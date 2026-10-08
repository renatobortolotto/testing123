# Databricks notebook source
# MAGIC %md
# MAGIC # NBA V2.3 — Parte 02B: auditoria temporal, sem retreino
# MAGIC
# MAGIC Consome um ajuste concluído da Parte 02. Não modifica parâmetros,
# MAGIC previsões, pesos, limites nem a tabela de negócio. Não executa Multi-step.
# MAGIC
# MAGIC Objetivos: verificar reprodução de scores; identificar os parâmetros
# MAGIC nos limites; localizar deterioração por contexto/idade; distinguir
# MAGIC extrapolação, falta de suporte e concentração em poucos clientes.
# MAGIC A comparação é descritiva e pareada. Não atribui causalidade aos alertas.
# MAGIC Idade = tempo JÁ transcorrido, não horizonte futuro de ocorrência.

# COMMAND ----------

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
from scipy import special, stats
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


AUD_CFG = {
    "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
    "id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
    "tabela_ajustes": "ctg_dsti.renato_nba.nba_sm_v23_ajustes_hml",
    "prefixo_saida": "ctg_dsti.renato_nba.nba_sm_v23_auditoria",
    "gravar": True,
    "autotestes": True,
    "max_modelos_driver": 5000,
    "max_agregados_driver": 100000,
    "max_linhas_exibir": 30,
    "bootstrap_replicas": 2000,
    "semente": 20261008,
    "top_casos_por_comparacao_idade": 20,
    # Limiares de TRIAGEM, não cortes de elegibilidade/promoção/calibração.
    "prob_baixa_triagem": 1e-6,
    "min_clientes_cauda_triagem": 20,
    "tol_prob": 1e-9,
    "tol_metrica": 1e-7,
    "tol_limite": 1e-4,  # Mesma tolerância usada na Parte 02.
}
AUD_VERSAO = "v2.3_auditoria_temporal_02b_v1"
AUD_FORMATO_SUPORTADO = "v2.3_temporal_memoria_pesos_02_v1"
AUD_BASE_CTX = "__BASE__"
AUD_VARIANTES = (
    "A_REFERENCIA", "B_MEMORIA", "C_PONDERACAO", "D_MEMORIA_PONDERACAO",
)
AUD_COMPARACOES = (
    ("B_VS_A", "A_REFERENCIA", "B_MEMORIA"),
    ("C_VS_A", "A_REFERENCIA", "C_PONDERACAO"),
    ("D_VS_A", "A_REFERENCIA", "D_MEMORIA_PONDERACAO"),
    ("D_VS_B", "B_MEMORIA", "D_MEMORIA_PONDERACAO"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Funções numéricas de auditoria
# MAGIC Só reavaliam modelos já ajustados. Não chamam nenhum otimizador.
# MAGIC Recalculam q em log-space e verificam a função de sobrevivência
# MAGIC também por `scipy.stats.lognorm.logsf` (sigma, scale=exp(mu), dias).
# MAGIC O log loss mantém o MESMO piso do ajuste original; também preservamos
# MAGIC log(q) sem piso para inspecionar destinos suportados nas caudas.

# COMMAND ----------


def aud_modelo_id(variante, estado, contexto):
    texto = json.dumps([variante, estado, contexto], ensure_ascii=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def aud_validar_modelo(m):
    """Contrato exato do modelo persistido na Parte 02."""
    if m.get("formato") != "sm_v23_lognormal_grupos":
        raise ValueError("FORMATO_MODELO_NAO_SUPORTADO")
    if m.get("unidade_tempo") != "dias":
        raise ValueError("UNIDADE_TEMPO_DIFERENTE_DE_DIAS")
    pi = np.asarray(m["pi_grupo"], dtype=float)
    mu = np.asarray(m["mu_grupo"], dtype=float)
    sigma = np.asarray(m["sigma_grupo"], dtype=float)
    r = np.asarray(m["r_destino_no_grupo"], dtype=float)
    g = np.asarray(m["grupo"], dtype=int)
    p0 = np.asarray(m["p_destino"], dtype=float)
    ng, nd = len(pi), len(m["destinos"])
    if not ng or not nd or len(set(m["destinos"])) != nd:
        raise ValueError("CATALOGO_VAZIO_OU_DUPLICADO")
    if not (len(mu) == len(sigma) == len(m["grupos_nomes"]) == ng):
        raise ValueError("DIMENSAO_GRUPOS_INVALIDA")
    if not (len(g) == len(r) == len(p0) == nd):
        raise ValueError("DIMENSAO_DESTINOS_INVALIDA")
    if g.min() < 0 or g.max() >= ng:
        raise ValueError("INDICE_GRUPO_INVALIDO")
    if not all(np.isfinite(a).all() for a in (pi, mu, sigma, r, p0)):
        raise ValueError("PARAMETROS_NAO_FINITOS")
    if min(pi.min(), sigma.min(), r.min(), p0.min()) <= 0:
        raise ValueError("PARAMETRO_NAO_POSITIVO")
    if not np.isclose(pi.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError("SOMA_PI_INVALIDA")
    if not np.allclose(np.bincount(g, weights=r, minlength=ng), 1.0):
        raise ValueError("SOMA_R_INVALIDA")
    if not np.allclose(pi[g] * r, p0, atol=1e-10, rtol=0):
        raise ValueError("P0_INCONSISTENTE")
    if m["estrutura"]["destinos"] != m["destinos"]:
        raise ValueError("ESTRUTURA_DESTINOS_INCONSISTENTE")
    if m["estrutura"]["grupo"] != m["grupo"]:
        raise ValueError("ESTRUTURA_GRUPOS_INCONSISTENTE")
    if not (np.isfinite(m["max_tempo_observado_dias"])
            and m["max_tempo_observado_dias"] > 0):
        raise ValueError("MAX_TEMPO_INVALIDO")


def aud_prever(m, idade_seg):
    """Reproduz q, sem alterar parâmetros ou renormalizar Top 5."""
    if not np.isfinite(idade_seg) or idade_seg < 0:
        raise ValueError("IDADE_INVALIDA")
    pi = np.asarray(m["pi_grupo"], dtype=float)
    mu = np.asarray(m["mu_grupo"], dtype=float)
    sigma = np.asarray(m["sigma_grupo"], dtype=float)
    r = np.asarray(m["r_destino_no_grupo"], dtype=float)
    g = np.asarray(m["grupo"], dtype=int)
    log_sf = np.zeros(len(pi), dtype=float)
    erro_sf = 0.0
    if idade_seg > 0:
        t = idade_seg / 86400.0
        log_sf = special.log_ndtr(-(np.log(t) - mu) / sigma)
        log_sf_stats = stats.lognorm.logsf(t, s=sigma, scale=np.exp(mu))
        if not np.isfinite(log_sf_stats).all():
            raise ValueError("LOGSF_NAO_FINITA_NA_VERIFICACAO")
        erro_sf = float(np.max(np.abs(log_sf - log_sf_stats)))
        if not np.allclose(log_sf, log_sf_stats, atol=1e-8, rtol=1e-10):
            raise ValueError("LOGSF_DIVERGE_ENTRE_FORMULAS")
    log_mass = np.log(pi) + log_sf
    log_s_mix = float(special.logsumexp(log_mass))
    log_q = (log_mass - log_s_mix)[g] + np.log(r)
    q = np.exp(log_q)
    if not np.isfinite(log_q).all() or not np.isclose(q.sum(), 1, atol=1e-9):
        raise ValueError("DISTRIBUICAO_CONDICIONAL_INVALIDA")
    p0 = np.asarray(m["p_destino"], dtype=float)
    if idade_seg == 0 and not np.allclose(q, p0, atol=1e-10):
        raise ValueError("Q_ZERO_DIFERENTE_DE_P0")
    return {
        "q": q, "log_q": log_q, "log_sf": log_sf,
        "log_s_mix": log_s_mix, "erro_sf": erro_sf,
        "soma_quadrados_q": float(np.dot(q, q)),
        "tv": float(np.abs(q - p0).sum() / 2),
        "ordem": np.argsort(-q, kind="mergesort"),
    }


def aud_parametros(m, cfg_num, pai=None, tolerancia=1e-4):
    """Reconstrói valores e lados dos limites, não só o nome do alerta."""
    pi = np.asarray(m["pi_grupo"], float)
    mu = np.asarray(m["mu_grupo"], float)
    sigma = np.asarray(m["sigma_grupo"], float)
    ng = len(pi)
    valores = [
        (f"logit_{k}", "LOGIT", k, float(np.log(pi[k]) - np.log(pi[-1])),
         -25.0, 25.0) for k in range(ng - 1)
    ]
    valores += [
        (f"mu_{k}", "MU_LOG_DIAS", k, float(mu[k]),
         float(m["estrutura"]["limite_mu"][0]),
         float(m["estrutura"]["limite_mu"][1])) for k in range(ng)
    ]
    valores += [
        (f"log_sigma_{k}", "LOG_SIGMA", k, float(np.log(sigma[k])),
         math.log(cfg_num["sigma_min"]), math.log(cfg_num["sigma_max"]))
        for k in range(ng)
    ]
    parametros_pai = {}
    if pai is not None:
        pp = np.asarray(pai["pi_grupo"], float)
        if (pai["destinos"] != m["destinos"] or pai["grupo"] != m["grupo"]):
            raise ValueError("ESTRUTURAS_PAI_CONTEXTO_DIFERENTES")
        parametros_pai.update({f"logit_{k}": float(np.log(pp[k]) - np.log(pp[-1]))
                              for k in range(ng - 1)})
        parametros_pai.update({f"mu_{k}": float(pai["mu_grupo"][k])
                              for k in range(ng)})
        parametros_pai.update({f"log_sigma_{k}": math.log(pai["sigma_grupo"][k])
                              for k in range(ng)})
    linhas, encontrados = [], set()
    for nome, tipo, k, valor, lo, hi in valores:
        if lo >= hi or valor < lo - 1e-7 or valor > hi + 1e-7:
            raise ValueError(f"PARAMETRO_FORA_LIMITES: {nome}")
        perto = min(abs(valor - lo), abs(valor - hi)) < tolerancia
        if perto:
            encontrados.add(nome)
        vp = parametros_pai.get(nome)
        linhas.append({
            "parametro": nome, "tipo_parametro": tipo, "grupo_indice": int(k),
            "grupo_nome": str(m["grupos_nomes"][k]), "valor": valor,
            "limite_inferior": lo, "limite_superior": hi,
            "distancia_limite": float(min(abs(valor - lo), abs(valor - hi))),
            "no_limite": bool(perto),
            "lado_limite": ("INFERIOR" if abs(valor - lo) <= abs(valor - hi)
                            else "SUPERIOR") if perto else "INTERIOR",
            "valor_pai": vp, "delta_pai": None if vp is None else valor - vp,
            "pi_grupo": float(pi[k]), "sigma_grupo": float(sigma[k]),
            "mediana_condicional_grupo_seg": float(np.exp(mu[k]) * 86400),
        })
    registrados = set(m["parametros_no_limite"])
    if encontrados != registrados or bool(encontrados) != bool(m["alerta_limite"]):
        raise ValueError("ALERTA_LIMITE_NAO_REPRODUZIDO")
    return linhas


def aud_bootstrap(df, replicas, seed):
    """Recebe SOMENTE médias por cliente. Ganho positivo = melhora."""
    rng = np.random.default_rng(seed)
    out = []
    for (seg, idade, comp), grupo in df.groupby(
        ["segmento", "idade_seg", "comparacao"], sort=True
    ):
        grupo = grupo.sort_values("cd_bv")
        n = len(grupo)
        if not n:
            continue
        indices = rng.integers(0, n, size=(replicas, n))
        for metrica in ("logloss", "brier", "top1", "top5"):
            a = grupo[f"media_{metrica}_ref"].to_numpy(float)
            b = grupo[f"media_{metrica}_var"].to_numpy(float)
            if not np.isfinite(np.r_[a, b]).all():
                raise ValueError("METRICA_NAO_FINITA_NO_BOOTSTRAP")
            ganho = a - b if metrica in ("logloss", "brier") else b - a
            lo, hi = np.quantile(ganho[indices].mean(axis=1), [0.025, 0.975])
            out.append({
                "segmento": str(seg), "idade_seg": float(idade),
                "comparacao": str(comp), "metrica": metrica,
                "n_clientes": int(n), "n_casos": int(grupo["n_casos"].sum()),
                "media_ref": float(a.mean()), "media_variante": float(b.mean()),
                "ganho_medio": float(ganho.mean()),
                "ganho_ic95_lo": float(lo), "ganho_ic95_hi": float(hi),
            })
    return out


def aud_testes_numericos():
    pi, r, g = [0.6, 0.4], [0.75, 0.25, 1.0], [0, 0, 1]
    m = {
        "formato": "sm_v23_lognormal_grupos", "unidade_tempo": "dias",
        "destinos": ["a", "b", "c"], "grupo": g,
        "grupos_nomes": ["g0", "g1"], "pi_grupo": pi,
        "mu_grupo": [-7.0, -0.5], "sigma_grupo": [0.15, 1.2],
        "r_destino_no_grupo": r, "p_destino": [0.45, 0.15, 0.4],
        "estrutura": {"destinos": ["a", "b", "c"], "grupo": g,
                      "limite_mu": [-15.0, 10.0]},
        "max_tempo_observado_dias": 30.0,
        "alerta_limite": True, "parametros_no_limite": ["log_sigma_0"],
    }
    aud_validar_modelo(m)
    for idade in (0, 1, 1800, 86400, 604800):
        p = aud_prever(m, idade)
        assert np.isclose(p["q"].sum(), 1.0)
        assert np.isfinite(p["log_q"]).all()
    assert np.allclose(aud_prever(m, 0)["q"], m["p_destino"])
    assert aud_prever(m, 604800)["q"][2] > 0.99
    pars = aud_parametros(m, {"sigma_min": 0.15, "sigma_max": 4.5})
    assert [p["parametro"] for p in pars if p["no_limite"]] == ["log_sigma_0"]
    assert [p for p in pars if p["no_limite"]][0]["lado_limite"] == "INFERIOR"
    for bad in (-1.0, float("nan")):
        try:
            aud_prever(m, bad)
        except ValueError:
            continue
        raise AssertionError("Idade inválida não foi rejeitada")
    print("Autotestes: q(0)=p, sobrevivência, massa e limites OK.")


if AUD_CFG["autotestes"]:
    aud_testes_numericos()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fontes e integridade
# MAGIC Usa o ID explícito e o manifesto concluído. Para treino/suporte,
# MAGIC lê as versões Delta registradas na Parte 02. Se expiraram, PARA:
# MAGIC não troca silenciosamente por dados atuais. Não lê customers_query.
# MAGIC Os pais reutilizados por B/D não são contados como novos ajustes.

# COMMAND ----------

if "spark" not in globals():
    raise RuntimeError("Execute este notebook no Databricks com sessão Spark.")

AUD_ID_EXP = str(uuid.UUID(AUD_CFG["id_experimento"]))
AUD_ID_FIT = str(uuid.UUID(AUD_CFG["id_ajuste"]))
AUD_ID = str(uuid.uuid4())
AUD_FONTES = {}
AUD_RELATORIOS = {}
AUD_CHECKS = []
AUD_BASE_CACHE = []
AUD_CHAVE_CASO = ["cd_bv", "passo", "idade_seg"]
AUD_CHAVE_MODELO = ["variante", "estado", "contexto_modelo"]


def aud_nome_sql(nome):
    partes = nome.split(".")
    if len(partes) != 3 or any(not p or "`" in p for p in partes):
        raise ValueError("Esperado catalogo.schema.tabela, sem crase")
    return ".".join(f"`{p}`" for p in partes)


def aud_ler(tabela, versao=None):
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela não encontrada: {tabela}")
    if versao is None:
        versao = int(spark.sql(
            f"DESCRIBE HISTORY {aud_nome_sql(tabela)} LIMIT 1"
        ).first()["version"])
    AUD_FONTES[tabela] = int(versao)
    return spark.read.option("versionAsOf", int(versao)).table(tabela)


def aud_cols(df, cols, nome):
    faltam = set(cols) - set(df.columns)
    if faltam:
        raise RuntimeError(f"{nome}: colunas ausentes: {sorted(faltam)}")


def aud_sem_erros(nome, erros):
    n = erros.count()
    AUD_CHECKS.append((nome, int(n), "OK" if n == 0 else "FALHA"))
    if n:
        print(f"FALHA ESTRUTURAL: {nome}; n={n}. Nenhum modelo será alterado.")
        erros.drop("cd_bv", "modelo_json").show(5, truncate=False)
        raise RuntimeError(f"Auditoria interrompida: {nome}")


def aud_cache(df):
    df = df.persist(StorageLevel.DISK_ONLY)
    df.count()
    AUD_BASE_CACHE.append(df)
    return df


def aud_mostrar(nome, df, ordenar=None):
    AUD_RELATORIOS[nome] = df
    print(f"\n{nome}")
    if ordenar is not None:
        df = df.orderBy(*ordenar)
    # O relatório persistido conserva todas as colunas; o print é reduzido.
    compactos = {
        "V23_AUD_02_PARAMETROS_LIMITES": [
            "familia_modelo", "estado_modelo", "contexto_ajustado", "parametro",
            "lado_limite", "valor", "valor_pai", "n_exatas_grupo",
            "n_clientes_exatas_grupo",
        ],
        "V23_AUD_05_ERROS_POR_ALERTA": [
            "segmento", "idade_seg", "comparacao", "categoria_variante",
            "alerta_limite_ref", "alerta_limite_var", "n_casos", "n_clientes",
            "delta_logloss_evento", "contrib_delta_logloss_media_cliente",
        ],
        "V23_AUD_06_CONTEXTOS_PRIORITARIOS": [
            "idade_seg", "estado", "contexto_modelo", "destino_real",
            "n_casos", "n_clientes", "tipos_limite_var",
            "n_clientes_treino_risco_var", "contrib_delta_logloss_media_cliente",
        ],
        "V23_AUD_07_CONCENTRACAO_CLIENTES": [
            "segmento", "idade_seg", "comparacao", "n_clientes",
            "delta_medio_cliente", "n_clientes_pioraram",
            "fracao_piora_positiva_top_1_clientes",
            "fracao_piora_positiva_top_5_clientes",
            "fracao_piora_positiva_top_10_clientes",
        ],
        "V23_AUD_09_CASOS_TRIAGEM": [
            "idade_seg", "estado", "contexto_modelo", "destino_real",
            "prob_destino_real_ref", "prob_destino_real_var", "delta_logloss",
            "extrapolacao_var", "parametros_limite_var",
        ],
    }
    if nome in compactos:
        df = df.select(*compactos[nome])
    df.show(AUD_CFG["max_linhas_exibir"], truncate=False)


def aud_filtro_fit(df):
    aud_cols(df, {"id_experimento", "id_ajuste", "hash_ajuste"}, "Artefato")
    return df.filter((F.col("id_experimento") == AUD_ID_EXP)
                     & (F.col("id_ajuste") == AUD_ID_FIT))



def aud_campos_rota():
    """Família e contexto efetivamente usados, sem criar novo fallback."""
    familia = F.when(
        F.col("rota_modelo") == "CONTEXTO_AJUSTADO", F.col("variante")
    ).otherwise(F.when(
        F.col("variante").isin("C_PONDERACAO", "D_MEMORIA_PONDERACAO"),
        "C_PONDERACAO",
    ).otherwise("A_REFERENCIA"))
    contexto = F.when(
        F.col("rota_modelo") == "CONTEXTO_AJUSTADO", F.col("contexto_modelo")
    ).otherwise(F.lit(AUD_BASE_CTX))
    return familia, contexto


def aud_testes_spark():
    teste = spark.createDataFrame([
        ("B_MEMORIA", "CONTEXTO_AJUSTADO", "pix", "B_MEMORIA", "pix"),
        ("B_MEMORIA", "ORIGEM_CONTEXTO_RARO", "raro", "A_REFERENCIA", AUD_BASE_CTX),
        ("D_MEMORIA_PONDERACAO", "CONTEXTO_AJUSTADO", "pix",
         "D_MEMORIA_PONDERACAO", "pix"),
        ("D_MEMORIA_PONDERACAO", "ORIGEM_SEM_MEMORIA", "__SEM_MEMORIA__",
         "C_PONDERACAO", AUD_BASE_CTX),
        ("A_REFERENCIA", "ORIGEM_REFERENCIA", "pix", "A_REFERENCIA", AUD_BASE_CTX),
        ("C_PONDERACAO", "ORIGEM_REFERENCIA", "pix", "C_PONDERACAO", AUD_BASE_CTX),
    ], "variante string, rota_modelo string, contexto_modelo string, "
       "familia_esperada string, contexto_esperado string")
    f, c = aud_campos_rota()
    resultado = teste.withColumn("familia_calc", f).withColumn("ctx_calc", c)
    n = resultado.filter(
        ~F.col("familia_calc").eqNullSafe(F.col("familia_esperada"))
        | ~F.col("ctx_calc").eqNullSafe(F.col("contexto_esperado"))
    ).count()
    if n:
        raise RuntimeError("Autoteste Spark: roteamento não confere.")
    print("Autoteste Spark: reuso de pais e seleção contextual OK.")


if AUD_CFG["autotestes"]:
    aud_testes_spark()


manifesto_rows = aud_ler(AUD_CFG["tabela_ajustes"]).filter(
    (F.col("id_experimento") == AUD_ID_EXP)
    & (F.col("id_ajuste") == AUD_ID_FIT)
    & (F.col("status") == "CONCLUIDO_TREINO_VALIDACAO")
).limit(2).collect()
if len(manifesto_rows) != 1:
    raise RuntimeError("O ID deve possuir exatamente um ajuste concluído.")
AUD_MANIFESTO = manifesto_rows[0].asDict()
if AUD_MANIFESTO["versao_ajuste"] != AUD_FORMATO_SUPORTADO:
    raise RuntimeError("Versão de ajuste diferente: revisar contrato/limites primeiro.")
AUD_FIT = json.loads(AUD_MANIFESTO["config_json"])
AUD_NUM = AUD_FIT["numerico"]
AUD_PREPARO = AUD_FIT["fonte_preparo"]
AUD_IDADES = [float(a) for a in AUD_FIT["idades_seg"]]
AUD_LOG_FLOOR = math.log(float(AUD_FIT["logloss_piso"]))
if spark.conf.get("spark.sql.session.timeZone") != AUD_PREPARO["fuso_base"]:
    raise RuntimeError("Fuso diferente do preparo. Não converta os timestamps.")

modelos_brutos = aud_filtro_fit(aud_ler(AUD_FIT["tabelas_saida"]["tabela_modelos"]))
validacao_bruta = aud_filtro_fit(aud_ler(AUD_FIT["tabelas_saida"]["tabela_validacao"]))
base = aud_ler(AUD_PREPARO["tabela_base_v23"], AUD_FIT["versao_delta_base_v23"]).filter(
    F.col("id_experimento") == AUD_ID_EXP
)
suporte = aud_ler(AUD_PREPARO["tabela_suporte_v23"], AUD_FIT["versao_delta_suporte_v23"]).filter(
    F.col("id_experimento") == AUD_ID_EXP
)
aud_cols(modelos_brutos, {
    "variante", "nivel", "estado", "contexto_modelo", "reuso", "status_modelo",
    "modelo_json", "alerta_limite", "detalhe", "n_observacoes", "n_clientes",
}, "Modelos")
aud_cols(validacao_bruta, {
    "cd_bv", "passo", "estado", "contexto_modelo", "idade_seg", "origem_tecnica",
    "variante", "tipo_censura", "destino_real", "rota_modelo", "tem_previsao",
    "suporte_destino", "alerta_limite", "extrapolacao", "prob_destino_real",
    "acerto_top1", "acerto_top5", "top1_previsto", "top5_previsto", "brier",
    "logloss_clip", "logloss_sem_tempo_clip", "nll_clip_aplicado",
    "variacao_temporal_tv",
}, "Validação")
aud_cols(base, {
    "cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
    "origem_tecnica", "contexto_modelo", "validacao_cliente", "elegivel_ajuste",
    "peso_cliente_treino", "peso_evento_treino", "politica_sha256",
}, "Base")
for df, nome in ((modelos_brutos, "modelos"), (validacao_bruta, "validação")):
    aud_sem_erros(f"hash_ajuste_{nome}", df.filter(
        ~F.col("hash_ajuste").eqNullSafe(AUD_MANIFESTO["hash_ajuste"])
    ))
if base.count() != int(AUD_PREPARO["n_linhas_base"]):
    raise RuntimeError("Contagem da base não confere com o preparo.")
aud_sem_erros("politica_base", base.filter(
    ~F.col("politica_sha256").eqNullSafe(AUD_MANIFESTO["politica_sha256"])
))
aud_sem_erros("duplicidade_base", base.groupBy("cd_bv", "passo").count().filter("count != 1"))
aud_sem_erros("duplicidade_modelos", modelos_brutos.groupBy(
    *AUD_CHAVE_MODELO
).count().filter("count != 1"))
aud_sem_erros("duplicidade_validacao", validacao_bruta.groupBy(
    *AUD_CHAVE_CASO, "variante"
).count().filter("count != 1"))
aud_sem_erros("quatro_variantes_por_caso", validacao_bruta.groupBy(*AUD_CHAVE_CASO).agg(
    F.countDistinct("variante").alias("n"), F.count("variante").alias("n_linhas")
).filter("n != 4 OR n_linhas != 4"))
aud_sem_erros("cliente_nos_dois_splits", base.groupBy("cd_bv").agg(
    F.countDistinct("validacao_cliente").alias("n")
).filter("n != 1"))
aud_sem_erros("valores_chave_nulos", validacao_bruta.filter(
    F.col("cd_bv").isNull() | F.col("passo").isNull()
    | F.col("contexto_modelo").isNull() | F.col("idade_seg").isNull()
    | ~F.col("idade_seg").isin(AUD_IDADES)
    | ~F.col("variante").isin(*AUD_VARIANTES)
    | F.col("tem_previsao").isNull() | F.col("origem_tecnica").isNull()
    | F.col("tipo_censura").isNull()
    | ~F.col("tipo_censura").isin("exata", "direita")
))

# A chave de observação é fixa; as censuras não recebem rótulos de destino.
colunas_base_join = ["estado", "contexto_modelo", "tipo_censura", "origem_tecnica"]
base_aud = base.select(
    "cd_bv", "passo", "dur_min", "destino", "validacao_cliente",
    "elegivel_ajuste",
    *[F.col(c).alias(f"base_{c}") for c in colunas_base_join],
)
val = validacao_bruta.join(base_aud, ["cd_bv", "passo"], "left")
cond_erro = (F.col("dur_min").isNull() | ~F.col("validacao_cliente")
             | ~F.col("elegivel_ajuste") | (F.col("dur_min") <= F.col("idade_seg")))
for c in colunas_base_join:
    cond_erro = cond_erro | ~F.col(c).eqNullSafe(F.col(f"base_{c}"))
cond_erro = cond_erro | ~F.col("destino_real").eqNullSafe(F.col("destino"))
aud_sem_erros("observacao_e_marco_holdout", val.filter(cond_erro))

# Modelos B/D no nível ORIGEM são reusos, não ajustes adicionais.
modelos = modelos_brutos.filter(~F.col("reuso"))
modelos_ajustados = modelos.filter(F.col("status_modelo") == "AJUSTADO")
rows_modelos = modelos_ajustados.limit(AUD_CFG["max_modelos_driver"] + 1).collect()
if not rows_modelos or len(rows_modelos) > AUD_CFG["max_modelos_driver"]:
    raise RuntimeError("Catálogo vazio/excede proteção de memória do driver.")
AUD_MODELOS = {}
for row in rows_modelos:
    d = row.asDict()
    k = (d["variante"], d["estado"], d["contexto_modelo"])
    m = json.loads(d["modelo_json"])
    aud_validar_modelo(m)
    if bool(d["alerta_limite"]) != bool(m["alerta_limite"]):
        raise RuntimeError(f"Flag externo/JSON divergente em {k}")
    AUD_MODELOS[k] = (d, m)
print("Ajuste auditado:", AUD_ID_FIT, "| Auditoria:", AUD_ID)
print("Corte histórico:", AUD_PREPARO["corte_estado_iso"])
print("Nenhum modelo será treinado ou alterado.")
aud_mostrar("V23_AUD_00_INVENTARIO", modelos.groupBy(
    "variante", "nivel", "status_modelo", "alerta_limite"
).agg(F.count("*").alias("n_modelos"), F.sum("n_observacoes").alias("n_linhas_treino")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Catálogo de parâmetros e kernels de auditoria
# MAGIC `valor` está na escala do parâmetro: logit, log-dias ou log-sigma.
# MAGIC `mediana_condicional_grupo_seg` não é duração média do cliente.
# MAGIC A soma de kernels é verificada sobre TODOS os destinos, não só Top 5.

# COMMAND ----------

AUD_META_SCHEMA = (
    "id_modelo string, familia_modelo string, nivel_modelo string, "
    "estado_modelo string, contexto_ajustado string, peso_utilizado string, "
    "n_obs_modelo long, n_clientes_modelo long, n_grupos_modelo long, "
    "massa_pesos_modelo double, alerta_json boolean, tipos_limite string, "
    "parametros_limite array<string>, max_tempo_modelo_seg double"
)
AUD_PAR_SCHEMA = (
    "id_modelo string, parametro string, tipo_parametro string, grupo_indice int, "
    "grupo_nome string, valor double, limite_inferior double, limite_superior double, "
    "distancia_limite double, no_limite boolean, lado_limite string, "
    "valor_pai double, delta_pai double, pi_grupo double, sigma_grupo double, "
    "mediana_condicional_grupo_seg double"
)
AUD_KERNEL_SCHEMA = (
    "id_modelo string, idade_seg double, log_s_mistura double, "
    "erro_sf_formula double, soma_quadrados_q double, tv_recalc double, "
    "top1_recalc string, top5_recalc array<string>, prob_top1_recalc double, "
    "destinos_q array<struct<destino:string,grupo_indice:int,prob_recalc:double,"
    "log_q_recalc:double,p0_recalc:double,log_sf_grupo:double,"
    "limite_temporal_grupo:boolean>>"
)
meta_rows, par_rows, kernel_rows = [], [], []
for (familia, origem, contexto), (row, m) in AUD_MODELOS.items():
    mid = aud_modelo_id(familia, origem, contexto)
    pai = None
    if row["nivel"] == "CONTEXTO":
        familia_pai = "A_REFERENCIA" if familia == "B_MEMORIA" else "C_PONDERACAO"
        pai = AUD_MODELOS[(familia_pai, origem, AUD_BASE_CTX)][1]
    pars = aud_parametros(m, AUD_NUM, pai, AUD_CFG["tol_limite"])
    tipos = sorted({p["tipo_parametro"] for p in pars if p["no_limite"]})
    meta_rows.append({
        "id_modelo": mid, "familia_modelo": familia, "nivel_modelo": row["nivel"],
        "estado_modelo": origem, "contexto_ajustado": contexto,
        "peso_utilizado": m["peso_utilizado"], "n_obs_modelo": int(m["n_observacoes"]),
        "n_clientes_modelo": int(m["n_clientes"]), "n_grupos_modelo": int(m["n_grupos"]),
        "massa_pesos_modelo": float(m["massa_pesos"]), "alerta_json": bool(m["alerta_limite"]),
        "tipos_limite": "+".join(tipos) if tipos else "SEM_LIMITE",
        "parametros_limite": sorted(m["parametros_no_limite"]),
        "max_tempo_modelo_seg": float(m["max_tempo_observado_dias"] * 86400),
    })
    par_rows.extend(dict(id_modelo=mid, **p) for p in pars)
    limite_grupos = {p["grupo_indice"] for p in pars
                     if p["no_limite"] and p["tipo_parametro"] != "LOGIT"}
    for idade in AUD_IDADES:
        p = aud_prever(m, idade)
        top = [m["destinos"][j] for j in p["ordem"][:5]]
        destinos_q = [{
            "destino": d, "grupo_indice": int(m["grupo"][j]),
            "prob_recalc": float(p["q"][j]), "log_q_recalc": float(p["log_q"][j]),
            "p0_recalc": float(m["p_destino"][j]),
            "log_sf_grupo": float(p["log_sf"][m["grupo"][j]]),
            "limite_temporal_grupo": m["grupo"][j] in limite_grupos,
        } for j, d in enumerate(m["destinos"])]
        kernel_rows.append({
            "id_modelo": mid, "idade_seg": idade, "log_s_mistura": p["log_s_mix"],
            "erro_sf_formula": p["erro_sf"], "soma_quadrados_q": p["soma_quadrados_q"],
            "tv_recalc": p["tv"], "top1_recalc": top[0], "top5_recalc": top,
            "prob_top1_recalc": float(p["q"][p["ordem"][0]]), "destinos_q": destinos_q,
        })
meta = spark.createDataFrame(meta_rows, AUD_META_SCHEMA)
parametros = spark.createDataFrame(par_rows, AUD_PAR_SCHEMA)
kernels = spark.createDataFrame(kernel_rows, AUD_KERNEL_SCHEMA)
kd = kernels.select("id_modelo", "idade_seg", F.explode("destinos_q").alias("d")).select(
    "id_modelo", "idade_seg", F.col("d.destino").alias("destino_real"),
    *[F.col(f"d.{c}").alias(c) for c in (
        "grupo_indice", "prob_recalc", "log_q_recalc", "p0_recalc",
        "log_sf_grupo", "limite_temporal_grupo",
    )]
)
aud_mostrar("V23_AUD_01_LIMITES_RESUMO", parametros.filter("no_limite").join(
    meta, "id_modelo"
).groupBy("familia_modelo", "nivel_modelo", "tipo_parametro", "lado_limite").agg(
    F.count("*").alias("n_parametros"), F.countDistinct("id_modelo").alias("n_modelos")
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Suporte do treino por grupo e por idade
# MAGIC Censuras não têm grupo/destino conhecido: são contadas no modelo,
# MAGIC não atribuídas artificialmente ao grupo de um destino.
# MAGIC O suporte remanescente é descritivo, não uma estimativa de sobrevivência.
# MAGIC O limite de clientes de cauda abaixo apenas sinaliza triagem.

# COMMAND ----------

train = base.filter(~F.col("validacao_cliente") & F.col("elegivel_ajuste")).select(
    "cd_bv", "passo", "estado", "contexto_modelo", "destino", "dur_min",
    "tipo_censura", "peso_evento_treino", "peso_cliente_treino"
)
mapa_pai = meta.filter("nivel_modelo = 'ORIGEM'").select(
    F.col("estado_modelo").alias("estado"), "id_modelo", "peso_utilizado"
)
mapa_ctx = meta.filter("nivel_modelo = 'CONTEXTO'").select(
    F.col("estado_modelo").alias("estado"), F.col("contexto_ajustado").alias("contexto_modelo"),
    "id_modelo", "peso_utilizado"
)
train_expand = train.join(mapa_pai, "estado").unionByName(
    train.join(mapa_ctx, ["estado", "contexto_modelo"])
).withColumn("peso_ajuste", F.when(
    F.col("peso_utilizado") == "peso_evento_treino", F.col("peso_evento_treino")
).otherwise(F.col("peso_cliente_treino")))
train_expand = aud_cache(train_expand)
train_m = train_expand.groupBy("id_modelo").agg(
    F.count("*").alias("n_obs_recontado"),
    F.countDistinct("cd_bv").alias("n_clientes_recontado"),
    F.sum("peso_ajuste").alias("massa_recontada"),
    F.sum((F.col("tipo_censura") == "direita").cast("long")).alias("n_censuras_modelo"),
)
aud_sem_erros("contagens_modelos_treino", meta.join(train_m, "id_modelo", "left").filter(
    F.col("n_obs_recontado").isNull() | (F.col("n_obs_recontado") != F.col("n_obs_modelo"))
    | (F.col("n_clientes_recontado") != F.col("n_clientes_modelo"))
    | (F.abs(F.col("massa_recontada") - F.col("massa_pesos_modelo")) > 1e-7)
))
mapa_dest = kd.filter(F.col("idade_seg") == 0).select(
    "id_modelo", F.col("destino_real").alias("destino"), "grupo_indice"
)
train_exact = train_expand.filter("tipo_censura = 'exata'").join(
    mapa_dest, ["id_modelo", "destino"], "left"
)
aud_sem_erros("destino_treino_fora_catalogo", train_exact.filter(F.col("grupo_indice").isNull()))
suporte_grupo = train_exact.groupBy("id_modelo", "grupo_indice").agg(
    F.count("*").alias("n_exatas_grupo"),
    F.countDistinct("cd_bv").alias("n_clientes_exatas_grupo"),
    F.countDistinct("dur_min").alias("n_tempos_distintos_grupo"),
    F.sum("peso_ajuste").alias("massa_exata_grupo"),
    F.min("dur_min").alias("min_tempo_exato_seg"), F.max("dur_min").alias("max_tempo_exato_seg"),
    F.expr("percentile_approx(dur_min, array(0.1, 0.5, 0.9))").alias("p10_p50_p90_exatas_seg"),
    F.stddev_pop(F.log(F.col("dur_min") / 86400)).alias("sd_log_t_exatas_nao_ponderado"),
)
parametros_det = parametros.join(meta, "id_modelo").join(
    suporte_grupo, ["id_modelo", "grupo_indice"], "left"
).fillna(0, subset=["n_exatas_grupo", "n_clientes_exatas_grupo", "n_tempos_distintos_grupo"])
parametros_det = aud_cache(parametros_det)
aud_mostrar("V23_AUD_02_PARAMETROS_LIMITES", parametros_det.filter("no_limite").select(
    "familia_modelo", "estado_modelo", "contexto_ajustado", "parametro", "grupo_nome",
    "lado_limite", "valor", "valor_pai", "delta_pai", "sigma_grupo",
    "pi_grupo", "n_exatas_grupo", "n_clientes_exatas_grupo", "n_tempos_distintos_grupo",
    "p10_p50_p90_exatas_seg",
), ["familia_modelo", "estado_modelo", "contexto_ajustado", "parametro"])

idades_df = spark.createDataFrame([(a,) for a in AUD_IDADES], "idade_seg double")
train_risco = train_expand.crossJoin(F.broadcast(idades_df)).filter(
    F.col("dur_min") > F.col("idade_seg")
)
risco = train_risco.groupBy("id_modelo", "idade_seg").agg(
    F.count("*").alias("n_obs_treino_risco"),
    F.countDistinct("cd_bv").alias("n_clientes_treino_risco"),
    F.sum((F.col("tipo_censura") == "exata").cast("long")).alias("n_exatas_treino_risco"),
    F.sum((F.col("tipo_censura") == "direita").cast("long")).alias("n_censuras_treino_risco"),
    F.sum("peso_ajuste").alias("massa_treino_risco"),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Roteamento efetivo e reprodução de todas as previsões
# MAGIC CONTEXTO_AJUSTADO usa B/D. Todas as rotas de origem em B usam A;
# MAGIC em D usam C. Isso impede atribuir a um contexto o alerta de outro.
# MAGIC Scores exatos são auditados; censuras ficam na cobertura e sem métricas
# MAGIC de destino. Casos sem previsão não recebem probabilidade inventada.

# COMMAND ----------

familia_efetiva, contexto_efetivo = aud_campos_rota()
val = val.withColumn("_familia_efetiva", familia_efetiva).withColumn(
    "_ctx_efetivo", contexto_efetivo,
)
mapa_modelos = meta.select(
    F.col("familia_modelo").alias("_familia_efetiva"),
    F.col("estado_modelo").alias("estado"), F.col("contexto_ajustado").alias("_ctx_efetivo"),
    "id_modelo", "alerta_json", "tipos_limite", "parametros_limite", "max_tempo_modelo_seg",
    "nivel_modelo",
)
val = val.join(mapa_modelos, ["_familia_efetiva", "estado", "_ctx_efetivo"], "left")
aud_sem_erros("previsao_sem_modelo_persistido", val.filter(
    F.col("tem_previsao") & F.col("id_modelo").isNull()
))
aud_sem_erros("rota_contexto_em_origem_nao_tecnica", val.filter(
    (F.col("rota_modelo") == "CONTEXTO_AJUSTADO")
    & (~F.col("origem_tecnica") | ~F.col("variante").isin("B_MEMORIA", "D_MEMORIA_PONDERACAO"))
))
val = val.join(kernels.drop("destinos_q"), ["id_modelo", "idade_seg"], "left").join(
    kd, ["id_modelo", "idade_seg", "destino_real"], "left"
).join(risco, ["id_modelo", "idade_seg"], "left").fillna(0, subset=[
    "n_obs_treino_risco", "n_clientes_treino_risco",
    "n_exatas_treino_risco", "n_censuras_treino_risco",
])
val = val.withColumn("suporte_recalc", F.col("grupo_indice").isNotNull()).withColumn(
    "prob_recalc", F.coalesce("prob_recalc", F.lit(0.0))
).withColumn("logloss_recalc", -F.greatest(
    F.coalesce("log_q_recalc", F.lit(float("-inf"))), F.lit(AUD_LOG_FLOOR)
)).withColumn("brier_recalc", F.col("soma_quadrados_q") - 2 * F.col("prob_recalc") + 1).withColumn(
    "top1_correto_recalc", F.col("destino_real").eqNullSafe(F.col("top1_recalc")).cast("double")
).withColumn(
    "top5_correto_recalc",
    F.expr("array_contains(top5_recalc, destino_real)").cast("double"),
)
val = val.withColumn(
    "clip_recalc",
    F.coalesce("log_q_recalc", F.lit(float("-inf"))) < AUD_LOG_FLOOR,
).withColumn(
    "logloss0_recalc",
    -F.greatest(
        F.coalesce(F.log("p0_recalc"), F.lit(float("-inf"))),
        F.lit(AUD_LOG_FLOOR),
    ),
)
val = aud_cache(val)
aud_sem_erros("flag_limite_roteado", val.filter(F.col("tem_previsao") & (
    ~F.col("alerta_limite").eqNullSafe(F.col("alerta_json"))
)))
aud_sem_erros("flag_extrapolacao", val.filter(F.col("tem_previsao") & (
    ~F.col("extrapolacao").eqNullSafe(F.col("idade_seg") > F.col("max_tempo_modelo_seg"))
)))
verificacao = val.filter(F.col("tem_previsao") & (F.col("tipo_censura") == "exata"))
for c in ("prob_destino_real", "brier", "logloss_clip", "acerto_top1", "acerto_top5"):
    aud_sem_erros(f"finito_{c}", verificacao.filter(
        F.col(c).isNull() | F.isnan(c) | (F.abs(F.col(c)) == float("inf"))
    ))
comparacoes_valores = [
    ("prob_destino_real", "prob_recalc", AUD_CFG["tol_prob"]),
    ("brier", "brier_recalc", AUD_CFG["tol_metrica"]),
    ("logloss_clip", "logloss_recalc", AUD_CFG["tol_metrica"]),
    ("logloss_sem_tempo_clip", "logloss0_recalc", AUD_CFG["tol_metrica"]),
    ("variacao_temporal_tv", "tv_recalc", AUD_CFG["tol_prob"]),
    ("acerto_top1", "top1_correto_recalc", 0.0), ("acerto_top5", "top5_correto_recalc", 0.0),
]
for c1, c2, tol in comparacoes_valores:
    aud_sem_erros(f"reproducao_{c1}", verificacao.filter(
        F.col(c2).isNull() | (F.abs(F.col(c1) - F.col(c2)) > tol)
    ))
aud_sem_erros("clipping_reportado", verificacao.filter(
    ~F.col("nll_clip_aplicado").eqNullSafe(F.col("clip_recalc"))
))
aud_sem_erros("suporte_destino", verificacao.filter(
    ~F.col("suporte_destino").eqNullSafe(F.col("suporte_recalc"))
))
aud_sem_erros("top5_ordenado", verificacao.filter(
    ~F.col("top5_previsto").eqNullSafe(F.col("top5_recalc"))
))
aud_sem_erros("censura_sem_rotulo", val.filter(
    (F.col("tipo_censura") == "direita") & (F.col("destino_real").isNotNull()
    | F.col("logloss_clip").isNotNull() | F.col("acerto_top1").isNotNull())
))
aud_mostrar("V23_AUD_03_INTEGRIDADE", spark.createDataFrame(
    AUD_CHECKS, "verificacao string, n_divergencias long, status string"
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pares exatos nos MESMOS casos das quatro variantes
# MAGIC Mantém a interseção de casos com previsão nas quatro variantes, igual
# MAGIC à Parte 02. Reporta a cobertura antes. Não remove destinos sem suporte,
# MAGIC probabilidades pequenas, extrapolações ou casos difíceis das métricas.
# MAGIC `delta_loss = variante - referência`: positivo = PIORA.
# MAGIC `ganho = referência - variante` para perdas: positivo = MELHORA.

# COMMAND ----------

exatas = val.filter("tipo_censura = 'exata'")
cobertura = exatas.groupBy("variante", "idade_seg", "origem_tecnica").agg(
    F.count("*").alias("n_casos"), F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum(F.col("tem_previsao").cast("long")).alias("n_com_previsao"),
    F.sum(F.when(
        F.col("tem_previsao") & ~F.col("suporte_destino"), 1
    ).otherwise(0)).alias("n_destinos_sem_suporte"),
)
aud_mostrar("V23_AUD_04_COBERTURA", cobertura)
chaves_comuns = exatas.groupBy(*AUD_CHAVE_CASO).agg(
    F.sum(F.col("tem_previsao").cast("long")).alias("n_prev")
).filter("n_prev = 4").select(*AUD_CHAVE_CASO)
comuns = exatas.join(chaves_comuns, AUD_CHAVE_CASO, "inner")
cols_pares = [
    "id_modelo", "alerta_limite", "tipos_limite", "parametros_limite", "extrapolacao",
    "suporte_destino", "prob_destino_real", "log_q_recalc", "logloss_clip", "brier",
    "acerto_top1", "acerto_top5", "top1_previsto", "rota_modelo", "nll_clip_aplicado",
    "limite_temporal_grupo", "grupo_indice", "log_sf_grupo", "log_s_mistura",
    "n_clientes_treino_risco", "n_obs_treino_risco", "n_exatas_treino_risco", "nivel_modelo",
]
pares_list = []
for comp, referencia, variante in AUD_COMPARACOES:
    ref = comuns.filter(F.col("variante") == referencia).select(
        *AUD_CHAVE_CASO, "estado", "contexto_modelo", "origem_tecnica", "destino_real", "dur_min",
        *[F.col(c).alias(f"{c}_ref") for c in cols_pares]
    )
    vr = comuns.filter(F.col("variante") == variante).select(
        *AUD_CHAVE_CASO, *[F.col(c).alias(f"{c}_var") for c in cols_pares]
    )
    par = ref.join(vr, AUD_CHAVE_CASO).withColumn("comparacao", F.lit(comp))
    for metrica, col in (("logloss", "logloss_clip"), ("brier", "brier"),
                         ("top1", "acerto_top1"), ("top5", "acerto_top5")):
        par = par.withColumn(f"delta_{metrica}", F.col(f"{col}_var") - F.col(f"{col}_ref"))
    pares_list.append(par)
pares = pares_list[0]
for par in pares_list[1:]:
    pares = pares.unionByName(par)
pares = pares.withColumn("segmento", F.explode(F.when(
    F.col("origem_tecnica"), F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS"))
).otherwise(F.array(F.lit("TODAS")))))
recorte = ["segmento", "idade_seg", "comparacao"]
wc = Window.partitionBy(*recorte, "cd_bv")
pares = pares.withColumn("n_casos_cliente", F.count("*").over(wc)).withColumn(
    "peso_cliente_avaliacao", 1.0 / F.col("n_casos_cliente")
).withColumn("categoria_variante", F.when(~F.col("suporte_destino_var"), "DESTINO_NAO_SUPORTADO")
    .when(F.col("extrapolacao_var"), "EXTRAPOLACAO_GLOBAL")
    .when(
        F.col("n_clientes_treino_risco_var") < AUD_CFG["min_clientes_cauda_triagem"],
        "POUCOS_CLIENTES_TREINO_NA_CAUDA",
    )
    .when(F.col("alerta_limite_var"), "LIMITE_COM_SUPORTE_GLOBAL")
    .otherwise("SEM_ALERTA_DE_TRIAGEM"))
pares = aud_cache(pares)
totais = pares.groupBy(*recorte).agg(
    F.count("*").alias("total_casos_recorte"),
    F.countDistinct("cd_bv").alias("total_clientes_recorte"),
)


def aud_resumir_pares(df, dims):
    """Contribuição aditiva na média por cliente, não média causal do grupo."""
    chaves = recorte + dims
    agregado = df.groupBy(*chaves).agg(
        F.count("*").alias("n_casos"), F.countDistinct("cd_bv").alias("n_clientes"),
        F.avg("logloss_clip_ref").alias("logloss_ref_evento"),
        F.avg("logloss_clip_var").alias("logloss_var_evento"),
        F.avg("delta_logloss").alias("delta_logloss_evento"),
        F.avg("delta_brier").alias("delta_brier_evento"),
        F.avg("delta_top1").alias("delta_top1_evento"),
        F.sum(F.col("delta_logloss") * F.col("peso_cliente_avaliacao"))
        .alias("soma_delta_ponderada_cliente"),
        F.sum(
            F.greatest("delta_logloss", F.lit(0.0))
            * F.col("peso_cliente_avaliacao")
        ).alias("soma_piora_ponderada"),
        F.sum(
            F.least("delta_logloss", F.lit(0.0))
            * F.col("peso_cliente_avaliacao")
        ).alias("soma_melhora_ponderada"),
        F.sum(F.col("nll_clip_aplicado_var").cast("long")).alias("n_clips_var"),
        F.sum((F.col("prob_destino_real_var") < AUD_CFG["prob_baixa_triagem"])
              .cast("long")).alias("n_prob_muito_baixa_var"),
    ).join(totais, recorte)
    return (
        agregado.withColumn(
            "contrib_delta_logloss_media_cliente",
            F.col("soma_delta_ponderada_cliente") / F.col("total_clientes_recorte"),
        ).withColumn(
            "contrib_piora_media_cliente",
            F.col("soma_piora_ponderada") / F.col("total_clientes_recorte"),
        ).withColumn(
            "contrib_melhora_media_cliente",
            F.col("soma_melhora_ponderada") / F.col("total_clientes_recorte"),
        ).withColumn(
            "fracao_casos", F.col("n_casos") / F.col("total_casos_recorte"),
        )
    )


alertas = aud_resumir_pares(pares, [
    "alerta_limite_ref", "alerta_limite_var", "extrapolacao_ref", "extrapolacao_var",
    "categoria_variante", "rota_modelo_var",
])
aud_mostrar("V23_AUD_05_ERROS_POR_ALERTA", alertas, [
    "segmento", "idade_seg", "comparacao", F.desc("contrib_piora_media_cliente")
])
contextos = aud_resumir_pares(pares, [
    "estado", "contexto_modelo", "destino_real", "id_modelo_ref", "id_modelo_var",
    "alerta_limite_ref", "alerta_limite_var", "extrapolacao_var", "rota_modelo_var",
    "tipos_limite_var", "limite_temporal_grupo_var", "n_clientes_treino_risco_var",
])
contextos = aud_cache(contextos)
# Toda a tabela é preservada. Só a exibição é limitada a uma amostra ordenada.
AUD_RELATORIOS["V23_AUD_06_CONTEXTOS_COMPLETO"] = contextos
aud_mostrar("V23_AUD_06_CONTEXTOS_PRIORITARIOS", contextos.filter(
    (F.col("segmento") == "ORIGENS_TECNICAS") & (F.col("idade_seg") > 0)
    & (F.col("comparacao") == "B_VS_A")
).orderBy(F.desc("contrib_delta_logloss_media_cliente")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Concentração por cliente e comparação direta D x B
# MAGIC Ganho positivo nos intervalos = melhora. Bootstrap de clientes, não
# MAGIC de observações independentes. ICs exploratórios, sem ajuste por testes
# MAGIC múltiplos. Não selecionamos/removemos os piores clientes da avaliação.
# MAGIC As contribuições por contexto somam a diferença média por cliente.

# COMMAND ----------

agg_pc = [F.count("*").alias("n_casos")]
for metrica, col in (("logloss", "logloss_clip"), ("brier", "brier"),
                     ("top1", "acerto_top1"), ("top5", "acerto_top5")):
    agg_pc += [F.avg(f"{col}_{lado}").alias(f"media_{metrica}_{lado}") for lado in ("ref", "var")]
por_cliente = pares.groupBy(*recorte, "cd_bv").agg(*agg_pc).withColumn(
    "delta_logloss_cliente", F.col("media_logloss_var") - F.col("media_logloss_ref")
)
# Garantia de que a decomposição não trocou de denominador.
decomp = contextos.groupBy(*recorte).agg(
    F.sum("contrib_delta_logloss_media_cliente").alias("soma_contrib")
).join(
    por_cliente.groupBy(*recorte).agg(
        F.avg("delta_logloss_cliente").alias("delta_global")
    ), recorte,
)
aud_sem_erros("decomposicao_por_cliente", decomp.filter(
    F.abs(F.col("soma_contrib") - F.col("delta_global")) > AUD_CFG["tol_metrica"]
))
wp = Window.partitionBy(*recorte).orderBy(F.desc("delta_logloss_cliente"), "cd_bv")
ww = Window.partitionBy(*recorte)
client_rank = por_cliente.withColumn("rank_piora", F.row_number().over(wp)).withColumn(
    "n_clientes_recorte", F.count("*").over(ww)
).withColumn("piora_positiva_cliente", F.greatest("delta_logloss_cliente", F.lit(0.0)))
agg_conc = [
    F.count("*").alias("n_clientes"), F.avg("delta_logloss_cliente").alias("delta_medio_cliente"),
    F.sum((F.col("delta_logloss_cliente") > 0).cast("long")).alias("n_clientes_pioraram"),
    F.sum("piora_positiva_cliente").alias("piora_positiva_total"),
    F.expr("percentile_approx(delta_logloss_cliente, array(0.1, 0.5, 0.9, 0.99))")
    .alias("p10_p50_p90_p99_delta_cliente"),
]
for n in (1, 5, 10):
    agg_conc.append(F.sum(F.when(F.col("rank_piora") <= n, F.col("piora_positiva_cliente"))
                         .otherwise(0.0)).alias(f"piora_top_{n}_clientes"))
concentracao = client_rank.groupBy(*recorte).agg(*agg_conc)
for n in (1, 5, 10):
    concentracao = concentracao.withColumn(f"fracao_piora_positiva_top_{n}_clientes", F.when(
        F.col("piora_positiva_total") > 0,
        F.col(f"piora_top_{n}_clientes") / F.col("piora_positiva_total"),
    ))
aud_mostrar("V23_AUD_07_CONCENTRACAO_CLIENTES", concentracao, recorte)

pc_driver = por_cliente.limit(AUD_CFG["max_agregados_driver"] + 1).toPandas()
if len(pc_driver) > AUD_CFG["max_agregados_driver"]:
    raise RuntimeError(
        "Muitos agregados por cliente para bootstrap. Nada foi truncado."
    )
boot_rows = aud_bootstrap(pc_driver, AUD_CFG["bootstrap_replicas"], AUD_CFG["semente"])
boot_schema = (
    "segmento string, idade_seg double, comparacao string, metrica string, "
    "n_clientes long, n_casos long, media_ref double, media_variante double, "
    "ganho_medio double, ganho_ic95_lo double, ganho_ic95_hi double"
)
bootstrap = spark.createDataFrame(boot_rows, boot_schema)
AUD_RELATORIOS["V23_AUD_08_PARES_COMPLETO"] = bootstrap
for comp in ("B_VS_A", "D_VS_B"):
    aud_mostrar(f"V23_AUD_08_PARES_{comp}", bootstrap.filter(
        (F.col("segmento") == "ORIGENS_TECNICAS") & (F.col("comparacao") == comp)
    ), ["idade_seg", "metrica"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Casos de piora, clipping e suporte na cauda
# MAGIC Salva no máximo K casos por comparação/idade no segmento técnico.
# MAGIC IDs ficam somente na tabela interna; a exibição não mostra cd_bv.
# MAGIC O log(q) é mantido, mesmo quando exp(log(q)) é zero por underflow.
# MAGIC Destino fora do vocabulário tem p=0; não é erro de cauda Lognormal.

# COMMAND ----------

w_piores = Window.partitionBy("comparacao", "idade_seg").orderBy(
    F.desc("delta_logloss"), "cd_bv", "passo"
)
casos = pares.filter(
    (F.col("segmento") == "ORIGENS_TECNICAS") & (F.col("delta_logloss") > 0)
).withColumn(
    "ranking_piora", F.row_number().over(w_piores)
).filter(F.col("ranking_piora") <= AUD_CFG["top_casos_por_comparacao_idade"])
casos = casos.select(
    "comparacao", "cd_bv", "passo", "idade_seg", "ranking_piora", "estado", "contexto_modelo",
    "destino_real", "dur_min", "id_modelo_ref", "id_modelo_var", "prob_destino_real_ref",
    "prob_destino_real_var", "log_q_recalc_ref", "log_q_recalc_var", "logloss_clip_ref",
    "logloss_clip_var", "delta_logloss", "delta_brier", "top1_previsto_ref", "top1_previsto_var",
    "suporte_destino_ref", "suporte_destino_var", "alerta_limite_ref", "alerta_limite_var",
    "parametros_limite_var", "extrapolacao_ref", "extrapolacao_var", "grupo_indice_var",
    "limite_temporal_grupo_var", "log_sf_grupo_var", "log_s_mistura_var",
    "n_clientes_treino_risco_var", "n_exatas_treino_risco_var",
    "rota_modelo_var", "categoria_variante",
)
aud_mostrar("V23_AUD_09_CASOS_TRIAGEM", casos.drop("cd_bv").filter(
    (F.col("comparacao") == "B_VS_A") & (F.col("idade_seg") > 0)
).orderBy("idade_seg", "ranking_piora"))

# Exposição real aos alertas/caudas, antes de olhar acertos/erros.
exposicao = exatas.filter("tem_previsao").groupBy("variante", "idade_seg", "origem_tecnica").agg(
    F.count("*").alias("n_casos_previstos"),
    F.sum(F.col("alerta_limite").cast("long")).alias("n_expostos_limite"),
    F.sum(F.col("extrapolacao").cast("long")).alias("n_extrapolados"),
    F.sum((F.col("n_clientes_treino_risco") < AUD_CFG["min_clientes_cauda_triagem"])
          .cast("long")).alias("n_com_pouco_suporte_cauda"),
    F.sum(F.col("nll_clip_aplicado").cast("long")).alias("n_logloss_clip"),
    F.sum((F.col("prob_destino_real") < AUD_CFG["prob_baixa_triagem"])
          .cast("long")).alias("n_prob_real_muito_baixa"),
)
aud_mostrar("V23_AUD_10_EXPOSICAO_CAUDA", exposicao, ["origem_tecnica", "idade_seg", "variante"])
# Curvas/q para os modelos usados nos contextos com maior PIORA de B.
prior_ids = contextos.filter(
    (F.col("segmento") == "ORIGENS_TECNICAS") & (F.col("comparacao") == "B_VS_A")
    & (F.col("idade_seg") > 0) & (F.col("contrib_delta_logloss_media_cliente") > 0)
).orderBy(F.desc("contrib_delta_logloss_media_cliente")).limit(10).select(
    F.col("id_modelo_var").alias("id_modelo")
).distinct()
curvas = kernels.drop("destinos_q").join(meta, "id_modelo").join(
    risco, ["id_modelo", "idade_seg"], "left"
)
aud_mostrar("V23_AUD_11_CURVAS_MODELOS_PRIORITARIOS", curvas.join(prior_ids, "id_modelo").select(
    "familia_modelo", "estado_modelo", "contexto_ajustado", "idade_seg", "tipos_limite",
    "top1_recalc", "prob_top1_recalc", "tv_recalc", "log_s_mistura",
    "max_tempo_modelo_seg", "n_clientes_treino_risco", "n_exatas_treino_risco",
), ["estado_modelo", "contexto_ajustado", "idade_seg"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Persistir somente auditoria e encerrar
# MAGIC Append em novas tabelas com id_auditoria. Nenhuma atualização das
# MAGIC fontes/modelos. Não é transação conjunta; somente o manifesto final
# MAGIC CONCLUIDA_AUDITORIA indica que todos os relatórios foram gravados.
# MAGIC Conclusão técnica NÃO autoriza promoção: revisar os resultados primeiro.
# MAGIC A inspeção de limites não demonstra ótimo global, calibração ou
# MAGIC qualidade do Multi-step; não reestima parâmetros nem muda bounds.

# COMMAND ----------

AUD_SAIDAS = {
    "resumos": AUD_CFG["prefixo_saida"] + "_resumos_hml",
    "parametros": AUD_CFG["prefixo_saida"] + "_parametros_hml",
    "casos": AUD_CFG["prefixo_saida"] + "_casos_hml",
    "execucoes": AUD_CFG["prefixo_saida"] + "_execucoes_hml",
}
AUD_TS = datetime.now(timezone.utc).isoformat()
for nome in AUD_SAIDAS.values():
    aud_nome_sql(nome)
    if nome in AUD_FONTES:
        raise RuntimeError("Destino da auditoria coincide com uma fonte.")


def aud_gravar(df, tabela):
    out = df.withColumn("id_experimento", F.lit(AUD_ID_EXP)).withColumn(
        "id_ajuste", F.lit(AUD_ID_FIT)
    ).withColumn("id_auditoria", F.lit(AUD_ID)).withColumn(
        "versao_auditoria", F.lit(AUD_VERSAO)
    ).withColumn("registrado_em_utc", F.lit(AUD_TS))
    if spark.catalog.tableExists(tabela):
        aud_cols(spark.table(tabela), {"id_auditoria"}, tabela)
        if spark.table(tabela).filter(F.col("id_auditoria") == AUD_ID).limit(1).count():
            raise RuntimeError("ID já gravado; execute o notebook inteiro para um novo ID.")
    out.write.format("delta").mode("append").saveAsTable(tabela)


AUD_RELATORIOS["V23_AUD_03_INTEGRIDADE"] = spark.createDataFrame(
    AUD_CHECKS, "verificacao string, n_divergencias long, status string"
)
# Guardar todos os resumos tipados como JSON por relatório permite um único
# destino para esquemas distintos. Os DataFrames tipados ficam em AUD_RELATORIOS.
json_resumos = None
contagens_relatorios = {}
for nome, df in AUD_RELATORIOS.items():
    contagens_relatorios[nome] = int(df.count())
    tmp = df.select(F.lit(nome).alias("relatorio"), F.to_json(
        F.struct(*[F.col(c) for c in df.columns]), {"ignoreNullFields": "false"}
    ).alias("dados_json"))
    json_resumos = tmp if json_resumos is None else json_resumos.unionByName(tmp)
if AUD_CFG["gravar"]:
    aud_gravar(json_resumos, AUD_SAIDAS["resumos"])
    aud_gravar(parametros_det, AUD_SAIDAS["parametros"])
    aud_gravar(casos, AUD_SAIDAS["casos"])
    manifesto_auditoria = spark.createDataFrame([(
        "CONCLUIDA_AUDITORIA", "AGUARDA_REVISAO_DOS_RESULTADOS", False,
        json.dumps(AUD_CFG, ensure_ascii=False, sort_keys=True),
        json.dumps(AUD_FONTES, sort_keys=True),
        json.dumps(contagens_relatorios, sort_keys=True),
        json.dumps({"scipy": scipy.__version__, "numpy": np.__version__,
                    "spark": spark.version}, sort_keys=True),
    )], "status string, status_revisao string, autoriza_promocao boolean, "
        "config_json string, fontes_versoes_json string, relatorios_json string, "
        "ambiente_json string")
    aud_gravar(manifesto_auditoria, AUD_SAIDAS["execucoes"])
print("\nV23_AUD_12_CONCLUSAO")
print("id_experimento:", AUD_ID_EXP)
print("id_ajuste:", AUD_ID_FIT)
print("id_auditoria:", AUD_ID)
print("Modo:", "PERSISTIDA" if AUD_CFG["gravar"] else "SOMENTE_EM_MEMORIA")
print("Status de revisão: AGUARDA_REVISAO_DOS_RESULTADOS; nenhuma promoção.")
print("Enviar: LIMITES_RESUMO, PARAMETROS_LIMITES, ERROS_POR_ALERTA,")
print("CONTEXTOS_PRIORITARIOS, CONCENTRACAO_CLIENTES e PARES_D_VS_B.")
print("Arquivos/modelos originais e tabela de negócio preservados.")
for k, v in AUD_SAIDAS.items():
    print(k + ":", v)
for df in AUD_BASE_CACHE:
    df.unpersist()
