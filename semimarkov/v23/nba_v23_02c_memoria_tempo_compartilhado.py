# Databricks notebook source
 %md
 # NBA V2.3 — Parte 02C: memória com tempo compartilhado

 Compara A × B × B2 no MESMO holdout persistido na Parte 02.
 B2 usa p(destino) de B e sobrevivências de A, alinhadas por destino.
 Não treina, não seleciona lambdas/limites e não executa Multi-step.
 Não altera V2.2, V2.3 anteriores nem a tabela de negócio.

 B2 é uma recombinação de parâmetros, NÃO um ajuste restrito otimizado.
 Onde B já reutiliza A, B2 também reutiliza A sem nenhuma alteração.
 Idade é o tempo JÁ transcorrido, não uma janela futura em dias.

# COMMAND ----------

import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import special, stats
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


C02_CFG = {
    "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
    "id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
    "tabela_ajustes": "ctg_dsti.renato_nba.nba_sm_v23_ajustes_hml",
    "prefixo_saida": "ctg_dsti.renato_nba.nba_sm_v23_02c",
    "gravar": True,
    "autotestes": True,
    "bootstrap_replicas": 2000,
    "semente": 20261008,
    "prob_baixa_triagem": 1e-6,  # Só relatório; não altera os scores.
    "max_modelos_driver": 5000,
    "max_entradas_kernels": 2000000,
    "max_agregados_driver": 100000,
    "max_celulas_bootstrap": 10000000,
    "max_linhas_exibir": 40,
    "casos_criticos_por_idade": 10,
    "tol_prob": 1e-9,
    "tol_metrica": 1e-7,
}
C02_VERSAO = "v2.3_02c_memoria_tempo_compartilhado_v1"
C02_AJUSTE_SUPORTADO = "v2.3_temporal_memoria_pesos_02_v1"
C02_BASE = "__BASE__"
C02_A = "A_REFERENCIA"
C02_B = "B_MEMORIA"
C02_B2 = "B2_MEMORIA_TEMPO_COMPARTILHADO"
C02_CHAVE = ["cd_bv", "passo", "idade_seg"]
C02_INICIO = datetime.now(timezone.utc).isoformat()
C02_ID = str(uuid.uuid4())
C02_FONTES = {}
C02_CHECKS = []
C02_RELATORIOS = {}
C02_CACHES = []

# COMMAND ----------

 %md
 ## 1. Núcleo numérico e alinhamento
 Conserva p_B, pi_B e r_B. Só troca mu/sigma por A.
 O alinhamento compara a partição de DESTINOS, nunca assume que o índice
 do grupo seja igual nos dois JSONs. Partições incompatíveis interrompem.
 A/B usam a mesma fórmula da Parte 02, incluindo o desempate estável.

# COMMAND ----------


def c02_hash(objeto: Any) -> str:
    texto = json.dumps(objeto, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def c02_validar_modelo(m: dict) -> None:
    if m.get("formato") != "sm_v23_lognormal_grupos":
        raise ValueError("Formato de modelo diferente do contrato da Parte 02.")
    if m.get("unidade_tempo") != "dias":
        raise ValueError("Unidade do modelo deve ser dias.")
    pi, mu, sigma, r, p0 = [np.asarray(m[k], float) for k in (
        "pi_grupo", "mu_grupo", "sigma_grupo", "r_destino_no_grupo", "p_destino"
    )]
    g = np.asarray(m["grupo"], int)
    nd, ng = len(m["destinos"]), len(pi)
    if int(m["n_grupos"]) != ng:
        raise ValueError("n_grupos incompatível com os parâmetros.")
    if not nd or not ng or len(set(m["destinos"])) != nd:
        raise ValueError("Destinos vazios ou duplicados.")
    if any(len(x) != nd for x in (g, r, p0)):
        raise ValueError("Dimensões dos destinos incompatíveis.")
    if any(len(x) != ng for x in (mu, sigma, m["grupos_nomes"])):
        raise ValueError("Dimensões dos grupos incompatíveis.")
    if not np.array_equal(np.unique(g), np.arange(ng)):
        raise ValueError("Grupos inválidos ou sem destinos.")
    if not all(np.isfinite(x).all() for x in (pi, mu, sigma, r, p0)):
        raise ValueError("Parâmetro não finito.")
    if min(pi.min(), sigma.min(), r.min(), p0.min()) <= 0:
        raise ValueError("Massa ou sigma não positivo.")
    if not np.isclose(pi.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("A massa dos grupos não fecha.")
    if not np.allclose(np.bincount(g, weights=r), 1, atol=1e-10, rtol=0):
        raise ValueError("A massa dentro dos grupos não fecha.")
    if not np.allclose(pi[g] * r, p0, atol=1e-10, rtol=0):
        raise ValueError("p_destino incompatível com pi e r.")
    if m["estrutura"]["destinos"] != m["destinos"]:
        raise ValueError("Estrutura/destinos divergentes.")
    if bool(m["alerta_limite"]) != bool(m["parametros_no_limite"]):
        raise ValueError("Flag e lista de limites divergem.")
    if m["estrutura"]["grupo"] != m["grupo"]:
        raise ValueError("Estrutura/grupos divergentes.")
    if not (np.isfinite(m["max_tempo_observado_dias"])
            and m["max_tempo_observado_dias"] > 0):
        raise ValueError("Máximo temporal inválido.")


def c02_compor_modelo(b: dict, a: dict) -> tuple[dict, list[int]]:
    """B2 = p_B × S_A; sem otimização, sem suavização ou novo piso."""
    c02_validar_modelo(a)
    c02_validar_modelo(b)
    if set(a["destinos"]) != set(b["destinos"]):
        raise ValueError("A e B precisam ter o mesmo vocabulário de destinos.")
    grupos_a = dict(zip(a["destinos"], a["grupo"]))
    grupos_b = np.asarray(b["grupo"], int)
    alinhamento = []
    for gb in range(b["n_grupos"]):
        membros = {d for d, g in zip(b["destinos"], grupos_b) if g == gb}
        candidatos = {grupos_a[d] for d in membros}
        if len(candidatos) != 1:
            raise ValueError("Um grupo de B cruza grupos temporais de A.")
        ga = next(iter(candidatos))
        if membros != {d for d, g in grupos_a.items() if g == ga}:
            raise ValueError("A partição temporal de A difere da de B.")
        alinhamento.append(int(ga))
    if len(set(alinhamento)) != len(a["pi_grupo"]):
        raise ValueError("Não há correspondência bijetiva entre grupos.")
    novo = copy.deepcopy(b)
    novo["mu_grupo"] = [a["mu_grupo"][g] for g in alinhamento]
    novo["sigma_grupo"] = [a["sigma_grupo"][g] for g in alinhamento]
    # Remapeia alertas temporais de A aos índices dos grupos de B.
    mapa_inverso = {ga: gb for gb, ga in enumerate(alinhamento)}
    limites = [p for p in b["parametros_no_limite"] if p.startswith("logit_")]
    for p in a["parametros_no_limite"]:
        if p.startswith(("mu_", "log_sigma_")):
            prefixo, ga = p.rsplit("_", 1)
            limites.append(f"{prefixo}_{mapa_inverso[int(ga)]}")
    novo["parametros_no_limite"] = limites
    novo["alerta_limite"] = bool(limites)
    novo["max_tempo_observado_dias"] = a["max_tempo_observado_dias"]
    novo["max_tempo_prob_fonte_dias"] = b["max_tempo_observado_dias"]
    novo["especificacao"] = "p_B_com_sobrevivencia_A_sem_reajuste"
    # Não atribui à recombinação objetivo/iterações de outro ajuste.
    for campo in ("objetivo_medio_penalizado", "n_iteracoes",
                  "regularizacao_prob", "regularizacao_tempo"):
        novo.pop(campo, None)
    novo["treinado_nesta_etapa"] = False
    c02_validar_modelo(novo)
    if not np.array_equal(novo["p_destino"], b["p_destino"]):
        raise RuntimeError("B2 alterou indevidamente p_B.")
    return novo, alinhamento


def c02_prever(m: dict, idade_seg: float) -> dict:
    if not np.isfinite(idade_seg) or idade_seg < 0:
        raise ValueError("Idade inválida.")
    pi, mu, sigma = [np.asarray(m[k], float) for k in (
        "pi_grupo", "mu_grupo", "sigma_grupo"
    )]
    g = np.asarray(m["grupo"], int)
    log_sf = np.zeros(len(pi), float)
    if idade_seg > 0:
        t = idade_seg / 86400.0
        log_sf = special.log_ndtr(-(np.log(t) - mu) / sigma)
        independente = stats.lognorm.logsf(t, s=sigma, scale=np.exp(mu))
        if not np.allclose(log_sf, independente, rtol=1e-10, atol=1e-8):
            raise ValueError("Sobrevivência difere da verificação independente.")
    log_mass = np.log(pi) + log_sf
    log_s_mix = float(special.logsumexp(log_mass))
    log_q = (log_mass - log_s_mix)[g] + np.log(m["r_destino_no_grupo"])
    q = np.exp(log_q)
    if not np.isfinite(log_q).all() or not np.isclose(q.sum(), 1, atol=1e-9):
        raise ValueError("Distribuição condicional inválida.")
    p0 = np.asarray(m["p_destino"], float)
    if idade_seg == 0 and not np.allclose(q, p0, atol=1e-10, rtol=0):
        raise ValueError("q(0) diferente de p_destino.")
    ordem = np.argsort(-q, kind="mergesort")
    ordem0 = np.argsort(-p0, kind="mergesort")
    return {
        "q": q, "log_q": log_q, "log_s_mistura": log_s_mix,
        "soma_quadrados_q": float(np.dot(q, q)),
        "variacao_temporal_tv": float(np.abs(q - p0).sum() / 2),
        "top5": [m["destinos"][int(j)] for j in ordem[:5]],
        "top1_sem_tempo": m["destinos"][int(ordem0[0])],
    }


def c02_bootstrap(pdf: pd.DataFrame, replicas: int, seed: int) -> list[dict]:
    """Mesmos clientes/casos em A/B/B2; reamostra clientes, não eventos."""
    rng = np.random.default_rng(seed)
    linhas = []
    comparacoes = (("B_VS_A", C02_A, C02_B), ("B2_VS_A", C02_A, C02_B2),
                   ("B2_VS_B", C02_B, C02_B2))
    for (segmento, idade), sub in pdf.groupby(["segmento", "idade_seg"], sort=True):
        ids = sorted(sub["cd_bv"].unique())
        if replicas * len(ids) > C02_CFG["max_celulas_bootstrap"]:
            raise RuntimeError("Bootstrap excede a proteção de memória do driver.")
        indices = rng.integers(0, len(ids), size=(replicas, len(ids)))
        for metrica in ("top1", "top5", "brier", "logloss"):
            p = sub.pivot(index="cd_bv", columns="variante", values=metrica).reindex(ids)
            if set(p.columns) != {C02_A, C02_B, C02_B2} or p.isna().any().any():
                raise ValueError("Bootstrap com pares incompletos.")
            n_casos = int(sub.loc[sub.variante.eq(C02_A), "n_casos"].sum())
            for nome, ref, var in comparacoes:
                a, b = p[ref].to_numpy(float), p[var].to_numpy(float)
                ganho = a - b if metrica in {"brier", "logloss"} else b - a
                boot = ganho[indices].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                linhas.append({
                    "segmento": str(segmento), "idade_seg": float(idade),
                    "comparacao": nome, "metrica": metrica, "n_clientes": len(ids),
                    "n_casos": n_casos, "media_ref": float(a.mean()),
                    "media_variante": float(b.mean()), "ganho_medio": float(ganho.mean()),
                    "ganho_ic95_lo": float(lo), "ganho_ic95_hi": float(hi),
                })
    return linhas


def c02_autotestes_numericos() -> None:
    a = {
        "formato": "sm_v23_lognormal_grupos", "unidade_tempo": "dias",
        "destinos": ["x", "y", "z"], "grupo": [0, 1, 1],
        "grupos_nomes": ["x", "compartilhado"], "n_grupos": 2,
        "pi_grupo": [0.4, 0.6], "r_destino_no_grupo": [1.0, 0.25, 0.75],
        "p_destino": [0.4, 0.15, 0.45], "mu_grupo": [-3.0, 1.0],
        "sigma_grupo": [0.8, 1.2], "max_tempo_observado_dias": 30.0,
        "parametros_no_limite": ["log_sigma_0"], "alerta_limite": True,
        "estrutura": {"destinos": ["x", "y", "z"], "grupo": [0, 1, 1]},
    }
    # Reordena destinos E os índices de grupo de propósito.
    b = copy.deepcopy(a)
    b.update(destinos=["z", "x", "y"], grupo=[0, 1, 0],
             grupos_nomes=["compartilhado", "x"], pi_grupo=[0.5, 0.5],
             r_destino_no_grupo=[0.6, 1.0, 0.4], p_destino=[0.3, 0.5, 0.2],
             mu_grupo=[-5.0, -8.0], sigma_grupo=[0.3, 0.2],
             parametros_no_limite=[], alerta_limite=False)
    b["estrutura"] = {"destinos": b["destinos"], "grupo": b["grupo"]}
    antes = c02_hash(b)
    b2, mapa = c02_compor_modelo(b, a)
    assert mapa == [1, 0] and c02_hash(b) == antes
    assert b2["parametros_no_limite"] == ["log_sigma_1"]
    for idade in (0.0, 1800.0, 86400.0, 604800.0):
        r2 = c02_prever(b2, idade)
        t = idade / 86400.0
        sf_a = np.ones(2) if not idade else stats.lognorm.sf(
            t, s=a["sigma_grupo"], scale=np.exp(a["mu_grupo"])
        )
        por_destino = dict(zip(a["destinos"], np.asarray(a["grupo"])))
        massa = np.asarray(b["p_destino"]) * np.array([
            sf_a[por_destino[d]] for d in b["destinos"]
        ])
        assert np.allclose(r2["q"], massa / massa.sum(), atol=1e-12)
        if idade == 0:
            rb = c02_prever(b, idade)
            assert np.array_equal(r2["q"], rb["q"])
            assert r2["top5"] == rb["top5"]
    copia_a, _ = c02_compor_modelo(a, a)
    assert np.allclose(c02_prever(copia_a, 1800)["q"], c02_prever(a, 1800)["q"])
    ruim = copy.deepcopy(a)
    ruim["destinos"] = ["x", "y", "outro"]
    ruim["estrutura"]["destinos"] = ruim["destinos"]
    try:
        c02_compor_modelo(b, ruim)
    except ValueError:
        pass
    else:
        raise AssertionError("Vocabulário incompatível não foi bloqueado.")
    if not np.isfinite(c02_prever(b, 1e6)["log_q"]).all():
        raise AssertionError("Log-space falhou na cauda.")
    print("02C: autotestes numéricos, alinhamento e identidade em idade zero OK.")


if C02_CFG["autotestes"]:
    c02_autotestes_numericos()

# COMMAND ----------

 %md
 ## 2. Fontes fixadas e contrato do ajuste
 Lê manifesto, modelos e validação; não relê eventos nem reconstrói memória.
 A seleção de casos/rotas é a persistida em A/B. A referência de auditoria
 não é usada para selecionar clientes, excluir erros ou calibrar parâmetros.

# COMMAND ----------


def c02_nome_sql(tabela: str) -> str:
    partes = tabela.split(".")
    if len(partes) != 3 or not all(p.replace("_", "").isalnum() for p in partes):
        raise ValueError("Use nome catalogo.schema.tabela sem crases.")
    return ".".join(f"`{p}`" for p in partes)


def c02_ler(tabela: str) -> DataFrame:
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela não encontrada: {tabela}")
    if tabela not in C02_FONTES:
        C02_FONTES[tabela] = int(spark.sql(
            f"DESCRIBE HISTORY {c02_nome_sql(tabela)} LIMIT 1"
        ).first()["version"])
    return spark.read.option("versionAsOf", C02_FONTES[tabela]).table(tabela)


def c02_colunas(df: DataFrame, nomes: set, rotulo: str) -> None:
    faltam = nomes - set(df.columns)
    if faltam:
        raise RuntimeError(f"{rotulo}: colunas ausentes: {sorted(faltam)}")


def c02_cache(df: DataFrame) -> DataFrame:
    df = df.persist(StorageLevel.DISK_ONLY)
    df.count()
    C02_CACHES.append(df)
    return df


def c02_exigir_zero(nome: str, erros: DataFrame) -> None:
    n = erros.count()
    C02_CHECKS.append((nome, int(n), "OK" if n == 0 else "FALHA"))
    if n:
        raise RuntimeError(f"02C interrompida: {nome}; {n} divergências. Sem promoção.")


def c02_filtro_id(df: DataFrame) -> DataFrame:
    c02_colunas(df, {"id_experimento", "id_ajuste", "hash_ajuste"}, "Artefato")
    return df.filter(
        (F.col("id_experimento") == C02_CFG["id_experimento"])
        & (F.col("id_ajuste") == C02_CFG["id_ajuste"])
    )


def c02_mostrar(nome: str, df: DataFrame, ordem: list | None = None) -> None:
    C02_RELATORIOS[nome] = df
    print(f"\n{nome}")
    mostrar = df.orderBy(*ordem) if ordem else df
    mostrar = mostrar.drop("cd_bv", "modelo_json")
    # Compacta só a exibição. As tabelas conservam a precisão original.
    for campo in mostrar.schema.fields:
        if campo.dataType.simpleString() in {"double", "float"}:
            valor = F.col(campo.name)
            if campo.name.endswith("_prob"):
                valor = F.when(valor.isNotNull(), F.format_string("%.6g", valor))
            else:
                valor = F.round(valor, 6)
            mostrar = mostrar.withColumn(campo.name, valor)
    mostrar.show(C02_CFG["max_linhas_exibir"], truncate=False)


def c02_autotestes_spark() -> None:
    """Verifica mapas, destino desconhecido e censura no Spark do cluster."""
    dados = [
        ("conhecido", "x", "exata", {"x": (0.25, math.log(0.25))}),
        ("desconhecido", "z", "exata", {"x": (0.25, math.log(0.25))}),
        ("censurado", None, "direita", {"x": (0.25, math.log(0.25))}),
    ]
    teste = spark.createDataFrame(dados, (
        "caso string, destino string, tipo string, "
        "massas map<string,struct<prob:double,log_prob:double>>"
    )).withColumn("real", F.element_at(F.col("massas"), F.col("destino")))
    teste = teste.withColumn("ll", F.when(F.col("tipo") == "exata",
        -F.greatest(F.coalesce(F.col("real.log_prob"), F.lit(math.log(1e-15))),
                    F.lit(math.log(1e-15)))))
    valores = {r["caso"]: r.asDict(recursive=True) for r in teste.collect()}
    assert np.isclose(valores["conhecido"]["ll"], -math.log(0.25))
    assert np.isclose(valores["desconhecido"]["ll"], -math.log(1e-15))
    assert valores["desconhecido"]["real"] is None
    assert valores["censurado"]["real"] is None and valores["censurado"]["ll"] is None
    print("02C: autotestes Spark de mapas, destino sem suporte e censura OK.")


if C02_CFG["autotestes"]:
    c02_autotestes_spark()


c02_manifestos = c02_ler(C02_CFG["tabela_ajustes"]).filter(
    (F.col("id_experimento") == C02_CFG["id_experimento"])
    & (F.col("id_ajuste") == C02_CFG["id_ajuste"])
    & (F.col("status") == "CONCLUIDO_TREINO_VALIDACAO")
).limit(2).collect()
if len(c02_manifestos) != 1:
    raise RuntimeError("Exige exatamente um manifesto de treino concluído para este ID.")
C02_MANIFESTO = c02_manifestos[0].asDict()
if C02_MANIFESTO["versao_ajuste"] != C02_AJUSTE_SUPORTADO:
    raise RuntimeError("Versão de ajuste não suportada; não adaptar silenciosamente.")
C02_FIT = json.loads(C02_MANIFESTO["config_json"])
C02_IDADES = [float(a) for a in C02_FIT["idades_seg"]]
C02_PISO = float(C02_FIT["logloss_piso"])
if C02_IDADES != [0.0, 1800.0, 86400.0, 604800.0] or not 0 < C02_PISO < 1:
    raise RuntimeError("Contrato de marcos/piso diferente do ajuste esperado.")
C02_LOG_PISO = math.log(C02_PISO)
C02_TAB_MODELOS = C02_FIT["tabelas_saida"]["tabela_modelos"]
C02_TAB_VALIDACAO = C02_FIT["tabelas_saida"]["tabela_validacao"]
c02_modelos_raw = c02_filtro_id(c02_ler(C02_TAB_MODELOS)).filter(
    F.col("variante").isin(C02_A, C02_B)
)
c02_valid_raw = c02_cache(c02_filtro_id(c02_ler(C02_TAB_VALIDACAO)).filter(
    F.col("variante").isin(C02_A, C02_B)
))
c02_colunas(c02_modelos_raw, {
    "variante", "estado", "contexto_modelo", "nivel", "reuso", "status_modelo",
    "modelo_json", "alerta_limite",
}, "Modelos")
C02_VAL_COLS = {
    "cd_bv", "passo", "estado", "contexto_modelo", "origem_tecnica", "idade_seg",
    "tipo_censura", "destino_real", "variante", "rota_modelo", "tem_previsao",
    "suporte_destino", "alerta_limite", "extrapolacao", "top1_previsto", "top5_previsto",
    "acerto_top1", "acerto_top5", "brier", "logloss_clip", "logloss_sem_tempo_clip",
    "acerto_top1_sem_tempo", "nll_clip_aplicado", "prob_destino_real", "variacao_temporal_tv",
}
c02_colunas(c02_valid_raw, C02_VAL_COLS, "Validação")
for df, nome in ((c02_modelos_raw, "modelos"), (c02_valid_raw, "validacao")):
    c02_exigir_zero(f"hash_{nome}", df.filter(
        ~F.col("hash_ajuste").eqNullSafe(F.lit(C02_MANIFESTO["hash_ajuste"]))
    ))
c02_exigir_zero("duplicidade_modelos", c02_modelos_raw.groupBy(
    "variante", "estado", "contexto_modelo"
).count().filter("count != 1"))
c02_exigir_zero("pares_A_B", c02_valid_raw.groupBy(*C02_CHAVE).agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v")
).filter("n != 2 OR v != 2"))
c02_exigir_zero("chaves_invalidas", c02_valid_raw.filter(
    F.col("cd_bv").isNull() | F.col("passo").isNull() | F.col("idade_seg").isNull()
    | ~F.col("idade_seg").isin(C02_IDADES) | F.col("estado").isNull()
    | F.col("contexto_modelo").isNull() | F.col("origem_tecnica").isNull()
    | F.col("tem_previsao").isNull() | F.col("tipo_censura").isNull()
    | ~F.col("tipo_censura").isin("exata", "direita")
    | ((F.col("tipo_censura") == "exata") & F.col("destino_real").isNull())
    | ((F.col("tipo_censura") == "direita") & F.col("destino_real").isNotNull())
))
if not c02_valid_raw.limit(1).count():
    raise RuntimeError("Validação vazia para os IDs informados.")
print("02C:", C02_ID, "| ajuste fonte:", C02_CFG["id_ajuste"])
print("Corte preservado:", C02_FIT["fonte_preparo"]["corte_estado_iso"])
print("Nenhum modelo será retreinado. Piso de log loss preservado:", C02_PISO)

# COMMAND ----------

 %md
 ## 3. Catálogo B2 e distribuições por modelo/idade
 Somente o catálogo pequeno vai ao driver. Casos de validação ficam no Spark.
 Limites temporais de A continuam sinalizados; não são removidos.
 Sem modelo na validação original continua sem modelo nesta comparação.

# COMMAND ----------

c02_rows = c02_modelos_raw.filter(
    ~F.col("reuso") & (F.col("status_modelo") == "AJUSTADO")
).limit(C02_CFG["max_modelos_driver"] + 1).collect()
if not c02_rows or len(c02_rows) > C02_CFG["max_modelos_driver"]:
    raise RuntimeError("Catálogo vazio ou maior que a proteção do driver.")
C02_MODELOS = {}
for row in c02_rows:
    m = json.loads(row["modelo_json"])
    c02_validar_modelo(m)
    if bool(m["alerta_limite"]) != bool(row["alerta_limite"]):
        raise RuntimeError("Alerta externo/JSON divergente.")
    chave = (row["variante"], row["estado"], row["contexto_modelo"])
    if chave[0] == C02_A and (chave[2] != C02_BASE or row["nivel"] != "ORIGEM"):
        raise RuntimeError("Referência A fora do nível ORIGEM.")
    if chave[0] == C02_B and row["nivel"] != "CONTEXTO":
        raise RuntimeError("B não reutilizado deve ser CONTEXTO.")
    C02_MODELOS[chave] = m

c02_candidatos = []
C02_COMPOSTOS = {}
for (familia, estado, contexto), b in C02_MODELOS.items():
    if familia != C02_B:
        continue
    a = C02_MODELOS.get((C02_A, estado, C02_BASE))
    if a is None:
        raise RuntimeError(f"Contexto sem pai A: {estado} / {contexto}")
    b2, mapa = c02_compor_modelo(b, a)
    b2["componentes_02c"] = {
        "id_ajuste_fonte": C02_CFG["id_ajuste"], "estado": estado,
        "contexto_prob": contexto, "familia_prob": C02_B,
        "contexto_tempo": C02_BASE, "familia_tempo": C02_A,
        "hash_modelo_prob": c02_hash(b), "hash_modelo_tempo": c02_hash(a),
        "grupo_B_para_A": mapa,
    }
    r0, rb0 = c02_prever(b2, 0.0), c02_prever(b, 0.0)
    if not np.array_equal(r0["q"], rb0["q"]) or r0["top5"] != rb0["top5"]:
        raise RuntimeError("B2 não reproduz B exatamente em idade zero.")
    C02_COMPOSTOS[(C02_B2, estado, contexto)] = b2
    c02_candidatos.append((
        C02_B2, estado, contexto, "CONTEXTO", "COMPOSTO_SEM_RETREINO",
        c02_hash(b), c02_hash(a), mapa, b2["alerta_limite"],
        b2["parametros_no_limite"], json.dumps(b2, ensure_ascii=False, allow_nan=False),
    ))
C02_MODELOS.update(C02_COMPOSTOS)
C02_MODELOS_SCHEMA = (
    "variante string, estado string, contexto_modelo string, nivel string, "
    "status_modelo string, hash_prob_fonte string, hash_tempo_fonte string, "
    "grupo_B_para_A array<int>, alerta_limite boolean, "
    "parametros_no_limite array<string>, modelo_json string"
)
c02_modelos_b2 = spark.createDataFrame(c02_candidatos, C02_MODELOS_SCHEMA)
C02_CHECKS.append(("catalogo_B2_p0_e_alinhamento", 0, "OK"))

C02_KERNEL_SCHEMA = (
    "familia_kernel string, estado string, contexto_kernel string, idade_seg double, "
    "kernel_ok boolean, alerta_limite_calc boolean, alerta_prob_fonte boolean, "
    "alerta_tempo_fonte boolean, parametros_limite_calc array<string>, "
    "max_tempo_prob_fonte_dias double, max_tempo_tempo_fonte_dias double, "
    "top5_calc array<string>, top1_sem_tempo_calc string, soma_quadrados_q double, "
    "variacao_temporal_calc double, log_s_mistura double, "
    "destinos_calc map<string,struct<prob:double,log_prob:double,log_p0:double>>"
)
entradas = sum(len(m["destinos"]) for m in C02_MODELOS.values()) * len(C02_IDADES)
if entradas > C02_CFG["max_entradas_kernels"]:
    raise RuntimeError("Catálogo de distribuições excede a proteção; não foi truncado.")
c02_kernels_rows = []
for (familia, estado, contexto), m in C02_MODELOS.items():
    prob_fonte = C02_MODELOS[(C02_B, estado, contexto)] if familia == C02_B2 else m
    tempo_fonte = C02_MODELOS[(C02_A, estado, C02_BASE)] if familia == C02_B2 else m
    alerta_prob = any(p.startswith("logit_") for p in prob_fonte["parametros_no_limite"])
    alerta_tempo = any(p.startswith(("mu_", "log_sigma_"))
                       for p in tempo_fonte["parametros_no_limite"])
    for idade in C02_IDADES:
        r = c02_prever(m, idade)
        destinos = {d: (float(r["q"][j]), float(r["log_q"][j]),
                        float(np.log(m["p_destino"][j])))
                    for j, d in enumerate(m["destinos"])}
        c02_kernels_rows.append((
            familia, estado, contexto, idade, True, m["alerta_limite"],
            alerta_prob, alerta_tempo, m["parametros_no_limite"],
            float(prob_fonte["max_tempo_observado_dias"]),
            float(tempo_fonte["max_tempo_observado_dias"]),
            r["top5"], r["top1_sem_tempo"], r["soma_quadrados_q"],
            r["variacao_temporal_tv"], r["log_s_mistura"], destinos,
        ))
c02_kernels = spark.createDataFrame(c02_kernels_rows, C02_KERNEL_SCHEMA)

# COMMAND ----------

 %md
 ## 4. Reavaliar A/B e B2 sem mudar casos, rotas ou cobertura
 A validação contém exatas e censuras. Censura recebe previsão, mas NÃO
 um rótulo de destino; só as exatas entram nas métricas de classificação.
 Destino desconhecido: p=0 e log loss no mesmo piso de antes.

# COMMAND ----------

c02_base_a = c02_valid_raw.filter(F.col("variante") == C02_A)
c02_base_b = c02_valid_raw.filter(F.col("variante") == C02_B)
c02_meta = ["estado", "contexto_modelo", "origem_tecnica", "tipo_censura", "destino_real"]
c02_meta_pares = c02_base_a.select(
    *C02_CHAVE, *[F.col(c).alias(f"a_{c}") for c in c02_meta]
).join(c02_base_b.select(*C02_CHAVE, *c02_meta), C02_CHAVE, "inner")
c02_dif = F.lit(False)
for c in c02_meta:
    c02_dif = c02_dif | ~F.col(c).eqNullSafe(F.col(f"a_{c}"))
c02_exigir_zero("mesmas_observacoes_A_B", c02_meta_pares.filter(c02_dif))
c02_exigir_zero("rotas_invalidas_A", c02_base_a.filter(
    F.col("rota_modelo").isNull()
    | ~F.col("rota_modelo").isin("ORIGEM_REFERENCIA", "SEM_MODELO_ORIGEM")
))
C02_ROTAS_B = ["CONTEXTO_AJUSTADO", "ORIGEM_NAO_TECNICA", "ORIGEM_SEM_MEMORIA",
               "ORIGEM_CONTEXTO_RARO", "ORIGEM_CONTEXTO_NAO_VISTO",
               "ORIGEM_FALHA_AJUSTE_CONTEXTO", "SEM_MODELO_ORIGEM"]
c02_exigir_zero("rotas_invalidas_B", c02_base_b.filter(
    F.col("rota_modelo").isNull() | ~F.col("rota_modelo").isin(C02_ROTAS_B)
    | ((F.col("rota_modelo") == "CONTEXTO_AJUSTADO") & ~F.col("origem_tecnica"))
))
c02_entrada = c02_valid_raw.select(*sorted(C02_VAL_COLS)).unionByName(
    c02_base_b.select(*sorted(C02_VAL_COLS)).withColumn("variante", F.lit(C02_B2))
).withColumn("familia_kernel", F.when(
    F.col("rota_modelo") == "CONTEXTO_AJUSTADO", F.col("variante")
).otherwise(F.lit(C02_A))).withColumn("contexto_kernel", F.when(
    F.col("rota_modelo") == "CONTEXTO_AJUSTADO", F.col("contexto_modelo")
).otherwise(F.lit(C02_BASE)))
c02_join = c02_entrada.join(
    F.broadcast(c02_kernels), ["familia_kernel", "estado", "contexto_kernel", "idade_seg"], "left"
)
c02_exigir_zero("disponibilidade_modelos", c02_join.filter(
    F.col("tem_previsao") != F.coalesce(F.col("kernel_ok"), F.lit(False))
))
c02_join = c02_join.withColumn("_real", F.element_at(
    F.col("destinos_calc"), F.col("destino_real")
)).withColumn("_prob", F.coalesce(F.col("_real.prob"), F.lit(0.0)))
c02_exata = F.col("tem_previsao") & (F.col("tipo_censura") == "exata")
c02_join = (
    c02_join
    .withColumn("calc_prob", F.when(c02_exata, F.col("_prob")))
    .withColumn("calc_suporte", F.when(c02_exata, F.col("_real").isNotNull()))
    .withColumn("calc_top1", F.when(c02_exata,
        (F.col("destino_real") == F.element_at("top5_calc", 1)).cast("double")))
    .withColumn("calc_top5", F.when(c02_exata,
        F.expr("array_contains(top5_calc, destino_real)").cast("double")))
    .withColumn("calc_brier", F.when(c02_exata,
        F.col("soma_quadrados_q") - 2 * F.col("_prob") + 1))
    .withColumn("calc_logloss", F.when(c02_exata,
        -F.greatest(F.coalesce(F.col("_real.log_prob"), F.lit(C02_LOG_PISO)), F.lit(C02_LOG_PISO))))
    .withColumn("calc_logloss_0", F.when(c02_exata,
        -F.greatest(F.coalesce(F.col("_real.log_p0"), F.lit(C02_LOG_PISO)), F.lit(C02_LOG_PISO))))
    .withColumn("calc_clip", F.when(c02_exata,
        F.col("_real").isNull() | (F.col("_real.log_prob") < C02_LOG_PISO)))
    .withColumn("calc_top1_0", F.when(c02_exata,
        (F.col("destino_real") == F.col("top1_sem_tempo_calc")).cast("double")))
    .withColumn("extrap_prob_fonte", F.coalesce(
        F.col("idade_seg") / 86400 > F.col("max_tempo_prob_fonte_dias"), F.lit(False)))
    .withColumn("extrap_tempo_fonte", F.coalesce(
        F.col("idade_seg") / 86400 > F.col("max_tempo_tempo_fonte_dias"), F.lit(False)))
)
c02_join = c02_cache(c02_join.drop("destinos_calc"))

# Confere A/B contra TODOS os valores originais, antes de comparar B2.
c02_replay = c02_join.filter(F.col("variante").isin(C02_A, C02_B))
c02_metricas_replay = {
    "prob_destino_real": "calc_prob", "brier": "calc_brier", "logloss_clip": "calc_logloss",
    "logloss_sem_tempo_clip": "calc_logloss_0", "acerto_top1": "calc_top1",
    "acerto_top5": "calc_top5", "acerto_top1_sem_tempo": "calc_top1_0",
    "variacao_temporal_tv": "variacao_temporal_calc",
}
c02_erros_replay = []
for original, calculado in c02_metricas_replay.items():
    tol = C02_CFG["tol_prob"] if original == "prob_destino_real" else C02_CFG["tol_metrica"]
    erro = (F.col(original).isNull() != F.col(calculado).isNull()) | (
        F.col(original).isNotNull() & ((F.abs(F.col(original) - F.col(calculado)) > tol)
                                       | F.isnan(calculado)))
    c02_erros_replay.append(F.sum(erro.cast("long")).alias(original))
for original, calculado in (("suporte_destino", "calc_suporte"),
                            ("nll_clip_aplicado", "calc_clip")):
    c02_erros_replay.append(F.sum(
        (~F.col(original).eqNullSafe(F.col(calculado))).cast("long")
    ).alias(original))
c02_erros_replay += [F.sum((F.col("tem_previsao") & (
    ~F.col("top5_previsto").eqNullSafe(F.col("top5_calc"))
    | ~F.col("top1_previsto").eqNullSafe(F.element_at("top5_calc", 1))
    | ~F.col("alerta_limite").eqNullSafe(F.col("alerta_limite_calc"))
    | ~F.col("extrapolacao").eqNullSafe(F.col("extrap_tempo_fonte"))
)).cast("long")).alias("ranking_flags")]
for nome, n in c02_replay.agg(*c02_erros_replay).first().asDict().items():
    C02_CHECKS.append((f"reproducao_{nome}", int(n or 0), "OK" if not n else "FALHA"))
    if n:
        raise RuntimeError(f"A/B não reproduzem a validação original: {nome}, n={n}")

c02_validacao = c02_join.select(
    *C02_CHAVE, "estado", "contexto_modelo", "origem_tecnica", "tipo_censura", "destino_real",
    "variante", F.col("rota_modelo").alias("rota_modelo_original"),
    "familia_kernel", "contexto_kernel",
    "tem_previsao", F.col("calc_suporte").alias("suporte_destino"),
    F.when(F.col("tem_previsao"), F.col("familia_kernel") != C02_A).alias("usa_prob_contextual"),
    F.when(F.col("tem_previsao"), F.when(F.col("variante") == C02_B2, C02_A)
           .otherwise(F.col("familia_kernel"))).alias("familia_tempo_utilizada"),
    "alerta_prob_fonte", "alerta_tempo_fonte", "parametros_limite_calc",
    "extrap_prob_fonte", "extrap_tempo_fonte",
    F.element_at("top5_calc", 1).alias("top1_previsto"),
    F.coalesce("top5_calc", F.array().cast("array<string>")).alias("top5_previsto"),
    F.col("calc_top1").alias("acerto_top1"), F.col("calc_top5").alias("acerto_top5"),
    F.col("calc_brier").alias("brier"), F.col("calc_logloss").alias("logloss_clip"),
    F.col("calc_logloss_0").alias("logloss_sem_tempo_clip"),
    F.col("calc_clip").alias("nll_clip_aplicado"),
    F.col("calc_prob").alias("prob_destino_real"),
    F.when(c02_exata, F.col("_real.log_prob")).alias("log_prob_destino_real"),
    F.col("variacao_temporal_calc").alias("variacao_temporal_tv"), "log_s_mistura",
)
c02_validacao = c02_cache(c02_validacao)
c02_exigir_zero("tres_variantes_mesma_cobertura", c02_validacao.groupBy(*C02_CHAVE).agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v"),
    F.countDistinct("tem_previsao").alias("p")
).filter("n != 3 OR v != 3 OR p != 1"))
# Identidade em idade zero inclui rankings, métricas e também casos censurados.
c02_b_zero = c02_validacao.filter((F.col("variante") == C02_B) & (F.col("idade_seg") == 0))
c02_b2_zero = c02_validacao.filter((F.col("variante") == C02_B2) & (F.col("idade_seg") == 0))
c02_zero_cols = [
    "top5_previsto", "prob_destino_real", "brier", "logloss_clip", "acerto_top1", "acerto_top5",
]
c02_zero = c02_b_zero.select(*C02_CHAVE, *c02_zero_cols).join(
    c02_b2_zero.select(*C02_CHAVE, *[F.col(c).alias(f"b2_{c}") for c in c02_zero_cols]),
    C02_CHAVE, "inner",
)
c02_zero_erro = F.lit(False)
for c in c02_zero_cols:
    c02_zero_erro = c02_zero_erro | ~F.col(c).eqNullSafe(F.col(f"b2_{c}"))
c02_exigir_zero("B2_igual_B_em_idade_zero", c02_zero.filter(c02_zero_erro))
c02_b2_fallback = c02_validacao.filter(
    (F.col("variante") == C02_B2)
    & (F.col("rota_modelo_original") != "CONTEXTO_AJUSTADO")
).select(*C02_CHAVE, *[F.col(c).alias(f"b2_{c}") for c in c02_zero_cols])
c02_a_fallback = c02_validacao.filter(F.col("variante") == C02_A).select(
    *C02_CHAVE, *c02_zero_cols
).join(c02_b2_fallback, C02_CHAVE, "inner")
c02_exigir_zero("B2_igual_A_nas_rotas_de_reuso", c02_a_fallback.filter(c02_zero_erro))
c02_checks = spark.createDataFrame(
    C02_CHECKS, "verificacao string, n_divergencias long, status string"
)
c02_mostrar("V23_02C_01_INTEGRIDADE", c02_checks)

# COMMAND ----------

 %md
 ## 5. Métricas e comparação pareada
 Sem previsões: ficam na cobertura, não nas médias de qualidade.
 Sem suporte ao destino: penalização mantida, sem exclusão das métricas.
 Brier/log loss menores são melhores; Top 1/5 maiores são melhores.
 Ganho positivo nos pares = melhora. ICs não são corrigidos por múltiplas
 comparações e descrevem este holdout de DESENVOLVIMENTO já examinado.

# COMMAND ----------


def c02_segmentos(df: DataFrame) -> DataFrame:
    return df.withColumn("segmento", F.explode(F.when(
        F.col("origem_tecnica"), F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS"))
    ).otherwise(F.array(F.lit("TODAS")))))


c02_rotas = c02_validacao.groupBy("variante", "idade_seg", "rota_modelo_original").agg(
    F.count("*").alias("n_observacoes"), F.countDistinct("cd_bv").alias("n_clientes")
)
c02_mostrar("V23_02C_02_ROTAS", c02_rotas, ["idade_seg", "variante", "rota_modelo_original"])
c02_exatas = c02_cache(c02_validacao.filter(F.col("tipo_censura") == "exata"))
c02_seg = c02_segmentos(c02_exatas)
c02_metricas = c02_seg.groupBy("segmento", "variante", "idade_seg").agg(
    F.count("*").alias("n_eventos"), F.sum(F.col("tem_previsao").cast("long")).alias("n_previstos"),
    F.sum(F.coalesce(F.col("suporte_destino").cast("long"), F.lit(0))).alias("n_suportados"),
    F.avg("acerto_top1").alias("top1"), F.avg("acerto_top5").alias("top5"),
    F.avg("brier").alias("brier"), F.avg("logloss_clip").alias("logloss"),
    F.sum(F.coalesce(F.col("nll_clip_aplicado").cast("long"), F.lit(0))).alias("n_logloss_clip"),
    F.avg(F.col("logloss_sem_tempo_clip") - F.col("logloss_clip")).alias("ganho_logloss_tempo"),
).withColumn("cobertura_modelo", F.col("n_previstos") / F.col("n_eventos"))
c02_mostrar("V23_02C_03_METRICAS", c02_metricas.filter(
    F.col("segmento") == "ORIGENS_TECNICAS"
).select(
    "variante", "idade_seg", "n_eventos", "n_previstos", "top1", "top5",
    "brier", "logloss", "n_logloss_clip",
),
    ["idade_seg", "variante"])
C02_RELATORIOS["V23_02C_03_METRICAS_COMPLETAS"] = c02_metricas
# Mesmos casos com A/B/B2; ausência numérica causaria interrupção acima.
c02_pc = c02_segmentos(c02_exatas.filter("tem_previsao")).groupBy(
    "segmento", "idade_seg", "cd_bv", "variante"
).agg(F.count("*").alias("n_casos"), F.avg("acerto_top1").alias("top1"),
      F.avg("acerto_top5").alias("top5"), F.avg("brier").alias("brier"),
      F.avg("logloss_clip").alias("logloss"))
c02_pc_rows = c02_pc.limit(C02_CFG["max_agregados_driver"] + 1).collect()
if not c02_pc_rows or len(c02_pc_rows) > C02_CFG["max_agregados_driver"]:
    raise RuntimeError("Bootstrap vazio ou acima da proteção; nenhum evento foi amostrado.")
# Apenas médias por cliente vão ao Pandas. Não exige Arrow/mapInPandas.
c02_pdf_pc = pd.DataFrame([r.asDict() for r in c02_pc_rows])
c02_boot = c02_bootstrap(c02_pdf_pc, C02_CFG["bootstrap_replicas"], C02_CFG["semente"])
C02_PARES_SCHEMA = (
    "segmento string, idade_seg double, comparacao string, metrica string, n_clientes long, "
    "n_casos long, media_ref double, media_variante double, ganho_medio double, "
    "ganho_ic95_lo double, ganho_ic95_hi double"
)
c02_pares = spark.createDataFrame(c02_boot, C02_PARES_SCHEMA)
for comp in ("B_VS_A", "B2_VS_A", "B2_VS_B"):
    c02_mostrar(f"V23_02C_04_{comp}", c02_pares.filter(
        (F.col("segmento") == "ORIGENS_TECNICAS") & (F.col("comparacao") == comp)
    ).drop("segmento", "comparacao"), ["idade_seg", "metrica"])

# COMMAND ----------

 %md
 ## 6. Probabilidades extremas e casos críticos
 Mesma triagem para todas as variantes. Não aumenta o piso da métrica.
 Os casos críticos são escolhidos pela piora ORIGINAL B−A, não pelo
 resultado de B2. Servem à investigação; a comparação global usa todos.

# COMMAND ----------

c02_extremos = c02_seg.filter("tem_previsao").groupBy("segmento", "idade_seg", "variante").agg(
    F.count("*").alias("n_casos"), F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum((~F.col("suporte_destino")).cast("long")).alias("n_destinos_sem_suporte"),
    F.sum(F.col("nll_clip_aplicado").cast("long")).alias("n_logloss_clip"),
    F.sum((F.col("prob_destino_real") < C02_CFG["prob_baixa_triagem"])
          .cast("long")).alias("n_prob_muito_baixa"),
    F.sum((F.col("suporte_destino") & (F.col("prob_destino_real") < C02_CFG["prob_baixa_triagem"]))
          .cast("long")).alias("n_baixas_com_suporte"),
    F.sum(F.col("extrap_tempo_fonte").cast("long")).alias("n_extrap_tempo_fonte"),
)
c02_mostrar("V23_02C_05_EXTREMOS", c02_extremos.filter(
    F.col("segmento") == "ORIGENS_TECNICAS"
).drop("segmento"), ["idade_seg", "variante"])
C02_RELATORIOS["V23_02C_05_EXTREMOS_COMPLETOS"] = c02_extremos
c02_wide = c02_exatas.filter("tem_previsao").groupBy(
    *C02_CHAVE, "estado", "contexto_modelo", "origem_tecnica", "destino_real"
).agg(*[
    F.max(F.when(F.col("variante") == variante, F.col(coluna))).alias(f"{curto}_{sufixo}")
    for variante, curto in ((C02_A, "a"), (C02_B, "b"), (C02_B2, "b2"))
    for coluna, sufixo in (("prob_destino_real", "prob"), ("log_prob_destino_real", "log_prob"),
                           ("logloss_clip", "logloss"))
]).withColumn("delta_b_a", F.col("b_logloss") - F.col("a_logloss")).withColumn(
    "delta_b2_a", F.col("b2_logloss") - F.col("a_logloss")
).withColumn("delta_b2_b", F.col("b2_logloss") - F.col("b_logloss"))
w_critico = Window.partitionBy("idade_seg").orderBy(F.desc("delta_b_a"), "cd_bv", "passo")
c02_criticos = c02_wide.filter(
    F.col("origem_tecnica") & (F.col("idade_seg") > 0) & (F.col("delta_b_a") > 0)
).withColumn("ranking_piora_original", F.row_number().over(w_critico)).filter(
    F.col("ranking_piora_original") <= C02_CFG["casos_criticos_por_idade"]
)
c02_mostrar("V23_02C_06_CASOS_CRITICOS", c02_criticos.select(
    "idade_seg", "ranking_piora_original", "estado", "contexto_modelo", "destino_real",
    "a_prob", "b_prob", "b2_prob", "delta_b2_a", "delta_b2_b"
), ["idade_seg", "ranking_piora_original"])
# Esta tabela interna inclui IDs; não os imprime nos relatórios agregados.
C02_RELATORIOS["V23_02C_06_CASOS_CRITICOS_COMPLETOS"] = c02_criticos

# COMMAND ----------

 %md
 ## 7. Persistência isolada e conclusão
 Novas tabelas 02C, append por id_02c; sem promoção automática.
 As escritas são separadas. Consuma somente IDs com manifesto concluído.
 Falha/reexecução: executar o notebook inteiro cria outro id_02c.

# COMMAND ----------

C02_CONFIG_EXEC = {
    "versao": C02_VERSAO, "config": C02_CFG, "fontes_delta": C02_FONTES,
    "hash_ajuste_fonte": C02_MANIFESTO["hash_ajuste"],
    "corte_estado_iso": C02_FIT["fonte_preparo"]["corte_estado_iso"],
    "logloss_piso": C02_PISO, "idades_seg": C02_IDADES,
    "coorte": "todos_os_casos_A_B_persistidos_sem_nova_amostragem",
    "regra_B2": "p_B_com_S_A; fallback_B_preservado; grupos_alinhados_por_destino",
    "estatistica": "media_por_cliente;bootstrap_pareado;IC95_nao_ajustado_multiplicidade",
    "spark": spark.version, "numpy": np.__version__, "pandas": pd.__version__,
    "scipy": scipy.__version__, "sem_retreino": True,
}
C02_HASH = c02_hash(C02_CONFIG_EXEC)
C02_SAIDAS = {nome: f"{C02_CFG['prefixo_saida']}_{nome}_hml" for nome in (
    "modelos", "validacao", "comparacoes", "resumos", "execucoes"
)}
if len(set(C02_SAIDAS.values())) != 5 or set(C02_SAIDAS.values()) & set(C02_FONTES):
    raise RuntimeError("Uma saída coincide com outra saída ou com uma fonte.")
for tabela in C02_SAIDAS.values():
    c02_nome_sql(tabela)


def c02_gravar(df: DataFrame, nome: str) -> None:
    tabela = C02_SAIDAS[nome]
    saida = (df.withColumn("id_experimento", F.lit(C02_CFG["id_experimento"]))
             .withColumn("id_ajuste_fonte", F.lit(C02_CFG["id_ajuste"]))
             .withColumn("id_02c", F.lit(C02_ID))
             .withColumn("versao_02c", F.lit(C02_VERSAO))
             .withColumn("hash_02c", F.lit(C02_HASH)))
    if spark.catalog.tableExists(tabela):
        existente = spark.table(tabela)
        c02_colunas(existente, {"id_02c"}, tabela)
        if existente.filter(F.col("id_02c") == C02_ID).limit(1).count():
            raise RuntimeError("Etapa já escrita neste id_02c; reexecute com novo ID.")
    saida.write.format("delta").mode("append").saveAsTable(tabela)


c02_resumos = None
for nome, df in C02_RELATORIOS.items():
    lote = df.select(F.lit(nome).alias("relatorio"), F.to_json(
        F.struct(*[F.col(c) for c in df.columns]), {"ignoreNullFields": "false"}
    ).alias("dados_json"))
    c02_resumos = lote if c02_resumos is None else c02_resumos.unionByName(lote)
if C02_CFG["gravar"]:
    c02_gravar(c02_modelos_b2, "modelos")
    c02_gravar(c02_validacao, "validacao")
    c02_gravar(c02_pares, "comparacoes")
    c02_gravar(c02_resumos, "resumos")
    c02_fim = spark.createDataFrame([(
        "CONCLUIDO_COMPARACAO_02C", False, False, C02_INICIO,
        datetime.now(timezone.utc).isoformat(),
        json.dumps(C02_CONFIG_EXEC, ensure_ascii=False, sort_keys=True, allow_nan=False),
        json.dumps(C02_SAIDAS, ensure_ascii=False, sort_keys=True),
    )], "status string, publicavel boolean, houve_retreino boolean, inicio_utc string, "
        "fim_utc string, config_json string, tabelas_json string")
    c02_gravar(c02_fim, "execucoes")
    print("\nV23_02C_07_CONCLUSAO: comparação persistida, sem promoção.")
else:
    print("\nV23_02C_07_CONCLUSAO: apenas em memória, sem manifesto persistido.")
print("id_02c:", C02_ID)
print("id_ajuste_fonte:", C02_CFG["id_ajuste"])
print("Tabelas 02C:", json.dumps(C02_SAIDAS, indent=2))
print("Nenhum modelo foi treinado; nenhuma tabela anterior foi sobrescrita.")
print("Próximo passo: analisar A/B/B2 antes de decidir sobre Multi-step.")
for df in C02_CACHES:
    df.unpersist()
