# Databricks notebook source
# MAGIC %md
# MAGIC # NBA V2.3 — Parte 02: treino temporal e validação pareada
# MAGIC
# MAGIC Execute depois da Parte 01 concluída. Não precisa reexecutá-la nem
# MAGIC manter suas variáveis em memória. Este notebook consome um ID explícito.
# MAGIC
# MAGIC A: referência; B: memória; C: ponderação; D: memória + ponderação.
# MAGIC Usa TODAS as observações elegíveis de treino, sem nova amostragem.
# MAGIC A é uma referência refeita sob a mesma especificação de B/C/D, não uma
# MAGIC reprodução numérica do ajuste V2.2 que limitava linhas por origem.
# MAGIC
# MAGIC Etapas: pais A/C -> contextos B/D -> validação do próximo estado RAW
# MAGIC nos mesmos marcos -> comparação por cliente. Não publica NBA, não
# MAGIC altera V2.2, não faz scoring de toda a população nem multi-step aqui.
# MAGIC O multi-step contextual será validado em etapa posterior: não usar
# MAGIC a antiga matriz P0 compartilhada para propagar estes novos modelos.

# COMMAND ----------

import hashlib
import json
import math
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import optimize, special
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


SM23_FIT_CFG = {
    "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
    "tabela_preparo": "ctg_dsti.renato_nba.nba_sm_v23_experimentos_hml",
    "tabela_modelos": "ctg_dsti.renato_nba.nba_sm_v23_modelos_hml",
    "tabela_validacao": "ctg_dsti.renato_nba.nba_sm_v23_validacao_destino_hml",
    "tabela_metricas": "ctg_dsti.renato_nba.nba_sm_v23_metricas_destino_hml",
    "tabela_pares": "ctg_dsti.renato_nba.nba_sm_v23_comparacao_pareada_hml",
    "tabela_ajustes": "ctg_dsti.renato_nba.nba_sm_v23_ajustes_hml",
    "idades_seg": [0.0, 1800.0, 86400.0, 604800.0],
    "gravar": True,
    "executar_autotestes": True,
    "max_linhas_por_origem": 150000,  # Proteção: para, NÃO amostra/trunca.
    "max_modelos_driver": 5000,
    "max_agregados_bootstrap": 100000,
    "bootstrap_replicas": 1000,
    "semente": 20261007,
    "max_linhas_exibir": 60,
}
SM23_NUM = {
    "min_eventos_origem": 80, "min_clientes_origem": 20,
    "min_eventos_grupo": 100, "min_clientes_grupo": 15,
    "max_grupos_proprios": 8,
    "sigma_min": 0.15, "sigma_max": 4.5, "maxiter": 800,
    "lambda_prob_base": 0.0001,
    "lambda_prob_contexto": 0.02,
    "lambda_tempo_contexto": 0.02,
}
VERSAO_AJUSTE = "v2.3_temporal_memoria_pesos_02_v1"
LOG_FLOOR = math.log(1e-15)  # Só para log loss REPORTADA; não altera q.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Modelo e regularização
# MAGIC
# MAGIC Para origem i e contexto c, p(j)=pi(grupo(j))*r(j|grupo(j)).
# MAGIC Duração Lognormal por grupo. A contribuição exata é log[p(j) f_g(t)],
# MAGIC e a censurada é log[sum_g pi_g S_g(censura)]. O peso da Parte 01
# MAGIC multiplica TODA essa contribuição. A função média divide por sum(w)
# MAGIC para não confundir a escala dos pesos com a força da penalidade.
# MAGIC
# MAGIC Grupos/vocabulário definidos por contagens do treino sem peso são os
# MAGIC mesmos para A/C e seus filhos B/D. Contextos herdam todos os destinos
# MAGIC da origem: não removemos destinos ausentes no contexto.
# MAGIC B regulariza em direção a A; D em direção a C (probabilidade e duração).
# MAGIC Os lambdas são uma especificação inicial fixa, não hiperparâmetros
# MAGIC otimizados no holdout. Alertas de limite são expostos, não ocultados.
# MAGIC
# MAGIC Contexto raro, não visto, desconhecido ou com falha numérica usa o pai
# MAGIC TEMPORAL da mesma família de peso, com rota explícita. Pai sem ajuste:
# MAGIC SEM_MODELO_ORIGEM, sem inventar um ranking estático.

# COMMAND ----------
VARIANTES = (
    "A_REFERENCIA", "B_MEMORIA", "C_PONDERACAO", "D_MEMORIA_PONDERACAO",
)
BASE_CTX = "__BASE__"
SEM_MEMORIA = "__SEM_MEMORIA__"
LOG_2PI = math.log(2.0 * math.pi)


def criar_estrutura(pdf, cfg):
    """Vocabulário/grupos definidos só no treino, iguais entre A/B/C/D."""
    ex = pdf.loc[pdf["tipo_censura"].eq("exata")]
    if len(ex) < cfg["min_eventos_origem"]:
        raise ValueError("SUPORTE_EVENTOS_ORIGEM")
    if ex["cd_bv"].nunique() < cfg["min_clientes_origem"]:
        raise ValueError("SUPORTE_CLIENTES_ORIGEM")
    suporte = ex.groupby("destino").agg(
        n=("cd_bv", "size"), clientes=("cd_bv", "nunique")
    ).reset_index().sort_values(
        ["n", "clientes", "destino"], ascending=[False, False, True]
    )
    proprios = suporte.loc[
        (suporte["n"] >= cfg["min_eventos_grupo"])
        & (suporte["clientes"] >= cfg["min_clientes_grupo"]), "destino"
    ].head(cfg["max_grupos_proprios"]).tolist()
    destinos = sorted(suporte["destino"].astype(str))
    nomes = list(proprios)
    mapa = {d: k for k, d in enumerate(proprios)}
    if any(d not in mapa for d in destinos):
        nomes.append("__GRUPO_COMPARTILHADO__")
        for d in destinos:
            mapa.setdefault(d, len(nomes) - 1)
    log_t = np.log(pdf["dur_min"].to_numpy(float) / 86400.0)
    return {
        "destinos": destinos, "grupo": [mapa[d] for d in destinos],
        "grupos_nomes": nomes,
        "limite_mu": [float(log_t.min() - 3), float(log_t.max() + 3)],
    }


def validar_modelo(m):
    pi = np.asarray(m["pi_grupo"], float)
    r = np.asarray(m["r_destino_no_grupo"], float)
    g = np.asarray(m["grupo"], int)
    sigma = np.asarray(m["sigma_grupo"], float)
    mu = np.asarray(m["mu_grupo"], float)
    if not (len(m["destinos"]) == len(r) == len(g)):
        raise ValueError("CONTRATO_DESTINOS")
    if len(set(m["destinos"])) != len(g) or g.min() < 0 or g.max() >= len(pi):
        raise ValueError("CONTRATO_GRUPOS")
    if len(pi) != len(sigma) or len(pi) != len(mu):
        raise ValueError("CONTRATO_PARAMETROS")
    if not all(np.isfinite(x).all() for x in (pi, r, sigma, mu)):
        raise ValueError("MODELO_NAO_FINITO")
    if min(pi.min(), r.min(), sigma.min()) <= 0:
        raise ValueError("PARAMETRO_NAO_POSITIVO")
    if not np.isclose(pi.sum(), 1.0, atol=1e-10):
        raise ValueError("MASSA_GRUPOS")
    if not np.allclose(np.bincount(g, weights=r, minlength=len(pi)), 1):
        raise ValueError("MASSA_DENTRO_GRUPO")
    if not np.allclose(pi[g] * r, m["p_destino"], atol=1e-10):
        raise ValueError("P_DESTINO_INCONSISTENTE")
    return True


def preparar_objetivo(pdf, peso_col, estrutura, cfg, pai=None):
    """L = -log-verossimilhança média ponderada + regularização explícita.

    Exata: log(pi_g * r_j|g * f_g(t)). Direita: log(sum_g pi_g S_g(c)).
    Mesmos pesos da Parte 01: não renormaliza por cliente/contexto.
    Dividir a FUNÇÃO por sum(w) estabiliza sua escala para a penalidade.
    """
    w = pdf[peso_col].to_numpy(float)
    t = pdf["dur_min"].to_numpy(float) / 86400.0
    exata = pdf["tipo_censura"].eq("exata").to_numpy()
    if not (np.isfinite(w).all() and np.isfinite(t).all()):
        raise ValueError("DADOS_NAO_FINITOS")
    if np.any(w <= 0) or np.any(t <= 0) or not exata.any():
        raise ValueError("PESO_TEMPO_OU_SAIDAS_INVALIDAS")
    destinos = estrutura["destinos"]
    jmap = {d: j for j, d in enumerate(destinos)}
    j = np.array([jmap[str(d)] for d in pdf.loc[exata, "destino"]], int)
    grupos = np.asarray(estrutura["grupo"], int)
    gex = grupos[j]
    ng = len(estrutura["grupos_nomes"])
    ne_logits = ng - 1
    we, xe = w[exata], np.log(t[exata])
    wc, xc = w[~exata], np.log(t[~exata])
    wg = np.bincount(gex, weights=we, minlength=ng)
    sx = np.bincount(gex, weights=we * xe, minlength=ng)
    sx2 = np.bincount(gex, weights=we * xe**2, minlength=ng)
    wj = np.bincount(j, weights=we, minlength=len(destinos))
    total_w = float(w.sum())
    if pai is not None:
        pi_prior = np.asarray(pai["pi_grupo"], float)
        r_prior = np.asarray(pai["r_destino_no_grupo"], float)
        mu_prior = np.asarray(pai["mu_grupo"], float)
        sigma_prior = np.asarray(pai["sigma_grupo"], float)
        lam_p = float(cfg["lambda_prob_contexto"])
        lam_t = float(cfg["lambda_tempo_contexto"])
    else:
        tamanhos = np.bincount(grupos, minlength=ng)
        pi_prior = tamanhos / len(destinos)  # Prior uniforme por destino.
        r_prior = 1.0 / tamanhos[grupos]
        mu_global = float(np.average(xe, weights=we))
        sigma_global = max(float(np.sqrt(np.average(
            (xe - mu_global)**2, weights=we
        ))), 0.5)
        mu_prior = np.full(ng, mu_global)
        sigma_prior = np.full(ng, sigma_global)
        for k in range(ng):
            mask = gex == k
            if mask.any():
                mu_prior[k] = np.average(xe[mask], weights=we[mask])
                sigma_prior[k] = np.sqrt(np.average(
                    (xe[mask] - mu_prior[k])**2, weights=we[mask]
                ))
        sigma_prior = np.clip(
            sigma_prior, cfg["sigma_min"] * 1.2, cfg["sigma_max"] * 0.8
        )
        lam_p = float(cfg["lambda_prob_base"])
        lam_t = 0.0
    eta_prior = np.log(sigma_prior)
    # MAP das proporções dentro dos grupos: o mesmo prior probabilístico
    # da função abaixo, em escala média ponderada (sem pseudocontagem bruta).
    suavizado = wj / total_w + lam_p * pi_prior[grupos] * r_prior
    soma_g = np.bincount(grupos, weights=suavizado, minlength=ng)
    r = suavizado / soma_g[grupos]
    pi0 = (wg / total_w + lam_p * pi_prior)
    pi0 /= pi0.sum()
    if pai is not None:
        pi0 = pi_prior.copy()
    theta0 = np.r_[np.log(pi0[:-1] / pi0[-1]), mu_prior, eta_prior]
    bounds = (
        [(-25.0, 25.0)] * ne_logits
        + [tuple(estrutura["limite_mu"])] * ng
        + [(math.log(cfg["sigma_min"]), math.log(cfg["sigma_max"]))] * ng
    )
    theta0 = np.clip(theta0, np.array(bounds)[:, 0] + 1e-8,
                     np.array(bounds)[:, 1] - 1e-8)
    const_r = float(np.dot(wj, np.log(r)))
    const_prior_r = -lam_p * float(np.dot(pi_prior[grupos] * r_prior, np.log(r)))

    def unpack(theta):
        logits = np.r_[theta[:ne_logits], 0.0]
        log_pi = logits - special.logsumexp(logits)
        mu = theta[ne_logits:ne_logits + ng]
        eta = theta[ne_logits + ng:]
        return log_pi, np.exp(log_pi), mu, eta, np.exp(eta)

    def objetivo(theta):
        log_pi, pi, mu, eta, sigma = unpack(theta)
        rss = np.maximum(sx2 - 2 * mu * sx + mu**2 * wg, 0.0)
        ll = float(np.sum(wg * log_pi - sx - wg * (eta + 0.5 * LOG_2PI)
                          - 0.5 * rss / sigma**2)) + const_r
        contagem_latente = wg.copy()
        grad_mu = (sx - mu * wg) / sigma**2
        grad_eta = -wg + rss / sigma**2
        if len(wc):
            z = (xc[:, None] - mu) / sigma
            log_sf = special.log_ndtr(-z)
            log_mix = special.logsumexp(log_pi + log_sf, axis=1)
            resp_w = wc[:, None] * np.exp(log_pi + log_sf - log_mix[:, None])
            mills = np.exp(-0.5 * z**2 - 0.5 * LOG_2PI - log_sf)
            ll += float(np.dot(wc, log_mix))
            contagem_latente += resp_w.sum(axis=0)
            grad_mu += (resp_w * mills / sigma).sum(axis=0)
            grad_eta += (resp_w * mills * z).sum(axis=0)
        grad_logits = (contagem_latente - total_w * pi)[:ne_logits]
        prob_pen = -lam_p * float(np.dot(pi_prior, log_pi)) + const_prior_r
        dm = (mu - mu_prior) / sigma_prior
        de = eta - eta_prior
        tempo_pen = 0.5 * lam_t * float(np.dot(pi_prior, dm**2 + de**2))
        grad = -np.r_[grad_logits, grad_mu, grad_eta] / total_w
        grad[:ne_logits] += lam_p * (pi - pi_prior)[:ne_logits]
        grad[ne_logits:ne_logits + ng] += lam_t * pi_prior * dm / sigma_prior
        grad[ne_logits + ng:] += lam_t * pi_prior * de
        valor = -ll / total_w + prob_pen + tempo_pen
        if not (np.isfinite(valor) and np.isfinite(grad).all()):
            raise FloatingPointError("OBJETIVO_NAO_FINITO")
        return float(valor), grad

    return objetivo, theta0, bounds, unpack, r, total_w


def ajustar_modelo(pdf, peso_col, estrutura, cfg, pai=None):
    objetivo, inicial, bounds, unpack, r, massa = preparar_objetivo(
        pdf, peso_col, estrutura, cfg, pai
    )
    resultados = []
    sementes = [inicial.copy()]
    alternativo = inicial.copy()
    ng = len(estrutura["grupos_nomes"])
    alternativo[-ng:] = np.clip(
        alternativo[-ng:] + 0.25, math.log(cfg["sigma_min"]) + 1e-5,
        math.log(cfg["sigma_max"]) - 1e-5
    )
    sementes.append(alternativo)
    for s in sementes:
        res = optimize.minimize(
            objetivo, s, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": cfg["maxiter"], "ftol": 1e-10,
                     "gtol": 1e-6, "maxls": 40},
        )
        resultados.append(res)
        if res.success and np.isfinite(res.fun):
            break
    bons = [x for x in resultados if x.success and np.isfinite(x.fun)]
    if not bons:
        raise ValueError("OTIMIZACAO_NAO_CONVERGIU: " + str(resultados[-1].message))
    res = min(bons, key=lambda x: x.fun)
    _, pi, mu, _, sigma = unpack(res.x)
    g = np.asarray(estrutura["grupo"], int)
    nomes_parametros = (
        [f"logit_{k}" for k in range(ng - 1)]
        + [f"mu_{k}" for k in range(ng)] + [f"log_sigma_{k}" for k in range(ng)]
    )
    limites = [nomes_parametros[k] for k, (v, (lo, hi)) in enumerate(
        zip(res.x, bounds)
    ) if min(abs(v - lo), abs(v - hi)) < 1e-4]
    m = {
        "formato": "sm_v23_lognormal_grupos", "unidade_tempo": "dias",
        "estrutura": estrutura, "destinos": estrutura["destinos"],
        "grupo": estrutura["grupo"], "grupos_nomes": estrutura["grupos_nomes"],
        "pi_grupo": pi.tolist(), "mu_grupo": mu.tolist(),
        "sigma_grupo": sigma.tolist(), "r_destino_no_grupo": r.tolist(),
        "p_destino": (pi[g] * r).tolist(), "n_grupos": ng,
        "peso_utilizado": peso_col, "massa_pesos": massa,
        "n_observacoes": len(pdf), "n_clientes": int(pdf["cd_bv"].nunique()),
        "max_tempo_observado_dias": float(pdf["dur_min"].max() / 86400),
        "alerta_limite": bool(limites), "parametros_no_limite": limites,
        "objetivo_medio_penalizado": float(res.fun), "n_iteracoes": int(res.nit),
        "regularizacao_prob": cfg["lambda_prob_contexto"] if pai else cfg["lambda_prob_base"],
        "regularizacao_tempo": cfg["lambda_tempo_contexto"] if pai else 0.0,
        "usa_pai": pai is not None,
    }
    validar_modelo(m)
    json.dumps(m, allow_nan=False)
    return m


def prever_probabilidades(m, idades_seg):
    """Retorna q(j|i,c,T>a), log q e variação total contra p(j|i,c)."""
    idade = np.asarray(idades_seg, float).reshape(-1)
    if np.any(idade < 0) or not np.isfinite(idade).all():
        raise ValueError("IDADE_INVALIDA")
    pi = np.asarray(m["pi_grupo"], float)
    mu = np.asarray(m["mu_grupo"], float)
    sigma = np.asarray(m["sigma_grupo"], float)
    g = np.asarray(m["grupo"], int)
    log_r = np.log(np.asarray(m["r_destino_no_grupo"], float))
    sf = np.zeros((len(idade), len(pi)), float)
    pos = idade > 0
    if pos.any():
        z = (np.log(idade[pos, None] / 86400) - mu) / sigma
        sf[pos] = special.log_ndtr(-z)
    log_mass = np.log(pi) + sf
    log_grupos = log_mass - special.logsumexp(log_mass, axis=1)[:, None]
    log_q = log_grupos[:, g] + log_r
    q = np.exp(log_q)
    if not np.allclose(q.sum(axis=1), 1, atol=1e-9):
        raise ValueError("MASSA_DESTINOS_NAO_FECHA")
    tv = 0.5 * np.abs(q - np.asarray(m["p_destino"])).sum(axis=1)
    return q, log_q, tv


def selecionar_modelo(registro, suporte, variante, origem, contexto, tecnica):
    """Fallback sempre para o modelo TEMPORAL da mesma família de peso."""
    familia = "C_PONDERACAO" if variante in (
        "C_PONDERACAO", "D_MEMORIA_PONDERACAO"
    ) else "A_REFERENCIA"
    pai = registro.get((familia, origem, BASE_CTX))
    if pai is None:
        return None, "SEM_MODELO_ORIGEM"
    if variante in ("A_REFERENCIA", "C_PONDERACAO"):
        return pai, "ORIGEM_REFERENCIA"
    if not tecnica:
        return pai, "ORIGEM_NAO_TECNICA"
    if contexto in (None, SEM_MEMORIA):
        return pai, "ORIGEM_SEM_MEMORIA"
    filho = registro.get((variante, origem, contexto))
    if filho is not None:
        return filho, "CONTEXTO_AJUSTADO"
    status = suporte.get((origem, contexto), "NAO_VISTO_NO_TREINO")
    if status == "CANDIDATO_CONTEXTUAL":
        return pai, "ORIGEM_FALHA_AJUSTE_CONTEXTO"
    if status == "NAO_VISTO_NO_TREINO":
        return pai, "ORIGEM_CONTEXTO_NAO_VISTO"
    return pai, "ORIGEM_CONTEXTO_RARO"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Autotestes numéricos e de roteamento
# MAGIC Executam antes das fontes reais. Não testam acesso Delta/permissões.

# COMMAND ----------


def sm23_testes_numericos():
    rng = np.random.default_rng(2718)
    dados = []
    for u in range(36):
        contexto = "pix" if u < 18 else "boleto"
        for _ in range(12):
            pix = rng.random() < (0.85 if contexto == "pix" else 0.15)
            censura = rng.random() < 0.1
            t = float(rng.lognormal(-4 if pix else -1, 0.7) * 86400)
            dados.append({"cd_bv": str(u), "estado": "login:::topo",
                          "contexto_modelo": contexto, "dur_min": t,
                          "tipo_censura": "direita" if censura else "exata",
                          "destino": None if censura else ("pix" if pix else "boleto"),
                          "peso_evento_treino": 1.0, "peso_cliente_treino": 1 / 12})
    pdf = pd.DataFrame(dados)
    cfg = dict(SM23_NUM, min_eventos_origem=20, min_clientes_origem=5,
               min_eventos_grupo=20, min_clientes_grupo=5)
    estrutura = criar_estrutura(pdf, cfg)
    obj, theta, _, _, _, _ = preparar_objetivo(
        pdf, "peso_cliente_treino", estrutura, cfg
    )
    erro = optimize.check_grad(lambda x: obj(x)[0], lambda x: obj(x)[1],
                               theta, epsilon=1e-6)
    if erro > 1e-4:
        raise RuntimeError(f"Gradiente incorreto no autoteste: {erro}")
    pai = ajustar_modelo(pdf, "peso_cliente_treino", estrutura, cfg)
    q, _, tv = prever_probabilidades(pai, [0, 1800, 86400])
    assert np.allclose(q.sum(axis=1), 1.0)
    assert np.allclose(q[0], pai["p_destino"]) and tv[0] < 1e-10
    filho = ajustar_modelo(pdf.loc[pdf.contexto_modelo.eq("boleto")],
                           "peso_cliente_treino", estrutura, cfg, pai)
    j = estrutura["destinos"].index("boleto")
    assert filho["p_destino"][j] > pai["p_destino"][j]
    registro = {("A_REFERENCIA", "login", BASE_CTX): pai,
                ("B_MEMORIA", "login", "boleto"): filho}
    assert selecionar_modelo(registro, {}, "B_MEMORIA", "login", "boleto", True)[0] is filho
    assert selecionar_modelo(registro, {}, "B_MEMORIA", "login", SEM_MEMORIA, True)[0] is pai
    assert selecionar_modelo(registro, {}, "B_MEMORIA", "login", "raro", True)[0] is pai
    print("V2.3 Parte 02 — gradiente, censura, massa e roteamento: OK.")


if SM23_FIT_CFG["executar_autotestes"]:
    sm23_testes_numericos()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Consumir somente o experimento de preparo concluído
# MAGIC Seleção explícita; nenhuma leitura das fontes brutas ou customers_query.
# MAGIC As versões Delta lidas nesta Parte 02 também ficam registradas.

# COMMAND ----------


def sql_nome(nome):
    partes = nome.split(".")
    if len(partes) != 3 or not all(partes) or any("`" in p for p in partes):
        raise ValueError("Esperado catalogo.schema.tabela, sem crase.")
    return ".".join(f"`{p}`" for p in partes)


def ler_snapshot(tabela):
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela não encontrada: {tabela}")
    versao = int(spark.sql(f"DESCRIBE HISTORY {sql_nome(tabela)} LIMIT 1").first()["version"])
    return spark.read.option("versionAsOf", versao).table(tabela), versao


def exigir_colunas(df, colunas, origem):
    faltantes = set(colunas) - set(df.columns)
    if faltantes:
        raise ValueError(f"{origem}: colunas ausentes: {sorted(faltantes)}")


def ha_linhas(df):
    return bool(df.limit(1).count())


def mostrar(nome, df):
    print(f"\n{nome}")
    df.show(SM23_FIT_CFG["max_linhas_exibir"], truncate=False)


if "spark" not in globals():
    raise RuntimeError("Execute no Databricks com sessão Spark.")
ID_EXPERIMENTO = str(SM23_FIT_CFG["id_experimento"]).strip()
uuid.UUID(ID_EXPERIMENTO)
ID_AJUSTE = str(uuid.uuid4())
preparos, V_MANIFESTO = ler_snapshot(SM23_FIT_CFG["tabela_preparo"])
registros = preparos.filter(
    (F.col("id_experimento") == ID_EXPERIMENTO)
    & (F.col("status") == "CONCLUIDO_PREPARO")
).limit(2).collect()
if len(registros) != 1:
    raise RuntimeError("O ID deve possuir exatamente um CONCLUIDO_PREPARO.")
PREPARO = registros[0].asDict()
POLITICA = json.loads(PREPARO["politica_json"])
if hashlib.sha256(PREPARO["politica_json"].encode()).hexdigest() != PREPARO["politica_sha256"]:
    raise RuntimeError("Hash da política de preparo não confere.")
if spark.conf.get("spark.sql.session.timeZone") != PREPARO["fuso_base"]:
    raise RuntimeError("Fuso diferente do preparo. Não altere datas para contornar o teste.")
if POLITICA["min_eventos_contexto"] != 80 or POLITICA["min_clientes_contexto"] != 20:
    print("Atenção: suporte contextual da Parte 01 tem limites personalizados.")

base_snapshot, V_BASE = ler_snapshot(PREPARO["tabela_base_v23"])
suporte_snapshot, V_SUPORTE = ler_snapshot(PREPARO["tabela_suporte_v23"])
base_v23 = base_snapshot.filter(F.col("id_experimento") == ID_EXPERIMENTO)
suporte_v23 = suporte_snapshot.filter(F.col("id_experimento") == ID_EXPERIMENTO)
exigir_colunas(base_v23, {
    "cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
    "validacao_cliente", "elegivel_ajuste", "origem_tecnica", "contexto_modelo",
    "peso_evento_treino", "peso_cliente_treino", "politica_sha256",
}, "Base V2.3")
exigir_colunas(suporte_v23, {
    "estado", "contexto_modelo", "status_suporte", "contexto_tem_suporte_candidato",
}, "Suporte V2.3")
if base_v23.count() != int(PREPARO["n_linhas_base"]):
    raise RuntimeError("Contagem da base diverge do manifesto de preparo.")
if ha_linhas(base_v23.filter(
    F.col("politica_sha256").isNull()
    | (F.col("politica_sha256") != PREPARO["politica_sha256"])
)):
    raise RuntimeError("A base mistura políticas de memória/pesos.")
if ha_linhas(base_v23.groupBy("cd_bv", "passo").count().filter("count != 1")):
    raise RuntimeError("Cliente/passo duplicado. Não deduplicar silenciosamente.")
if ha_linhas(suporte_v23.groupBy("estado", "contexto_modelo").count().filter("count != 1")):
    raise RuntimeError("Contexto duplicado na tabela de suporte.")
if ha_linhas(base_v23.groupBy("cd_bv").agg(
    F.countDistinct("validacao_cliente").alias("n")
).filter("n != 1")):
    raise RuntimeError("Cliente presente em ambos os splits.")

COLS_TREINO = ["cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
               "contexto_modelo", "origem_tecnica", "peso_evento_treino",
               "peso_cliente_treino"]
treino_v23 = base_v23.filter(
    ~F.col("validacao_cliente") & F.col("elegivel_ajuste")
).select(*COLS_TREINO).persist(StorageLevel.DISK_ONLY)
holdout_v23 = base_v23.filter(
    F.col("validacao_cliente") & F.col("elegivel_ajuste")
).select("cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
         "contexto_modelo", "origem_tecnica")
for df, nome in ((treino_v23, "treino"), (holdout_v23, "holdout")):
    if not ha_linhas(df):
        raise RuntimeError(f"{nome} vazio.")
    invalidas = (F.col("estado").isNull() | F.col("contexto_modelo").isNull()
                 | F.col("dur_min").isNull() | F.isnan("dur_min")
                 | (F.col("dur_min") <= 0) | (F.abs("dur_min") == float("inf"))
                 | ~F.col("tipo_censura").isin("exata", "direita")
                 | F.col("tipo_censura").isNull()
                 | ((F.col("tipo_censura") == "exata") & F.col("destino").isNull())
                 | ((F.col("tipo_censura") == "direita") & F.col("destino").isNotNull()))
    if ha_linhas(df.filter(invalidas)):
        raise RuntimeError(f"Observação elegível inválida no {nome}.")
for col in ("peso_evento_treino", "peso_cliente_treino"):
    if ha_linhas(treino_v23.filter(
        F.col(col).isNull() | F.isnan(col) | (F.col(col) <= 0)
        | (F.abs(F.col(col)) == float("inf"))
    )):
        raise RuntimeError(f"Peso inválido: {col}")
if ha_linhas(treino_v23.groupBy("cd_bv", "estado").agg(
    F.sum("peso_cliente_treino").alias("s")
).filter(F.abs(F.col("s") - 1) > 1e-8)):
    raise RuntimeError("Pesos não somam 1 por cliente-origem.")
max_grupo = treino_v23.groupBy("estado").count().agg(F.max("count")).first()[0]
if max_grupo > SM23_FIT_CFG["max_linhas_por_origem"]:
    raise RuntimeError(
        f"Maior origem tem {max_grupo} linhas; excede proteção de memória. "
        "Revisar capacidade, não truncar nem amostrar silenciosamente."
    )
print("Experimento:", ID_EXPERIMENTO, "| Ajuste:", ID_AJUSTE)
print("Corte preservado:", PREPARO["corte_estado_iso"])
print("Linhas de treino:", treino_v23.count(), "| maior origem:", max_grupo)
mostrar("V23_20_DADOS_AJUSTE", treino_v23.groupBy("tipo_censura").agg(
    F.count("*").alias("n_linhas"), F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum("peso_cliente_treino").alias("massa_cliente_origem"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ajustar pais A/C e persistir antes de ajustar os contextos
# MAGIC Spark distribui por origem. Pandas recebe SOMENTE um grupo, não toda a base.
# MAGIC O limite de memória é uma proteção, não um parâmetro de amostragem.

# COMMAND ----------

SCHEMA_MODELO = (
    "variante string, nivel string, estado string, contexto_modelo string, "
    "status_modelo string, n_observacoes long, n_saidas_exatas long, "
    "n_clientes long, massa_pesos double, n_grupos long, alerta_limite boolean, "
    "modelo_json string, detalhe string, reuso boolean"
)


def linha_modelo(pdf, variante, nivel, contexto, peso):
    return dict(variante=variante, nivel=nivel, estado=str(pdf["estado"].iloc[0]),
                contexto_modelo=contexto, status_modelo="NAO_AJUSTADO",
                n_observacoes=int(len(pdf)),
                n_saidas_exatas=int(pdf["tipo_censura"].eq("exata").sum()),
                n_clientes=int(pdf["cd_bv"].nunique()),
                massa_pesos=float(pdf[peso].sum()), n_grupos=0,
                alerta_limite=False, modelo_json=None, detalhe="", reuso=False)


def preencher_modelo(linha, modelo):
    linha.update(status_modelo="AJUSTADO", n_grupos=int(modelo["n_grupos"]),
                 alerta_limite=bool(modelo["alerta_limite"]),
                 modelo_json=json.dumps(modelo, allow_nan=False),
                 detalhe=",".join(modelo["parametros_no_limite"]))
    return linha


def ajustar_pais(pdf):
    pdf = pdf.sort_values(["cd_bv", "passo"], kind="mergesort")
    linhas = []
    try:
        estrutura = criar_estrutura(pdf, SM23_NUM)
        erro_suporte = None
    except ValueError as erro:
        estrutura, erro_suporte = None, str(erro)
    for variante, peso in (("A_REFERENCIA", "peso_evento_treino"),
                           ("C_PONDERACAO", "peso_cliente_treino")):
        linha = linha_modelo(pdf, variante, "ORIGEM", BASE_CTX, peso)
        if erro_suporte:
            linha["detalhe"] = erro_suporte
        else:
            try:
                preencher_modelo(linha, ajustar_modelo(pdf, peso, estrutura, SM23_NUM))
            except (ValueError, FloatingPointError) as erro:
                linha["detalhe"] = f"{type(erro).__name__}: {erro}"[:1000]
        linhas.append(linha)
    return pd.DataFrame(linhas)


HASH_AJUSTE = hashlib.sha256(json.dumps(
    {"versao": VERSAO_AJUSTE, "numerico": SM23_NUM,
     "politica_preparo": PREPARO["politica_sha256"],
     "idades": SM23_FIT_CFG["idades_seg"]}, sort_keys=True
).encode()).hexdigest()


def guardar(df, tabela, etapa):
    """Append isolado; não sobrescreve nenhuma execução. Releitura materializa."""
    saida = (df.withColumn("id_experimento", F.lit(ID_EXPERIMENTO))
             .withColumn("id_ajuste", F.lit(ID_AJUSTE))
             .withColumn("versao_ajuste", F.lit(VERSAO_AJUSTE))
             .withColumn("hash_ajuste", F.lit(HASH_AJUSTE))
             .withColumn("etapa_artefato", F.lit(etapa)))
    if SM23_FIT_CFG["gravar"]:
        if tabela in {PREPARO["tabela_base_v23"], PREPARO["tabela_suporte_v23"],
                      PREPARO["tabela_base_v22"], PREPARO["tabela_config_v22"],
                      SM23_FIT_CFG["tabela_preparo"]}:
            raise ValueError("Uma tabela de saída coincide com uma fonte.")
        if spark.catalog.tableExists(tabela):
            existente = spark.table(tabela)
            exigir_colunas(existente, {"id_ajuste", "etapa_artefato"}, tabela)
            if ha_linhas(existente.filter(
                (F.col("id_ajuste") == ID_AJUSTE) & (F.col("etapa_artefato") == etapa)
            )):
                raise RuntimeError(
                    "Etapa já escrita neste ID. Reexecute a Parte 02 "
                    "inteira para novo ID."
                )
        saida.write.format("delta").mode("append").saveAsTable(tabela)
        return spark.table(tabela).filter(
            (F.col("id_ajuste") == ID_AJUSTE) & (F.col("etapa_artefato") == etapa)
        )
    saida = saida.persist(StorageLevel.DISK_ONLY)
    saida.count()
    return saida


pais_brutos = treino_v23.groupBy("estado").applyInPandas(
    ajustar_pais, schema=SCHEMA_MODELO
)
pais = guardar(pais_brutos, SM23_FIT_CFG["tabela_modelos"], "PAIS")
mostrar("V23_21_MODELOS_ORIGEM", pais.groupBy(
    "variante", "status_modelo", "alerta_limite"
).agg(F.count("*").alias("n_modelos"), F.sum("n_observacoes").alias("n_linhas")))
mostrar("V23_21_FALHAS_ORIGEM", pais.filter("status_modelo != 'AJUSTADO'").groupBy(
    "variante", "detalhe"
).count().orderBy(F.desc("count")))
linhas_pais = pais.filter("status_modelo = 'AJUSTADO'").select(
    "variante", "estado", "modelo_json"
).limit(SM23_FIT_CFG["max_modelos_driver"] + 1).collect()
if len(linhas_pais) > SM23_FIT_CFG["max_modelos_driver"]:
    raise RuntimeError("Muitos modelos para broadcast. Rever dimensionamento.")
PAIS = {(r["variante"], r["estado"]): json.loads(r["modelo_json"]) for r in linhas_pais}
if not all(any(v == familia for v, _ in PAIS) for familia in ("A_REFERENCIA", "C_PONDERACAO")):
    raise RuntimeError("Nenhum pai ajustado em uma das duas famílias. Rever falhas.")
BC_PAIS = spark.sparkContext.broadcast(PAIS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ajustar somente contextos candidatos B/D
# MAGIC Sem memória/raros/não vistos continuam usando o pai temporal na avaliação.
# MAGIC Não utilizamos o holdout para criar grupos, suporte, prior ou pesos.

# COMMAND ----------


def ajustar_contextos(pdf):
    pdf = pdf.sort_values(["cd_bv", "passo"], kind="mergesort")
    origem, contexto = str(pdf["estado"].iloc[0]), str(pdf["contexto_modelo"].iloc[0])
    linhas = []
    for variante, familia, peso in (
        ("B_MEMORIA", "A_REFERENCIA", "peso_evento_treino"),
        ("D_MEMORIA_PONDERACAO", "C_PONDERACAO", "peso_cliente_treino"),
    ):
        linha = linha_modelo(pdf, variante, "CONTEXTO", contexto, peso)
        pai = BC_PAIS.value.get((familia, origem))
        if pai is None:
            linha["detalhe"] = "PAI_SEM_AJUSTE"
        else:
            try:
                preencher_modelo(linha, ajustar_modelo(
                    pdf, peso, pai["estrutura"], SM23_NUM, pai
                ))
            except (ValueError, FloatingPointError) as erro:
                linha["detalhe"] = f"{type(erro).__name__}: {erro}"[:1000]
        linhas.append(linha)
    return pd.DataFrame(linhas)


candidatos = suporte_v23.filter(F.col("contexto_tem_suporte_candidato")).select(
    "estado", "contexto_modelo"
)
base_contextos = treino_v23.join(candidatos, ["estado", "contexto_modelo"], "inner")
if ha_linhas(base_contextos):
    contextos_brutos = base_contextos.groupBy("estado", "contexto_modelo").applyInPandas(
        ajustar_contextos, schema=SCHEMA_MODELO
    )
else:
    contextos_brutos = spark.createDataFrame([], SCHEMA_MODELO)
contextos = guardar(contextos_brutos, SM23_FIT_CFG["tabela_modelos"], "CONTEXTOS")
mostrar("V23_22_MODELOS_CONTEXTO", contextos.groupBy(
    "variante", "status_modelo", "alerta_limite"
).agg(F.count("*").alias("n_modelos"), F.sum("n_observacoes").alias("n_linhas")))
mostrar("V23_22_FALHAS_CONTEXTO", contextos.filter("status_modelo != 'AJUSTADO'").groupBy(
    "variante", "detalhe"
).count().orderBy(F.desc("count")))

# Registrar explicitamente que B reutiliza A e D reutiliza C nas origens.
reusos = pais.select(*spark.createDataFrame([], SCHEMA_MODELO).columns).withColumn(
    "variante", F.when(F.col("variante") == "A_REFERENCIA", "B_MEMORIA")
    .otherwise("D_MEMORIA_PONDERACAO")
).withColumn("reuso", F.lit(True))
reusos = guardar(reusos, SM23_FIT_CFG["tabela_modelos"], "PAIS_REUTILIZADOS")
REGISTRO = {(v, origem, BASE_CTX): m for (v, origem), m in PAIS.items()}
ctx_coletados = contextos.filter("status_modelo = 'AJUSTADO'").select(
    "variante", "estado", "contexto_modelo", "modelo_json"
).limit(SM23_FIT_CFG["max_modelos_driver"] + 1).collect()
if len(ctx_coletados) > SM23_FIT_CFG["max_modelos_driver"]:
    raise RuntimeError("Muitos contextos para broadcast. Rever dimensionamento.")
for r in ctx_coletados:
    REGISTRO[(r["variante"], r["estado"], r["contexto_modelo"])] = json.loads(r["modelo_json"])
SUPORTE = {(r["estado"], r["contexto_modelo"]): r["status_suporte"]
           for r in suporte_v23.select("estado", "contexto_modelo", "status_suporte").collect()}
BC_REGISTRO = spark.sparkContext.broadcast(REGISTRO)
BC_SUPORTE = spark.sparkContext.broadcast(SUPORTE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validação nos mesmos marcos e observações
# MAGIC O marco a considera dur_min > a. Censuras continuam na tabela técnica,
# MAGIC mas não viram rótulo de "nenhuma ação" e não entram no acerto do destino.
# MAGIC A comparação de destino usa saídas exatas observadas antes do corte;
# MAGIC não certifica probabilidade de agir em N dias nem valida multi-step.
# MAGIC Log loss usa piso 1e-15 explícito e contagem de clipping. Destino não
# MAGIC aprendido tem p=0 e é penalizado: não o excluímos silenciosamente.
# MAGIC Registros sem modelo ficam nas contagens de cobertura.

# COMMAND ----------

SCHEMA_VALIDACAO = (
    "cd_bv string, passo long, estado string, contexto_modelo string, "
    "origem_tecnica boolean, idade_seg double, tipo_censura string, "
    "destino_real string, variante string, rota_modelo string, "
    "tem_previsao boolean, suporte_destino boolean, alerta_limite boolean, "
    "extrapolacao boolean, top1_previsto string, top5_previsto array<string>, "
    "acerto_top1 double, acerto_top5 double, brier double, logloss_clip double, "
    "logloss_sem_tempo_clip double, acerto_top1_sem_tempo double, "
    "nll_clip_aplicado boolean, prob_destino_real double, variacao_temporal_tv double"
)
VALID_COLS = [campo.strip().split()[0] for campo in SCHEMA_VALIDACAO.split(",")]


def validar_lotes(iterator: Iterator[pd.DataFrame]):
    registro, suporte = BC_REGISTRO.value, BC_SUPORTE.value
    for pdf in iterator:
        saidas = []
        for (origem, contexto, tecnica), grupo_df in pdf.groupby(
            ["estado", "contexto_modelo", "origem_tecnica"], sort=False, dropna=False
        ):
            for idade in SM23_FIT_CFG["idades_seg"]:
                casos = grupo_df.loc[grupo_df["dur_min"] > idade]
                if casos.empty:
                    continue
                for variante in VARIANTES:
                    m, rota = selecionar_modelo(registro, suporte, variante,
                                               str(origem), str(contexto), bool(tecnica))
                    q = logq = p0 = None
                    top5, ordem0 = [], []
                    tv, extrap = None, False
                    if m is not None:
                        try:
                            q2, lq2, tv2 = prever_probabilidades(m, [idade])
                            q, logq, tv = q2[0], lq2[0], float(tv2[0])
                            p0 = np.asarray(m["p_destino"], float)
                            ordem = np.argsort(-q, kind="mergesort")
                            top5 = [m["destinos"][int(j)] for j in ordem[:5]]
                            ordem0 = np.argsort(-p0, kind="mergesort")
                            jmap = {d: j for j, d in enumerate(m["destinos"])}
                            extrap = idade / 86400 > m["max_tempo_observado_dias"]
                        except (ValueError, FloatingPointError):
                            m, q, logq = None, None, None
                            rota = "FALHA_NUMERICA_INFERENCIA"
                    for row in casos.itertuples(index=False):
                        exata = row.tipo_censura == "exata"
                        destino = str(row.destino) if exata else None
                        out = dict(cd_bv=str(row.cd_bv), passo=int(row.passo),
                                   estado=str(origem), contexto_modelo=str(contexto),
                                   origem_tecnica=bool(tecnica), idade_seg=float(idade),
                                   tipo_censura=str(row.tipo_censura), destino_real=destino,
                                   variante=variante, rota_modelo=rota,
                                   tem_previsao=m is not None, suporte_destino=None,
                                   alerta_limite=bool(m["alerta_limite"]) if m else False,
                                   extrapolacao=bool(extrap),
                                   top1_previsto=top5[0] if m else None,
                                   top5_previsto=top5 if m else [],
                                   acerto_top1=None, acerto_top5=None, brier=None,
                                   logloss_clip=None, logloss_sem_tempo_clip=None,
                                   acerto_top1_sem_tempo=None, nll_clip_aplicado=None,
                                   prob_destino_real=None, variacao_temporal_tv=tv if m else None)
                        if m is not None and exata:
                            j = jmap.get(destino)
                            prob = float(q[j]) if j is not None else 0.0
                            lq = float(logq[j]) if j is not None else -np.inf
                            lp0 = float(np.log(p0[j])) if j is not None else -np.inf
                            out.update(
                                suporte_destino=j is not None,
                                acerto_top1=float(destino == top5[0]),
                                acerto_top5=float(destino in top5),
                                brier=float(np.dot(q, q) - 2 * prob + 1),
                                logloss_clip=float(-max(lq, LOG_FLOOR)),
                                logloss_sem_tempo_clip=float(-max(lp0, LOG_FLOOR)),
                                acerto_top1_sem_tempo=float(
                                    destino == m["destinos"][int(ordem0[0])]
                                ),
                                nll_clip_aplicado=bool(lq < LOG_FLOOR),
                                prob_destino_real=prob,
                            )
                        saidas.append(out)
        result = pd.DataFrame(saidas, columns=VALID_COLS)
        if not result.empty:
            result["passo"] = pd.array(result["passo"], dtype="Int64")
        yield result


arrow_anterior = spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")
try:
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "3000")
    validacao_bruta = holdout_v23.mapInPandas(validar_lotes, SCHEMA_VALIDACAO)
    validacao_v23 = guardar(validacao_bruta, SM23_FIT_CFG["tabela_validacao"], "VALIDACAO")
finally:
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", arrow_anterior)
if ha_linhas(validacao_v23.groupBy("cd_bv", "passo", "idade_seg").agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v")
).filter("n != 4 OR v != 4")):
    raise RuntimeError("Validação não preservou quatro variantes por observação/marco.")
mostrar("V23_23_ROTAS_HOLDOUT", validacao_v23.groupBy(
    "variante", "idade_seg", "rota_modelo"
).agg(F.count("*").alias("n_observacoes"), F.countDistinct("cd_bv").alias("n_clientes"))
 .orderBy("idade_seg", "variante", F.desc("n_observacoes")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Cobertura, desempenho e comparação pareada por cliente
# MAGIC Todos os destinos fora do suporte são penalizados com p=0; as métricas
# MAGIC de qualidade ficam condicionadas à existência de previsão, cuja
# MAGIC cobertura aparece ao lado. Pares usam INTERSEÇÃO de casos modelados
# MAGIC pelas quatro variantes. NLL/Brier menores são melhores; Top-1/5 maiores.
# MAGIC Bootstrap reamostra clientes e não linhas individuais. Um cliente conta
# MAGIC uma vez após calcular sua média nos mesmos casos do marco/segmento.
# MAGIC Os ICs descrevem este holdout de desenvolvimento, não garantem produção.

# COMMAND ----------


def adicionar_segmento(df):
    return df.withColumn("segmento", F.explode(F.when(
        F.col("origem_tecnica"), F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS"))
    ).otherwise(F.array(F.lit("TODAS")))))


exatas_v23 = validacao_v23.filter(F.col("tipo_censura") == "exata")
seg = adicionar_segmento(exatas_v23)
metricas = seg.groupBy("segmento", "variante", "idade_seg").agg(
    F.count("*").alias("n_eventos"),
    F.sum(F.col("tem_previsao").cast("long")).alias("n_previstos"),
    F.sum(F.coalesce(F.col("suporte_destino").cast("long"), F.lit(0))).alias("n_suportados"),
    F.sum(F.coalesce(F.col("nll_clip_aplicado").cast("long"), F.lit(0))).alias("n_logloss_clip"),
    F.avg("acerto_top1").alias("top1_por_evento_previsto"),
    F.avg("acerto_top5").alias("top5_por_evento_previsto"),
    F.avg("logloss_clip").alias("logloss_por_evento_previsto"),
    F.avg("brier").alias("brier_por_evento_previsto"),
    F.avg("variacao_temporal_tv").alias("variacao_temporal_media"),
    F.avg(F.col("logloss_sem_tempo_clip") - F.col("logloss_clip")).alias("ganho_logloss_tempo"),
).withColumn("cobertura_modelo", F.col("n_previstos") / F.col("n_eventos"))
metricas = guardar(metricas, SM23_FIT_CFG["tabela_metricas"], "METRICAS")
mostrar("V23_24_METRICAS_DESTINO", metricas.select(
    "segmento", "variante", "idade_seg", "n_eventos", "n_previstos", "n_suportados",
    "cobertura_modelo", "top1_por_evento_previsto", "top5_por_evento_previsto",
    "logloss_por_evento_previsto", "brier_por_evento_previsto", "n_logloss_clip",
    "ganho_logloss_tempo",
).orderBy("segmento", "idade_seg", "variante"))

w_caso = Window.partitionBy("cd_bv", "passo", "idade_seg")
pares = exatas_v23.withColumn("_n_prev", F.sum(
    F.col("tem_previsao").cast("long")
).over(w_caso)).filter(F.col("_n_prev") == 4)
pares_seg = adicionar_segmento(pares)
por_cliente = pares_seg.groupBy("segmento", "idade_seg", "cd_bv", "variante").agg(
    F.count("*").alias("n_casos"),
    F.avg("logloss_clip").alias("logloss"), F.avg("brier").alias("brier"),
    F.avg("acerto_top1").alias("top1"), F.avg("acerto_top5").alias("top5"),
)
if por_cliente.count() > SM23_FIT_CFG["max_agregados_bootstrap"]:
    raise RuntimeError("Muitos agregados para bootstrap no driver. Rever dimensão.")
# Somente médias por cliente (~506 clientes), nunca eventos individuais.
pdf_clientes = por_cliente.toPandas()
if pdf_clientes.empty:
    raise RuntimeError("Nenhum caso comum modelado nas quatro variantes.")
rng = np.random.default_rng(SM23_FIT_CFG["semente"])
linhas_pares = []
for (segmento, idade), grupo in pdf_clientes.groupby(["segmento", "idade_seg"], sort=True):
    ids = sorted(grupo["cd_bv"].unique())
    indices = rng.integers(0, len(ids), size=(SM23_FIT_CFG["bootstrap_replicas"], len(ids)))
    for metrica in ("logloss", "brier", "top1", "top5"):
        tabela = grupo.pivot(index="cd_bv", columns="variante", values=metrica).reindex(ids)
        if set(tabela.columns) != set(VARIANTES) or tabela.isna().any().any():
            raise RuntimeError("Pares incompletos por cliente no bootstrap.")
        a = tabela["A_REFERENCIA"].to_numpy(float)
        for variante in VARIANTES[1:]:
            b = tabela[variante].to_numpy(float)
            ganho = a - b if metrica in ("logloss", "brier") else b - a
            boot = ganho[indices].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            ncasos = int(grupo.loc[grupo.variante.eq("A_REFERENCIA"), "n_casos"].sum())
            linhas_pares.append(dict(
                segmento=str(segmento), idade_seg=float(idade), variante=variante,
                metrica=metrica, n_clientes=len(ids), n_casos=ncasos,
                media_a=float(a.mean()), media_variante=float(b.mean()),
                ganho_medio=float(ganho.mean()), ganho_ic95_lo=float(lo), ganho_ic95_hi=float(hi),
            ))
SCHEMA_PARES = (
    "segmento string, idade_seg double, variante string, metrica string, "
    "n_clientes long, n_casos long, media_a double, media_variante double, "
    "ganho_medio double, ganho_ic95_lo double, ganho_ic95_hi double"
)
comparacao = guardar(spark.createDataFrame(linhas_pares, SCHEMA_PARES),
                    SM23_FIT_CFG["tabela_pares"], "COMPARACAO_PAREADA")
mostrar("V23_25_COMPARACAO_PAREADA", comparacao.select(
    "segmento", "idade_seg", "variante", "metrica", "n_clientes", "n_casos",
    "media_a", "media_variante", "ganho_medio", "ganho_ic95_lo", "ganho_ic95_hi",
).filter("segmento = 'ORIGENS_TECNICAS'").orderBy("idade_seg", "metrica", "variante"))
print("Ganho positivo = melhora sobre A, tanto para perdas quanto para acertos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Conclusão da etapa e manifesto
# MAGIC Cada tabela é escrita separadamente: não é uma transação conjunta.
# MAGIC Somente CONCLUIDO_TREINO_VALIDACAO autoriza consumir este ID de ajuste.
# MAGIC Falhas/modelos nos limites ficam registrados. Não há promoção automática.
# MAGIC O resultado NÃO é a nova tabela de negócio nem mede Top-5 macro acionável.

# COMMAND ----------

config_execucao = {
    "versao_ajuste": VERSAO_AJUSTE, "numerico": SM23_NUM,
    "rotas": "B->contexto ou A; D->contexto ou C; sempre temporal",
    "sem_nova_amostragem": True, "idades_seg": SM23_FIT_CFG["idades_seg"],
    "validacao": "proximo_estado_raw_condicionado_a_duracao_maior_que_marco",
    "pares": "casos_modelados_nas_quatro_variantes; media_por_cliente",
    "logloss_piso": 1e-15, "bootstrap_replicas": SM23_FIT_CFG["bootstrap_replicas"],
    "semente": SM23_FIT_CFG["semente"], "fonte_preparo": PREPARO,
    "versao_delta_preparo": V_MANIFESTO, "versao_delta_base_v23": V_BASE,
    "versao_delta_suporte_v23": V_SUPORTE,
    "acoes_tecnicas": POLITICA["acoes_tecnicas"],
    "tabelas_saida": {k: v for k, v in SM23_FIT_CFG.items() if k.startswith("tabela_")},
}
meta_final = dict(
    id_experimento=ID_EXPERIMENTO, id_ajuste=ID_AJUSTE,
    status="CONCLUIDO_TREINO_VALIDACAO", versao_ajuste=VERSAO_AJUSTE,
    hash_ajuste=HASH_AJUSTE, politica_sha256=PREPARO["politica_sha256"],
    config_json=json.dumps(config_execucao, ensure_ascii=False, sort_keys=True),
    registrado_em_utc=datetime.now(timezone.utc).isoformat(), publicavel=False,
)
SCHEMA_MANIFESTO = (
    "id_experimento string, id_ajuste string, status string, versao_ajuste string, "
    "hash_ajuste string, politica_sha256 string, config_json string, "
    "registrado_em_utc string, publicavel boolean"
)
if SM23_FIT_CFG["gravar"]:
    # Todas as escritas anteriores já foram materializadas pela releitura.
    if spark.catalog.tableExists(SM23_FIT_CFG["tabela_ajustes"]):
        if ha_linhas(spark.table(SM23_FIT_CFG["tabela_ajustes"]).filter(
            F.col("id_ajuste") == ID_AJUSTE
        )):
            raise RuntimeError("Manifesto já concluído para este ID de ajuste.")
    spark.createDataFrame([meta_final], SCHEMA_MANIFESTO).write.format("delta").mode(
        "append"
    ).saveAsTable(SM23_FIT_CFG["tabela_ajustes"])
    print("V23_26_CONCLUSAO: treino e validação persistidos.")
else:
    print("V23_26_CONCLUSAO: resultados apenas em memória; sem manifesto persistido.")
print("id_experimento:", ID_EXPERIMENTO)
print("id_ajuste:", ID_AJUSTE)
print("Não houve alteração da V2.2 nem da tabela de negócio.")
print("Próxima etapa: analisar comparação; depois propagar com memória por trajetória.")
treino_v23.unpersist()
