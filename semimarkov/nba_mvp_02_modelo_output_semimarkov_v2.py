# Databricks notebook source
# NBA | Etapa 2 V2: semi-Markov com duracao DEPENDENTE DO DESTINO.
# Execute apos 01 V2, no mesmo notebook. Nao utiliza modelos/outputs V1.
#
# prob_proxima_acao(j | i, idade=a) = p_ij*S_ij(a) / sum_k p_ik*S_ik(a).
# Estima p e duracoes conjuntamente. Censurados sem destino contribuem pela
# sobrevivencia da mistura, NAO sao descartados nem replicados como rotulos.
# Pares raros compartilham uma distribuicao temporal. Isso e explicitado.
# Sem potencial de conversao. Sem transformar hazard instantanea em probabilidade.
# Sem top 5 precomputado por origem: o ranking e calculado DEPOIS de usar a idade.
#
# Compromisso do MVP: lognormais regulares por grupos de destino; calibracao,
# hipoteses de censura e qualidade do relogio ainda exigem validacao no banco.
# A familia do piloto agregado nao prova adequacao a todos os destinos/estados.

import hashlib
import json
from collections.abc import Iterator

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import functions as F

# Aplicacao diaria: 01 com preparar_treino=False e 02 com TREINAR_MODELO=False.
TREINAR_MODELO = True
TABELA_MODELO_EXISTENTE = "ctg_dsti.renato_nba.nba_semimarkov_modelos_v2_hml"
VERSAO_EXISTENTE = None
ALVO_LINHAS_POR_ORIGEM = 20000
MAX_LINHAS_POR_ORIGEM = 50000
TOP_K = 5
IDADES_VALIDACAO_SEG = [0.0, 1800.0, 86400.0, 604800.0]

# COMMAND ----------
# NUCLEO NUMERICO: puro NumPy/SciPy, independente do Spark, testavel isoladamente.
"""Semi-Markov lognormal por origem/destino; unidades internas em dias.

Para origem i, grupos temporais g(j):
  L_exato = p_j f_g(t)
  L_intervalo = p_j [F_g(U) - F_g(L)]
  L_direita = sum_g pi_g S_g(C)
Pares raros compartilham f_g, mas preservam seus destinos e probabilidades.
A funcao objetivo inclui regularizacao fraca, declarada; nao e MLE puro.
"""
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


@dataclass(frozen=True)
class AjusteSM:
    minimo_eventos: int = 80
    minimo_clientes: int = 20
    minimo_eventos_grupo: int = 100
    minimo_clientes_grupo: int = 15
    max_grupos_proprios: int = 8
    regularizacao: float = 2.0
    pseudocontagem_grupo: float = 0.25
    maxiter: int = 500
    sigma_min: float = 0.15
    sigma_max: float = 4.5


def _logsub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log(exp(a)-exp(b)), para a >= b; nao aplica piso de probabilidade."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return a + np.log(-np.expm1(b - a))


def _termos_lognormais(
    lo: np.ndarray, hi: np.ndarray, tipo: np.ndarray,
    mu: np.ndarray, ls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log-contribuicao e derivadas em mu e log(sigma), por observacao."""
    sigma = np.exp(ls)
    log_l = np.zeros(lo.shape, dtype=float)
    dmu = np.zeros_like(log_l)
    dls = np.zeros_like(log_l)
    ex = tipo == 0
    inte = tipo == 1
    ce = tipo == 2
    if np.any(ex):
        lt = np.log(lo[ex])
        z = (lt - mu[ex]) / sigma[ex]
        log_l[ex] = -lt - ls[ex] - .5 * z**2 - .5 * np.log(2 * np.pi)
        dmu[ex] = z / sigma[ex]
        dls[ex] = z**2 - 1
    if np.any(ce):
        with np.errstate(divide="ignore"):
            z = (np.log(lo[ce]) - mu[ce]) / sigma[ce]
        lp = special.log_ndtr(-z)
        log_l[ce] = lp
        with np.errstate(invalid="ignore"):
            mills = np.exp(-.5 * z**2 - .5 * np.log(2 * np.pi) - lp)
            dmu[ce] = mills / sigma[ce]
            dls[ce] = np.where(np.isfinite(z), z * mills, 0.0)
    if np.any(inte):
        with np.errstate(divide="ignore"):
            zl = (np.log(lo[inte]) - mu[inte]) / sigma[inte]
        zu = (np.log(hi[inte]) - mu[inte]) / sigma[inte]
        # Escolhe CDF ou sobrevivencia para nao subtrair numeros proximos de 1.
        use_sf = zl > 0
        a = np.where(use_sf, special.log_ndtr(-zl), special.log_ndtr(zu))
        b = np.where(use_sf, special.log_ndtr(-zu), special.log_ndtr(zl))
        lp = _logsub(a, b)
        log_l[inte] = lp
        ll = -.5 * zl**2 - .5 * np.log(2 * np.pi)
        lu = -.5 * zu**2 - .5 * np.log(2 * np.pi)
        # Diferencas com sinal, em escala log, para as derivadas analiticas.
        # O limite L=0 tem phi(-inf)=0 e (-inf)*phi(-inf)=0.
        pl = np.exp(ll - lp)
        pu = np.exp(lu - lp)
        dmu[inte] = (pl - pu) / sigma[inte]
        with np.errstate(invalid="ignore"):
            dls[inte] = np.where(np.isfinite(zl), zl * pl, 0.) - zu * pu
    return log_l, dmu, dls


def _arrays(pdf: pd.DataFrame) -> dict[str, Any]:
    required = {"dur_min", "dur_max", "tipo_censura", "destino", "cd_bv"}
    if required - set(pdf.columns):
        raise ValueError(f"Campos ausentes: {sorted(required - set(pdf.columns))}")
    df = pdf.copy()
    tipos = df["tipo_censura"].map({"exata": 0, "intervalo": 1, "direita": 2})
    if tipos.isna().any():
        raise ValueError("Tipo de censura desconhecido.")
    t = tipos.to_numpy(int)
    lo = df["dur_min"].to_numpy(float) / 86400.0
    hi = df["dur_max"].to_numpy(float) / 86400.0
    if not np.isfinite(lo).all() or (lo < 0).any():
        raise ValueError("Limite inferior invalido.")
    if np.any((t == 0) & ((lo <= 0) | (hi != lo))):
        raise ValueError("Duracao exata precisa ser positiva e ter L=U.")
    if np.any((t == 1) & ((hi <= lo) | ~np.isfinite(hi))):
        raise ValueError("Intervalo invalido; nao sera imputado.")
    if np.any((t == 2) & ~np.isnan(hi)):
        raise ValueError("Censura direita exige limite superior nulo.")
    known = t != 2
    if df.loc[known, "destino"].isna().any():
        raise ValueError("Evento observado sem destino identificado.")
    if df.loc[~known, "destino"].notna().any():
        raise ValueError("Observacao censurada nao pode ter destino atribuido.")
    w = df["peso"].to_numpy(float) if "peso" in df else np.ones(len(df))
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("Pesos invalidos.")
    return {"df": df, "tipo": t, "lo": lo, "hi": hi, "w": w, "known": known}


def _objetivo_conjunto(
    theta: np.ndarray, dados: dict[str, Any], n_grupos: int,
    mu_ref: float, sigma_ref: float, cfg: AjusteSM,
) -> tuple[float, np.ndarray]:
    """NLL penalizada media e gradiente; direita usa SOMA de sobrevivencias."""
    g = n_grupos
    logits = np.r_[theta[:g - 1], 0.0]
    logp = logits - special.logsumexp(logits)
    p = np.exp(logp)
    mu = theta[g - 1:2 * g - 1]
    ls = theta[2 * g - 1:]
    known, grupo = dados["known"], dados["grupo"]
    w, lo, hi, tipo = dados["w"], dados["lo"], dados["hi"], dados["tipo"]
    grad_mu = np.zeros(g)
    grad_ls = np.zeros(g)
    nk = np.zeros(g)
    ll = 0.0
    if np.any(known):
        ids = grupo[known]
        logf, dm, ds = _termos_lognormais(
            lo[known], hi[known], tipo[known], mu[ids], ls[ids]
        )
        ww = w[known]
        ll += np.sum(ww * (logp[ids] + dados["log_r"][known] + logf))
        nk += np.bincount(ids, weights=ww, minlength=g)
        grad_mu += np.bincount(ids, weights=ww * dm, minlength=g)
        grad_ls += np.bincount(ids, weights=ww * ds, minlength=g)
    # Processa censuras em lotes: evita criar uma matriz clientes x destinos enorme.
    idx = np.flatnonzero(~known)
    for inicio in range(0, len(idx), 2048):
        ix = idx[inicio:inicio + 2048]
        shape = (len(ix), g)
        log_s, dm, ds = _termos_lognormais(
            np.broadcast_to(lo[ix, None], shape),
            np.full(shape, np.nan), np.full(shape, 2),
            np.broadcast_to(mu, shape), np.broadcast_to(ls, shape),
        )
        comp = log_s + logp
        den = special.logsumexp(comp, axis=1)
        resp_w = np.exp(comp - den[:, None]) * w[ix, None]
        ll += np.sum(w[ix] * den)
        nk += resp_w.sum(axis=0)
        grad_mu += np.sum(resp_w * dm, axis=0)
        grad_ls += np.sum(resp_w * ds, axis=0)
    alpha = cfg.pseudocontagem_grupo
    shrink = cfg.regularizacao
    penalty = .5 * shrink * (
        np.sum(((mu - mu_ref) / sigma_ref)**2)
        + np.sum((ls - np.log(sigma_ref))**2)
    ) - alpha * logp.sum()
    grad_logits = nk - w.sum() * p + alpha * (1 - g * p)
    grad = np.r_[
        -grad_logits[:-1],
        -grad_mu + shrink * (mu - mu_ref) / sigma_ref**2,
        -grad_ls + shrink * (ls - np.log(sigma_ref)),
    ] / w.sum()
    loss = (-ll + penalty) / w.sum()
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        return np.inf, np.zeros_like(theta)
    return float(loss), grad


def ajustar_origem(pdf: pd.DataFrame, cfg: AjusteSM = AjusteSM()) -> dict[str, Any]:
    """Estima conjuntamente pi_g, mu_g e sigma_g, com pooling dos pares raros."""
    x = _arrays(pdf)
    df, known, w = x["df"], x["known"], x["w"]
    n_clientes = int(df["cd_bv"].nunique())
    n_eventos = int(w[known].sum())
    if n_eventos < cfg.minimo_eventos or n_clientes < cfg.minimo_clientes:
        raise ValueError("SUPORTE_INSUFICIENTE")
    obs = df.loc[known].assign(_peso=w[known])
    suporte = obs.groupby("destino", sort=True).agg(
        n=("_peso", "sum"), n_clientes=("cd_bv", "nunique")
    ).sort_values(["n"], ascending=False, kind="mergesort")
    proprios = list(suporte.loc[
        (suporte["n"] >= cfg.minimo_eventos_grupo)
        & (suporte["n_clientes"] >= cfg.minimo_clientes_grupo)
    ].head(cfg.max_grupos_proprios).index)
    destinos = sorted(suporte.index.astype(str))
    grupo_destino = {j: k for k, j in enumerate(proprios)}
    raros = [j for j in destinos if j not in grupo_destino]
    if raros:
        grupo_destino.update({j: len(proprios) for j in raros})
    g = max(grupo_destino.values()) + 1
    grupos = np.array([grupo_destino[j] for j in destinos], dtype=int)
    n_j = np.array([float(suporte.loc[j, "n"]) for j in destinos])
    n_g = np.bincount(grupos, weights=n_j, minlength=g)
    r_j = n_j / n_g[grupos]
    r_map = dict(zip(destinos, r_j))
    x["grupo"] = np.array([
        grupo_destino[str(j)] if k else -1
        for j, k in zip(df["destino"], known)
    ])
    x["log_r"] = np.array([
        np.log(r_map[str(j)]) if k else 0.
        for j, k in zip(df["destino"], known)
    ])
    # Intervalos: ponto interno SOMENTE para inicializar, nunca como observacao.
    ref = x["lo"][known].copy()
    inter = x["tipo"][known] == 1
    ref[inter] = .5 * (x["lo"][known][inter] + x["hi"][known][inter])
    lt = np.log(ref)
    ww = w[known]
    mu0 = float(np.average(lt, weights=ww))
    sig0 = float(np.clip(np.sqrt(np.average((lt - mu0)**2, weights=ww)), .35, 2.5))
    mus = np.array([
        np.average(lt[x["grupo"][known] == k], weights=ww[x["grupo"][known] == k])
        for k in range(g)
    ])
    probs0 = (n_g + cfg.pseudocontagem_grupo) / (n_g.sum() + cfg.pseudocontagem_grupo * g)
    logits0 = np.log(probs0[:-1]) - np.log(probs0[-1])
    # Limites amplos, declarados. Ajustes que encostam neles nao sao aprovados.
    mu_lo, mu_hi = -20., 15.
    bounds = ([(-25., 25.)] * (g - 1)
              + [(mu_lo, mu_hi)] * g
              + [(np.log(cfg.sigma_min), np.log(cfg.sigma_max))] * g)
    resultados = []
    for init_mu in (mus, np.full(g, mu0)):
        init = np.r_[logits0, np.clip(init_mu, -19., 14.), np.full(g, np.log(sig0))]
        res = optimize.minimize(
            _objetivo_conjunto, init, args=(x, g, mu0, sig0, cfg),
            method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": cfg.maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40},
        )
        if res.success and np.isfinite(res.fun) and np.isfinite(res.x).all():
            resultados.append(res)
    if not resultados:
        raise ValueError("NAO_CONVERGIU")
    res = min(resultados, key=lambda v: v.fun)
    th = res.x
    lp = np.r_[th[:g - 1], 0.]
    pi = special.softmax(lp)
    mu = th[g - 1:2 * g - 1]
    sigma = np.exp(th[2 * g - 1:])
    hit_bound = any(
        abs(val - low) < 1e-4 or abs(val - high) < 1e-4
        for val, (low, high) in zip(th, bounds)
    )
    if hit_bound:
        raise ValueError("LIMITE_PARAMETRICO_ATINGIDO")
    model = {
        "formato": "semimarkov_lognormal_grupos_v2",
        "familia": "lognormal", "unidade": "dias",
        "destinos": destinos, "grupo": grupos.tolist(),
        "r_destino_no_grupo": r_j.tolist(), "pi_grupo": pi.tolist(),
        "mu_grupo": mu.tolist(), "sigma_grupo": sigma.tolist(),
        "p_destino": (pi[grupos] * r_j).tolist(),
        "n_destino": n_j.astype(int).tolist(),
        "n_clientes_destino": [int(suporte.loc[j, "n_clientes"]) for j in destinos],
        "n_eventos": n_eventos, "n_direita": int(w[~known].sum()),
        "n_clientes": n_clientes, "n_grupos": int(g),
        "regularizacao": cfg.regularizacao,
        "pseudocontagem_grupo": cfg.pseudocontagem_grupo,
        "perda_penalizada_media": float(res.fun),
        "n_iteracoes": int(res.nit),
        "max_tempo_observado_dias": float(x["lo"].max()),
        "tempo_dependente_destino": bool(g > 1),
    }
    json.dumps(model, allow_nan=False)
    return model


def log_s_grupos(modelo: dict[str, Any], idade_dias: np.ndarray) -> np.ndarray:
    a = np.asarray(idade_dias, dtype=float).reshape(-1)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Idade precisa ser finita e nao negativa.")
    with np.errstate(divide="ignore"):
        z = (np.log(a[:, None]) - np.asarray(modelo["mu_grupo"])) / np.asarray(modelo["sigma_grupo"])
    return special.log_ndtr(-z)


def prever_destinos(
    modelo: dict[str, Any], idade_dias: np.ndarray, horizonte_dias: float = 7.,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q_j(a), P(proxima=j,T<=a+h | T>a), P(T>a+h | T>a)."""
    if horizonte_dias <= 0 or not np.isfinite(horizonte_dias):
        raise ValueError("Horizonte invalido.")
    a = np.asarray(idade_dias, float).reshape(-1)
    log_s = log_s_grupos(modelo, a)
    lp = np.log(np.asarray(modelo["pi_grupo"]))
    den = special.logsumexp(lp + log_s, axis=1)
    log_wg = lp + log_s - den[:, None]
    g = np.asarray(modelo["grupo"], int)
    r = np.asarray(modelo["r_destino_no_grupo"], float)
    q = np.exp(log_wg[:, g]) * r
    log_sh = log_s_grupos(modelo, a + horizonte_dias)
    delta = log_sh - log_s
    prob_sair_grupo = -np.expm1(delta)
    qh = q * prob_sair_grupo[:, g]
    sem_saida = np.exp(special.logsumexp(lp + log_sh, axis=1) - den)
    if not np.isfinite(q).all() or not np.isfinite(qh).all():
        raise ValueError("Falha numerica na previsao.")
    if not np.allclose(q.sum(axis=1), 1., atol=1e-9):
        raise ValueError("Probabilidades nao somam 1.")
    if not np.allclose(qh.sum(axis=1) + sem_saida, 1., atol=1e-9):
        raise ValueError("Probabilidades do horizonte nao fecham.")
    return q, qh, sem_saida


def loglik_linhas(modelo: dict[str, Any], pdf: pd.DataFrame) -> np.ndarray:
    """Contribuicao nao penalizada; destino nunca aprendido retorna -inf."""
    x = _arrays(pdf)
    dest_map = {j: k for k, j in enumerate(modelo["destinos"])}
    groups = np.asarray(modelo["grupo"], int)
    p = np.asarray(modelo["p_destino"], float)
    mus = np.asarray(modelo["mu_grupo"], float)
    lss = np.log(np.asarray(modelo["sigma_grupo"], float))
    ll = np.full(len(pdf), -np.inf)
    labels = pdf["destino"].map(dest_map)
    ids = np.flatnonzero(x["known"] & labels.notna().to_numpy())
    if len(ids):
        j = labels.iloc[ids].to_numpy(int)
        g = groups[j]
        lf, _, _ = _termos_lognormais(x["lo"][ids], x["hi"][ids], x["tipo"][ids], mus[g], lss[g])
        ll[ids] = np.log(p[j]) + lf
    idx = np.flatnonzero(~x["known"])
    if len(idx):
        ls = log_s_grupos(modelo, x["lo"][idx])
        ll[idx] = special.logsumexp(np.log(modelo["pi_grupo"]) + ls, axis=1)
    return ll

# COMMAND ----------
# Estimacao distribuida: uma linha de modelo por origem, nunca 600M linhas no driver.
SM_AJUSTE_CFG = AjusteSM()
SM_ESQUEMA_MODELO = (
    "origem string, status_modelo string, detalhe string, modelo_json string, "
    "id_modelo_origem string, n_amostra long, n_grupos long, "
    "versao_modelo string, corte_treino string, relogio string"
)
sm_versao = SM_CFG["versao_modelo"]


def sm_amostrar_origens(df, alvo=ALVO_LINHAS_POR_ORIGEM):
    tamanhos = df.groupBy("estado").count().collect()  # <= numero de estados.
    fracoes = {r["estado"]: min(1., alvo / r["count"]) for r in tamanhos}
    # Mesma fracao para exatas/intervalares/censuradas de cada origem.
    return df.sampleBy("estado", fracoes, seed=20260927)


def sm_ajustar_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
    origem = str(pdf["estado"].iloc[0])
    linha = dict(origem=origem, status_modelo="SEM_AJUSTE", detalhe="",
                 modelo_json=None, id_modelo_origem=None, n_amostra=int(len(pdf)),
                 n_grupos=0, versao_modelo=sm_versao,
                 corte_treino=SM_CFG["corte_treino_exclusivo"], relogio=SM_CFG["relogio"])
    if len(pdf) > MAX_LINHAS_POR_ORIGEM:
        linha.update(status_modelo="LIMITE_MEMORIA", detalhe="Reduzir alvo da amostra; nao truncamos o grupo.")
        return pd.DataFrame([linha])
    try:
        pdf = pdf.sort_values(["cd_bv", "passo"], kind="mergesort")
        model = ajustar_origem(pdf, SM_AJUSTE_CFG)
        model["origem"] = origem
        model["corte_treino"] = SM_CFG["corte_treino_exclusivo"]
        model["relogio"] = SM_CFG["relogio"]
        payload = json.dumps(model, sort_keys=True, allow_nan=False)
        linha.update(status_modelo="AJUSTADO", modelo_json=payload,
                     id_modelo_origem=hashlib.sha256(payload.encode()).hexdigest(),
                     n_grupos=model["n_grupos"])
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        linha.update(status_modelo="FALHA_OU_SUPORTE_INSUFICIENTE", detalhe=str(exc)[:240])
    return pd.DataFrame([linha])


if TREINAR_MODELO:
    if not SM_CFG["preparar_treino"]:
        raise ValueError("Etapa 1 precisa preparar o treino para esta opcao.")
    sm_base = spark.table(SM_VIEWS["treino"])
    sm_dados_ajuste = sm_amostrar_origens(
        sm_base.filter(F.col("elegivel_ajuste") & ~F.col("validacao_cliente"))
        .select("cd_bv", "passo", "estado", "destino", "dur_min", "dur_max", "tipo_censura")
    )
    sm_modelos = (
        sm_dados_ajuste.groupBy("estado").applyInPandas(sm_ajustar_pdf, schema=SM_ESQUEMA_MODELO)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
else:
    if not VERSAO_EXISTENTE:
        raise ValueError("Informe VERSAO_EXISTENTE para nao retreinar.")
    sm_versao = VERSAO_EXISTENTE
    sm_modelos = (
        spark.table(TABELA_MODELO_EXISTENTE)
        .filter(F.col("versao_modelo") == VERSAO_EXISTENTE)
        .select(*[p.strip().split()[0] for p in SM_ESQUEMA_MODELO.split(",")])
        .distinct()
    )
    if sm_ha(sm_modelos.groupBy("origem").count().filter("count > 1")):
        raise ValueError("Mais de um modelo da mesma origem/versao; nao selecionar arbitrariamente.")
    if sm_ha(sm_modelos.filter(
        (F.col("relogio") != SM_CFG["relogio"])
        | (F.col("corte_treino").cast("timestamp") > F.lit(SM_CFG["corte_estado_exclusivo"]).cast("timestamp"))
    )):
        raise ValueError("Modelo com outro relogio ou treinamento posterior ao corte do estado.")

sm_modelos.createOrReplaceTempView("nba_sm_v2_modelos")
sm_modelos.groupBy("status_modelo", "n_grupos").count().show(truncate=False)
sm_coleta_modelos = sm_modelos.collect()  # Uma linha por origem, nao por cliente.
SM_MODELOS = {}
SM_STATUS_ORIGEM = {}
for linha in sm_coleta_modelos:
    SM_STATUS_ORIGEM[linha["origem"]] = linha["status_modelo"]
    if linha["status_modelo"] == "AJUSTADO":
        mod = json.loads(linha["modelo_json"])
        mod["id_modelo_origem"] = linha["id_modelo_origem"]
        SM_MODELOS[linha["origem"]] = mod
if not SM_MODELOS:
    raise ValueError("Nenhuma origem ajustada. Revise suporte/erros antes de gerar previsoes.")
SM_BROADCAST_MODELOS = spark.sparkContext.broadcast(SM_MODELOS)

# Tabela pequena e legivel de parametros, alem do JSON tecnico.
sm_parametros = []
for origem, mod in SM_MODELOS.items():
    for j, destino in enumerate(mod["destinos"]):
        g = mod["grupo"][j]
        sm_parametros.append((
            origem, destino, int(g), float(mod["p_destino"][j]),
            float(mod["mu_grupo"][g]), float(mod["sigma_grupo"][g]),
            int(mod["n_destino"][j]), int(mod["n_clientes_destino"][j]),
            sm_versao, mod["id_modelo_origem"],
        ))
spark.createDataFrame(sm_parametros, (
    "origem string, destino string, grupo_temporal long, p_na_entrada double, "
    "meanlog_dias double, sdlog double, n_eventos_destino long, n_clientes_destino long, "
    "versao_modelo string, id_modelo_origem string"
)).createOrReplaceTempView("nba_sm_v2_parametros")

# COMMAND ----------
# Aplicacao: primeiro calcula as probabilidades condicionadas a idade; so depois rankeia.
SM_ESQUEMA_OUTPUT = (
    "cd_bv string, data_referencia date, ts_corte_estado timestamp, acao_atual string, "
    "tempo_no_estado_seg double, ranking long, proxima_acao string, "
    "prob_proxima_acao double, prob_na_entrada double, prob_proxima_acao_7d double, "
    "prob_sem_saida_7d double, massa_top5 double, status_previsao string, "
    "status_temporal string, status_dados string, versao_modelo string, "
    "id_modelo_origem string, relogio string, publicavel boolean"
)
SM_COLUNAS_OUTPUT = [p.strip().split()[0] for p in SM_ESQUEMA_OUTPUT.split(",")]


def sm_prever_pdf(pdf: pd.DataFrame, modelos: dict, versao: str, horizonte: float) -> pd.DataFrame:
    saidas = []
    for origem, parte in pdf.groupby("acao_atual", dropna=False, sort=False):
        model = modelos.get(origem)
        for start in range(0, len(parte), 512):
            bloco = parte.iloc[start:start + 512].copy()
            apto = ((bloco["status_input"] == "OK")
                    & bloco["tempo_no_estado_seg"].notna()
                    & (bloco["tempo_no_estado_seg"] >= 0))
            indices = np.flatnonzero(apto.to_numpy()) if model else np.array([], dtype=int)
            calculos = {}
            if len(indices):
                age = bloco.iloc[indices]["tempo_no_estado_seg"].to_numpy(float) / 86400.
                q, qh, ns = prever_destinos(model, age, horizonte)
                # Destinos estao ordenados por nome; mergesort desempata apenas exibicao.
                ordem = np.argsort(-q, axis=1, kind="mergesort")[:, :TOP_K]
                for k, ix in enumerate(indices):
                    calculos[ix] = (q[k], qh[k], ns[k], ordem[k], age[k])
            for ix, (_, row) in enumerate(bloco.iterrows()):
                base = dict(
                    cd_bv=str(row["cd_bv"]), data_referencia=row["data_referencia"],
                    ts_corte_estado=row["ts_corte_estado"],
                    acao_atual=None if pd.isna(row["acao_atual"]) else str(row["acao_atual"]),
                    tempo_no_estado_seg=None if pd.isna(row["tempo_no_estado_seg"]) else float(row["tempo_no_estado_seg"]),
                    ranking=None, proxima_acao=None, prob_proxima_acao=None,
                    prob_na_entrada=None, prob_proxima_acao_7d=None, prob_sem_saida_7d=None,
                    massa_top5=None, status_previsao="SEM_MODELO_ORIGEM",
                    status_temporal="NAO_CALCULADO", status_dados=str(row["status_dados"]),
                    versao_modelo=versao, id_modelo_origem=None,
                    relogio=str(row["relogio"]), publicavel=False,
                )
                if ix not in calculos:
                    if row["status_input"] != "OK":
                        base["status_previsao"] = str(row["status_input"])
                    saidas.append(base)
                    continue
                q, qh, ns, ordem, age = calculos[ix]
                status_t = ("EXTRAPOLACAO_TEMPORAL" if age > model["max_tempo_observado_dias"]
                            else "GRUPOS_TEMPORAIS_POR_DESTINO" if model["n_grupos"] > 1
                            else "POOLING_UM_GRUPO_TEMPORAL")
                for pos, j in enumerate(ordem, 1):
                    saidas.append(dict(
                        base, ranking=pos, proxima_acao=model["destinos"][j],
                        prob_proxima_acao=float(q[j]), prob_na_entrada=float(model["p_destino"][j]),
                        prob_proxima_acao_7d=float(qh[j]), prob_sem_saida_7d=float(ns),
                        massa_top5=float(q[ordem].sum()), status_previsao="PREVISAO_SEMIMARKOV",
                        status_temporal=status_t, id_modelo_origem=model["id_modelo_origem"],
                    ))
    result = pd.DataFrame(saidas, columns=SM_COLUNAS_OUTPUT)
    if len(result):
        result["ranking"] = pd.array(result["ranking"], dtype="Int64")
    return result


def sm_prever_lotes(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    modelos = SM_BROADCAST_MODELOS.value
    for pdf in iterator:
        yield sm_prever_pdf(pdf, modelos, sm_versao, SM_CFG["horizonte_dias"])


sm_atuais_entrada = spark.table(SM_VIEWS["atuais"]).select(
    "cd_bv", "data_referencia", "ts_corte_estado", "acao_atual", "tempo_no_estado_seg",
    "status_input", "status_dados", "relogio",
)
sm_previsoes = sm_atuais_entrada.mapInPandas(sm_prever_lotes, schema=SM_ESQUEMA_OUTPUT)
sm_previsoes = sm_previsoes.persist(StorageLevel.MEMORY_AND_DISK)
sm_previsoes.createOrReplaceTempView("nba_sm_v2_previsoes_top5")
sm_previsoes.groupBy("status_previsao", "status_temporal").agg(
    F.countDistinct("cd_bv").alias("n_clientes"), F.count("*").alias("n_linhas")
).show(truncate=False)
print("prob_proxima_acao usa tempo_no_estado_seg. Nao e o ranking de frequencias V1.")
print("prob_7d e a probabilidade da PROXIMA transicao ate 7 dias; nao do estado exato em D+7.")

# COMMAND ----------
# Validacao fora do treino: marcos temporais FIXOS, nao metade da duracao futura.
# NLL e ranking sao condicionados a estar no estado no marco. Casos cujo
# intervalo cruza o marco nao sao adjudicados. Destinos nao aprendidos sao contados.
SM_ESQUEMA_VALIDACAO = (
    "origem string, idade_marco_seg double, n_elegiveis long, n_eventos long, "
    "n_eventos_suportados long, n_top1_temporal long, n_top5_temporal long, "
    "n_top1_semtempo long, n_ll_finitos long, soma_nll double, "
    "status_validacao string"
)


def sm_validar_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
    origem = str(pdf["estado"].iloc[0])
    model = SM_BROADCAST_MODELOS.value.get(origem)
    out = []
    for marco in IDADES_VALIDACAO_SEG:
        case = pdf[(pdf["dur_min"] >= marco) & (
            (pdf["tipo_censura"] != "exata") | (pdf["dur_min"] > marco)
        )].copy()
        row = dict(origem=origem, idade_marco_seg=marco, n_elegiveis=int(len(case)),
                   n_eventos=int((case["tipo_censura"] != "direita").sum()),
                   n_eventos_suportados=0, n_top1_temporal=0, n_top5_temporal=0,
                   n_top1_semtempo=0, n_ll_finitos=0, soma_nll=0., status_validacao="SEM_MODELO")
        if model and len(case) and len(pdf) <= MAX_LINHAS_POR_ORIGEM:
            q, _, _ = prever_destinos(model, [marco / 86400.], SM_CFG["horizonte_dias"])
            order = np.argsort(-q[0], kind="mergesort")[:TOP_K]
            top = [model["destinos"][j] for j in order]
            static = model["destinos"][int(np.argmax(model["p_destino"]))]
            conhecidos = case[case["tipo_censura"] != "direita"]["destino"]
            ll = loglik_linhas(model, case)
            denom = special.logsumexp(np.log(model["pi_grupo"]) + log_s_grupos(model, [marco / 86400.])[0])
            finite = np.isfinite(ll)
            row.update(
                n_eventos_suportados=int(conhecidos.isin(model["destinos"]).sum()),
                n_top1_temporal=int((conhecidos == top[0]).sum()),
                n_top5_temporal=int(conhecidos.isin(top).sum()),
                n_top1_semtempo=int((conhecidos == static).sum()),
                n_ll_finitos=int(finite.sum()), soma_nll=float(-(ll[finite] - denom).sum()),
                status_validacao="AVALIADO_SUPORTE_EXPLICITO",
            )
        elif len(pdf) > MAX_LINHAS_POR_ORIGEM:
            row["status_validacao"] = "LIMITE_MEMORIA_VALIDACAO"
        out.append(row)
    return pd.DataFrame(out)


if TREINAR_MODELO:
    sm_val_dados = sm_amostrar_origens(
        sm_base.filter(F.col("elegivel_ajuste") & F.col("validacao_cliente"))
        .select("cd_bv", "estado", "destino", "dur_min", "dur_max", "tipo_censura")
    )
    sm_validacao = sm_val_dados.groupBy("estado").applyInPandas(sm_validar_pdf, schema=SM_ESQUEMA_VALIDACAO)
else:
    sm_validacao = spark.createDataFrame([], schema=SM_ESQUEMA_VALIDACAO)
sm_validacao = sm_validacao.persist(StorageLevel.MEMORY_AND_DISK)
sm_validacao.createOrReplaceTempView("nba_sm_v2_validacao")
sm_validacao.groupBy("idade_marco_seg").agg(
    F.sum("n_elegiveis").alias("n_elegiveis"),
    F.sum("n_eventos").alias("n_eventos"),
    F.sum("n_eventos_suportados").alias("n_eventos_suportados"),
    (F.sum("n_top1_temporal") / F.sum("n_eventos")).alias("top1_temporal"),
    (F.sum("n_top1_semtempo") / F.sum("n_eventos")).alias("top1_semtempo"),
    (F.sum("n_top5_temporal") / F.sum("n_eventos")).alias("top5_temporal"),
    (F.sum("soma_nll") / F.sum("n_ll_finitos")).alias("nll_media_suporte_finito"),
).orderBy("idade_marco_seg").show(truncate=False)
print("Validacao por clientes e marcos fixos; nao e teste futuro nem certificacao de calibracao.")
