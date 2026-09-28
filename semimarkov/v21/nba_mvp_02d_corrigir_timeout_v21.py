# Databricks notebook source
# NBA | 02D V2.1: correcao dirigida ao timeout, NAO outro diagnostico.
#
# EXECUTAR NO MESMO NOTEBOOK, DEPOIS DE 02C. Mantem a base 01 V2.
# Reajusta SOMENTE as origens rejeitadas por LIMITE_PARAMETRICO_ATINGIDO.
# Conserva, sem reestimar, os modelos AJUSTADO anteriores.
# Nao reduz minimos, nao aumenta sigma_max, nao altera duracoes, nao cria
# ruido/jitter, nao grava Delta e nao autoriza previsoes para producao.
#
# A correcao acrescenta:
#  - silencio separado do conjunto de destinos raros;
#  - massa pontual no timeout para span=0;
#  - cauda LN no EXCESSO T-timeout quando span>0 (nao LN no tempo total).
# Valores de referencia, estado e fontes continuam exatamente os da parte 01.
#
# Outputs NOVOS: nba_sm_v21_modelos, nba_sm_v21_previsoes_top5,
#                nba_sm_v21_validacao, nba_sm_v21_parametros.
# A parte 03 V2 antiga nao sabe auditar massas pontuais: NAO a execute.
# Use nba_mvp_03_auditar_timeout_v21.py apos revisar este output.
#
# API numerica: SciPy >=1.13; dependencia Spark: runtime do notebook.
# Validacao compara ranking ANTES/DEPOIS nos MESMOS registros materializados.
# NLL antiga (densidade continua) NAO e comparada com massa pontual.

# COMMAND ----------


"""Nucleo semi-Markov: lognormal e massa pontual no timeout operacional.

Convencao de aplicacao: o corte e exclusivo. Condicionamos em T >= idade,
e o horizonte cobre [idade, idade + h). Para distribuicoes continuas isso
coincide, em probabilidade, com as convencoes usuais de fronteira.
O suporte pontual so e admitido na transicao para o silencio, no timeout.
"""
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


@dataclass(frozen=True)
class AjusteTimeout:
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


SMD_TOL_SEG = 1e-9  # Apenas tolerancia de ponto flutuante; nao arredonda eventos.
SMD_FORMATO = "semimarkov_timeout_misto_v21"


def smd_arrays(pdf: pd.DataFrame) -> dict[str, Any]:
    campos = {"cd_bv", "destino", "tipo_censura", "dur_min", "dur_max"}
    if campos - set(pdf.columns) or len(pdf) == 0:
        raise ValueError("CAMPOS_AUSENTES_OU_AMOSTRA_VAZIA")
    df = pdf.reset_index(drop=True).copy()
    tipos = df["tipo_censura"].map({"exata": 0, "intervalo": 1, "direita": 2})
    if tipos.isna().any():
        raise ValueError("TIPO_CENSURA_INVALIDO")
    tipo = tipos.to_numpy(int)
    lo_seg = df["dur_min"].to_numpy(float)
    hi_seg = df["dur_max"].to_numpy(float)
    if not np.isfinite(lo_seg).all() or np.any(lo_seg < 0):
        raise ValueError("LIMITE_INFERIOR_INVALIDO")
    if np.any((tipo == 0) & ((lo_seg <= 0) | (hi_seg != lo_seg))):
        raise ValueError("EXATA_INVALIDA")
    if np.any((tipo == 1) & ((hi_seg <= lo_seg) | ~np.isfinite(hi_seg))):
        raise ValueError("INTERVALO_INVALIDO")
    if np.any((tipo == 2) & ~np.isnan(hi_seg)):
        raise ValueError("DIREITA_EXIGE_SUPERIOR_NULO")
    known = tipo != 2
    if df.loc[known, "destino"].isna().any():
        raise ValueError("EVENTO_SEM_DESTINO")
    if df.loc[~known, "destino"].notna().any():
        raise ValueError("CENSURA_COM_DESTINO")
    if df["cd_bv"].isna().any():
        raise ValueError("CLIENTE_NULO")
    w = df["peso"].to_numpy(float) if "peso" in df else np.ones(len(df))
    if not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("PESOS_INVALIDOS")
    return dict(
        df=df, tipo=tipo, known=known, w=w, lo_seg=lo_seg, hi_seg=hi_seg,
        lo=lo_seg / 86400., hi=hi_seg / 86400.,
    )


def smd_logsub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return a + np.log(-np.expm1(b - a))


def smd_lognormal_termos(lo, hi, tipo, mu, ls):
    """Termos de uma LN positiva: derivadas em mu e log(sigma)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    tipo = np.asarray(tipo, int)
    sig = np.exp(ls)
    ll = np.full(lo.shape, -np.inf)
    dm = np.zeros_like(lo)
    ds = np.zeros_like(lo)
    ex = (tipo == 0) & (lo > 0)
    if ex.any():
        lt = np.log(lo[ex])
        z = (lt - mu) / sig
        ll[ex] = -lt - ls - 0.5 * z**2 - 0.5 * np.log(2 * np.pi)
        dm[ex] = z / sig
        ds[ex] = z**2 - 1.
    ce0 = (tipo == 2) & (lo <= 0)
    ll[ce0] = 0.0
    ce = (tipo == 2) & (lo > 0)
    if ce.any():
        z = (np.log(lo[ce]) - mu) / sig
        val = special.log_ndtr(-z)
        mills = np.exp(-0.5 * z**2 - 0.5 * np.log(2 * np.pi) - val)
        ll[ce], dm[ce], ds[ce] = val, mills / sig, mills * z
    it = (tipo == 1) & (hi > 0) & (hi > np.maximum(lo, 0.))
    if it.any():
        l = np.maximum(lo[it], 0.)
        with np.errstate(divide="ignore"):
            zl = (np.log(l) - mu) / sig
        zu = (np.log(hi[it]) - mu) / sig
        sf = zl > 0
        a = np.where(sf, special.log_ndtr(-zl), special.log_ndtr(zu))
        b = np.where(sf, special.log_ndtr(-zu), special.log_ndtr(zl))
        val = smd_logsub(a, b)
        pl = np.exp(-0.5 * zl**2 - 0.5 * np.log(2 * np.pi) - val)
        pu = np.exp(-0.5 * zu**2 - 0.5 * np.log(2 * np.pi) - val)
        with np.errstate(invalid="ignore"):
            dz = np.where(np.isfinite(zl), zl * pl, 0.) - zu * pu
        ll[it], dm[it], ds[it] = val, (pl - pu) / sig, dz
    return ll, dm, ds


def smd_termos_grupo(lo_seg, hi_seg, tipo, familia, atraso_seg, mu, ls, eta):
    """Contribuicoes conjunta/pontual/continua, sem pisos de probabilidade.

    'exata' no atomo usa massa; exata fora do atomo usa densidade continua.
    Intervalos usam (L,U]. Censura de corte exclusivo usa P(T>=C).
    """
    lo_seg, hi_seg = np.asarray(lo_seg, float), np.asarray(hi_seg, float)
    tipo = np.asarray(tipo, int)
    shape = lo_seg.shape
    at_atom = np.abs(lo_seg - atraso_seg) <= SMD_TOL_SEG
    atom_ok = (
        ((tipo == 0) & at_atom)
        | ((tipo == 1) & (lo_seg < atraso_seg) & (hi_seg >= atraso_seg))
        | ((tipo == 2) & (lo_seg <= atraso_seg))
    )
    la = np.where(atom_ok, 0., -np.inf)
    zeros = np.zeros(shape)
    if familia == "atomo_timeout":
        return la, zeros, zeros, zeros

    lo = (lo_seg - atraso_seg) / 86400.
    hi = (hi_seg - atraso_seg) / 86400.
    lc, dm, ds = smd_lognormal_termos(lo, hi, tipo, mu, ls)
    if familia == "lognormal":
        return lc, dm, ds, zeros
    if familia != "atomo_mais_lognormal":
        raise ValueError("FAMILIA_DESCONHECIDA")

    # eta = logit da massa pontual condicional ao grupo do silencio.
    lmass = -np.logaddexp(0., -eta)
    ltail = -np.logaddexp(0., eta)
    mass = special.expit(eta)
    both = np.stack([lmass + la, ltail + lc], axis=1)
    val = special.logsumexp(both, axis=1)
    finite = np.isfinite(val)
    ra, rc = zeros.copy(), zeros.copy()
    ra[finite] = np.exp(both[finite, 0] - val[finite])
    rc[finite] = np.exp(both[finite, 1] - val[finite])
    return val, rc * dm, rc * ds, np.where(finite, ra - mass, 0.)


def smd_preparar_ajuste(pdf, cfg, timeout_seg=1800., estado_silencio="sem_acao:::classe"):
    if not np.isfinite(timeout_seg) or timeout_seg <= 0:
        raise ValueError("TIMEOUT_INVALIDO")
    x = smd_arrays(pdf)
    df, known, w = x["df"], x["known"], x["w"]
    if int(w[known].sum()) < cfg.minimo_eventos or df["cd_bv"].nunique() < cfg.minimo_clientes:
        raise ValueError("SUPORTE_INSUFICIENTE")
    # A base V2 operacional possui apenas exatas/direita. Nao inferir massa
    # pontual a partir de intervalos ou de observacoes imputadas.
    if np.any(x["tipo"] == 1):
        raise ValueError("PATCH_REQUER_EXATAS_DIREITA_DA_BASE_OPERACIONAL_V2")
    obs = df.loc[known].assign(_peso=w[known])
    suporte = (
        obs.groupby("destino", sort=True)
        .agg(n=("_peso", "sum"), n_clientes=("cd_bv", "nunique"))
        .sort_values("n", ascending=False, kind="mergesort")
    )
    destinos = sorted(suporte.index.astype(str))
    # Silencio nunca e misturado com destinos comportamentais raros.
    nao_sil = suporte.drop(index=estado_silencio, errors="ignore")
    proprios = list(nao_sil.loc[
        (nao_sil["n"] >= cfg.minimo_eventos_grupo)
        & (nao_sil["n_clientes"] >= cfg.minimo_clientes_grupo)
    ].head(cfg.max_grupos_proprios).index)
    grupo_destino = {j: k for k, j in enumerate(proprios)}
    raros = [j for j in destinos if j != estado_silencio and j not in grupo_destino]
    if raros:
        idx = len(proprios)
        grupo_destino.update({j: idx for j in raros})
    if estado_silencio in destinos:
        grupo_destino[estado_silencio] = len(set(grupo_destino.values()))
    g = max(grupo_destino.values()) + 1
    grupo = np.array([grupo_destino[j] for j in destinos], int)
    n_j = np.array([float(suporte.loc[j, "n"]) for j in destinos])
    n_g = np.bincount(grupo, weights=n_j, minlength=g)
    r_j = n_j / n_g[grupo]
    r_map = dict(zip(destinos, r_j))
    x["grupo"] = np.array([
        grupo_destino[str(j)] if k else -1 for j, k in zip(df["destino"], known)
    ])
    x["log_r"] = np.array([
        np.log(r_map[str(j)]) if k else 0. for j, k in zip(df["destino"], known)
    ])
    familias = ["lognormal"] * g
    atrasos = np.zeros(g)
    atom_count, tail_count = np.zeros(g), np.zeros(g)
    if estado_silencio in destinos:
        k = grupo_destino[estado_silencio]
        mask = known & (x["grupo"] == k)
        t = x["lo_seg"][mask]
        if np.any(t < timeout_seg - SMD_TOL_SEG):
            raise ValueError("SAIDA_PARA_SILENCIO_ANTES_TIMEOUT")
        is_atom = np.abs(t - timeout_seg) <= SMD_TOL_SEG
        atom_count[k] = w[mask][is_atom].sum()
        tail_count[k] = w[mask][~is_atom].sum()
        atrasos[k] = timeout_seg
        if tail_count[k] == 0:
            familias[k] = "atomo_timeout"
        elif atom_count[k] > 0:
            familias[k] = "atomo_mais_lognormal"
        # Caso sem massa: LN deslocada em timeout, e nao LN no tempo total.
    cont = np.array([k for k in range(g) if familias[k] != "atomo_timeout"], int)
    mix = np.array([k for k in range(g) if familias[k] == "atomo_mais_lognormal"], int)

    ref_values, ref_weights = [], []
    medias = {}
    for k in cont:
        sel = known & (x["grupo"] == k)
        r = (x["lo_seg"][sel] - atrasos[k]) / 86400.
        positive = r > SMD_TOL_SEG / 86400.
        if not positive.any():
            raise ValueError("CAUDA_CONTINUA_SEM_OBSERVACOES")
        lt, ww = np.log(r[positive]), w[sel][positive]
        medias[k] = float(np.average(lt, weights=ww))
        ref_values.extend(lt.tolist())
        ref_weights.extend(ww.tolist())
    if len(cont):
        vals, ww = np.asarray(ref_values), np.asarray(ref_weights)
        mu_ref = float(np.average(vals, weights=ww))
        sig_ref = float(np.clip(
            np.sqrt(np.average((vals - mu_ref)**2, weights=ww)), 0.35, 2.5
        ))
    else:
        mu_ref, sig_ref = 0., 1.
    mus = np.array([medias[k] for k in cont])
    probs0 = (n_g + cfg.pseudocontagem_grupo) / (
        n_g.sum() + cfg.pseudocontagem_grupo * g
    )
    logits = np.log(probs0[:-1]) - np.log(probs0[-1])
    etas = np.array([
        np.log((atom_count[k] + 0.25) / (tail_count[k] + 0.25)) for k in mix
    ])
    initial = np.r_[logits, np.clip(mus, -19., 14.),
                    np.full(len(cont), np.log(sig_ref)), etas]
    bounds = (
        [(-25., 25.)] * (g - 1) + [(-20., 15.)] * len(cont)
        + [(np.log(cfg.sigma_min), np.log(cfg.sigma_max))] * len(cont)
        + [(-25., 25.)] * len(mix)
    )
    return dict(
        x=x, suporte=suporte, destinos=destinos, grupo=grupo, n_j=n_j, r_j=r_j,
        g=g, familias=familias, atrasos=atrasos, cont=cont, mix=mix,
        atom_count=atom_count, tail_count=tail_count, mu_ref=mu_ref, sig_ref=sig_ref,
        initial=initial, bounds=bounds, timeout_seg=float(timeout_seg),
    )


def smd_unpack(theta, prep):
    g, cont, mix = prep["g"], prep["cont"], prep["mix"]
    c, m = len(cont), len(mix)
    lp = np.r_[theta[:g - 1], 0.]
    lp = lp - special.logsumexp(lp)
    mu, ls, eta = np.zeros(g), np.zeros(g), np.zeros(g)
    mu[cont] = theta[g - 1:g - 1 + c]
    ls[cont] = theta[g - 1 + c:g - 1 + 2 * c]
    if m:
        eta[mix] = theta[g - 1 + 2 * c:]
    return lp, mu, ls, eta


def smd_objetivo(theta, prep, cfg):
    x, g = prep["x"], prep["g"]
    lp, mu, ls, eta = smd_unpack(theta, prep)
    pi = np.exp(lp)
    nk, dmu, dls, de = [np.zeros(g) for _ in range(4)]
    ll = 0.
    for start in range(0, len(x["lo"]), 2048):
        ix = np.arange(start, min(start + 2048, len(x["lo"])))
        known = x["known"][ix]
        logf = np.full((len(ix), g), -np.inf)
        dm, ds, da = [np.zeros_like(logf) for _ in range(3)]
        for k in range(g):
            eligible = ~known | (x["grupo"][ix] == k)
            if not eligible.any():
                continue
            ii = ix[eligible]
            terms = smd_termos_grupo(
                x["lo_seg"][ii], x["hi_seg"][ii], x["tipo"][ii],
                prep["familias"][k], prep["atrasos"][k], mu[k], ls[k], eta[k],
            )
            logf[eligible, k], dm[eligible, k], ds[eligible, k], da[eligible, k] = terms
        logf += np.where(known[:, None], x["log_r"][ix, None], 0.)
        joint = logf + lp
        den = special.logsumexp(joint, axis=1)
        if not np.isfinite(den).all():
            return np.inf, np.zeros_like(theta)
        resp = np.exp(joint - den[:, None]) * x["w"][ix, None]
        ll += float(np.sum(x["w"][ix] * den))
        nk += resp.sum(axis=0)
        dmu += (resp * dm).sum(axis=0)
        dls += (resp * ds).sum(axis=0)
        de += (resp * da).sum(axis=0)
    cont, mix = prep["cont"], prep["mix"]
    ref, sr = prep["mu_ref"], prep["sig_ref"]
    alpha, shrink = cfg.pseudocontagem_grupo, cfg.regularizacao
    penalty = (
        0.5 * shrink * (
            np.sum(((mu[cont] - ref) / sr)**2)
            + np.sum((ls[cont] - np.log(sr))**2)
        ) - alpha * lp.sum()
    )
    if len(mix):
        penalty += alpha * np.sum(
            np.logaddexp(0., -eta[mix]) + np.logaddexp(0., eta[mix])
        )
    wsum = x["w"].sum()
    glogits = nk - wsum * pi + alpha * (1. - g * pi)
    grad = np.r_[
        -glogits[:-1],
        -dmu[cont] + shrink * (mu[cont] - ref) / sr**2,
        -dls[cont] + shrink * (ls[cont] - np.log(sr)),
        -de[mix] + alpha * (2 * special.expit(eta[mix]) - 1.),
    ] / wsum
    loss = (-ll + penalty) / wsum
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        return np.inf, np.zeros_like(theta)
    return float(loss), grad


def smd_ajustar_origem(pdf, cfg=AjusteTimeout(), timeout_seg=1800.,
                      estado_silencio="sem_acao:::classe"):
    prep = smd_preparar_ajuste(pdf, cfg, timeout_seg, estado_silencio)
    init = prep["initial"]
    if len(init) == 0:
        loss, _ = smd_objetivo(init, prep, cfg)
        if not np.isfinite(loss):
            raise ValueError("ATOMO_PURO_INCOMPATIVEL_COM_CENSURAS")
        theta, nit = init, 0
    else:
        alt = init.copy()
        a, c = prep["g"] - 1, len(prep["cont"])
        alt[a:a + c] = np.clip(prep["mu_ref"], -19., 14.)
        results = []
        for x0 in (init, alt):
            res = optimize.minimize(
                smd_objetivo, x0, args=(prep, cfg), method="L-BFGS-B",
                jac=True, bounds=prep["bounds"],
                options={"maxiter": cfg.maxiter, "ftol": 1e-10,
                         "gtol": 1e-6, "maxls": 40},
            )
            if res.success and np.isfinite(res.fun) and np.isfinite(res.x).all():
                results.append(res)
        if not results:
            raise ValueError("NAO_CONVERGIU_TIMEOUT_MISTO")
        # Preservar a regra original: nao escolhe uma solucao inferior interior
        # apenas para esconder a melhor tentativa situada no limite.
        res = min(results, key=lambda r: r.fun)
        theta, loss, nit = res.x, float(res.fun), int(res.nit)
        hits = []
        g, cont, mix = prep["g"], prep["cont"], prep["mix"]
        names = (
            [f"logit_grupo_{k}" for k in range(g - 1)]
            + [f"meanlog_grupo_{k}" for k in cont]
            + [f"log_sdlog_grupo_{k}" for k in cont]
            + [f"logit_atomo_grupo_{k}" for k in mix]
        )
        for name, val, (low, high) in zip(names, theta, prep["bounds"]):
            if abs(val - low) < 1e-4 or abs(val - high) < 1e-4:
                lado = "INFERIOR" if abs(val - low) < 1e-4 else "SUPERIOR"
                hits.append(f"{name}:{lado}:{val:.7g}")
        if hits:
            raise ValueError("LIMITE_REMANESCENTE|" + "|".join(hits))
    lp, mu, ls, eta = smd_unpack(theta, prep)
    pi, r, groups = np.exp(lp), prep["r_j"], prep["grupo"]
    mass = [
        1. if f == "atomo_timeout" else float(special.expit(eta[k]))
        if f == "atomo_mais_lognormal" else 0.
        for k, f in enumerate(prep["familias"])
    ]
    x, df = prep["x"], prep["x"]["df"]
    model = {
        "formato": SMD_FORMATO, "familia": "timeout_misto_lognormal",
        "unidade": "dias", "fronteira_corte": "T_MAIOR_OU_IGUAL",
        "horizonte": "[idade, idade+h)",
        "destinos": prep["destinos"], "grupo": groups.tolist(),
        "r_destino_no_grupo": r.tolist(), "pi_grupo": pi.tolist(),
        "tipo_grupo": prep["familias"],
        "atraso_grupo_seg": prep["atrasos"].tolist(),
        "massa_atomo_grupo": mass,
        "mu_grupo": [None if f == "atomo_timeout" else float(mu[k])
                     for k, f in enumerate(prep["familias"])],
        "sigma_grupo": [None if f == "atomo_timeout" else float(np.exp(ls[k]))
                        for k, f in enumerate(prep["familias"])],
        "p_destino": (pi[groups] * r).tolist(),
        "n_destino": prep["n_j"].astype(int).tolist(),
        "n_clientes_destino": [
            int(prep["suporte"].loc[j, "n_clientes"]) for j in prep["destinos"]
        ],
        "n_eventos": int(x["w"][x["known"]].sum()),
        "n_direita": int(x["w"][~x["known"]].sum()),
        "n_clientes": int(df["cd_bv"].nunique()), "n_grupos": int(prep["g"]),
        "n_atomicos_grupo": prep["atom_count"].tolist(),
        "n_cauda_grupo": prep["tail_count"].tolist(),
        "n_exatas_timeout": int(prep["atom_count"].sum()),
        "timeout_seg": float(timeout_seg), "ajuste_cfg": asdict(cfg),
        "regularizacao": cfg.regularizacao,
        "pseudocontagem_grupo": cfg.pseudocontagem_grupo,
        "perda_penalizada_media": float(loss), "n_iteracoes": nit,
        "max_tempo_observado_dias": float(x["lo"].max()),
        "tempo_dependente_destino": bool(prep["g"] > 1),
        "alerta_caudas": "Caudas sem observacoes positivas nao foram inventadas.",
    }
    # Garante que o JSON nao contem NaN/Infinity.
    import json
    json.dumps(model, allow_nan=False)
    return model


def smd_log_s_grupos(modelo, idade_dias):
    """P(T>=a), inclusive em atomos, pois o snapshot usa corte exclusivo."""
    a = np.asarray(idade_dias, float).reshape(-1)
    if not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("IDADE_INVALIDA")
    g = int(modelo["n_grupos"])
    types = modelo.get("tipo_grupo", ["lognormal"] * g)
    shifts = np.asarray(modelo.get("atraso_grupo_seg", [0.] * g), float)
    mass = np.asarray(modelo.get("massa_atomo_grupo", [0.] * g), float)
    result = np.full((len(a), g), -np.inf)
    for k, family in enumerate(types):
        # Compara em dias para evitar que a ida e volta dia->segundo
        # desloque um corte exatamente situado no atomo.
        shift_days = shifts[k] / 86400.
        atom_s = np.where(a <= shift_days, 0., -np.inf)
        if family == "atomo_timeout":
            result[:, k] = atom_s
            continue
        residual = a - shift_days
        lc = np.zeros(len(a))
        positive = residual > 0
        lc[positive] = special.log_ndtr(
            -(np.log(residual[positive]) - modelo["mu_grupo"][k])
            / modelo["sigma_grupo"][k]
        )
        if family == "lognormal":
            result[:, k] = lc
        elif family == "atomo_mais_lognormal":
            result[:, k] = np.logaddexp(
                np.log(mass[k]) + atom_s, np.log1p(-mass[k]) + lc
            )
        else:
            raise ValueError("FAMILIA_DESCONHECIDA_PREVISAO")
    return result


def smd_prever_destinos(modelo, idade_dias, horizonte_dias=7.):
    if not np.isfinite(horizonte_dias) or horizonte_dias <= 0:
        raise ValueError("HORIZONTE_INVALIDO")
    a = np.asarray(idade_dias, float).reshape(-1)
    ls = smd_log_s_grupos(modelo, a)
    lp = np.log(np.asarray(modelo["pi_grupo"], float))
    den = special.logsumexp(lp + ls, axis=1)
    if not np.isfinite(den).all():
        raise ValueError("IDADE_FORA_SUPORTE_DA_DISTRIBUICAO")
    wg = np.exp(lp + ls - den[:, None])
    groups = np.asarray(modelo["grupo"], int)
    r = np.asarray(modelo["r_destino_no_grupo"], float)
    q = wg[:, groups] * r
    lsh = smd_log_s_grupos(modelo, a + horizonte_dias)
    # Se uma massa ja expirou, wg=0 e sua chance residual tambem e zero.
    exits = np.zeros_like(ls)
    valid = np.isfinite(ls)
    delta = lsh[valid] - ls[valid]
    if np.any(delta > 1e-10):
        raise ValueError("SOBREVIVENCIA_NAO_MONOTONA")
    exits[valid] = -np.expm1(np.minimum(delta, 0.))
    qh = (wg * exits)[:, groups] * r
    ns = np.exp(special.logsumexp(lp + lsh, axis=1) - den)
    if not all(np.isfinite(z).all() for z in (q, qh, ns)):
        raise ValueError("PROBABILIDADE_NAO_FINITA")
    if not np.allclose(q.sum(axis=1), 1., atol=1e-9):
        raise ValueError("MASSA_PROXIMO_DESTINO_NAO_FECHA")
    if not np.allclose(qh.sum(axis=1) + ns, 1., atol=1e-9):
        raise ValueError("MASSA_HORIZONTE_NAO_FECHA")
    return q, qh, ns


# COMMAND ----------
# Preparacao e congelamento do recorte ja usado no 02C. Nada e reamostrado.
import hashlib
import json
import uuid
from collections.abc import Iterator

from pyspark import StorageLevel
from pyspark.sql import functions as F

_smd_required = [
    "SM_CFG", "SM_VIEWS", "SM_AJUSTE_CFG", "smc_dados",
    "sm_modelos", "sm_val_dados", "MAX_LINHAS_POR_ORIGEM",
]
_smd_missing = [name for name in _smd_required if name not in globals()]
if _smd_missing:
    raise RuntimeError(f"Execute 01/02/02C antes. Ausentes: {_smd_missing}")
if SM_CFG["relogio"] != "SILENCIO_OPERACIONAL_TIMEOUT_V2":
    raise ValueError("Este patch exige o relogio operacional da base 01 V2.")

SMD_TOP_K = 5
SMD_HORIZONTE = 7.0
SMD_MARCOS_SEG = [0., 1800., 86400., 604800.]
SMD_MAX_ORIGENS = 40
SMD_CONFIG_AJUSTE = AjusteTimeout(**asdict(SM_AJUSTE_CFG))
SMD_ID_EXECUCAO = str(uuid.uuid4())
SMD_CFG = dict(SM_CFG)
SMD_CFG["versao_modelo"] = SM_CFG["versao_modelo"] + "_timeout_v21"
SMD_CFG["correcao_estimador"] = SMD_FORMATO
SMD_CFG["id_execucao"] = SMD_ID_EXECUCAO
SMD_CFG["corte_exclusivo_condicionamento"] = "T>=idade"
SMD_CFG["horizonte"] = "[idade, idade+7dias)"
SMD_VERSAO = SMD_CFG["versao_modelo"]
SMD_VIEWS = {
    "modelos": "nba_sm_v21_modelos",
    "previsoes": "nba_sm_v21_previsoes_top5",
    "validacao": "nba_sm_v21_validacao",
    "parametros": "nba_sm_v21_parametros",
    "config": "nba_sm_v21_configuracao",
}

# Os checkpoints sao locais e efemeros; nao sao persistencia de producao.
# Se houver perda dos checkpoints/executores, este teste precisa ser repetido.
smd_base_modelos = sm_modelos.localCheckpoint(eager=True)
smd_alvos = smd_base_modelos.filter(
    (F.col("status_modelo") != "AJUSTADO")
    & F.col("detalhe").contains("LIMITE_PARAMETRICO_ATINGIDO")
).select("origem").distinct()
smd_n_alvos = smd_alvos.count()
if not 0 < smd_n_alvos <= SMD_MAX_ORIGENS:
    raise ValueError(f"Numero de origens alvo fora do limite: {smd_n_alvos}")
if smd_base_modelos.groupBy("origem").count().filter("count > 1").limit(1).count():
    raise ValueError("Mais de um resultado por origem.")

smd_dados = (
    smc_dados.join(
        F.broadcast(smd_alvos.select(F.col("origem").alias("estado"))),
        "estado", "left_semi",
    )
    .select("cd_bv", "passo", "estado", "destino", "dur_min", "dur_max", "tipo_censura")
    .localCheckpoint(eager=True)
)
if not smd_dados.limit(1).count():
    raise ValueError("Recorte de reajuste vazio.")
# Reutiliza o dataset de validacao da parte 02 sem novos sorteios.
smd_val_dados = sm_val_dados.localCheckpoint(eager=True)
smd_contagens = (
    smd_alvos.join(smd_base_modelos.select("origem", "n_amostra"), "origem")
    .join(
        smd_dados.groupBy("estado").count().select(
            F.col("estado").alias("origem"), F.col("count").alias("n_reajuste")
        ), "origem", "left",
    )
)
if smd_contagens.filter(F.col("n_reajuste").isNull()).limit(1).count():
    raise ValueError("Uma origem alvo nao existe no recorte do 02C.")
print("Origem dos dados: amostra materializada do 02C; sem nova amostragem.")
print("Origens a reajustar:", smd_n_alvos)
print("Linhas no recorte:", smd_dados.count())
print("Limites e minimos mantidos:", asdict(SMD_CONFIG_AJUSTE))
smd_contagens.orderBy("origem").show(40, truncate=False)
print("Novo identificador do conjunto:", SMD_VERSAO)

# COMMAND ----------
# Reajuste: uma origem por tarefa Spark. Poucas linhas de modelo no driver.
SMD_SCHEMA_MODELO = (
    "origem string, status_modelo string, detalhe string, modelo_json string, "
    "id_modelo_origem string, n_amostra long, n_grupos long, "
    "versao_modelo string, corte_treino string, relogio string"
)


def smd_ajustar_pdf(pdf):
    origem = str(pdf["estado"].iloc[0])
    row = dict(
        origem=origem, status_modelo="SEM_AJUSTE", detalhe="",
        modelo_json=None, id_modelo_origem=None, n_amostra=int(len(pdf)),
        n_grupos=0, versao_modelo=SMD_VERSAO,
        corte_treino=SM_CFG["corte_treino_exclusivo"], relogio=SM_CFG["relogio"],
    )
    if len(pdf) > MAX_LINHAS_POR_ORIGEM:
        row.update(status_modelo="LIMITE_MEMORIA", detalhe="Recorte nao foi truncado.")
        return pd.DataFrame([row])
    try:
        ordered = pdf.sort_values(["cd_bv", "passo"], kind="mergesort").reset_index(drop=True)
        model = smd_ajustar_origem(
            ordered, SMD_CONFIG_AJUSTE, float(SM_CFG["timeout_seg"]),
            "sem_acao:::classe",
        )
        model["origem"] = origem
        model["corte_treino"] = SM_CFG["corte_treino_exclusivo"]
        model["relogio"] = SM_CFG["relogio"]
        hashes = pd.util.hash_pandas_object(
            ordered[["cd_bv", "passo", "destino", "dur_min", "dur_max", "tipo_censura"]],
            index=False,
        ).to_numpy(np.uint64)
        model["assinatura_amostra"] = hashlib.sha256(hashes.tobytes()).hexdigest()
        payload = json.dumps(model, sort_keys=True, allow_nan=False)
        row.update(
            status_modelo="AJUSTADO", modelo_json=payload,
            id_modelo_origem=hashlib.sha256(payload.encode()).hexdigest(),
            n_grupos=int(model["n_grupos"]),
        )
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        row.update(status_modelo="FALHA_CORRECAO_TIMEOUT", detalhe=str(exc)[:500])
    return pd.DataFrame([row])


smd_tentativas = (
    smd_dados.groupBy("estado")
    .applyInPandas(smd_ajustar_pdf, schema=SMD_SCHEMA_MODELO)
    .localCheckpoint(eager=True)
)
# Mesmo schema: conserva os JSONs/hashes dos ajustes que nao foram selecionados.
smd_preservados = smd_base_modelos.join(smd_alvos, "origem", "left_anti")
smd_modelos = (
    smd_preservados.unionByName(smd_tentativas)
    .withColumn("versao_modelo", F.lit(SMD_VERSAO))
    .localCheckpoint(eager=True)
)
smd_bons_antes = smd_base_modelos.filter("status_modelo = 'AJUSTADO'").select(
    "origem", F.col("id_modelo_origem").alias("hash_antes"),
    F.col("modelo_json").alias("json_antes"),
)
smd_check = smd_bons_antes.join(smd_modelos, "origem", "left")
if smd_check.filter(
    ~F.col("hash_antes").eqNullSafe(F.col("id_modelo_origem"))
    | ~F.col("json_antes").eqNullSafe(F.col("modelo_json"))
).limit(1).count():
    raise RuntimeError("Um modelo anteriormente ajustado foi alterado.")
smd_modelos.createOrReplaceTempView(SMD_VIEWS["modelos"])
print("1. RESULTADO DA CORRECAO, APENAS NAS ORIGENS REJEITADAS")
smd_tentativas.groupBy("status_modelo", "detalhe").agg(
    F.count("*").alias("n_origens"), F.sum("n_amostra").alias("n_registros")
).show(50, truncate=False)
smd_tentativas.select("origem", "status_modelo", "n_grupos", "detalhe").orderBy(
    "status_modelo", "origem"
).show(40, truncate=False)
print("2. COBERTURA TOTAL DE MODELOS, INCLUINDO OS PRESERVADOS")
smd_modelos.groupBy("status_modelo", "n_grupos").count().show(50, truncate=False)


def smd_dict_modelos(df):
    result = {}
    for row in df.filter("status_modelo = 'AJUSTADO'").collect():
        model = json.loads(row["modelo_json"])
        model["id_modelo_origem"] = row["id_modelo_origem"]
        result[row["origem"]] = model
    return result


SMD_MODELOS = smd_dict_modelos(smd_modelos)
SMD_MODELOS_ANTES = smd_dict_modelos(smd_base_modelos)
if not SMD_MODELOS:
    raise ValueError("Nenhuma origem ajustada.")
SMD_BROADCAST = spark.sparkContext.broadcast(SMD_MODELOS)
SMD_BROADCAST_ANTES = spark.sparkContext.broadcast(SMD_MODELOS_ANTES)

smd_params = []
for origem, mod in SMD_MODELOS.items():
    g = int(mod["n_grupos"])
    families = mod.get("tipo_grupo", ["lognormal"] * g)
    shifts = mod.get("atraso_grupo_seg", [0.] * g)
    masses = mod.get("massa_atomo_grupo", [0.] * g)
    for j, dest in enumerate(mod["destinos"]):
        k = int(mod["grupo"][j])
        smd_params.append((
            origem, dest, k, families[k], float(shifts[k]), float(masses[k]),
            mod["mu_grupo"][k], mod["sigma_grupo"][k],
            float(mod["p_destino"][j]), int(mod["n_destino"][j]),
            SMD_VERSAO, mod["id_modelo_origem"],
        ))
spark.createDataFrame(smd_params, (
    "origem string, destino string, grupo_temporal long, familia_temporal string, "
    "deslocamento_seg double, massa_no_timeout_condicional double, "
    "meanlog_cauda_dias double, sdlog_cauda double, prob_na_entrada double, "
    "n_eventos_destino long, versao_modelo string, id_modelo_origem string"
)).createOrReplaceTempView(SMD_VIEWS["parametros"])
spark.createDataFrame(
    [(json.dumps(SMD_CFG, ensure_ascii=False, sort_keys=True),)], "config_json string"
).createOrReplaceTempView(SMD_VIEWS["config"])
print("Parametro massa_no_timeout_condicional pertence ao grupo de destino.")
print("Nao significa que todos os blocos terminam aos 1800 segundos.")


# COMMAND ----------
# Aplicacao V2.1: probabilidades por idade, sem reaplicar a parte 1.
# Mantem a familia anterior em origens preservadas, incluindo o silencio.
SMD_SCHEMA_OUTPUT = (
    "cd_bv string, data_referencia date, ts_corte_estado timestamp, acao_atual string, "
    "tempo_no_estado_seg double, ranking long, proxima_acao string, "
    "prob_proxima_acao double, prob_na_entrada double, prob_proxima_acao_7d double, "
    "prob_sem_saida_7d double, massa_top5 double, status_previsao string, "
    "status_temporal string, status_dados string, versao_modelo string, "
    "id_modelo_origem string, relogio string, publicavel boolean"
)
SMD_COLUMNS_OUTPUT = [item.strip().split()[0] for item in SMD_SCHEMA_OUTPUT.split(",")]


def smd_prever_pdf(pdf, modelos, versao, horizonte):
    saidas = []
    for origem, parte in pdf.groupby("acao_atual", sort=False, dropna=False):
        model = modelos.get(origem)
        for start in range(0, len(parte), 512):
            block = parte.iloc[start:start + 512].reset_index(drop=True)
            valid = (
                (block["status_input"] == "OK")
                & block["tempo_no_estado_seg"].notna()
                & (block["tempo_no_estado_seg"] >= 0)
            )
            selected = np.flatnonzero(valid.to_numpy()) if model else np.array([], int)
            forecasts, outside = {}, set()
            if len(selected):
                ages = block.iloc[selected]["tempo_no_estado_seg"].to_numpy(float) / 86400.
                # Um atomo puro tem suporte finito. Nao desvia para q estatico
                # quando a idade observada e incompatível com esse modelo.
                ls = smd_log_s_grupos(model, ages)
                den = special.logsumexp(
                    np.log(np.asarray(model["pi_grupo"])) + ls, axis=1
                )
                supported = np.isfinite(den)
                outside = set(selected[~supported].tolist())
                inside = selected[supported]
                if len(inside):
                    q, qh, ns = smd_prever_destinos(model, ages[supported], horizonte)
                    for k, ix in enumerate(inside):
                        rank = np.argsort(-q[k], kind="mergesort")
                        # Nao completa top 5 com destinos de probabilidade zero.
                        rank = rank[q[k, rank] > 0][:SMD_TOP_K]
                        forecasts[int(ix)] = (
                            q[k], qh[k], ns[k], rank, ages[supported][k]
                        )
            for ix, row in block.iterrows():
                base = dict(
                    cd_bv=str(row["cd_bv"]), data_referencia=row["data_referencia"],
                    ts_corte_estado=row["ts_corte_estado"],
                    acao_atual=None if pd.isna(row["acao_atual"]) else str(row["acao_atual"]),
                    tempo_no_estado_seg=(
                        None if pd.isna(row["tempo_no_estado_seg"])
                        else float(row["tempo_no_estado_seg"])
                    ),
                    ranking=None, proxima_acao=None, prob_proxima_acao=None,
                    prob_na_entrada=None, prob_proxima_acao_7d=None,
                    prob_sem_saida_7d=None, massa_top5=None,
                    status_previsao="SEM_MODELO_ORIGEM",
                    status_temporal="NAO_CALCULADO", status_dados=str(row["status_dados"]),
                    versao_modelo=versao, id_modelo_origem=None,
                    relogio=str(row["relogio"]), publicavel=False,
                )
                if ix not in forecasts:
                    if row["status_input"] != "OK":
                        base["status_previsao"] = str(row["status_input"])
                    elif ix in outside:
                        base["status_previsao"] = "IDADE_FORA_SUPORTE_MODELO"
                    saidas.append(base)
                    continue
                q, qh, ns, rank, age = forecasts[ix]
                tipo = (
                    "EXTRAPOLACAO_TEMPORAL"
                    if age > model["max_tempo_observado_dias"]
                    else "TIMEOUT_MISTO_POR_DESTINO"
                    if "atomo" in "|".join(model.get("tipo_grupo", []))
                    else "GRUPOS_TEMPORAIS_POR_DESTINO"
                    if model["n_grupos"] > 1
                    else "POOLING_UM_GRUPO_TEMPORAL"
                )
                for pos, j in enumerate(rank, 1):
                    saidas.append(dict(
                        base, ranking=pos, proxima_acao=model["destinos"][j],
                        prob_proxima_acao=float(q[j]),
                        prob_na_entrada=float(model["p_destino"][j]),
                        prob_proxima_acao_7d=float(qh[j]), prob_sem_saida_7d=float(ns),
                        massa_top5=float(q[rank].sum()),
                        status_previsao="PREVISAO_SEMIMARKOV",
                        status_temporal=tipo, id_modelo_origem=model["id_modelo_origem"],
                    ))
    result = pd.DataFrame(saidas, columns=SMD_COLUMNS_OUTPUT)
    if len(result):
        result["ranking"] = pd.array(result["ranking"], dtype="Int64")
    return result


def smd_prever_lotes(iterator: Iterator[pd.DataFrame]):
    modelos = SMD_BROADCAST.value
    for pdf in iterator:
        yield smd_prever_pdf(pdf, modelos, SMD_VERSAO, SMD_HORIZONTE)


smd_input = spark.table(SM_VIEWS["atuais"]).select(
    "cd_bv", "data_referencia", "ts_corte_estado", "acao_atual", "tempo_no_estado_seg",
    "status_input", "status_dados", "relogio",
)
smd_previsoes = (
    smd_input.mapInPandas(smd_prever_lotes, schema=SMD_SCHEMA_OUTPUT)
    .localCheckpoint(eager=True)
)
smd_previsoes.createOrReplaceTempView(SMD_VIEWS["previsoes"])
print("3. OUTPUT CANDIDATO V2.1")
smd_previsoes.groupBy("status_previsao", "status_temporal").agg(
    F.countDistinct("cd_bv").alias("n_clientes"), F.count("*").alias("n_linhas")
).show(truncate=False)
print("Publico e corte conservados. publicavel=False. Nao ha escrita permanente.")


# COMMAND ----------
# Comparacao ANTES/DEPOIS com mesmo dataset e mesma convencao no marco.
# Agora inclui eventos exatamente no marco, coerente com T>=a/corte exclusivo.
# Por isso nao compare diretamente com o quadro antigo de 30 minutos (T>a).
# Nao compara NLL de densidade antiga com massa atomica.
SMD_SCHEMA_VALIDACAO = (
    "origem string, cenario string, idade_marco_seg double, n_elegiveis long, "
    "n_eventos long, n_eventos_suportados long, n_top1_temporal long, "
    "n_top5_temporal long, n_top1_entrada long, status_validacao string"
)


def smd_validar_pdf(pdf):
    origem = str(pdf["estado"].iloc[0])
    outputs = []
    for marco in SMD_MARCOS_SEG:
        case = pdf[pdf["dur_min"] >= marco].copy()
        # A base operacional V2 nao tem intervalos. Recusa outra semantica.
        if (case["tipo_censura"] == "intervalo").any():
            raise ValueError("VALIDACAO_EXIGE_BASE_OPERACIONAL_EXATA_DIREITA")
        conhecidos = case.loc[case["tipo_censura"] == "exata", "destino"]
        for label, models in (
            ("ANTES", SMD_BROADCAST_ANTES.value),
            ("DEPOIS", SMD_BROADCAST.value),
        ):
            row = dict(
                origem=origem, cenario=label, idade_marco_seg=marco,
                n_elegiveis=int(len(case)), n_eventos=int(len(conhecidos)),
                n_eventos_suportados=0, n_top1_temporal=0, n_top5_temporal=0,
                n_top1_entrada=0, status_validacao="SEM_MODELO",
            )
            model = models.get(origem)
            if len(pdf) > MAX_LINHAS_POR_ORIGEM:
                row["status_validacao"] = "LIMITE_MEMORIA_VALIDACAO"
            elif model and len(case):
                den = special.logsumexp(
                    np.log(np.asarray(model["pi_grupo"]))
                    + smd_log_s_grupos(model, [marco / 86400.])[0]
                )
                if np.isfinite(den):
                    q, _, _ = smd_prever_destinos(
                        model, [marco / 86400.], SMD_HORIZONTE
                    )
                    order = np.argsort(-q[0], kind="mergesort")
                    order = order[q[0, order] > 0][:SMD_TOP_K]
                    top = [model["destinos"][j] for j in order]
                    available = [
                        d for j, d in enumerate(model["destinos"]) if q[0, j] > 0
                    ]
                    static = model["destinos"][int(np.argmax(model["p_destino"]))]
                    row.update(
                        n_eventos_suportados=int(conhecidos.isin(available).sum()),
                        n_top1_temporal=int((conhecidos == top[0]).sum()),
                        n_top5_temporal=int(conhecidos.isin(top).sum()),
                        n_top1_entrada=int((conhecidos == static).sum()),
                        status_validacao="AVALIADO_T_MAIOR_IGUAL_MARCO",
                    )
                else:
                    row["status_validacao"] = "IDADE_FORA_SUPORTE_MODELO"
            outputs.append(row)
    return pd.DataFrame(outputs)


smd_validacao = (
    smd_val_dados.groupBy("estado")
    .applyInPandas(smd_validar_pdf, schema=SMD_SCHEMA_VALIDACAO)
    .localCheckpoint(eager=True)
)
smd_validacao.createOrReplaceTempView(SMD_VIEWS["validacao"])
print("4. VALIDACAO PAREADA: MESMAS LINHAS, MESMOS MARCOS, SEM MUDAR AMOSTRA")
smd_validacao.groupBy("cenario", "idade_marco_seg").agg(
    F.sum("n_elegiveis").alias("n_elegiveis"),
    F.sum("n_eventos").alias("n_eventos"),
    F.sum("n_eventos_suportados").alias("n_eventos_suportados"),
    (F.sum("n_eventos_suportados") / F.sum("n_eventos")).alias("cobertura_destinos"),
    (F.sum("n_top1_temporal") / F.sum("n_eventos")).alias("top1_temporal_total"),
    (F.sum("n_top5_temporal") / F.sum("n_eventos")).alias("top5_temporal_total"),
    (F.sum("n_top1_entrada") / F.sum("n_eventos")).alias("top1_entrada_total"),
).orderBy("idade_marco_seg", "cenario").show(truncate=False)
print("NLL continua e massa pontual nao sao comparadas como se fossem a mesma medida.")
print("Casos ainda sem suporte/limite permanecem sinalizados; nao ha fallback estatico.")
print("Parte 3: usar somente a revisao V2.1, inicialmente sem gravar.")
