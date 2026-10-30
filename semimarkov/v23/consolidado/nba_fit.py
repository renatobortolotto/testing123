"""Ajuste A/B original: mesma verossimilhança, priors, limites e otimizador."""
import copy
import hashlib
import json
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import optimize, special

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
