# Databricks notebook source
# NBA | V2.2 - Parte 02
# Ajuste Semi-Markov sobre:
#   ultima acao real + tempo desde a ultima acao.
#
# Pre-requisito:
#   executar nba_v22_01_base_ultima_acao_debounce.py.
#
# Esta etapa:
# - usa apenas clientes de treino para o ajuste;
# - ajusta probabilidades de destino + Lognormal por grupo temporal;
# - trata a ultima permanencia como censura a direita;
# - valida em clientes que nao participaram do ajuste;
# - gera Top 5 para a amostra atual da customers_query;
# - nao grava tabelas permanentes.

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import json
import math

import numpy as np
import pandas as pd
from scipy import optimize, special
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


# =========================
# Configuracao
# =========================

ALVO_LINHAS_POR_ORIGEM = 20_000
SEMENTE_AJUSTE = 20260930

IDADES_VALIDACAO_SEG = (
    0.0,
    1800.0,
    86400.0,
    604800.0,
)

TOP_K = 5


@dataclass(frozen=True)
class AjusteV22:
    minimo_eventos_origem: int = 80
    minimo_clientes_origem: int = 20
    minimo_eventos_grupo: int = 100
    minimo_clientes_grupo: int = 15
    max_grupos_proprios: int = 8
    pseudocontagem_destino: float = 0.25
    regularizacao_logits: float = 0.05
    sigma_min: float = 0.15
    sigma_max: float = 4.5
    maxiter: int = 800


SM22_AJUSTE = AjusteV22()

SM22_MODEL_VIEW = "nba_sm_v22_modelos"
SM22_VALID_VIEW = "nba_sm_v22_validacao"
SM22_PREV_VIEW = "nba_sm_v22_previsoes_top5"


# =========================
# Validacoes
# =========================

_requeridos = [
    "SM22_CFG",
    "SM22_VIEWS",
    "base_treino_v22",
    "atuais_v22",
]

_faltantes = [
    nome
    for nome in _requeridos
    if nome not in globals()
]

if _faltantes:
    raise RuntimeError(
        "Execute a Parte 01 V2.2 antes. "
        f"Ausentes: {_faltantes}"
    )

if SM22_CFG["relogio"] != "ULTIMA_ACAO_TEMPO_V22":
    raise ValueError(
        "Esta etapa exige o relogio ULTIMA_ACAO_TEMPO_V22."
    )


# =========================
# Amostragem por origem
# =========================

treino_elegivel = (
    base_treino_v22
    .filter(
        (~F.col("validacao_cliente"))
        & F.col("elegivel_ajuste")
    )
    .select(
        "cd_bv",
        "estado",
        "destino",
        "dur_min",
        "tipo_censura",
    )
)

tamanhos = {
    row["estado"]: int(row["count"])
    for row in (
        treino_elegivel
        .groupBy("estado")
        .count()
        .collect()
    )
}

fracoes = {
    estado: min(
        1.0,
        ALVO_LINHAS_POR_ORIGEM / max(n, 1),
    )
    for estado, n in tamanhos.items()
}

amostra_ajuste = (
    treino_elegivel
    .sampleBy(
        "estado",
        fractions=fracoes,
        seed=SEMENTE_AJUSTE,
    )
    .localCheckpoint(eager=True)
)

print(
    "Origens candidatas:",
    len(tamanhos),
)

print(
    "Linhas na amostra de ajuste:",
    amostra_ajuste.count(),
)


# =========================
# Nucleo estatistico
# =========================

def _softmax_referencia(
    logits_reduzidos: np.ndarray,
) -> np.ndarray:
    if logits_reduzidos.size == 0:
        return np.array([1.0])

    logits = np.concatenate(
        [
            logits_reduzidos,
            np.array([0.0]),
        ]
    )

    logits -= logits.max()
    exp_logits = np.exp(logits)

    return exp_logits / exp_logits.sum()


def _logpdf_lognormal(
    t: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    log_t = np.log(t)

    return (
        -log_t
        - math.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((log_t - mu) / sigma) ** 2
    )


def _logsf_lognormal(
    t: np.ndarray,
    mu: float,
    sigma: float,
) -> np.ndarray:
    z = (np.log(t) - mu) / sigma

    return special.log_ndtr(-z)


def _preparar_origem(
    pdf: pd.DataFrame,
    cfg: AjusteV22,
) -> dict:
    exatas = pdf[
        pdf["tipo_censura"] == "exata"
    ].copy()

    direitas = pdf[
        pdf["tipo_censura"] == "direita"
    ].copy()

    if exatas.empty:
        raise ValueError("SEM_EVENTOS_EXATOS")

    suporte = (
        exatas.groupby("destino")
        .agg(
            n=("destino", "size"),
            n_clientes=("cd_bv", "nunique"),
        )
        .sort_values(
            ["n", "n_clientes"],
            ascending=False,
        )
    )

    destinos = suporte.index.astype(str).tolist()

    if len(exatas) < cfg.minimo_eventos_origem:
        raise ValueError("SUPORTE_EVENTOS_ORIGEM")

    if (
        exatas["cd_bv"].nunique()
        < cfg.minimo_clientes_origem
    ):
        raise ValueError("SUPORTE_CLIENTES_ORIGEM")

    proprios = (
        suporte[
            (suporte["n"] >= cfg.minimo_eventos_grupo)
            & (
                suporte["n_clientes"]
                >= cfg.minimo_clientes_grupo
            )
        ]
        .head(cfg.max_grupos_proprios)
        .index.astype(str)
        .tolist()
    )

    grupo_destino = {}
    grupos_nomes = []

    for destino in proprios:
        grupo_destino[destino] = len(grupos_nomes)
        grupos_nomes.append(destino)

    restantes = [
        destino
        for destino in destinos
        if destino not in grupo_destino
    ]

    if restantes:
        grupo_outros = len(grupos_nomes)
        grupos_nomes.append("__OUTROS__")

        for destino in restantes:
            grupo_destino[destino] = grupo_outros

    if not grupos_nomes:
        raise ValueError("SEM_GRUPOS")

    grupo = np.array(
        [
            grupo_destino[destino]
            for destino in destinos
        ],
        dtype=int,
    )

    n_j = suporte.loc[destinos, "n"].to_numpy(float)
    r_destino = np.empty(len(destinos), dtype=float)

    for g in range(len(grupos_nomes)):
        idx = np.flatnonzero(grupo == g)

        contagens = (
            n_j[idx]
            + cfg.pseudocontagem_destino
        )

        r_destino[idx] = (
            contagens / contagens.sum()
        )

    destino_pos = {
        destino: i
        for i, destino in enumerate(destinos)
    }

    exatas["destino_pos"] = (
        exatas["destino"]
        .map(destino_pos)
        .astype(int)
    )

    exatas["grupo"] = exatas[
        "destino_pos"
    ].map(
        lambda j: int(grupo[j])
    )

    t_exata = (
        exatas["dur_min"]
        .to_numpy(float)
        / 86400.0
    )

    t_direita = (
        direitas["dur_min"]
        .to_numpy(float)
        / 86400.0
    )

    if np.any(t_exata <= 0):
        raise ValueError("TEMPO_EXATO_NAO_POSITIVO")

    if len(t_direita) and np.any(t_direita <= 0):
        raise ValueError("TEMPO_CENSURA_NAO_POSITIVO")

    g_exata = exatas["grupo"].to_numpy(int)

    contagem_grupo = np.bincount(
        g_exata,
        minlength=len(grupos_nomes),
    ).astype(float)

    pi_inicial = (
        contagem_grupo
        + cfg.pseudocontagem_destino
    )

    pi_inicial /= pi_inicial.sum()

    mu_inicial = np.empty(
        len(grupos_nomes),
        dtype=float,
    )

    sigma_inicial = np.empty(
        len(grupos_nomes),
        dtype=float,
    )

    log_t = np.log(t_exata)

    mu_global = float(log_t.mean())
    sigma_global = max(
        float(log_t.std(ddof=0)),
        0.5,
    )

    for g in range(len(grupos_nomes)):
        valores = log_t[
            g_exata == g
        ]

        if len(valores):
            mu_inicial[g] = float(
                valores.mean()
            )

            sigma_inicial[g] = float(
                max(
                    valores.std(ddof=0),
                    cfg.sigma_min * 1.2,
                )
            )
        else:
            mu_inicial[g] = mu_global
            sigma_inicial[g] = sigma_global

        sigma_inicial[g] = min(
            sigma_inicial[g],
            cfg.sigma_max * 0.8,
        )

    todos_t = np.concatenate(
        [
            t_exata,
            t_direita,
        ]
    )

    log_min = float(
        np.log(todos_t.min())
    )

    log_max = float(
        np.log(todos_t.max())
    )

    return {
        "destinos": destinos,
        "grupo": grupo,
        "grupos_nomes": grupos_nomes,
        "r_destino": r_destino,
        "n_j": n_j,
        "suporte": suporte,
        "t_exata": t_exata,
        "t_direita": t_direita,
        "g_exata": g_exata,
        "pi_inicial": pi_inicial,
        "mu_inicial": mu_inicial,
        "sigma_inicial": sigma_inicial,
        "log_min": log_min,
        "log_max": log_max,
    }


def _ajustar_origem_pdf(
    pdf: pd.DataFrame,
    cfg: AjusteV22,
) -> dict:
    prep = _preparar_origem(
        pdf,
        cfg,
    )

    g = len(prep["grupos_nomes"])

    if g > 1:
        logits_iniciais = np.log(
            prep["pi_inicial"][:-1]
            / prep["pi_inicial"][-1]
        )
    else:
        logits_iniciais = np.array(
            [],
            dtype=float,
        )

    theta0 = np.concatenate(
        [
            logits_iniciais,
            prep["mu_inicial"],
            np.log(
                prep["sigma_inicial"]
            ),
        ]
    )

    bounds = (
        [(-12.0, 12.0)] * max(g - 1, 0)
        + [
            (
                prep["log_min"] - 3.0,
                prep["log_max"] + 3.0,
            )
        ] * g
        + [
            (
                math.log(cfg.sigma_min),
                math.log(cfg.sigma_max),
            )
        ] * g
    )

    n_logits = max(g - 1, 0)

    def unpack(
        theta: np.ndarray,
    ):
        logits = theta[
            :n_logits
        ]

        inicio_mu = n_logits
        fim_mu = inicio_mu + g

        mu = theta[
            inicio_mu:fim_mu
        ]

        sigma = np.exp(
            theta[fim_mu:]
        )

        pi = _softmax_referencia(
            logits
        )

        return pi, mu, sigma, logits

    def objetivo(
        theta: np.ndarray,
    ) -> float:
        pi, mu, sigma, logits = unpack(
            theta
        )

        log_pi = np.log(pi)

        ll = 0.0

        for grupo_id in range(g):
            mask = (
                prep["g_exata"]
                == grupo_id
            )

            if not np.any(mask):
                continue

            ll += float(
                (
                    log_pi[grupo_id]
                    + _logpdf_lognormal(
                        prep["t_exata"][mask],
                        mu[grupo_id],
                        sigma[grupo_id],
                    )
                ).sum()
            )

        pos_destino = {
            destino: i
            for i, destino
            in enumerate(
                prep["destinos"]
            )
        }

        exatas = pdf[
            pdf["tipo_censura"]
            == "exata"
        ]

        for destino, n in (
            exatas["destino"]
            .value_counts()
            .items()
        ):
            j = pos_destino[
                str(destino)
            ]

            ll += (
                float(n)
                * math.log(
                    prep["r_destino"][j]
                )
            )

        if len(
            prep["t_direita"]
        ):
            log_sf = np.column_stack(
                [
                    _logsf_lognormal(
                        prep["t_direita"],
                        mu[k],
                        sigma[k],
                    )
                    for k in range(g)
                ]
            )

            ll += float(
                special.logsumexp(
                    log_pi[None, :]
                    + log_sf,
                    axis=1,
                ).sum()
            )

        if g > 1:
            alvo_logits = np.log(
                prep["pi_inicial"][:-1]
                / prep["pi_inicial"][-1]
            )

            penalidade = (
                cfg.regularizacao_logits
                * np.square(
                    logits
                    - alvo_logits
                ).sum()
            )
        else:
            penalidade = 0.0

        valor = -ll + penalidade

        if not np.isfinite(valor):
            return np.inf

        return float(valor)

    resultado = optimize.minimize(
        objetivo,
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": cfg.maxiter,
            "ftol": 1e-10,
            "gtol": 1e-6,
        },
    )

    if (
        not resultado.success
        or not np.isfinite(
            resultado.fun
        )
    ):
        raise ValueError(
            "OTIMIZACAO_NAO_CONVERGIU"
        )

    pi, mu, sigma, _ = unpack(
        resultado.x
    )

    hits = []

    for valor, (
        limite_inferior,
        limite_superior,
    ) in zip(
        resultado.x,
        bounds,
    ):
        if (
            abs(
                valor
                - limite_inferior
            )
            < 1e-4
        ):
            hits.append("INFERIOR")

        elif (
            abs(
                valor
                - limite_superior
            )
            < 1e-4
        ):
            hits.append("SUPERIOR")

    p_destino = (
        pi[prep["grupo"]]
        * prep["r_destino"]
    )

    modelo = {
        "formato": "sm_v22_lognormal_destino",
        "versao": SM22_CFG["versao_modelo"],
        "relogio": SM22_CFG["relogio"],
        "unidade_tempo": "dias",
        "destinos": prep["destinos"],
        "grupo": prep["grupo"].tolist(),
        "grupos_nomes": prep[
            "grupos_nomes"
        ],
        "r_destino_no_grupo": prep[
            "r_destino"
        ].tolist(),
        "pi_grupo": pi.tolist(),
        "mu_grupo": mu.tolist(),
        "sigma_grupo": sigma.tolist(),
        "p_destino": p_destino.tolist(),
        "n_destino": prep[
            "n_j"
        ].astype(int).tolist(),
        "n_grupos": int(g),
        "n_eventos": int(
            len(
                prep["t_exata"]
            )
        ),
        "n_direita": int(
            len(
                prep["t_direita"]
            )
        ),
        "n_clientes": int(
            pdf["cd_bv"].nunique()
        ),
        "max_tempo_observado_dias": float(
            max(
                np.max(
                    prep["t_exata"]
                ),
                np.max(
                    prep["t_direita"]
                )
                if len(
                    prep["t_direita"]
                )
                else 0.0,
            )
        ),
        "tempo_dependente_destino": bool(
            g > 1
        ),
        "alerta_limite": bool(hits),
        "ajuste_cfg": asdict(cfg),
        "funcao_objetivo": float(
            resultado.fun
        ),
        "n_iteracoes": int(
            resultado.nit
        ),
    }

    json.dumps(
        modelo,
        allow_nan=False,
    )

    return modelo


# =========================
# Ajuste distribuido
# =========================

MODELO_SCHEMA = StructType(
    [
        StructField(
            "origem",
            StringType(),
            False,
        ),
        StructField(
            "status_modelo",
            StringType(),
            False,
        ),
        StructField(
            "n_amostra",
            LongType(),
            False,
        ),
        StructField(
            "n_eventos",
            LongType(),
            True,
        ),
        StructField(
            "n_direita",
            LongType(),
            True,
        ),
        StructField(
            "n_clientes",
            LongType(),
            True,
        ),
        StructField(
            "n_destinos",
            LongType(),
            True,
        ),
        StructField(
            "n_grupos",
            LongType(),
            True,
        ),
        StructField(
            "alerta_limite",
            StringType(),
            True,
        ),
        StructField(
            "modelo_json",
            StringType(),
            True,
        ),
        StructField(
            "detalhe",
            StringType(),
            True,
        ),
    ]
)


def ajustar_grupo(
    pdf: pd.DataFrame,
) -> pd.DataFrame:
    origem = str(
        pdf["estado"].iloc[0]
    )

    base = {
        "origem": origem,
        "status_modelo": (
            "FALHA_OU_SUPORTE_INSUFICIENTE"
        ),
        "n_amostra": int(
            len(pdf)
        ),
        "n_eventos": None,
        "n_direita": None,
        "n_clientes": int(
            pdf["cd_bv"].nunique()
        ),
        "n_destinos": None,
        "n_grupos": None,
        "alerta_limite": None,
        "modelo_json": None,
        "detalhe": None,
    }

    try:
        modelo = _ajustar_origem_pdf(
            pdf,
            SM22_AJUSTE,
        )

        base.update(
            {
                "status_modelo": "AJUSTADO",
                "n_eventos": int(
                    modelo["n_eventos"]
                ),
                "n_direita": int(
                    modelo["n_direita"]
                ),
                "n_clientes": int(
                    modelo["n_clientes"]
                ),
                "n_destinos": int(
                    len(
                        modelo["destinos"]
                    )
                ),
                "n_grupos": int(
                    modelo["n_grupos"]
                ),
                "alerta_limite": (
                    "SIM"
                    if modelo[
                        "alerta_limite"
                    ]
                    else "NAO"
                ),
                "modelo_json": json.dumps(
                    modelo,
                    allow_nan=False,
                ),
                "detalhe": "",
            }
        )

    except Exception as erro:
        base["detalhe"] = str(
            erro
        )[:500]

    return pd.DataFrame(
        [base]
    )


sm22_modelos = (
    amostra_ajuste
    .groupBy("estado")
    .applyInPandas(
        ajustar_grupo,
        schema=MODELO_SCHEMA,
    )
    .localCheckpoint(
        eager=True
    )
)

sm22_modelos.createOrReplaceTempView(
    SM22_MODEL_VIEW
)

print("MODELOS V2.2")

(
    sm22_modelos
    .groupBy(
        "status_modelo",
        "n_grupos",
        "alerta_limite",
    )
    .count()
    .orderBy(
        "status_modelo",
        "n_grupos",
    )
    .show(
        100,
        truncate=False,
    )
)

print("FALHAS V2.2")

(
    sm22_modelos
    .filter(
        F.col("status_modelo")
        != "AJUSTADO"
    )
    .groupBy(
        "detalhe"
    )
    .count()
    .orderBy(
        F.desc("count")
    )
    .show(
        50,
        truncate=False,
    )
)


# =========================
# Carregar modelos
# =========================

MODELOS_V22 = {}

for row in (
    sm22_modelos
    .filter(
        F.col("status_modelo")
        == "AJUSTADO"
    )
    .collect()
):
    MODELOS_V22[
        row["origem"]
    ] = json.loads(
        row["modelo_json"]
    )

SM22_BROADCAST = (
    spark.sparkContext.broadcast(
        MODELOS_V22
    )
)

print(
    "Origens ajustadas V2.2:",
    len(MODELOS_V22),
)


# =========================
# Inferencia
# =========================

def sm22_log_s_grupos(
    modelo: dict,
    idade_dias,
) -> np.ndarray:
    idade = np.asarray(
        idade_dias,
        dtype=float,
    ).reshape(-1)

    if (
        not np.isfinite(
            idade
        ).all()
        or np.any(
            idade < 0
        )
    ):
        raise ValueError(
            "IDADE_INVALIDA"
        )

    mus = np.asarray(
        modelo["mu_grupo"],
        dtype=float,
    )

    sigmas = np.asarray(
        modelo["sigma_grupo"],
        dtype=float,
    )

    resultado = np.zeros(
        (
            len(idade),
            len(mus),
        ),
        dtype=float,
    )

    zero = (
        idade <= 0
    )

    resultado[
        zero,
        :,
    ] = 0.0

    positivo = ~zero

    if np.any(positivo):
        log_idade = np.log(
            idade[
                positivo
            ]
        )

        z = (
            log_idade[:, None]
            - mus[None, :]
        ) / sigmas[None, :]

        resultado[
            positivo,
            :,
        ] = special.log_ndtr(
            -z
        )

    return resultado


def sm22_prever_destinos(
    modelo: dict,
    idade_dias,
    horizonte_dias: float = 7.0,
):
    idade = np.asarray(
        idade_dias,
        dtype=float,
    ).reshape(-1)

    log_s = sm22_log_s_grupos(
        modelo,
        idade,
    )

    log_pi = np.log(
        np.asarray(
            modelo["pi_grupo"],
            dtype=float,
        )
    )

    denominador = special.logsumexp(
        log_pi[None, :]
        + log_s,
        axis=1,
    )

    if not np.isfinite(
        denominador
    ).all():
        raise ValueError(
            "IDADE_FORA_SUPORTE"
        )

    peso_grupo = np.exp(
        log_pi[None, :]
        + log_s
        - denominador[:, None]
    )

    grupo = np.asarray(
        modelo["grupo"],
        dtype=int,
    )

    r = np.asarray(
        modelo[
            "r_destino_no_grupo"
        ],
        dtype=float,
    )

    q = (
        peso_grupo[:, grupo]
        * r[None, :]
    )

    log_s_h = sm22_log_s_grupos(
        modelo,
        idade
        + horizonte_dias,
    )

    exits = np.zeros_like(
        log_s
    )

    finito = np.isfinite(
        log_s
    )

    delta = (
        log_s_h[finito]
        - log_s[finito]
    )

    exits[finito] = -np.expm1(
        np.minimum(
            delta,
            0.0,
        )
    )

    q_h = (
        (
            peso_grupo
            * exits
        )[:, grupo]
        * r[None, :]
    )

    sem_saida = np.exp(
        special.logsumexp(
            log_pi[None, :]
            + log_s_h,
            axis=1,
        )
        - denominador
    )

    if not np.allclose(
        q.sum(axis=1),
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "MASSA_DESTINO_NAO_FECHA"
        )

    if not np.allclose(
        q_h.sum(axis=1)
        + sem_saida,
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "MASSA_HORIZONTE_NAO_FECHA"
        )

    return q, q_h, sem_saida


# =========================
# Validacao em holdout
# =========================

VALID_SCHEMA = (
    "origem string, idade_seg double, "
    "n_eventos long, n_suportados long, "
    "n_top1 long, n_top5 long"
)


def validar_origem(
    pdf: pd.DataFrame,
) -> pd.DataFrame:
    origem = str(
        pdf["estado"].iloc[0]
    )

    modelo = (
        SM22_BROADCAST.value.get(
            origem
        )
    )

    linhas = []

    for idade_seg in (
        IDADES_VALIDACAO_SEG
    ):
        caso = pdf[
            (
                pdf["tipo_censura"]
                == "exata"
            )
            & (
                pdf["dur_min"]
                >= idade_seg
            )
        ].copy()

        row = {
            "origem": origem,
            "idade_seg": float(
                idade_seg
            ),
            "n_eventos": int(
                len(caso)
            ),
            "n_suportados": 0,
            "n_top1": 0,
            "n_top5": 0,
        }

        if (
            modelo is None
            or caso.empty
        ):
            linhas.append(row)
            continue

        q, _, _ = (
            sm22_prever_destinos(
                modelo,
                np.array(
                    [
                        idade_seg
                        / 86400.0
                    ],
                    dtype=float,
                ),
                SM22_CFG[
                    "horizonte_dias"
                ],
            )
        )

        ordem = np.argsort(
            -q[0],
            kind="mergesort",
        )

        top1 = (
            modelo[
                "destinos"
            ][
                int(
                    ordem[0]
                )
            ]
        )

        top5 = {
            modelo[
                "destinos"
            ][int(j)]
            for j in ordem[:TOP_K]
        }

        suportados = caso[
            "destino"
        ].isin(
            modelo["destinos"]
        )

        caso_sup = caso[
            suportados
        ]

        row[
            "n_suportados"
        ] = int(
            len(
                caso_sup
            )
        )

        row[
            "n_top1"
        ] = int(
            (
                caso_sup[
                    "destino"
                ]
                == top1
            ).sum()
        )

        row[
            "n_top5"
        ] = int(
            caso_sup[
                "destino"
            ].isin(
                top5
            ).sum()
        )

        linhas.append(row)

    return pd.DataFrame(
        linhas
    )


validacao_base = (
    base_treino_v22
    .filter(
        F.col(
            "validacao_cliente"
        )
        & F.col(
            "elegivel_ajuste"
        )
    )
    .select(
        "cd_bv",
        "estado",
        "destino",
        "dur_min",
        "tipo_censura",
    )
)

sm22_validacao = (
    validacao_base
    .groupBy("estado")
    .applyInPandas(
        validar_origem,
        schema=VALID_SCHEMA,
    )
    .localCheckpoint(
        eager=True
    )
)

sm22_validacao.createOrReplaceTempView(
    SM22_VALID_VIEW
)

resumo_validacao = (
    sm22_validacao
    .groupBy(
        "idade_seg"
    )
    .agg(
        F.sum(
            "n_eventos"
        ).alias(
            "n_eventos"
        ),
        F.sum(
            "n_suportados"
        ).alias(
            "n_suportados"
        ),
        F.sum(
            "n_top1"
        ).alias(
            "n_top1"
        ),
        F.sum(
            "n_top5"
        ).alias(
            "n_top5"
        ),
    )
    .withColumn(
        "cobertura",
        F.when(
            F.col("n_eventos") > 0,
            F.col("n_suportados")
            / F.col("n_eventos"),
        ),
    )
    .withColumn(
        "acerto_top1_suporte",
        F.when(
            F.col(
                "n_suportados"
            )
            > 0,
            F.col("n_top1")
            / F.col(
                "n_suportados"
            ),
        ),
    )
    .withColumn(
        "acerto_top5_suporte",
        F.when(
            F.col(
                "n_suportados"
            )
            > 0,
            F.col("n_top5")
            / F.col(
                "n_suportados"
            ),
        ),
    )
    .orderBy(
        "idade_seg"
    )
)

print(
    "VALIDACAO V2.2"
)

resumo_validacao.show(
    truncate=False
)


# =========================
# Scoring da amostra atual
# =========================

PREV_SCHEMA = (
    "cd_bv string, data_referencia date, "
    "ultima_acao string, "
    "tempo_desde_ultima_acao_seg double, "
    "ranking long, proxima_acao string, "
    "prob_proxima_acao double, "
    "status_previsao string, "
    "status_temporal string, "
    "versao_modelo string"
)


def prever_lote(
    iterator: Iterator[
        pd.DataFrame
    ],
):
    modelos = (
        SM22_BROADCAST.value
    )

    for pdf in iterator:
        saidas = []

        for row in (
            pdf.itertuples(
                index=False
            )
        ):
            base = {
                "cd_bv": str(
                    row.cd_bv
                ),
                "data_referencia": (
                    row.data_referencia
                ),
                "ultima_acao": (
                    None
                    if pd.isna(
                        row.ultima_acao
                    )
                    else str(
                        row.ultima_acao
                    )
                ),
                "tempo_desde_ultima_acao_seg": (
                    None
                    if pd.isna(
                        row.tempo_desde_ultima_acao_seg
                    )
                    else float(
                        row.tempo_desde_ultima_acao_seg
                    )
                ),
                "ranking": None,
                "proxima_acao": None,
                "prob_proxima_acao": None,
                "status_previsao": (
                    "SEM_MODELO_ORIGEM"
                ),
                "status_temporal": (
                    "NAO_CALCULADO"
                ),
                "versao_modelo": (
                    SM22_CFG[
                        "versao_modelo"
                    ]
                ),
            }

            if row.status_input != "OK":
                base[
                    "status_previsao"
                ] = str(
                    row.status_input
                )

                saidas.append(
                    base
                )

                continue

            modelo = modelos.get(
                str(
                    row.ultima_acao
                )
            )

            if modelo is None:
                saidas.append(
                    base
                )

                continue

            idade = (
                float(
                    row.tempo_desde_ultima_acao_seg
                )
                / 86400.0
            )

            try:
                q, _, _ = (
                    sm22_prever_destinos(
                        modelo,
                        np.array(
                            [idade],
                            dtype=float,
                        ),
                        SM22_CFG[
                            "horizonte_dias"
                        ],
                    )
                )

            except ValueError:
                base[
                    "status_previsao"
                ] = (
                    "IDADE_FORA_SUPORTE_MODELO"
                )

                saidas.append(
                    base
                )

                continue

            ordem = np.argsort(
                -q[0],
                kind="mergesort",
            )

            ordem = ordem[
                q[0][ordem] > 0
            ][:TOP_K]

            status_temporal = (
                "GRUPOS_TEMPORAIS_POR_DESTINO"
                if modelo[
                    "n_grupos"
                ]
                > 1
                else "POOLING_UM_GRUPO_TEMPORAL"
            )

            if (
                idade
                > modelo[
                    "max_tempo_observado_dias"
                ]
            ):
                status_temporal = (
                    "EXTRAPOLACAO_TEMPORAL"
                )

            for pos, j in enumerate(
                ordem,
                start=1,
            ):
                saidas.append(
                    {
                        **base,
                        "ranking": int(
                            pos
                        ),
                        "proxima_acao": (
                            modelo[
                                "destinos"
                            ][int(j)]
                        ),
                        "prob_proxima_acao": float(
                            q[0][int(j)]
                        ),
                        "status_previsao": (
                            "PREVISAO_SEMIMARKOV_V22"
                        ),
                        "status_temporal": (
                            status_temporal
                        ),
                    }
                )

        resultado = pd.DataFrame(
            saidas
        )

        if len(
            resultado
        ):
            resultado[
                "ranking"
            ] = pd.array(
                resultado[
                    "ranking"
                ],
                dtype="Int64",
            )

        yield resultado


scoring_input = (
    atuais_v22
    .select(
        "cd_bv",
        "data_referencia",
        "ultima_acao",
        "tempo_desde_ultima_acao_seg",
        "status_input",
    )
)

sm22_previsoes = (
    scoring_input
    .mapInPandas(
        prever_lote,
        schema=PREV_SCHEMA,
    )
    .localCheckpoint(
        eager=True
    )
)

sm22_previsoes.createOrReplaceTempView(
    SM22_PREV_VIEW
)

print(
    "SCORING V2.2 - status"
)

(
    sm22_previsoes
    .groupBy(
        "status_previsao",
        "status_temporal",
    )
    .agg(
        F.countDistinct(
            "cd_bv"
        ).alias(
            "n_clientes"
        ),
        F.count("*").alias(
            "n_linhas"
        ),
    )
    .orderBy(
        F.desc(
            "n_clientes"
        )
    )
    .show(
        50,
        truncate=False,
    )
)

print(
    "SCORING V2.2 - Top 1"
)

(
    sm22_previsoes
    .filter(
        F.col("ranking") == 1
    )
    .groupBy(
        "ultima_acao",
        "proxima_acao",
    )
    .agg(
        F.countDistinct(
            "cd_bv"
        ).alias(
            "n_clientes"
        ),
        F.avg(
            "prob_proxima_acao"
        ).alias(
            "prob_media"
        ),
    )
    .orderBy(
        F.desc(
            "n_clientes"
        )
    )
    .show(
        50,
        truncate=False,
    )
)

print(
    "V2.2 Parte 02 concluida."
)

print(
    "Nao ha escrita permanente. "
    "Compare cobertura e diversidade com a V2.1."
)
