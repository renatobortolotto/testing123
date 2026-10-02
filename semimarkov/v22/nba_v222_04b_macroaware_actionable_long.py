# Databricks notebook source
# NBA | V2.2.2 CLEAN - Parte 04B
# Multi-step MACRO-AWARE + proxima acao acionavel de NBA.
#
# Esta etapa NAO retreina o Semi-Markov e NAO reconstrui o estado atual.
# Ela reutiliza o scoring FULL da Parte 04 e corrige a propagacao:
#
# RAW:
#   app_login:::topo -> app_login:::navegacao -> app_login:::sucesso
#
# MACRO-AWARE:
#   app_login
#
# Portanto, topo/navegacao/sucesso do mesmo funil nao consomem varios
# "passos de negocio".
#
# Saidas:
# - RAW: mantido da Parte 04;
# - NEXT MACRO: primeira macroacao DIFERENTE da macro atual;
# - ACTIONABLE: primeira acao configurada como acionavel em ate N passos MACRO.
#
# O actionable_score e first-passage em numero de transicoes MACRO.
# Nao e probabilidade dentro de uma janela de minutos/dias.

from collections import defaultdict
from collections.abc import Iterator
import json

import numpy as np
import pandas as pd
from scipy import special
from pyspark import StorageLevel
from pyspark.sql import functions as F


# ============================================================
# 1. Configuracao
# ============================================================

CFG = {
    "tabela_scoring_parte04": (
        "ctg_dsti.renato_nba.nba_semimarkov_v222_full_multistep_hml"
    ),
    "tabela_modelos": (
        "ctg_dsti.renato_nba.nba_sm_v22_modelos_clean_hml"
    ),
    "tabela_config": (
        "ctg_dsti.renato_nba.nba_config_estados_v222_clean"
    ),
    "tabela_saida": (
        "ctg_dsti.renato_nba.nba_semimarkov_v222_macroaware_long_hml"
    ),
    "max_passos_macro": 5,
    "top_k": 5,
    "gravar": True,
}

# Lista EXPLORATORIA. Ela define somente a camada "acao acionavel".
# O modelo continua aprendendo e propagando por TODAS as macroacoes.
#
# Ajuste esta whitelist com negocio quando necessario.
ACOES_ACIONAVEIS_NBA = {
    "pix_transferencia",
    "pix_pagamento",
    "pix_cadastro_chave",
    "pix_trazer_chave",
    "pagamentos_boleto",
    "gerar_boleto_financeira",
    "cobrancas_gerar_pix",
    "cobrancas_gerar_boleto",
    "cartoes_aumentar_limite",
    "cartoes_ativar_credito",
    "investimentos_aplicacao",
    "seguros_auto_avulso_cotacao",
    "tag_veicular_solicitar",
    "tag_veicular_ativar",
    "rodas_assinar_contrato",
}


# ============================================================
# 2. Validacoes
# ============================================================

for tabela in (
    CFG["tabela_scoring_parte04"],
    CFG["tabela_modelos"],
    CFG["tabela_config"],
):
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela nao encontrada: {tabela}")


# ============================================================
# 3. Carregar scoring, modelos e configuracao
# ============================================================

base_full = (
    spark.table(CFG["tabela_scoring_parte04"])
    .persist(StorageLevel.MEMORY_AND_DISK)
)

config_df = (
    spark.table(CFG["tabela_config"])
    .select(
        "estado",
        "tipo_estado",
        "acao_macro",
        "funil",
        "etapa",
        "detalhe",
        "terminal_real",
    )
    .dropDuplicates(["estado"])
)

modelos_df = (
    spark.table(CFG["tabela_modelos"])
    .filter(F.col("status_modelo") == "AJUSTADO")
    .select("origem", "modelo_json")
    .dropDuplicates(["origem"])
)

MODELOS = {
    row["origem"]: json.loads(row["modelo_json"])
    for row in modelos_df.collect()
}

CONFIG = {
    row["estado"]: {
        "tipo_estado": row["tipo_estado"],
        "acao_macro": row["acao_macro"],
        "funil": row["funil"],
        "etapa": row["etapa"],
        "detalhe": row["detalhe"],
        "terminal_real": bool(row["terminal_real"]),
    }
    for row in config_df.collect()
}

if not MODELOS:
    raise RuntimeError("Nenhum modelo clean ajustado encontrado.")


def macro_estado(estado):
    cfg = CONFIG.get(estado)

    if cfg and cfg["acao_macro"]:
        return cfg["acao_macro"]

    return estado


MACROS_DISPONIVEIS = {
    macro_estado(estado)
    for estado in CONFIG
}

ACOES_INVALIDAS = sorted(
    ACOES_ACIONAVEIS_NBA - MACROS_DISPONIVEIS
)

if ACOES_INVALIDAS:
    print(
        "ATENCAO - acoes acionaveis nao encontradas na config:",
        ACOES_INVALIDAS,
    )

ACOES_ACIONAVEIS = sorted(
    ACOES_ACIONAVEIS_NBA & MACROS_DISPONIVEIS
)

if not ACOES_ACIONAVEIS:
    raise RuntimeError("Nenhuma acao acionavel valida foi configurada.")

print("ACOES ACIONAVEIS UTILIZADAS:", len(ACOES_ACIONAVEIS))
print(ACOES_ACIONAVEIS)


# ============================================================
# 4. Kernel RAW em idade zero
# ============================================================

ESTADOS = sorted(
    set(CONFIG)
    | set(MODELOS)
    | {
        destino
        for modelo in MODELOS.values()
        for destino in modelo["destinos"]
    }
)

ESTADO_IDX = {
    estado: i
    for i, estado in enumerate(ESTADOS)
}

N_ESTADOS = len(ESTADOS)

MACRO_POR_ESTADO = np.array(
    [macro_estado(estado) for estado in ESTADOS],
    dtype=object,
)

TERMINAL_REAL = np.array(
    [
        bool(
            CONFIG.get(estado, {}).get(
                "terminal_real",
                False,
            )
        )
        for estado in ESTADOS
    ],
    dtype=bool,
)

P0 = np.zeros(
    (N_ESTADOS, N_ESTADOS),
    dtype=np.float64,
)

for origem, modelo in MODELOS.items():
    i = ESTADO_IDX.get(origem)

    if i is None or TERMINAL_REAL[i]:
        continue

    probs = np.asarray(
        modelo["p_destino"],
        dtype=np.float64,
    )

    for destino, prob in zip(
        modelo["destinos"],
        probs,
    ):
        j = ESTADO_IDX.get(destino)

        if j is not None:
            P0[i, j] += float(prob)

row_sum = P0.sum(axis=1)

for i in np.flatnonzero(row_sum > 0):
    P0[i] /= row_sum[i]

TEM_MODELO = row_sum > 0


# ============================================================
# 5. Colapsar etapas consecutivas da mesma macroacao
# ============================================================

def construir_jump_macro():
    """
    Para cada estado RAW i, calcula a distribuicao do primeiro estado
    alcancado cuja macroacao e diferente de macro(i).

    Assim:
      login:::topo -> login:::navegacao -> login:::sucesso -> PIX:::topo

    vira:
      login -> PIX

    sem gastar tres passos com o fluxo interno de login.
    """
    jump = np.zeros(
        (N_ESTADOS, N_ESTADOS),
        dtype=np.float64,
    )

    perda = np.zeros(
        N_ESTADOS,
        dtype=np.float64,
    )

    terminal = np.zeros(
        N_ESTADOS,
        dtype=np.float64,
    )

    macros = sorted(
        set(MACRO_POR_ESTADO),
        key=str,
    )

    for macro in macros:
        idx_macro = np.flatnonzero(
            MACRO_POR_ESTADO == macro
        )

        if not len(idx_macro):
            continue

        idx_terminal_macro = idx_macro[
            TERMINAL_REAL[idx_macro]
        ]

        idx_transiente = idx_macro[
            ~TERMINAL_REAL[idx_macro]
        ]

        if len(idx_terminal_macro):
            terminal[
                idx_terminal_macro
            ] = 1.0

        if not len(idx_transiente):
            continue

        idx_fora = np.flatnonzero(
            MACRO_POR_ESTADO != macro
        )

        pos_transiente = {
            estado_idx: pos
            for pos, estado_idx
            in enumerate(idx_transiente)
        }

        q = P0[
            np.ix_(
                idx_transiente,
                idx_transiente,
            )
        ]

        r_fora = P0[
            np.ix_(
                idx_transiente,
                idx_fora,
            )
        ]

        if len(idx_terminal_macro):
            r_terminal = P0[
                np.ix_(
                    idx_transiente,
                    idx_terminal_macro,
                )
            ]
        else:
            r_terminal = np.zeros(
                (
                    len(idx_transiente),
                    0,
                ),
                dtype=np.float64,
            )

        identidade = np.eye(
            len(idx_transiente),
            dtype=np.float64,
        )

        sistema = identidade - q

        try:
            fundamental = np.linalg.solve(
                sistema,
                identidade,
            )
        except np.linalg.LinAlgError:
            fundamental = np.linalg.pinv(
                sistema
            )

        saida_fora = (
            fundamental @ r_fora
        )

        if r_terminal.shape[1]:
            saida_terminal = (
                fundamental
                @ r_terminal
            ).sum(axis=1)
        else:
            saida_terminal = np.zeros(
                len(idx_transiente),
                dtype=np.float64,
            )

        saida_fora = np.clip(
            saida_fora,
            0.0,
            1.0,
        )

        saida_terminal = np.clip(
            saida_terminal,
            0.0,
            1.0,
        )

        for local_i, global_i in enumerate(
            idx_transiente
        ):
            jump[
                global_i,
                idx_fora,
            ] = saida_fora[
                local_i,
                :,
            ]

            terminal[
                global_i
            ] = float(
                saida_terminal[
                    local_i
                ]
            )

            contabilizado = (
                jump[
                    global_i,
                    :
                ].sum()
                + terminal[
                    global_i
                ]
            )

            perda[
                global_i
            ] = max(
                0.0,
                1.0 - contabilizado,
            )

    fechamento = (
        jump.sum(axis=1)
        + perda
        + terminal
    )

    if not np.allclose(
        fechamento,
        1.0,
        atol=1e-7,
    ):
        erro = float(
            np.max(
                np.abs(
                    fechamento - 1.0
                )
            )
        )

        raise RuntimeError(
            "Jump macro nao fechou massa. "
            f"Erro maximo={erro}"
        )

    return jump, perda, terminal


JUMP_MACRO, JUMP_PERDA, JUMP_TERMINAL = (
    construir_jump_macro()
)

print("JUMP MACRO construido.")
print(
    "Media de massa perdida por estado:",
    float(JUMP_PERDA.mean()),
)


# ============================================================
# 6. Semi-Markov condicionado a idade atual
# ============================================================

def prever_q(modelo, idades_dias):
    idades = np.asarray(
        idades_dias,
        dtype=float,
    ).reshape(-1)

    mus = np.asarray(
        modelo["mu_grupo"],
        dtype=float,
    )

    sigmas = np.asarray(
        modelo["sigma_grupo"],
        dtype=float,
    )

    log_s = np.zeros(
        (len(idades), len(mus)),
        dtype=np.float64,
    )

    positivo = idades > 0

    if np.any(positivo):
        z = (
            np.log(
                idades[positivo]
            )[:, None]
            - mus[None, :]
        ) / sigmas[None, :]

        log_s[
            positivo,
            :
        ] = special.log_ndtr(-z)

    log_pi = np.log(
        np.asarray(
            modelo["pi_grupo"],
            dtype=np.float64,
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
        modelo["r_destino_no_grupo"],
        dtype=np.float64,
    )

    q = (
        peso_grupo[:, grupo]
        * r[None, :]
    )

    if not np.allclose(
        q.sum(axis=1),
        1.0,
        atol=1e-9,
    ):
        raise ValueError(
            "MASSA_DESTINO_NAO_FECHA"
        )

    return q


# ============================================================
# 7. Primeira macroacao diferente da atual
# ============================================================

def primeira_macro_distribuicao(
    modelo,
    q,
    macro_atual,
):
    """
    q e a distribuicao RAW da proxima transicao, condicionada a idade atual.

    Se q cair em outro estado da mesma macroacao atual, atravessamos
    internamente o funil ate encontrar a primeira macroacao diferente.
    """
    dist = np.zeros(
        N_ESTADOS,
        dtype=np.float64,
    )

    perda = 0.0
    terminal = 0.0

    for destino, prob in zip(
        modelo["destinos"],
        q,
    ):
        j = ESTADO_IDX.get(destino)
        p = float(prob)

        if j is None:
            perda += p
            continue

        macro_destino = (
            MACRO_POR_ESTADO[j]
        )

        if macro_destino != macro_atual:
            dist[j] += p
            continue

        dist += (
            p
            * JUMP_MACRO[
                j,
                :
            ]
        )

        perda += (
            p
            * JUMP_PERDA[j]
        )

        terminal += (
            p
            * JUMP_TERMINAL[j]
        )

    fechamento = (
        dist.sum()
        + perda
        + terminal
    )

    if abs(
        fechamento - 1.0
    ) > 1e-7:
        raise RuntimeError(
            "Primeira macro nao fechou massa: "
            f"{fechamento}"
        )

    return dist, perda, terminal


# ============================================================
# 8. First-passage das acoes acionaveis em passos MACRO
# ============================================================

ACAO_IDX = {
    acao: i
    for i, acao
    in enumerate(ACOES_ACIONAVEIS)
}

N_ACOES = len(ACOES_ACIONAVEIS)

ALVO_POR_ESTADO = np.full(
    N_ESTADOS,
    -1,
    dtype=np.int32,
)

for i, macro in enumerate(
    MACRO_POR_ESTADO
):
    if macro in ACAO_IDX:
        ALVO_POR_ESTADO[i] = (
            ACAO_IDX[macro]
        )


def first_passage_macro(
    dist_inicial,
    perda_inicial,
    terminal_inicial,
):
    passos = CFG[
        "max_passos_macro"
    ]

    dist = dist_inicial.copy()

    contrib = np.zeros(
        (
            N_ACOES,
            passos,
        ),
        dtype=np.float64,
    )

    perda = float(
        perda_inicial
    )

    terminal = float(
        terminal_inicial
    )

    for passo in range(passos):
        # Uma acao acionavel e absorvida somente na primeira chegada.
        for k in range(N_ACOES):
            mask = (
                ALVO_POR_ESTADO
                == k
            )

            if np.any(mask):
                massa = float(
                    dist[
                        mask
                    ].sum()
                )

                contrib[
                    k,
                    passo,
                ] = massa

                dist[
                    mask
                ] = 0.0

        # Terminais reais fora dos alvos.
        mask_terminal = (
            TERMINAL_REAL
            & (
                ALVO_POR_ESTADO
                < 0
            )
        )

        if np.any(mask_terminal):
            terminal += float(
                dist[
                    mask_terminal
                ].sum()
            )

            dist[
                mask_terminal
            ] = 0.0

        if passo < passos - 1:
            perda += float(
                np.dot(
                    dist,
                    JUMP_PERDA,
                )
            )

            terminal += float(
                np.dot(
                    dist,
                    JUMP_TERMINAL,
                )
            )

            dist = (
                dist
                @ JUMP_MACRO
            )

    sem_acao = float(
        dist.sum()
    )

    scores = contrib.sum(
        axis=1
    )

    fechamento = (
        float(
            scores.sum()
        )
        + perda
        + terminal
        + sem_acao
    )

    if abs(
        fechamento - 1.0
    ) > 1e-7:
        raise RuntimeError(
            "First-passage macro nao fechou massa: "
            f"{fechamento}"
        )

    return {
        "scores": scores,
        "contrib": contrib,
        "perda": perda,
        "terminal": terminal,
        "sem_acao": sem_acao,
    }


# ============================================================
# 9. Broadcasts
# ============================================================

BC_MODELOS = spark.sparkContext.broadcast(
    MODELOS
)

BC_ESTADO_IDX = spark.sparkContext.broadcast(
    ESTADO_IDX
)

BC_MACRO_POR_ESTADO = (
    spark.sparkContext.broadcast(
        MACRO_POR_ESTADO
    )
)

BC_JUMP_MACRO = (
    spark.sparkContext.broadcast(
        JUMP_MACRO
    )
)

BC_JUMP_PERDA = (
    spark.sparkContext.broadcast(
        JUMP_PERDA
    )
)

BC_JUMP_TERMINAL = (
    spark.sparkContext.broadcast(
        JUMP_TERMINAL
    )
)

BC_TERMINAL_REAL = (
    spark.sparkContext.broadcast(
        TERMINAL_REAL
    )
)

BC_ALVO_POR_ESTADO = (
    spark.sparkContext.broadcast(
        ALVO_POR_ESTADO
    )
)

BC_ACOES = spark.sparkContext.broadcast(
    ACOES_ACIONAVEIS
)


# ============================================================
# 10. Schema das novas colunas
# ============================================================

campos = [
    "cd_bv string",
]

for k in range(
    1,
    CFG["top_k"] + 1,
):
    campos += [
        f"next_macro_acao_{k} string",
        f"next_macro_score_{k} double",
        f"next_macro_share_{k} double",
    ]

for k in range(
    1,
    CFG["top_k"] + 1,
):
    campos += [
        f"actionable_acao_{k} string",
        f"actionable_score_{k} double",
        f"actionable_passo_macro_maior_contribuicao_{k} long",
    ]

campos += [
    "massa_perdida_macro double",
    "massa_terminal_macro double",
    "massa_sem_acao_acionavel_apos_5_passos_macro double",
]

SCHEMA = ", ".join(
    campos
)


# ============================================================
# 11. Scoring MACRO-AWARE
# ============================================================

def scoring_macro_lote(
    iterator: Iterator[
        pd.DataFrame
    ],
):
    modelos = BC_MODELOS.value
    estado_idx = (
        BC_ESTADO_IDX.value
    )
    macro_por_estado = (
        BC_MACRO_POR_ESTADO.value
    )
    jump_macro = (
        BC_JUMP_MACRO.value
    )
    jump_perda = (
        BC_JUMP_PERDA.value
    )
    jump_terminal = (
        BC_JUMP_TERMINAL.value
    )
    terminal_real = (
        BC_TERMINAL_REAL.value
    )
    alvo_por_estado = (
        BC_ALVO_POR_ESTADO.value
    )
    acoes = BC_ACOES.value

    def primeira_macro_local(
        modelo,
        q,
        macro_atual,
    ):
        dist = np.zeros(
            len(macro_por_estado),
            dtype=np.float64,
        )

        perda = 0.0
        terminal = 0.0

        for destino, prob in zip(
            modelo["destinos"],
            q,
        ):
            j = estado_idx.get(
                destino
            )

            p = float(prob)

            if j is None:
                perda += p
                continue

            if (
                macro_por_estado[j]
                != macro_atual
            ):
                dist[j] += p

            else:
                dist += (
                    p
                    * jump_macro[
                        j,
                        :
                    ]
                )

                perda += (
                    p
                    * jump_perda[j]
                )

                terminal += (
                    p
                    * jump_terminal[j]
                )

        return (
            dist,
            perda,
            terminal,
        )

    for pdf in iterator:
        saidas = []

        for row in pdf.itertuples(
            index=False
        ):
            out = {
                "cd_bv": str(
                    row.cd_bv
                ),
                "massa_perdida_macro": None,
                "massa_terminal_macro": None,
                (
                    "massa_sem_acao_acionavel_"
                    "apos_5_passos_macro"
                ): None,
            }

            for k in range(
                1,
                CFG["top_k"] + 1,
            ):
                out[
                    f"next_macro_acao_{k}"
                ] = None

                out[
                    f"next_macro_score_{k}"
                ] = None

                out[
                    f"next_macro_share_{k}"
                ] = None

                out[
                    f"actionable_acao_{k}"
                ] = None

                out[
                    f"actionable_score_{k}"
                ] = None

                out[
                    (
                        "actionable_passo_macro_"
                        f"maior_contribuicao_{k}"
                    )
                ] = None

            if (
                row.status_previsao
                != "PREVISAO_V222_CLEAN"
            ):
                saidas.append(out)
                continue

            origem = str(
                row.ultima_acao
            )

            modelo = modelos.get(
                origem
            )

            if modelo is None:
                saidas.append(out)
                continue

            idade_dias = (
                float(
                    row.tempo_desde_ultima_acao_seg
                )
                / 86400.0
            )

            try:
                q = prever_q(
                    modelo,
                    np.array(
                        [idade_dias],
                        dtype=float,
                    ),
                )[0]
            except ValueError:
                saidas.append(out)
                continue

            macro_atual = str(
                row.macro_atual
            )

            (
                dist_macro,
                perda_inicial,
                terminal_inicial,
            ) = primeira_macro_local(
                modelo,
                q,
                macro_atual,
            )

            # --------------------------
            # NEXT MACRO
            # --------------------------
            macro_scores = (
                defaultdict(float)
            )

            for j in np.flatnonzero(
                dist_macro > 0
            ):
                macro_scores[
                    macro_por_estado[j]
                ] += float(
                    dist_macro[j]
                )

            total_macro = float(
                sum(
                    macro_scores.values()
                )
            )

            ranking_macro = sorted(
                macro_scores.items(),
                key=lambda item: (
                    -item[1],
                    str(item[0]),
                ),
            )[
                :CFG["top_k"]
            ]

            for rank, (
                macro,
                score,
            ) in enumerate(
                ranking_macro,
                start=1,
            ):
                out[
                    f"next_macro_acao_{rank}"
                ] = str(macro)

                out[
                    f"next_macro_score_{rank}"
                ] = float(score)

                out[
                    f"next_macro_share_{rank}"
                ] = (
                    float(
                        score
                        / total_macro
                    )
                    if total_macro > 0
                    else None
                )

            # --------------------------
            # ACTIONABLE first-passage
            # --------------------------
            dist = dist_macro.copy()

            contrib = np.zeros(
                (
                    len(acoes),
                    CFG[
                        "max_passos_macro"
                    ],
                ),
                dtype=np.float64,
            )

            perda = float(
                perda_inicial
            )

            terminal = float(
                terminal_inicial
            )

            for passo in range(
                CFG[
                    "max_passos_macro"
                ]
            ):
                for k in range(
                    len(acoes)
                ):
                    mask = (
                        alvo_por_estado
                        == k
                    )

                    if np.any(mask):
                        contrib[
                            k,
                            passo,
                        ] = float(
                            dist[
                                mask
                            ].sum()
                        )

                        dist[
                            mask
                        ] = 0.0

                mask_terminal = (
                    terminal_real
                    & (
                        alvo_por_estado
                        < 0
                    )
                )

                if np.any(
                    mask_terminal
                ):
                    terminal += float(
                        dist[
                            mask_terminal
                        ].sum()
                    )

                    dist[
                        mask_terminal
                    ] = 0.0

                if (
                    passo
                    < CFG[
                        "max_passos_macro"
                    ]
                    - 1
                ):
                    perda += float(
                        np.dot(
                            dist,
                            jump_perda,
                        )
                    )

                    terminal += float(
                        np.dot(
                            dist,
                            jump_terminal,
                        )
                    )

                    dist = (
                        dist
                        @ jump_macro
                    )

            scores = contrib.sum(
                axis=1
            )

            sem_acao = float(
                dist.sum()
            )

            out[
                "massa_perdida_macro"
            ] = perda

            out[
                "massa_terminal_macro"
            ] = terminal

            out[
                (
                    "massa_sem_acao_acionavel_"
                    "apos_5_passos_macro"
                )
            ] = sem_acao

            ranking_acao = np.argsort(
                -scores,
                kind="mergesort",
            )

            ranking_acao = [
                int(k)
                for k in ranking_acao
                if scores[int(k)] > 0
            ][
                :CFG["top_k"]
            ]

            for rank, k in enumerate(
                ranking_acao,
                start=1,
            ):
                out[
                    f"actionable_acao_{rank}"
                ] = acoes[k]

                out[
                    f"actionable_score_{rank}"
                ] = float(
                    scores[k]
                )

                out[
                    (
                        "actionable_passo_macro_"
                        f"maior_contribuicao_{rank}"
                    )
                ] = int(
                    np.argmax(
                        contrib[k]
                    )
                    + 1
                )

            fechamento = (
                float(
                    scores.sum()
                )
                + perda
                + terminal
                + sem_acao
            )

            if abs(
                fechamento - 1.0
            ) > 1e-7:
                raise RuntimeError(
                    "Massa macro nao fecha "
                    f"para {row.cd_bv}: "
                    f"{fechamento}"
                )

            saidas.append(out)

        resultado = pd.DataFrame(
            saidas
        )

        for k in range(
            1,
            CFG["top_k"] + 1,
        ):
            col = (
                "actionable_passo_macro_"
                f"maior_contribuicao_{k}"
            )

            resultado[col] = pd.array(
                resultado[col],
                dtype="Int64",
            )

        yield resultado


entrada = (
    base_full
    .select(
        "cd_bv",
        "ultima_acao",
        "macro_atual",
        "tempo_desde_ultima_acao_seg",
        "status_previsao",
    )
)

macro_scores = (
    entrada
    .mapInPandas(
        scoring_macro_lote,
        schema=SCHEMA,
    )
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)


# ============================================================
# 12. Montar output final
# ============================================================
# ============================================================
# 12. Montar output LONG para negocio
# ============================================================
#
# Estrutura persistida:
# - ate 5 linhas por cd_bv;
# - 1 linha por acao acionavel/ranking;
# - RAW Top 5 e NEXT MACRO Top 5 ficam em JSON para contexto/auditoria;
# - clientes sem acao acionavel continuam aparecendo com ranking NULL.
#
# Isso evita dezenas de colunas do tipo *_1, *_2, ... na tabela final.

colunas_contexto = [
    "cd_bv",
    "dt_snapshot_publico",
    "ts_corte_eventos",
    "ts_execucao",
    "ultima_acao",
    "macro_atual",
    "etapa_atual",
    "tipo_estado_atual",
    "tempo_desde_ultima_acao_seg",
    "status_previsao",
    "status_temporal",
]

colunas_raw = []

for k in range(
    1,
    CFG["top_k"] + 1,
):
    colunas_raw += [
        f"raw_estado_{k}",
        f"raw_prob_{k}",
    ]

resultado_wide = (
    base_full
    .select(
        *colunas_contexto,
        *colunas_raw,
    )
    .join(
        macro_scores,
        "cd_bv",
        "left",
    )
)

raw_array = F.array(
    *[
        F.struct(
            F.lit(k).cast("int").alias("ranking"),
            F.col(
                f"raw_estado_{k}"
            ).alias("estado"),
            F.col(
                f"raw_prob_{k}"
            ).alias("probabilidade"),
        )
        for k in range(
            1,
            CFG["top_k"] + 1,
        )
    ]
)

macro_array = F.array(
    *[
        F.struct(
            F.lit(k).cast("int").alias("ranking"),
            F.col(
                f"next_macro_acao_{k}"
            ).alias("acao"),
            F.col(
                f"next_macro_score_{k}"
            ).alias("score"),
            F.col(
                f"next_macro_share_{k}"
            ).alias("share"),
        )
        for k in range(
            1,
            CFG["top_k"] + 1,
        )
    ]
)

actionable_array = F.array(
    *[
        F.struct(
            F.lit(k).cast("int").alias("ranking"),
            F.col(
                f"actionable_acao_{k}"
            ).alias("acao"),
            F.col(
                f"actionable_score_{k}"
            ).alias("score"),
            F.col(
                (
                    "actionable_passo_macro_"
                    f"maior_contribuicao_{k}"
                )
            )
            .cast("long")
            .alias(
                "passo_macro_maior_contribuicao"
            ),
        )
        for k in range(
            1,
            CFG["top_k"] + 1,
        )
    ]
)

actionable_vazio = F.array(
    F.struct(
        F.lit(None)
        .cast("int")
        .alias("ranking"),
        F.lit(None)
        .cast("string")
        .alias("acao"),
        F.lit(None)
        .cast("double")
        .alias("score"),
        F.lit(None)
        .cast("long")
        .alias(
            "passo_macro_maior_contribuicao"
        ),
    )
)

resultado_pre_long = (
    resultado_wide
    .withColumn(
        "_raw_array",
        raw_array,
    )
    .withColumn(
        "_macro_array",
        macro_array,
    )
    .withColumn(
        "_actionable_array",
        actionable_array,
    )
    .withColumn(
        "_raw_validos",
        F.expr(
            "filter("
            "_raw_array, "
            "x -> x.estado is not null"
            ")"
        ),
    )
    .withColumn(
        "_macro_validos",
        F.expr(
            "filter("
            "_macro_array, "
            "x -> x.acao is not null"
            ")"
        ),
    )
    .withColumn(
        "_actionable_validos",
        F.expr(
            "filter("
            "_actionable_array, "
            "x -> x.acao is not null"
            ")"
        ),
    )
    .withColumn(
        "raw_top5_json",
        F.to_json(
            F.col("_raw_validos")
        ),
    )
    .withColumn(
        "next_macro_top5_json",
        F.to_json(
            F.col("_macro_validos")
        ),
    )
    .withColumn(
        "_actionable_para_explodir",
        F.when(
            F.size(
                "_actionable_validos"
            ) > 0,
            F.col(
                "_actionable_validos"
            ),
        ).otherwise(
            actionable_vazio
        ),
    )
)

resultado_negocio = (
    resultado_pre_long
    .withColumn(
        "_acao_rank",
        F.explode(
            "_actionable_para_explodir"
        ),
    )
    .select(
        *[
            F.col(c)
            for c in colunas_contexto
        ],
        "raw_top5_json",
        "next_macro_top5_json",
        F.col(
            "_acao_rank.ranking"
        ).alias(
            "ranking_actionable"
        ),
        F.col(
            "_acao_rank.acao"
        ).alias(
            "actionable_acao"
        ),
        F.col(
            "_acao_rank.score"
        ).alias(
            "actionable_score"
        ),
        F.col(
            (
                "_acao_rank."
                "passo_macro_maior_contribuicao"
            )
        ).alias(
            "actionable_passo_macro_"
            "maior_contribuicao"
        ),
        "massa_perdida_macro",
        "massa_terminal_macro",
        (
            "massa_sem_acao_acionavel_"
            "apos_5_passos_macro"
        ),
    )
    .withColumn(
        "versao_output",
        F.lit(
            "v2.2.2_clean_macroaware_long"
        ),
    )
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)


# ============================================================
# 13. Diagnosticos
# ============================================================

print("ESTRUTURA LONG")

(
    resultado_negocio
    .agg(
        F.count("*").alias(
            "n_linhas"
        ),
        F.countDistinct(
            "cd_bv"
        ).alias(
            "n_clientes"
        ),
        F.avg(
            F.when(
                F.col(
                    "ranking_actionable"
                ).isNotNull(),
                F.lit(1.0),
            ).otherwise(
                F.lit(0.0)
            )
        ).alias(
            "share_linhas_com_acao"
        ),
    )
    .show(
        truncate=False
    )
)

print("ACTIONABLE TOP 1")

(
    resultado_negocio
    .filter(
        F.col(
            "ranking_actionable"
        ) == 1
    )
    .groupBy(
        "actionable_acao"
    )
    .agg(
        F.countDistinct(
            "cd_bv"
        ).alias(
            "n_clientes"
        ),
        F.avg(
            "actionable_score"
        ).alias(
            "score_medio"
        ),
        F.avg(
            (
                "actionable_passo_macro_"
                "maior_contribuicao"
            )
        ).alias(
            "passo_macro_medio"
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

print("DISTRIBUICAO DE RANKS")

(
    resultado_negocio
    .groupBy(
        "ranking_actionable"
    )
    .agg(
        F.countDistinct(
            "cd_bv"
        ).alias(
            "n_clientes"
        )
    )
    .orderBy(
        "ranking_actionable"
    )
    .show(
        truncate=False
    )
)

print("MASSA MACRO-AWARE")

(
    resultado_negocio
    .filter(
        F.col(
            "status_previsao"
        )
        == "PREVISAO_V222_CLEAN"
    )
    .groupBy(
        "cd_bv"
    )
    .agg(
        F.first(
            "massa_perdida_macro"
        ).alias(
            "massa_perdida_macro"
        ),
        F.first(
            "massa_terminal_macro"
        ).alias(
            "massa_terminal_macro"
        ),
        F.first(
            (
                "massa_sem_acao_acionavel_"
                "apos_5_passos_macro"
            )
        ).alias(
            "massa_sem_acao"
        ),
    )
    .agg(
        F.avg(
            "massa_perdida_macro"
        ).alias(
            "media_perdida_macro"
        ),
        F.avg(
            "massa_terminal_macro"
        ).alias(
            "media_terminal_macro"
        ),
        F.avg(
            "massa_sem_acao"
        ).alias(
            "media_sem_acao_acionavel"
        ),
    )
    .show(
        truncate=False
    )
)


# ============================================================
# 14. Persistencia
# ============================================================

if CFG["gravar"]:
    (
        resultado_negocio
        .write
        .format("delta")
        .mode("overwrite")
        .option(
            "overwriteSchema",
            "true",
        )
        .saveAsTable(
            CFG["tabela_saida"]
        )
    )

    print(
        "Output LONG salvo em:",
        CFG["tabela_saida"],
    )


# ============================================================
# 15. Exemplos de consumo
# ============================================================

print(
    "\nValidar uma pessoa conhecida:\n"
    "display(\n"
    "    resultado_negocio\n"
    "    .filter(F.col('cd_bv') == 'ID_AQUI')\n"
    "    .orderBy('ranking_actionable')\n"
    ")\n"
)

print(
    "Top 1 de negocio:\n"
    "resultado_negocio.filter("
    "F.col('ranking_actionable') == 1"
    ")"
)

print(
    "Cada cd_bv possui ate 5 linhas de acao acionavel. "
    "RAW Top 5 e NEXT MACRO Top 5 ficam armazenados "
    "como JSON para contexto/auditoria."
)
