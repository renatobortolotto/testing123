# Databricks notebook source
# NBA | V2.2.2 CLEAN - Parte 04
# Scoring FULL: RAW + MACRO + ACAO UTIL MULTI-STEP.
#
# Usa:
# - publico completo da customers_query;
# - eventos comportamentais reais de base_jornadas_onboarding;
# - modelo Semi-Markov V2.2.2 clean persistido;
# - config de estados V2.2.2 clean.
#
# Nao retreina o modelo.
#
# Definicoes:
# RAW   = proximo estado tecnico.
# MACRO = proxima acao/funil no passo imediatamente seguinte.
# UTIL  = primeira acao macro nao-tecnica atingida em ate N transicoes.
#
# O util_score e first-passage por numero de transicoes.
# Nao e probabilidade de ocorrer dentro de X minutos/dias.

from collections import defaultdict
from collections.abc import Iterator
import json

import numpy as np
import pandas as pd
from scipy import special
from pyspark import StorageLevel
from pyspark.sql import Window
from pyspark.sql import functions as F


# ============================================================
# 1. Configuracao
# ============================================================

CFG = {
    "fonte_publico": "customers_query",
    "fonte_eventos": "ctg_dsti.renato_nba.base_jornadas_onboarding",
    "tabela_modelos": "ctg_dsti.renato_nba.nba_sm_v222_modelos_clean_hml",
    "tabela_config": "ctg_dsti.renato_nba.nba_config_estados_v222_clean",
    "tabela_saida": "ctg_dsti.renato_nba.nba_semimarkov_v222_full_multistep_hml",
    "debounce_seg": 30.0,
    "max_passos": 5,
    "top_k": 5,
    "gravar": True,
}

# Acoes explicitamente tecnicas. Elas continuam na cadeia e podem aparecer
# em RAW/MACRO, mas o Multi-step atravessa essas acoes em busca da proxima
# acao util.
ACOES_TECNICAS = {
    "app_login",
    "app_habilitacao_device",
    "app_primeiro_acesso",
    "app_reset_senha",
    "app_atualizacao_cadastral",
}


# ============================================================
# 2. Validacoes
# ============================================================

for tabela in (
    CFG["fonte_eventos"],
    CFG["tabela_modelos"],
    CFG["tabela_config"],
):
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela nao encontrada: {tabela}")

if not spark.catalog.tableExists(CFG["fonte_publico"]):
    raise RuntimeError(
        "customers_query nao encontrada. "
        "Crie/atualize a view antes do scoring."
    )


# ============================================================
# 3. Publico completo + metadata do snapshot
# ============================================================

publico_raw = spark.table(CFG["fonte_publico"])

if "dt_snapshot_publico" in publico_raw.columns:
    publico = (
        publico_raw
        .select(
            F.col("cd_bv").cast("string"),
            F.col("dt_snapshot_publico").cast("date"),
        )
        .filter(F.col("cd_bv").isNotNull())
        .dropDuplicates(["cd_bv"])
    )
else:
    publico = (
        publico_raw
        .select(F.col("cd_bv").cast("string"))
        .filter(F.col("cd_bv").isNotNull())
        .distinct()
        .withColumn(
            "dt_snapshot_publico",
            F.lit(None).cast("date"),
        )
    )

publico = publico.persist(StorageLevel.MEMORY_AND_DISK)

print("PUBLICO COMPLETO")
publico.agg(
    F.count("*").alias("n_clientes")
).show(truncate=False)


# ============================================================
# 4. Frescor dos eventos
# ============================================================

fonte_eventos = spark.table(CFG["fonte_eventos"])

campos_obrigatorios = {
    "cd_bv",
    "dm_navegacao",
    "estado",
    "profundidade_max",
}

faltantes = campos_obrigatorios - set(fonte_eventos.columns)

if faltantes:
    raise RuntimeError(
        f"Fonte de eventos sem colunas: {sorted(faltantes)}"
    )

if (
    fonte_eventos.schema["dm_navegacao"]
    .dataType.simpleString()
    != "timestamp"
):
    raise RuntimeError("dm_navegacao deve ser timestamp.")

ts_corte_eventos = (
    fonte_eventos
    .agg(F.max("dm_navegacao").alias("ts"))
    .first()["ts"]
)

if ts_corte_eventos is None:
    raise RuntimeError("Fonte de eventos vazia.")

ts_execucao = spark.sql(
    "SELECT current_timestamp() AS ts"
).first()["ts"]

print("FRESCOR")
print("ts_corte_eventos:", ts_corte_eventos)
print("ts_execucao:", ts_execucao)


# ============================================================
# 5. Resolver simultaneos e debounce para o estado atual
# ============================================================

eventos = (
    fonte_eventos
    .select(
        F.col("cd_bv").cast("string"),
        F.col("dm_navegacao").alias("ts_evento"),
        F.col("estado").cast("string"),
        F.col("profundidade_max").cast("double"),
    )
    .filter(F.col("cd_bv").isNotNull())
    .filter(F.col("ts_evento").isNotNull())
    .filter(F.col("estado").isNotNull())
    .filter(F.length(F.trim("estado")) > 0)
    .filter(~F.lower("estado").contains("sem_acao"))
    .filter(~F.lower("estado").contains("conversao"))
    .join(
        publico.select("cd_bv"),
        "cd_bv",
        "left_semi",
    )
)

por_estado = (
    eventos
    .groupBy("cd_bv", "ts_evento", "estado")
    .agg(
        F.max("profundidade_max").alias("profundidade_max"),
        F.count("*").alias("n_ev"),
    )
    .withColumn(
        "_prof_ordem",
        F.coalesce(
            F.col("profundidade_max"),
            F.lit(-1.0),
        ),
    )
)

w_instante = Window.partitionBy("cd_bv", "ts_evento")

candidatos = (
    por_estado
    .withColumn(
        "_max_prof",
        F.max("_prof_ordem").over(w_instante),
    )
    .withColumn(
        "_eh_max",
        F.col("_prof_ordem") == F.col("_max_prof"),
    )
    .withColumn(
        "_n_max",
        F.sum(F.col("_eh_max").cast("int")).over(w_instante),
    )
)

instantes = (
    candidatos
    .groupBy("cd_bv", "ts_evento")
    .agg(
        F.max("_n_max").alias("n_estados_max_prof"),
        F.sum("n_ev").alias("n_ev_instante"),
        F.first(
            F.when(F.col("_eh_max"), F.col("estado")),
            ignorenulls=True,
        ).alias("_estado_max"),
        F.first(
            F.when(F.col("_eh_max"), F.col("profundidade_max")),
            ignorenulls=True,
        ).alias("profundidade_max"),
    )
    .withColumn(
        "ordem_ambigua",
        F.col("n_estados_max_prof") > 1,
    )
    .withColumn(
        "estado",
        F.when(
            ~F.col("ordem_ambigua"),
            F.col("_estado_max"),
        ),
    )
    .drop("_estado_max")
)

w_cliente = (
    Window.partitionBy("cd_bv")
    .orderBy("ts_evento")
)

com_lag = (
    instantes
    .withColumn(
        "_estado_anterior",
        F.lag("estado").over(w_cliente),
    )
    .withColumn(
        "_ts_anterior",
        F.lag("ts_evento").over(w_cliente),
    )
    .withColumn(
        "_ambiguo_anterior",
        F.lag("ordem_ambigua").over(w_cliente),
    )
    .withColumn(
        "_gap_seg",
        (
            F.unix_micros("ts_evento")
            - F.unix_micros("_ts_anterior")
        ) / F.lit(1_000_000.0),
    )
)

mesma_acao = (
    F.col("estado").isNotNull()
    & F.col("_estado_anterior").isNotNull()
    & ~F.col("ordem_ambigua")
    & ~F.coalesce(F.col("_ambiguo_anterior"), F.lit(False))
    & (F.col("estado") == F.col("_estado_anterior"))
    & (F.col("_gap_seg") <= F.lit(CFG["debounce_seg"]))
)

w_acumulada = w_cliente.rowsBetween(
    Window.unboundedPreceding,
    Window.currentRow,
)

acoes = (
    com_lag
    .withColumn(
        "_abre_acao",
        F.when(mesma_acao, F.lit(0)).otherwise(F.lit(1)),
    )
    .withColumn(
        "_id_acao",
        F.sum("_abre_acao").over(w_acumulada),
    )
    .groupBy("cd_bv", "_id_acao")
    .agg(
        F.max("ts_evento").alias("ts_ultima_evidencia"),
        F.max("estado").alias("estado"),
        F.max(F.col("ordem_ambigua").cast("int"))
        .cast("boolean")
        .alias("ordem_ambigua"),
    )
)

w_ultimo = (
    Window.partitionBy("cd_bv")
    .orderBy(
        F.desc("ts_ultima_evidencia"),
        F.desc("_id_acao"),
    )
)

ultimo_estado = (
    acoes
    .withColumn("_rn", F.row_number().over(w_ultimo))
    .filter(F.col("_rn") == 1)
    .select(
        "cd_bv",
        F.col("estado").alias("ultima_acao"),
        F.col("ts_ultima_evidencia").alias("ts_ultima_acao"),
        "ordem_ambigua",
    )
)


# ============================================================
# 6. Config clean e estado atual
# ============================================================

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

atuais = (
    publico
    .join(ultimo_estado, "cd_bv", "left")
    .withColumn(
        "ts_corte_eventos",
        F.lit(ts_corte_eventos).cast("timestamp"),
    )
    .withColumn(
        "ts_execucao",
        F.lit(ts_execucao).cast("timestamp"),
    )
    .withColumn(
        "tempo_desde_ultima_acao_seg",
        F.when(
            F.col("ts_ultima_acao").isNotNull(),
            (
                F.unix_micros("ts_corte_eventos")
                - F.unix_micros("ts_ultima_acao")
            ) / F.lit(1_000_000.0),
        ),
    )
    .withColumn(
        "status_input",
        F.when(
            F.col("ts_ultima_acao").isNull(),
            F.lit("SEM_HISTORICO"),
        )
        .when(
            F.col("ordem_ambigua"),
            F.lit("ESTADO_ATUAL_AMBIGUO"),
        )
        .otherwise(F.lit("OK")),
    )
    .join(
        config_df.select(
            F.col("estado").alias("_estado_cfg"),
            F.col("acao_macro").alias("macro_atual"),
            F.col("etapa").alias("etapa_atual"),
            F.col("tipo_estado").alias("tipo_estado_atual"),
        ),
        F.col("ultima_acao") == F.col("_estado_cfg"),
        "left",
    )
    .drop("_estado_cfg")
    .persist(StorageLevel.MEMORY_AND_DISK)
)

print("ESTADO ATUAL")
(
    atuais
    .groupBy("status_input")
    .count()
    .orderBy(F.desc("count"))
    .show(truncate=False)
)


# ============================================================
# 7. Modelos clean
# ============================================================

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

if not MODELOS:
    raise RuntimeError("Nenhum modelo clean ajustado encontrado.")

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


def macro_estado(estado):
    cfg = CONFIG.get(estado)
    if cfg and cfg["acao_macro"]:
        return cfg["acao_macro"]
    return estado


# ============================================================
# 8. Definir acoes uteis exploratorias
# ============================================================

# Toda acao de funil e candidata, exceto as explicitamente tecnicas.
# Atendimento continua na cadeia, mas nao vira acao util.
ACOES_UTEIS = sorted(
    {
        row["acao_macro"]
        for row in config_df
        .filter(F.col("funil").isNotNull())
        .select("acao_macro")
        .distinct()
        .collect()
        if row["acao_macro"] not in ACOES_TECNICAS
    }
)

print("ACOES UTEIS EXPLORATORIAS:", len(ACOES_UTEIS))
print(ACOES_UTEIS)


# ============================================================
# 9. Kernel de transicao em entrada no estado
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
N_UTEIS = len(ACOES_UTEIS)
UTIL_IDX = {
    acao: i
    for i, acao in enumerate(ACOES_UTEIS)
}

P0 = np.zeros(
    (N_ESTADOS, N_ESTADOS),
    dtype=np.float64,
)

for origem, modelo in MODELOS.items():
    i = ESTADO_IDX.get(origem)
    if i is None:
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

MACRO_POR_ESTADO = np.array(
    [macro_estado(estado) for estado in ESTADOS],
    dtype=object,
)

TERMINAL_REAL = np.array(
    [
        bool(
            CONFIG.get(
                estado,
                {},
            ).get(
                "terminal_real",
                False,
            )
        )
        for estado in ESTADOS
    ],
    dtype=bool,
)


# ============================================================
# 10. First-passage precomputado
# ============================================================

def precomputar_first_passage(macro_atual):
    """
    Calcula a primeira acao util em ate max_passos.

    A macro atual e suprimida como alvo durante o horizonte para evitar
    interpretar topo -> navegacao -> sucesso do mesmo funil como uma nova
    acao util.
    """
    passos = CFG["max_passos"]

    alvo = np.full(
        N_ESTADOS,
        -1,
        dtype=np.int32,
    )

    for s, macro in enumerate(MACRO_POR_ESTADO):
        if (
            macro in UTIL_IDX
            and macro != macro_atual
        ):
            alvo[s] = UTIL_IDX[macro]

    dist = np.eye(
        N_ESTADOS,
        dtype=np.float64,
    )

    contrib = np.zeros(
        (
            N_ESTADOS,
            N_UTEIS,
            passos,
        ),
        dtype=np.float64,
    )

    perdida = np.zeros(
        N_ESTADOS,
        dtype=np.float64,
    )

    terminal = np.zeros(
        N_ESTADOS,
        dtype=np.float64,
    )

    for passo in range(passos):
        for k in range(N_UTEIS):
            mask = alvo == k

            if np.any(mask):
                contrib[:, k, passo] = (
                    dist[:, mask].sum(axis=1)
                )
                dist[:, mask] = 0.0

        mask_terminal = (
            TERMINAL_REAL
            & (alvo < 0)
        )

        if np.any(mask_terminal):
            terminal += (
                dist[:, mask_terminal].sum(axis=1)
            )
            dist[:, mask_terminal] = 0.0

        mask_sem_modelo = (
            (~TEM_MODELO)
            & (~TERMINAL_REAL)
            & (alvo < 0)
        )

        if np.any(mask_sem_modelo):
            perdida += (
                dist[:, mask_sem_modelo].sum(axis=1)
            )
            dist[:, mask_sem_modelo] = 0.0

        if passo < passos - 1:
            dist = dist @ P0

    sem_util = dist.sum(axis=1)

    fechamento = (
        contrib.sum(axis=(1, 2))
        + perdida
        + terminal
        + sem_util
    )

    if not np.allclose(
        fechamento,
        1.0,
        atol=1e-8,
    ):
        erro = float(
            np.max(
                np.abs(
                    fechamento - 1.0
                )
            )
        )
        raise RuntimeError(
            "First-passage nao fechou massa. "
            f"Erro maximo={erro}"
        )

    return {
        "contrib": contrib,
        "perdida": perdida,
        "terminal": terminal,
        "sem_util": sem_util,
    }


MACROS_ATUAIS = sorted(
    {
        macro_estado(origem)
        for origem in MODELOS
    }
)

FIRST_PASSAGE = {
    macro: precomputar_first_passage(macro)
    for macro in MACROS_ATUAIS
}


# ============================================================
# 11. Semi-Markov condicional a idade atual
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
        dtype=float,
    )

    positivo = idades > 0

    if np.any(positivo):
        z = (
            np.log(
                idades[positivo]
            )[:, None]
            - mus[None, :]
        ) / sigmas[None, :]

        log_s[positivo, :] = (
            special.log_ndtr(-z)
        )

    log_pi = np.log(
        np.asarray(
            modelo["pi_grupo"],
            dtype=float,
        )
    )

    den = special.logsumexp(
        log_pi[None, :]
        + log_s,
        axis=1,
    )

    if not np.isfinite(den).all():
        raise ValueError(
            "IDADE_FORA_SUPORTE"
        )

    peso_grupo = np.exp(
        log_pi[None, :]
        + log_s
        - den[:, None]
    )

    grupo = np.asarray(
        modelo["grupo"],
        dtype=int,
    )

    r = np.asarray(
        modelo["r_destino_no_grupo"],
        dtype=float,
    )

    q = (
        peso_grupo[:, grupo]
        * r[None, :]
    )

    return q


BC_MODELOS = spark.sparkContext.broadcast(MODELOS)
BC_CONFIG = spark.sparkContext.broadcast(CONFIG)
BC_ESTADO_IDX = spark.sparkContext.broadcast(ESTADO_IDX)
BC_FIRST_PASSAGE = spark.sparkContext.broadcast(FIRST_PASSAGE)
BC_ACOES_UTEIS = spark.sparkContext.broadcast(ACOES_UTEIS)


# ============================================================
# 12. Schema de saida
# ============================================================

campos = [
    "cd_bv string",
    "dt_snapshot_publico date",
    "ts_corte_eventos timestamp",
    "ts_execucao timestamp",
    "ultima_acao string",
    "macro_atual string",
    "etapa_atual string",
    "tipo_estado_atual string",
    "tempo_desde_ultima_acao_seg double",
    "status_previsao string",
    "status_temporal string",
]

for k in range(1, CFG["top_k"] + 1):
    campos += [
        f"raw_estado_{k} string",
        f"raw_prob_{k} double",
    ]

for k in range(1, CFG["top_k"] + 1):
    campos += [
        f"macro_acao_{k} string",
        f"macro_prob_{k} double",
    ]

for k in range(1, CFG["top_k"] + 1):
    campos += [
        f"util_acao_{k} string",
        f"util_score_{k} double",
        f"util_passo_maior_contribuicao_{k} long",
    ]

campos += [
    "massa_perdida_sem_modelo double",
    "massa_terminal_real double",
    "massa_sem_acao_util_apos_5_passos double",
]

SCHEMA = ", ".join(campos)


# ============================================================
# 13. Scoring por lote
# ============================================================

def scoring_lote(
    iterator: Iterator[pd.DataFrame],
):
    modelos = BC_MODELOS.value
    config = BC_CONFIG.value
    estado_idx = BC_ESTADO_IDX.value
    first_passage = BC_FIRST_PASSAGE.value
    acoes_uteis = BC_ACOES_UTEIS.value

    def macro_local(estado):
        cfg = config.get(estado)
        if cfg and cfg.get("acao_macro"):
            return cfg["acao_macro"]
        return estado

    for pdf in iterator:
        saidas = []

        for row in pdf.itertuples(index=False):
            out = {
                "cd_bv": str(row.cd_bv),
                "dt_snapshot_publico": (
                    None
                    if pd.isna(row.dt_snapshot_publico)
                    else row.dt_snapshot_publico
                ),
                "ts_corte_eventos": row.ts_corte_eventos,
                "ts_execucao": row.ts_execucao,
                "ultima_acao": (
                    None
                    if pd.isna(row.ultima_acao)
                    else str(row.ultima_acao)
                ),
                "macro_atual": (
                    None
                    if pd.isna(row.macro_atual)
                    else str(row.macro_atual)
                ),
                "etapa_atual": (
                    None
                    if pd.isna(row.etapa_atual)
                    else str(row.etapa_atual)
                ),
                "tipo_estado_atual": (
                    None
                    if pd.isna(row.tipo_estado_atual)
                    else str(row.tipo_estado_atual)
                ),
                "tempo_desde_ultima_acao_seg": (
                    None
                    if pd.isna(row.tempo_desde_ultima_acao_seg)
                    else float(row.tempo_desde_ultima_acao_seg)
                ),
                "status_previsao": str(row.status_input),
                "status_temporal": "NAO_CALCULADO",
                "massa_perdida_sem_modelo": None,
                "massa_terminal_real": None,
                "massa_sem_acao_util_apos_5_passos": None,
            }

            for k in range(1, CFG["top_k"] + 1):
                out[f"raw_estado_{k}"] = None
                out[f"raw_prob_{k}"] = None
                out[f"macro_acao_{k}"] = None
                out[f"macro_prob_{k}"] = None
                out[f"util_acao_{k}"] = None
                out[f"util_score_{k}"] = None
                out[f"util_passo_maior_contribuicao_{k}"] = None

            if row.status_input != "OK":
                saidas.append(out)
                continue

            origem = str(row.ultima_acao)
            modelo = modelos.get(origem)

            if modelo is None:
                out["status_previsao"] = "SEM_MODELO_ORIGEM"
                saidas.append(out)
                continue

            idade_dias = (
                float(row.tempo_desde_ultima_acao_seg)
                / 86400.0
            )

            try:
                q = prever_q(
                    modelo,
                    np.array([idade_dias]),
                )[0]
            except ValueError:
                out["status_previsao"] = "IDADE_FORA_SUPORTE"
                saidas.append(out)
                continue

            out["status_previsao"] = "PREVISAO_V222_CLEAN"

            if idade_dias > modelo["max_tempo_observado_dias"]:
                out["status_temporal"] = "EXTRAPOLACAO_TEMPORAL"
            elif modelo["n_grupos"] > 1:
                out["status_temporal"] = "GRUPOS_TEMPORAIS_POR_DESTINO"
            else:
                out["status_temporal"] = "POOLING_UM_GRUPO_TEMPORAL"

            # RAW
            ordem_raw = np.argsort(
                -q,
                kind="mergesort",
            )[:CFG["top_k"]]

            for rank, j in enumerate(
                ordem_raw,
                start=1,
            ):
                out[f"raw_estado_{rank}"] = (
                    modelo["destinos"][int(j)]
                )
                out[f"raw_prob_{rank}"] = float(
                    q[int(j)]
                )

            # MACRO imediato
            macro_probs = defaultdict(float)

            for destino, prob in zip(
                modelo["destinos"],
                q,
            ):
                macro_probs[
                    macro_local(destino)
                ] += float(prob)

            macro_rank = sorted(
                macro_probs.items(),
                key=lambda item: (
                    -item[1],
                    str(item[0]),
                ),
            )[:CFG["top_k"]]

            for rank, (
                macro,
                prob,
            ) in enumerate(
                macro_rank,
                start=1,
            ):
                out[f"macro_acao_{rank}"] = macro
                out[f"macro_prob_{rank}"] = float(prob)

            # UTIL multi-step
            macro_atual = macro_local(origem)
            fp = first_passage.get(macro_atual)

            if fp is None:
                out["status_previsao"] = (
                    "SEM_FIRST_PASSAGE_MACRO"
                )
                saidas.append(out)
                continue

            idx_dest = np.array(
                [
                    estado_idx.get(
                        destino,
                        -1,
                    )
                    for destino
                    in modelo["destinos"]
                ],
                dtype=int,
            )

            validos = idx_dest >= 0

            q_fp = q[validos]
            idx_fp = idx_dest[validos]

            massa_idx_valida = float(
                q_fp.sum()
            )

            perdida_destino_fora_catalogo = (
                1.0 - massa_idx_valida
            )

            if massa_idx_valida > 0:
                contrib = np.tensordot(
                    q_fp,
                    fp["contrib"][
                        idx_fp,
                        :,
                        :,
                    ],
                    axes=(0, 0),
                )

                perdida = float(
                    np.dot(
                        q_fp,
                        fp["perdida"][idx_fp],
                    )
                )
                terminal = float(
                    np.dot(
                        q_fp,
                        fp["terminal"][idx_fp],
                    )
                )
                sem_util = float(
                    np.dot(
                        q_fp,
                        fp["sem_util"][idx_fp],
                    )
                )
            else:
                contrib = np.zeros(
                    (
                        len(acoes_uteis),
                        CFG["max_passos"],
                    ),
                    dtype=float,
                )
                perdida = 0.0
                terminal = 0.0
                sem_util = 0.0

            perdida += max(
                perdida_destino_fora_catalogo,
                0.0,
            )

            scores = contrib.sum(axis=1)

            out["massa_perdida_sem_modelo"] = perdida
            out["massa_terminal_real"] = terminal
            out["massa_sem_acao_util_apos_5_passos"] = sem_util

            ordem_util = np.argsort(
                -scores,
                kind="mergesort",
            )

            ordem_util = [
                int(k)
                for k in ordem_util
                if scores[int(k)] > 0
            ][:CFG["top_k"]]

            for rank, k in enumerate(
                ordem_util,
                start=1,
            ):
                out[f"util_acao_{rank}"] = (
                    acoes_uteis[k]
                )
                out[f"util_score_{rank}"] = float(
                    scores[k]
                )
                out[
                    f"util_passo_maior_contribuicao_{rank}"
                ] = int(
                    np.argmax(
                        contrib[k]
                    )
                    + 1
                )

            fechamento = (
                float(scores.sum())
                + perdida
                + terminal
                + sem_util
            )

            if abs(
                fechamento - 1.0
            ) > 1e-7:
                raise RuntimeError(
                    "Massa nao fecha para "
                    f"{row.cd_bv}: {fechamento}"
                )

            saidas.append(out)

        resultado = pd.DataFrame(saidas)

        for k in range(
            1,
            CFG["top_k"] + 1,
        ):
            col = (
                "util_passo_maior_contribuicao_"
                f"{k}"
            )
            resultado[col] = pd.array(
                resultado[col],
                dtype="Int64",
            )

        yield resultado


entrada = atuais.select(
    "cd_bv",
    "dt_snapshot_publico",
    "ts_corte_eventos",
    "ts_execucao",
    "ultima_acao",
    "macro_atual",
    "etapa_atual",
    "tipo_estado_atual",
    "tempo_desde_ultima_acao_seg",
    "status_input",
)

resultado_v222 = (
    entrada
    .mapInPandas(
        scoring_lote,
        schema=SCHEMA,
    )
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)


# ============================================================
# 14. Diagnosticos
# ============================================================

print("STATUS FULL")
(
    resultado_v222
    .groupBy(
        "status_previsao",
        "status_temporal",
    )
    .count()
    .orderBy(F.desc("count"))
    .show(50, truncate=False)
)

print("RAW TOP 1")
(
    resultado_v222
    .filter(F.col("raw_estado_1").isNotNull())
    .groupBy("raw_estado_1")
    .agg(
        F.count("*").alias("n_clientes"),
        F.avg("raw_prob_1").alias("prob_media"),
    )
    .orderBy(F.desc("n_clientes"))
    .show(50, truncate=False)
)

print("MACRO TOP 1")
(
    resultado_v222
    .filter(F.col("macro_acao_1").isNotNull())
    .groupBy("macro_acao_1")
    .agg(
        F.count("*").alias("n_clientes"),
        F.avg("macro_prob_1").alias("prob_media"),
    )
    .orderBy(F.desc("n_clientes"))
    .show(50, truncate=False)
)

print("UTIL TOP 1")
(
    resultado_v222
    .filter(F.col("util_acao_1").isNotNull())
    .groupBy("util_acao_1")
    .agg(
        F.count("*").alias("n_clientes"),
        F.avg("util_score_1").alias("score_medio"),
        F.avg(
            "util_passo_maior_contribuicao_1"
        ).alias("passo_medio_maior_contribuicao"),
    )
    .orderBy(F.desc("n_clientes"))
    .show(50, truncate=False)
)

print("MASSA MULTI-STEP")
(
    resultado_v222
    .filter(
        F.col("status_previsao")
        == "PREVISAO_V222_CLEAN"
    )
    .agg(
        F.avg(
            "massa_perdida_sem_modelo"
        ).alias(
            "media_perdida_sem_modelo"
        ),
        F.avg(
            "massa_terminal_real"
        ).alias(
            "media_terminal_real"
        ),
        F.avg(
            "massa_sem_acao_util_apos_5_passos"
        ).alias(
            "media_sem_acao_util"
        ),
    )
    .show(truncate=False)
)


# ============================================================
# 15. Persistencia
# ============================================================

if CFG["gravar"]:
    (
        resultado_v222
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
        "SCORING FULL salvo em:",
        CFG["tabela_saida"],
    )


# ============================================================
# 16. Exemplo de validacao individual
# ============================================================

print(
    "\nPara validar um cliente:\n"
    "display(\n"
    "    resultado_v222.filter("
    "F.col('cd_bv') == 'ID_AQUI')\n"
    ")\n"
)

print(
    "Interpretacao: raw_prob e probabilidade condicional "
    "do proximo estado; util_score e first-passage da "
    "primeira acao util em ate 5 transicoes."
)
