# Databricks notebook source
# NBA | V2.2 - Parte 01
# Base Semi-Markov com ultima acao real + tempo desde a ultima acao.
#
# Revisao:
# - remove sem_acao e timeout de 30 minutos como estado;
# - resolve estados simultaneos pela maior profundidade_max;
# - consolida repeticoes consecutivas do mesmo estado em ate 30 segundos;
# - usa o ultimo timestamp do burst como instante representativo da acao;
# - o ultimo estado antes do corte entra como censura a direita.
#
# Nenhuma tabela permanente e alterada nesta etapa.

from datetime import date, datetime, timedelta
import json
import uuid

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F


SM22_CFG = {
    "data_publico": "2026-09-26",
    "fonte_publico": "customers_query",
    "fonte_treino": "base_passo_raw",
    "fonte_aplicacao": "base_passo_raw",
    "inicio_treino": "2026-06-01 00:00:00",
    "corte_treino_exclusivo": "2026-09-25 00:00:00",
    "corte_estado_exclusivo": "2026-09-25 00:00:00",
    "fuso": "Etc/UTC",
    "preparar_treino": True,
    "modulo_amostra_treino": 1000,
    "modulo_amostra_publico": 1000,
    "sal_treino": "sm_v2_amostra_historica",
    "sal_publico": "sm_v2_amostra_publico",
    "sal_validacao": "sm_v2_validacao_clientes",
    "versao_modelo": "sm_v22_ultima_acao_debounce_001",
    "relogio": "ULTIMA_ACAO_TEMPO_V22",
    "horizonte_dias": 7.0,
    "debounce_seg": 30.0,
}

SM22_ESTADOS_DATA_SEM_HORA = [
    f"{produto}:::conversao"
    for produto in (
        "ativacao_conta",
        "solicitacao_cartao",
        "contrato_leves",
        "contrato_egv",
        "contrato_motos",
        "contrato_solar",
        "contrato_scp",
    )
]

SM22_ID_EXECUCAO = str(uuid.uuid4())

SM22_VIEWS = {
    "treino": "nba_sm_v22_base_treino",
    "atuais": "nba_sm_v22_estados_publico",
    "publico": "nba_sm_v22_publico",
    "config": "nba_sm_v22_configuracao",
}


def sm22_segundos(inicio: str, fim: str) -> Column:
    return (
        F.unix_micros(fim) - F.unix_micros(inicio)
    ) / F.lit(1_000_000.0)


def sm22_ha(df: DataFrame) -> bool:
    return bool(df.limit(1).count())


def sm22_exigir_colunas(
    df: DataFrame,
    campos: set[str],
    nome: str,
) -> None:
    faltantes = campos - set(df.columns)
    if faltantes:
        raise ValueError(
            f"{nome}: faltam colunas {sorted(faltantes)}"
        )


def sm22_ler_eventos(
    tabela: str,
    inicio: str | None,
    corte: str,
) -> DataFrame:
    df = spark.table(tabela)

    sm22_exigir_colunas(
        df,
        {
            "cd_bv",
            "dm_navegacao",
            "estado",
            "profundidade_max",
        },
        tabela,
    )

    if (
        df.schema["dm_navegacao"]
        .dataType.simpleString()
        != "timestamp"
    ):
        raise ValueError(
            "dm_navegacao deve ser timestamp."
        )

    df = (
        df.select(
            F.col("cd_bv").cast("string"),
            F.col("dm_navegacao").alias("ts_evento"),
            F.col("estado").cast("string"),
            F.col("profundidade_max").cast("double"),
        )
        .filter(
            (
                F.col("ts_evento")
                < F.lit(corte).cast("timestamp")
            )
            | F.col("ts_evento").isNull()
        )
    )

    if inicio:
        df = df.filter(
            (
                F.col("ts_evento")
                >= F.lit(inicio).cast("timestamp")
            )
            | F.col("ts_evento").isNull()
        )

    return df


def sm22_resolver_instantes(
    eventos: DataFrame,
) -> DataFrame:
    """Resolve multiplos estados no mesmo instante por profundidade."""

    invalido = (
        F.col("cd_bv").isNull()
        | F.col("estado").isNull()
        | (F.length(F.trim("estado")) == 0)
        | F.col("ts_evento").isNull()
    )

    if sm22_ha(eventos.filter(invalido)):
        raise ValueError(
            "Evento sem ID, estado ou timestamp."
        )

    if sm22_ha(
        eventos.filter(
            F.lower("estado").contains("sem_acao")
        )
    ):
        raise ValueError(
            "A fonte contem sem_acao. "
            "Use a fonte bruta sem estado sintetico."
        )

    eventos = (
        eventos
        .withColumn("dia", F.to_date("ts_evento"))
        .withColumn(
            "dia_impreciso",
            F.col("estado").isin(
                SM22_ESTADOS_DATA_SEM_HORA
            ),
        )
        .withColumn(
            "ts_momento",
            F.when(
                F.col("dia_impreciso"),
                F.col("dia").cast("timestamp"),
            ).otherwise(F.col("ts_evento")),
        )
        .withColumn(
            "prof_ordem",
            F.coalesce(
                F.col("profundidade_max"),
                F.lit(-1.0),
            ),
        )
    )

    por_estado = (
        eventos
        .groupBy(
            "cd_bv",
            "ts_momento",
            "estado",
        )
        .agg(
            F.max("prof_ordem").alias("prof_ordem"),
            F.max("profundidade_max").alias(
                "profundidade_max"
            ),
            F.count("*").alias("n_ev"),
            F.max(
                F.col("dia_impreciso").cast("int")
            )
            .cast("boolean")
            .alias("dia_impreciso"),
        )
    )

    janela_instante = Window.partitionBy(
        "cd_bv",
        "ts_momento",
    )

    candidatos = (
        por_estado
        .withColumn(
            "max_prof",
            F.max("prof_ordem").over(
                janela_instante
            ),
        )
        .withColumn(
            "eh_max_prof",
            F.col("prof_ordem")
            == F.col("max_prof"),
        )
        .withColumn(
            "n_estados_instante",
            F.count("*").over(
                janela_instante
            ),
        )
        .withColumn(
            "n_estados_max_prof",
            F.sum(
                F.col("eh_max_prof").cast("int")
            ).over(janela_instante),
        )
    )

    instantes = (
        candidatos
        .groupBy(
            "cd_bv",
            "ts_momento",
        )
        .agg(
            F.max("n_estados_instante").alias(
                "n_estados_instante"
            ),
            F.max("n_estados_max_prof").alias(
                "n_estados_max_prof"
            ),
            F.max("dia_impreciso").alias(
                "dia_impreciso"
            ),
            F.sum("n_ev").alias("n_ev_instante"),
            F.first(
                F.when(
                    F.col("eh_max_prof"),
                    F.col("estado"),
                ),
                ignorenulls=True,
            ).alias("estado_max_prof"),
            F.first(
                F.when(
                    F.col("eh_max_prof"),
                    F.col("profundidade_max"),
                ),
                ignorenulls=True,
            ).alias("profundidade_max"),
        )
        .withColumn(
            "ordem_ambigua",
            F.col("dia_impreciso")
            | (F.col("n_estados_max_prof") > 1),
        )
        .withColumn(
            "resolvido_por_profundidade",
            (
                F.col("n_estados_instante") > 1
            )
            & ~F.col("dia_impreciso")
            & (
                F.col("n_estados_max_prof") == 1
            ),
        )
        .withColumn(
            "estado",
            F.when(
                ~F.col("ordem_ambigua"),
                F.col("estado_max_prof"),
            ),
        )
        .drop("estado_max_prof")
    )

    return instantes


def sm22_aplicar_debounce(
    instantes: DataFrame,
) -> DataFrame:
    """Consolida repeticoes consecutivas do mesmo estado em ate 30s."""

    janela = (
        Window.partitionBy("cd_bv")
        .orderBy("ts_momento")
    )

    anteriores = (
        instantes
        .withColumn(
            "estado_anterior",
            F.lag("estado").over(janela),
        )
        .withColumn(
            "ts_anterior",
            F.lag("ts_momento").over(janela),
        )
        .withColumn(
            "ambiguo_anterior",
            F.lag("ordem_ambigua").over(janela),
        )
        .withColumn(
            "gap_anterior_seg",
            sm22_segundos(
                "ts_anterior",
                "ts_momento",
            ),
        )
    )

    mesma_acao = (
        F.col("estado").isNotNull()
        & F.col("estado_anterior").isNotNull()
        & ~F.col("ordem_ambigua")
        & ~F.coalesce(
            F.col("ambiguo_anterior"),
            F.lit(False),
        )
        & (
            F.col("estado")
            == F.col("estado_anterior")
        )
        & (
            F.col("gap_anterior_seg")
            <= F.lit(
                SM22_CFG["debounce_seg"]
            )
        )
    )

    acumulada = janela.rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow,
    )

    linhas = (
        anteriores
        .withColumn(
            "abre_acao",
            F.when(
                mesma_acao,
                F.lit(0),
            ).otherwise(F.lit(1)),
        )
        .withColumn(
            "id_acao",
            F.sum("abre_acao").over(
                acumulada
            ),
        )
    )

    acoes = (
        linhas
        .groupBy(
            "cd_bv",
            "id_acao",
        )
        .agg(
            F.max("ts_momento").alias(
                "ts_estado"
            ),
            F.max("estado").alias("estado"),
            F.sum("n_ev_instante").alias(
                "n_ev"
            ),
            F.max(
                F.col("ordem_ambigua").cast("int")
            )
            .cast("boolean")
            .alias("ordem_ambigua"),
            F.max(
                F.col("dia_impreciso").cast("int")
            )
            .cast("boolean")
            .alias("dia_impreciso"),
            F.max(
                F.col(
                    "resolvido_por_profundidade"
                ).cast("int")
            )
            .cast("boolean")
            .alias("resolvido_por_profundidade"),
            F.max("profundidade_max").alias(
                "profundidade_max"
            ),
            F.count("*").alias(
                "n_instantes_no_burst"
            ),
        )
    )

    return acoes


def sm22_preparar_passos(
    eventos: DataFrame,
    corte: str,
    nome: str,
) -> DataFrame:
    instantes = sm22_resolver_instantes(
        eventos
    )
    acoes = sm22_aplicar_debounce(
        instantes
    )

    janela = (
        Window.partitionBy("cd_bv")
        .orderBy(
            "ts_estado",
            "id_acao",
        )
    )

    passos = (
        acoes
        .withColumn(
            "passo",
            F.row_number().over(janela),
        )
        .withColumn(
            "proximo_estado_bruto",
            F.lead("estado").over(janela),
        )
        .withColumn(
            "proximo_ts",
            F.lead("ts_estado").over(janela),
        )
        .withColumn(
            "proximo_ambiguo",
            F.lead("ordem_ambigua").over(janela),
        )
        .withColumn(
            "ts_corte_estado",
            F.lit(corte).cast("timestamp"),
        )
        .withColumn(
            "ts_fim",
            F.when(
                F.col("proximo_ts").isNotNull(),
                F.col("proximo_ts"),
            ).otherwise(
                F.col("ts_corte_estado")
            ),
        )
        .withColumn(
            "dur_min",
            sm22_segundos(
                "ts_estado",
                "ts_fim",
            ),
        )
        .withColumn(
            "tipo_censura",
            F.when(
                F.col("ordem_ambigua"),
                F.lit("origem_ambigua"),
            )
            .when(
                F.col("proximo_ts").isNull(),
                F.lit("direita"),
            )
            .when(
                F.col("proximo_ambiguo"),
                F.lit("direita"),
            )
            .otherwise(F.lit("exata")),
        )
        .withColumn(
            "destino",
            F.when(
                F.col("tipo_censura")
                == "exata",
                F.col("proximo_estado_bruto"),
            ),
        )
        .withColumn(
            "dur_max",
            F.when(
                F.col("tipo_censura")
                == "exata",
                F.col("dur_min"),
            ),
        )
        .withColumn(
            "status_observacao",
            F.when(
                F.col("ordem_ambigua"),
                F.lit("ORIGEM_AMBIGUA"),
            )
            .when(
                F.col("proximo_ts").isNull(),
                F.lit("CENSURA_CORTE"),
            )
            .when(
                F.col("proximo_ambiguo"),
                F.lit(
                    "CENSURA_ANTES_AMBIGUIDADE"
                ),
            )
            .otherwise(
                F.lit("SAIDA_OBSERVADA")
            ),
        )
        .withColumn(
            "elegivel_ajuste",
            F.col("estado").isNotNull()
            & (
                F.col("tipo_censura")
                != "origem_ambigua"
            )
            & (
                F.col("dur_min") > 0
            ),
        )
        .withColumn(
            "autotransicao",
            (
                F.col("tipo_censura")
                == "exata"
            )
            & (
                F.col("estado")
                == F.col("destino")
            ),
        )
        .withColumn(
            "relogio",
            F.lit(
                SM22_CFG["relogio"]
            ),
        )
        .select(
            "cd_bv",
            "passo",
            "ts_estado",
            "estado",
            "n_ev",
            "profundidade_max",
            "n_instantes_no_burst",
            "destino",
            "dur_min",
            "dur_max",
            "tipo_censura",
            "status_observacao",
            "elegivel_ajuste",
            "autotransicao",
            "ordem_ambigua",
            "dia_impreciso",
            "resolvido_por_profundidade",
            "ts_corte_estado",
            "relogio",
        )
    )

    invalidas = passos.filter(
        F.col("estado").isNotNull()
        & (
            F.col("dur_min").isNull()
            | (F.col("dur_min") <= 0)
        )
    )

    if sm22_ha(invalidas):
        raise ValueError(
            f"{nome}: duracao nao positiva encontrada."
        )

    return passos


if (
    spark.conf.get(
        "spark.sql.session.timeZone"
    )
    != SM22_CFG["fuso"]
):
    raise ValueError(
        "Fuso da sessao difere do configurado."
    )

referencia = date.fromisoformat(
    SM22_CFG["data_publico"]
)
corte_referencia = datetime.combine(
    referencia + timedelta(days=1),
    datetime.min.time(),
)
corte_treino = datetime.fromisoformat(
    SM22_CFG["corte_treino_exclusivo"]
)
corte_estado = datetime.fromisoformat(
    SM22_CFG["corte_estado_exclusivo"]
)

if (
    corte_treino > corte_estado
    or corte_estado > corte_referencia
):
    raise ValueError(
        "Exigir corte_treino <= corte_estado "
        "<= fechamento da referencia."
    )

for chave in (
    "modulo_amostra_treino",
    "modulo_amostra_publico",
):
    if SM22_CFG[chave] < 1:
        raise ValueError(
            f"{chave} precisa ser >= 1."
        )

if SM22_CFG["debounce_seg"] < 0:
    raise ValueError(
        "debounce_seg precisa ser >= 0."
    )

SM22_CFG["id_execucao"] = SM22_ID_EXECUCAO
SM22_CFG["estados_data_sem_hora"] = (
    SM22_ESTADOS_DATA_SEM_HORA
)

print(
    "V2.2 | ultima acao real + tempo desde a ultima acao"
)
print(
    "Debounce do mesmo estado:",
    SM22_CFG["debounce_seg"],
    "segundos",
)
print(
    "Nao existe estado sem_acao nesta versao."
)


# COMMAND ----------
# Treinamento

if SM22_CFG["preparar_treino"]:
    eventos_treino = (
        sm22_ler_eventos(
            SM22_CFG["fonte_treino"],
            SM22_CFG["inicio_treino"],
            SM22_CFG[
                "corte_treino_exclusivo"
            ],
        )
        .filter(
            F.pmod(
                F.xxhash64(
                    "cd_bv",
                    F.lit(
                        SM22_CFG["sal_treino"]
                    ),
                ),
                F.lit(
                    SM22_CFG[
                        "modulo_amostra_treino"
                    ]
                ),
            )
            == 0
        )
    )

    instantes_treino_v22 = (
        sm22_resolver_instantes(
            eventos_treino
        )
        .persist(
            StorageLevel.MEMORY_AND_DISK
        )
    )

    acoes_treino_v22 = (
        sm22_aplicar_debounce(
            instantes_treino_v22
        )
        .persist(
            StorageLevel.MEMORY_AND_DISK
        )
    )

    base_treino_v22 = sm22_preparar_passos(
        eventos_treino,
        SM22_CFG[
            "corte_treino_exclusivo"
        ],
        "treino_v22",
    )

    base_treino_v22 = (
        base_treino_v22
        .withColumn(
            "validacao_cliente",
            F.pmod(
                F.xxhash64(
                    "cd_bv",
                    F.lit(
                        SM22_CFG[
                            "sal_validacao"
                        ]
                    ),
                ),
                F.lit(5),
            )
            == 0,
        )
        .persist(
            StorageLevel.MEMORY_AND_DISK
        )
    )

    base_treino_v22.createOrReplaceTempView(
        SM22_VIEWS["treino"]
    )

    print("TREINO V2.2 - status")
    (
        base_treino_v22
        .groupBy(
            "validacao_cliente",
            "status_observacao",
        )
        .count()
        .orderBy(
            "validacao_cliente",
            "status_observacao",
        )
        .show(truncate=False)
    )

    print("TREINO V2.2 - resumo")
    (
        base_treino_v22
        .agg(
            F.count("*").alias(
                "n_linhas"
            ),
            F.countDistinct(
                "cd_bv"
            ).alias(
                "n_clientes"
            ),
            F.countDistinct(
                "estado"
            ).alias(
                "n_estados"
            ),
            F.sum(
                F.col(
                    "autotransicao"
                ).cast("long")
            ).alias(
                "n_autotransicoes"
            ),
        )
        .show(truncate=False)
    )

    print("TREINO V2.2 - debounce")
    (
        instantes_treino_v22
        .agg(
            F.count("*").alias(
                "n_instantes_resolvidos"
            ),
        )
        .crossJoin(
            acoes_treino_v22
            .agg(
                F.count("*").alias(
                    "n_acoes_apos_debounce"
                ),
                F.sum(
                    (
                        F.col(
                            "n_instantes_no_burst"
                        )
                        > 1
                    ).cast("long")
                ).alias(
                    "n_bursts_consolidados"
                ),
            )
        )
        .withColumn(
            "n_instantes_absorvidos",
            F.col(
                "n_instantes_resolvidos"
            )
            - F.col(
                "n_acoes_apos_debounce"
            ),
        )
        .show(truncate=False)
    )

    print(
        "TREINO V2.2 - quantis de duracao exata"
    )
    (
        base_treino_v22
        .filter(
            F.col("tipo_censura")
            == "exata"
        )
        .selectExpr(
            "percentile_approx("
            "dur_min, "
            "array(0.5, 0.9, 0.95, 0.99), "
            "10000"
            ") as quantis_seg"
        )
        .show(truncate=False)
    )

    print(
        "TREINO V2.2 - autotransicoes restantes"
    )
    (
        base_treino_v22
        .filter(
            F.col("autotransicao")
        )
        .agg(
            F.count("*").alias(
                "n_autotransicoes"
            ),
            F.avg(
                (
                    F.col("dur_min")
                    <= 30
                ).cast("double")
            ).alias(
                "pct_ate_30s"
            ),
            F.avg(
                (
                    F.col("dur_min")
                    <= 300
                ).cast("double")
            ).alias(
                "pct_ate_5min"
            ),
        )
        .show(truncate=False)
    )


# COMMAND ----------
# Publico de aplicacao

publico_raw = spark.table(
    SM22_CFG["fonte_publico"]
)

sm22_exigir_colunas(
    publico_raw,
    {"cd_bv"},
    SM22_CFG["fonte_publico"],
)

publico_v22 = (
    publico_raw
    .select(
        F.col("cd_bv").cast("string")
    )
    .filter(
        F.col("cd_bv").isNotNull()
    )
    .distinct()
    .filter(
        F.pmod(
            F.xxhash64(
                "cd_bv",
                F.lit(
                    SM22_CFG["sal_publico"]
                ),
            ),
            F.lit(
                SM22_CFG[
                    "modulo_amostra_publico"
                ]
            ),
        )
        == 0
    )
    .withColumn(
        "data_referencia",
        F.lit(
            SM22_CFG["data_publico"]
        ).cast("date"),
    )
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)

if not sm22_ha(publico_v22):
    raise ValueError(
        "Publico V2.2 vazio."
    )

publico_v22.createOrReplaceTempView(
    SM22_VIEWS["publico"]
)

eventos_app = (
    sm22_ler_eventos(
        SM22_CFG["fonte_aplicacao"],
        None,
        SM22_CFG[
            "corte_estado_exclusivo"
        ],
    )
    .join(
        publico_v22.select("cd_bv"),
        "cd_bv",
        "left_semi",
    )
)

passos_app_v22 = sm22_preparar_passos(
    eventos_app,
    SM22_CFG["corte_estado_exclusivo"],
    "aplicacao_v22",
)

janela_ultimo = (
    Window.partitionBy("cd_bv")
    .orderBy(F.desc("passo"))
)

ultimo_v22 = (
    passos_app_v22
    .withColumn(
        "_rn",
        F.row_number().over(
            janela_ultimo
        ),
    )
    .filter(F.col("_rn") == 1)
    .select(
        "cd_bv",
        F.col("estado").alias(
            "ultima_acao"
        ),
        F.col("ts_estado").alias(
            "ts_ultima_acao"
        ),
        "ordem_ambigua",
    )
)

atuais_v22 = (
    publico_v22
    .join(
        ultimo_v22,
        "cd_bv",
        "left",
    )
    .withColumn(
        "ts_corte_estado",
        F.lit(
            SM22_CFG[
                "corte_estado_exclusivo"
            ]
        ).cast("timestamp"),
    )
    .withColumn(
        "tempo_desde_ultima_acao_seg",
        F.when(
            F.col(
                "ts_ultima_acao"
            ).isNotNull(),
            sm22_segundos(
                "ts_ultima_acao",
                "ts_corte_estado",
            ),
        ),
    )
    .withColumn(
        "status_input",
        F.when(
            F.col(
                "ts_ultima_acao"
            ).isNull(),
            F.lit("SEM_HISTORICO"),
        )
        .when(
            F.col("ordem_ambigua"),
            F.lit(
                "ESTADO_ATUAL_AMBIGUO"
            ),
        )
        .otherwise(F.lit("OK")),
    )
    .withColumn(
        "relogio",
        F.lit(
            SM22_CFG["relogio"]
        ),
    )
    .persist(
        StorageLevel.MEMORY_AND_DISK
    )
)

atuais_v22.createOrReplaceTempView(
    SM22_VIEWS["atuais"]
)

print("PUBLICO V2.2 - status")
(
    atuais_v22
    .groupBy("status_input")
    .count()
    .orderBy(F.desc("count"))
    .show(truncate=False)
)

print("PUBLICO V2.2 - ultimas acoes")
(
    atuais_v22
    .groupBy("ultima_acao")
    .count()
    .orderBy(F.desc("count"))
    .show(
        20,
        truncate=False,
    )
)

if sm22_ha(
    atuais_v22.filter(
        F.lower(
            F.coalesce(
                F.col("ultima_acao"),
                F.lit(""),
            )
        ).contains("sem_acao")
    )
):
    raise RuntimeError(
        "V2.2 encontrou sem_acao no estado atual."
    )

meta_json = json.dumps(
    SM22_CFG,
    ensure_ascii=False,
    sort_keys=True,
)

spark.createDataFrame(
    [(meta_json,)],
    "config_json string",
).createOrReplaceTempView(
    SM22_VIEWS["config"]
)

print(
    "V2.2 Parte 01 com debounce concluida."
)
print(
    "Proxima etapa: ajustar o Semi-Markov "
    "sobre ultima acao + tempo ate a proxima acao."
)
