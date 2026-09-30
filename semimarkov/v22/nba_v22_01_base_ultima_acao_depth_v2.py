# Databricks notebook source
# NBA | V2.2 - Parte 01
# Estado = ultima acao real; tempo = tempo desde essa acao.
# Nao existe sem_acao e nao existe timeout de 30 min nesta versao.

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
    # Mesma amostra da V2.1 para comparacao justa.
    "modulo_amostra_treino": 1000,
    "modulo_amostra_publico": 1000,
    "sal_treino": "sm_v2_amostra_historica",
    "sal_publico": "sm_v2_amostra_publico",
    "sal_validacao": "sm_v2_validacao_clientes",
    "versao_modelo": "sm_v22_ultima_acao_001",
    "relogio": "ULTIMA_ACAO_TEMPO_V22",
    "horizonte_dias": 7.0,
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
        raise ValueError(f"{nome}: faltam {sorted(faltantes)}")


def sm22_ler_eventos(
    tabela: str,
    inicio: str | None,
    corte: str,
) -> DataFrame:
    df = spark.table(tabela)
    sm22_exigir_colunas(
        df,
        {"cd_bv", "dm_navegacao", "estado", "profundidade_max"},
        tabela,
    )

    if df.schema["dm_navegacao"].dataType.simpleString() != "timestamp":
        raise ValueError("dm_navegacao deve ser timestamp.")

    df = (
        df.select(
            F.col("cd_bv").cast("string"),
            F.col("dm_navegacao").alias("ts_evento"),
            F.col("estado").cast("string"),
            F.col("profundidade_max").cast("double"),
        )
        .filter(
            (F.col("ts_evento") < F.lit(corte).cast("timestamp"))
            | F.col("ts_evento").isNull()
        )
    )

    if inicio:
        df = df.filter(
            (F.col("ts_evento") >= F.lit(inicio).cast("timestamp"))
            | F.col("ts_evento").isNull()
        )

    return df


def sm22_preparar_passos(
    eventos: DataFrame,
    corte: str,
) -> DataFrame:
    """Constroi a cadeia somente com acoes reais observadas."""
    invalido = (
        F.col("cd_bv").isNull()
        | F.col("estado").isNull()
        | (F.length(F.trim("estado")) == 0)
        | F.col("ts_evento").isNull()
    )
    if sm22_ha(eventos.filter(invalido)):
        raise ValueError("Evento sem ID, estado ou timestamp.")

    if sm22_ha(
        eventos.filter(
            F.lower(F.col("estado")).contains("sem_acao")
        )
    ):
        raise ValueError(
            "A fonte contem sem_acao. Use base_passo_raw sem estado sintetico."
        )

    eventos = eventos.withColumn("dia", F.to_date("ts_evento"))

    dias_imprecisos = (
        eventos.filter(F.col("estado").isin(SM22_ESTADOS_DATA_SEM_HORA))
        .select("cd_bv", "dia")
        .distinct()
        .withColumn("dia_impreciso", F.lit(True))
    )

    eventos = (
        eventos.join(dias_imprecisos, ["cd_bv", "dia"], "left")
        .fillna({"dia_impreciso": False})
        .withColumn(
            "ts_momento",
            F.when(
                F.col("dia_impreciso"),
                F.col("dia").cast("timestamp"),
            ).otherwise(F.col("ts_evento")),
        )
    )

    # Um mesmo evento tecnico pode emitir varios estados no mesmo timestamp.
    # Nao os tratamos como transicoes sucessivas, pois isso criaria
    # permanencias de zero segundos. Mantemos o estado de maior profundidade
    # normalizada como representante daquele instante. Empate na maior
    # profundidade continua ambiguo. Dias com timestamp impreciso tambem.
    por_estado = (
        eventos.groupBy("cd_bv", "ts_momento", "estado")
        .agg(
            F.max("profundidade_max").alias("profundidade_max"),
            F.count("*").alias("n_ev_estado"),
            F.max(F.col("dia_impreciso").cast("int"))
            .cast("boolean")
            .alias("dia_impreciso_estado"),
        )
    )

    janela_momento = Window.partitionBy("cd_bv", "ts_momento")
    por_estado = por_estado.withColumn(
        "profundidade_max_instante",
        F.max("profundidade_max").over(janela_momento),
    )

    candidatos_max = (
        por_estado.filter(
            F.col("profundidade_max").eqNullSafe(
                F.col("profundidade_max_instante")
            )
        )
        .groupBy("cd_bv", "ts_momento")
        .agg(
            F.sort_array(F.collect_set("estado")).alias("estados_no_max"),
            F.first("profundidade_max_instante").alias(
                "profundidade_escolhida"
            ),
        )
    )

    resumo_momento = (
        por_estado.groupBy("cd_bv", "ts_momento")
        .agg(
            F.sum("n_ev_estado").alias("n_ev"),
            F.countDistinct("estado").alias("n_estados_no_instante"),
            F.max(F.col("dia_impreciso_estado").cast("int"))
            .cast("boolean")
            .alias("dia_impreciso"),
        )
    )

    momentos = (
        resumo_momento.join(
            candidatos_max,
            ["cd_bv", "ts_momento"],
            "left",
        )
        .withColumn(
            "n_estados_no_max",
            F.size("estados_no_max"),
        )
        .withColumn(
            "ordem_ambigua",
            F.col("dia_impreciso")
            | F.col("estados_no_max").isNull()
            | (F.col("n_estados_no_max") != 1),
        )
        .withColumn(
            "estado",
            F.when(
                ~F.col("ordem_ambigua"),
                F.element_at("estados_no_max", 1),
            ),
        )
    )

    janela = Window.partitionBy("cd_bv").orderBy("ts_momento")

    passos = (
        momentos.withColumn("passo", F.row_number().over(janela))
        .withColumn("proximo_estado", F.lead("estado").over(janela))
        .withColumn("proximo_ts", F.lead("ts_momento").over(janela))
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
            F.coalesce("proximo_ts", "ts_corte_estado"),
        )
        .withColumn(
            "dur_min",
            sm22_segundos("ts_momento", "ts_fim"),
        )
        .withColumn(
            "tipo_censura",
            F.when(F.col("ordem_ambigua"), "origem_ambigua")
            .when(F.col("proximo_ts").isNull(), "direita")
            .when(F.col("proximo_ambiguo"), "direita")
            .otherwise("exata"),
        )
        .withColumn(
            "destino",
            F.when(
                F.col("tipo_censura") == "exata",
                F.col("proximo_estado"),
            ),
        )
        .withColumn(
            "dur_max",
            F.when(
                F.col("tipo_censura") == "exata",
                F.col("dur_min"),
            ),
        )
        .withColumn(
            "status_observacao",
            F.when(F.col("ordem_ambigua"), "ORIGEM_AMBIGUA")
            .when(F.col("proximo_ts").isNull(), "CENSURA_CORTE")
            .when(
                F.col("proximo_ambiguo"),
                "CENSURA_ANTES_AMBIGUIDADE",
            )
            .otherwise("SAIDA_OBSERVADA"),
        )
        .withColumn(
            "elegivel_ajuste",
            F.col("estado").isNotNull()
            & (F.col("tipo_censura") != "origem_ambigua")
            & (F.col("dur_min") > 0),
        )
        .withColumn(
            "autotransicao",
            (F.col("tipo_censura") == "exata")
            & (F.col("estado") == F.col("destino")),
        )
        .withColumn("relogio", F.lit(SM22_CFG["relogio"]))
        .select(
            "cd_bv",
            "passo",
            F.col("ts_momento").alias("ts_estado"),
            "estado",
            "n_ev",
            "destino",
            "dur_min",
            "dur_max",
            "tipo_censura",
            "status_observacao",
            "elegivel_ajuste",
            "autotransicao",
            "ordem_ambigua",
            "dia_impreciso",
            "profundidade_escolhida",
            "n_estados_no_instante",
            "n_estados_no_max",
            "ts_corte_estado",
            "relogio",
        )
    )

    invalidas = passos.filter(
        F.col("estado").isNotNull()
        & (F.col("dur_min").isNull() | (F.col("dur_min") <= 0))
    )
    if sm22_ha(invalidas):
        raise ValueError(
            "Duracao nao positiva encontrada; nao aplicar piso artificial."
        )

    return passos


# =========================
# Validar configuracao
# =========================

if spark.conf.get("spark.sql.session.timeZone") != SM22_CFG["fuso"]:
    raise ValueError("Fuso da sessao difere do configurado.")

referencia = date.fromisoformat(SM22_CFG["data_publico"])
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

if corte_treino > corte_estado or corte_estado > corte_referencia:
    raise ValueError(
        "Exigir corte_treino <= corte_estado <= fechamento da referencia."
    )

SM22_CFG["id_execucao"] = SM22_ID_EXECUCAO
SM22_CFG["estados_data_sem_hora"] = SM22_ESTADOS_DATA_SEM_HORA

print("V2.2 | ultima acao real + tempo desde a ultima acao")
print("Nao existe estado sem_acao nesta versao.")


# COMMAND ----------
# Treino

if SM22_CFG["preparar_treino"]:
    eventos_treino = (
        sm22_ler_eventos(
            SM22_CFG["fonte_treino"],
            SM22_CFG["inicio_treino"],
            SM22_CFG["corte_treino_exclusivo"],
        )
        .filter(
            F.pmod(
                F.xxhash64(
                    "cd_bv",
                    F.lit(SM22_CFG["sal_treino"]),
                ),
                F.lit(SM22_CFG["modulo_amostra_treino"]),
            )
            == 0
        )
    )

    base_treino_v22 = sm22_preparar_passos(
        eventos_treino,
        SM22_CFG["corte_treino_exclusivo"],
    )

    base_treino_v22 = (
        base_treino_v22.withColumn(
            "validacao_cliente",
            F.pmod(
                F.xxhash64(
                    "cd_bv",
                    F.lit(SM22_CFG["sal_validacao"]),
                ),
                F.lit(5),
            )
            == 0,
        )
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    base_treino_v22.createOrReplaceTempView(SM22_VIEWS["treino"])

    print("TREINO V2.2 - status")
    (
        base_treino_v22.groupBy(
            "validacao_cliente",
            "status_observacao",
        )
        .count()
        .show(truncate=False)
    )

    print("TREINO V2.2 - resumo")
    (
        base_treino_v22.agg(
            F.count("*").alias("n_linhas"),
            F.countDistinct("cd_bv").alias("n_clientes"),
            F.countDistinct("estado").alias("n_estados"),
            F.sum(F.col("autotransicao").cast("long")).alias(
                "n_autotransicoes"
            ),
        )
        .show(truncate=False)
    )

    print("TREINO V2.2 - quantis de duracao exata")
    (
        base_treino_v22.filter(F.col("tipo_censura") == "exata")
        .selectExpr(
            "percentile_approx("
            "dur_min, array(0.5, 0.9, 0.95, 0.99), 10000"
            ") as quantis_seg"
        )
        .show(truncate=False)
    )


# COMMAND ----------
# Aplicacao

publico_raw = spark.table(SM22_CFG["fonte_publico"])
sm22_exigir_colunas(
    publico_raw,
    {"cd_bv"},
    SM22_CFG["fonte_publico"],
)

publico_v22 = (
    publico_raw.select(F.col("cd_bv").cast("string"))
    .filter(F.col("cd_bv").isNotNull())
    .distinct()
    .filter(
        F.pmod(
            F.xxhash64(
                "cd_bv",
                F.lit(SM22_CFG["sal_publico"]),
            ),
            F.lit(SM22_CFG["modulo_amostra_publico"]),
        )
        == 0
    )
    .withColumn(
        "data_referencia",
        F.lit(SM22_CFG["data_publico"]).cast("date"),
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

if not sm22_ha(publico_v22):
    raise ValueError("Publico V2.2 vazio.")

publico_v22.createOrReplaceTempView(SM22_VIEWS["publico"])

eventos_app = (
    sm22_ler_eventos(
        SM22_CFG["fonte_aplicacao"],
        None,
        SM22_CFG["corte_estado_exclusivo"],
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
)

janela_ultimo = Window.partitionBy("cd_bv").orderBy(F.desc("passo"))

ultimo_v22 = (
    passos_app_v22.withColumn(
        "_rn",
        F.row_number().over(janela_ultimo),
    )
    .filter(F.col("_rn") == 1)
    .select(
        "cd_bv",
        F.col("estado").alias("ultima_acao"),
        F.col("ts_estado").alias("ts_ultima_acao"),
        "ordem_ambigua",
    )
)

atuais_v22 = (
    publico_v22.join(ultimo_v22, "cd_bv", "left")
    .withColumn(
        "ts_corte_estado",
        F.lit(SM22_CFG["corte_estado_exclusivo"]).cast("timestamp"),
    )
    .withColumn(
        "tempo_desde_ultima_acao_seg",
        F.when(
            F.col("ts_ultima_acao").isNotNull(),
            sm22_segundos("ts_ultima_acao", "ts_corte_estado"),
        ),
    )
    .withColumn(
        "status_input",
        F.when(F.col("ts_ultima_acao").isNull(), "SEM_HISTORICO")
        .when(F.col("ordem_ambigua"), "ESTADO_ATUAL_AMBIGUO")
        .otherwise("OK"),
    )
    .withColumn("relogio", F.lit(SM22_CFG["relogio"]))
    .persist(StorageLevel.MEMORY_AND_DISK)
)

atuais_v22.createOrReplaceTempView(SM22_VIEWS["atuais"])

print("PUBLICO V2.2 - status")
(
    atuais_v22.groupBy("status_input")
    .count()
    .orderBy(F.desc("count"))
    .show(truncate=False)
)

print("PUBLICO V2.2 - ultimas acoes")
(
    atuais_v22.groupBy("ultima_acao")
    .count()
    .orderBy(F.desc("count"))
    .show(20, truncate=False)
)

if sm22_ha(
    atuais_v22.filter(
        F.lower(F.coalesce(F.col("ultima_acao"), F.lit("")))
        .contains("sem_acao")
    )
):
    raise RuntimeError("V2.2 encontrou sem_acao no estado atual.")

meta_json = json.dumps(
    SM22_CFG,
    ensure_ascii=False,
    sort_keys=True,
)

spark.createDataFrame(
    [(meta_json,)],
    "config_json string",
).createOrReplaceTempView(SM22_VIEWS["config"])

print("V2.2 - resolucao de estados simultaneos")
(
    base_treino_v22
    .agg(
        F.sum(
            (F.col("n_estados_no_instante") > 1).cast("long")
        ).alias("n_instantes_multiplos"),
        F.sum(
            (
                (F.col("n_estados_no_instante") > 1)
                & ~F.col("ordem_ambigua")
            ).cast("long")
        ).alias("n_resolvidos_por_profundidade"),
        F.sum(
            F.col("ordem_ambigua").cast("long")
        ).alias("n_ambiguos_finais"),
    )
    .show(truncate=False)
)

print("V2.2 Parte 01 concluida.")
print(
    "Proxima etapa: ajustar o Semi-Markov sobre ultima acao + "
    "duracao ate a proxima acao."
)
