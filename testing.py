from pyspark.sql import functions as F

TABELA_ORIGEM = (
    "ctg_dsti.renato_nba."
    "nba_semimarkov_v222_macroaware_long_hml"
)

TABELA_NEGOCIO = (
    "ctg_dsti.renato_nba."
    "nba_semimarkov_v222_negocio_hml"
)

schema_raw = (
    "array<struct<ranking:int,"
    "estado:string,probabilidade:double>>"
)

base_negocio = (
    spark.table(TABELA_ORIGEM)
    .filter(
        F.col("ranking_actionable").between(1, 5)
        & F.col("actionable_acao").isNotNull()
    )
    .withColumn(
        "_raw_array",
        F.from_json("raw_top5_json", schema_raw),
    )
    .withColumn(
        "_raw_rank",
        F.element_at(
            F.col("_raw_array"),
            F.col("ranking_actionable").cast("int"),
        ),
    )
    .select(
        F.to_date("ts_corte_eventos").alias("data_referencia"),
        "cd_bv",
        F.col("macro_atual").alias("acao_atual"),
        F.col("ranking_actionable").cast("int").alias("ranking"),
        F.col("_raw_rank.estado").alias("proxima_acao_raw"),
        F.col("_raw_rank.probabilidade").alias("score_raw"),
        F.col("actionable_acao").alias("proxima_acao_util"),
        F.col("actionable_score").alias("score_util"),
    )
)

(
    base_negocio.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_NEGOCIO)
)

display(
    spark.table(TABELA_NEGOCIO)
    .orderBy("cd_bv", "ranking")
    .limit(20)
)