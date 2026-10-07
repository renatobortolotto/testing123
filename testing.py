from pyspark.sql import functions as F


TABELA_ORIGEM = (
    "ctg_dsti.renato_nba."
    "nba_semimarkov_v222_macroaware_long_hml"
)

TABELA_NEGOCIO = (
    "ctg_dsti.renato_nba."
    "nba_semimarkov_v222_negocio_hml"
)

if not spark.catalog.tableExists(TABELA_ORIGEM):
    raise RuntimeError(f"Tabela não encontrada: {TABELA_ORIGEM}")

# Seleciona somente clientes com ações previstas, preservando o ranking.
base_negocio = (
    spark.table(TABELA_ORIGEM)
    .filter(
        F.col("ranking_actionable").between(1, 5)
        & F.col("actionable_acao").isNotNull()
        & F.col("actionable_score").isNotNull()
    )
    .select(
        F.to_date("ts_corte_eventos").alias("data_referencia"),
        F.col("cd_bv"),
        F.col("macro_atual").alias("acao_atual"),
        F.col("ranking_actionable").cast("int").alias("ranking"),
        F.col("actionable_acao").alias("proxima_acao"),
        F.col("actionable_score").cast("double").alias("score_acao"),
    )
)

# Substitui apenas a tabela de negócio; a tabela analítica é preservada.
(
    base_negocio.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_NEGOCIO)
)

print(f"Tabela de negócio salva em: {TABELA_NEGOCIO}")

display(spark.table(TABELA_NEGOCIO).limit(20))


CD_BV_ANALISE = "COLOQUE_O_ID_AQUI"

display(
    spark.table(TABELA_NEGOCIO)
    .filter(F.col("cd_bv") == CD_BV_ANALISE)
    .orderBy("ranking")
)