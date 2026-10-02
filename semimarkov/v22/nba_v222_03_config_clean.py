# Databricks notebook source
# NBA | V2.2.2 CLEAN - Parte 03
# Configuracao limpa de estados e diagnostico de acoes macro.
#
# Esta etapa NAO retreina o modelo.
# Ela cria metadata a partir dos estados comportamentais reais.

from pyspark.sql import functions as F

FONTE_ESTADOS = "ctg_dsti.renato_nba.base_jornadas_onboarding"
TABELA_CONFIG = "ctg_dsti.renato_nba.nba_config_estados_v222_clean"

config_clean = (
    spark.table(FONTE_ESTADOS)
    .select(F.trim(F.col("estado")).alias("estado"))
    .filter(F.col("estado").isNotNull())
    .filter(F.length("estado") > 0)
    .filter(~F.lower("estado").contains("sem_acao"))
    .filter(~F.lower("estado").contains("conversao"))
    .distinct()
    .withColumn(
        "_parte_1",
        F.split(F.col("estado"), ":::").getItem(0),
    )
    .withColumn(
        "_parte_2",
        F.split(F.col("estado"), ":::").getItem(1),
    )
    .withColumn(
        "eh_funil",
        F.col("_parte_2").isin("topo", "navegacao", "sucesso"),
    )
    .withColumn(
        "tipo_estado",
        F.when(F.col("_parte_2") == "topo", F.lit("funil_abertura"))
        .when(F.col("_parte_2") == "navegacao", F.lit("funil_meio"))
        .when(F.col("_parte_2") == "sucesso", F.lit("funil_conclusao"))
        .when(F.lower("estado").contains("perdido"), F.lit("terminal"))
        .when(F.col("_parte_1") == "atendimento", F.lit("atendimento"))
        .otherwise(F.lit("nao_mapeado")),
    )
    .withColumn(
        "funil",
        F.when(F.col("eh_funil"), F.col("_parte_1")),
    )
    .withColumn(
        "etapa",
        F.when(F.col("eh_funil"), F.col("_parte_2")),
    )
    .withColumn(
        "detalhe",
        F.when(~F.col("eh_funil"), F.col("_parte_2")),
    )
    .withColumn(
        "acao_macro",
        F.when(F.col("eh_funil"), F.col("_parte_1"))
        .otherwise(F.col("estado")),
    )
    .withColumn(
        "terminal_real",
        F.col("tipo_estado") == "terminal",
    )
    .select(
        "estado",
        "tipo_estado",
        "acao_macro",
        "funil",
        "etapa",
        "detalhe",
        "terminal_real",
    )
)

(
    config_clean
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABELA_CONFIG)
)

print("CONFIG CLEAN criada:")
print(TABELA_CONFIG)

print("CONFIG CLEAN - RESUMO")
(
    config_clean
    .agg(
        F.count("*").alias("n_estados"),
        F.countDistinct("acao_macro").alias("n_acoes_macro"),
        F.sum(F.col("terminal_real").cast("long")).alias("n_terminais_reais"),
    )
    .show(truncate=False)
)

print("CONFIG CLEAN - TIPOS")
(
    config_clean
    .groupBy("tipo_estado")
    .count()
    .orderBy(F.desc("count"))
    .show(50, truncate=False)
)

n_conversao = (
    config_clean
    .filter(F.lower("estado").contains("conversao"))
    .count()
)

n_sem_acao = (
    config_clean
    .filter(F.lower("estado").contains("sem_acao"))
    .count()
)

print("Estados com conversao:", n_conversao)
print("Estados com sem_acao:", n_sem_acao)

if n_conversao != 0 or n_sem_acao != 0:
    raise RuntimeError(
        "A configuracao CLEAN ainda contem estado artificial."
    )

eventos = (
    spark.table(FONTE_ESTADOS)
    .select("estado")
    .filter(F.col("estado").isNotNull())
)

macro_resumo = (
    eventos
    .join(config_clean, "estado", "inner")
    .groupBy("acao_macro")
    .agg(
        F.count("*").alias("n_eventos"),
        F.countDistinct("estado").alias("n_estados"),
        F.sort_array(F.collect_set("etapa")).alias("etapas"),
        F.sort_array(F.collect_set("tipo_estado")).alias("tipos_estado"),
    )
    .orderBy(F.desc("n_eventos"))
)

print("ACOES MACRO - TOP 100")
macro_resumo.show(100, truncate=False)

print("ESTADOS NAO MAPEADOS")
(
    config_clean
    .filter(F.col("tipo_estado") == "nao_mapeado")
    .orderBy("estado")
    .show(100, truncate=False)
)

print(
    "\nParte 03 concluida.\n"
    "Proxima etapa: definir quais acoes_macro sao relevantes para NBA "
    "e executar o Multi-step na customers_query completa."
)
