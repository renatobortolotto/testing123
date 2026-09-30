from pyspark.sql import functions as F


fonte = spark.table(SM22_CFG["fonte_aplicacao"])

print(
    "Tem profundidade_max:",
    "profundidade_max" in fonte.columns,
)

empatados = (
    fonte
    .select(
        "cd_bv",
        "dm_navegacao",
        "estado",
        "profundidade_max",
    )
    .filter(
        F.col("cd_bv").isNotNull()
        & F.col("dm_navegacao").isNotNull()
        & F.col("estado").isNotNull()
    )
    .groupBy(
        "cd_bv",
        "dm_navegacao",
    )
    .agg(
        F.countDistinct("estado").alias("n_estados"),
        F.countDistinct("profundidade_max").alias(
            "n_profundidades"
        ),
        F.sort_array(
            F.collect_set("estado")
        ).alias("estados"),
        F.sort_array(
            F.collect_set("profundidade_max")
        ).alias("profundidades"),
    )
    .filter(
        F.col("n_estados") > 1
    )
)

print("RESUMO DOS EMPATES")

empatados.agg(
    F.count("*").alias("n_momentos_ambiguos"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum(
        (
            F.col("n_estados")
            == F.col("n_profundidades")
        ).cast("long")
    ).alias("n_potencialmente_ordenaveis"),
).show(
    truncate=False
)


print("COMBINACOES MAIS FREQUENTES")

(
    empatados
    .withColumn(
        "combinacao",
        F.concat_ws(
            " -> ",
            "estados",
        ),
    )
    .groupBy(
        "combinacao",
        "n_estados",
        "n_profundidades",
    )
    .count()
    .orderBy(
        F.desc("count")
    )
    .show(
        30,
        truncate=False,
    )
)


print("EXEMPLOS COM PROFUNDIDADE")

(
    empatados
    .select(
        "cd_bv",
        "dm_navegacao",
        "estados",
        "profundidades",
        "n_estados",
        "n_profundidades",
    )
    .show(
        30,
        truncate=False,
    )
)