from pyspark.sql import functions as F


fonte = spark.table(SM22_CFG["fonte_aplicacao"])

clientes_amostra = (
    base_treino_v22
    .select("cd_bv")
    .distinct()
)

eventos_amostra = (
    fonte
    .select(
        "cd_bv",
        "dm_navegacao",
        "estado",
        "profundidade_max",
    )
    .join(
        F.broadcast(clientes_amostra),
        "cd_bv",
        "left_semi",
    )
    .filter(
        F.col("dm_navegacao").isNotNull()
        & F.col("estado").isNotNull()
    )
)

empates = (
    eventos_amostra
    .groupBy(
        "cd_bv",
        "dm_navegacao",
    )
    .agg(
        F.countDistinct("estado").alias("n_estados"),
        F.countDistinct("profundidade_max").alias(
            "n_profundidades"
        ),
        F.sum(
            F.col("profundidade_max").isNull().cast("long")
        ).alias("n_profundidade_null"),
        F.count("*").alias("n_linhas"),
    )
    .filter(
        F.col("n_estados") > 1
    )
)

print("RESUMO DOS EMPATES")

empates.agg(
    F.count("*").alias("n_momentos_ambiguos"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum(
        (
            (F.col("n_estados") == F.col("n_profundidades"))
            & (F.col("n_profundidade_null") == 0)
        ).cast("long")
    ).alias("n_potencialmente_ordenaveis"),
).show(truncate=False)







chaves_exemplo = (
    empates
    .orderBy(
        F.desc("n_estados")
    )
    .limit(100)
    .select(
        "cd_bv",
        "dm_navegacao",
    )
)

exemplos = (
    eventos_amostra
    .join(
        F.broadcast(chaves_exemplo),
        ["cd_bv", "dm_navegacao"],
        "inner",
    )
    .orderBy(
        "cd_bv",
        "dm_navegacao",
        "profundidade_max",
    )
)

exemplos.show(
    300,
    truncate=False,
)