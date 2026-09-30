from pyspark.sql import functions as F


autos = base_treino_v22.filter(
    F.col("autotransicao")
)

print("RESUMO DAS AUTOTRANSICOES")

autos.agg(
    F.count("*").alias("n_autotransicoes"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.avg(
        (F.col("dur_min") <= 1).cast("double")
    ).alias("pct_ate_1s"),
    F.avg(
        (F.col("dur_min") <= 5).cast("double")
    ).alias("pct_ate_5s"),
    F.avg(
        (F.col("dur_min") <= 30).cast("double")
    ).alias("pct_ate_30s"),
    F.avg(
        (F.col("dur_min") <= 300).cast("double")
    ).alias("pct_ate_5min"),
).show(
    truncate=False
)


print("ESTADOS COM MAIS AUTOTRANSICOES")

(
    autos
    .groupBy("estado")
    .agg(
        F.count("*").alias("n"),
        F.countDistinct("cd_bv").alias("n_clientes"),
        F.expr(
            "percentile_approx("
            "dur_min, array(0.5, 0.9, 0.99), 10000)"
        ).alias("quantis_seg"),
    )
    .orderBy(F.desc("n"))
    .show(
        30,
        truncate=False,
    )
)