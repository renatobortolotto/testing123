from pyspark.sql import functions as F


diagnostico_temporal = (
    sm_previsoes
    .filter(F.col("ranking").isNotNull())
    .groupBy("acao_atual", "status_temporal")
    .agg(
        F.countDistinct("cd_bv").alias("n_clientes"),
        F.min("tempo_no_estado_seg").alias("menor_idade_seg"),
        F.max("tempo_no_estado_seg").alias("maior_idade_seg"),
        F.max(
            F.abs(
                F.col("prob_proxima_acao")
                - F.col("prob_na_entrada")
            )
        ).alias("maior_alteracao_probabilidade"),
    )
)

diagnostico_temporal.show(truncate=False)

sm_modelos.filter(
    F.col("origem") == "sem_acao:::classe"
).select(
    "origem",
    "status_modelo",
    "n_amostra",
    "n_grupos",
    "detalhe",
).show(truncate=False)