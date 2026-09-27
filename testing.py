from pyspark.sql import functions as F


ESTADO_PILOTO = "sem_acao:::classe"

cliente_validacao = (
    F.pmod(F.xxhash64("cd_bv"), F.lit(1000)) == 1
)

validacao_temporal = (
    transicoes_sm
    .filter(F.col("origem") == ESTADO_PILOTO)
    .filter(cliente_validacao)
    .groupBy(
        "tipo_censura",
        "dur_min",
        "dur_max",
    )
    .agg(F.count("*").alias("peso"))
)

validacao_temporal.createOrReplaceGlobalTempView(
    "nba_sm_tempo_validacao"
)