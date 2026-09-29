from pyspark.sql import functions as F


HOLDOUT_BUCKETS = [1, 2]

eventos_holdout = (
    sm_ler_eventos(
        SM_CFG["fonte_treino"],
        SM_CFG["inicio_treino"],
        SM_CFG["corte_treino_exclusivo"],
    )
    .filter(
        F.pmod(
            F.xxhash64(
                "cd_bv",
                F.lit(SM_CFG["sal_treino"]),
            ),
            F.lit(SM_CFG["modulo_amostra_treino"]),
        ).isin(HOLDOUT_BUCKETS)
    )
)

base = (
    sm_preparar_passos(
        eventos_holdout,
        SM_CFG["corte_treino_exclusivo"],
        "holdout_externo",
    )
    .withColumn(
        "validacao_cliente",
        F.lit(True),
    )
    .persist()
)

print(
    "Clientes holdout externo:",
    base.select("cd_bv").distinct().count(),
)

print(
    "Linhas holdout externo:",
    base.count(),
)