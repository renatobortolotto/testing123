from datetime import datetime, timezone

from pyspark.sql import functions as F


COLUNA_DATA = "data_ultimo_egv_alvo"

# Corte da execução original: OOT começa em 04/12/2025.
INICIO_OOT = datetime(2025, 12, 4, tzinfo=timezone.utc)

# Mesma limpeza básica mostrada no notebook de treinamento.
base_valida = df_treino_original.filter(
    F.col(COLUNA_DATA).isNotNull()
    & F.col("target").isNotNull()
)

df_treino_sem_oot = base_valida.filter(
    F.col(COLUNA_DATA) < F.lit(INICIO_OOT)
)

df_oot_conferencia = base_valida.filter(
    F.col(COLUNA_DATA) >= F.lit(INICIO_OOT)
)

resumo = (
    base_valida
    .withColumn(
        "conjunto",
        F.when(
            F.col(COLUNA_DATA) < F.lit(INICIO_OOT),
            F.lit("TREINO"),
        ).otherwise(F.lit("OOT")),
    )
    .groupBy("conjunto")
    .agg(
        F.count("*").alias("linhas"),
        F.min(COLUNA_DATA).alias("min_data"),
        F.max(COLUNA_DATA).alias("max_data"),
        F.avg(F.col("target").cast("double")).alias("taxa_positivos"),
    )
)

display(resumo)

contagens = {
    linha["conjunto"]: linha["linhas"]
    for linha in resumo.collect()
}

esperado = {"TREINO": 411_704, "OOT": 102_742}

if contagens != esperado:
    raise ValueError(
        "A origem não reproduziu as contagens da execução original. "
        f"Esperado: {esperado}. Encontrado: {contagens}. "
        "Não prepare a referência antes de verificar essa diferença."
    )

print("Recorte conferido: 411.704 registros de treino, sem OOT.")