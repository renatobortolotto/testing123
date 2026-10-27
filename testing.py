# Origem dos dados: independente do caminho final utilizado pelo Kraken.
CAMINHO_TREINO_ORIGINAL = (
    "/Volumes/ctg_dsti/renato-egv/egv-apnp/"
    "part-00000-tid-6645282771968014706-"
    "39a03543-922e-4df2-95c8-6bd9f0c1e9b7-26-1-c000.snappy.parquet"
)

df_treino_original = spark.read.parquet(CAMINHO_TREINO_ORIGINAL)

df_treino_original.printSchema()
display(df_treino_original.limit(5))