from datetime import datetime
from zoneinfo import ZoneInfo

# Utiliza exclusivamente o conjunto original de treinamento.
df_treino = df_treino_sem_oot

# Confirma o recorte antes de criar a referência.
assert df_treino.count() == 411_704

bundle = carregar_bundle()
features = list(bundle["feature_cols"])

# Mantém o target observado e utiliza o modelo já treinado.
saida = escorar_dataframe(
    df=df_treino.toPandas(),
    bundle=bundle,
    safra=get_safra(),
    model_grc=CONTRACT["model_grc"],
    treino=True,
)

df_referencia = para_spark_big(saida, features)

# Define um novo destino sem substituir referências anteriores.
timestamp = datetime.now(
    ZoneInfo("America/Sao_Paulo")
).strftime("%Y%m%d_%H%M%S")

caminho_referencia = (
    "/Volumes/kraken_dasc/users/train_data/"
    f"propensao_egv_treino_{timestamp}"
)

df_referencia.write.mode("errorifexists").parquet(
    caminho_referencia
)

# Confirma a persistência.
df_conferencia = spark.read.parquet(caminho_referencia)

assert df_conferencia.count() == 411_704

print("Referência criada:", caminho_referencia)
df_conferencia.printSchema()