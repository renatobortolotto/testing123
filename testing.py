from mlflow.artifacts import load_dict

run_id = CONTRACT["mlflow"][0]["run_id"]

metadata = load_dict(
    f"runs:/{run_id}/propensao/deploy_metadata.json"
)

caminho_registrado = metadata["path_to_data_train"]

print("Caminho atual no contrato:")
print(CONTRACT["path_to_data_train"])

print("\nCaminho registrado na run:")
print(caminho_registrado)

# Teste mínimo de leitura da referência, sem gravar ou sobrescrever dados.
df_referencia = spark.read.parquet(caminho_registrado)

if not df_referencia.take(1):
    raise ValueError("A referência de treinamento está vazia.")

df_referencia.printSchema()

# Atualiza somente depois que a leitura tiver funcionado.
CONTRACT["path_to_data_train"] = caminho_registrado

print("\nLeitura realizada. Use este caminho no DEPLOY_CONTRACT:")
print(caminho_registrado)

pasta = "/Volumes/kraken_dasc/users/train_data/"

for item in dbutils.fs.ls(pasta):
    if item.name.startswith("propensao_egv_"):
        print(item.path)


assert "preparar_artefatos_deploy" not in globals(), (
    "A função de preparação ainda está carregada nesta sessão."
)

resultado_validacao = validar_contrato()