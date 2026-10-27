print("Run atual:", CONTRACT["mlflow"][0]["run_id"])

run_candidata = "eca97cbaffae49cea3458a6ec4856692"

bundle_teste = _carregar_bundle_run(run_candidata)

print("Bundle carregado da run:", run_candidata)
print("Features:", len(bundle_teste["feature_cols"]))
print("GHs:", len(bundle_teste["splits"]) + 1)