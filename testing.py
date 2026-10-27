import joblib

CAMINHO_BUNDLE = (
    "/Workspace/Users/renato.netto@bv.com.br/"
    "first-test/EGV/egv_model_bundle_latest.joblib"
)

bundle = joblib.load(CAMINHO_BUNDLE)

validar_bundle(bundle)

features = list(bundle["feature_cols"])

print("Bundle carregado com sucesso.")
print(f"Quantidade de features: {len(features)}")
print(f"Quantidade de cortes GH: {len(bundle['splits'])}")
print(f"Calibrador: {bundle['calibrator'] is not None}")

assert len(features) == 63
assert len(bundle["splits"]) == 9