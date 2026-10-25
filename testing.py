bibliotecas = [
    f"{pacote}=={version(pacote)}"
    for pacote in [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "joblib",
    ]
]
bibliotecas.append(f"mlflow=={mlflow.__version__}")

sufixo = datetime.now(FUSO).strftime("%Y%m%d_%H%M%S_%f")

novo_destino = (
    "/Volumes/kraken_dasc/users/train_data/"
    f"propensao_egv_referencia_{sufixo}"
)