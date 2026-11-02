# Preencha com o ID real exibido na página do experimento.
EXPERIMENTO_ID = "COLE_AQUI_O_ID_DO_EXPERIMENTO"

mlflow.set_tracking_uri(CFG["tracking_uri"])

EXPERIMENTO_ID = EXPERIMENTO_ID.strip()

if not EXPERIMENTO_ID.isdecimal():
    raise ValueError(
        "Informe o ID numérico real do experimento MLflow. "
        "Não utilize o nome do experimento, o run_id ou 'None'."
    )

# Confirma que o experimento pode ser consultado neste workspace.
experimento = mlflow.get_experiment(EXPERIMENTO_ID)

if experimento is None:
    raise RuntimeError(
        f"Experimento {EXPERIMENTO_ID} não encontrado."
    )

if experimento.lifecycle_stage != "active":
    raise RuntimeError(
        f"O experimento {EXPERIMENTO_ID} não está ativo: "
        f"{experimento.lifecycle_stage}."
    )

# Registra na configuração o destino efetivamente utilizado.
CFG["experimento_mlflow"] = experimento.name
CFG["experimento_mlflow_id"] = experimento.experiment_id

print("Tracking URI:", mlflow.get_tracking_uri())
print("Experimento:", experimento.name)
print("Experiment ID:", experimento.experiment_id)

START = datetime.now(timezone.utc)
corte_treino = timestamp_utc(CFG["corte_historico_exclusivo"])

nome_run = (
    f"v23_B2__{START:%Y%m%d_%H%M%S_UTC}"
    f"__corte_{corte_treino:%Y%m%d}"
    f"__mod{CFG['modulo_amostra_clientes']}"
)

RUN = mlflow.start_run(
    experiment_id=experimento.experiment_id,
    run_name=nome_run,
)
RUN_ID = RUN.info.run_id