import scipy
from scipy import stats


SCHEMA = "ctg_dsti.renato_nba"

TABELAS_ORIGEM = {
    "eventos_jornada": f"{SCHEMA}.base_jornadas_onboarding",
    "base_passo_atual": f"{SCHEMA}.nba_sample",
    "config_estados": f"{SCHEMA}.nba_config_sample",
}

TABELAS_DESTINO = {
    "base_passo": f"{SCHEMA}.nba_base_passo_v1",
    "modelo_transicoes": f"{SCHEMA}.nba_modelo_transicoes_v1",
    "previsoes": f"{SCHEMA}.nba_previsoes_top5_v1",
}

print(f"Spark: {spark.version}")
print(f"SciPy: {scipy.__version__}")
print(
    "Fuso da sessão:",
    spark.conf.get("spark.sql.session.timeZone"),
)
print("CensoredData disponível:", hasattr(stats, "CensoredData"))

for descricao, tabela in TABELAS_ORIGEM.items():
    print(f"\nFonte: {descricao}")
    print(f"Tabela: {tabela}")
    spark.table(tabela).printSchema()

try:
    spark.table("base_passo_raw").printSchema()
except Exception as erro:
    print(
        "\nA view temporária base_passo_raw não pôde ser lida:",
        type(erro).__name__,
    )