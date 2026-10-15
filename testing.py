import math
import re

TABELA = (
    "ctg_dsti.renato_nba."
    "nba_sm_v23_base_memoria_pesos_hml"
)

RETENCAO_MINIMA_DIAS = 30

propriedades = {
    row["key"]: row["value"]
    for row in spark.sql(f"SHOW TBLPROPERTIES {TABELA}").collect()
}


def intervalo_em_dias(valor: str) -> float:
    """Interpreta intervalos simples sem reduzir uma retenção existente."""
    resultado = re.fullmatch(
        r"\s*interval\s+(\d+(?:\.\d+)?)\s+(weeks?|days?|hours?)\s*",
        valor,
        flags=re.IGNORECASE,
    )
    if resultado is None:
        raise ValueError(
            f"Formato de retenção não reconhecido: {valor!r}. "
            "Nenhuma propriedade foi alterada."
        )

    quantidade, unidade = resultado.groups()
    fator = {"week": 7, "day": 1, "hour": 1 / 24}
    return float(quantidade) * fator[unidade.lower().rstrip("s")]


dados_atuais = intervalo_em_dias(
    propriedades.get("delta.deletedFileRetentionDuration", "interval 7 days")
)
logs_atuais = intervalo_em_dias(
    propriedades.get("delta.logRetentionDuration", "interval 30 days")
)

dias_dados = math.ceil(max(RETENCAO_MINIMA_DIAS, dados_atuais))
dias_logs = math.ceil(max(dias_dados, logs_atuais))

# Altera apenas propriedades; não sobrescreve os registros da tabela.
spark.sql(f"""
    ALTER TABLE {TABELA}
    SET TBLPROPERTIES (
        'delta.deletedFileRetentionDuration' = 'interval {dias_dados} days',
        'delta.logRetentionDuration' = 'interval {dias_logs} days'
    )
""")

print(f"Retenção configurada: dados={dias_dados} dias; logs={dias_logs} dias.")


from pyspark.sql import functions as F

teste_base_v1 = (
    spark.read
    .option("versionAsOf", 1)
    .table(TABELA)
    .filter(
        F.col("id_experimento")
        == "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72"
    )
)

amostra = teste_base_v1.select(
    "cd_bv", "passo", "estado", "ts_estado"
).limit(1).collect()

if not amostra:
    raise RuntimeError("O experimento não foi encontrado na versão 1.")

print("Leitura inicial da versão 1 concluída.")