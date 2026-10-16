from pyspark.sql import functions as F

PREFIXO = "ctg_dsti.renato_nba.nba_sm_v23_operacional"

CD_BV_TESTE = "2fa51926-49a5-4e02-b5be-ebe3768fd819"

publico = spark.table(f"{PREFIXO}_publico_hml")
estados = spark.table(f"{PREFIXO}_estados_hml")
negocio = spark.table(f"{PREFIXO}_negocio_hml")

# Identifica execuções presentes nas três tabelas.
exec_publico = publico.select("id_execucao").distinct()
exec_estados = estados.select("id_execucao").distinct()
exec_negocio = negocio.select("id_execucao").distinct()

exec_comuns = (
    exec_publico
    .join(exec_estados, "id_execucao", "inner")
    .join(exec_negocio, "id_execucao", "inner")
)

# Prioriza a execução mais recente pela data registrada.
exec_recentes = (
    publico
    .select("id_execucao", "data_snapshot_publico")
    .join(exec_comuns, "id_execucao", "inner")
    .groupBy("id_execucao")
    .agg(F.max("data_snapshot_publico").alias("data_snapshot"))
    .orderBy(
        F.col("data_snapshot").desc_nulls_last(),
        F.col("id_execucao").desc()
    )
    .limit(1)
    .collect()
)

if not exec_recentes:
    raise RuntimeError(
        "Nenhuma execução comum encontrada nas três tabelas."
    )

ID_EXECUCAO = exec_recentes[0]["id_execucao"]

# Mantém apenas os dados da execução selecionada.
publico = publico.filter(F.col("id_execucao") == ID_EXECUCAO)
estados = estados.filter(F.col("id_execucao") == ID_EXECUCAO)
negocio = negocio.filter(F.col("id_execucao") == ID_EXECUCAO)

print(f"Execução selecionada: {ID_EXECUCAO}")

for nome, df in (
    ("publico", publico),
    ("estados", estados),
    ("negocio", negocio),
):
    print(f"{nome}: {df.count()} registros")