# Adicione esta célula DEPOIS de PRED_05_CONCLUSAO.
# Lê somente a saída persistida: não executa novamente o modelo.

from pyspark.sql import functions as F
from pyspark.sql.types import DateType

TABELA_FINAL = "ctg_dsti.renato_nba.nba_v23_scoring_final"
ID_PUBLICACAO = str(SCORE_ID)

# 1. Aceitar somente uma execução concluída da população completa.
registros = (
    spark.table(TABELAS["execucoes"])
    .filter(F.col("id_execucao") == ID_PUBLICACAO)
    .limit(2)
    .collect()
)

if len(registros) != 1:
    raise RuntimeError("Esperado um único manifesto para esta execução.")

execucao = registros[0].asDict()

if execucao["status"] != "CONCLUIDO_SCORING_B2":
    raise RuntimeError("O scoring ainda não foi concluído.")

if execucao["modo_scoring"] != "TODOS":
    raise RuntimeError("A tabela final recebe apenas o modo TODOS.")

# 2. Preservar os scores e a data de referência já calculados.
df_final = (
    spark.table(TABELAS["negocio"])
    .filter(F.col("id_execucao") == ID_PUBLICACAO)
    .select(
        "data_referencia",
        "cd_bv",
        "acao_atual",
        "ranking",
        "proxima_acao_raw",
        "score_raw",
        "proxima_acao_util",
        "score_util",
        "id_execucao",
        "run_id_treinamento",
        "model_uri",
        "modo_scoring",
    )
    .withColumn(
        "ts_corte_eventos",
        F.expr(f"timestamp_micros({int(execucao['corte_micros'])})"),
    )
    .withColumn(
        "ts_execucao",
        F.lit(execucao["inicio_utc"]).cast("timestamp"),
    )
    .withColumn(
        "ts_conclusao_scoring",
        F.lit(execucao["fim_utc"]).cast("timestamp"),
    )
)

if not isinstance(df_final.schema["data_referencia"].dataType, DateType):
    raise TypeError("data_referencia deve ser DATE, como no scoring original.")

invalido = (
    F.col("data_referencia").isNull()
    | F.col("cd_bv").isNull()
    | F.col("ranking").isNull()
    | ~F.col("ranking").between(1, 5)
    | ~F.col("data_referencia").eqNullSafe(F.to_date("ts_corte_eventos"))
    | ~F.col("run_id_treinamento").eqNullSafe(
        F.lit(execucao["run_id_treinamento"])
    )
    | ~F.col("model_uri").eqNullSafe(F.lit(execucao["model_uri"]))
    | ~F.col("modo_scoring").eqNullSafe(F.lit("TODOS"))
)

resumo = df_final.agg(
    F.count("*").alias("n_linhas"),
    F.min("data_referencia").alias("data_referencia"),
    F.countDistinct("data_referencia").alias("n_datas"),
    F.sum(F.when(invalido, 1).otherwise(0)).alias("n_invalidas"),
).first()

n_esperado = int(execucao["n_linhas_negocio"])

if n_esperado <= 0 or resumo["n_linhas"] != n_esperado:
    raise RuntimeError("Saída vazia ou contagem diferente do manifesto.")

if resumo["n_datas"] != 1 or resumo["n_invalidas"] != 0:
    raise RuntimeError("Há datas, chaves ou metadados incompatíveis na saída.")

# O scoring já verifica a unicidade de cd_bv + ranking por execução.
# Não aplicamos dropDuplicates: preservamos integralmente o resultado.

# 3. Impedir que a repetição desta célula duplique a mesma execução.
ja_publicado = False

if spark.catalog.tableExists(TABELA_FINAL):
    detalhe = spark.sql(
        f"DESCRIBE DETAIL {TABELA_FINAL}"
    ).first().asDict()

    if detalhe["format"].lower() != "delta":
        raise RuntimeError("A tabela de destino precisa ser Delta.")

    if detalhe["partitionColumns"] != ["data_referencia"]:
        raise RuntimeError(
            "O destino existente não está particionado por data_referencia."
        )

    n_existente = (
        spark.table(TABELA_FINAL)
        .filter(
            (F.col("data_referencia") == F.lit(resumo["data_referencia"]))
            & (F.col("id_execucao") == ID_PUBLICACAO)
        )
        .count()
    )

    if n_existente not in (0, n_esperado):
        raise RuntimeError(
            "Execução já existe com contagem divergente; revisar destino."
        )

    ja_publicado = n_existente == n_esperado

if ja_publicado:
    print("Execução já publicada:", ID_PUBLICACAO)
    print("Nenhuma linha acrescentada.")
else:
    (
        df_final.write.format("delta")
        .mode("append")
        .partitionBy("data_referencia")
        # Um identificador de transação por lote, para repetir a gravação.
        .option("txnAppId", f"{TABELA_FINAL}:{ID_PUBLICACAO}")
        .option("txnVersion", 0)
        .saveAsTable(TABELA_FINAL)
    )

    print("Publicação concluída:", TABELA_FINAL)
    print("Data de referência:", resumo["data_referencia"])
    print("Execução:", ID_PUBLICACAO, "| linhas:", n_esperado)