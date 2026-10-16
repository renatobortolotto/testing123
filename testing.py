from pyspark.sql import functions as F

fonte = spark.table(OP_CFG["fonte_publico"])

print("Fonte:", OP_CFG["fonte_publico"])
print("Modo:", OP_CFG["modo_scoring"])
print("Módulo:", OP_CFG["modulo_amostra"])

fonte.agg(
    F.count("*").alias("total_linhas"),
    F.count("cd_bv").alias("ids_validos"),
    F.countDistinct("cd_bv").alias("clientes_distintos")
).show()

fonte.select("cd_bv").show(10, truncate=False)