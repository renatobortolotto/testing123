from pyspark.sql import functions as F

display(
    df.groupBy("fase", "TesteDigito").agg(
        F.count("*").alias("total_clientes"),
        F.sum("converteu").alias("total_convertidos"),
        F.sum("converteu_mesma_proposta").alias("mesma_proposta"),
        F.sum("converteu_nova_proposta").alias("nova_proposta"),
        F.sum("conversao_recuperada").alias("recuperados"),
    )
)