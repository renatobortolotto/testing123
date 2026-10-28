tabela = (
    "kraken_dasc.dvry_anly_cienciadado."
    "tbpropensaoegvapnp_kraken"
)

display(
    spark.sql(f"DESCRIBE HISTORY {tabela} LIMIT 5")
    .select(
        "version",
        "timestamp",
        "operation",
        "operationParameters",
        "job",
        "notebook",
    )
)