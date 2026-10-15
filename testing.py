from pyspark.sql import functions as F

# IDs do treinamento executado anteriormente neste mesmo notebook.
if "ID_EXPERIMENTO" not in globals() or "ID_AJUSTE" not in globals():
    raise RuntimeError(
        "IDs da Parte 02 não encontrados nesta sessão. "
        "Preencha AUD_CFG com os IDs exibidos em V23_26_CONCLUSAO."
    )

AUD_CFG.update({
    "id_experimento": str(ID_EXPERIMENTO),
    "id_ajuste": str(ID_AJUSTE),
})

ajuste_concluido = (
    spark.table(AUD_CFG["tabela_ajustes"])
    .filter(
        (F.col("id_experimento") == AUD_CFG["id_experimento"])
        & (F.col("id_ajuste") == AUD_CFG["id_ajuste"])
        & (F.col("status") == "CONCLUIDO_TREINO_VALIDACAO")
    )
)

if ajuste_concluido.limit(2).count() != 1:
    raise RuntimeError(
        "Não foi encontrado um único manifesto concluído para esse ajuste. "
        "Confira o output final da Parte 02 antes de continuar."
    )

print("Experimento que será auditado:", AUD_CFG["id_experimento"])
print("Ajuste que será auditado:", AUD_CFG["id_ajuste"])


def aud_ler(tabela, versao=None):
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela não encontrada: {tabela}")

    if versao is None:
        versao = int(
            spark.sql(
                f"DESCRIBE HISTORY {aud_nome_sql(tabela)} LIMIT 1"
            ).first()["version"]
        )

    versao = int(versao)
    AUD_FONTES[tabela] = versao

    print(f"02B — lendo {tabela} | versão Delta: {versao}", flush=True)

    df = spark.read.option("versionAsOf", versao).table(tabela)

    # Executa uma leitura pequena para localizar a falha nesta fonte.
    df.limit(1).collect()

    return df