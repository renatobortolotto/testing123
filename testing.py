from pyspark.sql import functions as F


TABELA_MODELOS_V22 = (
    "ctg_dsti.renato_nba.nba_sm_v22_modelos_hml"
)

TABELA_VALIDACAO_V22 = (
    "ctg_dsti.renato_nba.nba_sm_v22_validacao_hml"
)

TABELA_PREVISOES_V22 = (
    "ctg_dsti.renato_nba.nba_sm_v22_previsoes_hml"
)

TABELA_BASE_V22 = (
    "ctg_dsti.renato_nba.nba_sm_v22_base_treino_hml"
)


def adicionar_metadados(df):
    return (
        df
        .withColumn(
            "id_execucao",
            F.lit(SM22_ID_EXECUCAO),
        )
        .withColumn(
            "versao_modelo",
            F.lit(SM22_CFG["versao_modelo"]),
        )
        .withColumn(
            "gravado_em",
            F.current_timestamp(),
        )
    )


# 1. Modelo ajustado
(
    adicionar_metadados(sm22_modelos)
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABELA_MODELOS_V22)
)


# 2. Validacao
(
    adicionar_metadados(sm22_validacao)
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABELA_VALIDACAO_V22)
)


# 3. Scoring da amostra
(
    adicionar_metadados(sm22_previsoes)
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABELA_PREVISOES_V22)
)


# 4. Base preparada
# Vale salvar porque evita ter que refazer toda a Parte 01.
(
    adicionar_metadados(base_treino_v22)
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(TABELA_BASE_V22)
)


print("Artefatos V2.2 persistidos.")

print(TABELA_MODELOS_V22)
print(TABELA_VALIDACAO_V22)
print(TABELA_PREVISOES_V22)
print(TABELA_BASE_V22)