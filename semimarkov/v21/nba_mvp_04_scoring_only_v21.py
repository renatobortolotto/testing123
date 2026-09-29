# Databricks notebook source
# NBA | Etapa 04 V2.1: scoring-only usando o modelo ja ajustado.
#
# Executar no MESMO notebook depois das etapas 01/02/02D/03.
# Nao retreina o modelo. Apenas recalcula o publico, estado + tempo e Top 5.
#
# Para testar um cliente:
#   MODO_SCORING = "TESTE_CD_BV"
#   CD_BV_TESTE = "..."
#
# Para pontuar 100% da customers_query:
#   MODO_SCORING = "TODOS"
#
# O output fica na view nba_sm_v21_scoring_only.

from pyspark import StorageLevel
from pyspark.sql import Window
from pyspark.sql import functions as F


# =========================
# CONFIGURACAO
# =========================

MODO_SCORING = "TESTE_CD_BV"  # "TESTE_CD_BV" ou "TODOS"
CD_BV_TESTE = None

# Mantem o contrato oficial: por padrao, o teste precisa estar no publico.
EXIGIR_CLIENTE_NO_PUBLICO = True

FONTE_PUBLICO = SM_CFG["fonte_publico"]
FONTE_APLICACAO = SM_CFG["fonte_aplicacao"]
VIEW_SCORING = "nba_sm_v21_scoring_only"

# Opcional. Primeiro valide sem gravar.
GRAVAR_SCORING = False
TABELA_SCORING = (
    "ctg_dsti.renato_nba.nba_semimarkov_scoring_v21_hml"
)


# =========================
# VALIDACOES DE CONTEXTO
# =========================

_requeridos = [
    "SM_CFG",
    "SM_VIEWS",
    "SMD_BROADCAST",
    "SMD_VERSAO",
    "SMD_HORIZONTE",
    "SMD_SCHEMA_OUTPUT",
    "smd_prever_lotes",
    "sm_ler_eventos",
    "sm_preparar_passos",
    "sm_segundos",
    "sm_ha",
]

_faltantes = [nome for nome in _requeridos if nome not in globals()]
if _faltantes:
    raise RuntimeError(
        "Execute 01/02/02D/03 antes desta etapa. "
        f"Ausentes: {_faltantes}"
    )

if MODO_SCORING not in {"TESTE_CD_BV", "TODOS"}:
    raise ValueError("MODO_SCORING invalido.")


# =========================
# 1. PUBLICO SEM AMOSTRAGEM
# =========================

publico_raw = spark.table(FONTE_PUBLICO)

if "cd_bv" not in publico_raw.columns:
    raise ValueError(f"{FONTE_PUBLICO}: coluna cd_bv ausente.")

publico_base = (
    publico_raw
    .select(F.col("cd_bv").cast("string"))
    .filter(F.col("cd_bv").isNotNull())
    .distinct()
)

if MODO_SCORING == "TESTE_CD_BV":
    if CD_BV_TESTE is None or not str(CD_BV_TESTE).strip():
        raise ValueError("Preencha CD_BV_TESTE antes de executar.")

    cd_bv_teste = str(CD_BV_TESTE)
    esta_no_publico = sm_ha(
        publico_base.filter(F.col("cd_bv") == cd_bv_teste)
    )
    print("Cliente esta na customers_query:", esta_no_publico)

    if EXIGIR_CLIENTE_NO_PUBLICO and not esta_no_publico:
        raise ValueError(
            "O cd_bv nao esta na customers_query. Para um teste de QA "
            "fora do publico, altere EXIGIR_CLIENTE_NO_PUBLICO=False."
        )

    publico_scoring = spark.createDataFrame(
        [(cd_bv_teste,)],
        "cd_bv string",
    )
else:
    # 100% do publico. Nenhum modulo/hash e aplicado aqui.
    publico_scoring = publico_base

publico_scoring = (
    publico_scoring
    .withColumn(
        "data_referencia",
        F.lit(SM_CFG["data_publico"]).cast("date"),
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

if not sm_ha(publico_scoring):
    raise ValueError("Publico de scoring vazio.")

print(
    "Clientes no scoring:",
    publico_scoring.count(),
    "| modo:",
    MODO_SCORING,
)


# =========================
# 2. ESTADO + TEMPO NO ESTADO
# =========================

# Mesma preparacao usada no treino V2.1; apenas troca o publico.
eventos_scoring = (
    sm_ler_eventos(
        FONTE_APLICACAO,
        None,
        SM_CFG["corte_estado_exclusivo"],
    )
    .join(
        publico_scoring.select("cd_bv"),
        "cd_bv",
        "left_semi",
    )
)

passos_scoring = sm_preparar_passos(
    eventos_scoring,
    SM_CFG["corte_estado_exclusivo"],
    "scoring_only",
)

janela_ultimo = (
    Window.partitionBy("cd_bv")
    .orderBy(F.desc("passo"))
)

ultimo = (
    passos_scoring
    .withColumn(
        "_ultima",
        F.row_number().over(janela_ultimo),
    )
    .filter(F.col("_ultima") == 1)
    .select(
        "cd_bv",
        "estado",
        "ts_inicio",
        "ts_ultima_atividade",
        "ultima_acao_observada",
        "ordem_ambigua",
        "inicio_observado",
    )
)

atuais_scoring = (
    publico_scoring
    .join(ultimo, "cd_bv", "left")
    .withColumn(
        "ts_corte_estado",
        F.lit(SM_CFG["corte_estado_exclusivo"]).cast("timestamp"),
    )
    .withColumn("acao_atual", F.col("estado"))
    .withColumn(
        "tempo_no_estado_seg",
        F.when(
            F.col("inicio_observado"),
            sm_segundos("ts_inicio", "ts_corte_estado"),
        ),
    )
    .withColumn(
        "status_input",
        F.when(F.col("ts_inicio").isNull(), "SEM_HISTORICO")
        .when(F.col("ordem_ambigua"), "ESTADO_ATUAL_AMBIGUO")
        .when(~F.col("inicio_observado"), "IDADE_ESTADO_DESCONHECIDA")
        .otherwise("OK"),
    )
    .withColumn(
        "status_dados",
        F.lit("SCORING_ONLY_HOMOLOGACAO"),
    )
    .withColumn("relogio", F.lit(SM_CFG["relogio"]))
    .persist(StorageLevel.MEMORY_AND_DISK)
)

atuais_scoring.groupBy("status_input", "acao_atual").count().orderBy(
    F.desc("count")
).show(50, truncate=False)


# =========================
# 3. APLICAR MODELO SALVO EM MEMORIA
# =========================

input_scoring = atuais_scoring.select(
    "cd_bv",
    "data_referencia",
    "ts_corte_estado",
    "acao_atual",
    "tempo_no_estado_seg",
    "status_input",
    "status_dados",
    "relogio",
)

previsoes_scoring = (
    input_scoring
    .mapInPandas(
        smd_prever_lotes,
        schema=SMD_SCHEMA_OUTPUT,
    )
    .localCheckpoint(eager=True)
)

previsoes_scoring.createOrReplaceTempView(VIEW_SCORING)

resumo = previsoes_scoring.agg(
    F.count("*").alias("n_linhas"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.countDistinct(
        F.when(F.col("ranking") == 1, F.col("cd_bv"))
    ).alias("n_clientes_previstos"),
).first().asDict()

print("Resumo scoring-only:", resumo)

previsoes_scoring.groupBy(
    "status_previsao",
    "status_temporal",
).agg(
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.count("*").alias("n_linhas"),
).orderBy(F.desc("n_clientes")).show(50, truncate=False)

if MODO_SCORING == "TESTE_CD_BV":
    print("Top 5 do cliente de teste:")
    previsoes_scoring.orderBy("ranking").show(20, truncate=False)


# =========================
# 4. GRAVACAO OPCIONAL
# =========================

if GRAVAR_SCORING:
    if not sm_ha(previsoes_scoring):
        raise ValueError("Output vazio: gravacao recusada.")

    output = (
        previsoes_scoring
        .withColumn("modo_scoring", F.lit(MODO_SCORING))
        .withColumn("gravado_em", F.current_timestamp())
    )

    predicado = (
        f"data_referencia = DATE '{SM_CFG['data_publico']}' "
        f"AND versao_modelo = '{SMD_VERSAO}'"
    )

    if spark.catalog.tableExists(TABELA_SCORING):
        (
            output.write.format("delta")
            .mode("overwrite")
            .option("replaceWhere", predicado)
            .saveAsTable(TABELA_SCORING)
        )
    else:
        (
            output.write.format("delta")
            .mode("errorifexists")
            .saveAsTable(TABELA_SCORING)
        )

    print("Scoring gravado em:", TABELA_SCORING)
else:
    print(
        "Nenhuma tabela gravada. Primeiro valide o teste; "
        "depois use MODO_SCORING='TODOS'."
    )

# Consultas:
# %sql
# SELECT * FROM nba_sm_v21_scoring_only
# WHERE ranking = 1
# ORDER BY cd_bv;
#
# %sql
# SELECT * FROM nba_sm_v21_scoring_only
# WHERE ranking IS NOT NULL
# ORDER BY cd_bv, ranking;
