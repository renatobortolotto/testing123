# Databricks notebook source
# NBA MVP | Etapa 2: modelo de frequencias + top 5 candidato.
# Executar no MESMO notebook/sessao da etapa 1.
# Nao grava tabelas permanentes, nao treina tempos e nao altera a etapa 1.
#
# CONTRATO DA PROBABILIDADE NESTE PROTOTIPO:
# prob_proxima_acao = frequencia do destino entre as saidas com origem
# e destino identificados. Nao redistribui a massa somente do top 5.
# Excluir destinos ambiguos condiciona a estimativa a observacoes
# identificaveis; cobertura_destino_identificado_origem explicita isso.
# Ainda NAO e uma probabilidade calibrada sobre todas as proximas acoes.

from datetime import timedelta

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

VIEW_BASE = "nba_base_passo_v1_amostra"
VIEW_ATUAIS = "nba_estados_atuais_v1_amostra"
FONTE_EVENTOS = "base_passo_raw"
TOP_K = 5
DIAS_DIAGNOSTICO = 7
SAL_SPLIT = "nba_mvp_v1_split_clientes"
MODULO_SPLIT = 5  # Aproximadamente 20% dos clientes para validacao.
RESTO_VALIDACAO = 0
TOLERANCIA_PROB = 1e-9


def total_condicao(condicao: Column) -> Column:
    """Conta uma condicao, retornando zero em uma entrada vazia."""
    return F.coalesce(
        F.sum(F.when(condicao, 1).otherwise(0)), F.lit(0)
    )


def razao_segura(numerador: Column, denominador: Column) -> Column:
    """Retorna NULL quando nao existe denominador positivo."""
    return F.when(denominador > 0, numerador.cast("double") / denominador)


def exigir_colunas(df: DataFrame, colunas: set[str], nome: str) -> None:
    faltantes = colunas - set(df.columns)
    if faltantes:
        raise ValueError(f"{nome}: colunas ausentes: {sorted(faltantes)}")


base_piloto = spark.table(VIEW_BASE)
atuais_piloto = spark.table(VIEW_ATUAIS)
exigir_colunas(
    base_piloto,
    {
        "cd_bv", "estado", "destino", "status_transicao",
        "elegivel_transicao", "data_referencia", "ts_corte",
        "fuso_referencia",
    },
    VIEW_BASE,
)
exigir_colunas(
    atuais_piloto,
    {"cd_bv", "estado", "ordem_ambigua", "data_referencia", "ts_corte"},
    VIEW_ATUAIS,
)

metadados = (
    base_piloto.select("data_referencia", "ts_corte", "fuso_referencia")
    .distinct().limit(2).collect()  # Apenas metadados, nao registros de clientes.
)
if len(metadados) != 1:
    raise ValueError("A etapa 2 exige uma unica referencia e base nao vazia.")

meta = metadados[0]
data_ref_modelo = meta["data_referencia"]
ts_corte_modelo = meta["ts_corte"]
fuso_modelo = meta["fuso_referencia"]
if any(item is None for item in (data_ref_modelo, ts_corte_modelo, fuso_modelo)):
    raise ValueError("Metadados temporais incompletos.")
if spark.conf.get("spark.sql.session.timeZone") != fuso_modelo:
    raise ValueError("O fuso mudou desde a etapa 1. Nao continue sem revisa-lo.")

VERSAO_MODELO = (
    f"mvp_v1_{data_ref_modelo.isoformat()}_amostra_frequencia_identificada"
)

# COMMAND ----------

# 1. Atualidade: consulta a FONTE antes da selecao de clientes da etapa 1.
# Contar registros em D-1 e um diagnostico, nao atesta carga completa.
# Mesmo quando existem eventos, cobertura e semantica do fuso sao pendencias.
fonte_eventos = spark.table(FONTE_EVENTOS)
exigir_colunas(fonte_eventos, {"dm_navegacao"}, FONTE_EVENTOS)
primeiro_dia = data_ref_modelo - timedelta(days=DIAS_DIAGNOSTICO - 1)

eventos_recentes = fonte_eventos.filter(
    (F.col("dm_navegacao") >= F.lit(primeiro_dia).cast("timestamp"))
    & (F.col("dm_navegacao") < F.lit(ts_corte_modelo))
)
por_dia = (
    eventos_recentes
    .groupBy(F.to_date("dm_navegacao").alias("dia"))
    .agg(
        F.count("*").alias("n_eventos_fonte"),
        F.min("dm_navegacao").alias("primeiro_evento_fonte"),
        F.max("dm_navegacao").alias("ultimo_evento_fonte"),
    )
)
calendario = spark.createDataFrame(
    [(primeiro_dia + timedelta(days=i),) for i in range(DIAS_DIAGNOSTICO)],
    "dia date",
)
cobertura_fonte = (
    calendario.join(por_dia, "dia", "left")
    .fillna({"n_eventos_fonte": 0})
    .orderBy("dia")
    .cache()  # Apenas sete linhas agregadas.
)
cobertura_fonte.createOrReplaceTempView("nba_cobertura_fonte_v1")
print("COBERTURA DA FONTE, ANTES DA AMOSTRAGEM")
cobertura_fonte.show(truncate=False)

n_eventos_referencia = (
    cobertura_fonte.filter(F.col("dia") == F.lit(data_ref_modelo))
    .select("n_eventos_fonte").first()[0]
)
STATUS_DADOS = (
    "SEM_EVENTOS_D1_NA_FONTE"
    if n_eventos_referencia == 0
    else "COBERTURA_E_FUSO_NAO_CONFIRMADOS"
)
print(f"STATUS DOS DADOS: {STATUS_DADOS}")
print("Existencia de eventos nao confirma completude nem corrige o fuso.")
print("ESTADOS ATUAIS DA AMOSTRA")
atuais_piloto.groupBy("estado").count().orderBy(F.desc("count")).show(
    15, truncate=False
)

# COMMAND ----------

# 2. Separa CLIENTES; todos os passos de um cliente ficam no mesmo grupo.
# O sal e diferente da selecao da etapa 1. Nao se reutiliza hash % 1000.
base_split = base_piloto.withColumn(
    "validacao_cliente",
    F.pmod(
        F.xxhash64("cd_bv", F.lit(SAL_SPLIT)), F.lit(MODULO_SPLIT)
    ) == RESTO_VALIDACAO,
)
base_split.groupBy("validacao_cliente").agg(
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.count("*").alias("n_passos"),
    total_condicao(F.col("elegivel_transicao")).alias("n_elegiveis"),
).show(truncate=False)

observada = (
    (F.col("status_transicao") == "observada")
    & F.col("elegivel_transicao")
    & F.col("estado").isNotNull()
    & F.col("destino").isNotNull()
)

treino = base_split.filter(~F.col("validacao_cliente"))
validacao = base_split.filter(F.col("validacao_cliente"))
treino_observado = treino.filter(observada)

suporte_origem = (
    treino.filter(F.col("estado").isNotNull())
    .groupBy(F.col("estado").alias("origem"))
    .agg(
        total_condicao(observada).alias("n_saidas_identificadas"),
        total_condicao(
            F.col("status_transicao") == "destino_ambiguo"
        ).alias("n_destinos_ambiguos"),
        F.countDistinct(
            F.when(observada, F.col("cd_bv"))
        ).alias("n_clientes_saidas_identificadas"),
    )
    .withColumn(
        "cobertura_destino_identificado_origem",
        razao_segura(
            F.col("n_saidas_identificadas"),
            F.col("n_saidas_identificadas") + F.col("n_destinos_ambiguos"),
        ),
    )
)
suporte_origem.createOrReplaceTempView("nba_suporte_origem_v1_amostra")

modelo_transicoes = (
    treino_observado
    .groupBy(F.col("estado").alias("origem"), "destino")
    .agg(
        F.count("*").alias("n_transicoes"),
        F.countDistinct("cd_bv").alias("n_clientes_par"),
    )
    .join(suporte_origem, "origem", "inner")
    .withColumn(
        "prob_proxima_acao",
        F.col("n_transicoes").cast("double") / F.col("n_saidas_identificadas"),
    )
    .withColumn("versao_modelo", F.lit(VERSAO_MODELO))
    .cache()  # Tabela de pares agregados, nao a base inteira.
)
if not modelo_transicoes.limit(1).count():
    raise ValueError("Nenhuma transicao identificada no conjunto de treino.")

invalida = (
    F.col("prob_proxima_acao").isNull()
    | F.isnan("prob_proxima_acao")
    | ~F.col("prob_proxima_acao").between(0.0, 1.0)
)
if modelo_transicoes.filter(invalida).limit(1).count():
    raise ValueError("O modelo contem probabilidades invalidas.")

somas = modelo_transicoes.groupBy("origem").agg(
    F.sum("prob_proxima_acao").alias("soma_probabilidade")
)
if somas.filter(
    F.abs(F.col("soma_probabilidade") - 1.0) > TOLERANCIA_PROB
).limit(1).count():
    raise ValueError("As probabilidades nao somam 1 por origem identificada.")

# Empates de probabilidade sao resolvidos por nome apenas para exibir o ranking.
# Nao implica diferenca de probabilidade entre os destinos empatados.
janela_ranking = Window.partitionBy("origem").orderBy(
    F.desc("prob_proxima_acao"), F.asc("destino")
)
modelo_top5 = (
    modelo_transicoes
    .withColumn("ranking", F.row_number().over(janela_ranking))
    .filter(F.col("ranking") <= TOP_K)
    .withColumn(
        "massa_probabilidade_top5",
        F.sum("prob_proxima_acao").over(Window.partitionBy("origem")),
    )
    .cache()
)
modelo_transicoes.createOrReplaceTempView("nba_modelo_transicoes_v1_amostra")
modelo_top5.createOrReplaceTempView("nba_modelo_top5_v1_amostra")

print("MODELO DA AMOSTRA")
modelo_transicoes.agg(
    F.countDistinct("origem").alias("n_origens_estimadas"),
    F.count("*").alias("n_pares"),
    F.sum("n_transicoes").alias("n_transicoes_treino"),
).show(vertical=True, truncate=False)

# COMMAND ----------

# 3. Avaliacao fora do treino, POR CLIENTES, sobre destinos identificados.
# Acertos ponderados pelo numero de transicoes, nao pela contagem de pares.
# NAO e teste temporal, nem acuracia em passos com destino ambiguo/censurado.
pares_validacao = (
    validacao.filter(observada)
    .groupBy(F.col("estado").alias("origem"), "destino")
    .agg(F.count("*").alias("n"))
)
origens_treino = modelo_transicoes.select("origem").distinct().withColumn(
    "tem_modelo", F.lit(True)
)
resultado_validacao = (
    pares_validacao
    .join(F.broadcast(origens_treino), "origem", "left")
    .join(
        F.broadcast(modelo_top5.select("origem", "destino", "ranking")),
        ["origem", "destino"],
        "left",
    )
)
validacao_modelo = resultado_validacao.agg(
    F.coalesce(F.sum("n"), F.lit(0)).alias("n_transicoes_avaliadas"),
    F.coalesce(F.sum(
        F.when(F.col("tem_modelo"), F.col("n")).otherwise(0)
    ), F.lit(0)).alias("n_com_origem_modelada"),
    F.coalesce(F.sum(
        F.when(F.col("ranking") == 1, F.col("n")).otherwise(0)
    ), F.lit(0)).alias("n_acertos_top1"),
    F.coalesce(F.sum(
        F.when(F.col("ranking").isNotNull(), F.col("n")).otherwise(0)
    ), F.lit(0)).alias("n_acertos_top5"),
)
for nome, numerador in [
    ("cobertura_origem_modelada", "n_com_origem_modelada"),
    ("acerto_top1", "n_acertos_top1"),
    ("acerto_top5", "n_acertos_top5"),
]:
    validacao_modelo = validacao_modelo.withColumn(
        nome,
        razao_segura(F.col(numerador), F.col("n_transicoes_avaliadas")),
    )
validacao_modelo.createOrReplaceTempView("nba_validacao_modelo_v1_amostra")
print("VALIDACAO: METRICAS RESTRITAS A TRANSICOES IDENTIFICADAS")
validacao_modelo.show(vertical=True, truncate=False)
print("COBERTURA DA BASE DE VALIDACAO")
validacao.groupBy("status_transicao").count().show(truncate=False)

# COMMAND ----------

# 4. Aplicacao do modelo de treino a todos os estados atuais da amostra.
# A aplicacao nao e a avaliacao: o destino futuro desses clientes nao e conhecido.
# Mantem clientes sem previsao com ranking/probabilidade NULL e status explicito.
if atuais_piloto.groupBy("cd_bv").count().filter(
    F.col("count") != 1
).limit(1).count():
    raise ValueError("Mais de um estado atual por cliente.")
if atuais_piloto.filter(
    F.col("data_referencia").isNull()
    | (F.col("data_referencia") != F.lit(data_ref_modelo))
    | F.col("ts_corte").isNull()
    | (F.col("ts_corte") != F.lit(ts_corte_modelo))
).limit(1).count():
    raise ValueError("O corte dos estados atuais difere do corte da base.")

ranking_saida = modelo_top5.select(
    F.col("origem").alias("acao_atual"),
    "ranking",
    F.col("destino").alias("proxima_acao"),
    "prob_proxima_acao", "n_transicoes", "n_clientes_par",
    "n_saidas_identificadas", "n_destinos_ambiguos",
    "n_clientes_saidas_identificadas",
    "cobertura_destino_identificado_origem", "massa_probabilidade_top5",
)
previsoes_top5 = (
    atuais_piloto
    .select(
        "cd_bv", "data_referencia", "ts_corte",
        F.col("estado").alias("acao_atual"), "ordem_ambigua",
    )
    .join(F.broadcast(ranking_saida), "acao_atual", "left")
    .withColumn(
        "status_previsao",
        F.when(
            F.col("acao_atual").isNull() | F.col("ordem_ambigua"),
            "ESTADO_ATUAL_AMBIGUO",
        )
        .when(F.col("ranking").isNull(), "SEM_TRANSICAO_ESTIMADA")
        .otherwise("CANDIDATA"),
    )
    .withColumn("prob_proxima_acao_7d", F.lit(None).cast("double"))
    .withColumn("status_temporal", F.lit("NAO_AJUSTADO"))
    .withColumn("status_dados", F.lit(STATUS_DADOS))
    .withColumn(
        "condicao_probabilidade", F.lit("DESTINO_IDENTIFICADO_NA_BASE")
    )
    .withColumn("publicavel", F.lit(False))
    .withColumn("fuso_referencia", F.lit(fuso_modelo))
    .withColumn("versao_modelo", F.lit(VERSAO_MODELO))
    .drop("ordem_ambigua")
    .select(
        "data_referencia", "cd_bv", "acao_atual", "ranking",
        "proxima_acao", "prob_proxima_acao", "prob_proxima_acao_7d",
        "status_previsao", "status_temporal", "status_dados", "publicavel",
        "n_transicoes", "n_clientes_par", "n_saidas_identificadas",
        "n_destinos_ambiguos", "n_clientes_saidas_identificadas",
        "cobertura_destino_identificado_origem", "massa_probabilidade_top5",
        "condicao_probabilidade", "ts_corte", "fuso_referencia", "versao_modelo",
    )
    .cache()
)

if previsoes_top5.groupBy("cd_bv", "ranking").count().filter(
    F.col("count") > 1
).limit(1).count():
    raise ValueError("Duplicidade de cliente/ranking no resultado.")

resumo_output = previsoes_top5.agg(
    F.countDistinct("cd_bv").alias("n_clientes_output"),
    F.countDistinct(F.when(
        F.col("ranking").isNotNull(), F.col("cd_bv")
    )).alias("n_clientes_com_previsao"),
    F.countDistinct(F.when(
        F.col("ranking").isNull(), F.col("cd_bv")
    )).alias("n_clientes_sem_previsao"),
    F.countDistinct(F.when(
        F.col("ranking") == TOP_K, F.col("cd_bv")
    )).alias("n_clientes_com_cinco_alternativas"),
    total_condicao(F.col("ranking").isNotNull()).alias("n_linhas_previsao"),
    total_condicao(F.col("prob_proxima_acao_7d").isNotNull()).alias(
        "n_linhas_com_prob_7d"
    ),
)

previsoes_top5.createOrReplaceTempView("nba_previsoes_top5_v1_amostra")
resumo_output.createOrReplaceTempView("nba_resumo_output_v1_amostra")
print("RESUMO DO OUTPUT CANDIDATO")
resumo_output.show(vertical=True, truncate=False)
print("SUPORTE DO RANKING DO SILENCIO")
modelo_top5.filter(F.col("origem") == "sem_acao:::classe").select(
    "origem", "destino", "ranking", "prob_proxima_acao",
    "n_transicoes", "n_saidas_identificadas", "n_destinos_ambiguos",
    "cobertura_destino_identificado_origem",
).orderBy("ranking").show(truncate=False)

print("Output: nba_previsoes_top5_v1_amostra. Top 1: filtrar ranking = 1.")
print("Somente views temporarias: nenhuma tabela permanente foi alterada.")
print("Probabilidades condicionadas a destinos identificados; nao ha calibracao.")
print("Nenhum estado/destino foi ocultado para renormalizar o top 5.")
print("Cobertura de D-1, fuso e ambiguidades ainda impedem publicacao.")

# Para INSPECIONAR localmente, sem enviar identificadores de clientes ao chat:
# display(spark.table("nba_previsoes_top5_v1_amostra").limit(20))
