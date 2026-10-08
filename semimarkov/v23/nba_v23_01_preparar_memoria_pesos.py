# Databricks notebook source
# MAGIC %md
# MAGIC # NBA V2.3 — Parte 01: memória de contexto e pesos por cliente
# MAGIC
# MAGIC Execute em um **novo notebook**. Esta parte lê a base V2.2 clean já
# MAGIC preparada, conserva os mesmos eventos, durações, cortes e holdout, e
# MAGIC acrescenta memória e pesos. **Não treina nem muda o scoring existente.**
# MAGIC
# MAGIC Quatro variantes futuras usarão a mesma base:
# MAGIC `A_REFERENCIA`, `B_MEMORIA`, `C_PONDERACAO`, `D_MEMORIA_PONDERACAO`.
# MAGIC
# MAGIC Memória: último `funil` não técnico observado até o estado atual.
# MAGIC Ela atravessa as etapas técnicas e canais de atendimento, sem usar o
# MAGIC destino futuro. Ambiguidade ou lacuna na sequência zera a memória.
# MAGIC Na primeira candidata, somente origens técnicas terão parâmetros
# MAGIC condicionados à memória; os demais estados mantêm o modelo por origem.
# MAGIC
# MAGIC Pesos de treino: `peso_evento_treino = 1` e
# MAGIC `peso_cliente_treino = 1 / n_observacoes_cliente_origem`.
# MAGIC O denominador inclui observações exatas **e censuradas elegíveis**.
# MAGIC Assim cada cliente tem massa total 1 em cada origem. Os pesos são os
# MAGIC mesmos com/sem memória e não são recalculados por contexto.
# MAGIC
# MAGIC Não há balanceamento de destinos, nova amostragem, `sem_acao`, conversão
# MAGIC sintética, alteração do debounce, calendário ou sucesso como novo alvo.
# MAGIC A validação não participa do suporte nem do cálculo dos pesos de ajuste.

# COMMAND ----------

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


SM23_CFG = {
    "tabela_base_v22": (
        "ctg_dsti.renato_nba.nba_sm_v22_base_treino_clean_hml"
    ),
    "tabela_config_v22": (
        "ctg_dsti.renato_nba.nba_config_estados_v222_clean"
    ),
    "id_execucao_base": None,  # Preencher somente se a base tiver várias execuções.
    "fuso_esperado": "Etc/UTC",
    "min_eventos_contexto": 80,
    "min_clientes_contexto": 20,
    "gravar": True,
    "executar_autotestes_spark": True,
    "tabela_base_v23": "ctg_dsti.renato_nba.nba_sm_v23_base_memoria_pesos_hml",
    "tabela_suporte_v23": "ctg_dsti.renato_nba.nba_sm_v23_suporte_contexto_hml",
    "tabela_experimentos": "ctg_dsti.renato_nba.nba_sm_v23_experimentos_hml",
    "max_linhas_exibir": 40,
}

SM23_TECNICAS = frozenset({
    "app_login", "app_habilitacao_device", "app_primeiro_acesso",
    "app_reset_senha", "app_atualizacao_cadastral",
})
SM23_VARIANTES = [
    {"variante": "A_REFERENCIA", "memoria": False,
     "coluna_peso": "peso_evento_treino"},
    {"variante": "B_MEMORIA", "memoria": True,
     "coluna_peso": "peso_evento_treino"},
    {"variante": "C_PONDERACAO", "memoria": False,
     "coluna_peso": "peso_cliente_treino"},
    {"variante": "D_MEMORIA_PONDERACAO", "memoria": True,
     "coluna_peso": "peso_cliente_treino"},
]
SM23_SEM_MEMORIA = "__SEM_MEMORIA__"  # Categoria de contexto, NÃO evento.
SM23_BASE = "__BASE__"  # Origem que não utiliza contexto específico.
SM23_VERSAO_PREPARO = "v2.3_preparo_memoria_pesos_01"


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Funções sem dependência do notebook anterior
# MAGIC
# MAGIC `contexto_negocio` atualiza quando um funil não técnico é observado.
# MAGIC Atendimento não define um novo funil de negócio nem apaga o anterior.
# MAGIC Não confundimos `funil` com a whitelist de ações acionáveis: consultas
# MAGIC também podem ser contexto informativo.
# MAGIC
# MAGIC `contexto_modelo` usa essa memória apenas se a origem for técnica.
# MAGIC Se ela não estiver disponível, registra `__SEM_MEMORIA__`, sem inventar
# MAGIC um comportamento. O treino futuro usará o modelo de origem como suporte.

# COMMAND ----------


def sm23_tem_linhas(df: DataFrame) -> bool:
    return bool(df.limit(1).count())


def sm23_exigir_colunas(df: DataFrame, nomes: set[str], origem: str) -> None:
    faltantes = nomes - set(df.columns)
    if faltantes:
        raise ValueError(f"{origem}: colunas ausentes: {sorted(faltantes)}")


def sm23_identificador_sql(nome: str) -> str:
    partes = nome.split(".")
    if len(partes) != 3 or not all(partes) or any("`" in p for p in partes):
        raise ValueError("Use um nome completo: catalogo.schema.tabela.")
    return ".".join(f"`{p}`" for p in partes)


def sm23_snapshot_delta(sessao: SparkSession, tabela: str):
    """Fixa uma versão física Delta; não altera a fonte."""
    if not sessao.catalog.tableExists(tabela):
        raise ValueError(f"Tabela não encontrada: {tabela}")
    versao = int(sessao.sql(
        f"DESCRIBE HISTORY {sm23_identificador_sql(tabela)} LIMIT 1"
    ).first()["version"])
    df = sessao.read.option("versionAsOf", versao).table(tabela)
    return df, versao


def sm23_selecionar_execucao(df: DataFrame, identificador: str | None):
    if "id_execucao" not in df.columns:
        if identificador is not None:
            raise ValueError("A base não possui id_execucao para filtrar.")
        return df, None
    if identificador is not None:
        df = df.filter(F.col("id_execucao") == identificador)
    ids = df.select("id_execucao").distinct().limit(2).collect()
    if len(ids) != 1 or ids[0]["id_execucao"] is None:
        raise ValueError(
            "Selecione uma única id_execucao_base não nula em SM23_CFG."
        )
    return df, str(ids[0]["id_execucao"])


def sm23_atualizar_memoria(
    memoria: str | None, estado: str | None, mapa: dict,
    tecnicas: frozenset[str] = SM23_TECNICAS,
) -> str | None:
    """Regra de atualização que o futuro scoring também deverá aplicar."""
    if estado is None:
        return None
    if estado not in mapa:
        raise ValueError(f"Estado sem configuração: {estado}")
    funil = mapa[estado].get("funil")
    if funil and funil not in tecnicas:
        return funil
    return memoria


def sm23_adicionar_memoria(base: DataFrame, config: DataFrame) -> DataFrame:
    """Calcula contexto pelo prefixo da sequência, antes de filtrar o treino."""
    df = base.join(config, "estado", "left")
    df = df.withColumn(
        "barreira_memoria", F.col("estado").isNull()
        | F.coalesce(F.col("ordem_ambigua"), F.lit(False))
    ).withColumn(
        "origem_tecnica",
        F.coalesce(F.col("funil").isin(sorted(SM23_TECNICAS)), F.lit(False)),
    )
    janela = Window.partitionBy("cd_bv").orderBy("passo")
    df = df.withColumn("_passo_anterior", F.lag("passo").over(janela))
    df = df.withColumn("_barreira_anterior", F.lag("barreira_memoria").over(janela))
    df = df.withColumn(
        "_novo_trecho",
        F.when(
            F.col("_passo_anterior").isNull() | F.col("barreira_memoria")
            | F.coalesce(F.col("_barreira_anterior"), F.lit(False))
            | (F.col("passo") != F.col("_passo_anterior") + 1), 1
        ).otherwise(0),
    ).withColumn(
        "trecho_memoria",
        F.sum("_novo_trecho").over(janela.rowsBetween(Window.unboundedPreceding, 0)),
    )
    prefixo = (
        Window.partitionBy("cd_bv", "trecho_memoria").orderBy("passo")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    negocio_observado = (
        ~F.col("barreira_memoria") & F.col("funil").isNotNull()
        & ~F.col("origem_tecnica")
    )
    df = df.withColumn(
        "_evidencia_negocio",
        F.when(negocio_observado, F.struct(
            F.col("funil").alias("acao"),
            F.col("passo").alias("passo"),
            F.col("ts_estado").alias("ts"),
        )),
    ).withColumn(
        "_memoria", F.last("_evidencia_negocio", ignorenulls=True).over(prefixo)
    ).withColumn("contexto_negocio", F.col("_memoria.acao"))
    df = df.withColumn("passo_evidencia_memoria", F.col("_memoria.passo"))
    df = df.withColumn("ts_evidencia_memoria", F.col("_memoria.ts"))
    df = df.withColumn(
        "contexto_modelo",
        F.when(F.col("barreira_memoria"), F.lit(None).cast("string"))
        .when(F.col("origem_tecnica"), F.coalesce(
            F.col("contexto_negocio"), F.lit(SM23_SEM_MEMORIA)
        )).otherwise(F.lit(SM23_BASE)),
    )
    return df.drop(
        "_passo_anterior", "_barreira_anterior", "_novo_trecho",
        "_evidencia_negocio", "_memoria",
    )


def sm23_adicionar_pesos(base: DataFrame) -> DataFrame:
    """Massa total 1 por cliente-origem no treino, incluindo censuras."""
    treino = ~F.col("validacao_cliente") & F.col("elegivel_ajuste")
    janela = Window.partitionBy("cd_bv", "estado")
    df = base.withColumn(
        "_n_observacoes_treino",
        F.sum(F.when(treino, 1).otherwise(0)).over(janela),
    )
    return (
        df.withColumn(
            "n_observacoes_cliente_origem",
            F.when(treino, F.col("_n_observacoes_treino")),
        ).withColumn("peso_evento_treino", F.when(treino, F.lit(1.0)))
        .withColumn("peso_cliente_treino", F.when(
            treino & (F.col("_n_observacoes_treino") > 0),
            F.lit(1.0) / F.col("_n_observacoes_treino"),
        )).drop("_n_observacoes_treino")
    )


def sm23_suporte_contexto(base: DataFrame) -> DataFrame:
    treino = base.filter(~F.col("validacao_cliente") & F.col("elegivel_ajuste"))
    exata = F.col("tipo_censura") == "exata"
    suporte = treino.groupBy("estado", "contexto_modelo", "origem_tecnica").agg(
        F.count("*").alias("n_observacoes_treino"),
        F.sum(exata.cast("long")).alias("n_saidas_exatas"),
        F.sum((F.col("tipo_censura") == "direita").cast("long"))
        .alias("n_censuras_direita"),
        F.countDistinct("cd_bv").alias("n_clientes_treino"),
        F.countDistinct(F.when(exata, F.col("cd_bv"))).alias("n_clientes_exatas"),
        F.countDistinct(F.when(exata, F.col("destino"))).alias("n_destinos_exatos"),
        F.sum("peso_cliente_treino").alias("massa_pesos_cliente"),
    )
    return suporte.withColumn(
        "contexto_tem_suporte_candidato",
        F.col("origem_tecnica") & (F.col("contexto_modelo") != SM23_SEM_MEMORIA)
        & (F.col("n_saidas_exatas") >= SM23_CFG["min_eventos_contexto"])
        & (F.col("n_clientes_exatas") >= SM23_CFG["min_clientes_contexto"]),
    ).withColumn(
        "status_suporte",
        F.when(~F.col("origem_tecnica"), "MODELO_POR_ORIGEM")
        .when(F.col("contexto_modelo") == SM23_SEM_MEMORIA, "SEM_MEMORIA_OBSERVADA")
        .when(F.col("contexto_tem_suporte_candidato"), "CANDIDATO_CONTEXTUAL")
        .otherwise("CONTEXTO_RARO_USAR_SUPORTE_ORIGEM"),
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Autotestes pequenos, antes de consultar as tabelas reais
# MAGIC
# MAGIC Os testes Spark abaixo executam no seu cluster. Conferem propagação de
# MAGIC memória, barreiras, ausência de leitura do futuro, censura no denominador
# MAGIC e ausência de pesos de ajuste no holdout. Não criam tabelas.

# COMMAND ----------


def sm23_autotestes(sessao: SparkSession) -> None:
    cfg_schema = "estado string, funil string, etapa string, tipo_estado string"
    cfg = sessao.createDataFrame([
        ("boleto:::sucesso", "boleto", "sucesso", "funil_conclusao"),
        ("pix:::topo", "pix", "topo", "funil_abertura"),
        ("pix:::sucesso", "pix", "sucesso", "funil_conclusao"),
        ("app_login:::topo", "app_login", "topo", "funil_abertura"),
        ("app_login:::navegacao", "app_login", "navegacao", "funil_meio"),
        ("app_login:::sucesso", "app_login", "sucesso", "funil_conclusao"),
        ("atendimento:::chat", None, None, "atendimento"),
    ], cfg_schema)
    schema = (
        "cd_bv string, passo int, estado string, tipo_censura string, "
        "validacao_cliente boolean, elegivel_ajuste boolean, ordem_ambigua boolean"
    )
    registros = [
        ("A", 1, "boleto:::sucesso", "exata", False, True, False),
        ("A", 2, "app_login:::topo", "exata", False, True, False),
        ("A", 3, "app_login:::navegacao", "exata", False, True, False),
        ("A", 4, "app_login:::sucesso", "exata", False, True, False),
        ("A", 5, "pix:::topo", "exata", False, True, False),
        ("A", 6, "pix:::sucesso", "exata", False, True, False),
        ("A", 7, "app_login:::topo", "direita", False, True, False),
        ("B", 1, "app_login:::topo", "exata", True, True, False),
        ("B", 2, "pix:::sucesso", "direita", True, True, False),
        ("C", 1, "boleto:::sucesso", "exata", False, True, False),
        ("C", 2, None, "origem_ambigua", False, False, True),
        ("C", 3, "app_login:::topo", "direita", False, True, False),
        ("D", 1, "boleto:::sucesso", "exata", False, True, False),
        ("D", 3, "app_login:::topo", "direita", False, True, False),
        ("E", 1, "boleto:::sucesso", "exata", False, True, False),
        ("E", 2, "atendimento:::chat", "exata", False, True, False),
        ("E", 3, "app_login:::topo", "direita", False, True, False),
    ]
    base = sessao.createDataFrame(registros, schema).withColumn(
        "ts_estado", F.expr("timestamp_micros(1700000000000000 + passo * 1000000)")
    )
    completo = sm23_adicionar_pesos(sm23_adicionar_memoria(base, cfg))
    dados = {(r.cd_bv, r.passo): r.asDict() for r in completo.collect()}
    for passo in (2, 3, 4):
        assert dados[("A", passo)]["contexto_modelo"] == "boleto"
    assert dados[("A", 7)]["contexto_modelo"] == "pix"
    assert dados[("A", 5)]["contexto_modelo"] == SM23_BASE
    for chave in (("B", 1), ("C", 3), ("D", 3)):
        assert dados[chave]["contexto_modelo"] == SM23_SEM_MEMORIA
    assert dados[("E", 3)]["contexto_modelo"] == "boleto"
    assert dados[("C", 2)]["contexto_negocio"] is None
    for passo in (2, 7):
        assert math.isclose(dados[("A", passo)]["peso_cliente_treino"], 0.5)
    assert dados[("B", 1)]["peso_cliente_treino"] is None
    assert dados[("C", 2)]["peso_evento_treino"] is None
    prefixo = sm23_adicionar_memoria(base.filter("passo <= 4"), cfg)
    for r in prefixo.collect():
        assert r.contexto_negocio == dados[(r.cd_bv, r.passo)]["contexto_negocio"]
    print("V2.3 — autotestes Spark: memória, pesos e censura OK.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Congelar as fontes e validar o contrato
# MAGIC
# MAGIC O notebook não lê `customers_query`: elegibilidade operacional não define
# MAGIC o histórico deste experimento. O holdout já persistido é conservado.
# MAGIC Se houver versões/IDs conflitantes, a execução para em vez de escolher
# MAGIC arbitrariamente ou eliminar duplicatas.

# COMMAND ----------

if "spark" not in globals():
    raise RuntimeError("Execute no Databricks com uma sessão Spark.")
if spark.conf.get("spark.sql.session.timeZone") != SM23_CFG["fuso_esperado"]:
    raise ValueError("Fuso inesperado. Não altere timestamps para contornar a validação.")
if SM23_CFG["executar_autotestes_spark"]:
    sm23_autotestes(spark)

sm23_base_origem, SM23_VERSAO_DELTA_BASE = sm23_snapshot_delta(
    spark, SM23_CFG["tabela_base_v22"]
)
sm23_config_origem, SM23_VERSAO_DELTA_CONFIG = sm23_snapshot_delta(
    spark, SM23_CFG["tabela_config_v22"]
)
sm23_base_origem, SM23_ID_BASE = sm23_selecionar_execucao(
    sm23_base_origem, SM23_CFG["id_execucao_base"]
)
SM23_ID_EXPERIMENTO = str(uuid.uuid4())

sm23_exigir_colunas(sm23_base_origem, {
    "cd_bv", "passo", "ts_estado", "estado", "destino", "dur_min",
    "tipo_censura", "elegivel_ajuste", "validacao_cliente", "ordem_ambigua",
    "ts_corte_estado", "relogio",
}, "base V2.2")
sm23_exigir_colunas(sm23_config_origem, {
    "estado", "funil", "etapa", "tipo_estado",
}, "configuração clean")
for coluna in ("elegivel_ajuste", "validacao_cliente", "ordem_ambigua"):
    if sm23_base_origem.schema[coluna].dataType.simpleString() != "boolean":
        raise ValueError(f"{coluna} precisa ser boolean.")
for coluna in ("ts_estado", "ts_corte_estado"):
    if sm23_base_origem.schema[coluna].dataType.simpleString() != "timestamp":
        raise ValueError(f"{coluna} precisa ser timestamp.")
if {"contexto_negocio", "peso_cliente_treino", "funil"} & set(sm23_base_origem.columns):
    raise ValueError("A origem deve ser a base V2.2, não uma base V2.3 já enriquecida.")

sm23_config = sm23_config_origem.select(
    "estado", "funil", "etapa", "tipo_estado"
).distinct()
if sm23_tem_linhas(sm23_config.filter(F.col("estado").isNull())):
    raise ValueError("Configuração com estado nulo.")
if sm23_tem_linhas(sm23_config.groupBy("estado").count().filter("count > 1")):
    raise ValueError("Configuração com definições conflitantes para um estado.")
if sm23_tem_linhas(sm23_base_origem.filter(
    F.col("cd_bv").isNull() | F.col("passo").isNull()
    | F.col("ts_estado").isNull() | F.col("validacao_cliente").isNull()
    | F.col("elegivel_ajuste").isNull() | F.col("ordem_ambigua").isNull()
    | F.col("ts_corte_estado").isNull()
)):
    raise ValueError("Identificador, ordem, timestamp, corte ou flags nulos.")
if sm23_tem_linhas(sm23_config.filter(
    F.col("funil").isin(SM23_SEM_MEMORIA, SM23_BASE)
)):
    raise ValueError("Nome de funil colide com categoria reservada de contexto.")
if not sm23_tem_linhas(sm23_base_origem):
    raise ValueError("Base V2.2 vazia.")
if sm23_tem_linhas(sm23_base_origem.groupBy("cd_bv", "passo").count().filter("count > 1")):
    raise ValueError("Cliente/passo repetido. Não deduplicar silenciosamente.")
if sm23_tem_linhas(sm23_base_origem.groupBy("cd_bv").agg(
    F.countDistinct("validacao_cliente").alias("n_splits")
).filter("n_splits != 1")):
    raise ValueError("Cliente presente em mais de um split.")
if sm23_base_origem.select("validacao_cliente").distinct().count() != 2:
    raise ValueError("Esperados treino e holdout por cliente na fonte.")
if sm23_tem_linhas(sm23_base_origem.filter(
    F.col("relogio").isNull() | (F.col("relogio") != "ULTIMA_ACAO_TEMPO_V22")
)):
    raise ValueError("Relógio diferente de última ação real + tempo.")

for coluna in ("estado", "destino"):
    artificial = F.coalesce(F.lower(F.col(coluna)).rlike(
        r"(^sem_acao:::|:::conversao$|^perdido(?::::|$))"
    ), F.lit(False))
    if sm23_tem_linhas(sm23_base_origem.filter(artificial)):
        raise ValueError(f"Fonte não clean: estado sintético/terminal em {coluna}.")
    estados = sm23_base_origem.select(F.col(coluna).alias("estado")).where(
        F.col("estado").isNotNull()
    ).distinct()
    if sm23_tem_linhas(estados.join(sm23_config.select("estado"), "estado", "left_anti")):
        raise ValueError(f"{coluna}: há estados sem configuração. Atualize a config clean.")

sm23_checar_ordem = sm23_base_origem.withColumn(
    "_ts_prev", F.lag("ts_estado").over(Window.partitionBy("cd_bv").orderBy("passo"))
)
if sm23_tem_linhas(sm23_checar_ordem.filter(F.col("ts_estado") <= F.col("_ts_prev"))):
    raise ValueError("Timestamps repetidos/invertidos na sequência preparada.")
if sm23_tem_linhas(sm23_base_origem.filter(
    F.col("elegivel_ajuste") & (
        F.col("estado").isNull() | F.col("ordem_ambigua")
        | ~F.col("tipo_censura").isin("exata", "direita")
        | F.col("tipo_censura").isNull()
        | F.col("dur_min").isNull() | F.isnan("dur_min")
        | (F.col("dur_min") <= 0) | (F.abs(F.col("dur_min")) == float("inf"))
        | ((F.col("tipo_censura") == "exata") & F.col("destino").isNull())
        | ((F.col("tipo_censura") == "direita") & F.col("destino").isNotNull())
    )
)):
    raise ValueError("Linha elegível incompatível com o ajuste Semi-Markov.")

sm23_cortes = sm23_base_origem.select("ts_corte_estado").distinct().limit(2).collect()
if len(sm23_cortes) != 1:
    raise ValueError("O experimento exige um único corte comportamental na base.")
SM23_CORTE_ESTADO = sm23_cortes[0]["ts_corte_estado"]
if sm23_tem_linhas(sm23_base_origem.filter(F.col("ts_estado") >= F.col("ts_corte_estado"))):
    raise ValueError("Evento no corte exclusivo ou posterior ao corte.")
SM23_N_BASE = sm23_base_origem.count()
print("Experimento V2.3:", SM23_ID_EXPERIMENTO)
print("Execução-base V2.2:", SM23_ID_BASE)
print("Corte histórico preservado:", SM23_CORTE_ESTADO)
print("Versões Delta fixadas:", SM23_VERSAO_DELTA_BASE, SM23_VERSAO_DELTA_CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Preparar memória e pesos — sem treinar
# MAGIC
# MAGIC As flags de suporte são estimadas somente no treino. Contextos raros
# MAGIC continuam na base; não são excluídos. Esta parte não atribui uma matriz
# MAGIC contextual aos contextos raros nem aplica um bônus ao score.

# COMMAND ----------

sm23_base = sm23_adicionar_pesos(
    sm23_adicionar_memoria(sm23_base_origem, sm23_config)
).persist(StorageLevel.MEMORY_AND_DISK)
sm23_suporte = sm23_suporte_contexto(sm23_base).persist(StorageLevel.MEMORY_AND_DISK)
if sm23_base.count() != SM23_N_BASE:
    raise RuntimeError("O enriquecimento alterou o número de linhas da base.")
if sm23_tem_linhas(sm23_base.filter(
    (F.col("passo_evidencia_memoria") > F.col("passo"))
    | (F.col("ts_evidencia_memoria") > F.col("ts_estado"))
)):
    raise RuntimeError("Memória utilizou informação posterior ao estado.")
if sm23_tem_linhas(sm23_base.filter(
    (F.col("validacao_cliente") | ~F.col("elegivel_ajuste"))
    & (F.col("peso_evento_treino").isNotNull() | F.col("peso_cliente_treino").isNotNull())
)):
    raise RuntimeError("Pesos de treino atribuídos ao holdout/linhas inelegíveis.")

sm23_massa_cliente = sm23_base.filter(
    ~F.col("validacao_cliente") & F.col("elegivel_ajuste")
).groupBy("cd_bv", "estado").agg(
    F.sum("peso_cliente_treino").alias("massa_cliente_origem")
)
if sm23_tem_linhas(sm23_massa_cliente.filter(
    F.abs(F.col("massa_cliente_origem") - 1.0) > 1e-8
)):
    raise RuntimeError("Os pesos não somam 1 por cliente-origem.")

SM23_RESULTADOS = {}


def sm23_mostrar(nome: str, df: DataFrame) -> None:
    SM23_RESULTADOS[nome] = df
    print(f"\n{nome}")
    df.show(SM23_CFG["max_linhas_exibir"], truncate=False)


sm23_mostrar("V23_01_INVENTARIO", sm23_base.groupBy("validacao_cliente").agg(
    F.count("*").alias("n_linhas"), F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum(F.col("elegivel_ajuste").cast("long")).alias("n_elegiveis"),
    F.sum(F.col("barreira_memoria").cast("long")).alias("n_barreiras"),
    F.min("ts_estado").alias("primeiro_evento"),
    F.max("ts_estado").alias("ultimo_evento"),
).orderBy("validacao_cliente"))
sm23_mostrar("V23_02_COBERTURA_MEMORIA", sm23_base.filter(
    F.col("elegivel_ajuste") & F.col("origem_tecnica")
).groupBy("validacao_cliente", "estado").agg(
    F.count("*").alias("n_observacoes"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum(F.col("contexto_negocio").isNotNull().cast("long")).alias("n_com_memoria"),
    F.countDistinct("contexto_negocio").alias("n_contextos"),
).withColumn("cobertura_observacoes", F.col("n_com_memoria") / F.col("n_observacoes"))
 .orderBy("validacao_cliente", F.desc("n_observacoes")))
sm23_mostrar("V23_03_SUPORTE_CONTEXTOS", sm23_suporte.filter(
    F.col("origem_tecnica")
).orderBy(F.desc("n_saidas_exatas")))
sm23_mostrar("V23_04_AUDITORIA_PESOS", sm23_massa_cliente.agg(
    F.count("*").alias("n_pares_cliente_origem"),
    F.min("massa_cliente_origem").alias("min_massa_cliente_origem"),
    F.max("massa_cliente_origem").alias("max_massa_cliente_origem"),
    F.max(F.abs(F.col("massa_cliente_origem") - 1.0)).alias("maior_erro_soma"),
))
sm23_mostrar("V23_05_PESOS_POR_ORIGEM", sm23_base.filter(
    ~F.col("validacao_cliente") & F.col("elegivel_ajuste")
).groupBy("estado").agg(
    F.count("*").alias("n_observacoes"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.sum("peso_cliente_treino").alias("soma_pesos_cliente"),
    F.expr("percentile_approx(peso_cliente_treino, array(0.1, 0.5, 0.9))")
    .alias("p10_p50_p90_peso"),
).orderBy(F.desc("n_observacoes")))
sm23_mostrar("V23_06_CONTEXTO_HOLDOUT", sm23_base.filter(
    F.col("validacao_cliente") & F.col("elegivel_ajuste") & F.col("origem_tecnica")
).join(sm23_suporte.select(
    "estado", "contexto_modelo", "status_suporte"
), ["estado", "contexto_modelo"], "left")
 .withColumn("status_suporte", F.coalesce("status_suporte", F.lit("NAO_VISTO_NO_TREINO")))
 .groupBy("status_suporte").agg(
     F.count("*").alias("n_observacoes"),
     F.countDistinct("cd_bv").alias("n_clientes"),
 ).orderBy(F.desc("n_observacoes")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persistência isolada por experimento
# MAGIC
# MAGIC Somente novas tabelas V2.3. Escrita em append, com `id_experimento`.
# MAGIC A linha `CONCLUIDO_PREPARO` só é registrada depois das escritas e checagens.
# MAGIC As tabelas juntas não formam uma transação: a próxima parte deve consumir
# MAGIC apenas o ID com esse status, nunca inferir que tabelas existentes bastam.
# MAGIC O manifesto guarda versões das fontes, política de memória e pesos.

# COMMAND ----------

sm23_politica = {
    "versao": SM23_VERSAO_PREPARO,
    "acoes_tecnicas": sorted(SM23_TECNICAS),
    "memoria": "ultimo_funil_nao_tecnico_observado_no_prefixo",
    "atendimento": "preserva_memoria_mas_nao_recebe_modelo_contextual",
    "barreiras": ["origem_ambigua", "lacuna_de_passo"],
    "origens_contextualizadas": "somente_acoes_tecnicas",
    "peso": "1/n_observacoes_elegiveis_cliente_origem_incluindo_censuras",
    "nova_amostragem": False,
    "variantes_planejadas": SM23_VARIANTES,
    "min_eventos_contexto": SM23_CFG["min_eventos_contexto"],
    "min_clientes_contexto": SM23_CFG["min_clientes_contexto"],
}
SM23_POLITICA_JSON = json.dumps(sm23_politica, sort_keys=True, ensure_ascii=False)
SM23_POLITICA_HASH = hashlib.sha256(SM23_POLITICA_JSON.encode()).hexdigest()
sm23_meta = {
    "id_experimento": SM23_ID_EXPERIMENTO,
    "status": "CONCLUIDO_PREPARO",
    "versao_preparo": SM23_VERSAO_PREPARO,
    "politica_json": SM23_POLITICA_JSON,
    "politica_sha256": SM23_POLITICA_HASH,
    "tabela_base_v22": SM23_CFG["tabela_base_v22"],
    "versao_delta_base_v22": SM23_VERSAO_DELTA_BASE,
    "id_execucao_base_v22": SM23_ID_BASE,
    "tabela_config_v22": SM23_CFG["tabela_config_v22"],
    "versao_delta_config_v22": SM23_VERSAO_DELTA_CONFIG,
    "tabela_base_v23": SM23_CFG["tabela_base_v23"],
    "tabela_suporte_v23": SM23_CFG["tabela_suporte_v23"],
    "corte_estado_iso": SM23_CORTE_ESTADO.isoformat(),
    "n_linhas_base": SM23_N_BASE,
    "fuso_base": SM23_CFG["fuso_esperado"],
    "registrado_em_utc": datetime.now(timezone.utc).isoformat(),
}


def sm23_com_experimento(df: DataFrame) -> DataFrame:
    return (df.withColumn("id_experimento", F.lit(SM23_ID_EXPERIMENTO))
            .withColumn("versao_preparo", F.lit(SM23_VERSAO_PREPARO))
            .withColumn("politica_sha256", F.lit(SM23_POLITICA_HASH)))


if SM23_CFG["gravar"]:
    for chave in ("tabela_base_v23", "tabela_suporte_v23", "tabela_experimentos"):
        tabela = SM23_CFG[chave]
        if tabela in {SM23_CFG["tabela_base_v22"], SM23_CFG["tabela_config_v22"]}:
            raise ValueError("Destino de escrita não pode ser uma fonte V2.2.")
        if spark.catalog.tableExists(tabela):
            sm23_exigir_colunas(spark.table(tabela), {"id_experimento"}, tabela)
            if sm23_tem_linhas(spark.table(tabela).filter(
                F.col("id_experimento") == SM23_ID_EXPERIMENTO
            )):
                raise ValueError("ID já escrito. Inicie uma nova execução da Parte 01.")
    for df, chave in ((sm23_base, "tabela_base_v23"), (sm23_suporte, "tabela_suporte_v23")):
        (sm23_com_experimento(df).write.format("delta").mode("append")
         .partitionBy("id_experimento").saveAsTable(SM23_CFG[chave]))
    sm23_n_persistido = spark.table(SM23_CFG["tabela_base_v23"]).filter(
        F.col("id_experimento") == SM23_ID_EXPERIMENTO
    ).count()
    if sm23_n_persistido != SM23_N_BASE:
        raise RuntimeError("Contagem persistida diverge. Manifesto não será concluído.")
    manifesto_schema = (
        "id_experimento string, status string, versao_preparo string, "
        "politica_json string, politica_sha256 string, tabela_base_v22 string, "
        "versao_delta_base_v22 long, id_execucao_base_v22 string, "
        "tabela_config_v22 string, versao_delta_config_v22 long, "
        "tabela_base_v23 string, tabela_suporte_v23 string, corte_estado_iso string, "
        "n_linhas_base long, fuso_base string, registrado_em_utc string"
    )
    (spark.createDataFrame([sm23_meta], manifesto_schema).write.format("delta")
     .mode("append").saveAsTable(SM23_CFG["tabela_experimentos"]))
    print("\nV2.3 PREPARO PERSISTIDO. ID:", SM23_ID_EXPERIMENTO)
    print("BASE:", SM23_CFG["tabela_base_v23"])
    print("SUPORTE:", SM23_CFG["tabela_suporte_v23"])
    print("MANIFESTO:", SM23_CFG["tabela_experimentos"])
else:
    print("\nV2.3 preparada somente em memória; nenhum experimento persistido.")

print("\nNenhum modelo foi treinado, promovido ou sobrescrito nesta Parte 01.")
print("Próxima parte: ajuste temporal das quatro variantes e validação pareada.")
