# Databricks notebook source
# NBA MVP | Etapa 3: persistir SOMENTE a homologacao da amostra.
# Executar no mesmo notebook/sessao das etapas 1 e 2, APOS os testes.
# ESCRITAS: cria/atualiza apenas as tres tabelas _hml em TABELAS_HML.
# Nao altera fontes, nao altera a referencia, nao treina nem libera automacao.
# Um job por vez: as tres escritas nao constituem uma transacao unica.
# O registro de execucao so fica CONCLUIDA apos as duas tabelas serem conferidas.

import json
import re
from datetime import datetime, timezone
from uuid import uuid4

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

GRAVAR_HOMOLOGACAO = True  # False: somente verificar, sem escrever no catalogo.
SCHEMA_HML = "ctg_dsti.renato_nba"
TABELAS_HML = {
    "modelo": f"{SCHEMA_HML}.nba_mvp_modelo_transicoes_hml",
    "previsoes": f"{SCHEMA_HML}.nba_mvp_previsoes_top5_hml",
    "execucoes": f"{SCHEMA_HML}.nba_mvp_execucoes_hml",
}
VIEWS = {
    "modelo": "nba_modelo_transicoes_v1_amostra",
    "previsoes": "nba_previsoes_top5_v1_amostra",
    "atuais": "nba_estados_atuais_v1_amostra",
    "validacao": "nba_validacao_modelo_v1_amostra",
    "cobertura": "nba_cobertura_fonte_v1",
}
TOLERANCIA = 1e-9
ID_EXECUCAO = uuid4().hex
INICIO_EXECUCAO = datetime.now(timezone.utc)


def exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise ValueError(mensagem)


def exigir_colunas(df: DataFrame, nomes: set[str], origem: str) -> None:
    faltantes = nomes - set(df.columns)
    exigir(not faltantes, f"{origem}: colunas ausentes: {sorted(faltantes)}")


def ha_linhas(df: DataFrame) -> bool:
    return bool(df.limit(1).count())


for tabela in TABELAS_HML.values():
    exigir(
        bool(re.fullmatch(r"[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*", tabela))
        and tabela.endswith("_hml"),
        f"Destino fora do padrao de homologacao: {tabela}",
    )

entradas = {nome: spark.table(view) for nome, view in VIEWS.items()}
modelo = entradas["modelo"]
previsoes = entradas["previsoes"]
atuais = entradas["atuais"]

exigir_colunas(
    modelo,
    {"origem", "destino", "prob_proxima_acao", "versao_modelo", "n_transicoes"},
    VIEWS["modelo"],
)
exigir_colunas(
    previsoes,
    {
        "data_referencia", "versao_modelo", "ts_corte", "fuso_referencia",
        "status_dados", "cd_bv", "acao_atual", "ranking", "proxima_acao",
        "prob_proxima_acao", "prob_proxima_acao_7d", "publicavel",
    },
    VIEWS["previsoes"],
)
exigir_colunas(
    atuais,
    {
        "data_referencia", "cd_bv", "ts_entrada_min", "ts_entrada_max",
        "ts_ultima_atividade", "ts_corte",
    },
    VIEWS["atuais"],
)

metas = previsoes.select(
    "data_referencia", "versao_modelo", "ts_corte", "fuso_referencia",
    "status_dados",
).distinct().limit(2).collect()  # Somente metadados, nao clientes.
exigir(len(metas) == 1, "O output deve conter uma unica referencia/versao.")
meta = metas[0].asDict()
exigir(all(valor is not None for valor in meta.values()), "Metadados incompletos.")
exigir(
    spark.conf.get("spark.sql.session.timeZone") == meta["fuso_referencia"]
    and meta["fuso_referencia"] in {"UTC", "Etc/UTC"},
    "Fuso diferente do diagnostico anterior; revisar antes de continuar.",
)
DATA_REF = meta["data_referencia"].isoformat()
VERSAO = meta["versao_modelo"]
exigir(bool(re.fullmatch(r"[A-Za-z0-9_-]+", VERSAO)), "Versao de modelo invalida.")
PREDICADO = f"data_referencia = '{DATA_REF}' AND versao_modelo = '{VERSAO}'"

# COMMAND ----------

# Verificacoes antes de QUALQUER escrita permanente.
exigir(ha_linhas(modelo), "O modelo esta vazio.")
exigir(
    not ha_linhas(modelo.filter(
        F.col("versao_modelo").isNull() | (F.col("versao_modelo") != VERSAO)
    )),
    "As versoes do modelo e do output sao diferentes.",
)
exigir(
    not ha_linhas(modelo.groupBy("origem", "destino").count().filter("count > 1")),
    "O modelo possui pares duplicados.",
)
prob_invalida = (
    F.col("prob_proxima_acao").isNull()
    | F.isnan("prob_proxima_acao")
    | ~F.col("prob_proxima_acao").between(0.0, 1.0)
)
exigir(not ha_linhas(modelo.filter(prob_invalida)), "Probabilidade invalida no modelo.")
somas = modelo.groupBy("origem").agg(F.sum("prob_proxima_acao").alias("soma"))
exigir(
    not ha_linhas(somas.filter(F.abs(F.col("soma") - 1.0) > TOLERANCIA)),
    "As probabilidades nao somam um por origem.",
)
exigir(
    not ha_linhas(previsoes.filter(F.col("cd_bv").isNull())),
    "Cliente nulo no output.",
)
exigir(
    not ha_linhas(
        previsoes.groupBy("cd_bv", "ranking").count().filter("count > 1")
    ),
    "Cliente/ranking duplicado.",
)
previstas = previsoes.filter(F.col("ranking").isNotNull())
exigir(
    not ha_linhas(previstas.filter(
        prob_invalida
        | F.col("acao_atual").isNull()
        | F.col("proxima_acao").isNull()
        | ~F.col("ranking").between(1, 5)
    )),
    "Previsao invalida.",
)
exigir(
    not ha_linhas(previstas.groupBy("cd_bv", "proxima_acao").count().filter("count > 1")),
    "Destino repetido para o mesmo cliente.",
)
exigir(
    not ha_linhas(previsoes.filter(F.col("publicavel") | F.col("publicavel").isNull())),
    "Esta etapa aceita somente o output candidato com publicavel=False.",
)
exigir(
    not ha_linhas(previsoes.filter(F.col("prob_proxima_acao_7d").isNotNull())),
    "A etapa 3 consolida o MVP sem horizonte. Ha probabilidades temporais inesperadas.",
)

# Confere que as probabilidades mostradas vieram exatamente do modelo salvo.
checagem = previstas.alias("p").join(
    F.broadcast(modelo.alias("m")),
    (F.col("p.acao_atual") == F.col("m.origem"))
    & (F.col("p.proxima_acao") == F.col("m.destino")),
    "left",
)
exigir(
    not ha_linhas(checagem.filter(
        F.col("m.destino").isNull()
        | (F.abs(F.col("p.prob_proxima_acao") - F.col("m.prob_proxima_acao")) > TOLERANCIA)
    )),
    "Output e modelo nao sao consistentes.",
)
exigir(
    not ha_linhas(atuais.groupBy("cd_bv", "data_referencia").count().filter("count > 1")),
    "Estado atual duplicado.",
)
exigir(
    not ha_linhas(atuais.filter(
        F.col("data_referencia").isNull()
        | (F.col("data_referencia") != F.lit(meta["data_referencia"]))
        | F.col("ts_corte").isNull()
        | (F.col("ts_corte") != F.lit(meta["ts_corte"]))
    )),
    "Os estados atuais pertencem a outro corte.",
)
clientes_atuais = atuais.select("cd_bv").distinct()
clientes_output = previsoes.select("cd_bv").distinct()
exigir(
    not ha_linhas(clientes_atuais.join(clientes_output, "cd_bv", "left_anti"))
    and not ha_linhas(clientes_output.join(clientes_atuais, "cd_bv", "left_anti")),
    "A lista de clientes do output difere da lista de estados atuais.",
)

resumo_hml = previsoes.agg(
    F.count("*").alias("n_linhas_output"),
    F.countDistinct("cd_bv").alias("n_clientes_output"),
    F.countDistinct(F.when(F.col("ranking") == 1, F.col("cd_bv"))).alias("n_clientes_top1"),
).first().asDict()
metricas = entradas["validacao"].limit(2).collect()
exigir(len(metricas) == 1, "Esperada uma linha de metricas de validacao.")
metricas = metricas[0].asDict()
cobertura = entradas["cobertura"].orderBy("dia").limit(32).collect()
exigir(0 < len(cobertura) <= 31, "Cobertura vazia ou fora do tamanho esperado.")
cobertura_d1 = [linha for linha in cobertura if linha["dia"] == meta["data_referencia"]]
exigir(len(cobertura_d1) == 1, "A cobertura nao possui a data do output.")

print("Resumo candidato para homologacao:", resumo_hml)
print("Metricas entre transicoes identificadas:", metricas)
print("Status de dados preservado:", meta["status_dados"])
print("NENHUM cliente esta liberado para automacao nesta etapa.")

# COMMAND ----------

# Prepara tres artefatos. Os timestamps ajudam a auditar o estado conhecido.
# Mantem as probabilidades como double (0.70 = 70%), sem arredondar no storage.
modelo_hml = (
    modelo
    .withColumn("data_referencia", F.lit(meta["data_referencia"]).cast("date"))
    .withColumn("ts_corte", F.lit(meta["ts_corte"]).cast("timestamp"))
    .withColumn("fuso_referencia", F.lit(meta["fuso_referencia"]))
    .withColumn("status_dados", F.lit(meta["status_dados"]))
    .withColumn("escopo_modelo", F.lit("AMOSTRA_TREINO_ETAPA_2"))
    .withColumn("id_execucao", F.lit(ID_EXECUCAO))
    .withColumn("ambiente", F.lit("HOMOLOGACAO"))
    .withColumn("gravado_em", F.lit(INICIO_EXECUCAO).cast("timestamp"))
)
output_hml = (
    previsoes
    .join(
        atuais.select(
            "cd_bv", "data_referencia", "ts_entrada_min", "ts_entrada_max",
            "ts_ultima_atividade",
        ),
        ["cd_bv", "data_referencia"],
        "left",
    )
    .withColumn("id_execucao", F.lit(ID_EXECUCAO))
    .withColumn("ambiente", F.lit("HOMOLOGACAO"))
    .withColumn("gravado_em", F.lit(INICIO_EXECUCAO).cast("timestamp"))
)
manifesto = spark.createDataFrame(
    [(
        meta["data_referencia"], VERSAO, ID_EXECUCAO, meta["status_dados"],
        json.dumps(metricas, allow_nan=False, default=str),
        json.dumps(resumo_hml, allow_nan=False, default=str),
        json.dumps([linha.asDict() for linha in cobertura], default=str, allow_nan=False),
    )],
    "data_referencia date, versao_modelo string, id_execucao string, "
    "status_dados string, metricas_json string, resumo_output_json string, "
    "cobertura_fonte_json string",
).withColumn("ambiente", F.lit("HOMOLOGACAO")).withColumn(
    "publicavel", F.lit(False)
).withColumn("iniciado_em", F.lit(INICIO_EXECUCAO).cast("timestamp"))


def registro_execucao(status: str, erro: str | None = None) -> DataFrame:
    return (
        manifesto
        .withColumn("status_execucao", F.lit(status))
        .withColumn("classe_erro", F.lit(erro).cast("string"))
        .withColumn(
            "finalizado_em",
            F.lit(None if status == "EM_GRAVACAO" else datetime.now(timezone.utc))
            .cast("timestamp"),
        )
    )


def gravar_hml(df: DataFrame, tabela: str) -> None:
    """Substitui somente data/versao desta execucao, sem aceitar entradas vazias."""
    exigir(tabela in TABELAS_HML.values(), "Destino nao autorizado neste notebook.")
    exigir(ha_linhas(df), f"Entrada vazia para {tabela}; nenhuma exclusao sera feita.")
    exigir(
        not ha_linhas(df.filter(~F.coalesce(F.expr(PREDICADO), F.lit(False)))),
        f"Ha linhas fora do recorte de publicacao de {tabela}.",
    )
    if spark.catalog.tableExists(tabela):
        formato = spark.sql(f"DESCRIBE DETAIL {tabela}").select("format").first()[0]
        exigir(formato.lower() == "delta", f"{tabela} nao e uma tabela Delta.")
        esquema_existente = {campo.name: campo.dataType for campo in spark.table(tabela).schema}
        esquema_novo = {campo.name: campo.dataType for campo in df.schema}
        exigir(esquema_existente == esquema_novo, f"Schema alterado em {tabela}; revisar antes de escrever.")
        (
            df.write.format("delta").mode("overwrite")
            .option("replaceWhere", PREDICADO)
            .saveAsTable(tabela)
        )
    else:
        df.write.format("delta").mode("errorifexists").saveAsTable(tabela)


if GRAVAR_HOMOLOGACAO:
    gravar_hml(registro_execucao("EM_GRAVACAO"), TABELAS_HML["execucoes"])
    try:
        gravar_hml(modelo_hml, TABELAS_HML["modelo"])
        gravar_hml(output_hml, TABELAS_HML["previsoes"])
        n_salvo = spark.table(TABELAS_HML["previsoes"]).filter(
            (F.col("id_execucao") == ID_EXECUCAO) & F.expr(PREDICADO)
        ).count()
        exigir(n_salvo == resumo_hml["n_linhas_output"], "Contagem de previsoes gravadas diverge.")
        n_modelo_salvo = spark.table(TABELAS_HML["modelo"]).filter(
            (F.col("id_execucao") == ID_EXECUCAO) & F.expr(PREDICADO)
        ).count()
        exigir(n_modelo_salvo == modelo.count(), "Contagem de pares gravados diverge.")
        gravar_hml(registro_execucao("CONCLUIDA"), TABELAS_HML["execucoes"])
    except Exception as erro:
        try:
            gravar_hml(
                registro_execucao("FALHOU", type(erro).__name__),
                TABELAS_HML["execucoes"],
            )
        except Exception as erro_registro:
            print("Falha ao registrar o erro:", type(erro_registro).__name__)
        raise
    print("HOMOLOGACAO GRAVADA. Id da execucao:", ID_EXECUCAO)
    for finalidade, tabela in TABELAS_HML.items():
        print(finalidade, "->", tabela)
else:
    print("Modo somente verificacao. Nenhuma tabela permanente foi alterada.")

# COMMAND ----------

# Consulta de consumo: exige que a gravacao das duas tabelas tenha concluido.
# Nao exibe identificadores de clientes automaticamente.
consulta_top1_hml = f"""
SELECT
    p.data_referencia,
    p.cd_bv,
    p.acao_atual,
    p.proxima_acao,
    p.prob_proxima_acao,
    p.status_dados,
    p.publicavel,
    p.versao_modelo
FROM {TABELAS_HML['previsoes']} AS p
INNER JOIN {TABELAS_HML['execucoes']} AS e
    ON p.id_execucao = e.id_execucao
    AND p.data_referencia = e.data_referencia
    AND p.versao_modelo = e.versao_modelo
WHERE e.status_execucao = 'CONCLUIDA'
    AND p.data_referencia = DATE '{DATA_REF}'
    AND p.versao_modelo = '{VERSAO}'
    AND p.ranking = 1
"""
print("Consulta do top 1 de HOMOLOGACAO (dados NAO liberados para automacao):")
print(consulta_top1_hml)
print("Para top 5: retire apenas o filtro p.ranking = 1.")
print("Para liberar o diario: corrigir cobertura/fuso na origem e reexecutar as etapas 1 e 2.")
print("Nao basta mudar publicavel para True nem trocar a referencia por uma data anterior.")
