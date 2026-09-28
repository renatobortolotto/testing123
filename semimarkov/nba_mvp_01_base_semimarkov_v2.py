# Databricks notebook source
# NBA | Etapa 1 V2: historico para treinamento + publico diario independente.
# Execute as celulas dos arquivos 01, 02 e 03, nessa ordem, no MESMO notebook.
# Nao sobrescreve tabelas V1. Nenhuma escrita permanente nesta etapa.
#
# CONTRATO NOVO, EXPLICITO:
# - Estado comportamental observado. Silencio OPERACIONAL comeca exatamente
#   30 minutos apos o ultimo evento; nao estimamos um abandono latente.
# - Logo, o relogio do silencio e exato pela regra, nao o ponto medio de intervalo.
# - As duracoes e parametros da V1/R NAO sao reutilizados: e preciso reestimar.
# - Empates de estado nao sao ordenados artificialmente. Ha barreiras de
#   observacao. Uma origem conhecida termina censurada na barreira.
# - Sem imputacao de 1 segundo; sem perdido automatico; sem conversao absorvente
#   global (a jornada de comportamentos continua apos a contratacao de produto).
# - O modelo pressupoe saida eventual dos estados modelados. Churn permanente
#   ou acao bancaria intervencionista NAO sao estimados aqui.

from datetime import date, datetime, timedelta
import json
import uuid

from pyspark import StorageLevel
from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F

# =================== CONFIGURACAO: LEIA ANTES DE EXECUTAR ===================
SM_CFG = {
    "data_publico": "2026-09-26",  # Referencia do snapshot customers_query.
    "fonte_publico": "customers_query",
    # Podem ser a mesma fonte, mas a selecao do publico NAO filtra o treinamento.
    "fonte_treino": "base_passo_raw",
    "fonte_aplicacao": "base_passo_raw",
    "inicio_treino": "2026-06-01 00:00:00",
    "corte_treino_exclusivo": "2026-09-25 00:00:00",
    # Exemplo RETROSPECTIVO: nao afirmar que o estado abaixo e o estado de D-1.
    # Troque pelo corte realmente coberto quando a fonte estiver confirmada.
    "corte_estado_exclusivo": "2026-09-25 00:00:00",
    "fuso": "Etc/UTC",  # Manter a interpretacao atual; nao corrige a origem.
    "fuso_origem_confirmado": False,
    "cobertura_treino_confirmada": False,
    "cobertura_aplicacao_confirmada": False,
    "snapshot_publico_confirmado": False,
    "escopo_historico": "HISTORICO_DISPONIVEL_JA_FILTRADO_NA_FONTE_ORIGINAL",
    "relogio": "SILENCIO_OPERACIONAL_TIMEOUT_V2",
    "timeout_seg": 1800,
    "preparar_treino": True,  # False na aplicacao diaria de um modelo salvo.
    "modulo_amostra_treino": 1000,  # 1 = todos os clientes do historico.
    "modulo_amostra_publico": 1000,  # 1 = TODOS os IDs do publico diario.
    "sal_treino": "sm_v2_amostra_historica",
    "sal_publico": "sm_v2_amostra_publico",
    "sal_validacao": "sm_v2_validacao_clientes",
    "versao_modelo": "sm_v2_piloto_001",  # Incremente ao reestimar/publicar.
    "horizonte_dias": 7.0,
}
# Nas fotos, estes eventos sao posicionados em 23:59:59 a partir de uma DATA.
# Nao podemos usa-los como timestamps exatos. Os dias afetados viram barreiras
# de observacao para aquele cliente. Isso e conservador e reduz a cobertura.
# Ajuste a lista SOMENTE apos conferir a precisao na fonte.
SM_ESTADOS_DATA_SEM_HORA = [
    f"{produto}:::conversao" for produto in (
        "ativacao_conta", "solicitacao_cartao", "contrato_leves",
        "contrato_egv", "contrato_motos", "contrato_solar", "contrato_scp",
    )
]
SM_ID_EXECUCAO = str(uuid.uuid4())
SM_SILENCIO = "sem_acao:::classe"
SM_VIEWS = {
    "treino": "nba_sm_v2_base_treino",
    "atuais": "nba_sm_v2_estados_publico",
    "publico": "nba_sm_v2_publico",
    "config": "nba_sm_v2_configuracao",
}


def sm_segundos(inicio: str, fim: str) -> Column:
    return (F.unix_micros(fim) - F.unix_micros(inicio)) / F.lit(1_000_000.0)


def sm_exigir_colunas(df: DataFrame, campos: set[str], nome: str) -> None:
    if campos - set(df.columns):
        raise ValueError(f"{nome}: faltam {sorted(campos - set(df.columns))}")


def sm_ha(df: DataFrame) -> bool:
    return bool(df.limit(1).count())


def sm_ler_eventos(tabela: str, inicio: str | None, corte: str) -> DataFrame:
    df = spark.table(tabela)
    sm_exigir_colunas(df, {"cd_bv", "dm_navegacao", "estado"}, tabela)
    if df.schema["dm_navegacao"].dataType.simpleString() != "timestamp":
        raise ValueError("dm_navegacao deve ser timestamp; revisar parsing/fuso antes.")
    df = df.select(
        F.col("cd_bv").cast("string"), F.col("dm_navegacao").alias("ts_evento"),
        F.col("estado").cast("string"),
    ).filter((F.col("ts_evento") < F.lit(corte).cast("timestamp")) | F.col("ts_evento").isNull())
    if inicio:
        df = df.filter((F.col("ts_evento") >= F.lit(inicio).cast("timestamp")) | F.col("ts_evento").isNull())
    return df


def sm_preparar_passos(eventos: DataFrame, corte: str, nome: str) -> DataFrame:
    """Cronologia operacional consistente; nao liga acoes atraves de ambiguidades."""
    invalido = (F.col("cd_bv").isNull() | F.col("estado").isNull()
                | (F.length(F.trim("estado")) == 0) | F.col("ts_evento").isNull())
    if sm_ha(eventos.filter(invalido)):
        raise ValueError(f"{nome}: evento sem ID/estado/timestamp.")
    if sm_ha(eventos.filter(F.col("estado") == SM_SILENCIO)):
        raise ValueError("A fonte ja tem silencio: use eventos brutos, nao base_passo.")

    e = eventos.withColumn("dia", F.to_date("ts_evento"))
    dias_imprecisos = (
        e.filter(F.col("estado").isin(SM_ESTADOS_DATA_SEM_HORA))
        .select("cd_bv", "dia").distinct().withColumn("dia_impreciso", F.lit(True))
    )
    e = e.join(dias_imprecisos, ["cd_bv", "dia"], "left").fillna({"dia_impreciso": False})
    e = (
        e.withColumn("ts_momento", F.when(F.col("dia_impreciso"), F.col("dia").cast("timestamp"))
                     .otherwise(F.col("ts_evento")))
        .withColumn("ts_fim_indefinicao", F.when(
            F.col("dia_impreciso"),
            F.least(F.date_add("dia", 1).cast("timestamp"), F.lit(corte).cast("timestamp")),
        ).otherwise(F.col("ts_evento")))
    )
    momentos = (
        e.groupBy("cd_bv", "ts_momento")
        .agg(F.sort_array(F.collect_set("estado")).alias("estados_no_instante"),
             F.count("*").alias("n_ev"),
             F.max(F.col("dia_impreciso").cast("int")).cast("boolean").alias("dia_impreciso"),
             F.max("ts_fim_indefinicao").alias("ts_fim_indefinicao"))
        .withColumn("ordem_ambigua", F.col("dia_impreciso") | (F.size("estados_no_instante") > 1))
        .withColumn("estado", F.when(~F.col("ordem_ambigua"), F.element_at("estados_no_instante", 1)))
    )
    momentos.createOrReplaceTempView(f"nba_sm_v2_momentos_{nome}")
    w = Window.partitionBy("cd_bv").orderBy("ts_momento")
    rows = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    marcados = (
        momentos.withColumn("anterior", F.lag("estado").over(w))
        .withColumn("ts_anterior", F.lag("ts_fim_indefinicao").over(w))
        .withColumn("abre", F.when(
            F.col("estado").isNotNull() & (F.col("estado") == F.col("anterior"))
            & (sm_segundos("ts_anterior", "ts_momento") <= SM_CFG["timeout_seg"]), 0,
        ).otherwise(1))
        .withColumn("id_bloco", F.sum("abre").over(rows))
    )
    b = (
        marcados.groupBy("cd_bv", "id_bloco")
        .agg(F.min("estado").alias("estado"), F.sum("n_ev").alias("n_ev"),
             F.min("ts_momento").alias("ts_inicio"), F.max("ts_fim_indefinicao").alias("ts_ultimo"),
             F.max(F.col("ordem_ambigua").cast("int")).cast("boolean").alias("ordem_ambigua"),
             F.max(F.col("dia_impreciso").cast("int")).cast("boolean").alias("dia_impreciso"))
    )
    wb = Window.partitionBy("cd_bv").orderBy("id_bloco")
    b = (
        b.withColumn("prox_inicio", F.lead("ts_inicio").over(wb))
        .withColumn("ts_corte_estado", F.lit(corte).cast("timestamp"))
        .withColumn("ts_timeout", F.expr(f"ts_ultimo + INTERVAL {SM_CFG['timeout_seg']} SECONDS"))
        .withColumn("gera_silencio", ~F.col("dia_impreciso")
                    & (F.col("ts_timeout") < F.col("ts_corte_estado"))
                    & (F.col("prox_inicio").isNull() | (F.col("ts_timeout") < F.col("prox_inicio"))))
    )
    b = (b.withColumn("anterior_ambiguo", F.lag("ordem_ambigua").over(wb))
         .withColumn("anterior_gerou_silencio", F.lag("gera_silencio").over(wb)))
    campos = ["cd_bv", "id_bloco", "subpasso", "estado", "ts_inicio", "ts_fim",
              "ts_ultima_atividade", "ultima_acao_observada", "n_ev", "sintetico",
              "ordem_ambigua", "dia_impreciso", "inicio_observado", "ts_corte_estado"]
    atividade = (
        b.withColumn("subpasso", F.lit(0))
        .withColumn("ts_fim", F.when(F.col("gera_silencio"), F.col("ts_timeout"))
                    .otherwise(F.coalesce("prox_inicio", "ts_corte_estado")))
        .withColumn("ts_ultima_atividade", F.col("ts_ultimo"))
        .withColumn("ultima_acao_observada", F.col("estado"))
        .withColumn("sintetico", F.lit(False))
        # A janela de extracao pode comecar no meio do primeiro bloco.
        .withColumn("inicio_observado", (F.col("id_bloco") > 1) & ~F.col("ordem_ambigua")
                    & (~F.coalesce(F.col("anterior_ambiguo"), F.lit(False))
                       | F.coalesce(F.col("anterior_gerou_silencio"), F.lit(False))))
        .select(*campos)
    )
    silencio = (
        b.filter(F.col("gera_silencio"))
        .select("cd_bv", "id_bloco", F.lit(1).alias("subpasso"),
                F.lit(SM_SILENCIO).alias("estado"), F.col("ts_timeout").alias("ts_inicio"),
                F.coalesce("prox_inicio", "ts_corte_estado").alias("ts_fim"),
                F.col("ts_ultimo").alias("ts_ultima_atividade"),
                F.col("estado").alias("ultima_acao_observada"),
                F.lit(0).cast("long").alias("n_ev"), F.lit(True).alias("sintetico"),
                F.lit(False).alias("ordem_ambigua"), F.lit(False).alias("dia_impreciso"),
                F.lit(True).alias("inicio_observado"), "ts_corte_estado")
    )
    wp = Window.partitionBy("cd_bv").orderBy("id_bloco", "subpasso")
    result = (
        atividade.unionByName(silencio)
        .withColumn("passo", F.row_number().over(wp))
        .withColumn("proximo_estado_bruto", F.lead("estado").over(wp))
        .withColumn("proximo_passo", F.lead("passo").over(wp))
        .withColumn("dur_min", sm_segundos("ts_inicio", "ts_fim"))
        .withColumn("tipo_censura", F.when(
            F.col("proximo_passo").isNull() | F.col("proximo_estado_bruto").isNull(), "direita"
        ).otherwise("exata"))
        .withColumn("destino", F.when(F.col("tipo_censura") == "exata", F.col("proximo_estado_bruto")))
        .withColumn("dur_max", F.when(F.col("tipo_censura") == "exata", F.col("dur_min")))
        .withColumn("status_observacao", F.when(F.col("estado").isNull(), "ORIGEM_AMBIGUA")
                    .when(~F.col("inicio_observado"), "INICIO_NAO_OBSERVADO")
                    .when(F.col("proximo_passo").isNull(), "CENSURA_CORTE")
                    .when(F.col("proximo_estado_bruto").isNull(), "CENSURA_ANTES_AMBIGUIDADE")
                    .otherwise("SAIDA_OBSERVADA"))
        .withColumn("elegivel_ajuste", F.col("estado").isNotNull() & F.col("inicio_observado")
                    & (F.col("dur_min") > 0))
        .withColumn("relogio", F.lit(SM_CFG["relogio"]))
    )
    if sm_ha(result.filter(F.col("dur_min").isNull() | (F.col("dur_min") <= 0))):
        raise ValueError("Duracao nao positiva: interromper, sem aplicar piso.")
    return result


# Nao altera o fuso da sessao silenciosamente.
if spark.conf.get("spark.sql.session.timeZone") != SM_CFG["fuso"]:
    raise ValueError("Fuso da sessao difere do configurado; revisar antes.")
ref = date.fromisoformat(SM_CFG["data_publico"])
sm_corte_referencia = datetime.combine(ref + timedelta(days=1), datetime.min.time())
sm_corte_estado = datetime.fromisoformat(SM_CFG["corte_estado_exclusivo"])
sm_corte_treino = datetime.fromisoformat(SM_CFG["corte_treino_exclusivo"])
if sm_corte_treino > sm_corte_estado or sm_corte_estado > sm_corte_referencia:
    raise ValueError("Exigir corte_treino <= corte_estado <= fechamento da referencia.")
if SM_CFG["horizonte_dias"] != 7.0:
    raise ValueError("Esta versao usa colunas 7d: horizonte_dias deve permanecer 7.")
if any(SM_CFG[k] < 1 for k in ("modulo_amostra_treino", "modulo_amostra_publico")):
    raise ValueError("Modulo de amostra invalido.")
SM_CFG["estados_data_sem_hora"] = SM_ESTADOS_DATA_SEM_HORA
SM_CFG["id_execucao"] = SM_ID_EXECUCAO
print("Corte publico:", SM_CFG["data_publico"], "| corte do estado:", sm_corte_estado)
print("Corte e cobertura de comportamento NAO sao inferidos de customers_query.")
print("Relogio:", SM_CFG["relogio"], "| nenhuma tabela permanente sera alterada.")

# COMMAND ----------
# TREINAMENTO: somente o historico. Nunca faz join com o publico do dia.
if SM_CFG["preparar_treino"]:
    sm_evt_treino = sm_ler_eventos(
        SM_CFG["fonte_treino"], SM_CFG["inicio_treino"], SM_CFG["corte_treino_exclusivo"]
    ).filter(F.pmod(
        F.xxhash64("cd_bv", F.lit(SM_CFG["sal_treino"])),
        F.lit(SM_CFG["modulo_amostra_treino"]),
    ) == 0)
    sm_base_treino = sm_preparar_passos(sm_evt_treino, SM_CFG["corte_treino_exclusivo"], "treino")
    sm_base_treino = sm_base_treino.withColumn(
        "validacao_cliente", F.pmod(
            F.xxhash64("cd_bv", F.lit(SM_CFG["sal_validacao"])), F.lit(5)
        ) == 0,
    ).persist(StorageLevel.MEMORY_AND_DISK)
    sm_base_treino.createOrReplaceTempView(SM_VIEWS["treino"])
    sm_base_treino.groupBy("validacao_cliente", "status_observacao").count().show(truncate=False)

# COMMAND ----------
# APLICACAO: comeca pela LISTA DIARIA, inclusive IDs sem historico.
sm_publico_raw = spark.table(SM_CFG["fonte_publico"])
sm_exigir_colunas(sm_publico_raw, {"cd_bv"}, SM_CFG["fonte_publico"])
sm_publico = (
    sm_publico_raw.select(F.col("cd_bv").cast("string"))
    .filter(F.col("cd_bv").isNotNull()).distinct()
    .filter(F.pmod(F.xxhash64("cd_bv", F.lit(SM_CFG["sal_publico"])),
                   F.lit(SM_CFG["modulo_amostra_publico"])) == 0)
    .withColumn("data_referencia", F.lit(SM_CFG["data_publico"]).cast("date"))
    .persist(StorageLevel.MEMORY_AND_DISK)
)
if not sm_ha(sm_publico):
    raise ValueError("Publico diario/amostra vazio: nao publicar nem apagar resultados.")
sm_publico.createOrReplaceTempView(SM_VIEWS["publico"])
sm_evt_app = sm_ler_eventos(
    SM_CFG["fonte_aplicacao"], None, SM_CFG["corte_estado_exclusivo"]
).join(sm_publico.select("cd_bv"), "cd_bv", "left_semi")
sm_passos_app = sm_preparar_passos(sm_evt_app, SM_CFG["corte_estado_exclusivo"], "publico")
sm_ultimo = (
    sm_passos_app.withColumn("_ultima", F.row_number().over(
        Window.partitionBy("cd_bv").orderBy(F.desc("passo"))))
    .filter(F.col("_ultima") == 1)
    .select("cd_bv", "estado", "ts_inicio", "ts_ultima_atividade", "ultima_acao_observada",
            "ordem_ambigua", "inicio_observado")
)
sm_status_dados = (
    "CORTE_RETROSPECTIVO_NAO_D1" if sm_corte_estado != sm_corte_referencia
    else "COBERTURA_NAO_CONFIRMADA" if not SM_CFG["cobertura_aplicacao_confirmada"]
    else "FUSO_NAO_CONFIRMADO" if not SM_CFG["fuso_origem_confirmado"]
    else "SNAPSHOT_PUBLICO_NAO_CONFIRMADO" if not SM_CFG["snapshot_publico_confirmado"]
    else "DADOS_CONFIRMADOS"
)
sm_atuais = (
    sm_publico.join(sm_ultimo, "cd_bv", "left")
    .withColumn("ts_corte_estado", F.lit(SM_CFG["corte_estado_exclusivo"]).cast("timestamp"))
    .withColumn("acao_atual", F.col("estado"))
    .withColumn("tempo_no_estado_seg", F.when(F.col("inicio_observado"), sm_segundos("ts_inicio", "ts_corte_estado")))
    .withColumn("status_input", F.when(F.col("ts_inicio").isNull(), "SEM_HISTORICO")
                .when(F.col("ordem_ambigua"), "ESTADO_ATUAL_AMBIGUO")
                .when(~F.col("inicio_observado"), "IDADE_ESTADO_DESCONHECIDA")
                .otherwise("OK"))
    .withColumn("status_dados", F.lit(sm_status_dados))
    .withColumn("fuso_referencia", F.lit(SM_CFG["fuso"]))
    .withColumn("relogio", F.lit(SM_CFG["relogio"]))
    .persist(StorageLevel.MEMORY_AND_DISK)
)
sm_atuais.createOrReplaceTempView(SM_VIEWS["atuais"])
sm_atuais.groupBy("status_input", "status_dados").count().show(truncate=False)
sm_atuais.groupBy("acao_atual").count().orderBy(F.desc("count")).show(15, truncate=False)
sm_meta_json = json.dumps(SM_CFG, ensure_ascii=False, sort_keys=True)
spark.createDataFrame([(sm_meta_json,)], "config_json string").createOrReplaceTempView(SM_VIEWS["config"])
print("Views V2 criadas. Proxima etapa: ajuste conjunto do semi-Markov.")
