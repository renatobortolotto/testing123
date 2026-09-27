# Databricks notebook source
# Base candidata para diagnóstico. Não publica nem sobrescreve tabelas.
# Fuso/cobertura das fontes devem ser confirmados antes da execução final.

from datetime import date, timedelta

from pyspark import StorageLevel
from pyspark.sql import Column, Window
from pyspark.sql import functions as F

DATA_REFERENCIA = "2026-09-26"  # Último dia incluído, no fuso da sessão.
FONTE = "base_passo_raw"
TIMEOUT_SEG = 1800
MODULO_AMOSTRA = 1000  # Aproximadamente 0,1% dos clientes.
RESTO_AMOSTRA = 0

referencia = date.fromisoformat(DATA_REFERENCIA)
TS_CORTE = f"{referencia + timedelta(days=1)} 00:00:00"
fuso_sessao = spark.conf.get("spark.sql.session.timeZone")
if fuso_sessao not in {"UTC", "Etc/UTC"}:
    raise ValueError("A sessão mudou de fuso. Revise o corte antes de continuar.")

if not (MODULO_AMOSTRA >= 1 and 0 <= RESTO_AMOSTRA < MODULO_AMOSTRA):
    raise ValueError("Parâmetros inválidos para a amostra de clientes.")
if TIMEOUT_SEG <= 0:
    raise ValueError("TIMEOUT_SEG precisa ser positivo.")


def segundos(inicio: str, fim: str) -> Column:
    """Calcula segundos preservando os microssegundos disponíveis."""
    return (F.unix_micros(fim) - F.unix_micros(inicio)) / F.lit(1_000_000.0)


fonte = spark.table(FONTE)
necessarias = {"cd_bv", "dm_navegacao", "estado", "profundidade_max"}
if necessarias - set(fonte.columns):
    raise ValueError(f"Colunas ausentes: {sorted(necessarias - set(fonte.columns))}")

# Seleciona clientes, não eventos avulsos. Mantém o histórico disponível deles.
eventos_amostra = (
    fonte
    .filter(F.col("cd_bv").isNotNull())
    .filter(
        F.pmod(F.xxhash64("cd_bv"), F.lit(MODULO_AMOSTRA))
        == RESTO_AMOSTRA
    )
    .select(
        "cd_bv",
        F.col("dm_navegacao").alias("ts_evento"),
        "estado",
        F.col("profundidade_max").cast("double").alias("profundidade_max"),
    )
    .withColumn("ts_corte", F.lit(TS_CORTE).cast("timestamp"))
    .filter(
        (F.col("ts_evento") < F.col("ts_corte"))
        | F.col("ts_evento").isNull()
    )
    .persist(StorageLevel.MEMORY_AND_DISK)
)

invalido = (
    F.col("ts_evento").isNull()
    | F.col("estado").isNull()
    | (F.length(F.trim("estado")) == 0)
)
if eventos_amostra.filter(invalido).limit(1).count():
    raise ValueError("Há eventos com timestamp ou estado inválido na amostra.")
if not eventos_amostra.limit(1).count():
    raise ValueError("Nenhum evento foi encontrado na amostra antes do corte.")

print(f"Base CANDIDATA | referência: {DATA_REFERENCIA} | fuso: {fuso_sessao}")
print(f"Corte exclusivo: {TS_CORTE} | fonte: {FONTE}")

# COMMAND ----------

# Um instante por cliente. Não inventa ordem entre estados simultâneos.
momentos = (
    eventos_amostra
    .groupBy("cd_bv", "ts_evento")
    .agg(
        F.sort_array(F.collect_set("estado")).alias("estados_no_instante"),
        F.count("*").alias("n_ev"),
        F.max("profundidade_max").alias("profundidade_max"),
    )
    .withColumn("ordem_ambigua", F.size("estados_no_instante") > 1)
    .withColumn(
        "estado",
        F.when(
            ~F.col("ordem_ambigua"),
            F.element_at("estados_no_instante", 1),
        ),
    )
)
momentos.createOrReplaceTempView("nba_momentos_auditoria_v1")

janela_eventos = Window.partitionBy("cd_bv").orderBy("ts_evento")
acumulada = janela_eventos.rowsBetween(Window.unboundedPreceding, Window.currentRow)

marcados = (
    momentos
    .withColumn("estado_anterior", F.lag("estado").over(janela_eventos))
    .withColumn("ts_anterior", F.lag("ts_evento").over(janela_eventos))
    .withColumn(
        "abre_bloco",
        F.when(
            F.col("estado").isNotNull()
            & (F.col("estado") == F.col("estado_anterior"))
            & (segundos("ts_anterior", "ts_evento") <= TIMEOUT_SEG),
            0,
        ).otherwise(1),
    )
    .withColumn("id_bloco", F.sum("abre_bloco").over(acumulada))
)

blocos_v1 = (
    marcados.groupBy("cd_bv", "id_bloco")
    .agg(
        F.min("estado").alias("estado"),
        F.sum("n_ev").alias("n_ev"),
        F.max("profundidade_max").alias("profundidade_max"),
        F.min("ts_evento").alias("ts_inicio"),
        F.max("ts_evento").alias("ts_ultimo"),
        F.max(F.col("ordem_ambigua").cast("int"))
        .cast("boolean").alias("ordem_ambigua"),
    )
)

janela_blocos = Window.partitionBy("cd_bv").orderBy("id_bloco")
medidos = (
    blocos_v1
    .withColumn("ts_corte", F.lit(TS_CORTE).cast("timestamp"))
    .withColumn("prox_inicio", F.lead("ts_inicio").over(janela_blocos))
    .withColumn("span_seg", segundos("ts_inicio", "ts_ultimo"))
    .withColumn("gap_seg", segundos("ts_ultimo", "prox_inicio"))
    .withColumn("ate_prox_seg", segundos("ts_inicio", "prox_inicio"))
    .withColumn(
        "ts_fecha_sessao",
        F.expr(f"ts_ultimo + INTERVAL {TIMEOUT_SEG} SECONDS"),
    )
)

# Preserva a regra intervalar existente. Não aplica piso de 1 segundo
# nem converte silêncios longos em "perdido" nesta preparação.
direita = (
    F.col("prox_inicio").isNull()
    & (F.col("ts_fecha_sessao") >= F.col("ts_corte"))
)
exata = F.col("gap_seg") <= TIMEOUT_SEG

campos = [
    "cd_bv", "id_bloco", "subpasso", "estado", "n_ev", "profundidade_max",
    "ts_inicio", "ts_ultimo", "ts_ultima_atividade", "prox_inicio",
    "ts_entrada_min", "ts_entrada_max", "dur_min", "dur_max",
    "tipo_censura", "sintetico", "ordem_ambigua", "ts_corte",
]

atividade = (
    medidos
    .withColumn("subpasso", F.lit(0))
    .withColumn("ts_ultima_atividade", F.col("ts_ultimo"))
    .withColumn("ts_entrada_min", F.col("ts_inicio"))
    .withColumn("ts_entrada_max", F.col("ts_inicio"))
    .withColumn(
        "tipo_censura",
        F.when(direita, "direita").when(exata, "exata").otherwise("intervalo"),
    )
    .withColumn(
        "dur_min",
        F.when(direita, segundos("ts_inicio", "ts_corte"))
        .when(exata, F.col("ate_prox_seg"))
        .otherwise(F.col("span_seg")),
    )
    .withColumn(
        "dur_max",
        F.when(direita, F.lit(None).cast("double"))
        .when(exata, F.col("ate_prox_seg"))
        .otherwise(F.col("span_seg") + TIMEOUT_SEG),
    )
    .withColumn("sintetico", F.lit(False))
    .select(*campos)
)

silencio = (
    medidos
    .filter(
        (F.col("prox_inicio").isNull() | (F.col("gap_seg") > TIMEOUT_SEG))
        & (F.col("ts_fecha_sessao") < F.col("ts_corte"))
    )
    .select(
        "cd_bv", "id_bloco",
        F.lit(1).alias("subpasso"),
        F.lit("sem_acao:::classe").alias("estado"),
        F.lit(0).cast("long").alias("n_ev"),
        F.lit(None).cast("double").alias("profundidade_max"),
        F.col("ts_fecha_sessao").alias("ts_inicio"),
        F.lit(None).cast("timestamp").alias("ts_ultimo"),
        F.col("ts_ultimo").alias("ts_ultima_atividade"),
        "prox_inicio",
        F.col("ts_ultimo").alias("ts_entrada_min"),
        F.col("ts_fecha_sessao").alias("ts_entrada_max"),
        F.when(
            F.col("prox_inicio").isNull(),
            segundos("ts_fecha_sessao", "ts_corte"),
        ).otherwise(F.col("gap_seg") - TIMEOUT_SEG).alias("dur_min"),
        F.when(
            F.col("prox_inicio").isNotNull(), F.col("gap_seg")
        ).cast("double").alias("dur_max"),
        F.when(F.col("prox_inicio").isNull(), "direita")
        .otherwise("intervalo").alias("tipo_censura"),
        F.lit(False).alias("sintetico"),
        F.lit(False).alias("ordem_ambigua"),
        "ts_corte",
    )
)

sequencia = Window.partitionBy("cd_bv").orderBy("id_bloco", "subpasso")
base_passo_v1_amostra = (
    atividade.unionByName(silencio)
    .withColumn("passo", F.row_number().over(sequencia))
    # O lead é aplicado ANTES de excluir qualquer estado ambíguo.
    .withColumn("destino", F.lead("estado").over(sequencia))
    .withColumn(
        "destino_ambiguo",
        F.coalesce(F.lead("ordem_ambigua").over(sequencia), F.lit(False)),
    )
    .withColumn(
        "status_transicao",
        F.when(F.col("estado").isNull(), "origem_ambigua")
        .when(F.col("tipo_censura") == "direita", "censurada")
        .when(F.col("destino_ambiguo"), "destino_ambiguo")
        .otherwise("observada"),
    )
    .withColumn(
        "elegivel_transicao",
        F.col("estado").isNotNull()
        & F.col("destino").isNotNull()
        & (F.col("tipo_censura") != "direita"),
    )
    .withColumn("data_referencia", F.lit(DATA_REFERENCIA).cast("date"))
    .withColumn("fuso_referencia", F.lit(fuso_sessao))
    .persist(StorageLevel.MEMORY_AND_DISK)
)
base_passo_v1_amostra.createOrReplaceTempView("nba_base_passo_v1_amostra")

# COMMAND ----------

# Resultado agregado: sem exposição de identificadores de clientes.
def contar(condicao: Column) -> Column:
    return F.sum(F.when(condicao, 1).otherwise(0))


limites_validos = (
    (F.col("dur_min") >= 0)
    & (
        (
            (F.col("tipo_censura") == "direita")
            & F.col("dur_max").isNull()
        )
        | (
            (F.col("tipo_censura") == "exata")
            & (F.col("dur_max") == F.col("dur_min"))
        )
        | (
            (F.col("tipo_censura") == "intervalo")
            & (F.col("dur_max") > F.col("dur_min"))
        )
    )
)

resumo_base_v1 = base_passo_v1_amostra.agg(
    F.count("*").alias("n_passos"),
    F.countDistinct("cd_bv").alias("n_clientes"),
    F.countDistinct("estado").alias("n_estados_identificados"),
    contar(F.col("ordem_ambigua")).alias("n_passos_ambiguos"),
    contar(F.col("elegivel_transicao")).alias("n_transicoes_elegiveis"),
    contar(F.col("tipo_censura") == "direita").alias("n_direita"),
    contar(
        (F.col("tipo_censura") == "exata") & (F.col("dur_min") == 0)
    ).alias("n_exatas_zero"),
    contar(
        ~F.coalesce(limites_validos, F.lit(False))
    ).alias("n_duracoes_invalidas"),
)
resumo_base_v1.show(vertical=True, truncate=False)

eventos_amostra.agg(
    F.min("ts_evento").alias("primeiro_evento_amostra"),
    F.max("ts_evento").alias("ultimo_evento_amostra"),
).show(truncate=False)

# Esta view já identifica o último estado conhecido no corte.
# Não é o output de ranking e não está liberada para publicação.
estados_atuais_v1_amostra = (
    base_passo_v1_amostra
    .withColumn(
        "ordem_atual",
        F.row_number().over(
            Window.partitionBy("cd_bv").orderBy(F.desc("passo"))
        ),
    )
    .filter(F.col("ordem_atual") == 1)
    .select(
        "cd_bv", "data_referencia", "estado", "ordem_ambigua",
        "ts_entrada_min", "ts_entrada_max", "ts_ultima_atividade", "ts_corte",
    )
)
estados_atuais_v1_amostra.createOrReplaceTempView("nba_estados_atuais_v1_amostra")

base_passo_v1_amostra.groupBy("status_transicao").count().show(truncate=False)

print("Geradas as views de auditoria, passos e estados atuais da amostra.")
print("Nenhuma tabela permanente foi publicada ou sobrescrita.")
print("A presença do último evento NÃO comprova cobertura completa de D-1.")
