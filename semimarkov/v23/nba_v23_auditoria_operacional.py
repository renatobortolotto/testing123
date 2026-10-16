# Databricks notebook source
# V2.3: auditoria operacional, somente leitura.
from pyspark.sql import functions as F

PREFIXO = "ctg_dsti.renato_nba.nba_sm_v23_operacional"
CD_BV_TESTE = None  # Informe o identificador de um cliente conhecido.

publico = spark.table(f"{PREFIXO}_publico_hml")
estados = spark.table(f"{PREFIXO}_estados_hml")
negocio = spark.table(f"{PREFIXO}_negocio_hml")

for nome, df in (("publico", publico), ("estados", estados), ("negocio", negocio)):
    print(f"{nome}: {df.columns}")
    if "cd_bv" not in df.columns:
        raise ValueError(f"A tabela {nome} não possui cd_bv.")
    if "id_execucao" in df.columns and df.select("id_execucao").distinct().limit(2).count() > 1:
        raise ValueError(f"A tabela {nome} contém várias execuções. Filtre um único id_execucao antes de continuar.")

# COMMAND ----------
# 1. Cobertura e consistência entre fontes.
ids_publico = publico.select("cd_bv").distinct()
ids_estados = estados.select("cd_bv").distinct()
ids_negocio = negocio.select("cd_bv").distinct()

n_publico = ids_publico.count()
n_previstos = ids_negocio.count()
print("01_COBERTURA")
print("Elegíveis:", n_publico)
print("Com estado:", ids_estados.count())
print("Com previsão:", n_previstos)
print("Linhas de negócio:", negocio.count())
print("Cobertura:", round(n_previstos / n_publico, 4) if n_publico else None)
print("Elegíveis sem estado:", ids_publico.join(ids_estados, "cd_bv", "left_anti").count())
print("Previsões fora do público:", ids_negocio.join(ids_publico, "cd_bv", "left_anti").count())

# COMMAND ----------
# 2. Distribuição de estados e causas de ausência de previsão.
campos_status = [c for c in ("status_dado", "status_previsao") if c in estados.columns]
if campos_status:
    cobertura = estados.join(
        ids_negocio.withColumn("com_previsao", F.lit(True)),
        "cd_bv", "left"
    ).fillna({"com_previsao": False})
    cobertura.groupBy(*campos_status, "com_previsao").agg(
        F.countDistinct("cd_bv").alias("n_clientes")
    ).orderBy(F.desc("n_clientes")).show(50, truncate=False)
else:
    print("Colunas de status não identificadas. Inspecione o schema de estados.")

# COMMAND ----------
# 3. Identificação explícita das colunas da saída de negócio.
def identificar(df, candidatos):
    mapa = {c.lower(): c for c in df.columns}
    return next((mapa[c.lower()] for c in candidatos if c.lower() in mapa), None)

rank_col = identificar(negocio, [
    "ranking_acionavel", "ranking_actionable", "ranking_util", "ranking", "rank"
])
acao_col = identificar(negocio, [
    "actionable_acao", "proxima_acao_util", "acao_util", "acao_recomendada"
])
score_col = identificar(negocio, [
    "actionable_score", "score_util", "score_acao_util", "prob_util", "score"
])
print("03_MAPEAMENTO:", {"ranking": rank_col, "acao": acao_col, "score": score_col})

if not all((rank_col, acao_col, score_col)):
    raise ValueError(
        "Colunas da saída LONG não identificadas. Confira negocio.columns "
        "e ajuste as três listas de candidatos acima."
    )

rec = negocio.select(
    "cd_bv",
    F.col(rank_col).cast("int").alias("rank_util"),
    F.col(acao_col).cast("string").alias("acao_util"),
    F.col(score_col).cast("double").alias("score_util"),
)

# COMMAND ----------
# 4. Sanity checks do Top 5: não pressupor soma de scores igual a 1.
checks = rec.groupBy("cd_bv").agg(
    F.count("*").alias("linhas"),
    F.countDistinct("rank_util").alias("ranks"),
    F.countDistinct("acao_util").alias("acoes"),
    F.min("rank_util").alias("rank_min"),
    F.max("rank_util").alias("rank_max"),
    F.sum(F.when(
        F.col("score_util").isNull() | F.isnan("score_util")
        | (F.col("score_util") < 0) | (F.col("score_util") > 1), 1
    ).otherwise(0)).alias("scores_invalidos"),
)
print("04_DISTRIBUICAO_RANKS")
checks.groupBy("linhas", "ranks", "acoes", "rank_min", "rank_max").count().show(30)
print("04_ANOMALIAS")
checks.filter(
    (F.col("linhas") != 5) | (F.col("ranks") != 5)
    | (F.col("acoes") != 5) | (F.col("rank_min") != 1)
    | (F.col("rank_max") != 5) | (F.col("scores_invalidos") > 0)
).show(20, truncate=False)

# COMMAND ----------
# 5. Personalização: concentração e dispersão real dos scores.
top1 = rec.filter(F.col("rank_util") == 1)
print("05_TOP1")
top1.groupBy("acao_util").agg(
    F.countDistinct("cd_bv").alias("clientes"),
    F.avg("score_util").alias("score_medio"),
    F.stddev_pop("score_util").alias("desvio_score"),
    F.countDistinct(F.round("score_util", 6)).alias("scores_distintos_6_casas"),
).orderBy(F.desc("clientes")).show(30, truncate=False)

if "estado" in estados.columns:
    atributos = [c for c in (
        "estado", "contexto_negocio", "contexto_modelo", "origem_tecnica"
    ) if c in estados.columns]
    base_contexto = estados.select("cd_bv", *atributos).dropDuplicates(["cd_bv"])
    top1_contexto = top1.join(base_contexto, "cd_bv", "left")
    print("05_TOP1_POR_ESTADO")
    top1_contexto.groupBy("estado", "acao_util").agg(
        F.countDistinct("cd_bv").alias("clientes"),
        F.stddev_pop("score_util").alias("desvio_score"),
        F.countDistinct(F.round("score_util", 6)).alias("scores_distintos_6_casas"),
    ).orderBy(F.desc("clientes")).show(30, truncate=False)
    if "contexto_negocio" in atributos:
        print("05_VARIACAO_DE_CONTEXTO")
        top1_contexto.groupBy("estado").agg(
            F.countDistinct("cd_bv").alias("clientes"),
            F.countDistinct("contexto_negocio").alias("contextos_distintos"),
            F.countDistinct("acao_util").alias("top1_distintos"),
        ).orderBy(F.desc("clientes")).show(30, truncate=False)

# COMMAND ----------
# 6. Auditoria de um cliente conhecido e sua sequência observada.
if CD_BV_TESTE is not None:
    alvo = F.col("cd_bv").cast("string") == F.lit(str(CD_BV_TESTE))
    print("06_ESTADO_DO_CLIENTE")
    estados.filter(alvo).show(20, truncate=False)
    print("06_TOP5_DO_CLIENTE")
    negocio.filter(alvo).orderBy(F.col(rank_col).asc()).show(10, truncate=False)

    eventos = spark.table("ctg_dsti.renato_nba.base_jornadas_onboarding")
    if {"cd_bv", "dm_navegacao", "estado"}.issubset(eventos.columns):
        print("06_ULTIMOS_40_EVENTOS")
        eventos.filter(alvo).select(
            *[c for c in ("dm_navegacao", "estado", "profundidade_max")
              if c in eventos.columns]
        ).orderBy(F.desc("dm_navegacao")).show(40, truncate=False)
    else:
        print("Colunas da fonte histórica diferem das esperadas.")
else:
    print("06_CLIENTE_CONHECIDO: informe CD_BV_TESTE e execute esta célula.")

print("Auditoria finalizada. Nenhuma tabela foi modificada.")
