# Databricks notebook source
# NBA | Etapa 3 V2: auditar a probabilidade TEMPORAL e consolidar homologacao.
# Executar depois de 01/02 V2, no mesmo notebook. Nao usa a publicacao V1.
# IMPORTANTE: a verificacao antiga output == frequencia p_ij era incorreta para
# este MVP. Agora recalculamos p_ij*S_ij(idade) / sum_k p_ik*S_ik(idade).
# GRAVAR_HOMOLOGACAO=False por padrao. Mude conscientemente para persistir.
# Sem agendamento/automacao e sem sobrescrever fontes ou tabelas V1.

from datetime import datetime, timezone
import json
import re

import numpy as np
import pandas as pd
from scipy import special, stats
from pyspark.sql import functions as F

GRAVAR_HOMOLOGACAO = False
SM_TABELAS = {
    "modelos": "ctg_dsti.renato_nba.nba_semimarkov_modelos_v2_hml",
    "previsoes": "ctg_dsti.renato_nba.nba_semimarkov_previsoes_v2_hml",
    "execucoes": "ctg_dsti.renato_nba.nba_semimarkov_execucoes_v2_hml",
}
SM_TOL = 1e-8
SM_INICIO = datetime.now(timezone.utc)
if not re.fullmatch(r"[A-Za-z0-9_-]+", sm_versao):
    raise ValueError("Versao deve conter apenas letras, numeros, _ e -.")
SM_PREDICADO = (
    f"data_referencia = DATE '{SM_CFG['data_publico']}' "
    f"AND versao_modelo = '{sm_versao}'"
)


def sm_exigir(condicao: bool, mensagem: str) -> None:
    if not condicao:
        raise ValueError(mensagem)


sm_saida = spark.table("nba_sm_v2_previsoes_top5")
sm_mod = spark.table("nba_sm_v2_modelos")
sm_input = spark.table(SM_VIEWS["atuais"])
sm_publico_ids = spark.table(SM_VIEWS["publico"]).select("cd_bv").distinct()
sm_exigir(sm_ha(sm_saida), "Output vazio: nao escrever nem apagar recorte.")
sm_exigir(not sm_ha(sm_saida.filter(F.col("publicavel"))), "Esta etapa somente grava homologacao.")
sm_exigir(not sm_ha(sm_saida.filter(
    F.col("versao_modelo").isNull() | (F.col("versao_modelo") != sm_versao)
    | F.col("data_referencia").isNull() | F.col("ts_corte_estado").isNull()
    | (F.col("ts_corte_estado") != F.lit(SM_CFG["corte_estado_exclusivo"]).cast("timestamp"))
    | (F.col("relogio") != SM_CFG["relogio"])
    | (F.col("data_referencia") != F.lit(SM_CFG["data_publico"]).cast("date"))
)), "Mistura de versoes/referencias.")
sm_clientes_output = sm_saida.select("cd_bv").distinct()
sm_exigir(not sm_ha(sm_publico_ids.join(sm_clientes_output, "cd_bv", "left_anti"))
          and not sm_ha(sm_clientes_output.join(sm_publico_ids, "cd_bv", "left_anti")),
          "Output precisa preservar exatamente os clientes selecionados em customers_query.")
sm_prev = sm_saida.filter(F.col("ranking").isNotNull())
sm_exigir(not sm_ha(sm_prev.groupBy("cd_bv", "ranking").count().filter("count > 1")), "Ranking duplicado.")
sm_exigir(not sm_ha(sm_prev.groupBy("cd_bv", "proxima_acao").count().filter("count > 1")), "Destino duplicado.")
sm_exigir(not sm_ha(sm_prev.filter(
    F.col("prob_proxima_acao").isNull() | F.isnan("prob_proxima_acao")
    | ~F.col("prob_proxima_acao").between(0., 1.)
    | F.col("tempo_no_estado_seg").isNull() | (F.col("tempo_no_estado_seg") < 0)
    | F.col("prob_proxima_acao_7d").isNull()
    | (F.col("prob_proxima_acao_7d") < -SM_TOL)
    | (F.col("prob_proxima_acao_7d") > F.col("prob_proxima_acao") + SM_TOL)
)), "Probabilidade/idade invalida.")
sm_exigir(not sm_ha(sm_prev.groupBy("cd_bv").agg(
    F.count("*").alias("n"), F.sum("prob_proxima_acao").alias("soma")
).filter((F.col("n") > TOP_K) | (F.col("soma") > 1. + SM_TOL))), "Top 5 invalido ou renormalizado incorretamente.")

# Conferir a idade contra os timestamps que realmente foram usados na aplicacao.
sm_age_check = sm_prev.join(sm_input.select("cd_bv", "ts_inicio"), "cd_bv", "left")
sm_exigir(not sm_ha(sm_age_check.filter(
    F.col("ts_inicio").isNull()
    | (F.abs(sm_segundos("ts_inicio", "ts_corte_estado") - F.col("tempo_no_estado_seg")) > 1e-6)
)), "Tempo utilizado nao corresponde ao relogio de entrada no estado.")

# COMMAND ----------
# Recalculo independente usando scipy.stats.lognorm.logsf; NAO usa a funcao
# prever_destinos que produziu o output. Verifica normalizador com TODOS destinos.
def sm_auditar_temporal_pdf(pdf: pd.DataFrame, modelos: dict, horizonte: float) -> pd.DataFrame:
    n_checadas = 0
    n_erros = 0
    max_erro = 0.0
    for origem, parte in pdf.groupby("acao_atual", sort=False):
        m = modelos.get(origem)
        if m is None:
            n_erros += len(parte)
            continue
        g = np.asarray(m["grupo"], int)
        mu = np.asarray(m["mu_grupo"])[g]
        sig = np.asarray(m["sigma_grupo"])[g]
        lp = np.log(np.asarray(m["p_destino"]))
        mapa = {d: j for j, d in enumerate(m["destinos"])}
        for _, row in parte.iterrows():
            n_checadas += 1
            j = mapa.get(row["proxima_acao"])
            if j is None or row["id_modelo_origem"] != m["id_modelo_origem"]:
                n_erros += 1
                continue
            a = float(row["tempo_no_estado_seg"]) / 86400.
            ls = stats.lognorm.logsf(a, s=sig, scale=np.exp(mu))
            den = special.logsumexp(lp + ls)
            q = np.exp(lp + ls - den)
            lsh = stats.lognorm.logsf(a + horizonte, s=sig, scale=np.exp(mu))
            qh = q[j] * (-np.expm1(lsh[j] - ls[j]))
            ns = np.exp(special.logsumexp(lp + lsh) - den)
            ordem = np.argsort(-q, kind="mergesort")[:TOP_K]
            posto = int(row["ranking"]) - 1
            err = max(abs(q[j] - row["prob_proxima_acao"]),
                      abs(qh - row["prob_proxima_acao_7d"]),
                      abs(ns - row["prob_sem_saida_7d"]),
                      abs(q[ordem].sum() - row["massa_top5"]))
            max_erro = max(max_erro, float(err))
            if (not np.isfinite(err) or err > SM_TOL or posto >= len(ordem)
                    or j != ordem[posto]):
                n_erros += 1
    return pd.DataFrame([dict(n_checadas=n_checadas, n_erros=n_erros, max_erro=max_erro)])


def sm_auditar_lotes(iterator):
    modelos = SM_BROADCAST_MODELOS.value
    for pdf in iterator:
        yield sm_auditar_temporal_pdf(pdf, modelos, SM_CFG["horizonte_dias"])


if sm_ha(sm_prev):
    sm_audit = sm_prev.mapInPandas(
        sm_auditar_lotes, "n_checadas long, n_erros long, max_erro double"
    ).agg(F.sum("n_checadas").alias("n_checadas"), F.sum("n_erros").alias("n_erros"),
          F.max("max_erro").alias("max_erro")).first().asDict()
    sm_exigir(sm_audit["n_erros"] == 0, "Recalculo independente detectou inconsistencias temporais.")
else:
    sm_audit = {"n_checadas": 0, "n_erros": 0, "max_erro": 0.0}
    raise ValueError("Nenhuma previsao calculada: revisar suporte e estados; nao publicar output vazio de previsoes.")
print("Auditoria temporal independente:", sm_audit)

sm_resumo = sm_saida.agg(
    F.count("*").alias("n_linhas"), F.countDistinct("cd_bv").alias("n_clientes"),
    F.countDistinct(F.when(F.col("ranking") == 1, F.col("cd_bv"))).alias("n_clientes_previstos"),
).first().asDict()
sm_metricas = [r.asDict() for r in spark.table("nba_sm_v2_validacao").collect()]
print("Resumo do output:", sm_resumo)
print("Estado e idade sao no corte:", SM_CFG["corte_estado_exclusivo"])
print("Publico selecionado na referencia:", SM_CFG["data_publico"])
print("Probabilidades nao estao certificadas como calibradas; flags de cobertura nao sao inferidas de contagens.")

# COMMAND ----------
# Persistencia segura em tres tabelas novas. A escrita de cada tabela e atomica,
# mas as tres juntas nao sao uma transacao; consumo exige manifesto CONCLUIDA.
sm_modelos_hml = (
    sm_mod.withColumn("data_referencia", F.lit(SM_CFG["data_publico"]).cast("date"))
    .withColumn("id_execucao", F.lit(SM_ID_EXECUCAO))
    .withColumn("ambiente", F.lit("HOMOLOGACAO"))
    .withColumn("gravado_em", F.lit(SM_INICIO).cast("timestamp"))
)
sm_output_hml = (
    sm_saida.join(sm_input.select(
        "cd_bv", "ts_inicio", "ts_ultima_atividade", "ultima_acao_observada"
    ), "cd_bv", "left")
    .withColumn("id_execucao", F.lit(SM_ID_EXECUCAO))
    .withColumn("ambiente", F.lit("HOMOLOGACAO"))
    .withColumn("gravado_em", F.lit(SM_INICIO).cast("timestamp"))
)


def sm_manifesto(status: str, erro: str | None = None):
    return spark.createDataFrame([(
        SM_CFG["data_publico"], sm_versao, SM_ID_EXECUCAO, status, erro,
        json.dumps(SM_CFG, ensure_ascii=False, sort_keys=True),
        json.dumps(sm_audit, allow_nan=False), json.dumps(sm_resumo, allow_nan=False),
        json.dumps(sm_metricas, allow_nan=False),
    )], "data_ref_string string, versao_modelo string, id_execucao string, "
        "status_execucao string, erro string, config_json string, auditoria_json string, "
        "resumo_json string, validacao_json string").withColumn(
        "data_referencia", F.col("data_ref_string").cast("date")
    ).drop("data_ref_string").withColumn("ambiente", F.lit("HOMOLOGACAO")).withColumn(
        "gravado_em", F.lit(datetime.now(timezone.utc)).cast("timestamp")
    ).withColumn("publicavel", F.lit(False))


def sm_gravar_recorte(df, tabela: str) -> None:
    sm_exigir(tabela in SM_TABELAS.values(), "Destino nao autorizado.")
    sm_exigir(sm_ha(df), "Entrada vazia: escrita recusada.")
    sm_exigir(not sm_ha(df.filter(~F.coalesce(F.expr(SM_PREDICADO), F.lit(False)))),
              "Entrada contem dados fora do recorte.")
    if spark.catalog.tableExists(tabela):
        fmt = spark.sql(f"DESCRIBE DETAIL {tabela}").select("format").first()[0]
        sm_exigir(fmt.lower() == "delta", "Destino existente nao e Delta.")
        velho = {c.name: c.dataType for c in spark.table(tabela).schema}
        novo = {c.name: c.dataType for c in df.schema}
        sm_exigir(velho == novo, "Schema mudou: revisar, sem overwriteSchema automatico.")
        df.write.format("delta").mode("overwrite").option(
            "replaceWhere", SM_PREDICADO
        ).saveAsTable(tabela)
    else:
        df.write.format("delta").mode("errorifexists").saveAsTable(tabela)


if GRAVAR_HOMOLOGACAO:
    # Impede reutilizar a mesma versao para parametros diferentes em outra referencia.
    if spark.catalog.tableExists(SM_TABELAS["modelos"]):
        anteriores = spark.table(SM_TABELAS["modelos"]).filter(
            F.col("versao_modelo") == sm_versao
        ).select("origem", F.col("id_modelo_origem").alias("hash_anterior")).distinct()
        if sm_ha(anteriores):
            antigas_origens = anteriores.select("origem").distinct()
            novas_origens = sm_mod.select("origem").distinct()
            sm_exigir(
                not sm_ha(antigas_origens.join(novas_origens, "origem", "left_anti"))
                and not sm_ha(novas_origens.join(antigas_origens, "origem", "left_anti")),
                "Conjunto de origens mudou: incremente versao_modelo.",
            )
        conflito = sm_mod.join(anteriores, "origem", "inner").filter(
            ~F.col("id_modelo_origem").eqNullSafe(F.col("hash_anterior"))
        )
        sm_exigir(not sm_ha(conflito), "Parametros mudaram: incremente versao_modelo.")
    sm_gravar_recorte(sm_manifesto("EM_GRAVACAO"), SM_TABELAS["execucoes"])
    try:
        sm_gravar_recorte(sm_modelos_hml, SM_TABELAS["modelos"])
        sm_gravar_recorte(sm_output_hml, SM_TABELAS["previsoes"])
        n_salvo = spark.table(SM_TABELAS["previsoes"]).filter(
            F.expr(SM_PREDICADO) & (F.col("id_execucao") == SM_ID_EXECUCAO)
        ).count()
        sm_exigir(n_salvo == sm_resumo["n_linhas"], "Contagem gravada diverge.")
        n_modelos_salvos = spark.table(SM_TABELAS["modelos"]).filter(
            F.expr(SM_PREDICADO) & (F.col("id_execucao") == SM_ID_EXECUCAO)
        ).count()
        sm_exigir(n_modelos_salvos == sm_mod.count(), "Contagem de modelos diverge.")
        sm_gravar_recorte(sm_manifesto("CONCLUIDA"), SM_TABELAS["execucoes"])
    except Exception as exc:
        try:
            sm_gravar_recorte(sm_manifesto("FALHOU", type(exc).__name__), SM_TABELAS["execucoes"])
        except Exception:
            pass  # A excecao original e relancada; nao ocultamos falha de publicacao.
        raise
    print("Homologacao persistida. Execucao:", SM_ID_EXECUCAO)
else:
    print("Nenhuma tabela gravada. Apos revisar, altere GRAVAR_HOMOLOGACAO=True e execute esta celula.")

consulta_top1 = f"""
SELECT p.data_referencia, p.cd_bv, p.acao_atual, p.tempo_no_estado_seg,
       p.proxima_acao, p.prob_proxima_acao, p.prob_proxima_acao_7d,
       p.ts_corte_estado, p.status_dados, p.status_temporal, p.versao_modelo
FROM {SM_TABELAS['previsoes']} p
INNER JOIN {SM_TABELAS['execucoes']} e
  ON p.id_execucao = e.id_execucao
 AND p.versao_modelo = e.versao_modelo
 AND p.data_referencia = e.data_referencia
WHERE e.status_execucao = 'CONCLUIDA'
  AND p.data_referencia = DATE '{SM_CFG['data_publico']}'
  AND p.versao_modelo = '{sm_versao}'
  AND p.ranking = 1
"""
print(consulta_top1)
print("Para top 5 remova apenas p.ranking = 1. Probabilidades ja incluem a idade.")
