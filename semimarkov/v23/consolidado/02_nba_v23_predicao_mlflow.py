# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — NBA V2.3 | Predição a partir do MLflow
# MAGIC Independente do notebook 01 e das variáveis de sua sessão. **Não treina.**
# MAGIC Escolha explicitamente a URI/run aprovada. Sem configuração, apenas lista execuções disponíveis; não escolhe a mais recente automaticamente.
# MAGIC A preparação de estado/memória é carregada do pacote, junto com catálogo, parâmetros e coeficientes B2.
# MAGIC Fonte de público é apenas elegibilidade geral; não aplica elegibilidade de produtos nem regras de comunicação.
# MAGIC Requer o mesmo ambiente Python do treinamento nos executores (`env_manager="local"`), Spark >=3.5 e MLflow >=2.12.2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração e seleção do modelo

# COMMAND ----------

import json
import math
import uuid
from datetime import datetime, timezone
import mlflow
import numpy as np
import pandas as pd
from packaging.version import Version
from pyspark.sql import Window
from pyspark.sql import functions as F

CFG = {
    # Escolha exatamente UMA alternativa. Cole a URI impressa no notebook 01.
    'model_uri': '',
    'run_id_treinamento': '',  # Alternativa: ID de uma run concluída deste pacote.
    'experimento_mlflow': '/Shared/NBA/V23_B2',
    'tracking_uri': 'databricks',
    'fonte_publico': 'customers_query',  # Em job: tabela/view permanente ou ingestão explícita.
    'coluna_data_publico': 'AUTO',  # dt_snapshot_publico ou data_snapshot_publico; senão NULL.
    'fonte_eventos': 'ctg_dsti.renato_nba.base_jornadas_onboarding',
    'versao_eventos': None,
    'corte_eventos_iso': None,  # Watermark explícito com timezone; None usa máximo observado <= agora.
    'inicio_eventos_iso': None,  # None preserva todo o histórico disponível para memória.
    'max_atraso_eventos_horas': 72.0,
    'max_atraso_publico_dias': 7,
    'modo_scoring': 'AMOSTRA',  # AMOSTRA, TODOS ou IDS.
    'modulo_amostra': 1000,
    'sal_amostra': 'nba_v23_operacional',
    'ids_validacao': [],  # IDs adicionais só entram se estiverem no público elegível.
    'prefixo_saida': 'ctg_dsti.renato_nba.nba_v23_mlflow_scoring',
    'autotestes': True,
    'arrow_lote': 2048,
}
SCORE_ID = str(uuid.uuid4())
INICIO = datetime.now(timezone.utc)
TABELAS = {x: f"{CFG['prefixo_saida']}_{x}_hml" for x in (
    'publico', 'estados', 'predicoes', 'negocio', 'execucoes')}
# Não seleciona automaticamente a run mais recente; permite runs:/ e models:/ explícitos.

# COMMAND ----------

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pyspark.sql import functions as F


def nome_sql(nome):
    partes = nome.split('.')
    if not 1 <= len(partes) <= 3 or not all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', p) for p in partes):
        raise ValueError(f'Nome SQL inválido: {nome}')
    return '.'.join(f'`{p}`' for p in partes)


def timestamp_utc(texto):
    dt = datetime.fromisoformat(texto.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('A data deve informar offset/timezone. Exemplo: 2026-09-25T00:00:00+00:00')
    return dt.astimezone(timezone.utc)


def ler_delta_fixo(spark, tabela, versao=None):
    """Captura a versão atual UMA vez. Nenhuma troca por versão nova no meio do job."""
    if not spark.catalog.tableExists(tabela):
        raise ValueError(f'Fonte não encontrada: {tabela}')
    detail = spark.sql(f'DESCRIBE DETAIL {nome_sql(tabela)}').first().asDict()
    if detail.get('format', '').lower() != 'delta':
        raise ValueError(f'Fonte de eventos deve ser Delta: {tabela}')
    if versao is None:
        versao = int(spark.sql(f'DESCRIBE HISTORY {nome_sql(tabela)} LIMIT 1').first()['version'])
    df = spark.read.option('versionAsOf', int(versao)).table(tabela)
    df.limit(1).collect()
    print(f'Fonte fixa: {tabela} | versão={versao}')
    return df, {'tabela': tabela, 'versao_delta': int(versao),
                'table_id': detail.get('id'), 'location': detail.get('location')}


def conferir_identidade(spark, meta):
    current = spark.sql(f"DESCRIBE DETAIL {nome_sql(meta['tabela'])}").first().asDict()
    if current.get('id') != meta['table_id'] or current.get('location') != meta['location']:
        raise RuntimeError('A identidade da fonte mudou durante a execução.')
    # Novos commits são permitidos: todas as leituras usam a versão fixada.


def salvar_etapa(spark, df, tabela, campo_id, valor_id):
    """Append por execução; reexecução parcial com mesmo ID é bloqueada."""
    nome_sql(tabela)
    if spark.catalog.tableExists(tabela):
        if spark.table(tabela).filter(F.col(campo_id) == valor_id).limit(1).count():
            raise RuntimeError(f'{tabela}: ID já gravado. Reinicie o notebook para nova execução.')
    out = df.withColumn(campo_id, F.lit(valor_id))
    out.write.format('delta').mode('append').partitionBy(campo_id).saveAsTable(tabela)
    return spark.table(tabela).filter(F.col(campo_id) == valor_id)


def exibir(nome, df, limite=30):
    print('\n' + nome)
    df.show(limite, truncate=False)


def exigir_colunas(df, cols, label):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'{label}: colunas ausentes {sorted(missing)}')

# COMMAND ----------

if 'spark' not in globals():
    raise RuntimeError('Execute no Databricks com sessão Spark.')
if Version(spark.version.split('-')[0]) < Version('3.5.0'):
    raise RuntimeError('Requer Spark >= 3.5.')
if Version(mlflow.__version__) < Version('2.12.2'):
    raise RuntimeError('Requer MLflow >= 2.12.2 para carregar o modelo por código.')
mlflow.set_tracking_uri(CFG['tracking_uri'])
if CFG['model_uri'] and CFG['run_id_treinamento']:
    raise ValueError('Preencha model_uri OU run_id_treinamento, não os dois.')
if not CFG['model_uri'] and not CFG['run_id_treinamento']:
    experiment = mlflow.get_experiment_by_name(CFG['experimento_mlflow'])
    if experiment:
        runs = mlflow.search_runs([experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED' AND tags.arquitetura = 'NBA_V23_B2'",
            order_by=['attributes.start_time DESC'], max_results=30)
        cols = [x for x in ['run_id', 'start_time', 'tags.mlflow.runName', 'tags.model_uri',
                'params.corte_historico', 'params.modulo_amostra', 'metrics.n_clientes_treino',
                'tags.estado_pacote'] if x in runs.columns]
        display(runs[cols])
    raise ValueError('Escolha uma run na tabela acima e preencha model_uri (ou run_id_treinamento) na configuração. Nenhum modelo foi escolhido automaticamente.')
URI_SOLICITADA = CFG['model_uri']
if not URI_SOLICITADA:
    rr = mlflow.get_run(CFG['run_id_treinamento'])
    URI_SOLICITADA = rr.data.tags.get('model_uri', '')
    if not URI_SOLICITADA:
        raise ValueError('A run escolhida não contém model_uri registrada por este treinamento.')
loaded = mlflow.pyfunc.load_model(URI_SOLICITADA)
model_api = loaded.unwrap_python_model()
CONTRATO = model_api.contrato()
RUN_MODELO = CONTRATO['run_id_treinamento']
r = mlflow.get_run(RUN_MODELO)
if r.info.status != 'FINISHED' or r.data.tags.get('estado_pacote') != 'CONCLUIDO':
    raise ValueError('Treinamento não concluído: não consumir um pacote parcial.')
# Fixa a URI imutável retornada no treinamento; aliases não são re-resolvidos nos executores.
MODEL_URI = r.data.tags.get('model_uri')
if not MODEL_URI:
    raise ValueError('URI imutável do modelo não registrada.')
spark.conf.set('spark.sql.session.timeZone', CONTRATO['preparo']['fuso'])
spark.conf.set('spark.sql.execution.arrow.maxRecordsPerBatch', str(CFG['arrow_lote']))
if 'nba_v23_mlflow' not in CFG['prefixo_saida']:
    raise ValueError('Conserve prefixo separado nba_v23_mlflow; não sobrescreva as views antigas.')
if CFG['modo_scoring'] not in {'AMOSTRA', 'TODOS', 'IDS'} or CFG['modulo_amostra'] < 1:
    raise ValueError('Modo/módulo de scoring inválido.')
if CFG['modo_scoring'] == 'IDS' and not CFG['ids_validacao']:
    raise ValueError('IDS exige pelo menos um identificador.')
if CFG['autotestes']:
    model_api.teste_preparo(spark)
print('Modelo:', MODEL_URI)
print('Run de treinamento:', RUN_MODELO)
print('Ações úteis:', CONTRATO['alvos'])
print('Run de scoring:', SCORE_ID, '| modo:', CFG['modo_scoring'])


def salvar(df, nome):
    out = df.withColumn('run_id_treinamento', F.lit(RUN_MODELO)).withColumn('model_uri', F.lit(MODEL_URI))
    out = out.withColumn('modo_scoring', F.lit(CFG['modo_scoring']))
    return salvar_etapa(spark, out, TABELAS[nome], 'id_execucao', SCORE_ID)


OUTPUT_SCHEMA = (
    'status_modelo string, tem_previsao boolean, rota_modelo string, '
    'raw_top5_json string, util_top5_json string, raw_probs_json string, '
    'util_probs_json string, massa_acao_util double, massa_sem_modelo double, '
    'massa_terminal double, massa_retencao_interna double, massa_limite_macro double, '
    'erro_massa double, extrapolacao_tempo boolean, alerta_limite boolean'
)
model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=MODEL_URI, result_type=OUTPUT_SCHEMA, env_manager='local')
if CFG['autotestes']:
    exemplos = CONTRATO['exemplo_inferencia']
    exemplo_pd = pd.DataFrame(exemplos)[['estado', 'memoria', 'idade_seg']]
    esperado = loaded.predict(exemplo_pd)
    pequeno = spark.createDataFrame([(i, r['estado'], r['memoria'], float(r['idade_seg']))
        for i, r in enumerate(exemplos)], 'ordem int, estado string, memoria string, idade_seg double')
    observado = pequeno.withColumn('_p', model_udf(F.struct('estado', 'memoria', 'idade_seg')))\
        .select('ordem', '_p.*').orderBy('ordem').collect()
    for i, row in enumerate(observado):
        for col in esperado.columns:
            a, b = esperado.iloc[i][col], row[col]
            if isinstance(a, (float, np.floating)):
                if not np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True):
                    raise RuntimeError(f'UDF distribuída divergiu: {col}')
            elif a != b:
                raise RuntimeError(f'UDF distribuída divergiu: {col}')
    print('PRED_00: carregamento distribuído reproduziu a inferência do driver.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Público, corte operacional, estado e memória
# MAGIC O público é materializado por execução. Os eventos Delta são fixados por versão no começo; novos commits não alteram a leitura.
# MAGIC Máximo timestamp observado não prova completude da carga. Para produção, prefira um watermark confiável.
# MAGIC `customers_query` precisa existir nesta sessão; um job independente deve receber uma fonte permanente ou executar a ingestão antes.

# COMMAND ----------

if not spark.catalog.tableExists(CFG['fonte_publico']):
    raise ValueError('Fonte de público indisponível. Disponibilize customers_query nesta sessão ou configure uma tabela/view permanente; não repita o treino.')
pub = spark.table(CFG['fonte_publico'])
exigir_colunas(pub, {'cd_bv'}, 'Público')
datecol = CFG['coluna_data_publico']
if datecol == 'AUTO':
    datecol = next((c for c in ('dt_snapshot_publico', 'data_snapshot_publico') if c in pub.columns), None)
if datecol:
    exigir_colunas(pub, {datecol}, 'Público')
    pub = pub.select(F.col('cd_bv').cast('string'), F.to_date(datecol).alias('data_snapshot_publico'))
    date_pub = pub.agg(F.max('data_snapshot_publico').alias('d')).first()['d']
    if date_pub is None:
        raise ValueError('Data do público não encontrada.')
    # Mesma convenção de fuso do modelo; não confundir momento de leitura com data da fonte.
    atraso = spark.range(1).select(F.datediff(F.to_date(F.lit(INICIO)), F.lit(date_pub)).alias('d')).first()['d']
    if atraso < 0 or atraso > CFG['max_atraso_publico_dias']:
        raise ValueError('Público futuro/desatualizado para a tolerância configurada.')
    pub = pub.filter(F.col('data_snapshot_publico') == F.lit(date_pub))
else:
    date_pub = None
    pub = pub.select(F.col('cd_bv').cast('string')).withColumn('data_snapshot_publico', F.lit(None).cast('date'))
    print('Público sem data de snapshot: permanece NULL. Não foi criado D-1 artificial.')
pub = pub.filter(F.col('cd_bv').isNotNull() & (F.length(F.trim('cd_bv')) > 0)).dropDuplicates(['cd_bv'])
if CFG['modo_scoring'] == 'AMOSTRA':
    mask = F.pmod(F.xxhash64('cd_bv', F.lit(CFG['sal_amostra'])), F.lit(CFG['modulo_amostra'])) == 0
    if CFG['ids_validacao']:
        mask = mask | F.col('cd_bv').isin([str(x) for x in CFG['ids_validacao']])
    pub = pub.filter(mask)
elif CFG['modo_scoring'] == 'IDS':
    pub = pub.filter(F.col('cd_bv').isin([str(x) for x in CFG['ids_validacao']]))
# Materializa inclusive uma view temporária; as outras etapas usam este recorte persistido.
publico = salvar(pub, 'publico')
N_PUBLICO = publico.count()
if not N_PUBLICO:
    raise ValueError('Público selecionado vazio. Revise fonte/modo/amostra/IDs.')
fonte, META_FONTE = ler_delta_fixo(spark, CFG['fonte_eventos'], CFG['versao_eventos'])
exigir_colunas(fonte, {'cd_bv', 'dm_navegacao', 'estado', 'profundidade_max'}, 'Eventos')
if fonte.schema['dm_navegacao'].dataType.simpleString() != 'timestamp':
    raise ValueError('dm_navegacao precisa ser timestamp.')
maior = fonte.filter(F.col('dm_navegacao') <= F.lit(INICIO)).agg(F.max('dm_navegacao').alias('ts')).first()['ts']
if maior is None:
    raise ValueError('Eventos sem timestamp válido até o início da execução.')
# Não converte datetime coletado sem timezone assumindo fuso local do driver.
max_us = fonte.filter(F.col('dm_navegacao') <= F.lit(INICIO)).agg(F.max(F.expr('unix_micros(dm_navegacao)')).alias('v')).first()['v']
CORTE = timestamp_utc(CFG['corte_eventos_iso']) if CFG['corte_eventos_iso'] else datetime.fromtimestamp(max_us / 1e6, timezone.utc)
if CORTE > INICIO:
    raise ValueError('Corte operacional no futuro.')
ultimo_corte_us = fonte.filter(F.col('dm_navegacao') <= F.lit(CORTE)).agg(F.max(F.expr('unix_micros(dm_navegacao)')).alias('v')).first()['v']
if ultimo_corte_us is None or (INICIO.timestamp() - ultimo_corte_us / 1e6) / 3600 > CFG['max_atraso_eventos_horas']:
    raise ValueError('Eventos ausentes/desatualizados no corte escolhido. Atualize a ingestão; não repita o treino.')
print('Corte operacional:', CORTE.isoformat(), '| público:', N_PUBLICO)
print('Regra de corte:', 'WATERMARK_INFORMADO' if CFG['corte_eventos_iso'] else 'MAXIMO_OBSERVADO_NAO_COMPROVA_COMPLETUDE')
eventos = fonte.select(F.col('cd_bv').cast('string'), F.col('dm_navegacao').alias('ts_evento'),
    F.col('estado').cast('string'), F.col('profundidade_max').cast('double')).join(publico.select('cd_bv'), 'cd_bv', 'left_semi')
eventos = eventos.filter((F.col('ts_evento') <= F.lit(CORTE)) | F.col('ts_evento').isNull())
if CFG['inicio_eventos_iso']:
    eventos = eventos.filter((F.col('ts_evento') >= F.lit(timestamp_utc(CFG['inicio_eventos_iso']))) | F.col('ts_evento').isNull())
model_api.validar_eventos(eventos)
estados = model_api.preparar_estados(eventos, publico.select('cd_bv', 'data_snapshot_publico'), CORTE.isoformat())
estados = estados.withColumn('ts_execucao', F.lit(INICIO))
estados = salvar(estados, 'estados')
if estados.count() != N_PUBLICO or estados.groupBy('cd_bv').count().filter('count != 1').limit(1).count():
    raise RuntimeError('A preparação perdeu/duplicou clientes.')
conferir_identidade(spark, META_FONTE)
exibir('PRED_01_ESTADOS', estados.groupBy('status_dado').count().orderBy(F.desc('count')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Scoring distribuído via MLflow
# MAGIC A URI é fixada para os executores. Os coeficientes são carregados uma vez por instância de modelo, sem reconstruir o grafo para cada cliente.
# MAGIC Scores úteis não são renormalizados. Ações desconhecidas e origens sem modelo não recebem recomendações inventadas.

# COMMAND ----------

# result_type é explícito; saídas JSON evitam incompatibilidades Arrow com estruturas aninhadas.
entrada = estados.filter("status_dado = 'OK'").select('cd_bv', 'estado', 'memoria', F.col('idade_seg').cast('double'))
calculado = entrada.withColumn('_p', model_udf(F.struct('estado', 'memoria', 'idade_seg'))).select('cd_bv', '_p.*')
pred = estados.join(calculado, 'cd_bv', 'left')
pred = pred.withColumn('status_previsao', F.when(F.col('status_dado') != 'OK', F.col('status_dado'))
    .when(F.col('tem_previsao'), 'PREVISAO_B2_OPERACIONAL').otherwise(F.col('status_modelo')))
pred = pred.withColumn('tem_previsao', F.coalesce('tem_previsao', F.lit(False)))
raw_schema = 'array<struct<ranking:int,estado:string,score:double>>'
util_schema = 'array<struct<ranking:int,acao:string,score:double,passo_macro_maior_contribuicao:int>>'
pred = pred.withColumn('raw_top5', F.from_json('raw_top5_json', raw_schema))
pred = pred.withColumn('util_top5', F.from_json('util_top5_json', util_schema))
pred = pred.withColumn('util_probs', F.from_json('util_probs_json', 'map<string,double>'))
pred = pred.drop('raw_top5_json', 'util_top5_json', 'raw_probs_json', 'util_probs_json')
pred = salvar(pred, 'predicoes')
if pred.filter("status_dado = 'OK' AND status_previsao IS NULL").limit(1).count():
    raise RuntimeError('Cliente válido não recebeu status do modelo.')
if pred.filter(F.col('tem_previsao') & (
        F.col('erro_massa').isNull() | F.isnan('erro_massa') | (F.col('erro_massa') > 1e-8))).limit(1).count():
    raise RuntimeError('Conservação de massa falhou.')
if pred.groupBy('cd_bv').count().filter('count != 1').limit(1).count():
    raise RuntimeError('Predição duplicada por cliente.')
N_PREVISTOS = pred.filter('tem_previsao').count()
exibir('PRED_02_STATUS', pred.groupBy('status_previsao', 'rota_modelo').count().orderBy(F.desc('count')))
exibir('PRED_03_TOP1_UTIL', pred.filter('tem_previsao').withColumn('acao_top1', F.expr('try_element_at(util_top5, 1)')['acao'])
       .groupBy('acao_top1').count().orderBy(F.desc('count')))
exibir('PRED_04_MASSAS', pred.filter('tem_previsao').agg(*[F.avg(c).alias(c) for c in (
    'massa_acao_util', 'massa_sem_modelo', 'massa_terminal', 'massa_retencao_interna', 'massa_limite_macro')]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Saída LONG, manifesto e view
# MAGIC Até cinco linhas por cliente, RAW e útil como rankings independentes. A tabela técnica preserva status e os scores completos dos 17 alvos.
# MAGIC Tabelas em append por execução. A view aponta somente para uma execução concluída do modo/modelo selecionado. Nenhuma tabela anterior é sobrescrita.
# MAGIC Este código não autoriza disparos automáticos.

# COMMAND ----------

base_long = pred.filter('tem_previsao').withColumn('_n', F.greatest(F.size('raw_top5'), F.size('util_top5')))
base_long = base_long.filter('_n > 0').withColumn('ranking', F.explode(F.sequence(F.lit(1), F.col('_n'))))
base_long = base_long.withColumn('_raw', F.expr('try_element_at(raw_top5, ranking)'))
base_long = base_long.withColumn('_util', F.expr('try_element_at(util_top5, ranking)'))
negocio = base_long.select('data_referencia', 'cd_bv', 'acao_atual', 'ranking',
    F.col('_raw.estado').alias('proxima_acao_raw'), F.col('_raw.score').alias('score_raw'),
    F.col('_util.acao').alias('proxima_acao_util'), F.col('_util.score').alias('score_util'))
if negocio.groupBy('cd_bv', 'ranking').count().filter('count != 1').limit(1).count():
    raise RuntimeError('Ranking duplicado na saída de negócio.')
negocio = salvar(negocio, 'negocio')
N_LINHAS = negocio.count()
conferir_identidade(spark, META_FONTE)
manifesto = spark.createDataFrame([(
    'CONCLUIDO_SCORING_B2', INICIO.isoformat(), datetime.now(timezone.utc).isoformat(),
    int(round(CORTE.timestamp() * 1e6)), N_PUBLICO, N_PREVISTOS, N_LINHAS,
    json.dumps({'config': CFG, 'fonte_eventos': META_FONTE, 'model_uri_solicitada': URI_SOLICITADA,
                'model_uri_fixada': MODEL_URI, 'contrato': CONTRATO, 'tabelas': TABELAS}, ensure_ascii=False),
    False,
)], 'status string, inicio_utc string, fim_utc string, corte_micros long, n_elegiveis long, '
    'n_previstos long, n_linhas_negocio long, config_json string, autorizado_disparo boolean')
salvar(manifesto, 'execucoes')
suffix = {'TODOS': 'negocio_atual', 'AMOSTRA': 'amostra_atual', 'IDS': 'ids_atual'}[CFG['modo_scoring']]
VIEW = CFG['prefixo_saida'] + '_' + suffix
# A view não inclui saída de execução parcial e não mistura modelos/modos.
spark.sql(f"""
CREATE OR REPLACE VIEW {nome_sql(VIEW)} AS
WITH ultima AS (
    SELECT id_execucao FROM {nome_sql(TABELAS['execucoes'])}
    WHERE status = 'CONCLUIDO_SCORING_B2'
      AND modo_scoring = '{CFG['modo_scoring']}'
      AND run_id_treinamento = '{RUN_MODELO}'
    ORDER BY corte_micros DESC, fim_utc DESC, id_execucao DESC LIMIT 1
)
SELECT n.data_referencia, n.cd_bv, n.acao_atual, n.ranking,
       n.proxima_acao_raw, n.score_raw, n.proxima_acao_util, n.score_util
FROM {nome_sql(TABELAS['negocio'])} n INNER JOIN ultima u ON n.id_execucao = u.id_execucao
""")
print('\nPRED_05_CONCLUSAO')
print('id_execucao:', SCORE_ID)
print('Modelo utilizado:', MODEL_URI)
print('Público:', N_PUBLICO, '| previstos:', N_PREVISTOS, '| linhas negócio:', N_LINHAS)
print('VIEW PARA AVALIAÇÃO:', VIEW)
print('Não houve treinamento, elegibilidade por produto ou disparos automáticos.')
