"""Preparação comum ao treinamento e scoring. Regras da V2.3 preservadas."""
import math
import re
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

MS_SEM_MEMORIA = "__SEM_MEMORIA__"
MS_BASE = "__BASE__"
def op_preparar_acoes(eventos: DataFrame, config: DataFrame,
                     tecnicas: set, debounce_seg: float) -> DataFrame:
    # Recebe somente cd_bv, ts_evento, estado, profundidade_max, até o corte.
    por_estado = eventos.groupBy("cd_bv", "ts_evento", "estado").agg(
        F.max("profundidade_max").alias("profundidade_max"),
        F.count("*").alias("n_ev"),
    ).withColumn("_prof", F.coalesce("profundidade_max", F.lit(-1.0)))
    wi = Window.partitionBy("cd_bv", "ts_evento")
    candidatos = por_estado.withColumn("_max", F.max("_prof").over(wi))
    candidatos = candidatos.withColumn("_eh_max", F.col("_prof") == F.col("_max"))
    instantes = candidatos.groupBy("cd_bv", "ts_evento").agg(
        F.sum(F.col("_eh_max").cast("int")).alias("n_max"),
        F.max(F.when(F.col("_eh_max"), F.col("estado"))).alias("_estado_max"),
        F.sum("n_ev").alias("n_ev"),
    ).withColumn("ordem_ambigua", F.col("n_max") != 1)
    instantes = instantes.withColumn(
        "estado", F.when(~F.col("ordem_ambigua"), F.col("_estado_max"))
    )
    w = Window.partitionBy("cd_bv").orderBy("ts_evento")
    wa = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = instantes.withColumn("_s_ant", F.lag("estado").over(w))
    df = df.withColumn("_t_ant", F.lag("ts_evento").over(w))
    df = df.withColumn("_a_ant", F.lag("ordem_ambigua").over(w))
    gap = (F.expr("unix_micros(ts_evento)") - F.expr("unix_micros(_t_ant)")) / 1e6
    repetir = (
        F.col("estado").isNotNull() & F.col("_s_ant").isNotNull()
        & ~F.col("ordem_ambigua") & ~F.coalesce("_a_ant", F.lit(False))
        & (F.col("estado") == F.col("_s_ant")) & (gap <= debounce_seg)
    )
    df = df.withColumn("_abriu", F.when(repetir, 0).otherwise(1))
    df = df.withColumn("_burst", F.sum("_abriu").over(wa))
    acoes = df.groupBy("cd_bv", "_burst").agg(
        F.max("estado").alias("estado"),
        F.max("ts_evento").alias("ts_estado"),
        F.max(F.col("ordem_ambigua").cast("int")).cast("boolean").alias("ordem_ambigua"),
        F.sum("n_ev").alias("n_eventos_consolidados"),
    )
    w = Window.partitionBy("cd_bv").orderBy("ts_estado", "_burst")
    wa = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    acoes = acoes.withColumn("passo", F.row_number().over(w)).drop("_burst")
    w = Window.partitionBy("cd_bv").orderBy("passo")
    wa = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    cat = config.select("estado", "funil", "etapa", "tipo_estado").withColumn(
        "estado_catalogado", F.lit(True)
    )
    acoes = acoes.join(F.broadcast(cat), "estado", "left")
    acoes = acoes.withColumn("estado_catalogado", F.coalesce("estado_catalogado", F.lit(False)))
    acoes = acoes.withColumn(
        "barreira_memoria",
        F.col("ordem_ambigua") | ~F.col("estado_catalogado"),
    ).withColumn(
        "origem_tecnica", F.coalesce(F.col("funil").isin(sorted(tecnicas)), F.lit(False))
    )
    acoes = acoes.withColumn("_barreira_ant", F.lag("barreira_memoria").over(w))
    acoes = acoes.withColumn("_novo_trecho", F.when(
        (F.col("passo") == 1) | F.col("barreira_memoria")
        | F.coalesce("_barreira_ant", F.lit(False)), 1
    ).otherwise(0))
    acoes = acoes.withColumn("_trecho", F.sum("_novo_trecho").over(wa))
    wp = Window.partitionBy("cd_bv", "_trecho").orderBy("passo").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    evidence = ~F.col("barreira_memoria") & F.col("funil").isNotNull() & ~F.col("origem_tecnica")
    acoes = acoes.withColumn("_evidencia", F.when(evidence, F.struct(
        F.col("funil").alias("acao"), F.col("passo").alias("passo"),
        F.col("ts_estado").alias("ts"),
    )))
    acoes = acoes.withColumn("_memoria", F.last("_evidencia", ignorenulls=True).over(wp))
    return acoes.select(
        "cd_bv", "passo", "ts_estado", "estado", "funil", "etapa", "tipo_estado",
        "ordem_ambigua", "estado_catalogado", "barreira_memoria", "origem_tecnica",
        "n_eventos_consolidados",
        F.col("_memoria.acao").alias("contexto_negocio"),
        F.col("_memoria.passo").alias("passo_evidencia_memoria"),
        F.col("_memoria.ts").alias("ts_evidencia_memoria"),
    )

def ms03_rotular(base: DataFrame, alvos: list, k: int, tolerancia: float = 1e-3) -> DataFrame:
    """Primeira entrada em macro alvo diferente, em até k mudanças observadas.

    Não requer modelo para rotular. Continuação do macro atual não é alvo;
    reentrada nele depois de outro macro é. Futuro incompleto é censurado.
    """
    w = Window.partitionBy("cd_bv").orderBy("passo")
    atual_ok = F.col("estado").isNotNull() & ~F.coalesce(
        F.col("barreira_memoria"), F.lit(True)
    )
    df = base.withColumn("_ok", atual_ok).withColumn(
        "_macro", F.coalesce("funil", "estado")
    )
    for coluna in ("passo", "estado", "ts_estado", "_ok"):
        df = df.withColumn(f"_next{coluna}", F.lead(coluna).over(w))
    consecutivo = F.col("_nextpasso") == F.col("passo") + 1
    exata = F.col("elegivel_ajuste") & (F.col("tipo_censura") == "exata")
    delta = (F.expr("unix_micros(_nextts_estado)")
             - F.expr("unix_micros(ts_estado)")) / F.lit(1000000.0)
    erro_dest = ~F.col("destino").eqNullSafe(F.col("_nextestado"))
    erro_tempo = F.abs(delta - F.col("dur_min")) > tolerancia
    exigir_vazio("labels_arestas_exatas", df.filter(
        exata & consecutivo & F.col("_ok") & F.col("_next_ok")
        & (erro_dest | erro_tempo)
    ))
    df = df.withColumn("_link", F.coalesce(
        exata & consecutivo & F.col("_ok") & F.col("_next_ok")
        & ~erro_dest & ~erro_tempo, F.lit(False)
    )).withColumn("_prev_link", F.lag("_link").over(w))
    df = df.withColumn("_seg", F.sum(F.when(
        F.coalesce("_prev_link", F.lit(False)), 0
    ).otherwise(1)).over(w.rowsBetween(Window.unboundedPreceding, 0)))
    ws = Window.partitionBy("cd_bv", "_seg").orderBy("passo")
    df = df.withColumn("_prev_macro", F.lag("_macro").over(ws))
    df = df.withColumn("_bloco", F.sum(F.when(
        F.col("_macro").eqNullSafe(F.col("_prev_macro")), 0
    ).otherwise(1)).over(ws.rowsBetween(Window.unboundedPreceding, 0)))
    pontos = df.filter("_ok").select("cd_bv", "passo", "_seg", "_bloco")
    blocos = df.filter("_ok").groupBy("cd_bv", "_seg", "_bloco").agg(
        F.first("_macro").alias("_macro"),
        F.min("ts_estado").alias("_entrada"),
    )
    wb = Window.partitionBy("cd_bv", "_seg").orderBy("_bloco")
    for h in range(1, k + 1):
        blocos = blocos.withColumn(f"_acao_{h}", F.lead("_macro", h).over(wb))
        blocos = blocos.withColumn(f"_ts_{h}", F.lead("_entrada", h).over(wb))
    candidatos = [F.when(F.col(f"_acao_{h}").isin(alvos), F.struct(
        F.col(f"_acao_{h}").alias("acao"), F.lit(h).alias("mudanca"),
        F.col(f"_ts_{h}").alias("ts"),
    )) for h in range(1, k + 1)]
    blocos = blocos.withColumn("_primeira", F.coalesce(*candidatos))
    blocos = blocos.withColumn("status_label", F.when(
        F.col("_primeira").isNotNull(), "ALVO_OBSERVADO"
    ).when(F.col(f"_acao_{k}").isNotNull(), "HORIZONTE_OBSERVADO_SEM_ALVO")
        .otherwise("CENSURADO_ANTES_HORIZONTE"))
    labels = blocos.select(
        "cd_bv", "_seg", "_bloco", "status_label",
        F.col("_primeira.acao").alias("acao_real"),
        F.col("_primeira.mudanca").alias("passo_macro_real"),
        F.col("_primeira.ts").alias("ts_acao_real"),
        sum(F.col(f"_acao_{h}").isNotNull().cast("int") for h in range(1, k + 1))
        .alias("n_mudancas_futuras_observadas"),
    )
    return pontos.join(labels, ["cd_bv", "_seg", "_bloco"], "left").drop("_seg", "_bloco")


def exigir_vazio(nome: str, df: DataFrame) -> None:
    if df.limit(1).count():
        raise RuntimeError(f'Integridade: {nome}. Inspecione os registros; não foram descartados.')


def validar_eventos(eventos: DataFrame) -> None:
    needed = {'cd_bv', 'ts_evento', 'estado', 'profundidade_max'}
    if not needed.issubset(eventos.columns):
        raise ValueError(f'Eventos sem colunas: {sorted(needed - set(eventos.columns))}')
    if eventos.schema['ts_evento'].dataType.simpleString() != 'timestamp':
        raise ValueError('Timestamp de eventos deve ser timestamp, não string/NTZ/date.')
    bad = eventos.agg(
        F.sum((F.col('cd_bv').isNull() | F.col('estado').isNull() | F.col('ts_evento').isNull()
               | (F.length(F.trim('cd_bv')) == 0) | (F.length(F.trim('estado')) == 0)).cast('long')).alias('invalidos'),
        F.sum(F.lower('estado').rlike('sem_acao|conversao').cast('long')).alias('sinteticos'),
        F.sum((F.isnan('profundidade_max') | (F.abs('profundidade_max') == float('inf'))).cast('long')).alias('prof_invalida'),
    ).first().asDict()
    if any(int(x or 0) for x in bad.values()):
        raise ValueError(f'Fonte incompatível com a V2.3 clean: {bad}')


def construir_catalogo(eventos: DataFrame) -> DataFrame:
    """Mesmo contrato clean: jornadas topo/navegacao/sucesso e atendimento/canal."""
    cfg = eventos.select('estado').distinct().withColumn('_p', F.split('estado', ':::'))
    exigir_vazio('estado fora do formato família:::etapa/canal', cfg.filter(F.size('_p') != 2))
    cfg = cfg.withColumn('_familia', F.col('_p')[0]).withColumn('_parte', F.col('_p')[1])
    jornada = F.col('_parte').isin('topo', 'navegacao', 'sucesso')
    cfg = cfg.withColumn('funil', F.when(jornada, F.col('_familia')))
    cfg = cfg.withColumn('etapa', F.when(jornada, F.col('_parte')))
    cfg = cfg.withColumn('tipo_estado', F.when(F.col('_parte') == 'topo', 'funil_abertura')
        .when(F.col('_parte') == 'navegacao', 'funil_meio')
        .when(F.col('_parte') == 'sucesso', 'funil_conclusao')
        .when(F.col('_familia') == 'atendimento', 'atendimento').otherwise('nao_mapeado'))
    exigir_vazio('estado sem definição no catálogo clean', cfg.filter("tipo_estado = 'nao_mapeado'"))
    return cfg.select('estado', 'funil', 'etapa', 'tipo_estado',
                      F.when(~jornada, F.col('_parte')).alias('detalhe'),
                      F.coalesce('funil', 'estado').alias('acao_macro'),
                      F.lit(False).alias('terminal_real'))


def construir_transicoes(acoes: DataFrame, corte_iso: str, sal_validacao: str,
                          modulo_validacao: int = 5) -> DataFrame:
    """Dur_min é em SEGUNDOS. Último estado e saída antes de ambiguidade: censura."""
    if modulo_validacao < 2:
        raise ValueError('modulo_validacao deve ser >= 2.')
    w = Window.partitionBy('cd_bv').orderBy('passo')
    b = acoes.withColumn('_s_next', F.lead('estado').over(w))
    b = b.withColumn('_t_next', F.lead('ts_estado').over(w))
    b = b.withColumn('_b_next', F.lead('barreira_memoria').over(w))
    b = b.withColumn('ts_corte_estado', F.lit(corte_iso).cast('timestamp'))
    b = b.withColumn('_fim', F.coalesce('_t_next', 'ts_corte_estado'))
    b = b.withColumn('dur_min', (F.expr('unix_micros(_fim)') - F.expr('unix_micros(ts_estado)')) / 1e6)
    b = b.withColumn('tipo_censura', F.when(F.col('barreira_memoria'), 'origem_ambigua')
        .when(F.col('_t_next').isNull() | F.col('_b_next'), 'direita').otherwise('exata'))
    b = b.withColumn('destino', F.when(F.col('tipo_censura') == 'exata', F.col('_s_next')))
    b = b.withColumn('dur_max', F.when(F.col('tipo_censura') == 'exata', F.col('dur_min')))
    b = b.withColumn('status_observacao', F.when(F.col('barreira_memoria'), 'ORIGEM_AMBIGUA')
        .when(F.col('_t_next').isNull(), 'CENSURA_CORTE')
        .when(F.col('_b_next'), 'CENSURA_ANTES_AMBIGUIDADE').otherwise('SAIDA_OBSERVADA'))
    b = b.withColumn('elegivel_ajuste', ~F.col('barreira_memoria') & (F.col('dur_min') > 0))
    b = b.withColumn('contexto_modelo', F.when(F.col('barreira_memoria'), F.lit(None).cast('string'))
        .when(F.col('origem_tecnica'), F.coalesce('contexto_negocio', F.lit(MS_SEM_MEMORIA)))
        .otherwise(MS_BASE))
    b = b.withColumn('validacao_cliente', F.pmod(F.xxhash64('cd_bv', F.lit(sal_validacao)), F.lit(modulo_validacao)) == 0)
    b = b.withColumn('peso_evento_treino', F.when(~F.col('validacao_cliente') & F.col('elegivel_ajuste'), 1.0))
    exigir_vazio('duração não positiva', b.filter('dur_min <= 0 OR dur_min IS NULL'))
    return b.drop('_s_next', '_t_next', '_b_next', '_fim')


def preparar_estado_atual(eventos: DataFrame, catalogo: DataFrame, tecnicas: set,
                           debounce_seg: float, publico: DataFrame, corte_iso: str) -> DataFrame:
    a = op_preparar_acoes(eventos, catalogo, tecnicas, debounce_seg)
    w = Window.partitionBy('cd_bv').orderBy(F.desc('ts_estado'), F.desc('passo'))
    ultimo = a.withColumn('_rn', F.row_number().over(w)).filter('_rn = 1').drop('_rn')
    b = publico.join(ultimo, 'cd_bv', 'left')
    b = b.withColumn('status_dado', F.when(F.col('ts_estado').isNull(), 'SEM_HISTORICO_NA_FONTE')
        .when(F.col('ordem_ambigua'), 'ESTADO_ATUAL_AMBIGUO')
        .when(~F.coalesce('estado_catalogado', F.lit(False)), 'ESTADO_NAO_CATALOGADO').otherwise('OK'))
    b = b.withColumn('memoria', F.coalesce('contexto_negocio', F.lit(MS_SEM_MEMORIA)))
    b = b.withColumn('ts_corte_eventos', F.lit(corte_iso).cast('timestamp'))
    b = b.withColumn('idade_seg', (F.expr('unix_micros(ts_corte_eventos)') - F.expr('unix_micros(ts_estado)')) / 1e6)
    b = b.withColumn('acao_atual', F.coalesce('funil', 'estado'))
    b = b.withColumn('data_referencia', F.to_date('ts_corte_eventos'))
    exigir_vazio('idade negativa', b.filter('idade_seg < 0'))
    return b


def autoteste_preparo(spark):
    """Executado no cluster antes de ler dados reais; não exige tabelas."""
    catalogo = spark.createDataFrame([
        ('b:::sucesso', 'b', 'sucesso', 'funil_conclusao'),
        ('l:::topo', 'l', 'topo', 'funil_abertura'),
        ('l:::sucesso', 'l', 'sucesso', 'funil_conclusao'),
        ('p:::topo', 'p', 'topo', 'funil_abertura'),
    ], 'estado string, funil string, etapa string, tipo_estado string')
    rows = [('u1', 0, 'b:::sucesso', 1.), ('u1', 10, 'l:::topo', 0.),
            ('u1', 20, 'l:::topo', 0.), ('u1', 40, 'l:::sucesso', 1.),
            ('u1', 60, 'p:::topo', 0.), ('u2', 0, 'b:::sucesso', 1.),
            ('u2', 10, 'l:::topo', 1.), ('u2', 10, 'l:::sucesso', 1.),
            ('u2', 40, 'l:::topo', 0.)]
    e = spark.createDataFrame(rows, 'cd_bv string, seg long, estado string, profundidade_max double')
    e = e.withColumn('ts_evento', F.expr('timestamp_micros(1700000000000000 + seg*1000000)')).drop('seg')
    a = op_preparar_acoes(e, catalogo, {'l'}, 30.)
    corte = '2023-11-14T22:15:20+00:00'
    b = construir_transicoes(a, corte, 'test')
    r = { (x.cd_bv, x.passo): x.asDict() for x in b.collect() }
    assert len([x for x in r if x[0] == 'u1']) == 4
    assert r['u1', 2]['contexto_negocio'] == 'b'
    assert r['u1', 2]['dur_min'] == 20.0
    assert r['u1', 4]['tipo_censura'] == 'direita'
    assert r['u2', 1]['tipo_censura'] == 'direita'
    assert r['u2', 3]['contexto_negocio'] is None
    labels = ms03_rotular(b, ['b', 'p'], 5)
    l = {(x.cd_bv, x.passo): x.asDict() for x in labels.collect()}
    assert l['u1', 1]['acao_real'] == 'p'
    assert l['u1', 1]['passo_macro_real'] == 2
    assert l['u1', 4]['status_label'] == 'CENSURADO_ANTES_HORIZONTE'
    print('Autoteste Spark: debounce, memória, censura e rótulos aprovados.')
