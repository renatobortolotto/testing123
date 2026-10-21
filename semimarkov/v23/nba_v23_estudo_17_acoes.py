# Databricks notebook source
# MAGIC %md
# MAGIC # NBA V2.3 — estudo dos 17 alvos úteis
# MAGIC Diagnóstico somente leitura. Não treina, não refaz scoring, não cria regras
# MAGIC de elegibilidade e não altera tabelas ou views. Usa uma execução concluída
# MAGIC do operacional e o histórico que ela referencia, com a MESMA lista de alvos.
# MAGIC A presença no Top 5 não é acerto nem elegibilidade. As estatísticas de score
# MAGIC operacional abaixo são condicionadas à presença no Top 5.

# COMMAND ----------

import hashlib
import json
import re
from uuid import UUID

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

CFG = {
    "prefixo_operacional": "ctg_dsti.renato_nba.nba_sm_v23_operacional",
    "view_negocio": (
        "ctg_dsti.renato_nba.nba_sm_v23_operacional_negocio_atual"
    ),
    # None reproduz a seleção da view: modo/modelo/corte/conclusão.
    # Um ID explícito ignora a seleção da view, mas exige manifesto concluído.
    "id_execucao": None,
    "analisar_historico": True,
    "persistir_em_disco": True,
    "linhas_exibir": 40,
    "precisao_percentis": 10000,
    "tolerancia": 1e-8,
    "autotestes_spark": True,
    "ids_clientes": [],
}

ALVOS_ESPERADOS = sorted({
    "opf_iniciador_pagamentos", "opf_recepcao_dados",
    "tag_veicular_alterar_placa", "tag_veicular_ativar",
    "tag_veicular_bloquear", "tag_veicular_cancelar",
    "tag_veicular_contestacao", "tag_veicular_desbloquear",
    "tag_veicular_segunda_via", "tag_veicular_solicitar",
    "pix_cadastro_chave", "pix_excluir_chave", "pix_trazer_chave",
    "pagamentos_boleto", "debito_automatico",
    "emprestimos_pessoal_cartao", "portabilidade_salario",
})
B2 = "B2_MEMORIA_TEMPO_COMPARTILHADO"
A = "A_REFERENCIA"
PREVISAO = "PREVISAO_B2_OPERACIONAL"
CONHECIDOS = ["ALVO_OBSERVADO", "HORIZONTE_OBSERVADO_SEM_ALVO"]
RELATORIOS = {}
CACHES_ESTUDO = []

if "spark" not in globals():
    raise RuntimeError("Execute no Databricks, com acesso às tabelas persistidas.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Funções de diagnóstico e testes locais do contrato
# MAGIC Os testes Spark usam apenas dados artificiais pequenos. Nenhuma instalação
# MAGIC de biblioteca, chamada a outro notebook ou variável do scoring é necessária.

# COMMAND ----------


def nome_sql(nome: str) -> str:
    partes = nome.split(".")
    if not partes or any(not re.fullmatch(r"[A-Za-z_][\w-]*", p) for p in partes):
        raise ValueError(f"Nome de tabela/view inválido: {nome!r}")
    return ".".join(f"`{p}`" for p in partes)


def hash_json(objeto: dict) -> str:
    texto = json.dumps(
        objeto, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def validar_alvos(alvos, esperados):
    if not isinstance(alvos, list) or not alvos:
        raise ValueError("Manifesto sem uma lista não vazia de alvos.")
    if any(not isinstance(a, str) or not a for a in alvos):
        raise ValueError("Alvos inválidos no manifesto.")
    if len(set(alvos)) != len(alvos):
        raise ValueError("Há alvos duplicados no manifesto.")
    if set(alvos) != set(esperados):
        raise ValueError(
            "A execução não usa a lista de negócio esperada. "
            f"Ausentes={sorted(set(esperados) - set(alvos))}; "
            f"extras={sorted(set(alvos) - set(esperados))}. "
            "Não misture o histórico de 15 alvos com o operacional de 17."
        )
    return sorted(alvos)


def contrato_view(ddl: str):
    modelos = set(re.findall(
        r"`?id_ms03_validado`?\s*=\s*'([0-9a-fA-F-]{36})'",
        ddl, flags=re.IGNORECASE
    ))
    modos = set(re.findall(
        r"`?modo_scoring`?\s*=\s*'(TODOS|AMOSTRA|IDS)'",
        ddl, flags=re.IGNORECASE
    ))
    if len(modelos) != 1 or len(modos) != 1:
        raise ValueError(
            "Não foi possível reproduzir a seleção dessa view. "
            "Informe CFG['id_execucao'] com um ID concluído; "
            "não será escolhido um UUID aleatório ou outro modelo."
        )
    modelo, modo = next(iter(modelos)), next(iter(modos))
    UUID(modelo)
    return modelo, modo.upper()


def ler(tabela: str) -> DataFrame:
    nome_sql(tabela)
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Entrada persistida não encontrada: {tabela}")
    # Recortes imutáveis por ID, gravados em append pelo pipeline operacional.
    # Não solicita versões Delta antigas nem altera políticas de retenção.
    return spark.table(tabela)


def colunas(df: DataFrame, obrigatorias, rotulo: str) -> None:
    faltantes = sorted(set(obrigatorias) - set(df.columns))
    if faltantes:
        raise RuntimeError(f"{rotulo}: colunas ausentes: {faltantes}")


def unico(df: DataFrame, rotulo: str):
    linhas = df.limit(2).collect()
    if len(linhas) != 1:
        raise RuntimeError(
            f"{rotulo}: esperado um único registro; verificação retornou "
            f"{len(linhas)} de até 2 linhas."
        )
    return linhas[0].asDict(recursive=True)


def vazio(df: DataFrame, rotulo: str) -> None:
    if df.limit(1).count():
        raise RuntimeError(
            f"Inconsistência em {rotulo}. "
            "Não se corrigem dados nesta análise."
        )


def cache_disco(df: DataFrame) -> DataFrame:
    if CFG["persistir_em_disco"]:
        df = df.persist(StorageLevel.DISK_ONLY)
        CACHES_ESTUDO.append(df)
    return df


def mostrar(nome: str, df: DataFrame, ordem=None) -> None:
    RELATORIOS[nome] = df
    print(f"\n{nome}")
    saida = df.orderBy(*ordem) if ordem else df
    for campo in saida.schema.fields:
        if campo.dataType.simpleString() in {"double", "float"}:
            saida = saida.withColumn(campo.name, F.round(campo.name, 6))
    saida.show(CFG["linhas_exibir"], truncate=False)


def quantis(campo: str):
    return F.percentile_approx(
        campo, [0.1, 0.5, 0.9, 0.99], CFG["precisao_percentis"]
    )


def resumo_acoes(rec: DataFrame, dimensao: DataFrame, n_previstos: int):
    agregado = rec.groupBy("acao").agg(
        F.count("*").alias("clientes_top5"),
        *[
            F.sum(F.when(F.col("ranking") == r, 1).otherwise(0))
            .alias(f"clientes_rank_{r}") for r in range(1, 6)
        ],
        F.avg("score").alias("score_medio_no_top5"),
        F.stddev_pop("score").alias("desvio_score_no_top5"),
        F.min("score").alias("score_minimo_no_top5"),
        quantis("score").alias("score_p10_p50_p90_p99_no_top5"),
        F.max("score").alias("score_maximo_no_top5"),
    )
    contagens = ["clientes_top5", *[f"clientes_rank_{r}" for r in range(1, 6)]]
    return (
        dimensao.join(agregado, "acao", "left")
        .fillna(0, subset=contagens)
        .withColumn("presente_top5", F.col("clientes_top5") > 0)
        .withColumn("fracao_clientes_top5", F.col("clientes_top5") / n_previstos)
        .withColumn("fracao_clientes_top1", F.col("clientes_rank_1") / n_previstos)
    )


def testes_spark():
    dim = spark.createDataFrame([("x",), ("y",), ("z",)], "acao string")
    teste = spark.createDataFrame([
        ("c1", "x", 1, 0.5), ("c1", "y", 2, 0.2),
        ("c2", "y", 1, 0.4),
    ], "cd_bv string, acao string, ranking int, score double")
    out = {r.acao: r.asDict() for r in resumo_acoes(teste, dim, 2).collect()}
    assert out["x"]["clientes_top5"] == 1
    assert out["y"]["clientes_top5"] == 2
    assert out["y"]["fracao_clientes_top5"] == 1.0
    assert out["z"]["clientes_top5"] == 0
    assert out["z"]["score_medio_no_top5"] is None
    exemplo = spark.createDataFrame([
        ("ALVO_OBSERVADO", "x", "x"),
        ("HORIZONTE_OBSERVADO_SEM_ALVO", None, "x"),
        ("CENSURADO_ANTES_HORIZONTE", None, "x"),
    ], "status_label string, acao_real string, acao_prevista string")
    conhecidos = exemplo.filter(F.col("status_label").isin(CONHECIDOS))
    taxa = conhecidos.agg(F.avg(F.when(
        F.col("acao_real") == F.col("acao_prevista"), 1.0
    ).otherwise(0.0))).first()[0]
    assert taxa == 0.5  # Censura não vira negativo.
    print("Autotestes Spark: cobertura, ausências e censura OK.")


if CFG["autotestes_spark"]:
    testes_spark()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Selecionar UMA execução concluída
# MAGIC Com ID vazio, replica o filtro da view final pelo modo/modelo e pelo maior
# MAGIC corte; a conclusão mais recente desempata. Não ordena pela data do público
# MAGIC (que pode ser nula) nem usa um UUID como relógio. Um ID explícito é permitido.

# COMMAND ----------

manifestos = ler(f"{CFG['prefixo_operacional']}_execucoes_hml")
colunas(manifestos, {
    "id_execucao", "id_ms03_validado", "id_pacote_modelo", "modo_scoring",
    "status", "corte_micros", "fim_utc", "config_json", "hash_execucao",
    "tabelas_json", "n_clientes_elegiveis", "n_clientes_previstos",
    "n_linhas_negocio",
}, "Manifesto operacional")
concluidos = manifestos.filter(F.col("status") == "CONCLUIDO_SCORING_B2")
if CFG["id_execucao"]:
    execucao = unico(concluidos.filter(
        F.col("id_execucao") == str(CFG["id_execucao"])
    ), "Execução informada")
else:
    ddl_linhas = spark.sql(
        f"SHOW CREATE TABLE {nome_sql(CFG['view_negocio'])}"
    ).collect()
    ddl = "\n".join(str(r[0]) for r in ddl_linhas)
    modelo_view, modo_view = contrato_view(ddl)
    candidatas = concluidos.filter(
        (F.col("id_ms03_validado") == modelo_view)
        & (F.col("modo_scoring") == modo_view)
    ).orderBy(
        F.desc("corte_micros"), F.desc("fim_utc"), F.desc("id_execucao")
    )
    primeira = candidatas.limit(1).collect()
    if not primeira:
        raise RuntimeError("A view não possui execução concluída compatível.")
    execucao = unico(concluidos.filter(
        F.col("id_execucao") == primeira[0]["id_execucao"]
    ), "Execução selecionada pela view")

ID_EXECUCAO = execucao["id_execucao"]
config_run = json.loads(execucao["config_json"])
if hash_json(config_run) != execucao["hash_execucao"]:
    raise RuntimeError("Configuração do manifesto operacional divergente.")
ALVOS = validar_alvos(config_run["alvos"], ALVOS_ESPERADOS)
TABELAS = json.loads(execucao["tabelas_json"])
DIM_ALVOS = spark.createDataFrame([(a,) for a in ALVOS], "acao string")
print("id_execucao:", ID_EXECUCAO)
print("modo:", execucao["modo_scoring"])
print("id_ms03_validado:", execucao["id_ms03_validado"])
print("corte_micros:", execucao["corte_micros"])
print("Alvos do manifesto:", len(ALVOS), ALVOS)


def ler_execucao(tipo: str, campos):
    df = ler(TABELAS[tipo])
    colunas(df, {"id_execucao", *campos}, tipo)
    return df.filter(F.col("id_execucao") == ID_EXECUCAO).select(*campos)


publico = ler_execucao("publico", ["cd_bv"])
estados = cache_disco(ler_execucao("estados", [
    "cd_bv", "estado_atual", "contexto_negocio", "memoria_grafo",
    "origem_tecnica", "idade_seg", "status_dado", "status_previsao",
]))
negocio = cache_disco(ler_execucao("negocio", [
    "cd_bv", "data_referencia", "ranking", "proxima_acao_util", "score_util",
]))
predicoes = cache_disco(ler_execucao("predicoes", [
    "cd_bv", "status_previsao", "rota_modelo", "status_temporal",
    "massa_acao_util", "massa_sem_modelo", "massa_limite_macro",
    "massa_terminal", "massa_retencao_interna", "erro_massa",
]))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Integridade e presença dos 17 alvos
# MAGIC `clientes_top5` tem como denominador os clientes COM previsão. As frações
# MAGIC por ação podem somar mais de 100%, pois cada cliente pode receber cinco ações.
# MAGIC Para alvos ausentes, a contagem é zero, mas o score é DESCONHECIDO, não zero.

# COMMAND ----------

n_publico = publico.count()
n_estados = estados.count()
n_negocio = negocio.count()
n_predicoes = predicoes.count()
n_previstos = predicoes.filter(F.col("status_previsao") == PREVISAO).count()
if (
    n_publico != int(execucao["n_clientes_elegiveis"])
    or n_estados != n_publico or n_predicoes != n_publico
    or n_previstos != int(execucao["n_clientes_previstos"])
    or n_negocio != int(execucao["n_linhas_negocio"])
    or n_previstos == 0
):
    raise RuntimeError("Contagens divergentes do manifesto concluído.")
for nome, df in [("publico", publico), ("estados", estados), ("predicoes", predicoes)]:
    vazio(df.filter(F.col("cd_bv").isNull()), f"ID nulo em {nome}")
    vazio(df.groupBy("cd_bv").count().filter("count != 1"), f"duplicidade em {nome}")
vazio(
    negocio.groupBy("cd_bv", "ranking").count().filter("count != 1"),
    "rank duplicado",
)
vazio(negocio.filter(
    F.col("ranking").isNull() | ~F.col("ranking").between(1, 5)
), "ranking fora de 1..5")
rec = negocio.filter(F.col("proxima_acao_util").isNotNull()).select(
    "cd_bv", "ranking", F.col("proxima_acao_util").alias("acao"),
    F.col("score_util").alias("score"),
)
vazio(rec.filter(
    F.col("score").isNull() | F.isnan("score")
    | (F.col("score") <= 0) | (F.col("score") > 1)
), "score útil inválido")
vazio(rec.join(DIM_ALVOS, "acao", "left_anti"), "ação fora da lista do manifesto")
vazio(
    rec.groupBy("cd_bv", "acao").count().filter("count != 1"),
    "ação repetida por cliente",
)
ordem = Window.partitionBy("cd_bv").orderBy("ranking")
ordenado = rec.withColumn("score_anterior", F.lag("score").over(ordem))
vazio(ordenado.filter(
    F.col("score") > F.col("score_anterior") + CFG["tolerancia"]
), "scores fora de ordem")
contagem_rank = rec.groupBy("cd_bv").agg(
    F.count("*").alias("n"), F.min("ranking").alias("rmin"),
    F.max("ranking").alias("rmax"),
)
vazio(contagem_rank.filter("rmin != 1 OR rmax != n"), "lacuna de rank útil")
vazio(
    estados.select("cd_bv").join(publico, "cd_bv", "left_anti"),
    "estado fora do público",
)
vazio(negocio.select("cd_bv").distinct().join(
    predicoes.filter(F.col("status_previsao") == PREVISAO).select("cd_bv"),
    "cd_bv", "left_anti",
), "negócio sem previsão")

mostrar("EST17_01_COBERTURA", estados.groupBy(
    "status_dado", "status_previsao"
).agg(F.count("*").alias("n_clientes")).withColumn(
    "fracao_publico", F.col("n_clientes") / n_publico
), [F.desc("n_clientes")])
resumo = resumo_acoes(rec, DIM_ALVOS, n_previstos)
mostrar("EST17_02_COBERTURA_ALVOS", resumo.select(
    "acao", "presente_top5", "clientes_top5", "clientes_rank_1",
    "fracao_clientes_top5", "fracao_clientes_top1",
), [F.desc("clientes_top5"), "acao"])
RELATORIOS["EST17_03_SCORES_COMPLETOS"] = resumo
mostrar("EST17_03_RANKS", resumo.select(
    "acao", *[f"clientes_rank_{r}" for r in range(1, 6)]
), [F.desc("clientes_rank_1"), "acao"])
mostrar("EST17_03_SCORES", resumo.select(
    "acao", "clientes_top5",
    F.col("score_medio_no_top5").alias("score_medio"),
    F.col("desvio_score_no_top5").alias("desvio_score"),
    F.element_at("score_p10_p50_p90_p99_no_top5", 2).alias("score_p50"),
    F.element_at("score_p10_p50_p90_p99_no_top5", 3).alias("score_p90"),
    F.col("score_maximo_no_top5").alias("score_maximo"),
), [F.desc("clientes_top5"), "acao"])
mostrar(
    "EST17_02_ALVOS_AUSENTES",
    resumo.filter("NOT presente_top5").select("acao"), ["acao"],
)
n_presentes = resumo.filter("presente_top5").count()
print(f"Cobertura do catálogo: {n_presentes}/{len(ALVOS)} alvos no Top 5.")
print("Isso NÃO é cobertura de clientes, acurácia ou elegibilidade por produto.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Concentração, margens e massa fora do Top 5
# MAGIC Um alvo fora do Top 5 pode ter score positivo. A diferença entre a massa
# MAGIC total útil salva e a soma do Top 5 mede a massa dos demais alvos, sem atribuir
# MAGIC essa massa a uma ação específica. Não redistribuímos ou renormalizamos scores.

# COMMAND ----------

por_cliente = rec.groupBy("cd_bv").agg(
    F.sort_array(F.collect_list(F.struct("ranking", "acao"))).alias("lista"),
    F.count("*").alias("n_acoes_top5"),
    F.sum("score").alias("massa_top5"),
    F.max(F.when(F.col("ranking") == 1, F.col("score"))).alias("score_1"),
    F.max(F.when(F.col("ranking") == 2, F.col("score"))).alias("score_2"),
    F.max(F.when(F.col("ranking") == 5, F.col("score"))).alias("score_5"),
)
previstos = predicoes.filter(F.col("status_previsao") == PREVISAO)
por_cliente = previstos.select("cd_bv", "massa_acao_util").join(
    por_cliente, "cd_bv", "left"
).fillna({"n_acoes_top5": 0, "massa_top5": 0.0}).withColumn(
    "assinatura", F.coalesce(F.to_json("lista"), F.lit("[]"))
).withColumn(
    "massa_alvos_fora_top5", F.col("massa_acao_util") - F.col("massa_top5")
).withColumn(
    "fracao_massa_util_no_top5", F.when(
        F.col("massa_acao_util") > 0,
        F.col("massa_top5") / F.col("massa_acao_util"),
    )
).withColumn("margem_1_2", F.col("score_1") - F.col("score_2"))
vazio(por_cliente.filter(
    F.col("massa_alvos_fora_top5") < -CFG["tolerancia"]
), "Top 5 maior que a massa total útil")
assinaturas = por_cliente.groupBy("assinatura").agg(F.count("*").alias("clientes"))
mostrar("EST17_04_CONCENTRACAO", assinaturas.agg(
    F.count("*").alias("n_listas_distintas"),
    F.max("clientes").alias("clientes_lista_dominante"),
    F.sum("clientes").alias("clientes_previstos"),
).withColumn(
    "fracao_lista_dominante",
    F.col("clientes_lista_dominante") / F.col("clientes_previstos"),
))
mostrar("EST17_04_LISTAS", assinaturas.withColumn(
    "fracao_previstos", F.col("clientes") / n_previstos
), [F.desc("clientes")])
mostrar("EST17_04_MARGENS_E_MASSA", por_cliente.agg(
    F.count("score_2").alias("clientes_com_pelo_menos_2_acoes"),
    quantis("margem_1_2").alias("margem_1_2_p10_p50_p90_p99"),
    F.avg("massa_top5").alias("massa_media_top5"),
    F.avg("massa_acao_util").alias("massa_media_todos_alvos"),
    F.avg("massa_alvos_fora_top5").alias("massa_media_fora_top5"),
    quantis("fracao_massa_util_no_top5").alias("captura_massa_p10_p50_p90_p99"),
))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Estado, memória e tempo observados
# MAGIC Comparação descritiva. Faixas de tempo são somente de relatório: não mudam
# MAGIC a inferência e não isolam causalmente o efeito da memória. Não são faixas de
# MAGIC confiança. A tabela completa fica em RELATORIOS.

# COMMAND ----------

contextos = estados.filter(F.col("status_previsao") == PREVISAO).withColumn(
    "faixa_idade", F.when(F.col("idade_seg") < 1800, "01_menos_30min")
    .when(F.col("idade_seg") < 86400, "02_30min_a_1dia")
    .when(F.col("idade_seg") < 604800, "03_1_a_7dias")
    .otherwise("04_7dias_ou_mais"),
)
chave_contexto = ["estado_atual", "memoria_grafo", "faixa_idade"]
cont_contexto = contextos.groupBy(*chave_contexto).agg(
    F.count("*").alias("clientes_no_contexto")
)
top1_contexto = contextos.join(rec.filter("ranking = 1"), "cd_bv", "inner")
perfil = top1_contexto.groupBy(*chave_contexto, "acao").agg(
    F.count("*").alias("clientes_top1"),
    F.avg("score").alias("score_medio_top1"),
).join(cont_contexto, chave_contexto, "left").withColumn(
    "fracao_top1_dentro_contexto",
    F.col("clientes_top1") / F.col("clientes_no_contexto"),
)
mostrar("EST17_05_CONTEXTO_TOP1", perfil, [F.desc("clientes_top1"), "acao"])
mostrar("EST17_05_ROTAS", predicoes.groupBy(
    "status_previsao", "rota_modelo", "status_temporal"
).agg(F.count("*").alias("clientes")), [F.desc("clientes")])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Histórico vinculado: suporte e ranking completo dos 17 alvos
# MAGIC O operacional só persistiu Top 5 por pessoa. O kernel histórico contém
# MAGIC `util_probs`/`util_ranks` para todos os alvos e permite verificar score
# MAGIC positivo fora do Top 5 NO HISTÓRICO. Isso não revela o rank 6..17 dos clientes
# MAGIC operacionais. Não repetimos treino/scoring nem lemos a telemetria bruta.

# COMMAND ----------

if CFG["analisar_historico"]:
    cfg_operacional = config_run["config"]
    hist = unico(ler(cfg_operacional["tabela_ms03_execucoes"]).filter(
        (F.col("id_ms03") == execucao["id_ms03_validado"])
        & (F.col("status") == "CONCLUIDO_MULTISTEP_HISTORICO_03")
    ), "Histórico Multi-step vinculado")
    hist_cfg = json.loads(hist["config_json"])
    if hash_json(hist_cfg) != hist["hash_ms03"]:
        raise RuntimeError("Hash do manifesto histórico divergente.")
    validar_alvos(hist_cfg["alvos"], ALVOS)
    hist_tabelas = json.loads(hist["tabelas_json"])
    id_pacote = hist_cfg["pacote_entradas"]
    if id_pacote != execucao["id_pacote_modelo"]:
        raise RuntimeError("Pacote histórico diferente do pacote operacional.")
    pacote = unico(ler(cfg_operacional["tabela_pacotes"]).filter(
        (F.col("id_pacote") == id_pacote)
        & (F.col("status") == "CONCLUIDO_PACOTE")
    ), "Pacote persistido")
    meta = json.loads(pacote["meta_json"])
    if hash_json(meta) != pacote["hash_pacote"]:
        raise RuntimeError("Hash do manifesto do pacote divergente.")

    def ler_pacote(tipo, campos):
        info = meta["entradas"][tipo]
        df = ler(info["tabela"]).filter(F.col("id_pacote") == id_pacote)
        colunas(df, campos, f"Pacote/{tipo}")
        return df.select(*campos)

    config = ler_pacote("config", ["estado", "funil"])
    vazio(config.groupBy("estado").count().filter("count != 1"), "catálogo duplicado")
    catalogo = config.groupBy(F.col("funil").alias("acao")).agg(
        F.count("*").alias("n_estados_catalogados")
    )
    treino = ler_pacote("base", [
        "cd_bv", "estado", "funil", "destino", "tipo_censura",
        "validacao_cliente", "elegivel_ajuste",
    ]).filter(
        (F.col("validacao_cliente") == F.lit(False))
        & (F.col("elegivel_ajuste") == F.lit(True))
    )
    suporte_obs = treino.groupBy(F.col("funil").alias("acao")).agg(
        F.count("*").alias("n_observacoes_preparadas_treino"),
        F.countDistinct("cd_bv").alias("n_clientes_com_acao_treino"),
    )
    mapa_destino = config.select(
        F.col("estado").alias("destino"), F.col("funil").alias("acao")
    )
    entradas = treino.filter(F.col("tipo_censura") == "exata").join(
        F.broadcast(mapa_destino), "destino", "inner"
    ).filter(
        F.col("acao").isNotNull()
        & ~F.col("acao").eqNullSafe(F.col("funil"))
    ).groupBy("acao").agg(
        F.count("*").alias("n_entradas_macro_exatas_treino"),
        F.countDistinct("cd_bv").alias("n_clientes_entradas_treino"),
    )
    suporte = (
        DIM_ALVOS.join(catalogo, "acao", "left")
        .join(suporte_obs, "acao", "left")
        .join(entradas, "acao", "left").fillna(0)
    )
    mostrar("EST17_06_SUPORTE_TREINO", suporte, ["acao"])

    kernels = ler(hist_tabelas["kernels"]).filter(
        (F.col("id_ms03") == hist["id_ms03"])
        & (F.col("variante") == B2) & F.col("tem_previsao")
    )
    colunas(kernels, {
        "estado", "memoria_grafo", "idade_seg", "util_probs", "util_ranks",
    }, "Kernels históricos")
    if not kernels.limit(1).count():
        raise RuntimeError("Não há kernels B2 na execução histórica vinculada.")
    chaves_kernel = ["estado", "memoria_grafo", "idade_seg"]
    vazio(
        kernels.groupBy(*chaves_kernel).count().filter("count != 1"),
        "kernel repetido",
    )
    esperado = F.array(*[F.lit(a) for a in ALVOS])
    vazio(kernels.filter(
        ~F.sort_array(F.map_keys("util_probs")).eqNullSafe(esperado)
        | ~F.sort_array(F.map_keys("util_ranks")).eqNullSafe(esperado)
    ), "kernel com conjunto de alvos diferente")
    completos_hist = kernels.select(
        *chaves_kernel, "util_ranks",
        F.explode("util_probs").alias("acao", "score_hist"),
    ).withColumn("rank_hist", F.element_at("util_ranks", F.col("acao")))
    evidencia_kernel = completos_hist.groupBy("acao").agg(
        F.count("*").alias("n_condicoes_historicas_avaliadas"),
        F.sum(F.when(F.col("score_hist") > 0, 1).otherwise(0))
        .alias("n_condicoes_hist_score_positivo"),
        F.sum(F.when(
            (F.col("score_hist") > 0) & (F.col("rank_hist") <= 5), 1
        ).otherwise(0))
        .alias("n_condicoes_hist_top5"),
        F.sum(F.when(
            (F.col("score_hist") > 0) & (F.col("rank_hist") > 5), 1
        ).otherwise(0))
        .alias("n_condicoes_hist_positivo_fora_top5"),
        F.min(F.when(F.col("score_hist") > 0, F.col("rank_hist")))
        .alias("melhor_rank_historico_score_positivo"),
        F.max("score_hist").alias("maior_score_historico"),
    )
    mostrar("EST17_07_RANKING_COMPLETO_HISTORICO", evidencia_kernel, ["acao"])
    diagnostico_17 = resumo.select(
        "acao", "clientes_top5", "clientes_rank_1", "presente_top5"
    ).join(suporte, "acao", "left").join(evidencia_kernel, "acao", "left")
    RELATORIOS["EST17_08_EVIDENCIAS_COMPLETAS"] = diagnostico_17
    mostrar("EST17_08_AUSENCIAS_E_EVIDENCIAS", diagnostico_17.select(
        "acao", F.col("clientes_top5").alias("top5_atual"),
        F.col("n_clientes_com_acao_treino").alias("clientes_treino"),
        F.col("n_entradas_macro_exatas_treino").alias("entradas_treino"),
        F.col("n_condicoes_hist_score_positivo").alias("cond_hist_p_pos"),
        F.col("n_condicoes_hist_top5").alias("cond_hist_top5"),
        F.col("melhor_rank_historico_score_positivo")
        .alias("melhor_rank_hist"),
        F.col("maior_score_historico").alias("score_max_hist"),
    ), ["top5_atual", "acao"])
    print(
        "Condições históricas = estado + memória + idade; não são clientes. "
        "Ausência ou score zero nesses kernels não prova impossibilidade global."
    )
else:
    print("Seção histórica desativada: não serão inferidas causas das ausências.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Desempenho histórico por ação, com a mesma lista
# MAGIC Recuperação: acerto entre snapshots onde aquela ação foi o alvo observado.
# MAGIC Precisão: acerto entre recomendações daquela ação, apenas com futuro conhecido.
# MAGIC Censura fica fora das métricas, nunca é convertida em negativo. Há repetição
# MAGIC de clientes/snapshots e seleção do holdout durante desenvolvimento: indicadores
# MAGIC exploratórios, não prova de efeito de campanha nem confiança calibrada.

# COMMAND ----------

if CFG["analisar_historico"]:
    hp = ler(hist_tabelas["predicoes"]).filter(
        (F.col("id_ms03") == hist["id_ms03"])
        & F.col("variante").isin(A, B2)
    )
    colunas(hp, {
        "cd_bv", "passo", "idade_seg", "variante", "tem_previsao",
        "status_label", "acao_real", "top1", "top5", "mrr", "util_top5",
    }, "Previsões históricas")
    mostrar("EST17_09_COBERTURA_HISTORICA", hp.groupBy(
        "variante", "idade_seg", "status_label", "tem_previsao"
    ).agg(
        F.count("*").alias("n_snapshots"), F.countDistinct("cd_bv").alias("n_clientes")
    ), ["idade_seg", "variante", "status_label"])
    conhecidos = hp.filter(
        F.col("tem_previsao") & F.col("status_label").isin(CONHECIDOS)
    )
    positivos = conhecidos.filter(F.col("status_label") == "ALVO_OBSERVADO")
    keys = ["variante", "idade_seg", "acao"]
    positivos_acao = positivos.withColumnRenamed("acao_real", "acao")
    eventos_metricas = positivos_acao.groupBy(*keys).agg(
        F.count("*").alias("n_alvos_observados"),
        F.countDistinct("cd_bv").alias("n_clientes_alvo_observado"),
        F.avg("top1").alias("recuperacao_top1_por_snapshot"),
        F.avg("top5").alias("recuperacao_top5_por_snapshot"),
        F.avg("mrr").alias("mrr_por_snapshot"),
    )
    pc = positivos.withColumnRenamed("acao_real", "acao").groupBy(*keys, "cd_bv").agg(
        F.avg("top1").alias("top1_cliente"),
        F.avg("top5").alias("top5_cliente"),
    )
    cliente_metricas = pc.groupBy(*keys).agg(
        F.avg("top1_cliente").alias("recuperacao_top1_media_cliente"),
        F.avg("top5_cliente").alias("recuperacao_top5_media_cliente"),
    )
    candidatos_hist = conhecidos.select(
        "variante", "idade_seg", "acao_real", F.explode("util_top5").alias("u")
    ).select(
        "variante", "idade_seg", "acao_real", F.col("u.acao").alias("acao"),
        F.col("u.ranking").alias("ranking"),
    ).withColumn("acertou", F.when(
        F.col("acao_real") == F.col("acao"), 1.0
    ).otherwise(0.0))
    precisao = candidatos_hist.groupBy(*keys).agg(
        F.count("*").alias("n_top5_com_desfecho_conhecido"),
        F.sum(F.when(F.col("ranking") == 1, 1).otherwise(0))
        .alias("n_top1_com_desfecho_conhecido"),
        F.avg("acertou").alias("precisao_top5_desfecho_conhecido"),
        F.avg(F.when(F.col("ranking") == 1, F.col("acertou")))
        .alias("precisao_top1_desfecho_conhecido"),
    )
    grade = hp.select("variante", "idade_seg").distinct().crossJoin(
        F.broadcast(DIM_ALVOS)
    )
    desempenho = grade.join(eventos_metricas, keys, "left").join(
        cliente_metricas, keys, "left"
    ).join(precisao, keys, "left").fillna(0, subset=[
        "n_alvos_observados", "n_clientes_alvo_observado",
        "n_top5_com_desfecho_conhecido", "n_top1_com_desfecho_conhecido",
    ])
    # Relatório completo preserva A e B2 e todas as idades.
    RELATORIOS["EST17_10_DESEMPENHO_HISTORICO_COMPLETO"] = desempenho
    mostrar("EST17_10_DESEMPENHO_POR_ACAO", desempenho.filter(
        (F.col("variante") == B2) & (F.col("idade_seg") == 0)
    ).select(
        "acao", F.col("n_alvos_observados").alias("n_alvos"),
        F.col("n_clientes_alvo_observado").alias("n_clientes_alvo"),
        F.col("recuperacao_top1_media_cliente").alias("rec1_cliente"),
        F.col("recuperacao_top5_media_cliente").alias("rec5_cliente"),
        F.col("precisao_top1_desfecho_conhecido").alias("precisao1"),
        F.col("precisao_top5_desfecho_conhecido").alias("precisao5"),
    ), ["acao"])
    print(
        "A impressão acima é B2/idade zero; "
        "as demais idades e A estão no relatório completo."
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Clientes escolhidos e encerramento
# MAGIC IDs opcionais são consultados só nesta execução. Ter eventos em outra fonte
# MAGIC não garante participação na amostra nem no público selecionado.
# MAGIC Nenhuma linha é removida, publicada ou encaminhada para comunicação.

# COMMAND ----------

if CFG["ids_clientes"]:
    escolhidos = spark.createDataFrame(
        [(str(i),) for i in sorted(set(CFG["ids_clientes"]))], "cd_bv string"
    )
    situacao = escolhidos.join(
        publico.withColumn("no_publico_execucao", F.lit(True)), "cd_bv", "left"
    ).join(estados, "cd_bv", "left").fillna({"no_publico_execucao": False})
    mostrar("EST17_11_CLIENTES_STATUS", situacao, ["cd_bv"])
    mostrar("EST17_11_CLIENTES_TOP5", negocio.join(
        escolhidos, "cd_bv", "inner"
    ), ["cd_bv", "ranking"])

print("\nEST17_12_CONCLUSAO")
print("Execução operacional analisada:", ID_EXECUCAO)
print("Modelo histórico:", execucao["id_ms03_validado"])
print("Alvos presentes no Top 5:", n_presentes, "de", len(ALVOS))
print("Clientes elegíveis:", n_publico, "| clientes com previsão:", n_previstos)
print("Relatórios disponíveis:", sorted(RELATORIOS))
print(
    "Nenhum treino/scoring executado. Nenhuma tabela/view alterada. "
    "Nenhuma regra comercial criada."
)
print(
    "Não há scores operacionais completos 6..17 neste arquivo: "
    "não inventamos zeros para alvos ausentes."
)
for _df in CACHES_ESTUDO:
    _df.unpersist()
CACHES_ESTUDO.clear()
