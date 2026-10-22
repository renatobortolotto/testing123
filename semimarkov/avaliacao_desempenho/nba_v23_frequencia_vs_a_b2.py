# Databricks notebook source
# MAGIC %md
# MAGIC # NBA V2.3 — frequência geral × A × B2
# MAGIC **Estudo somente leitura.** Não retreina, não refaz scoring, não altera
# MAGIC tabelas/views e não aplica elegibilidade. Reutiliza o histórico aprovado
# MAGIC com os 17 alvos. Estima somente uma referência empírica F no treino.
# MAGIC
# MAGIC Perguntas: B2 melhora as probabilidades? Os maiores scores priorizam
# MAGIC clientes? A memória acrescenta valor a A? Diversidade não é uma meta.
# MAGIC O alvo continua sendo a primeira entrada em um dos 17 funis em até cinco
# MAGIC mudanças de macroação; não é conversão, conclusão de produto ou sete dias.
# MAGIC
# MAGIC Importe este arquivo em um notebook NOVO e execute as células em ordem.
# MAGIC Não depende das variáveis ou funções de outros notebooks.

# COMMAND ----------

import hashlib
import json
import re
from functools import reduce
from uuid import UUID

import numpy as np
import pandas as pd
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

CFG = {
    "prefixo_operacional": "ctg_dsti.renato_nba.nba_sm_v23_operacional",
    # Execução TODOS com os 17 alvos enviada no estudo anterior.
    # Para outra execução, altere SOMENTE este ID; as dependências são derivadas.
    "id_execucao": "52a577b9-4d54-41dd-9d24-7a871b675318",
    "autotestes_spark": True,
    "cache_disco": True,
    "linhas_exibir": 100,
    "bootstrap_replicas": 2000,
    "bootstrap_lote": 100,
    "semente": 20261022,
    "max_agregados_driver": 300000,
    "comparar_brier_por_acao_com_ic": True,
    # Escolha independente do alvo, feita ANTES de excluir censuras.
    # Alternativa: TODOS_PESO_CLIENTE (lift em snapshots, não pessoas distintas).
    "coorte_lift": "UM_SNAPSHOT_POR_CLIENTE",
    "fracoes_lift": [0.10, 0.20],
    "faixas_score": 10,
    # Aviso estatístico apenas: não bloqueia/exclui ações nem decide negócio.
    "min_clientes_positivos_alerta": 30,
    "tolerancia": 1e-8,
    "tolerancia_duracao_seg": 1e-3,
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
REF = "F_FREQUENCIA_TREINO"
A = "A_REFERENCIA"
B2 = "B2_MEMORIA_TEMPO_COMPARTILHADO"
SEM_ALVO = "__SEM_ALVO_NO_HORIZONTE__"
CONHECIDOS = ["ALVO_OBSERVADO", "HORIZONTE_OBSERVADO_SEM_ALVO"]
CHAVE = ["cd_bv", "passo", "idade_seg"]
CHAVE_KERNEL = ["estado", "memoria_grafo", "idade_seg"]
METRICAS = [
    "brier_acoes", "brier_algum_alvo", "logloss_conservador",
    "top1", "top5", "mrr",
]
PERDAS = {"brier_acoes", "brier_algum_alvo", "logloss_conservador"}
RELATORIOS = {}
_CACHES = []

if "spark" not in globals():
    raise RuntimeError("Execute este notebook em uma sessão Spark/Databricks.")
if CFG["coorte_lift"] not in {
    "UM_SNAPSHOT_POR_CLIENTE", "TODOS_PESO_CLIENTE"
}:
    raise ValueError("Configuração de coorte_lift inválida.")
UUID(CFG["id_execucao"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Funções independentes e verificações
# MAGIC Os inputs são partições por IDs concluídos de artefatos persistidos.
# MAGIC Nenhuma versão antiga do Delta é solicitada. As cópias físicas do pacote
# MAGIC e os registros da execução devem ser imutáveis; não são as fontes atuais.
# MAGIC Hash de configuração não é prova criptográfica de todo o conteúdo.

# COMMAND ----------


def nome_sql(nome: str) -> str:
    partes = nome.split(".")
    if not 1 <= len(partes) <= 3 or any(
        not re.fullmatch(r"[A-Za-z_][\w-]*", p) for p in partes
    ):
        raise ValueError(f"Nome de tabela inválido: {nome!r}")
    return ".".join(f"`{p}`" for p in partes)


def hash_json(objeto: dict) -> str:
    texto = json.dumps(
        objeto, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def ler(tabela: str) -> DataFrame:
    nome_sql(tabela)
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Artefato persistido não encontrado: {tabela}")
    return spark.table(tabela)


def colunas(df: DataFrame, campos, rotulo: str) -> None:
    faltantes = set(campos) - set(df.columns)
    if faltantes:
        raise RuntimeError(f"{rotulo}: faltam {sorted(faltantes)}.")


def unico(df: DataFrame, rotulo: str) -> dict:
    rows = df.limit(2).collect()
    if len(rows) != 1:
        raise RuntimeError(
            f"{rotulo}: esperado um registro concluído; retornou {len(rows)} "
            "de até dois. Não será escolhido outro experimento."
        )
    return rows[0].asDict(recursive=True)


def vazio(df: DataFrame, rotulo: str) -> None:
    if df.limit(1).count():
        raise RuntimeError(f"Verificação falhou: {rotulo}.")


def verificar_zero(rotulo: str, df: DataFrame) -> None:
    vazio(df, rotulo)


def cache(df: DataFrame) -> DataFrame:
    if CFG["cache_disco"]:
        df = df.persist(StorageLevel.DISK_ONLY)
        _CACHES.append(df)
    return df


def mostrar(nome: str, df: DataFrame, ordem=None) -> None:
    RELATORIOS[nome] = df
    print(f"\n{nome}")
    out = df.orderBy(*ordem) if ordem else df
    for campo in out.schema.fields:
        if campo.dataType.simpleString() in {"float", "double"}:
            out = out.withColumn(campo.name, F.round(campo.name, 6))
    out.show(CFG["linhas_exibir"], truncate=False)


def segmentos(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "segmento",
        F.explode(F.when(
            F.col("origem_tecnica"),
            F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS")),
        ).otherwise(F.array(F.lit("TODAS")))),
    )


def dividir(numerador, denominador):
    return F.when(denominador > 0, numerador / denominador)


def validar_alvos(alvos: list) -> list:
    if not isinstance(alvos, list) or any(
        not isinstance(a, str) or not a for a in alvos
    ):
        raise ValueError("Alvos inválidos no manifesto.")
    if len(alvos) != len(set(alvos)) or set(alvos) != set(ALVOS_ESPERADOS):
        raise ValueError("A execução não corresponde aos 17 alvos acordados.")
    return sorted(alvos)



def pequenas_linhas(df: DataFrame, rotulo: str) -> list:
    limite = CFG["max_agregados_driver"]
    rows = df.limit(limite + 1).collect()
    if len(rows) > limite:
        raise RuntimeError(
            f"{rotulo}: excedeu {limite} agregados no driver. "
            "As tabelas de eventos não são coletadas silenciosamente."
        )
    return [r.asDict(recursive=True) for r in rows]


def intervalo_pareado(
    referencia: np.ndarray,
    candidato: np.ndarray,
    perda: bool,
    replicas: int,
    seed: int,
    lote: int,
) -> dict:
    """Bootstrap pareado de médias de clientes; positivo sempre é melhora."""
    x = np.asarray(referencia, dtype=float)
    y = np.asarray(candidato, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size == 0:
        raise ValueError("Arrays pareados vazios ou incompatíveis.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Valor não finito no bootstrap.")
    sinal = -1.0 if perda else 1.0
    delta = sinal * (y - x)
    saida = {
        "n_clientes": int(x.size),
        "media_referencia": float(x.mean()),
        "media_candidato": float(y.mean()),
        "ganho": float(delta.mean()),
        "ic95_inferior": None,
        "ic95_superior": None,
        "skill_relativo": None,
        "skill_ic95_inferior": None,
        "skill_ic95_superior": None,
    }
    if perda and x.mean() > 0:
        saida["skill_relativo"] = float(1.0 - y.mean() / x.mean())
    if x.size < 2:
        return saida
    rng = np.random.default_rng(seed)
    ganhos = np.empty(replicas)
    skills = np.full(replicas, np.nan)
    for inicio in range(0, replicas, lote):
        tamanho = min(lote, replicas - inicio)
        indices = rng.integers(0, x.size, size=(tamanho, x.size))
        ganhos[inicio:inicio + tamanho] = delta[indices].mean(axis=1)
        if perda:
            xr = x[indices].mean(axis=1)
            yr = y[indices].mean(axis=1)
            np.divide(yr, xr, out=yr, where=xr > 0)
            skills[inicio:inicio + tamanho] = np.where(xr > 0, 1 - yr, np.nan)
    lo, hi = np.quantile(ganhos, [0.025, 0.975])
    saida.update(ic95_inferior=float(lo), ic95_superior=float(hi))
    if perda and np.isfinite(skills).all():
        lo, hi = np.quantile(skills, [0.025, 0.975])
        saida.update(
            skill_ic95_inferior=float(lo), skill_ic95_superior=float(hi)
        )
    return saida


def pares_bootstrap(
    cliente: DataFrame, grupos: list, metricas: list
) -> DataFrame:
    """Coleta somente agregados cliente/grupo, nunca eventos ou telemetria."""
    campos = [*grupos, "cd_bv", "variante", *metricas]
    pdf = pd.DataFrame(pequenas_linhas(cliente.select(*campos), "Bootstrap"))
    if pdf.empty:
        raise RuntimeError("Nenhum cliente para a comparação pareada.")
    comparacoes = [(REF, A), (REF, B2), (A, B2)]
    rows = []
    for valores, sub in pdf.groupby(grupos, dropna=False, sort=True):
        if not isinstance(valores, tuple):
            valores = (valores,)
        identidade = {
            c: float(v) if c == "idade_seg" else str(v)
            for c, v in zip(grupos, valores)
        }
        for metrica in metricas:
            pivot = sub.pivot(index="cd_bv", columns="variante", values=metrica)
            for ref, cand in comparacoes:
                if ref not in pivot or cand not in pivot:
                    raise RuntimeError("Falta variante na comparação pareada.")
                par = pivot[[ref, cand]]
                if not par[ref].isna().equals(par[cand].isna()):
                    raise RuntimeError("Disponibilidade desigual de métrica pareada.")
                par = par.dropna()
                if par.empty:
                    continue
                resultado = intervalo_pareado(
                    par[ref].to_numpy(float), par[cand].to_numpy(float),
                    metrica in PERDAS or metrica == "brier_acao",
                    CFG["bootstrap_replicas"], CFG["semente"],
                    CFG["bootstrap_lote"],
                )
                rows.append({
                    **identidade, "metrica": metrica,
                    "referencia": ref, "candidato": cand, **resultado,
                })
    schema = ", ".join(
        f"{c} {'double' if c == 'idade_seg' else 'string'}" for c in grupos
    ) + (
        ", metrica string, referencia string, candidato string, "
        "n_clientes long, media_referencia double, media_candidato double, "
        "ganho double, ic95_inferior double, ic95_superior double, "
        "skill_relativo double, skill_ic95_inferior double, "
        "skill_ic95_superior double"
    )
    out = spark.createDataFrame(rows, schema)
    return out.withColumn("leitura_ic", F.when(
        F.col("n_clientes") < 2, "SEM_IC_AMOSTRA_INSUFICIENTE"
    ).when(
        F.col("ic95_inferior") > 0, "GANHO_POSITIVO_NESTE_RECORTE"
    ).when(
        F.col("ic95_superior") < 0, "PERDA_NESTE_RECORTE"
    ).otherwise("IC_INCLUI_ZERO"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Mesma regra de primeiro alvo, sem vazamento entre clientes
# MAGIC A função abaixo preserva a regra do Multi-step validado. O rótulo é
# MAGIC construído sobre a trajetória inteira do split, inclusive as barreiras.
# MAGIC Só depois são selecionados snapshots elegíveis. Continuação no mesmo funil
# MAGIC não é nova entrada; reentrada após outro funil é válida.

# COMMAND ----------


def rotular_jornadas(base: DataFrame, alvos: list, k: int) -> DataFrame:
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
    erro_tempo = F.abs(delta - F.col("dur_min")) > CFG["tolerancia_duracao_seg"]
    verificar_zero("labels_arestas_exatas", df.filter(
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



def montar_referencia(treino: DataFrame, alvos: list) -> tuple:
    """Frequências empíricas com peso total um por cliente e idade no treino."""
    conhecidos = treino.filter(F.col("status_label").isin(CONHECIDOS))
    conhecidos = conhecidos.withColumn(
        "desfecho",
        F.when(F.col("status_label") == "ALVO_OBSERVADO", F.col("acao_real"))
        .otherwise(F.lit(SEM_ALVO)),
    )
    wc = Window.partitionBy("idade_seg", "cd_bv")
    conhecidos = conhecidos.withColumn(
        "peso", F.lit(1.0) / F.count("*").over(wc)
    )
    totais = conhecidos.groupBy("idade_seg").agg(
        F.count("*").alias("n_snapshots_treino_conhecidos"),
        F.countDistinct("cd_bv").alias("n_clientes_treino_conhecidos"),
        F.sum("peso").alias("peso_total"),
    )
    contagens = conhecidos.groupBy("idade_seg", "desfecho").agg(
        F.sum("peso").alias("peso_desfecho"),
        F.count("*").alias("n_snapshots_desfecho"),
        F.countDistinct("cd_bv").alias("n_clientes_desfecho"),
    )
    dimensao = spark.createDataFrame(
        [(a,) for a in [*alvos, SEM_ALVO]], "desfecho string"
    )
    grade = totais.crossJoin(F.broadcast(dimensao)).join(
        contagens, ["idade_seg", "desfecho"], "left"
    ).fillna(0, subset=[
        "peso_desfecho", "n_snapshots_desfecho", "n_clientes_desfecho"
    ])
    grade = grade.withColumn(
        "probabilidade_f", F.col("peso_desfecho") / F.col("peso_total")
    )
    ac = grade.filter(F.col("desfecho") != SEM_ALVO).withColumnRenamed(
        "desfecho", "acao"
    )
    wr = Window.partitionBy("idade_seg").orderBy(
        F.desc("probabilidade_f"), F.asc("acao")
    )
    ac = ac.withColumn("rank_f", F.row_number().over(wr))
    mapas = ac.groupBy("idade_seg").agg(
        F.map_from_entries(F.collect_list(F.struct(
            "acao", "probabilidade_f"
        ))).alias("util_probs"),
        F.map_from_entries(F.collect_list(F.struct(
            "acao", "rank_f"
        ))).alias("util_ranks"),
        F.sum("probabilidade_f").alias("massa_acao_util"),
    )
    mapas = mapas.join(
        grade.filter(F.col("desfecho") == SEM_ALVO).select(
            "idade_seg",
            F.col("probabilidade_f").alias("prob_sem_alvo_modelada"),
        ), "idade_seg", "inner"
    ).withColumn("massa_sem_modelo", F.lit(0.0))
    return grade, mapas


def faixas_ponderadas(long: DataFrame, faixas: list) -> DataFrame:
    """Fatias de score com empates fracionados sem olhar rótulos no corte.

    Para um bloco empatado que cruza a fronteira, usa a mesma fração para
    todos os seus registros. É a expectativa de desempate aleatório, sem
    usar CPF, label ou ordem física para fabricar ganho.
    """
    grupo = ["segmento", "idade_seg", "variante", "acao"]
    totais = long.groupBy(*grupo).agg(
        F.sum("peso").alias("peso_total"),
        F.sum(F.col("peso") * F.col("y")).alias("positivos_total"),
        F.count("*").alias("n_snapshots"),
        F.countDistinct("cd_bv").alias("n_clientes"),
        F.countDistinct(F.when(F.col("y") == 1, F.col("cd_bv")))
        .alias("n_clientes_positivos"),
        F.countDistinct("score").alias("n_scores_distintos"),
    )
    blocos = long.groupBy(*grupo, "score").agg(
        F.sum("peso").alias("peso_bloco"),
        F.sum(F.col("peso") * F.col("y")).alias("positivos_bloco"),
    )
    ordem = Window.partitionBy(*grupo).orderBy(F.desc("score"))
    antes = ordem.rowsBetween(Window.unboundedPreceding, -1)
    blocos = blocos.withColumn(
        "peso_antes", F.coalesce(F.sum("peso_bloco").over(antes), F.lit(0.0))
    ).join(totais, grupo, "inner")
    dim = spark.createDataFrame(
        faixas, "faixa string, fracao_inicio double, fracao_fim double"
    )
    grade = blocos.crossJoin(F.broadcast(dim))
    grade = grade.withColumn(
        "peso_selecionado",
        F.greatest(
            F.lit(0.0),
            F.least(
                F.col("peso_antes") + F.col("peso_bloco"),
                F.col("peso_total") * F.col("fracao_fim"),
            ) - F.greatest(
                F.col("peso_antes"),
                F.col("peso_total") * F.col("fracao_inicio"),
            ),
        ),
    ).withColumn(
        "fracao_bloco", F.col("peso_selecionado") / F.col("peso_bloco")
    )
    out = grade.groupBy(
        *grupo, "faixa", "fracao_inicio", "fracao_fim"
    ).agg(
        F.sum("peso_selecionado").alias("peso_faixa"),
        F.sum(F.col("fracao_bloco") * F.col("positivos_bloco"))
        .alias("positivos_faixa_esperados"),
        F.sum(F.col("peso_selecionado") * F.col("score"))
        .alias("soma_score_ponderado"),
    ).join(totais, grupo, "inner")
    out = out.withColumn(
        "taxa_populacao", dividir(F.col("positivos_total"), F.col("peso_total"))
    ).withColumn(
        "taxa_faixa", dividir(F.col("positivos_faixa_esperados"), F.col("peso_faixa"))
    ).withColumn(
        "score_medio_faixa", dividir(F.col("soma_score_ponderado"), F.col("peso_faixa"))
    ).withColumn(
        "lift", dividir(F.col("taxa_faixa"), F.col("taxa_populacao"))
    ).withColumn(
        "captura_positivos",
        dividir(F.col("positivos_faixa_esperados"), F.col("positivos_total")),
    ).withColumn(
        "suporte_leitura",
        F.when(F.col("n_clientes_positivos") == 0, "SEM_POSITIVOS_AVALIAVEIS")
        .when(
            F.col("n_clientes_positivos") < CFG["min_clientes_positivos_alerta"],
            "POUCOS_CLIENTES_POSITIVOS",
        ).otherwise("AVALIAR_CONTAGENS_E_INCERTEZA"),
    )
    return out.drop("soma_score_ponderado")


def autotestes_spark() -> None:
    """Pequenos testes no próprio cluster, sem acessar tabelas do banco."""
    sequencias = {
        "u1": [("b", "boleto"), ("l1", "login"), ("l2", "login"),
               ("b", "boleto"), ("q", "consulta")],
        "u2": [("l1", "login"), ("l2", "login")],
        "u3": [("q", "consulta"), (None, None), ("b", "boleto")],
        "u4": [("q", "consulta"), ("l1", "login"), ("q", "consulta")],
    }
    rows = []
    for cd, seq in sequencias.items():
        for i, (estado, funil) in enumerate(seq):
            destino = seq[i + 1][0] if i + 1 < len(seq) else None
            tipo = "exata" if estado and destino else "direita"
            rows.append((
                cd, i + 1, estado, funil,
                destino if tipo == "exata" else None, 1.0, tipo,
                estado is not None, estado is None,
            ))
    teste = spark.createDataFrame(rows, (
        "cd_bv string, passo long, estado string, funil string, destino string, "
        "dur_min double, tipo_censura string, elegivel_ajuste boolean, "
        "barreira_memoria boolean"
    )).withColumn(
        "ts_estado", F.expr("timestamp_micros(1700000000000000 + passo * 1000000)")
    )
    out = {(r.cd_bv, r.passo): r for r in rotular_jornadas(teste, ["boleto"], 2).collect()}
    assert out[("u1", 1)].acao_real == "boleto"
    assert out[("u1", 1)].passo_macro_real == 2
    assert out[("u1", 2)].passo_macro_real == 1
    assert out[("u2", 1)].status_label == "CENSURADO_ANTES_HORIZONTE"
    assert out[("u3", 1)].status_label == "CENSURADO_ANTES_HORIZONTE"
    assert out[("u4", 1)].status_label == "HORIZONTE_OBSERVADO_SEM_ALVO"
    referencia = spark.createDataFrame([
        ("c1", 0.0, "ALVO_OBSERVADO", "x"),
        ("c1", 0.0, "ALVO_OBSERVADO", "x"),
        ("c2", 0.0, "HORIZONTE_OBSERVADO_SEM_ALVO", None),
        ("c3", 0.0, "CENSURADO_ANTES_HORIZONTE", None),
    ], "cd_bv string, idade_seg double, status_label string, acao_real string")
    _, mapas = montar_referencia(referencia, ["x", "y"])
    r = mapas.first()
    assert abs(r.util_probs["x"] - 0.5) < 1e-12
    assert r.util_probs["y"] == 0
    assert abs(r.prob_sem_alvo_modelada - 0.5) < 1e-12
    empate = spark.createDataFrame([
        ("TODAS", 0.0, REF, "x", f"u{i}", 0.2, float(i < 2), 1.0)
        for i in range(10)
    ], (
        "segmento string, idade_seg double, variante string, acao string, "
        "cd_bv string, score double, y double, peso double"
    ))
    lift = faixas_ponderadas(empate, [("TOP10", 0.0, 0.1)]).first()
    assert abs(lift.lift - 1.0) < 1e-12
    assert abs(lift.captura_positivos - 0.1) < 1e-12
    assert abs(lift.peso_faixa - 1.0) < 1e-12
    print("Autotestes Spark: reentrada, censura, pesos e empates OK.")


if CFG["autotestes_spark"]:
    autotestes_spark()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Recuperar o experimento e os scores salvos
# MAGIC Não lê a base operacional de milhões de clientes. Lê somente seu
# MAGIC manifesto para resolver o histórico e o pacote corretos. Erro de ID ou
# MAGIC contrato interrompe a análise; não escolhe automaticamente outro treino.

# COMMAND ----------

execucao = unico(ler(f"{CFG['prefixo_operacional']}_execucoes_hml").filter(
    (F.col("id_execucao") == CFG["id_execucao"])
    & (F.col("status") == "CONCLUIDO_SCORING_B2")
), "Execução operacional")
run_cfg = json.loads(execucao["config_json"])
if hash_json(run_cfg) != execucao["hash_execucao"]:
    raise RuntimeError("Hash da configuração operacional divergente.")
ALVOS = validar_alvos(run_cfg["alvos"])
op_cfg = run_cfg["config"]
hist = unico(ler(op_cfg["tabela_ms03_execucoes"]).filter(
    (F.col("id_ms03") == execucao["id_ms03_validado"])
    & (F.col("status") == "CONCLUIDO_MULTISTEP_HISTORICO_03")
), "Multi-step histórico vinculado")
hist_cfg = json.loads(hist["config_json"])
if hash_json(hist_cfg) != hist["hash_ms03"]:
    raise RuntimeError("Hash da configuração histórica divergente.")
validar_alvos(hist_cfg["alvos"])
if int(hist_cfg["config"]["max_passos_macro"]) != 5:
    raise ValueError("Este estudo preserva o horizonte de cinco mudanças.")
IDADES = sorted(float(a) for a in hist_cfg["config"]["idades_seg"])
PISO_LOG = float(hist_cfg["logloss_piso"])
if not 0 < PISO_LOG < 1 or not IDADES or len(set(IDADES)) != len(IDADES):
    raise ValueError("Idades ou piso logarítmico inválidos.")
id_pacote = hist_cfg["pacote_entradas"]
if id_pacote != execucao["id_pacote_modelo"]:
    raise RuntimeError("Histórico e operacional não apontam para o mesmo pacote.")
pacote = unico(ler(op_cfg["tabela_pacotes"]).filter(
    (F.col("id_pacote") == id_pacote)
    & (F.col("status") == "CONCLUIDO_PACOTE")
), "Pacote de entradas")
meta = json.loads(pacote["meta_json"])
if hash_json(meta) != pacote["hash_pacote"]:
    raise RuntimeError("Hash de metadados do pacote divergente.")
if meta["id_pacote"] != id_pacote:
    raise RuntimeError("Identificação interna do pacote divergente.")
for interno, externo in (
    ("id_experimento", "id_experimento"),
    ("id_ajuste", "id_ajuste_fonte"),
    ("id_02c", "id_02c_fonte"),
):
    if meta[interno] != hist[externo]:
        raise RuntimeError(f"Dependência incompatível: {interno}")
preparo = meta["fit_config"]["fonte_preparo"]
spark.conf.set("spark.sql.session.timeZone", preparo["fuso_base"])
tabelas_hist = json.loads(hist["tabelas_json"])

campos_base = [
    "cd_bv", "passo", "estado", "funil", "destino", "dur_min",
    "tipo_censura", "elegivel_ajuste", "barreira_memoria", "validacao_cliente",
    "ts_estado", "ts_corte_estado", "origem_tecnica", "politica_sha256",
]
info_base = meta["entradas"]["base"]
base = ler(info_base["tabela"]).filter(F.col("id_pacote") == id_pacote)
colunas(base, {"id_experimento", *campos_base}, "Base do pacote")
base = cache(base.filter(
    F.col("id_experimento") == hist["id_experimento"]
).select(*campos_base))
vazio(base.filter(
    F.col("cd_bv").isNull() | F.col("passo").isNull()
    | F.col("validacao_cliente").isNull()
    | ~F.col("politica_sha256").eqNullSafe(F.lit(preparo["politica_sha256"]))
), "chaves, split ou política da base")
vazio(base.groupBy("cd_bv", "passo").count().filter("count != 1"), "passo duplicado")
vazio(base.groupBy("cd_bv").agg(
    F.countDistinct("validacao_cliente").alias("n")
).filter("n != 1"), "mesmo cliente nos dois splits")

campos_snap = [
    *CHAVE, "id_snapshot", "estado", "memoria_grafo", "origem_tecnica",
    "status_label", "acao_real", "ts_snapshot", "ts_corte_estado",
]
snapshots = ler(tabelas_hist["snapshots"]).filter(
    F.col("id_ms03") == hist["id_ms03"]
)
colunas(snapshots, {*campos_snap, "hash_ms03"}, "Snapshots")
vazio(snapshots.filter(
    ~F.col("hash_ms03").eqNullSafe(F.lit(hist["hash_ms03"]))
), "hash de execução nos snapshots")
snapshots = cache(snapshots.select(*campos_snap))
vazio(snapshots.groupBy(*CHAVE).count().filter("count != 1"), "snapshots duplicados")
vazio(snapshots.filter(
    F.col("id_snapshot").isNull() | F.col("origem_tecnica").isNull()
    | F.col("estado").isNull() | F.col("memoria_grafo").isNull()
    | ~F.col("idade_seg").isin(IDADES)
    | (F.col("ts_snapshot") >= F.col("ts_corte_estado"))
), "snapshots inválidos")
if not snapshots.limit(1).count():
    raise RuntimeError("O histórico selecionado não contém snapshots.")
campos_kernel = [
    "variante", *CHAVE_KERNEL, "tem_previsao", "util_probs", "util_ranks",
    "massa_acao_util", "massa_sem_modelo", "prob_sem_alvo_modelada",
]
kernels = ler(tabelas_hist["kernels"]).filter(
    (F.col("id_ms03") == hist["id_ms03"])
    & F.col("variante").isin(A, B2)
)
colunas(kernels, {*campos_kernel, "hash_ms03"}, "Kernels históricos")
vazio(kernels.filter(
    ~F.col("hash_ms03").eqNullSafe(F.lit(hist["hash_ms03"]))
), "hash de execução nos kernels")
kernels = cache(kernels.select(*campos_kernel))
vazio(kernels.groupBy("variante", *CHAVE_KERNEL).count().filter("count != 1"),
      "kernel duplicado")
expected_keys = F.array(*[F.lit(a) for a in ALVOS])
validos_kernel = kernels.filter("tem_previsao")
vazio(validos_kernel.filter(
    ~F.sort_array(F.map_keys("util_probs")).eqNullSafe(expected_keys)
    | ~F.sort_array(F.map_keys("util_ranks")).eqNullSafe(expected_keys)
    | ~F.sort_array(F.map_values("util_ranks")).eqNullSafe(
        F.array(*[F.lit(i) for i in range(1, len(ALVOS) + 1)])
    )
), "mapas de ações/ranks incompletos")
prob_kernel = validos_kernel.select(
    F.explode("util_probs").alias("acao", "p")
)
vazio(prob_kernel.filter(
    F.col("p").isNull() | F.isnan("p") | ~F.col("p").between(0.0, 1.0)
), "probabilidade de kernel inválida")
soma_kernel = reduce(lambda x, y: x + y, [
    F.element_at("util_probs", F.lit(a)) for a in ALVOS
])
vazio(validos_kernel.filter(
    F.col("massa_acao_util").isNull() | F.col("massa_sem_modelo").isNull()
    | F.col("prob_sem_alvo_modelada").isNull()
    | (F.abs(soma_kernel - F.col("massa_acao_util")) > CFG["tolerancia"])
    | (F.abs(
        F.col("massa_acao_util") + F.col("massa_sem_modelo")
        + F.col("prob_sem_alvo_modelada") - 1
    ) > CFG["tolerancia"])
), "massa dos kernels")
print("Execução operacional:", execucao["id_execucao"])
print("Histórico A/B2:", hist["id_ms03"], "| pacote:", id_pacote)
print("Idades:", IDADES, "| alvos:", len(ALVOS))
print("Somente leitura. Nenhum modelo será treinado ou classificado novamente.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Construir F exclusivamente no treino
# MAGIC Primeiro recompõe os rótulos nas trajetórias completas e verifica que a
# MAGIC regra reproduz os rótulos históricos do holdout. F usa apenas clientes
# MAGIC de treino, após essa separação. Peso um por cliente/idade entre desfechos
# MAGIC conhecidos. Não usa estado/memória, nem as frequências do holdout.
# MAGIC Sem suavização: alvo nunca observado no treino mantém probabilidade zero;
# MAGIC o mesmo piso numérico de log loss é aplicado a todos os métodos.

# COMMAND ----------

labels = cache(rotular_jornadas(base, ALVOS, 5))
labels_val = base.filter("validacao_cliente").select("cd_bv", "passo").join(
    labels.select("cd_bv", "passo", "status_label", "acao_real"),
    ["cd_bv", "passo"], "inner",
).select(
    "cd_bv", "passo", F.col("status_label").alias("_status_reconstruido"),
    F.col("acao_real").alias("_acao_reconstruida"),
)
validacao_labels = snapshots.join(labels_val, ["cd_bv", "passo"], "left")
vazio(validacao_labels.filter(
    ~F.col("status_label").eqNullSafe(F.col("_status_reconstruido"))
    | ~F.col("acao_real").eqNullSafe(F.col("_acao_reconstruida"))
), "reprodução de labels / presença do holdout no pacote")
idades_df = spark.createDataFrame([(a,) for a in IDADES], "idade_seg double")
origens_treino = base.filter(
    ~F.col("validacao_cliente") & F.col("elegivel_ajuste")
    & ~F.col("barreira_memoria") & F.col("estado").isNotNull()
).join(labels, ["cd_bv", "passo"], "inner")
treino_snap = origens_treino.crossJoin(F.broadcast(idades_df)).filter(
    F.col("dur_min") > F.col("idade_seg")
).withColumn(
    "ts_snapshot", F.expr(
        "timestamp_micros(unix_micros(ts_estado) "
        "+ cast(round(idade_seg * 1000000) as bigint))"
    )
).filter(F.col("ts_snapshot") < F.col("ts_corte_estado"))
treino_snap = cache(treino_snap.select(
    *CHAVE, "status_label", "acao_real", "ts_snapshot", "ts_acao_real"
))
vazio(treino_snap.filter(
    (F.col("status_label") == "ALVO_OBSERVADO")
    & (F.col("ts_acao_real") <= F.col("ts_snapshot"))
), "label de treino anterior ao snapshot")
vazio(treino_snap.select("cd_bv").distinct().join(
    snapshots.select("cd_bv").distinct(), "cd_bv", "inner"
), "vazamento de clientes do holdout na referência")
mostrar("FREQ_01_COBERTURA_TREINO", treino_snap.groupBy(
    "idade_seg", "status_label"
).agg(F.count("*").alias("n_snapshots"),
      F.countDistinct("cd_bv").alias("n_clientes")),
        ["idade_seg", "status_label"])
frequencias, mapa_f = montar_referencia(treino_snap, ALVOS)
mapa_f = cache(mapa_f)
vazio(idades_df.join(mapa_f.select("idade_seg"), "idade_seg", "left_anti"),
      "idade sem desfechos conhecidos no treino")
vazio(mapa_f.filter(
    F.abs(F.col("massa_acao_util") + F.col("prob_sem_alvo_modelada") - 1)
    > CFG["tolerancia"]
), "referência F não fecha massa")
mostrar("FREQ_02_REFERENCIA_F", frequencias.select(
    "idade_seg", "desfecho", "n_clientes_desfecho", "n_snapshots_desfecho",
    "n_clientes_treino_conhecidos", "n_snapshots_treino_conhecidos",
    "probabilidade_f",
), ["idade_seg", F.desc("probabilidade_f"), "desfecho"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Mesmos casos e probabilidades completas de F, A e B2
# MAGIC A comparação principal usa a interseção de snapshots suportados por A/B2.
# MAGIC A cobertura dos não suportados e dos censurados é reportada separadamente.
# MAGIC Não transforma falta de modelo em erro zero nem censura em negativo.
# MAGIC Brier usa os 17 indicadores. Log loss conservador preserva a massa sem
# MAGIC modelo como NÃO resolvida; não a transfere para "sem alvo". É secundário.

# COMMAND ----------

ab = cache(snapshots.join(kernels, CHAVE_KERNEL, "left"))
vazio(ab.groupBy(*CHAVE).agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v")
).filter("n != 2 OR v != 2"), "A/B2 ausentes ou repetidos por snapshot")
status_pares = ab.groupBy(*CHAVE).agg(
    F.sum(F.col("tem_previsao").cast("int")).alias("n_modelos_com_previsao")
)
cobertura = snapshots.join(status_pares, CHAVE, "inner")
mostrar("FREQ_03_COBERTURA_COMPARACAO", segmentos(cobertura).groupBy(
    "segmento", "idade_seg", "status_label", "n_modelos_com_previsao"
).agg(F.count("*").alias("n_snapshots"),
      F.countDistinct("cd_bv").alias("n_clientes")),
        ["segmento", "idade_seg", "status_label"])
comuns = cache(cobertura.filter("n_modelos_com_previsao = 2").drop(
    "n_modelos_com_previsao"
))
if not comuns.limit(1).count():
    raise RuntimeError("Sem snapshots suportados por ambas as variantes.")
vazio(comuns.filter(
    ~F.col("status_label").isin(*CONHECIDOS, "CENSURADO_ANTES_HORIZONTE")
    | F.col("status_label").isNull()
    | ((F.col("status_label") == "ALVO_OBSERVADO") & (
        F.col("acao_real").isNull() | ~F.col("acao_real").isin(ALVOS)
    ))
    | ((F.col("status_label") != "ALVO_OBSERVADO") & F.col("acao_real").isNotNull())
), "rótulo fora do contrato")
campos_score = [
    *CHAVE, "id_snapshot", "estado", "memoria_grafo", "origem_tecnica",
    "status_label", "acao_real", "ts_snapshot", "variante", "util_probs",
    "util_ranks", "massa_acao_util", "massa_sem_modelo", "prob_sem_alvo_modelada",
]
score_ab = ab.join(comuns.select(*CHAVE), CHAVE, "inner").select(*campos_score)
score_f = comuns.join(mapa_f, "idade_seg", "inner").withColumn(
    "variante", F.lit(REF)
).select(*campos_score)
scores_todos = cache(score_ab.unionByName(score_f))
vazio(scores_todos.groupBy(*CHAVE).agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v")
).filter("n != 3 OR v != 3"), "tripla F/A/B2 incompleta")

avaliaveis = scores_todos.filter(F.col("status_label").isin(CONHECIDOS))
positivo = F.col("status_label") == "ALVO_OBSERVADO"
metricas = avaliaveis.withColumn("y_algum_alvo", positivo.cast("double"))
metricas = metricas.withColumn(
    "_p_real", F.when(positivo, F.element_at("util_probs", F.col("acao_real")))
).withColumn(
    "_rank_real", F.when(positivo, F.element_at("util_ranks", F.col("acao_real")))
)
soma_quadrados = reduce(lambda x, y: x + y, [
    F.pow(F.element_at("util_probs", F.lit(a)), 2) for a in ALVOS
])
metricas = metricas.withColumn(
    "brier_acoes", soma_quadrados + F.when(
        positivo, 1.0 - 2.0 * F.col("_p_real")
    ).otherwise(0.0)
).withColumn(
    "brier_algum_alvo", F.pow(F.col("massa_acao_util") - F.col("y_algum_alvo"), 2)
).withColumn(
    "_prob_desfecho", F.when(positivo, F.col("_p_real"))
    .otherwise(F.col("prob_sem_alvo_modelada"))
).withColumn(
    "logloss_conservador", -F.log(F.greatest(F.col("_prob_desfecho"), F.lit(PISO_LOG)))
).withColumn(
    "nll_clip", (F.col("_prob_desfecho") < PISO_LOG).cast("long")
).withColumn(
    "top1", F.when(positivo, (
        (F.col("_rank_real") == 1) & (F.col("_p_real") > 0)
    ).cast("double"))
).withColumn(
    "top5", F.when(positivo, (
        (F.col("_rank_real") <= 5) & (F.col("_p_real") > 0)
    ).cast("double"))
).withColumn(
    "mrr", F.when(positivo, F.when(
        F.col("_p_real") > 0, 1.0 / F.col("_rank_real")
    ).otherwise(0.0))
)
metricas = cache(metricas)
vazio(metricas.filter(
    F.isnan("brier_acoes") | F.col("brier_acoes").isNull()
    | (F.col("brier_acoes") < -CFG["tolerancia"])
    | F.isnan("logloss_conservador") | F.col("logloss_conservador").isNull()
), "métricas inválidas")

# Confirma que a reconstrução de A/B2 reproduz as métricas já persistidas.
hp = ler(tabelas_hist["predicoes"]).filter(
    (F.col("id_ms03") == hist["id_ms03"])
    & F.col("variante").isin(A, B2)
)
colunas(hp, {*CHAVE, "variante", *METRICAS}, "Métricas históricas persistidas")
vazio(hp.groupBy(*CHAVE, "variante").count().filter("count != 1"),
      "previsão histórica duplicada")
originais = hp.select(
    *CHAVE, "variante", *[F.col(m).alias(f"_salvo_{m}") for m in METRICAS],
    F.lit(True).alias("_encontrado"),
)
check = metricas.filter(F.col("variante").isin(A, B2)).join(
    originais, [*CHAVE, "variante"], "left"
)
erros = [F.col("_encontrado").isNull()]
for m in METRICAS:
    erros.extend([
        F.col(m).isNull() != F.col(f"_salvo_{m}").isNull(),
        F.abs(F.col(m) - F.col(f"_salvo_{m}")) > CFG["tolerancia"],
    ])
vazio(check.filter(reduce(lambda x, y: x | y, erros)),
      "métricas reconstruídas diferentes de A/B2 salvos")
print("Rótulos e métricas de A/B2 reproduzidos; F calculada somente no treino.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Comparação por cliente e intervalos pareados
# MAGIC Cada cliente tem peso igual dentro do segmento/idade. Top1/5/MRR usam
# MAGIC clientes com alvo observado; Brier/log loss usam todos os desfechos
# MAGIC conhecidos. O relatório por snapshot é complementar e tem outra ponderação.
# MAGIC Ganho positivo = melhora em qualquer métrica. Para perdas,
# MAGIC skill = 1 - erro_candidato/erro_referência. IC95 é bootstrap de clientes,
# MAGIC condicional ao treino e ao baseline fixados, não retreina em cada réplica.
# MAGIC Resultados exploratórios: o holdout já orientou o desenvolvimento.

# COMMAND ----------

metricas_segmento = cache(segmentos(metricas))
por_cliente = cache(metricas_segmento.groupBy(
    "segmento", "idade_seg", "cd_bv", "variante"
).agg(
    F.count("*").alias("n_casos"),
    F.sum("y_algum_alvo").alias("n_alvos"),
    F.sum("nll_clip").alias("n_logloss_clip"),
    *[F.avg(m).alias(m) for m in METRICAS],
    F.avg("massa_sem_modelo").alias("massa_sem_modelo_media"),
))
mostrar("FREQ_04_METRICAS_CLIENTE", por_cliente.groupBy(
    "segmento", "idade_seg", "variante"
).agg(
    F.count("*").alias("n_clientes_conhecidos"),
    F.sum((F.col("n_alvos") > 0).cast("long")).alias("n_clientes_com_alvo"),
    F.sum("n_casos").alias("n_snapshots_conhecidos"),
    F.sum("n_alvos").alias("n_snapshots_com_alvo"),
    F.sum("n_logloss_clip").alias("n_logloss_clip"),
    *[F.avg(m).alias(m) for m in METRICAS],
    F.avg("massa_sem_modelo_media").alias("massa_sem_modelo_media"),
), ["segmento", "idade_seg", "variante"])
mostrar("FREQ_04_METRICAS_SNAPSHOT", metricas_segmento.groupBy(
    "segmento", "idade_seg", "variante"
).agg(
    F.count("*").alias("n_snapshots_conhecidos"),
    F.sum("y_algum_alvo").alias("n_snapshots_com_alvo"),
    *[F.avg(m).alias(m) for m in METRICAS],
), ["segmento", "idade_seg", "variante"])
pares = pares_bootstrap(por_cliente, ["segmento", "idade_seg"], METRICAS)
RELATORIOS["FREQ_05_COMPARACOES_COMPLETAS"] = pares
mostrar("FREQ_05_COMPARACAO_PAREADA", pares.filter(
    F.col("segmento") == "TODAS"
), ["idade_seg", "metrica", "referencia", "candidato"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Brier por ação e skill em relação a F
# MAGIC Todos os 17 scores participam, inclusive os que ficaram fora do Top 5.
# MAGIC Negativo significa que a ação NÃO foi o primeiro alvo naquele horizonte
# MAGIC conhecido. Não significa que ela nunca aconteceu depois na jornada.
# MAGIC A análise usa o mesmo conjunto de clientes/casos nas três variantes.

# COMMAND ----------

long_metricas = metricas_segmento.select(
    "segmento", *CHAVE, "variante", "acao_real",
    F.explode("util_probs").alias("acao", "score"),
).withColumn(
    "y", F.col("acao").eqNullSafe(F.col("acao_real")).cast("double")
).withColumn("erro_quadratico", F.pow(F.col("score") - F.col("y"), 2))
acao_cliente = cache(long_metricas.groupBy(
    "segmento", "idade_seg", "acao", "variante", "cd_bv"
).agg(
    F.avg("erro_quadratico").alias("brier_acao"),
    F.avg("score").alias("score_medio"),
    F.avg("y").alias("taxa_cliente"),
    F.count("*").alias("n_snapshots"),
    F.sum("y").alias("n_alvos"),
))
acao_resumo = acao_cliente.groupBy(
    "segmento", "idade_seg", "acao", "variante"
).agg(
    F.count("*").alias("n_clientes"),
    F.sum((F.col("n_alvos") > 0).cast("long")).alias("n_clientes_positivos"),
    F.sum("n_snapshots").alias("n_snapshots"),
    F.sum("n_alvos").alias("n_alvos"),
    F.avg("brier_acao").alias("brier_acao"),
    F.avg("score_medio").alias("score_medio_cliente"),
    F.avg("taxa_cliente").alias("taxa_media_cliente"),
)
RELATORIOS["FREQ_06_BRIER_COMPLETO"] = acao_resumo
mostrar("FREQ_06_BRIER_POR_ACAO", acao_resumo.filter(
    (F.col("segmento") == "TODAS") & (F.col("idade_seg") == 0.0)
), ["acao", "variante"])
if CFG["comparar_brier_por_acao_com_ic"]:
    pares_acao = pares_bootstrap(
        acao_cliente, ["segmento", "idade_seg", "acao"], ["brier_acao"]
    )
    suporte_acao = acao_resumo.filter(F.col("variante") == REF).select(
        "segmento", "idade_seg", "acao", "n_clientes_positivos", "n_alvos"
    )
    pares_acao = pares_acao.join(
        suporte_acao, ["segmento", "idade_seg", "acao"], "inner"
    )
    RELATORIOS["FREQ_06_PARES_ACAO_COMPLETOS"] = pares_acao
    mostrar("FREQ_06_PARES_BRIER_ACAO", pares_acao.filter(
        (F.col("segmento") == "TODAS") & (F.col("idade_seg") == 0.0)
        & (F.col("referencia") == REF) & (F.col("candidato") == B2)
    ), ["acao"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Lift: priorizar clientes para cada ação
# MAGIC Padrão: um snapshot por cliente/idade, por hash determinístico independente
# MAGIC do alvo. A seleção ocorre ANTES de excluir censuras. Os mesmos IDs são
# MAGIC usados por F/A/B2. A análise continua condicionada a desfechos conhecidos;
# MAGIC não corrige censura informativa nem reproduz um corte diário prospectivo.
# MAGIC
# MAGIC Empates na fronteira do Top 10%/20% são fracionados. O resultado é a
# MAGIC expectativa de um desempate aleatório. Assim, F tem lift exatamente 1
# MAGIC quando há positivos, e não ganha por ordenação arbitrária de IDs.
# MAGIC Lift é descritivo: não recebe IC neste notebook nem mede efeito causal.

# COMMAND ----------

if CFG["coorte_lift"] == "UM_SNAPSHOT_POR_CLIENTE":
    h = F.sha2(F.to_json(F.struct(
        *[F.col(c) for c in CHAVE], F.lit(CFG["semente"]).alias("semente")
    )), 256)
    wl = Window.partitionBy("cd_bv", "idade_seg").orderBy(h, F.col("passo"))
    ids_lift = comuns.withColumn("_rn", F.row_number().over(wl)).filter(
        "_rn = 1"
    ).select(*CHAVE)
else:
    ids_lift = comuns.select(*CHAVE)
coorte_lift = cache(comuns.join(ids_lift, CHAVE, "inner"))
mostrar("FREQ_07_COBERTURA_LIFT", segmentos(coorte_lift).groupBy(
    "segmento", "idade_seg", "status_label"
).agg(F.count("*").alias("n_snapshots"),
      F.countDistinct("cd_bv").alias("n_clientes")),
        ["segmento", "idade_seg", "status_label"])
score_lift = segmentos(scores_todos.join(ids_lift, CHAVE, "inner").filter(
    F.col("status_label").isin(CONHECIDOS)
))
wc_lift = Window.partitionBy("segmento", "idade_seg", "variante", "cd_bv")
score_lift = score_lift.withColumn(
    "peso", F.lit(1.0) / F.count("*").over(wc_lift)
)
long_lift = cache(score_lift.select(
    "segmento", *CHAVE, "variante", "acao_real", "peso",
    F.explode("util_probs").alias("acao", "score"),
).withColumn(
    "y", F.col("acao").eqNullSafe(F.col("acao_real")).cast("double")
))
fatias_top = [
    (f"TOP_{int(round(q * 100)):02d}", 0.0, float(q))
    for q in CFG["fracoes_lift"]
]
if any(not 0 < q <= 1 for q in CFG["fracoes_lift"]):
    raise ValueError("Frações de lift devem estar no intervalo (0, 1].")
lift = cache(faixas_ponderadas(long_lift, fatias_top))
vazio(lift.filter(
    (F.col("variante") == REF) & (F.col("taxa_populacao") > 0)
    & (F.abs(F.col("lift") - 1) > CFG["tolerancia"])
), "lift artificial na referência constante")
RELATORIOS["FREQ_08_LIFT_COMPLETO"] = lift
mostrar("FREQ_08_LIFT_TOP10_TOP20", lift.filter(
    (F.col("variante") == B2) & (F.col("idade_seg") == 0.0)
    & (F.col("segmento") == "TODAS")
).select(
    "acao", "faixa", "n_clientes", "n_clientes_positivos", "n_scores_distintos",
    "peso_faixa", "positivos_faixa_esperados", "taxa_populacao", "taxa_faixa",
    "lift", "captura_positivos", "score_medio_faixa", "suporte_leitura",
), ["acao", "faixa"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Curva por faixa de score (diagnóstico de ordenação e calibração)
# MAGIC Faixa 1 = maiores scores; última = menores. Faixas com empates podem ter
# MAGIC os mesmos scores/frequências: isso é informação, não erro. A comparação
# MAGIC score médio × taxa observada é exploratória e pode sofrer seleção por
# MAGIC censura. Não recalibra, não renormaliza, não cria confiança alta/média/baixa.

# COMMAND ----------

n_faixas = int(CFG["faixas_score"])
if n_faixas < 2:
    raise ValueError("Informe pelo menos duas faixas de score.")
fatias_curva = [
    (f"FAIXA_{i + 1:02d}", i / n_faixas, (i + 1) / n_faixas)
    for i in range(n_faixas)
]
curvas = faixas_ponderadas(long_lift, fatias_curva)
RELATORIOS["FREQ_09_CURVAS_COMPLETAS"] = curvas
mostrar("FREQ_09_CURVA_BOLETO", curvas.filter(
    (F.col("variante") == B2) & (F.col("idade_seg") == 0.0)
    & (F.col("segmento") == "TODAS") & (F.col("acao") == "pagamentos_boleto")
).select(
    "acao", "faixa", "peso_faixa", "positivos_faixa_esperados",
    "score_medio_faixa", "taxa_faixa", "taxa_populacao", "lift",
    "n_clientes", "n_clientes_positivos", "n_scores_distintos",
), ["faixa"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Painel final e limites de interpretação
# MAGIC Ganho = desempenho preditivo, não diversidade nem efeito de campanha.
# MAGIC Brier e ranking têm IC pareado por cliente. Lift/curvas são descritivos.
# MAGIC Os denominadores de loss, ranking positivo e coorte de lift são diferentes
# MAGIC e estão explícitos. Falta de positivos gera métrica indefinida, não zero.
# MAGIC Não usar este holdout já explorado como prova final de generalização.

# COMMAND ----------

brier_painel = acao_resumo.filter(
    (F.col("segmento") == "TODAS") & (F.col("idade_seg") == 0.0)
).groupBy("acao").agg(
    F.max("n_clientes_positivos").alias("n_clientes_positivos_brier"),
    F.max(F.when(F.col("variante") == REF, F.col("brier_acao"))).alias("brier_f"),
    F.max(F.when(F.col("variante") == A, F.col("brier_acao"))).alias("brier_a"),
    F.max(F.when(F.col("variante") == B2, F.col("brier_acao"))).alias("brier_b2"),
).withColumn("skill_b2_vs_f", F.when(
    F.col("brier_f") > 0, 1.0 - F.col("brier_b2") / F.col("brier_f")
))
lift_painel = lift.filter(
    (F.col("segmento") == "TODAS") & (F.col("idade_seg") == 0.0)
    & (F.col("variante") == B2)
).groupBy("acao").agg(
    F.max("n_clientes").alias("n_clientes_coorte_lift"),
    F.max("n_clientes_positivos").alias("n_clientes_positivos_lift"),
    *[
        F.max(F.when(F.col("faixa") == nome, F.col("lift")))
        .alias(f"lift_b2_{nome.lower()}") for nome, _, _ in fatias_top
    ],
)
mostrar("FREQ_10_PAINEL_ACOES", brier_painel.join(lift_painel, "acao", "left"),
        ["acao"])
print("\nFREQ_11_CONCLUSAO")
print("Execução operacional de referência:", execucao["id_execucao"])
print("Histórico comparado:", hist["id_ms03"])
print("Pacote preservado:", id_pacote)
print("Baseline: frequências do primeiro desfecho; treino; peso igual por cliente/idade.")
print("Modelos: F × A × B2; mesmo alvo, horizonte, snapshots e piso logarítmico.")
print("Coorte de lift:", CFG["coorte_lift"], "| desempate neutro por fração.")
print("Nenhum treinamento/scoring executado. Nenhuma tabela ou view modificada.")
print("IC95 exploratórios, sem ajuste por múltiplas comparações.")
print("Os relatórios completos, sem arredondamento, estão em RELATORIOS:")
print(sorted(RELATORIOS))
print("Priorize FREQ_05_COMPARACAO_PAREADA, FREQ_06_PARES_BRIER_ACAO e FREQ_08_LIFT_TOP10_TOP20.")
# Libera apenas os caches criados neste estudo. Os DataFrames continuam acessíveis.
for _df in _CACHES:
    _df.unpersist(blocking=False)
