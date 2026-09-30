# Databricks notebook source
# NBA | Parte 08 V3: output final do MVP
# Executar depois da Parte 04, no mesmo notebook.
# Nao retreina o modelo.
#
# Output por cliente:
# - Top 5 proximos estados imediatos;
# - Top 5 primeiras acoes relevantes em ate 5 transicoes.
#
# Os dois rankings sao independentes, mas usam a mesma coluna "ranking":
# ranking=1 traz o Top 1 de cada lista, ranking=2 o Top 2 etc.

from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy import special
from pyspark.sql import functions as F


# =========================
# Configuracao
# =========================

MAX_PASSOS = 5
TOP_K_ESTADOS = 5
TOP_K_ACOES = 5

ALVO_MVP = "pagamentos_boleto:::topo"

ACOES_RELEVANTES = {
    "pagamentos_boleto:::topo",
    "pix_cadastro_chave:::topo",
    "pix_trazer_chave:::topo",
    "seguros_auto_avulso_cotacao:::topo",
    "gerar_boleto_financeira:::topo",
}

GRAVAR_MVP = False
TABELA_MVP = (
    "ctg_dsti.renato_nba.nba_semimarkov_mvp_top5_v21_hml"
)
VIEW_MVP = "nba_sm_v21_mvp_top5"

REQUERIDOS = {
    "atuais_scoring",
    "SMD_BROADCAST",
    "smd_log_s_grupos",
    "smd_prever_destinos",
    "SMD_HORIZONTE",
    "SMD_VERSAO",
}

faltantes = sorted(
    nome for nome in REQUERIDOS if nome not in globals()
)
if faltantes:
    raise RuntimeError(
        "Execute a Parte 04 antes. Variaveis ausentes: "
        f"{faltantes}"
    )

if ALVO_MVP not in ACOES_RELEVANTES:
    raise ValueError("ALVO_MVP precisa pertencer a ACOES_RELEVANTES.")

if min(MAX_PASSOS, TOP_K_ESTADOS, TOP_K_ACOES) < 1:
    raise ValueError("MAX_PASSOS e TOP_K precisam ser >= 1.")

if not SMD_BROADCAST.value:
    raise ValueError("Nenhum modelo Semi-Markov disponivel.")

print("Acoes relevantes:", sorted(ACOES_RELEVANTES))
print("Alvo oficial do MVP:", ALVO_MVP)


# COMMAND ----------
# Funcoes auxiliares


def status_temporal_modelo(modelo: dict, idade_dias: float) -> str:
    """Resume o componente temporal usado no estado atual."""
    if idade_dias > float(modelo["max_tempo_observado_dias"]):
        return "EXTRAPOLACAO_TEMPORAL"

    tipos = modelo.get("tipo_grupo", [])
    if "atomo" in "|".join(tipos):
        return "TIMEOUT_MISTO_POR_DESTINO"
    if int(modelo["n_grupos"]) > 1:
        return "GRUPOS_TEMPORAIS_POR_DESTINO"
    return "POOLING_UM_GRUPO_TEMPORAL"


def probs_idade_zero(
    estado: str,
    modelos: dict,
    cache: dict,
) -> dict[str, float] | None:
    """P(destino | estado) ao entrar no estado."""
    if estado in cache:
        return cache[estado]

    modelo = modelos.get(estado)
    if modelo is None:
        cache[estado] = None
        return None

    probs = {
        destino: float(prob)
        for destino, prob in zip(
            modelo["destinos"],
            modelo["p_destino"],
        )
        if prob > 0
    }
    cache[estado] = probs
    return probs


def propagar_primeira_acao_relevante(
    q_primeiro: dict[str, float],
    modelos: dict,
    cache_zero: dict,
) -> tuple[dict, float, float, float]:
    """Primeira passagem por acoes relevantes em ate MAX_PASSOS."""
    contribuicoes = {
        acao: np.zeros(MAX_PASSOS, dtype=float)
        for acao in ACOES_RELEVANTES
    }
    vivos = defaultdict(float)

    for destino, prob in q_primeiro.items():
        if destino in ACOES_RELEVANTES:
            contribuicoes[destino][0] += prob
        else:
            vivos[destino] += prob

    massa_perdida = 0.0

    for passo_idx in range(1, MAX_PASSOS):
        proximos_vivos = defaultdict(float)

        for origem, massa in vivos.items():
            probs = probs_idade_zero(origem, modelos, cache_zero)
            if probs is None:
                massa_perdida += massa
                continue

            for destino, prob in probs.items():
                fluxo = massa * prob
                if destino in ACOES_RELEVANTES:
                    contribuicoes[destino][passo_idx] += fluxo
                else:
                    proximos_vivos[destino] += fluxo

        vivos = proximos_vivos

    scores = {}
    for acao, por_passo in contribuicoes.items():
        score = float(por_passo.sum())
        if score <= 0:
            continue

        passo_modal = int(np.argmax(por_passo) + 1)
        scores[acao] = {
            "score": score,
            "passo_maior_contribuicao": passo_modal,
            "prob_no_passo_maior": float(
                por_passo[passo_modal - 1]
            ),
        }

    prob_alguma = float(
        sum(info["score"] for info in scores.values())
    )
    prob_sem_acao = float(sum(vivos.values()))
    fechamento = prob_alguma + prob_sem_acao + massa_perdida

    if not np.isclose(fechamento, 1.0, atol=1e-8):
        raise ValueError(
            f"Massa da propagacao nao fecha: {fechamento}"
        )

    return scores, prob_alguma, prob_sem_acao, float(massa_perdida)


# COMMAND ----------
# Schema

SCHEMA_MVP = (
    "cd_bv string, data_referencia date, ts_corte_estado timestamp, "
    "estado_atual string, tempo_no_estado_seg double, ranking long, "
    "proximo_estado_imediato string, prob_proximo_estado double, "
    "proxima_acao_relevante string, score_acao_relevante double, "
    "passo_maior_contribuicao long, prob_no_passo_maior double, "
    "prob_alguma_acao_relevante_ate_5 double, "
    "prob_sem_acao_relevante_modelada_ate_5 double, "
    "massa_perdida_sem_modelo double, cobertura_propagacao double, "
    "status_previsao string, status_temporal string, "
    "status_dados string, versao_modelo string, "
    "alvo_mvp boolean, max_passos long, publicavel boolean"
)
COLUNAS_MVP = [
    item.strip().split()[0] for item in SCHEMA_MVP.split(",")
]


# COMMAND ----------
# Scoring multi-step


def prever_mvp_pdf(
    pdf: pd.DataFrame,
    modelos: dict,
    cache_zero: dict,
) -> pd.DataFrame:
    saidas = []

    for origem, parte in pdf.groupby(
        "acao_atual",
        sort=False,
        dropna=False,
    ):
        modelo = modelos.get(origem)

        for inicio in range(0, len(parte), 512):
            bloco = parte.iloc[inicio:inicio + 512].reset_index(
                drop=True
            )
            valido = (
                (bloco["status_input"] == "OK")
                & bloco["tempo_no_estado_seg"].notna()
                & (bloco["tempo_no_estado_seg"] >= 0)
            )
            selecionados = (
                np.flatnonzero(valido.to_numpy())
                if modelo is not None
                else np.array([], dtype=int)
            )

            previsoes_primeiro = {}
            fora_suporte = set()

            if len(selecionados):
                idades = (
                    bloco.iloc[selecionados]["tempo_no_estado_seg"]
                    .to_numpy(float)
                    / 86400.0
                )
                ls = smd_log_s_grupos(modelo, idades)
                den = special.logsumexp(
                    np.log(
                        np.asarray(modelo["pi_grupo"], float)
                    )
                    + ls,
                    axis=1,
                )
                suportados = np.isfinite(den)
                fora_suporte = set(
                    selecionados[~suportados].tolist()
                )
                dentro = selecionados[suportados]

                if len(dentro):
                    q, _, _ = smd_prever_destinos(
                        modelo,
                        idades[suportados],
                        SMD_HORIZONTE,
                    )
                    for k, ix in enumerate(dentro):
                        previsoes_primeiro[int(ix)] = (
                            q[k],
                            idades[suportados][k],
                        )

            for ix, row in bloco.iterrows():
                base = {
                    "cd_bv": str(row["cd_bv"]),
                    "data_referencia": row["data_referencia"],
                    "ts_corte_estado": row["ts_corte_estado"],
                    "estado_atual": (
                        None
                        if pd.isna(row["acao_atual"])
                        else str(row["acao_atual"])
                    ),
                    "tempo_no_estado_seg": (
                        None
                        if pd.isna(row["tempo_no_estado_seg"])
                        else float(row["tempo_no_estado_seg"])
                    ),
                    "ranking": None,
                    "proximo_estado_imediato": None,
                    "prob_proximo_estado": None,
                    "proxima_acao_relevante": None,
                    "score_acao_relevante": None,
                    "passo_maior_contribuicao": None,
                    "prob_no_passo_maior": None,
                    "prob_alguma_acao_relevante_ate_5": None,
                    "prob_sem_acao_relevante_modelada_ate_5": None,
                    "massa_perdida_sem_modelo": None,
                    "cobertura_propagacao": None,
                    "status_previsao": "SEM_MODELO_ORIGEM",
                    "status_temporal": "NAO_CALCULADO",
                    "status_dados": str(row["status_dados"]),
                    "versao_modelo": SMD_VERSAO,
                    "alvo_mvp": None,
                    "max_passos": int(MAX_PASSOS),
                    "publicavel": False,
                }

                if ix not in previsoes_primeiro:
                    if row["status_input"] != "OK":
                        base["status_previsao"] = str(
                            row["status_input"]
                        )
                    elif ix in fora_suporte:
                        base["status_previsao"] = (
                            "IDADE_FORA_SUPORTE_MODELO"
                        )
                    saidas.append(base)
                    continue

                q_primeiro, idade = previsoes_primeiro[ix]
                destinos = modelo["destinos"]

                idx_imediatos = np.argsort(
                    -q_primeiro,
                    kind="mergesort",
                )
                idx_imediatos = idx_imediatos[
                    q_primeiro[idx_imediatos] > 0
                ][:TOP_K_ESTADOS]

                top_imediatos = [
                    (
                        destinos[j],
                        float(q_primeiro[j]),
                    )
                    for j in idx_imediatos
                ]

                q_primeiro_dict = {
                    destino: float(prob)
                    for destino, prob in zip(
                        destinos,
                        q_primeiro,
                    )
                    if prob > 0
                }

                (
                    scores,
                    prob_alguma,
                    prob_sem_acao,
                    massa_perdida,
                ) = propagar_primeira_acao_relevante(
                    q_primeiro_dict,
                    modelos,
                    cache_zero,
                )

                top_acoes = sorted(
                    scores.items(),
                    key=lambda item: (
                        -item[1]["score"],
                        item[0],
                    ),
                )[:TOP_K_ACOES]

                cobertura = float(1.0 - massa_perdida)
                status_temporal = status_temporal_modelo(
                    modelo,
                    idade,
                )
                n_linhas = max(
                    len(top_imediatos),
                    len(top_acoes),
                )

                if n_linhas == 0:
                    comum = {
                        **base,
                        "prob_alguma_acao_relevante_ate_5": (
                            prob_alguma
                        ),
                        "prob_sem_acao_relevante_modelada_ate_5": (
                            prob_sem_acao
                        ),
                        "massa_perdida_sem_modelo": massa_perdida,
                        "cobertura_propagacao": cobertura,
                        "status_previsao": (
                            "SEM_ACAO_RELEVANTE_ATE_5_PASSOS"
                        ),
                        "status_temporal": status_temporal,
                    }
                    saidas.append(comum)
                    continue

                for pos in range(1, n_linhas + 1):
                    imediato = (
                        top_imediatos[pos - 1]
                        if pos <= len(top_imediatos)
                        else (None, None)
                    )
                    acao = (
                        top_acoes[pos - 1]
                        if pos <= len(top_acoes)
                        else (None, None)
                    )
                    acao_nome, info = acao

                    saidas.append({
                        **base,
                        "ranking": pos,
                        "proximo_estado_imediato": imediato[0],
                        "prob_proximo_estado": imediato[1],
                        "proxima_acao_relevante": acao_nome,
                        "score_acao_relevante": (
                            None
                            if info is None
                            else float(info["score"])
                        ),
                        "passo_maior_contribuicao": (
                            None
                            if info is None
                            else int(
                                info[
                                    "passo_maior_contribuicao"
                                ]
                            )
                        ),
                        "prob_no_passo_maior": (
                            None
                            if info is None
                            else float(
                                info["prob_no_passo_maior"]
                            )
                        ),
                        "prob_alguma_acao_relevante_ate_5": (
                            prob_alguma
                        ),
                        "prob_sem_acao_relevante_modelada_ate_5": (
                            prob_sem_acao
                        ),
                        "massa_perdida_sem_modelo": massa_perdida,
                        "cobertura_propagacao": cobertura,
                        "status_previsao": "PREVISAO_MVP_MULTISTEP",
                        "status_temporal": status_temporal,
                        "alvo_mvp": acao_nome == ALVO_MVP,
                    })

    resultado = pd.DataFrame(
        saidas,
        columns=COLUNAS_MVP,
    )
    if len(resultado):
        for coluna in (
            "ranking",
            "passo_maior_contribuicao",
            "max_passos",
        ):
            resultado[coluna] = pd.array(
                resultado[coluna],
                dtype="Int64",
            )

    return resultado


def prever_mvp_lotes(
    iterator: Iterator[pd.DataFrame],
):
    modelos = SMD_BROADCAST.value
    cache_zero = {}

    for pdf in iterator:
        yield prever_mvp_pdf(
            pdf,
            modelos,
            cache_zero,
        )


# COMMAND ----------
# Execucao

input_mvp = (
    atuais_scoring
    .select(
        "cd_bv",
        "data_referencia",
        "ts_corte_estado",
        "acao_atual",
        "tempo_no_estado_seg",
        "status_input",
        "status_dados",
        "relogio",
    )
    .repartition("acao_atual")
)

mvp_output = (
    input_mvp
    .mapInPandas(
        prever_mvp_lotes,
        schema=SCHEMA_MVP,
    )
    .localCheckpoint(eager=True)
)

mvp_output.createOrReplaceTempView(VIEW_MVP)


# COMMAND ----------
# Auditoria

n_input = atuais_scoring.select("cd_bv").distinct().count()
n_output = mvp_output.select("cd_bv").distinct().count()

if n_input != n_output:
    raise ValueError(
        "Output nao preservou todos os clientes: "
        f"input={n_input}, output={n_output}"
    )

prev = mvp_output.filter(F.col("ranking").isNotNull())

if prev.limit(1).count():
    invalidas = prev.filter(
        (
            F.col("prob_proximo_estado").isNotNull()
            & ~F.col("prob_proximo_estado").between(0.0, 1.0)
        )
        | (
            F.col("score_acao_relevante").isNotNull()
            & ~F.col("score_acao_relevante").between(0.0, 1.0)
        )
        | ~F.col("cobertura_propagacao").between(0.0, 1.0)
    )
    if invalidas.limit(1).count():
        raise ValueError("Probabilidade ou score invalido.")

    duplicados_estado = (
        prev
        .filter(F.col("proximo_estado_imediato").isNotNull())
        .groupBy("cd_bv", "proximo_estado_imediato")
        .count()
        .filter(F.col("count") > 1)
    )
    if duplicados_estado.limit(1).count():
        raise ValueError("Estado imediato duplicado no ranking.")

    duplicadas_acao = (
        prev
        .filter(F.col("proxima_acao_relevante").isNotNull())
        .groupBy("cd_bv", "proxima_acao_relevante")
        .count()
        .filter(F.col("count") > 1)
    )
    if duplicadas_acao.limit(1).count():
        raise ValueError("Acao relevante duplicada no ranking.")

check_massa = (
    mvp_output
    .filter(F.col("ranking") == 1)
    .filter(
        F.col("prob_alguma_acao_relevante_ate_5").isNotNull()
    )
    .select(
        "cd_bv",
        (
            F.col("prob_alguma_acao_relevante_ate_5")
            + F.col(
                "prob_sem_acao_relevante_modelada_ate_5"
            )
            + F.col("massa_perdida_sem_modelo")
        ).alias("soma"),
    )
    .filter(F.abs(F.col("soma") - 1.0) > 1e-8)
)

if check_massa.limit(1).count():
    raise ValueError("Massa multi-step nao fecha.")


# COMMAND ----------
# Resumos

print("RESUMO MVP")
(
    mvp_output
    .groupBy("status_previsao", "status_temporal")
    .agg(
        F.countDistinct("cd_bv").alias("n_clientes"),
        F.count("*").alias("n_linhas"),
    )
    .orderBy(F.desc("n_clientes"))
    .show(50, truncate=False)
)

print("TOP 1 - ESTADO IMEDIATO")
(
    mvp_output
    .filter(F.col("ranking") == 1)
    .groupBy("proximo_estado_imediato")
    .agg(
        F.countDistinct("cd_bv").alias("n_clientes"),
        F.avg("prob_proximo_estado").alias("prob_media"),
    )
    .orderBy(F.desc("n_clientes"))
    .show(50, truncate=False)
)

print("TOP 1 - ACAO RELEVANTE")
(
    mvp_output
    .filter(F.col("ranking") == 1)
    .groupBy("proxima_acao_relevante")
    .agg(
        F.countDistinct("cd_bv").alias("n_clientes"),
        F.avg("score_acao_relevante").alias("score_medio"),
        F.avg("cobertura_propagacao").alias("cobertura_media"),
    )
    .orderBy(F.desc("n_clientes"))
    .show(50, truncate=False)
)

if (
    "MODO_SCORING" in globals()
    and MODO_SCORING == "TESTE_CD_BV"
):
    print("OUTPUT DO CLIENTE DE TESTE")
    mvp_output.orderBy("ranking").show(20, truncate=False)


# COMMAND ----------
# Gravacao

if GRAVAR_MVP:
    if (
        "MODO_SCORING" not in globals()
        or MODO_SCORING != "TODOS"
    ):
        raise ValueError(
            "Gravacao so e permitida com MODO_SCORING='TODOS'."
        )

    if not mvp_output.limit(1).count():
        raise ValueError("Output MVP vazio.")

    output_gravar = (
        mvp_output
        .withColumn("gravado_em", F.current_timestamp())
        .withColumn("ambiente", F.lit("HOMOLOGACAO"))
    )

    data_ref = (
        atuais_scoring
        .select(
            F.date_format(
                "data_referencia",
                "yyyy-MM-dd",
            ).alias("data_ref")
        )
        .first()["data_ref"]
    )

    predicado = (
        f"data_referencia = DATE '{data_ref}' "
        f"AND versao_modelo = '{SMD_VERSAO}'"
    )

    writer = output_gravar.write.format("delta")

    if spark.catalog.tableExists(TABELA_MVP):
        (
            writer
            .mode("overwrite")
            .option("replaceWhere", predicado)
            .saveAsTable(TABELA_MVP)
        )
    else:
        (
            writer
            .mode("errorifexists")
            .saveAsTable(TABELA_MVP)
        )

    print("MVP gravado em:", TABELA_MVP)

else:
    print(
        "Nenhuma tabela gravada. "
        "Para o publico completo, rode a Parte 04 com "
        "MODO_SCORING='TODOS' e depois use GRAVAR_MVP=True."
    )
