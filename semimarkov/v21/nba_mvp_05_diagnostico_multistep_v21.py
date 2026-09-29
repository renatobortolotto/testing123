# Databricks notebook source
# NBA | Parte 05: diagnostico multi-step
# Executar depois da Parte 04 scoring-only, no mesmo notebook.

from collections import defaultdict

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CD_BV_ANALISE = str(CD_BV_TESTE)
N_PASSOS = 5
PADRAO_ALVO = "pix"
TOP_N = 15


def obter_probabilidades(estado: str, idade_dias: float = 0.0):
    modelo = MODELOS.get(estado)
    if modelo is None:
        return None

    q, _, _ = prever_destinos(
        modelo,
        np.array([idade_dias], dtype=float),
        HORIZONTE_DIAS,
    )

    return {
        destino: float(prob)
        for destino, prob in zip(modelo["destinos"], q[0])
        if prob > 0
    }


cliente = (
    atuais
    .filter(F.col("cd_bv") == CD_BV_ANALISE)
    .select(
        "cd_bv",
        "acao_atual",
        "tempo_no_estado_seg",
        "status_input",
        "ultima_acao_observada",
        "ts_ultima_atividade",
        "ts_inicio",
        "ts_corte_estado",
    )
    .first()
)

if cliente is None:
    raise ValueError(
        f"Cliente {CD_BV_ANALISE} nao encontrado no input de scoring."
    )

if cliente["status_input"] != "OK":
    raise ValueError(
        f"Cliente nao apto ao scoring: {cliente['status_input']}"
    )

estado_atual = cliente["acao_atual"]
idade_dias = float(cliente["tempo_no_estado_seg"]) / 86400.0

print("Cliente:", CD_BV_ANALISE)
print("Estado atual:", estado_atual)
print("Idade do estado em dias:", idade_dias)
print("Ultima acao observada:", cliente["ultima_acao_observada"])

# Distribuicao exata em cada numero de passos.
distribuicoes = []
massa_sem_modelo = []

primeiro = obter_probabilidades(estado_atual, idade_dias)
if primeiro is None:
    raise ValueError(f"Nao existe modelo para o estado atual: {estado_atual}")

dist = primeiro
distribuicoes.append(dist)
massa_sem_modelo.append(0.0)

for passo in range(2, N_PASSOS + 1):
    proxima_dist = defaultdict(float)
    perdida = 0.0

    for origem, massa in dist.items():
        probs = obter_probabilidades(origem, 0.0)

        if probs is None:
            perdida += massa
            continue

        for destino, prob in probs.items():
            proxima_dist[destino] += massa * prob

    dist = dict(proxima_dist)
    distribuicoes.append(dist)
    massa_sem_modelo.append(perdida)

# Top estados por passo.
linhas = []

for passo, dist_passo in enumerate(distribuicoes, start=1):
    ordenado = sorted(
        dist_passo.items(),
        key=lambda item: (-item[1], item[0]),
    )[:TOP_N]

    massa_modelada = sum(dist_passo.values())

    for ranking, (estado, prob) in enumerate(ordenado, start=1):
        linhas.append(
            {
                "passo": passo,
                "ranking": ranking,
                "estado": estado,
                "probabilidade": prob,
                "eh_pix": PADRAO_ALVO.lower() in estado.lower(),
                "massa_modelada_no_passo": massa_modelada,
                "massa_sem_modelo_anterior": massa_sem_modelo[passo - 1],
            }
        )

resultado_passos = pd.DataFrame(linhas)

display(
    spark.createDataFrame(resultado_passos)
    .orderBy("passo", "ranking")
)

# Probabilidade de PIX exatamente em cada passo.
resumo_pix = []

for passo, dist_passo in enumerate(distribuicoes, start=1):
    prob_pix = sum(
        prob
        for estado, prob in dist_passo.items()
        if PADRAO_ALVO.lower() in estado.lower()
    )

    resumo_pix.append(
        {
            "passo": passo,
            "prob_pix_exatamente_no_passo": prob_pix,
            "massa_modelada_no_passo": sum(dist_passo.values()),
        }
    )

display(
    spark.createDataFrame(pd.DataFrame(resumo_pix))
    .orderBy("passo")
)

# Probabilidade de atingir PIX pela primeira vez ate cada passo.
vivos = {estado_atual: 1.0}
prob_pix_acumulada = 0.0
linhas_primeira_passagem = []

for passo in range(1, N_PASSOS + 1):
    proxima_vivos = defaultdict(float)
    prob_pix_passo = 0.0
    massa_sem_modelo_passo = 0.0

    for origem, massa in vivos.items():
        idade = idade_dias if passo == 1 else 0.0
        probs = obter_probabilidades(origem, idade)

        if probs is None:
            massa_sem_modelo_passo += massa
            continue

        for destino, prob in probs.items():
            fluxo = massa * prob

            if PADRAO_ALVO.lower() in destino.lower():
                prob_pix_passo += fluxo
            else:
                proxima_vivos[destino] += fluxo

    prob_pix_acumulada += prob_pix_passo

    linhas_primeira_passagem.append(
        {
            "passo": passo,
            "prob_primeiro_pix_no_passo": prob_pix_passo,
            "prob_pix_ate_o_passo": prob_pix_acumulada,
            "massa_ainda_sem_pix": sum(proxima_vivos.values()),
            "massa_sem_modelo_no_passo": massa_sem_modelo_passo,
        }
    )

    vivos = dict(proxima_vivos)

display(
    spark.createDataFrame(pd.DataFrame(linhas_primeira_passagem))
    .orderBy("passo")
)

print(
    "Se PIX ganhar massa nos passos 2-5, o modelo aprendeu o caminho, "
    "mas o Top 5 imediato estava olhando curto demais."
)

print(
    "Se PIX continuar quase ausente, o problema esta no aprendizado, "
    "na amostra ou na falta de contexto, e nao apenas no horizonte."
)
