# Databricks notebook source
# NBA | Parte 05 V2: diagnostico multi-step alinhado ao pipeline V2.1
#
# Executar DEPOIS da Parte 04, no mesmo notebook.
# Nao retreina e nao altera modelos.
#
# Usa exatamente:
# - atuais_scoring: criado na Parte 04
# - SMD_BROADCAST.value: dicionario de modelos V2.1
# - smd_prever_destinos: nucleo de inferencia V2.1
# - SMD_HORIZONTE: horizonte configurado no pipeline
#
# Objetivo:
# verificar se comportamentos como PIX aparecem depois de estados
# intermediarios (login, navegacao etc.) em 2, 3, 4 ou 5 transicoes.

# COMMAND ----------

from collections import defaultdict

import numpy as np
import pandas as pd
from pyspark.sql import functions as F


# =========================
# CONFIGURACAO
# =========================

CD_BV_ANALISE = str(CD_BV_TESTE)
N_PASSOS = 5
PADRAO_ALVO = "pix"
TOP_N_POR_PASSO = 15


# =========================
# VALIDACOES
# =========================

_requeridos = [
    "atuais_scoring",
    "SMD_BROADCAST",
    "smd_prever_destinos",
    "SMD_HORIZONTE",
    "CD_BV_TESTE",
]

_faltantes = [
    nome for nome in _requeridos
    if nome not in globals()
]

if _faltantes:
    raise RuntimeError(
        "Execute a Parte 04 antes. "
        f"Variaveis ausentes: {_faltantes}"
    )

MODELOS_SMD = SMD_BROADCAST.value

if not MODELOS_SMD:
    raise ValueError(
        "SMD_BROADCAST nao contem modelos."
    )


# =========================
# FUNCOES
# =========================

def probabilidades_proximo_estado(
    estado: str,
    idade_dias: float,
) -> dict[str, float] | None:
    """Retorna P(proximo destino | estado, idade)."""

    modelo = MODELOS_SMD.get(estado)

    if modelo is None:
        return None

    try:
        q, _, _ = smd_prever_destinos(
            modelo,
            np.array([idade_dias], dtype=float),
            SMD_HORIZONTE,
        )
    except ValueError as exc:
        if "IDADE_FORA_SUPORTE" in str(exc):
            return None
        raise

    return {
        destino: float(prob)
        for destino, prob in zip(
            modelo["destinos"],
            q[0],
        )
        if prob > 0
    }


# =========================
# CLIENTE
# =========================

cliente = (
    atuais_scoring
    .filter(F.col("cd_bv") == CD_BV_ANALISE)
    .select(
        "cd_bv",
        "acao_atual",
        "tempo_no_estado_seg",
        "status_input",
        "ts_inicio",
        "ts_ultima_atividade",
    )
    .first()
)

if cliente is None:
    raise ValueError(
        f"Cliente {CD_BV_ANALISE} nao encontrado em atuais_scoring."
    )

if cliente["status_input"] != "OK":
    raise ValueError(
        "Cliente nao apto ao scoring: "
        f"{cliente['status_input']}"
    )

estado_atual = str(cliente["acao_atual"])
idade_atual_dias = (
    float(cliente["tempo_no_estado_seg"])
    / 86400.0
)

print("Cliente:", CD_BV_ANALISE)
print("Estado atual:", estado_atual)
print(
    "Tempo no estado (dias):",
    idade_atual_dias,
)
print(
    "Existe modelo para estado atual:",
    estado_atual in MODELOS_SMD,
)


# COMMAND ----------

# =====================================================
# 1. DISTRIBUICAO DO ESTADO EXATO EM CADA TRANSICAO
# =====================================================
#
# Passo 1:
#   usa a idade REAL do cliente no estado atual.
#
# Passos 2+:
#   idade = 0, pois a cadeia acabou de entrar naquele estado.
#
# Isto responde:
# P(X_n = estado | estado/idade atuais)

distribuicoes = []

primeiro = probabilidades_proximo_estado(
    estado_atual,
    idade_atual_dias,
)

if primeiro is None:
    raise ValueError(
        "Nao foi possivel calcular o primeiro passo "
        "para o estado atual."
    )

dist_atual = primeiro
distribuicoes.append(dist_atual)

perda_sem_modelo_por_passo = [0.0]

for passo in range(
    2,
    N_PASSOS + 1,
):
    proxima = defaultdict(float)
    perda_sem_modelo = 0.0

    for origem, massa_origem in dist_atual.items():
        probs = probabilidades_proximo_estado(
            origem,
            idade_dias=0.0,
        )

        if probs is None:
            perda_sem_modelo += massa_origem
            continue

        for destino, prob in probs.items():
            proxima[destino] += (
                massa_origem * prob
            )

    dist_atual = dict(proxima)
    distribuicoes.append(dist_atual)
    perda_sem_modelo_por_passo.append(
        perda_sem_modelo
    )


# COMMAND ----------

# =========================
# 2. TOP ESTADOS POR PASSO
# =========================

linhas_top = []

for passo, distribuicao in enumerate(
    distribuicoes,
    start=1,
):
    ordenado = sorted(
        distribuicao.items(),
        key=lambda x: (-x[1], x[0]),
    )

    massa_modelada = sum(
        distribuicao.values()
    )

    for ranking, (
        estado,
        probabilidade,
    ) in enumerate(
        ordenado[:TOP_N_POR_PASSO],
        start=1,
    ):
        linhas_top.append(
            {
                "passo": passo,
                "ranking": ranking,
                "estado": estado,
                "probabilidade": (
                    probabilidade
                ),
                "eh_pix": (
                    PADRAO_ALVO.lower()
                    in estado.lower()
                ),
                "massa_modelada": (
                    massa_modelada
                ),
                "massa_perdida_sem_modelo": (
                    perda_sem_modelo_por_passo[
                        passo - 1
                    ]
                ),
            }
        )

top_passos_df = pd.DataFrame(
    linhas_top
)

print(
    "Top estados em cada numero de transicoes:"
)

display(
    spark.createDataFrame(
        top_passos_df
    ).orderBy(
        "passo",
        "ranking",
    )
)


# COMMAND ----------

# =============================================
# 3. PROBABILIDADE DE PIX EXATAMENTE NO PASSO
# =============================================

linhas_pix_passo = []

for passo, distribuicao in enumerate(
    distribuicoes,
    start=1,
):
    prob_pix = sum(
        prob
        for estado, prob
        in distribuicao.items()
        if (
            PADRAO_ALVO.lower()
            in estado.lower()
        )
    )

    linhas_pix_passo.append(
        {
            "passo": passo,
            "prob_pix_exatamente_no_passo": (
                prob_pix
            ),
            "massa_modelada_no_passo": sum(
                distribuicao.values()
            ),
            "massa_perdida_sem_modelo": (
                perda_sem_modelo_por_passo[
                    passo - 1
                ]
            ),
        }
    )

pix_passo_df = pd.DataFrame(
    linhas_pix_passo
)

print(
    "Probabilidade de estar em um estado PIX "
    "exatamente em cada passo:"
)

display(
    spark.createDataFrame(
        pix_passo_df
    ).orderBy("passo")
)


# COMMAND ----------

# =================================================
# 4. PRIMEIRA PASSAGEM POR PIX ATE CADA TRANSICAO
# =================================================
#
# Aqui removemos caminhos assim que encontram PIX.
# Portanto:
#
# prob_pix_ate_o_passo
#
# representa a probabilidade acumulada de atingir
# algum estado cujo nome contenha "pix" pela primeira
# vez ate aquele numero de transicoes.

vivos = {
    estado_atual: 1.0
}

prob_pix_acumulada = 0.0
linhas_primeira_passagem = []

for passo in range(
    1,
    N_PASSOS + 1,
):
    proxima_vivos = defaultdict(float)
    prob_pix_passo = 0.0
    massa_sem_modelo = 0.0

    for origem, massa_origem in vivos.items():
        idade = (
            idade_atual_dias
            if passo == 1
            else 0.0
        )

        probs = probabilidades_proximo_estado(
            origem,
            idade,
        )

        if probs is None:
            massa_sem_modelo += (
                massa_origem
            )
            continue

        for destino, prob in probs.items():
            fluxo = massa_origem * prob

            if (
                PADRAO_ALVO.lower()
                in destino.lower()
            ):
                prob_pix_passo += fluxo
            else:
                proxima_vivos[
                    destino
                ] += fluxo

    prob_pix_acumulada += (
        prob_pix_passo
    )

    linhas_primeira_passagem.append(
        {
            "passo": passo,
            "prob_primeiro_pix_no_passo": (
                prob_pix_passo
            ),
            "prob_pix_ate_o_passo": (
                prob_pix_acumulada
            ),
            "massa_ainda_sem_pix": sum(
                proxima_vivos.values()
            ),
            "massa_sem_modelo_no_passo": (
                massa_sem_modelo
            ),
        }
    )

    vivos = dict(proxima_vivos)

primeira_passagem_df = pd.DataFrame(
    linhas_primeira_passagem
)

print(
    "Probabilidade acumulada de chegar "
    "a PIX pela primeira vez:"
)

display(
    spark.createDataFrame(
        primeira_passagem_df
    ).orderBy("passo")
)


# COMMAND ----------

# =========================
# 5. LEITURA RAPIDA
# =========================

ultimo = primeira_passagem_df.iloc[-1]

print(
    f"P(chegar a PIX em ate {N_PASSOS} transicoes): "
    f"{ultimo['prob_pix_ate_o_passo']:.4f}"
)

print(
    "Se PIX crescer principalmente nos passos 2-5, "
    "o modelo aprendeu o caminho indireto e o Top 5 "
    "imediato estava apenas olhando um passo."
)

print(
    "Se PIX continuar com probabilidade muito baixa, "
    "a proxima investigacao deve ser treinamento, "
    "cobertura dos estados e contexto/sazonalidade."
)
