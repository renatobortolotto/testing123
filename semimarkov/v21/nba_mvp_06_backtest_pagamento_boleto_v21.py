# Databricks notebook source
# NBA | Parte 06: backtest formal da acao relevante "pagamento_boleto"
#
# Objetivo:
# - avaliar o modelo em CLIENTES DE VALIDACAO, que nao participaram do ajuste;
# - usar no maximo 1 snapshot por cliente e por idade testada;
# - medir a probabilidade de atingir pagamento_boleto em ate 1, 3 e 5 transicoes;
# - comparar Semi-Markov temporal vs baseline sem tempo usando os MESMOS modelos;
# - nao retreina e nao grava tabelas permanentes.
#
# Executar depois do pipeline V2.1 (02D), no mesmo notebook.
#
# Definicao inicial de acao relevante:
# estados contendo "pagamento_boleto" ou "pagamentos_boleto".
# Ajuste TARGET_PATTERNS se o catalogo usar outra nomenclatura.

# COMMAND ----------

from collections import defaultdict

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F


TARGET_PATTERNS = (
    "pagamento_boleto",
    "pagamentos_boleto",
)

IDADES_TESTE_SEG = (
    0.0,
    1800.0,
    86400.0,
)

HORIZONTES_PASSOS = (
    1,
    3,
    5,
)

N_PASSOS_MAX = max(HORIZONTES_PASSOS)

# Apenas para relatorio de cobertura da propagacao.
LIMITE_MASSA_PERDIDA = 0.05

# =========================
# VALIDACOES DE CONTEXTO
# =========================

_requeridos = [
    "SM_VIEWS",
    "SMD_BROADCAST",
    "smd_prever_destinos",
    "SMD_HORIZONTE",
]

_faltantes = [
    nome
    for nome in _requeridos
    if nome not in globals()
]

if _faltantes:
    raise RuntimeError(
        "Execute o pipeline V2.1 antes. "
        f"Ausentes: {_faltantes}"
    )

MODELOS_SMD = SMD_BROADCAST.value

if not MODELOS_SMD:
    raise ValueError("Nenhum modelo V2.1 em memoria.")


def eh_target_python(estado: str | None) -> bool:
    if estado is None:
        return False

    estado_lower = str(estado).lower()

    return any(
        padrao in estado_lower
        for padrao in TARGET_PATTERNS
    )


def target_spark(coluna):
    texto = F.lower(
        F.coalesce(
            coluna.cast("string"),
            F.lit(""),
        )
    )

    condicao = F.lit(False)

    for padrao in TARGET_PATTERNS:
        condicao = condicao | texto.contains(
            padrao.lower()
        )

    return condicao


# Conferir o que a regra esta chamando de pagamento_boleto.
estados_target = sorted(
    {
        estado
        for modelo in MODELOS_SMD.values()
        for estado in modelo["destinos"]
        if eh_target_python(estado)
    }
)

print(
    "Estados considerados pagamento_boleto:",
    estados_target,
)

if not estados_target:
    raise ValueError(
        "Nenhum estado do modelo casou com TARGET_PATTERNS."
    )


# COMMAND ----------

# ==============================================
# 1. CRIAR SNAPSHOTS DE HOLDOUT SEM OLHAR O FUTURO
# ==============================================
#
# A base de treino V2 ja contem validacao_cliente.
# O modelo foi ajustado com validacao_cliente=False.
# Aqui usamos somente validacao_cliente=True.
#
# Para cada idade:
# - o estado precisa ter sobrevivido ate essa idade;
# - exigimos os proximos N estados observados para formar um rotulo completo;
# - escolhemos 1 snapshot deterministico por cliente, independente do alvo.

base = spark.table(SM_VIEWS["treino"])

colunas_necessarias = {
    "cd_bv",
    "passo",
    "estado",
    "dur_min",
    "validacao_cliente",
    "elegivel_ajuste",
}

faltantes_base = (
    colunas_necessarias
    - set(base.columns)
)

if faltantes_base:
    raise ValueError(
        "Base de treino sem colunas necessarias: "
        f"{sorted(faltantes_base)}"
    )

janela_jornada = (
    Window.partitionBy("cd_bv")
    .orderBy("passo")
)

base_futuro = base

for k in range(
    1,
    N_PASSOS_MAX + 1,
):
    base_futuro = base_futuro.withColumn(
        f"estado_futuro_{k}",
        F.lead("estado", k).over(
            janela_jornada
        ),
    )

snapshots = []

for idade_seg in IDADES_TESTE_SEG:
    candidatos = (
        base_futuro
        .filter(
            F.col("validacao_cliente")
            & F.col("elegivel_ajuste")
            & F.col("estado").isNotNull()
            & (F.col("dur_min") >= F.lit(idade_seg))
            & ~target_spark(F.col("estado"))
        )
    )

    # Para rotular "nao ocorreu em 5 passos", precisamos realmente
    # observar os 5 estados futuros. Assim evitamos tratar censura como negativo.
    for k in range(
        1,
        N_PASSOS_MAX + 1,
    ):
        candidatos = candidatos.filter(
            F.col(
                f"estado_futuro_{k}"
            ).isNotNull()
        )

    # Snapshot pseudoaleatorio, mas deterministico, por cliente.
    janela_snapshot = (
        Window.partitionBy("cd_bv")
        .orderBy(
            F.xxhash64(
                "cd_bv",
                "passo",
                F.lit(
                    f"boleto_{idade_seg}"
                ),
            )
        )
    )

    candidatos = (
        candidatos
        .withColumn(
            "_rn_snapshot",
            F.row_number().over(
                janela_snapshot
            ),
        )
        .filter(
            F.col("_rn_snapshot") == 1
        )
        .drop("_rn_snapshot")
        .withColumn(
            "idade_teste_seg",
            F.lit(float(idade_seg)),
        )
    )

    # Primeiro passo real em que pagamento_boleto aparece.
    primeiro_hit = F.lit(None).cast(
        "int"
    )

    # Construimos de tras para frente para preservar o menor k.
    for k in range(
        N_PASSOS_MAX,
        0,
        -1,
    ):
        primeiro_hit = F.when(
            target_spark(
                F.col(
                    f"estado_futuro_{k}"
                )
            ),
            F.lit(k),
        ).otherwise(primeiro_hit)

    candidatos = candidatos.withColumn(
        "primeiro_passo_boleto_real",
        primeiro_hit,
    )

    snapshots.append(
        candidatos.select(
            "cd_bv",
            "passo",
            "estado",
            "idade_teste_seg",
            "primeiro_passo_boleto_real",
            *[
                f"estado_futuro_{k}"
                for k in range(
                    1,
                    N_PASSOS_MAX + 1,
                )
            ],
        )
    )

snapshots_df = snapshots[0]

for df in snapshots[1:]:
    snapshots_df = (
        snapshots_df.unionByName(df)
    )

snapshots_df = snapshots_df.localCheckpoint(
    eager=True
)

print("Snapshots de validacao:")

snapshots_df.groupBy(
    "idade_teste_seg"
).agg(
    F.count("*").alias("n_snapshots"),
    F.countDistinct("cd_bv").alias(
        "n_clientes"
    ),
    F.sum(
        F.col(
            "primeiro_passo_boleto_real"
        ).isNotNull().cast("int")
    ).alias(
        "n_boleto_ate_5_passos"
    ),
).orderBy(
    "idade_teste_seg"
).show(
    truncate=False
)


# COMMAND ----------

# ====================================
# 2. PROPAGAR O MODELO
# ====================================


def probs_estado(
    estado: str,
    idade_dias: float,
    temporal: bool,
):
    modelo = MODELOS_SMD.get(estado)

    if modelo is None:
        return None

    if temporal:
        try:
            q, _, _ = smd_prever_destinos(
                modelo,
                np.array(
                    [idade_dias],
                    dtype=float,
                ),
                SMD_HORIZONTE,
            )

            probs = q[0]

        except ValueError as exc:
            if (
                "IDADE_FORA_SUPORTE"
                in str(exc)
            ):
                return None
            raise

    else:
        probs = np.asarray(
            modelo["p_destino"],
            dtype=float,
        )

    return {
        destino: float(prob)
        for destino, prob in zip(
            modelo["destinos"],
            probs,
        )
        if prob > 0
    }


def primeira_passagem_boleto(
    estado_inicial: str,
    idade_dias: float,
    temporal: bool,
    n_passos: int,
):
    """P(atingir pagamento_boleto pela primeira vez em ate n_passos)."""

    vivos = {
        estado_inicial: 1.0
    }

    prob_hit = 0.0
    massa_perdida = 0.0
    hit_por_passo = []

    for passo in range(
        1,
        n_passos + 1,
    ):
        proximos_vivos = defaultdict(
            float
        )
        hit_passo = 0.0

        for origem, massa in vivos.items():
            # O tempo atual somente existe no primeiro estado.
            # Depois da transicao, a idade do novo estado e zero.
            idade = (
                idade_dias
                if passo == 1
                else 0.0
            )

            probs = probs_estado(
                origem,
                idade,
                temporal=temporal,
            )

            if probs is None:
                massa_perdida += massa
                continue

            for destino, prob in probs.items():
                fluxo = massa * prob

                if eh_target_python(
                    destino
                ):
                    hit_passo += fluxo
                else:
                    proximos_vivos[
                        destino
                    ] += fluxo

        prob_hit += hit_passo
        hit_por_passo.append(
            hit_passo
        )
        vivos = dict(
            proximos_vivos
        )

    return {
        "prob_hit": prob_hit,
        "massa_perdida": (
            massa_perdida
        ),
        "massa_ainda_sem_hit": sum(
            vivos.values()
        ),
        "hit_por_passo": (
            hit_por_passo
        ),
    }


# O conjunto tem no maximo poucos snapshots por cliente; a coleta
# e intencionalmente feita DEPOIS da reducao a 1 snapshot/cliente/idade.
snap_pd = snapshots_df.toPandas()

predicoes = []

for row in snap_pd.itertuples(
    index=False
):
    idade_dias = (
        float(row.idade_teste_seg)
        / 86400.0
    )

    for n_passos in HORIZONTES_PASSOS:
        sm = primeira_passagem_boleto(
            row.estado,
            idade_dias,
            temporal=True,
            n_passos=n_passos,
        )

        mk = primeira_passagem_boleto(
            row.estado,
            idade_dias,
            temporal=False,
            n_passos=n_passos,
        )

        primeiro_real = (
            row.primeiro_passo_boleto_real
        )

        y = int(
            pd.notna(primeiro_real)
            and int(primeiro_real)
            <= n_passos
        )

        predicoes.append(
            {
                "cd_bv": row.cd_bv,
                "estado": row.estado,
                "idade_teste_seg": (
                    row.idade_teste_seg
                ),
                "n_passos": n_passos,
                "y_boleto": y,
                "primeiro_passo_real": (
                    None
                    if pd.isna(
                        primeiro_real
                    )
                    else int(
                        primeiro_real
                    )
                ),
                "p_semimarkov": (
                    sm["prob_hit"]
                ),
                "p_markov": (
                    mk["prob_hit"]
                ),
                "massa_perdida_sm": (
                    sm["massa_perdida"]
                ),
                "massa_perdida_markov": (
                    mk["massa_perdida"]
                ),
            }
        )

pred_pd = pd.DataFrame(
    predicoes
)

if pred_pd.empty:
    raise ValueError(
        "Nenhuma predicao de validacao criada."
    )


# COMMAND ----------

# =========================
# 3. METRICAS
# =========================


def logloss_binario(
    y,
    p,
    eps=1e-12,
):
    p = np.clip(
        np.asarray(p, float),
        eps,
        1 - eps,
    )

    y = np.asarray(
        y,
        float,
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1 - y) * np.log(
                1 - p
            )
        )
    )


def brier(
    y,
    p,
):
    y = np.asarray(
        y,
        float,
    )

    p = np.asarray(
        p,
        float,
    )

    return float(
        np.mean(
            (p - y) ** 2
        )
    )


try:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    TEM_SKLEARN = True

except ImportError:
    TEM_SKLEARN = False


linhas_metricas = []

for (
    idade_seg,
    n_passos,
), grupo in pred_pd.groupby(
    [
        "idade_teste_seg",
        "n_passos",
    ],
    sort=True,
):
    y = grupo["y_boleto"].to_numpy(
        int
    )

    p_sm = grupo[
        "p_semimarkov"
    ].to_numpy(float)

    p_mk = grupo[
        "p_markov"
    ].to_numpy(float)

    linha = {
        "idade_teste_seg": (
            idade_seg
        ),
        "n_passos": n_passos,
        "n": len(grupo),
        "taxa_real_boleto": float(
            y.mean()
        ),
        "media_p_semimarkov": float(
            p_sm.mean()
        ),
        "media_p_markov": float(
            p_mk.mean()
        ),
        "brier_semimarkov": brier(
            y,
            p_sm,
        ),
        "brier_markov": brier(
            y,
            p_mk,
        ),
        # Positivo => Semi-Markov teve menor Brier.
        "ganho_brier_sm": (
            brier(y, p_mk)
            - brier(y, p_sm)
        ),
        "logloss_semimarkov": (
            logloss_binario(
                y,
                p_sm,
            )
        ),
        "logloss_markov": (
            logloss_binario(
                y,
                p_mk,
            )
        ),
        "pct_massa_perdida_sm_le_5pct": float(
            (
                grupo[
                    "massa_perdida_sm"
                ]
                <= LIMITE_MASSA_PERDIDA
            ).mean()
        ),
        "massa_perdida_sm_media": float(
            grupo[
                "massa_perdida_sm"
            ].mean()
        ),
    }

    if (
        TEM_SKLEARN
        and len(
            np.unique(y)
        ) == 2
    ):
        linha[
            "roc_auc_semimarkov"
        ] = float(
            roc_auc_score(
                y,
                p_sm,
            )
        )

        linha[
            "pr_auc_semimarkov"
        ] = float(
            average_precision_score(
                y,
                p_sm,
            )
        )

        linha[
            "roc_auc_markov"
        ] = float(
            roc_auc_score(
                y,
                p_mk,
            )
        )

        linha[
            "pr_auc_markov"
        ] = float(
            average_precision_score(
                y,
                p_mk,
            )
        )

    else:
        linha[
            "roc_auc_semimarkov"
        ] = np.nan
        linha[
            "pr_auc_semimarkov"
        ] = np.nan
        linha[
            "roc_auc_markov"
        ] = np.nan
        linha[
            "pr_auc_markov"
        ] = np.nan

    linhas_metricas.append(
        linha
    )

metricas_pd = pd.DataFrame(
    linhas_metricas
)

print(
    "METRICAS GERAIS - pagamento_boleto"
)

display(
    spark.createDataFrame(
        metricas_pd
    ).orderBy(
        "idade_teste_seg",
        "n_passos",
    )
)


# COMMAND ----------

# =================================
# 4. CALIBRACAO EM FAIXAS
# =================================
#
# A pergunta:
# quando o modelo fala ~30%, o evento acontece perto de 30%?
#
# Fazemos isso por idade e horizonte em passos.

calibracoes = []

for (
    idade_seg,
    n_passos,
), grupo in pred_pd.groupby(
    [
        "idade_teste_seg",
        "n_passos",
    ],
    sort=True,
):
    g = grupo.copy()

    if g["p_semimarkov"].nunique() < 2:
        continue

    n_bins = min(
        10,
        g["p_semimarkov"].nunique(),
    )

    g["faixa"] = pd.qcut(
        g["p_semimarkov"],
        q=n_bins,
        duplicates="drop",
    )

    resumo = (
        g.groupby(
            "faixa",
            observed=True,
        )
        .agg(
            n=("y_boleto", "size"),
            prob_media=(
                "p_semimarkov",
                "mean",
            ),
            taxa_real=(
                "y_boleto",
                "mean",
            ),
            massa_perdida_media=(
                "massa_perdida_sm",
                "mean",
            ),
        )
        .reset_index()
    )

    resumo[
        "idade_teste_seg"
    ] = idade_seg

    resumo[
        "n_passos"
    ] = n_passos

    resumo[
        "faixa"
    ] = resumo["faixa"].astype(
        str
    )

    calibracoes.append(
        resumo
    )

if calibracoes:
    calibracao_pd = pd.concat(
        calibracoes,
        ignore_index=True,
    )

    print(
        "CALIBRACAO - probabilidade prevista vs taxa observada"
    )

    display(
        spark.createDataFrame(
            calibracao_pd
        ).orderBy(
            "idade_teste_seg",
            "n_passos",
            "prob_media",
        )
    )

else:
    print(
        "Sem variacao suficiente de probabilidade para criar faixas."
    )


# COMMAND ----------

# =========================================
# 5. AMOSTRA DOS CASOS PARA INSPECAO
# =========================================
#
# Sem necessidade de expor IDs para a analise agregada.
# Mostra apenas estado, idade, verdade e probabilidades.

print(
    "AMOSTRA DE CASOS SEM IDENTIFICADOR"
)

display(
    spark.createDataFrame(
        pred_pd[
            [
                "estado",
                "idade_teste_seg",
                "n_passos",
                "y_boleto",
                "primeiro_passo_real",
                "p_semimarkov",
                "p_markov",
                "massa_perdida_sm",
            ]
        ].head(50)
    )
)


# COMMAND ----------

print(
    "Leitura principal:"
)

print(
    "- ganho_brier_sm > 0 favorece o Semi-Markov temporal."
)

print(
    "- compare p_semimarkov com p_markov nas idades 1800s e 86400s;"
)

print(
    "  em idade 0, eles devem ser iguais ou praticamente iguais."
)

print(
    "- a calibracao compara probabilidade prevista de pagamento_boleto "
    "com a frequencia realmente observada no holdout."
)

print(
    "- massa_perdida alta indica que parte relevante dos caminhos entra "
    "em estados sem modelo; interpretar a probabilidade com cautela."
)
