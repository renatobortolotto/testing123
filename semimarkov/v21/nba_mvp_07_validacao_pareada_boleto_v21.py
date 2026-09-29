# Databricks notebook source
# NBA | Parte 07: validacao pareada Markov vs Semi-Markov
# Alvo: pagamento_boleto
#
# Executar DEPOIS da Parte 06, no mesmo notebook.
#
# Objetivos:
# 1) comparar Brier e Log Loss nas MESMAS observacoes;
# 2) bootstrap pareado por cliente para IC das diferencas;
# 3) verificar se o ganho vem apenas dos negativos;
# 4) avaliar calibracao;
# 5) medir sensibilidade a perda de massa na propagacao.
#
# Nao retreina modelo.
# Nao altera probabilidades.
# Nao grava tabelas permanentes.

# COMMAND ----------

import numpy as np
import pandas as pd


SEMENTE_BOOTSTRAP = 20260929
N_BOOTSTRAP = 2000
EPS = 1e-12

LIMITES_MASSA = (
    ("TODOS", 1.0),
    ("PERDA_ATE_5PCT", 0.05),
    ("PERDA_ATE_1PCT", 0.01),
)

_requeridos = [
    "pred_pd",
]

_faltantes = [
    nome
    for nome in _requeridos
    if nome not in globals()
]

if _faltantes:
    raise RuntimeError(
        "Execute a Parte 06 antes. "
        f"Ausentes: {_faltantes}"
    )

campos = {
    "cd_bv",
    "idade_teste_seg",
    "n_passos",
    "y_boleto",
    "p_semimarkov",
    "p_markov",
    "massa_perdida_sm",
}

faltantes_campos = campos - set(pred_pd.columns)

if faltantes_campos:
    raise ValueError(
        "pred_pd sem colunas esperadas: "
        f"{sorted(faltantes_campos)}"
    )

dados = pred_pd.copy()

for coluna in (
    "p_semimarkov",
    "p_markov",
    "massa_perdida_sm",
):
    dados[coluna] = pd.to_numeric(
        dados[coluna],
        errors="coerce",
    )

dados["y_boleto"] = pd.to_numeric(
    dados["y_boleto"],
    errors="coerce",
).astype("Int64")

dados = dados.dropna(
    subset=[
        "cd_bv",
        "idade_teste_seg",
        "n_passos",
        "y_boleto",
        "p_semimarkov",
        "p_markov",
        "massa_perdida_sm",
    ]
).copy()

if dados.empty:
    raise ValueError(
        "Nao ha observacoes validas em pred_pd."
    )

for coluna in (
    "p_semimarkov",
    "p_markov",
):
    if (
        (dados[coluna] < 0).any()
        or (dados[coluna] > 1).any()
    ):
        raise ValueError(
            f"{coluna} fora de [0, 1]."
        )

# COMMAND ----------

# =========================
# FUNCOES DE PERDA
# =========================

def brier_linha(
    y: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(p, float)
        - np.asarray(y, float)
    ) ** 2


def logloss_linha(
    y: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y, float)

    p = np.clip(
        np.asarray(p, float),
        EPS,
        1.0 - EPS,
    )

    return -(
        y * np.log(p)
        + (1.0 - y) * np.log(
            1.0 - p
        )
    )


dados["brier_sm_linha"] = brier_linha(
    dados["y_boleto"],
    dados["p_semimarkov"],
)

dados["brier_mk_linha"] = brier_linha(
    dados["y_boleto"],
    dados["p_markov"],
)

dados["logloss_sm_linha"] = logloss_linha(
    dados["y_boleto"],
    dados["p_semimarkov"],
)

dados["logloss_mk_linha"] = logloss_linha(
    dados["y_boleto"],
    dados["p_markov"],
)

# Positivo => Semi-Markov melhor.
dados["ganho_brier_linha"] = (
    dados["brier_mk_linha"]
    - dados["brier_sm_linha"]
)

dados["ganho_logloss_linha"] = (
    dados["logloss_mk_linha"]
    - dados["logloss_sm_linha"]
)

dados["delta_prob_sm_mk"] = (
    dados["p_semimarkov"]
    - dados["p_markov"]
)


# COMMAND ----------

# ==============================================
# 1. RESUMO PAREADO - MESMAS OBSERVACOES
# ==============================================

resumos = []

for (
    idade,
    n_passos,
), grupo in dados.groupby(
    [
        "idade_teste_seg",
        "n_passos",
    ],
    sort=True,
):
    y = grupo[
        "y_boleto"
    ].to_numpy(int)

    positivos = grupo[
        grupo["y_boleto"] == 1
    ]

    negativos = grupo[
        grupo["y_boleto"] == 0
    ]

    resumos.append(
        {
            "idade_teste_seg": float(
                idade
            ),
            "n_passos": int(
                n_passos
            ),
            "n": int(
                len(grupo)
            ),
            "n_positivos": int(
                y.sum()
            ),
            "taxa_real": float(
                y.mean()
            ),
            "p_media_sm": float(
                grupo[
                    "p_semimarkov"
                ].mean()
            ),
            "p_media_markov": float(
                grupo[
                    "p_markov"
                ].mean()
            ),
            "brier_sm": float(
                grupo[
                    "brier_sm_linha"
                ].mean()
            ),
            "brier_markov": float(
                grupo[
                    "brier_mk_linha"
                ].mean()
            ),
            "ganho_brier_sm": float(
                grupo[
                    "ganho_brier_linha"
                ].mean()
            ),
            "logloss_sm": float(
                grupo[
                    "logloss_sm_linha"
                ].mean()
            ),
            "logloss_markov": float(
                grupo[
                    "logloss_mk_linha"
                ].mean()
            ),
            "ganho_logloss_sm": float(
                grupo[
                    "ganho_logloss_linha"
                ].mean()
            ),
            "delta_prob_media_positivos": (
                float(
                    positivos[
                        "delta_prob_sm_mk"
                    ].mean()
                )
                if len(positivos)
                else np.nan
            ),
            "delta_prob_media_negativos": (
                float(
                    negativos[
                        "delta_prob_sm_mk"
                    ].mean()
                )
                if len(negativos)
                else np.nan
            ),
            "massa_perdida_media": float(
                grupo[
                    "massa_perdida_sm"
                ].mean()
            ),
        }
    )

resumo_pareado_pd = pd.DataFrame(
    resumos
)

print(
    "1. COMPARACAO PAREADA - Markov vs Semi-Markov"
)

display(
    spark.createDataFrame(
        resumo_pareado_pd
    ).orderBy(
        "idade_teste_seg",
        "n_passos",
    )
)


# COMMAND ----------

# =====================================
# 2. BOOTSTRAP PAREADO POR CLIENTE
# =====================================
#
# Cada estrato da Parte 06 possui no maximo 1 snapshot por cliente.
# Mesmo assim, reamostramos explicitamente cd_bv para deixar a
# unidade de incerteza clara.
#
# IC > 0 para o ganho => favorece Semi-Markov.
# IC cruzando 0 => evidencia insuficiente naquele estrato.

rng = np.random.default_rng(
    SEMENTE_BOOTSTRAP
)

bootstrap_rows = []

for (
    idade,
    n_passos,
), grupo in dados.groupby(
    [
        "idade_teste_seg",
        "n_passos",
    ],
    sort=True,
):
    clientes = (
        grupo["cd_bv"]
        .astype(str)
        .unique()
    )

    if len(clientes) < 2:
        continue

    ganho_brier_boot = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    ganho_logloss_boot = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    grupo_indexado = grupo.set_index(
        "cd_bv",
        drop=False,
    )

    for b in range(
        N_BOOTSTRAP
    ):
        amostra_clientes = rng.choice(
            clientes,
            size=len(clientes),
            replace=True,
        )

        amostra = grupo_indexado.loc[
            amostra_clientes
        ]

        ganho_brier_boot[b] = float(
            amostra[
                "ganho_brier_linha"
            ].mean()
        )

        ganho_logloss_boot[b] = float(
            amostra[
                "ganho_logloss_linha"
            ].mean()
        )

    bootstrap_rows.append(
        {
            "idade_teste_seg": float(
                idade
            ),
            "n_passos": int(
                n_passos
            ),
            "n_clientes": int(
                len(clientes)
            ),
            "n_positivos": int(
                grupo[
                    "y_boleto"
                ].sum()
            ),
            "ganho_brier_medio": float(
                grupo[
                    "ganho_brier_linha"
                ].mean()
            ),
            "ganho_brier_ic95_lo": float(
                np.quantile(
                    ganho_brier_boot,
                    0.025,
                )
            ),
            "ganho_brier_ic95_hi": float(
                np.quantile(
                    ganho_brier_boot,
                    0.975,
                )
            ),
            "prob_bootstrap_brier_sm_melhor": float(
                (
                    ganho_brier_boot > 0
                ).mean()
            ),
            "ganho_logloss_medio": float(
                grupo[
                    "ganho_logloss_linha"
                ].mean()
            ),
            "ganho_logloss_ic95_lo": float(
                np.quantile(
                    ganho_logloss_boot,
                    0.025,
                )
            ),
            "ganho_logloss_ic95_hi": float(
                np.quantile(
                    ganho_logloss_boot,
                    0.975,
                )
            ),
            "prob_bootstrap_logloss_sm_melhor": float(
                (
                    ganho_logloss_boot > 0
                ).mean()
            ),
        }
    )

bootstrap_pd = pd.DataFrame(
    bootstrap_rows
)

print(
    "2. BOOTSTRAP PAREADO - IC 95%"
)

display(
    spark.createDataFrame(
        bootstrap_pd
    ).orderBy(
        "idade_teste_seg",
        "n_passos",
    )
)


# COMMAND ----------

# =====================================
# 3. SENSIBILIDADE A MASSA PERDIDA
# =====================================

sensibilidade_rows = []

for (
    idade,
    n_passos,
), grupo_base in dados.groupby(
    [
        "idade_teste_seg",
        "n_passos",
    ],
    sort=True,
):
    for nome_filtro, limite in LIMITES_MASSA:
        if nome_filtro == "TODOS":
            grupo = grupo_base
        else:
            grupo = grupo_base[
                grupo_base[
                    "massa_perdida_sm"
                ]
                <= limite
            ]

        if grupo.empty:
            continue

        y = grupo[
            "y_boleto"
        ].to_numpy(int)

        sensibilidade_rows.append(
            {
                "idade_teste_seg": float(
                    idade
                ),
                "n_passos": int(
                    n_passos
                ),
                "filtro_massa": (
                    nome_filtro
                ),
                "n": int(
                    len(grupo)
                ),
                "n_positivos": int(
                    y.sum()
                ),
                "taxa_real": float(
                    y.mean()
                ),
                "massa_perdida_media": float(
                    grupo[
                        "massa_perdida_sm"
                    ].mean()
                ),
                "brier_sm": float(
                    grupo[
                        "brier_sm_linha"
                    ].mean()
                ),
                "brier_markov": float(
                    grupo[
                        "brier_mk_linha"
                    ].mean()
                ),
                "ganho_brier_sm": float(
                    grupo[
                        "ganho_brier_linha"
                    ].mean()
                ),
                "logloss_sm": float(
                    grupo[
                        "logloss_sm_linha"
                    ].mean()
                ),
                "logloss_markov": float(
                    grupo[
                        "logloss_mk_linha"
                    ].mean()
                ),
                "ganho_logloss_sm": float(
                    grupo[
                        "ganho_logloss_linha"
                    ].mean()
                ),
            }
        )

sensibilidade_pd = pd.DataFrame(
    sensibilidade_rows
)

print(
    "3. SENSIBILIDADE A PERDA DE MASSA"
)

display(
    spark.createDataFrame(
        sensibilidade_pd
    ).orderBy(
        "idade_teste_seg",
        "n_passos",
        "filtro_massa",
    )
)


# COMMAND ----------

# =====================================
# 4. CALIBRACAO - SEMI-MARKOV
# =====================================
#
# Como boleto e raro, usamos 5 bins no maximo.
# Nao interpretar bins sem positivos como prova de boa calibracao.

calibracao_rows = []

for (
    idade,
    n_passos,
), grupo in dados.groupby(
    [
        "idade_teste_seg",
        "n_passos",
    ],
    sort=True,
):
    if grupo[
        "p_semimarkov"
    ].nunique() < 2:
        continue

    n_bins = min(
        5,
        int(
            grupo[
                "p_semimarkov"
            ].nunique()
        ),
    )

    trabalho = grupo.copy()

    trabalho["faixa"] = pd.qcut(
        trabalho[
            "p_semimarkov"
        ],
        q=n_bins,
        duplicates="drop",
    )

    tabela = (
        trabalho
        .groupby(
            "faixa",
            observed=True,
        )
        .agg(
            n=(
                "y_boleto",
                "size",
            ),
            n_positivos=(
                "y_boleto",
                "sum",
            ),
            prob_media_sm=(
                "p_semimarkov",
                "mean",
            ),
            prob_media_markov=(
                "p_markov",
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

    tabela["faixa"] = tabela[
        "faixa"
    ].astype(str)

    tabela[
        "idade_teste_seg"
    ] = float(idade)

    tabela[
        "n_passos"
    ] = int(n_passos)

    calibracao_rows.append(
        tabela
    )

if calibracao_rows:
    calibracao_p07_pd = pd.concat(
        calibracao_rows,
        ignore_index=True,
    )

    print(
        "4. CALIBRACAO POR FAIXA"
    )

    display(
        spark.createDataFrame(
            calibracao_p07_pd
        ).orderBy(
            "idade_teste_seg",
            "n_passos",
            "prob_media_sm",
        )
    )

else:
    calibracao_p07_pd = pd.DataFrame()

    print(
        "Sem variacao suficiente para calibracao por faixas."
    )


# COMMAND ----------

# =====================================
# 5. FOCO NO CENARIO MAIS RELEVANTE
# =====================================
#
# Mostra 30 min, 3 e 5 passos.
# Aqui o componente temporal pode diferir do Markov.

foco = resumo_pareado_pd[
    (
        resumo_pareado_pd[
            "idade_teste_seg"
        ]
        == 1800.0
    )
    & (
        resumo_pareado_pd[
            "n_passos"
        ].isin(
            [3, 5]
        )
    )
]

print(
    "5. FOCO EXECUTIVO - 30 min / 3 e 5 passos"
)

display(
    spark.createDataFrame(
        foco
    ).orderBy(
        "n_passos"
    )
)

print(
    "INTERPRETACAO:"
)

print(
    "- ganho_brier_sm > 0: Semi-Markov teve Brier menor."
)

print(
    "- ganho_logloss_sm > 0: Semi-Markov teve Log Loss menor."
)

print(
    "- IC bootstrap inteiro > 0: evidencia mais consistente a favor do Semi-Markov."
)

print(
    "- delta_prob_media_positivos > 0 seria desejavel nos positivos;"
)

print(
    "  delta_prob_media_negativos < 0 seria desejavel nos negativos."
)

print(
    "- Se o Semi-Markov melhora apenas porque reduz probabilidades dos negativos,"
)

print(
    "  mas tambem reduz demais os positivos, precisamos revisar calibracao/objetivo."
)

print(
    "- Se o resultado muda muito ao filtrar massa perdida, a cobertura multi-step"
)

print(
    "  ainda e uma limitacao importante."
)
