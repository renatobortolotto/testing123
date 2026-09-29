import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from pyspark.sql import functions as F
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest


# 1. Preparação: mantenha convertidos E não convertidos.
base = (
    df.select(
        F.date_trunc("week", F.col("dtinclusao"))
        .cast("date")
        .alias("semana"),
        "cpf",
        F.col("TesteDigito").cast("int").alias("testedigito"),
        F.col("converteu").cast("int").alias("converteu"),
    )
    .filter(
        F.col("semana").isNotNull()
        & F.col("cpf").isNotNull()
        & F.col("testedigito").isin(0, 1)
        & F.col("converteu").isin(0, 1)
    )
)

# Impede que o mesmo CPF participe dos dois grupos na mesma semana.
conflitos = (
    base.groupBy("semana", "cpf")
    .agg(F.countDistinct("testedigito").alias("n_grupos"))
    .filter(F.col("n_grupos") > 1)
)

if conflitos.limit(1).count():
    raise ValueError(
        "Existem CPFs nos dois grupos na mesma semana. "
        "Revise a atribuição de teste e controle antes da comparação."
    )


# 2. Contagem de CPFs únicos por semana e grupo.
resumo_semanal = (
    base.groupBy("semana", "testedigito")
    .agg(
        F.countDistinct("cpf").alias("total"),
        F.countDistinct(
            F.when(F.col("converteu") == 1, F.col("cpf"))
        ).alias("aprovados"),
    )
    .orderBy("semana", "testedigito")
)

# Apenas o resultado agregado é transferido para pandas.
pdf = resumo_semanal.toPandas()

if pdf.empty:
    raise ValueError("Não há registros válidos para gerar o gráfico.")

metricas = ["total", "aprovados"]

resultado = (
    pdf.pivot(
        index="semana",
        columns="testedigito",
        values=metricas,
    )
    .reindex(
        columns=pd.MultiIndex.from_product([metricas, [0, 1]])
    )
    .fillna(0)
    .astype("int64")
    .sort_index()
)

resultado.columns = [
    f"{metrica}_{'teste' if grupo == 0 else 'controle'}"
    for metrica, grupo in resultado.columns
]

resultado = resultado.reset_index()
resultado["semana"] = pd.to_datetime(resultado["semana"])


# 3. Testes de Fisher e Z, ambos bilaterais, por semana.
def calcular_testes(linha: pd.Series) -> pd.Series:
    """Compara as proporções de aprovação entre teste e controle."""
    # Ordem dos grupos: teste primeiro, controle depois.
    aprovados = np.array(
        [linha["aprovados_teste"], linha["aprovados_controle"]],
        dtype=np.int64,
    )

    totais = np.array(
        [linha["total_teste"], linha["total_controle"]],
        dtype=np.int64,
    )

    testes = {
        "p_valor_fisher": np.nan,
        "p_valor_z": np.nan,
        "estatistica_z": np.nan,
    }

    # Sem clientes em algum grupo, não há comparação.
    if np.any(totais == 0):
        return pd.Series(testes)

    # Fisher: linhas = grupos; colunas = aprovados e não aprovados.
    tabela = np.column_stack(
        (aprovados, totais - aprovados)
    )

    testes["p_valor_fisher"] = float(
        fisher_exact(tabela, alternative="two-sided")[1]
    )

    # Z: evita variância zero quando todos, nos dois grupos,
    # foram aprovados ou quando ninguém foi aprovado.
    if 0 < aprovados.sum() < totais.sum():
        estatistica_z, p_valor_z = proportions_ztest(
            count=aprovados,
            nobs=totais,
            value=0,
            alternative="two-sided",
            prop_var=False,
        )

        testes["p_valor_z"] = float(p_valor_z)
        testes["estatistica_z"] = float(estatistica_z)

    return pd.Series(testes)


colunas_testes = [
    "p_valor_fisher",
    "p_valor_z",
    "estatistica_z",
]

resultado[colunas_testes] = resultado.apply(
    calcular_testes,
    axis=1,
)

# Taxas e diferença em pontos percentuais.
for grupo in ["teste", "controle"]:
    resultado[f"taxa_{grupo}_pct"] = (
        100
        * resultado[f"aprovados_{grupo}"]
        / resultado[f"total_{grupo}"].replace(0, np.nan)
    )

resultado["diferenca_pp"] = (
    resultado["taxa_teste_pct"] - resultado["taxa_controle_pct"]
)

display(
    resultado.round(
        {
            "taxa_teste_pct": 2,
            "taxa_controle_pct": 2,
            "diferenca_pp": 2,
            "estatistica_z": 3,
        }
    )
)


# 4. Gráfico: aprovados em barras e os dois p-valores acima.
def formatar_p_valor(valor: float) -> str:
    """Formata o p-valor sem arredondar valores pequenos para 0,000."""
    return "N/D" if pd.isna(valor) else f"{valor:.3g}"


x = np.arange(len(resultado))
largura = 0.36

fig, ax = plt.subplots(
    figsize=(max(12, len(resultado) * 1.8), 7)
)

barras_teste = ax.bar(
    x - largura / 2,
    resultado["aprovados_teste"],
    width=largura,
    label="Teste — TesteDigito = 0",
)

barras_controle = ax.bar(
    x + largura / 2,
    resultado["aprovados_controle"],
    width=largura,
    label="Controle — TesteDigito = 1",
)

# Quantidade de aprovados sobre cada barra.
ax.bar_label(barras_teste, fmt="%.0f", padding=3)
ax.bar_label(barras_controle, fmt="%.0f", padding=3)

maior_barra = max(
    1,
    resultado[["aprovados_teste", "aprovados_controle"]]
    .to_numpy()
    .max(),
)

# P-valores centralizados acima de cada comparação semanal.
for i, (_, linha) in enumerate(resultado.iterrows()):
    altura = max(
        linha["aprovados_teste"],
        linha["aprovados_controle"],
    )

    p_fisher = formatar_p_valor(linha["p_valor_fisher"])
    p_z = formatar_p_valor(linha["p_valor_z"])

    texto = (
        f"Fisher: p = {p_fisher}\n"
        f"Z: p = {p_z}"
    )

    ax.text(
        x[i],
        altura + maior_barra * 0.10,
        texto,
        ha="center",
        va="bottom",
        fontsize=10,
        linespacing=1.4,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    resultado["semana"].dt.strftime("%d/%m/%Y"),
    rotation=45,
    ha="right",
)

ax.set_xlabel("Semana de inclusão — início na segunda-feira")
ax.set_ylabel("CPFs aprovados (converteu = 1)")
ax.set_title(
    "Aprovados por semana — Teste x Controle\n"
    "p-valores: Fisher e Z para duas proporções, ambos bilaterais",
    pad=45,
)

# Espaço adicional para as duas linhas de p-valores.
ax.set_ylim(0, maior_barra * 1.50)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)

ax.legend(
    loc="lower left",
    bbox_to_anchor=(0, 1.02),
    ncol=2,
    frameon=False,
)

# Exibe a figura explicitamente no Databricks.
fig.tight_layout()
display(fig)
plt.close(fig)