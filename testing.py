import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter
from pyspark.sql import functions as F
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import proportions_ztest


# 1. Configuração. df deve conter apenas a população elegível ao A/B.
ANO = 2026
ALFA = 0.05  # Use 0.15 ou 0.20 para mudar apenas o corte de significância.

inicio = f"{ANO}-08-01"
fim_exclusivo = f"{ANO}-10-01"
meses = pd.date_range(inicio, periods=2, freq="MS", name="mes")

# 2. Agosto e setembro completos, pelo mês de inclusão.
base = (
    df.select(
        F.date_trunc("month", F.col("dtinclusao"))
        .cast("date")
        .alias("mes"),
        "cpf",
        F.col("TesteDigito").cast("int").alias("testedigito"),
        F.col("converteu").cast("int").alias("converteu"),
    )
    .filter(
        (F.col("mes") >= F.lit(inicio).cast("date"))
        & (F.col("mes") < F.lit(fim_exclusivo).cast("date"))
        & F.col("cpf").isNotNull()
        & F.col("testedigito").isin(0, 1)
        & F.col("converteu").isin(0, 1)
    )
)

# Uma observação por CPF/mês; converteu se houver algum registro com 1.
por_cpf = base.groupBy("mes", "cpf").agg(
    F.countDistinct("testedigito").alias("n_grupos"),
    F.min("testedigito").alias("testedigito"),
    F.max("converteu").alias("converteu"),
)

if por_cpf.filter(F.col("n_grupos") > 1).limit(1).count():
    raise ValueError(
        "Existem CPFs em teste e controle no mesmo mês. "
        "Revise a atribuição dos grupos antes de comparar."
    )

resumo_mensal = por_cpf.groupBy("mes", "testedigito").agg(
    F.count("*").alias("total"),
    F.sum("converteu").alias("aprovados"),
)

# Apenas as contagens agregadas são transferidas para pandas.
pdf = resumo_mensal.toPandas()
if pdf.empty:
    raise ValueError("Não há registros válidos em agosto/setembro.")

pdf["mes"] = pd.to_datetime(pdf["mes"])
metricas = ["total", "aprovados"]

resultado = (
    pdf.pivot(index="mes", columns="testedigito", values=metricas)
    .reindex(
        index=meses,
        columns=pd.MultiIndex.from_product([metricas, [0, 1]]),
    )
    .fillna(0)
    .astype("int64")
)
resultado.columns = [
    f"{metrica}_{'teste' if grupo == 0 else 'controle'}"
    for metrica, grupo in resultado.columns
]
resultado = resultado.reset_index()


# 3. Fisher e Z bilaterais: teste versus controle DENTRO de cada mês.
def calcular_testes(linha: pd.Series) -> pd.Series:
    """Compara as proporções usando as contagens mensais observadas."""
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

    if np.any(aprovados < 0) or np.any(aprovados > totais):
        raise ValueError("Contagem de aprovados incompatível com o total.")
    if np.any(totais == 0):
        return pd.Series(testes)

    tabela = np.column_stack((aprovados, totais - aprovados))
    testes["p_valor_fisher"] = float(
        fisher_exact(tabela, alternative="two-sided")[1]
    )

    # O Z não é definido quando todos ou ninguém converteu nos dois grupos.
    if 0 < aprovados.sum() < totais.sum():
        z, p_z = proportions_ztest(
            count=aprovados,
            nobs=totais,
            value=0,
            alternative="two-sided",
            prop_var=False,
        )
        testes["p_valor_z"] = float(p_z)
        testes["estatistica_z"] = float(z)

    return pd.Series(testes)


colunas_testes = ["p_valor_fisher", "p_valor_z", "estatistica_z"]
resultado[colunas_testes] = resultado.apply(calcular_testes, axis=1)

# 4. Totais gerais, taxas, diferença absoluta e lift relativo.
resultado["total_geral"] = (
    resultado["total_teste"] + resultado["total_controle"]
)
resultado["aprovados_geral"] = (
    resultado["aprovados_teste"] + resultado["aprovados_controle"]
)

for grupo in ("teste", "controle", "geral"):
    resultado[f"taxa_{grupo}_pct"] = (
        100
        * resultado[f"aprovados_{grupo}"]
        / resultado[f"total_{grupo}"].replace(0, np.nan)
    )

resultado["diferenca_pp"] = (
    resultado["taxa_teste_pct"] - resultado["taxa_controle_pct"]
)
resultado["lift_razao"] = (
    resultado["taxa_teste_pct"]
    / resultado["taxa_controle_pct"].replace(0, np.nan)
)
resultado["lift_pct"] = 100 * (resultado["lift_razao"] - 1)

colunas = [
    "mes", "total_geral", "aprovados_geral", "taxa_geral_pct",
    "total_teste", "aprovados_teste", "taxa_teste_pct",
    "total_controle", "aprovados_controle", "taxa_controle_pct",
    "diferenca_pp", "lift_razao", "lift_pct",
    "p_valor_fisher", "p_valor_z",
]
arredondamento = {
    "taxa_geral_pct": 2,
    "taxa_teste_pct": 2,
    "taxa_controle_pct": 2,
    "diferenca_pp": 2,
    "lift_razao": 3,
    "lift_pct": 2,
}
display(
    resultado[colunas]
    .assign(mes=resultado["mes"].dt.strftime("%m/%Y"))
    .round(arredondamento)
)

lift_agosto, lift_setembro = resultado["lift_pct"].to_numpy()
variacao_lift_pp = lift_setembro - lift_agosto
comparativo_lift = pd.DataFrame([{
    "lift_agosto_pct": lift_agosto,
    "lift_setembro_pct": lift_setembro,
    "variacao_lift_pp": variacao_lift_pp,
}])
display(comparativo_lift.round(2))


# 5. Gráfico de aprovados, com denominadores, taxas e p-valores.
def formatar_pvalor(valor: float) -> str:
    """Preserva p-valores pequenos e marca o corte escolhido."""
    if pd.isna(valor):
        return "N/D"
    marcador = "*" if valor < ALFA else ""
    return f"{valor:.3g}{marcador}"


def formatar_percentual(valor: float) -> str:
    """Formata percentuais, incluindo os casos não definidos."""
    return "N/D" if pd.isna(valor) else f"{valor:.1f}%"


x = np.arange(2)
rotulos_meses = [f"Agosto/{ANO}", f"Setembro/{ANO}"]
largura = 0.36
fig, ax = plt.subplots(figsize=(11, 7))

for grupo, deslocamento in (("teste", -0.5), ("controle", 0.5)):
    barras = ax.bar(
        x + deslocamento * largura,
        resultado[f"aprovados_{grupo}"],
        width=largura,
        label=grupo.capitalize(),
    )
    rotulos = [
        f"{aprovados}/{total}\n({formatar_percentual(taxa)})"
        for aprovados, total, taxa in zip(
            resultado[f"aprovados_{grupo}"],
            resultado[f"total_{grupo}"],
            resultado[f"taxa_{grupo}_pct"],
        )
    ]
    ax.bar_label(barras, labels=rotulos, padding=4, fontsize=10)

maior = max(
    1, resultado[["aprovados_teste", "aprovados_controle"]].to_numpy().max()
)
for i, linha in resultado.iterrows():
    altura = max(linha["aprovados_teste"], linha["aprovados_controle"])
    texto = (
        f"Fisher: p = {formatar_pvalor(linha['p_valor_fisher'])}\n"
        f"Z: p = {formatar_pvalor(linha['p_valor_z'])}\n"
        f"Lift: {formatar_percentual(linha['lift_pct'])}"
    )
    ax.annotate(
        texto, (x[i], altura), xytext=(0, 48),
        textcoords="offset points", ha="center", va="bottom",
    )

ax.set_xticks(x)
ax.set_xticklabels(rotulos_meses)
ax.set_ylim(0, maior * 1.85)
ax.set_ylabel("CPFs aprovados (converteu = 1)")
ax.set_title(
    "Totais mensais: teste versus controle\n"
    f"Rótulos: aprovados/total e taxa | * p < {ALFA:g}",
    pad=15,
)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="upper left")
fig.tight_layout()
display(fig)
plt.close(fig)

# 6. Comparação descritiva do lift entre agosto e setembro.
fig, ax = plt.subplots(figsize=(9, 5))
valores = resultado["lift_pct"]
ax.bar(x, valores.fillna(0), width=0.5)

for i, valor in enumerate(valores):
    altura = 0 if pd.isna(valor) else valor
    texto = "N/D" if pd.isna(valor) else f"{valor:+.1f}%"
    acima = altura >= 0
    ax.annotate(
        texto, (x[i], altura), xytext=(0, 6 if acima else -6),
        textcoords="offset points", ha="center",
        va="bottom" if acima else "top",
    )

minimo = min(0, valores.fillna(0).min())
maximo = max(0, valores.fillna(0).max())
margem = max(5, (maximo - minimo) * 0.3)
ax.set_ylim(minimo - margem, maximo + margem)
ax.axhline(0, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(rotulos_meses)
ax.set_ylabel("Lift relativo sobre a taxa do controle")
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))

subtitulo = (
    "Variação não disponível"
    if pd.isna(variacao_lift_pp)
    else f"Setembro − agosto: {variacao_lift_pp:+.2f} p.p. de lift"
)
ax.set_title(f"Lift mensal do teste sobre o controle\n{subtitulo}")
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
display(fig)
plt.close(fig)
