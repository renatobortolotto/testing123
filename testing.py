from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter

# ============================================================
# 1. CONFIGURAÇÕES
# ============================================================

# Caso ainda não tenha carregado os dados:
# df = spark.table("catalogo.schema.sua_tabela")

COL_DATA = "sua_coluna_de_data"        # Substitua pelo nome real
COL_METRICA = "sua_coluna_de_metrica"  # Substitua pelo nome real

AGREGACAO = "media"  # Opções: "media", "soma" ou "contagem"

# True para mostrar uma taxa decimal como percentual:
# por exemplo, 0.15 aparecerá como 15%.
FORMATAR_PERCENTUAL = False

# ============================================================
# 2. PREPARAÇÃO
# ============================================================

base = (
    df
    .withColumn("_data", F.to_date(F.col(COL_DATA)))
    .withColumn("_digito", F.col("testedigito").cast("double"))
    .filter(
        F.col("_data").isNotNull()
        & F.col("_digito").isin(0, 1)
    )
    .withColumn(
        "grupo",
        F.when(F.col("_digito") == 1, F.lit("Controle"))
         .otherwise(F.lit("Teste"))
    )
    .withColumn(
        "semana",
        F.date_trunc("week", F.col("_data")).cast("date")
    )
)

# Para datas em texto no formato dd/MM/yyyy, substitua acima por:
# F.to_date(F.col(COL_DATA), "dd/MM/yyyy")

# ============================================================
# 3. AGREGAÇÃO SEMANAL
# ============================================================

if AGREGACAO == "media":
    expressao = F.avg(F.col(COL_METRICA).cast("double"))
    rotulo_y = f"Média de {COL_METRICA}"

elif AGREGACAO == "soma":
    expressao = F.sum(F.col(COL_METRICA).cast("double"))
    rotulo_y = f"Soma de {COL_METRICA}"

elif AGREGACAO == "contagem":
    expressao = F.count(F.lit(1))
    rotulo_y = "Quantidade de registros"

else:
    raise ValueError("AGREGACAO deve ser 'media', 'soma' ou 'contagem'.")

resumo_semanal = (
    base
    .groupBy("semana", "grupo")
    .agg(expressao.alias("valor"))
    .orderBy("semana", "grupo")
)

display(resumo_semanal)

# Apenas o resultado agregado é levado para pandas.
pdf = resumo_semanal.toPandas()

if pdf.empty:
    raise ValueError(
        "Não há dados para plotar. Confira as datas e os valores de testedigito."
    )

pdf["semana"] = pd.to_datetime(pdf["semana"])

# Uma coluna por grupo e uma linha por semana.
dados_grafico = (
    pdf.pivot(index="semana", columns="grupo", values="valor")
       .sort_index()
       .reindex(columns=["Controle", "Teste"])
)

# Inclui semanas ausentes como lacunas, sem inventar valores zero.
semanas = pd.date_range(
    start=dados_grafico.index.min(),
    end=dados_grafico.index.max(),
    freq="W-MON"
)

dados_grafico = dados_grafico.reindex(semanas)

# ============================================================
# 4. GRÁFICO
# ============================================================

fig, ax = plt.subplots(figsize=(13, 5))

for grupo in ["Controle", "Teste"]:
    ax.plot(
        dados_grafico.index,
        dados_grafico[grupo].astype(float),
        marker="o",
        linewidth=2,
        label=f"{grupo} — testedigito = {1 if grupo == 'Controle' else 0}"
    )

ax.set_title("Comparação semanal — Controle x Teste")
ax.set_xlabel("Semana — data de início")
ax.set_ylabel(rotulo_y)

if FORMATAR_PERCENTUAL:
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))

ax.legend()
ax.grid(True, alpha=0.25)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()