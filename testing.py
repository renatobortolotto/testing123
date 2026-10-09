CONFIANCA = 0.85  # Use 0.95, 0.85 ou 0.80.
ALFA = 1.0 - CONFIANCA

for teste in ("fisher", "z"):
    p_valores = resultado[f"p_valor_{teste}"]

    resultado[f"significativo_{teste}"] = (
        p_valores.lt(ALFA)
        .astype("boolean")
        .mask(p_valores.isna())
    )

    from statsmodels.stats.proportion import confint_proportions_2indep


def calcular_ic_diferenca(linha: pd.Series, alfa: float) -> pd.Series:
    """Calcula o IC de teste menos controle, em pontos percentuais."""
    saida = {
        "ic_inferior_pp": np.nan,
        "ic_superior_pp": np.nan,
    }

    n_teste = int(linha["total_teste"])
    n_controle = int(linha["total_controle"])
    x_teste = int(linha["aprovados_teste"])
    x_controle = int(linha["aprovados_controle"])

    if n_teste <= 0 or n_controle <= 0:
        return pd.Series(saida)

    if not (
        0 <= x_teste <= n_teste
        and 0 <= x_controle <= n_controle
    ):
        raise ValueError(
            "Aprovados devem estar entre zero e o total do grupo."
        )

    inferior, superior = confint_proportions_2indep(
        count1=x_teste,
        nobs1=n_teste,
        count2=x_controle,
        nobs2=n_controle,
        method="newcomb",
        compare="diff",
        alpha=alfa,
    )

    saida["ic_inferior_pp"] = 100 * inferior
    saida["ic_superior_pp"] = 100 * superior

    return pd.Series(saida)


colunas_ic = ["ic_inferior_pp", "ic_superior_pp"]

resultado[colunas_ic] = resultado.apply(
    calcular_ic_diferenca,
    axis=1,
    alfa=ALFA,
)

resultado["nivel_confianca_ic"] = CONFIANCA