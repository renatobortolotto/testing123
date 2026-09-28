# Databricks notebook source
# NBA | Complemento 02B V2: diagnosticar os resultados da parte 2.
# Executar APOS a parte 2, no mesmo notebook. NAO substitui 01, 02 ou 03.
# Nao chama o ajuste, nao modifica parametros e nao escreve tabelas permanentes.
# Consulta somente as tabelas pequenas de modelos e validacao ja produzidas.
# Origem sem modelo e destino nao aprendido NAO desaparecem da cobertura.
# "p_na_entrada" e o mesmo modelo sem condicionar na idade; nao e outro modelo
# Markov treinado separadamente. Os denominadores ficam explicitos.

import numpy as np
import pandas as pd
from pyspark.sql import functions as F


# COMMAND ----------
def sm_diag_resumir(
    modelos: pd.DataFrame,
    validacao: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Resume falhas e metricas agregadas, sem ajustar modelos."""
    cols_m = {"origem", "status_modelo", "detalhe", "n_amostra", "n_grupos"}
    cols_v = {
        "origem", "idade_marco_seg", "n_elegiveis", "n_eventos",
        "n_eventos_suportados", "n_top1_temporal", "n_top1_semtempo",
        "n_top5_temporal", "n_ll_finitos", "soma_nll", "status_validacao",
    }
    if cols_m - set(modelos) or cols_v - set(validacao):
        raise ValueError("Schemas diferentes da parte 2 V2. Confira as views.")
    if modelos.empty or validacao.empty:
        raise ValueError("Modelo ou validacao vazios: este bloco exige um treino V2.")
    if modelos["origem"].duplicated().any():
        raise ValueError("Mais de um modelo por origem. Nao misture execucoes.")
    if validacao.duplicated(["origem", "idade_marco_seg"]).any():
        raise ValueError("Mais de uma linha por origem/marco na validacao.")

    m = modelos.copy()
    v = validacao.copy()
    contagens = sorted(cols_v - {"origem", "idade_marco_seg", "soma_nll", "status_validacao"})
    for coluna in contagens:
        v[coluna] = pd.to_numeric(v[coluna], errors="raise")
        if v[coluna].isna().any() or (v[coluna] < 0).any():
            raise ValueError(f"Contagem invalida em {coluna}.")
    if (v["n_eventos_suportados"] > v["n_eventos"]).any():
        raise ValueError("Eventos suportados excedem o total.")
    for coluna in ("n_top1_temporal", "n_top5_temporal", "n_top1_semtempo"):
        if (v[coluna] > v["n_eventos_suportados"]).any():
            raise ValueError(f"Acertos excedem eventos suportados em {coluna}.")

    zero = v.loc[v["idade_marco_seg"].eq(0), [
        "origem", "n_elegiveis", "n_eventos", "n_eventos_suportados",
    ]].rename(columns={
        "n_elegiveis": "n_observacoes_validacao_0",
        "n_eventos": "n_saidas_validacao_0",
        "n_eventos_suportados": "n_saidas_suportadas_0",
    })
    m = m.merge(zero, on="origem", how="left", validate="one_to_one")
    for coluna in zero.columns.drop("origem"):
        m[coluna] = m[coluna].fillna(0).astype("int64")
    m["motivo"] = m["detalhe"].fillna("").str.strip()
    m.loc[m["motivo"].eq(""), "motivo"] = "SEM_DETALHE"
    falhas = m.loc[~m["status_modelo"].eq("AJUSTADO")].copy()
    causas = (
        falhas.groupby(["status_modelo", "motivo"], dropna=False)
        .agg(
            n_origens=("origem", "size"),
            n_linhas_ajuste=("n_amostra", "sum"),
            n_saidas_validacao_0=("n_saidas_validacao_0", "sum"),
        )
        .reset_index()
        .sort_values(["n_saidas_validacao_0", "n_origens"], ascending=False)
    )
    impacto = falhas[[
        "origem", "motivo", "n_amostra", "n_saidas_validacao_0",
    ]].sort_values(["n_saidas_validacao_0", "n_amostra"], ascending=False)

    avaliada = v["status_validacao"].eq("AVALIADO_SUPORTE_EXPLICITO")
    v["n_eventos_origem_avaliada"] = v["n_eventos"].where(avaliada, 0)
    agregadas = contagens + ["n_eventos_origem_avaliada", "soma_nll"]
    metricas = v.groupby("idade_marco_seg", as_index=False)[agregadas].sum()

    def razao(numerador: str, denominador: str) -> pd.Series:
        return metricas[numerador] / metricas[denominador].replace(0, np.nan)

    metricas["cobertura_destinos"] = razao("n_eventos_suportados", "n_eventos")
    metricas["n_eventos_sem_origem_avaliada"] = (
        metricas["n_eventos"] - metricas["n_eventos_origem_avaliada"]
    )
    metricas["n_destinos_nao_aprendidos"] = (
        metricas["n_eventos_origem_avaliada"] - metricas["n_eventos_suportados"]
    )
    metricas["top1_temporal_total"] = razao("n_top1_temporal", "n_eventos")
    metricas["top1_entrada_total"] = razao("n_top1_semtempo", "n_eventos")
    metricas["top1_temporal_suporte"] = razao("n_top1_temporal", "n_eventos_suportados")
    metricas["top1_entrada_suporte"] = razao("n_top1_semtempo", "n_eventos_suportados")
    metricas["delta_top1_pp_total"] = 100 * (
        metricas["top1_temporal_total"] - metricas["top1_entrada_total"]
    )
    metricas["top5_temporal_total"] = razao("n_top5_temporal", "n_eventos")
    metricas["top5_temporal_suporte"] = razao("n_top5_temporal", "n_eventos_suportados")
    metricas["nll_media_somente_finitos"] = razao("soma_nll", "n_ll_finitos")
    metricas["n_observacoes_fora_nll_finita"] = (
        metricas["n_elegiveis"] - metricas["n_ll_finitos"]
    )
    return {"causas": causas, "impacto": impacto, "metricas": metricas}


# COMMAND ----------
# Coleta limitada a agregados por origem/marco. Nenhum ID de cliente e coletado.
def sm_diag_coletar(view: str, colunas: list[str], limite: int) -> pd.DataFrame:
    linhas = spark.table(view).select(*colunas).limit(limite + 1).collect()
    if len(linhas) > limite:
        raise ValueError(f"{view}: limite de agregados excedido; nao truncar.")
    return pd.DataFrame([linha.asDict() for linha in linhas], columns=colunas)


sm_diag_mod = sm_diag_coletar(
    "nba_sm_v2_modelos",
    ["origem", "status_modelo", "detalhe", "n_amostra", "n_grupos"],
    limite=5000,
)
sm_diag_val = sm_diag_coletar(
    "nba_sm_v2_validacao",
    [
        "origem", "idade_marco_seg", "n_elegiveis", "n_eventos",
        "n_eventos_suportados", "n_top1_temporal", "n_top1_semtempo",
        "n_top5_temporal", "n_ll_finitos", "soma_nll", "status_validacao",
    ],
    limite=25000,
)
sm_diag_resultado = sm_diag_resumir(sm_diag_mod, sm_diag_val)

print("1. CAUSAS DAS ORIGENS SEM AJUSTE")
print(sm_diag_resultado["causas"].head(20).to_string(index=False))
print("\n2. FALHAS COM MAIOR VOLUME DE SAIDAS NA VALIDACAO (MARCO ZERO)")
print(sm_diag_resultado["impacto"].head(15).to_string(index=False))

print("\n3. COBERTURA E ACERTO: DENOMINADORES EXPLICITOS")
sm_diag_metricas = sm_diag_resultado["metricas"]
print(sm_diag_metricas[[
    "idade_marco_seg", "n_eventos", "n_eventos_suportados",
    "cobertura_destinos", "top1_temporal_total", "top1_temporal_suporte",
    "top1_entrada_total", "delta_top1_pp_total",
]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n4. AUSENCIAS NA AVALIACAO")
print(sm_diag_metricas[[
    "idade_marco_seg", "n_eventos_sem_origem_avaliada",
    "n_destinos_nao_aprendidos", "n_observacoes_fora_nll_finita",
]].to_string(index=False))
print("\nNLL finita e uma media PARCIAL; nao inclui destinos com probabilidade zero.")
print("Nao comparar NLL entre marcos como se fossem as mesmas observacoes.")
print("Nenhum modelo reestimado por este codigo; nenhum parametro/tabela alterado.")
print("Nao diminuir minimos de suporte nem relaxar limites sem identificar a causa.")
print("Parte 3: manter GRAVAR_HOMOLOGACAO=False enquanto os resultados sao revisados.")
