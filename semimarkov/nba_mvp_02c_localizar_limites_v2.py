# Databricks notebook source
# NBA | 02C — Identificar exatamente o limite que rejeitou cada origem.
# Execute APOS 01/02/02B V2, no MESMO notebook. Nao substitui esses arquivos.
#
# Este e um DIAGNOSTICO NUMERICO, nao uma correcao automatica dos modelos.
# Repete a otimizacao SOMENTE das origens com LIMITE_PARAMETRICO_ATINGIDO.
# Nao refaz a base, nao usa validacao para ajustar, nao altera os 59 ajustes
# existentes, nao altera probabilidades, nao grava Delta e nao libera publicacao.
#
# A amostra de ajuste V2 nao foi persistida antes da primeira execucao. Sua
# reavaliacao pode produzir outra realizacao. Este bloco materializa a amostra
# agora e informa diferencas de contagem; contagens iguais NAO provam mesmas linhas.
#
# Dependencias: funcoes do nucleo da parte 2, sm_dados_ajuste, sm_modelos,
# SM_AJUSTE_CFG, SM_CFG e MAX_LINHAS_POR_ORIGEM.

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import optimize, special
from pyspark import StorageLevel
from pyspark.sql import functions as F

SMC_MAX_ORIGENS = 40
SMC_TOL_LIMITE = 1e-4  # Coordenada de otimizacao; igual ao criterio original.
SMC_TOL_TIMEOUT_SEG = 1e-6  # Comparacao com a resolucao de microssegundos.

requeridos = [
    "sm_dados_ajuste", "sm_modelos", "SM_AJUSTE_CFG", "SM_CFG",
    "_arrays", "_termos_lognormais", "_objetivo_conjunto",
    "MAX_LINHAS_POR_ORIGEM",
]
faltantes = [nome for nome in requeridos if nome not in globals()]
if faltantes:
    raise RuntimeError(f"Execute antes a parte 2 V2. Ausentes: {faltantes}")

# Seleciona apenas as origens rejeitadas pelo controle de limites da parte 2.
smc_alvos = (
    sm_modelos
    .filter(
        (F.col("status_modelo") != "AJUSTADO")
        & F.col("detalhe").contains("LIMITE_PARAMETRICO_ATINGIDO")
    )
    .select(
        "origem",
        F.col("n_amostra").alias("n_amostra_primeira_tentativa"),
    )
    .distinct()
    .persist(StorageLevel.MEMORY_AND_DISK)
)
smc_n_origens = smc_alvos.count()
if smc_n_origens == 0:
    raise ValueError("Nao ha origens com LIMITE_PARAMETRICO_ATINGIDO.")
if smc_n_origens > SMC_MAX_ORIGENS:
    raise ValueError(
        f"Foram encontradas {smc_n_origens} origens; limite de seguranca "
        f"deste diagnostico: {SMC_MAX_ORIGENS}. Revise antes de ampliar."
    )
if smc_alvos.groupBy("origem").count().filter("count > 1").limit(1).count():
    raise ValueError("Existe mais de um resultado de ajuste por origem.")

smc_dados = (
    sm_dados_ajuste
    .join(
        F.broadcast(smc_alvos.select(F.col("origem").alias("estado"))),
        "estado", "left_semi",
    )
    .select("cd_bv", "passo", "estado", "destino", "dur_min", "dur_max", "tipo_censura")
    .persist(StorageLevel.MEMORY_AND_DISK)
)
smc_n_linhas = smc_dados.count()  # Materializa somente o recorte das origens-alvo.
if smc_n_linhas == 0:
    raise ValueError("Amostra vazia; verifique as variaveis da parte 2.")

smc_amostra_atual = (
    smc_dados.groupBy(F.col("estado").alias("origem"))
    .agg(F.count("*").alias("n_amostra_diagnostico"))
)
smc_contagens = (
    smc_alvos.join(smc_amostra_atual, "origem", "left")
    .withColumn(
        "delta_contagem",
        F.col("n_amostra_diagnostico") - F.col("n_amostra_primeira_tentativa"),
    )
)
if smc_contagens.filter(F.col("n_amostra_diagnostico").isNull()).limit(1).count():
    raise ValueError("Uma origem com falha nao esta presente na amostra atual.")

SMC_TIMEOUT_SEG = float(SM_CFG["timeout_seg"])
print("Origens a inspecionar:", smc_n_origens)
print("Linhas nesta tentativa:", smc_n_linhas)
print("Configuracao original preservada:", asdict(SM_AJUSTE_CFG))
print("Timeout operacional (segundos):", SMC_TIMEOUT_SEG)
print("Comparacao de contagens; mesmo tamanho nao garante mesmas observacoes:")
smc_contagens.orderBy(F.abs(F.col("delta_contagem")).desc()).show(40, truncate=False)

# COMMAND ----------
# Nucleo isolado do diagnostico. Reusa _arrays e _objetivo_conjunto da parte 2.
# A preparacao e as duas inicializacoes abaixo reproduzem ajustar_origem V2.

def smc_repetir_otimizacao(pdf: pd.DataFrame, cfg) -> dict:
    """Repete o ajuste V2 sem alterar limites nem aprovar um modelo.

    Mesma funcao objetivo, agrupamento, inicializacoes e tolerancias da V2.
    Retem os parametros antes da rejeicao para permitir a inspecao.
    """
    x = _arrays(pdf)
    df, known, w = x["df"], x["known"], x["w"]
    n_clientes = int(df["cd_bv"].nunique())
    n_eventos = int(w[known].sum())
    if n_eventos < cfg.minimo_eventos or n_clientes < cfg.minimo_clientes:
        raise ValueError("SUPORTE_INSUFICIENTE")
    obs = df.loc[known].assign(_peso=w[known])
    suporte = obs.groupby("destino", sort=True).agg(
        n=("_peso", "sum"), n_clientes=("cd_bv", "nunique")
    ).sort_values(["n"], ascending=False, kind="mergesort")
    proprios = list(suporte.loc[
        (suporte["n"] >= cfg.minimo_eventos_grupo)
        & (suporte["n_clientes"] >= cfg.minimo_clientes_grupo)
    ].head(cfg.max_grupos_proprios).index)
    destinos = sorted(suporte.index.astype(str))
    grupo_destino = {j: k for k, j in enumerate(proprios)}
    raros = [j for j in destinos if j not in grupo_destino]
    if raros:
        grupo_destino.update({j: len(proprios) for j in raros})
    g = max(grupo_destino.values()) + 1
    grupos = np.array([grupo_destino[j] for j in destinos], dtype=int)
    n_j = np.array([float(suporte.loc[j, "n"]) for j in destinos])
    n_g = np.bincount(grupos, weights=n_j, minlength=g)
    r_j = n_j / n_g[grupos]
    r_map = dict(zip(destinos, r_j))
    x["grupo"] = np.array([
        grupo_destino[str(j)] if k else -1
        for j, k in zip(df["destino"], known)
    ])
    x["log_r"] = np.array([
        np.log(r_map[str(j)]) if k else 0.
        for j, k in zip(df["destino"], known)
    ])
    # Intervalos: ponto interno SOMENTE para inicializar, nunca como observacao.
    ref = x["lo"][known].copy()
    inter = x["tipo"][known] == 1
    ref[inter] = .5 * (x["lo"][known][inter] + x["hi"][known][inter])
    lt = np.log(ref)
    ww = w[known]
    mu0 = float(np.average(lt, weights=ww))
    sig0 = float(np.clip(np.sqrt(np.average((lt - mu0)**2, weights=ww)), .35, 2.5))
    mus = np.array([
        np.average(lt[x["grupo"][known] == k], weights=ww[x["grupo"][known] == k])
        for k in range(g)
    ])
    probs0 = (n_g + cfg.pseudocontagem_grupo) / (n_g.sum() + cfg.pseudocontagem_grupo * g)
    logits0 = np.log(probs0[:-1]) - np.log(probs0[-1])
    # Limites amplos, declarados. Ajustes que encostam neles nao sao aprovados.
    mu_lo, mu_hi = -20., 15.
    bounds = ([(-25., 25.)] * (g - 1)
              + [(mu_lo, mu_hi)] * g
              + [(np.log(cfg.sigma_min), np.log(cfg.sigma_max))] * g)
    resultados = []
    for init_mu in (mus, np.full(g, mu0)):
        init = np.r_[logits0, np.clip(init_mu, -19., 14.), np.full(g, np.log(sig0))]
        res = optimize.minimize(
            _objetivo_conjunto, init, args=(x, g, mu0, sig0, cfg),
            method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": cfg.maxiter, "ftol": 1e-10, "gtol": 1e-6, "maxls": 40},
        )
        if res.success and np.isfinite(res.fun) and np.isfinite(res.x).all():
            resultados.append(res)
    if not resultados:
        raise ValueError("NAO_CONVERGIU")
    res = min(resultados, key=lambda v: v.fun)
    th = res.x
    lp = np.r_[th[:g - 1], 0.]
    pi = special.softmax(lp)
    mu = th[g - 1:2 * g - 1]
    sigma = np.exp(th[2 * g - 1:])
    return {
        "dados": x,
        "destinos": destinos,
        "grupos_destino": grupos,
        "g": g,
        "theta": th,
        "bounds": bounds,
        "mu": mu,
        "sigma": sigma,
        "pi": pi,
        "resultado": res,
    }


def smc_identificar_limites(theta: np.ndarray, bounds: list, g: int) -> list[dict]:
    """Identifica coordenada, grupo e lado com o mesmo criterio da V2."""
    encontrados = []
    for indice, (valor, (low, high)) in enumerate(zip(theta, bounds)):
        if abs(valor - low) < SMC_TOL_LIMITE:
            lado = "INFERIOR"
        elif abs(valor - high) < SMC_TOL_LIMITE:
            lado = "SUPERIOR"
        else:
            continue
        if indice < g - 1:
            parametro, grupo, escala = "logit_grupo", indice, False
        elif indice < 2 * g - 1:
            parametro, grupo, escala = "meanlog_dias", indice - (g - 1), False
        else:
            parametro, grupo, escala = "sdlog", indice - (2 * g - 1), True
        encontrados.append({
            "parametro": parametro,
            "grupo_temporal": int(grupo),
            "lado": lado,
            "valor": float(np.exp(valor) if escala else valor),
            "limite_inferior": float(np.exp(low) if escala else low),
            "limite_superior": float(np.exp(high) if escala else high),
        })
    return encontrados


def smc_perfil_grupo(dados: dict, grupo: int, timeout: float) -> dict:
    """Perfil apenas das duracoes EXATAS do grupo; nao imputa intervalos."""
    mascara = (dados["grupo"] == grupo) & dados["known"]
    exatas = mascara & (dados["tipo"] == 0)
    tempos = dados["df"].loc[exatas, "dur_min"].to_numpy(float)
    saida = {
        "n_saidas_grupo": int(mascara.sum()),
        "n_exatas_grupo": int(exatas.sum()),
        "n_tempos_exatos_distintos": 0,
        "tempo_modal_seg": None,
        "fracao_moda_exatas": None,
        "fracao_timeout_exatas": None,
        "q05_exatas_seg": None,
        "mediana_exatas_seg": None,
        "q95_exatas_seg": None,
        "sd_log_t_exatas": None,
    }
    if not len(tempos):
        return saida
    unicos, contagens = np.unique(tempos, return_counts=True)
    pos = int(np.argmax(contagens))
    q05, mediana, q95 = np.quantile(tempos, [0.05, 0.5, 0.95])
    saida.update({
        "n_tempos_exatos_distintos": int(len(unicos)),
        "tempo_modal_seg": float(unicos[pos]),
        "fracao_moda_exatas": float(contagens[pos] / len(tempos)),
        "fracao_timeout_exatas": float(np.mean(
            np.isclose(tempos, timeout, rtol=0, atol=SMC_TOL_TIMEOUT_SEG)
        )),
        "q05_exatas_seg": float(q05),
        "mediana_exatas_seg": float(mediana),
        "q95_exatas_seg": float(q95),
        "sd_log_t_exatas": float(np.std(np.log(tempos), ddof=0)),
    })
    return saida


SMC_SCHEMA = (
    "origem string, status_diagnostico string, detalhe string, "
    "grupo_temporal long, destinos string, parametro string, lado string, "
    "valor double, limite_inferior double, limite_superior double, "
    "n_amostra_diagnostico long, n_grupos long, n_saidas_grupo long, "
    "n_exatas_grupo long, n_tempos_exatos_distintos long, tempo_modal_seg double, "
    "fracao_moda_exatas double, fracao_timeout_exatas double, "
    "q05_exatas_seg double, mediana_exatas_seg double, q95_exatas_seg double, "
    "sd_log_t_exatas double, perda_penalizada_media double, "
    "iteracoes long, mensagem_otimizador string, assinatura_amostra string"
)
SMC_COLUNAS = [campo.strip().split()[0] for campo in SMC_SCHEMA.split(",")]


def smc_diagnosticar_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
    """Retorna so parametros e estatisticas agregadas, sem IDs de clientes."""
    pdf = pdf.sort_values(["cd_bv", "passo"], kind="mergesort").reset_index(drop=True)
    hash_linhas = pd.util.hash_pandas_object(
        pdf[["cd_bv", "passo", "destino", "dur_min", "dur_max", "tipo_censura"]],
        index=False,
    ).to_numpy(np.uint64)
    # Assinatura do conjunto, nao por cliente; para rastrear esta tentativa apenas.
    assinatura = hashlib.sha256(hash_linhas.tobytes()).hexdigest()
    origem = str(pdf["estado"].iloc[0])
    base = {coluna: None for coluna in SMC_COLUNAS}
    base.update(
        origem=origem,
        n_amostra_diagnostico=int(len(pdf)),
        assinatura_amostra=assinatura,
        status_diagnostico="SEM_RESULTADO",
    )
    linhas = []
    try:
        if len(pdf) > MAX_LINHAS_POR_ORIGEM:
            raise ValueError("LIMITE_MEMORIA_DIAGNOSTICO")
        inspecao = smc_repetir_otimizacao(pdf, SM_AJUSTE_CFG)
        resultado = inspecao["resultado"]
        base.update(
            n_grupos=int(inspecao["g"]),
            perda_penalizada_media=float(resultado.fun),
            iteracoes=int(resultado.nit),
            mensagem_otimizador=str(resultado.message),
        )
        hits = smc_identificar_limites(
            inspecao["theta"], inspecao["bounds"], inspecao["g"]
        )
        if not hits:
            base.update(
                status_diagnostico="LIMITE_NAO_REPRODUZIDO",
                detalhe="Revisar realizacao da amostra e configuracao; nao substitui modelo.",
            )
            linhas.append(base)
        for hit in hits:
            grupo = hit["grupo_temporal"]
            destinos = [
                destino for destino, g in zip(
                    inspecao["destinos"], inspecao["grupos_destino"]
                ) if int(g) == grupo
            ]
            perfil = smc_perfil_grupo(inspecao["dados"], grupo, SMC_TIMEOUT_SEG)
            linhas.append(dict(
                base,
                **hit,
                **perfil,
                destinos=" | ".join(destinos),
                status_diagnostico="LIMITE_REPRODUZIDO",
                detalhe="Parametros mantidos; este resultado nao e modelo aprovado.",
            ))
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        base.update(status_diagnostico="DIAGNOSTICO_INTERROMPIDO", detalhe=str(exc)[:500])
        linhas.append(base)
    saida = pd.DataFrame(linhas, columns=SMC_COLUNAS)
    for campo in SMC_SCHEMA.split(","):
        nome, tipo = campo.strip().split()
        if tipo == "long":
            saida[nome] = pd.array(saida[nome], dtype="Int64")
        elif tipo == "double":
            saida[nome] = pd.to_numeric(saida[nome], errors="raise").astype(float)
    return saida

# COMMAND ----------
# Executa SOMENTE o diagnostico, sem atualizar sm_modelos ou seus broadcasts.
smc_limites = (
    smc_dados.groupBy("estado")
    .applyInPandas(smc_diagnosticar_pdf, schema=SMC_SCHEMA)
    .persist(StorageLevel.MEMORY_AND_DISK)
)
smc_limites.count()  # Materializa uma vez antes dos resumos.
smc_limites.createOrReplaceTempView("nba_sm_v2_diagnostico_limites")

print("1. QUAL PARAMETRO ATINGIU QUAL LIMITE")
smc_limites.groupBy("status_diagnostico", "parametro", "lado").agg(
    F.countDistinct("origem").alias("n_origens"),
    F.count("*").alias("n_ocorrencias"),
).show(40, truncate=False)

print("2. GRUPOS AFETADOS E CONCENTRACAO DAS DURACOES")
smc_limites.filter(F.col("status_diagnostico") == "LIMITE_REPRODUZIDO").select(
    "origem", "grupo_temporal", "destinos", "parametro", "lado",
    "valor", "limite_inferior", "limite_superior",
    "n_saidas_grupo", "n_exatas_grupo", "n_tempos_exatos_distintos",
    "tempo_modal_seg", "fracao_moda_exatas", "fracao_timeout_exatas",
    "q05_exatas_seg", "mediana_exatas_seg", "q95_exatas_seg", "sd_log_t_exatas",
).orderBy(F.desc("n_saidas_grupo"), "origem", "grupo_temporal").show(40, truncate=False)

print("3. CASOS NAO REPRODUZIDOS OU INTERROMPIDOS")
smc_limites.filter(F.col("status_diagnostico") != "LIMITE_REPRODUZIDO").select(
    "origem", "status_diagnostico", "detalhe", "n_amostra_diagnostico",
).show(40, truncate=False)

print("Diagnostico concluido. Nenhum modelo, previsao ou tabela permanente foi alterado.")
print("Nao remova a protecao hit_bound nem aumente limites antes de interpretar o perfil.")
print("A parte 3 permanece com GRAVAR_HOMOLOGACAO=False.")
