# Databricks notebook source
# MAGIC %md
# MAGIC # NBA V2.3 — Retomada estável e Multi-step A × B2
# MAGIC **Importe este arquivo em um notebook NOVO e use Executar tudo.**
# MAGIC Substitui a Parte 03 antiga e os patches de retomada; não retreina 01/02/02C.
# MAGIC Não copie este conteúdo para o notebook antigo e não execute patches junto.
# MAGIC Retoma somente o treinamento já aprovado; ignora variáveis de outras sessões.
# MAGIC O recorte é histórico. A saída não é uma lista de disparos do público atual.
# MAGIC Fontes não são sobrescritas nem têm retenção alterada.

# COMMAND ----------

import uuid
from datetime import datetime, timezone

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

# ÚNICO bloco de configuração. Não precisa copiar IDs de outras células.
MS03_CFG = {
    "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
    "id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
    "id_02c_preferido": "53d097ef-c176-485b-9848-4fa06b653064",
    "tabela_execucoes_02c": "ctg_dsti.renato_nba.nba_sm_v23_02c_execucoes_hml",
    "prefixo_saida": "ctg_dsti.renato_nba.nba_sm_v23_estavel",
    "gravar": True,
    "autotestes": True,
    "max_passos_macro": 5,
    "idades_seg": [0.0, 1800.0, 86400.0, 604800.0],
    "bootstrap_replicas": 2000,
    "semente": 20261009,
    "max_modelos_driver": 5000,
    "max_nos_expandidos": 6000,
    "max_arestas": 1000000,
    "max_kernels_driver": 30000,
    "max_agregados_driver": 100000,
    "max_celulas_bootstrap": 10000000,
    "max_celulas_numericas": 10000000,
    "max_linhas_exibir": 20,
    "tolerancia_prob": 1e-9,
    "tolerancia_massa": 1e-8,
    "tolerancia_duracao_seg": 1e-3,
}
MS03_ALVOS = sorted({
    "pix_transferencia", "pix_pagamento", "pix_cadastro_chave", "pix_trazer_chave",
    "pagamentos_boleto", "gerar_boleto_financeira", "cobrancas_gerar_pix",
    "cobrancas_gerar_boleto", "cartoes_aumentar_limite", "cartoes_ativar_credito",
    "investimentos_aplicacao", "seguros_auto_avulso_cotacao", "tag_veicular_solicitar",
    "tag_veicular_ativar", "rodas_assinar_contrato",
})
MS03_VERSAO = "v2.3_multistep_retomada_estavel_v1"
MS03_ID = str(uuid.uuid4())
MS03_INICIO = datetime.now(timezone.utc).isoformat()
MS03_FONTES, MS03_CHECKS, MS03_RELATORIOS, MS03_CACHES = {}, [], {}, []
MS03_CHAVE = ["cd_bv", "passo", "idade_seg"]
MS03_SAIDAS = {nome: f"{MS03_CFG['prefixo_saida']}_{nome}_hml" for nome in (
    "snapshots", "kernels", "predicoes", "comparacoes", "negocio_historico",
    "resumos", "execucoes",
)}
# Os pacotes são cópias físicas novas, lidas pelo ID, sem consulta histórica.
RET_PACOTES = f"{MS03_CFG['prefixo_saida']}_pacotes_hml"
RET_ENTRADAS = f"{MS03_CFG['prefixo_saida']}_entradas_hml"
RET_CONTRATO = "pacote_logico_verificado_v1"
RET_FONTES = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Motor e funções locais
# MAGIC Definições próprias deste notebook. Sem %run, arquivos auxiliares ou patch.
# MAGIC O motor numérico e o alvo histórico são os da Parte 03 anterior.

# COMMAND ----------

# Núcleo independente de Spark. Não ajusta parâmetros.
from collections import deque
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse, special, stats
from scipy.sparse.linalg import splu

MS_A = "A_REFERENCIA"
MS_B2 = "B2_MEMORIA_TEMPO_COMPARTILHADO"
MS_BASE = "__BASE__"
MS_SEM_MEMORIA = "__SEM_MEMORIA__"


def ms_hash(objeto: Any) -> str:
    texto = json.dumps(objeto, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def ms_validar_modelo(m: dict) -> None:
    if m.get("formato") != "sm_v23_lognormal_grupos":
        raise ValueError("Formato diferente do contrato V2.3.")
    if m.get("unidade_tempo") != "dias":
        raise ValueError("Unidade temporal diferente de dias.")
    pi, mu, sigma, r, p = [np.asarray(m[k], float) for k in (
        "pi_grupo", "mu_grupo", "sigma_grupo", "r_destino_no_grupo", "p_destino"
    )]
    g = np.asarray(m["grupo"], int)
    nd, ng = len(m["destinos"]), len(pi)
    if not nd or not ng or len(set(m["destinos"])) != nd:
        raise ValueError("Vocabulário vazio ou duplicado.")
    if m["n_grupos"] != ng or any(len(v) != ng for v in (mu, sigma)):
        raise ValueError("Dimensões temporais incompatíveis.")
    if any(len(v) != nd for v in (g, r, p)):
        raise ValueError("Dimensões dos destinos incompatíveis.")
    if not np.array_equal(np.unique(g), np.arange(ng)):
        raise ValueError("Grupos fora do contrato.")
    if not all(np.isfinite(v).all() for v in (pi, mu, sigma, r, p)):
        raise ValueError("Parâmetros não finitos.")
    if min(pi.min(), sigma.min(), r.min(), p.min()) <= 0:
        raise ValueError("Massa/sigma não positivo.")
    if not np.isclose(pi.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Massa dos grupos não fecha.")
    if not np.allclose(np.bincount(g, weights=r), 1, rtol=0, atol=1e-10):
        raise ValueError("Massas dentro dos grupos não fecham.")
    if not np.allclose(pi[g] * r, p, rtol=0, atol=1e-10):
        raise ValueError("p_destino divergente de pi/grupo/r.")
    if (m["estrutura"]["destinos"] != m["destinos"]
            or m["estrutura"]["grupo"] != m["grupo"]):
        raise ValueError("Estrutura divergente dos parâmetros.")


def ms_prever_raw(m: dict, idade_seg: float) -> dict:
    """Mesma inferência em log-space da 02C; não renormaliza o Top 5."""
    if not math.isfinite(idade_seg) or idade_seg < 0:
        raise ValueError("Idade inválida.")
    pi, mu, sigma = [np.asarray(m[k], float) for k in (
        "pi_grupo", "mu_grupo", "sigma_grupo"
    )]
    log_sf = np.zeros(len(pi), float)
    if idade_seg > 0:
        t = idade_seg / 86400.0
        log_sf = special.log_ndtr(-(np.log(t) - mu) / sigma)
        independente = stats.lognorm.logsf(t, s=sigma, scale=np.exp(mu))
        if not np.allclose(log_sf, independente, rtol=1e-10, atol=1e-8):
            raise ValueError("Sobrevivência não reproduz a referência independente.")
    log_mass = np.log(pi) + log_sf
    log_q = (log_mass - special.logsumexp(log_mass))[np.asarray(m["grupo"], int)]
    log_q += np.log(np.asarray(m["r_destino_no_grupo"], float))
    q = np.exp(log_q)
    if not np.isfinite(log_q).all() or not np.isclose(q.sum(), 1, atol=1e-9):
        raise ValueError("Distribuição RAW inválida.")
    ordem = np.argsort(-q, kind="mergesort")
    return {"q": q, "log_q": log_q, "ordem": ordem,
            "soma_quadrados": float(q @ q)}


class MotorMacroMemoria:
    """Estado expandido = (estado RAW, última memória de negócio).

    Mudança de funil custa 1; etapas internas custam 0. Primeira chegada a
    uma ação-alvo APÓS mudar de macro é absorvida. Reentrada no funil inicial,
    depois de sair dele, é permitida. Não estima prazo em minutos/dias.
    """

    def __init__(self, variante: str, config: dict, tecnicas: set,
                 a: dict, b2: dict, suporte: dict, alvos: list,
                 passos: int = 5, max_nos: int = 6000,
                 max_arestas: int = 1000000, tol: float = 1e-8):
        if variante not in (MS_A, MS_B2) or passos < 1:
            raise ValueError("Variante ou horizonte inválido.")
        self.variante, self.config = variante, config
        self.tecnicas, self.a, self.b2, self.suporte = set(tecnicas), a, b2, suporte
        self.alvos, self.k, self.tol = sorted(alvos), int(passos), float(tol)
        if len(set(alvos)) != len(alvos) or not alvos:
            raise ValueError("Whitelist vazia ou duplicada.")
        macros = {self.macro(s) for s in config}
        if not set(alvos) <= macros or set(alvos) & self.tecnicas:
            raise ValueError("Whitelist fora da config ou contendo funil técnico.")
        for s, m in a.items():
            if s not in config or not set(m["destinos"]) <= set(config):
                raise ValueError("Origem/destino sem configuração.")
            ms_validar_modelo(m)
        for (s, c), m in b2.items():
            if s not in a or config[s].get("funil") not in self.tecnicas:
                raise ValueError("Contexto sem pai ou em origem não técnica.")
            if not set(m["destinos"]) <= set(config):
                raise ValueError("Destino contextual sem configuração.")
            ms_validar_modelo(m)
        memorias = sorted({v.get("funil") for v in config.values()
                           if v.get("funil") and v["funil"] not in self.tecnicas}
                          | {MS_SEM_MEMORIA})
        self.nos = []
        for s in sorted(config):
            f = config[s].get("funil")
            self.nos.extend((s, c) for c in ([f] if f and f not in self.tecnicas
                                           else memorias))
        if len(self.nos) > max_nos:
            raise RuntimeError("Grafo expandido excede proteção; não foi truncado.")
        self.indice = {chave: i for i, chave in enumerate(self.nos)}
        self.n = len(self.nos)
        self.rotas, self.modelos = [], []
        self.term = np.array([bool(config[s].get("terminal_real", False))
                              for s, _ in self.nos], bool)
        self.alvo = np.array([self.alvos.index(self.macro(s))
                              if self.macro(s) in self.alvos else -1
                              for s, _ in self.nos], int)
        if np.any(self.term & (self.alvo >= 0)):
            raise ValueError("Alvo também marcado terminal; revisar semântica.")
        qi, qj, qv, ri, rj, rv = [], [], [], [], [], []
        self.sem_modelo = np.zeros(self.n)
        for i, (s, c) in enumerate(self.nos):
            m, rota = self.selecionar(s, c)
            self.modelos.append(m)
            self.rotas.append(rota)
            if self.term[i]:
                continue
            if m is None:
                self.sem_modelo[i] = 1.0
                continue
            for d, p in zip(m["destinos"], m["p_destino"]):
                j = self.indice[(d, self.atualizar(c, d))]
                if self.macro(s) == self.macro(d):
                    qi.append(i)
                    qj.append(j)
                    qv.append(float(p))
                else:
                    ri.append(i)
                    rj.append(j)
                    rv.append(float(p))
            if len(qv) + len(rv) > max_arestas:
                raise RuntimeError("Grafo excede proteção de arestas; não foi truncado.")
        self.q = sparse.csr_matrix((qv, (qi, qj)), shape=(self.n, self.n))
        self.r = sparse.csr_matrix((rv, (ri, rj)), shape=(self.n, self.n))
        self._precomputar()

    def macro(self, estado: str) -> str:
        return self.config[estado].get("funil") or estado

    def atualizar(self, memoria: str | None, destino: str) -> str:
        f = self.config[destino].get("funil")
        return f if f and f not in self.tecnicas else (memoria or MS_SEM_MEMORIA)

    def selecionar(self, estado: str, memoria: str) -> tuple:
        pai = self.a.get(estado)
        if pai is None:
            return None, "SEM_MODELO_ORIGEM"
        if self.variante == MS_A:
            return pai, "ORIGEM_REFERENCIA"
        if self.config[estado].get("funil") not in self.tecnicas:
            return pai, "ORIGEM_NAO_TECNICA"
        if memoria == MS_SEM_MEMORIA:
            return pai, "ORIGEM_SEM_MEMORIA"
        filho = self.b2.get((estado, memoria))
        if filho is not None:
            return filho, "CONTEXTO_AJUSTADO"
        status = self.suporte.get((estado, memoria), "NAO_VISTO_NO_TREINO")
        if status == "CANDIDATO_CONTEXTUAL":
            return pai, "ORIGEM_FALHA_AJUSTE_CONTEXTO"
        if status == "NAO_VISTO_NO_TREINO":
            return pai, "ORIGEM_CONTEXTO_NAO_VISTO"
        return pai, "ORIGEM_CONTEXTO_RARO"

    def _precomputar(self) -> None:
        # Nós sem acesso a mudança de macro/terminal/falta de modelo, por
        # transições internas, retêm massa. Não são tratados como sem modelo.
        saida = np.asarray(self.r.sum(axis=1)).ravel()
        ativos = (saida > 0) | (self.sem_modelo > 0) | self.term
        reverso = self.q.transpose().tocsr()
        fila = deque(np.flatnonzero(ativos).tolist())
        while fila:
            j = fila.popleft()
            for i in reverso.indices[reverso.indptr[j]:reverso.indptr[j + 1]]:
                if not ativos[i]:
                    ativos[i] = True
                    fila.append(int(i))
        self.vivos, self.fechados = np.flatnonzero(ativos), np.flatnonzero(~ativos)
        q_live = self.q[self.vivos][:, self.vivos].tocsc()
        sistema = sparse.eye(len(self.vivos), format="csc") - q_live
        lu = splu(sistema) if len(self.vivos) else None
        self.max_residuo = 0.0

        def resolver(rhs: np.ndarray, valor_fechado=None) -> np.ndarray:
            sol = np.zeros_like(rhs, dtype=float)
            if len(self.vivos):
                x = lu.solve(np.asarray(rhs[self.vivos], order="F"))
                resid = float(np.max(np.abs(sistema @ x - rhs[self.vivos])))
                self.max_residuo = max(self.max_residuo, resid)
                if resid > self.tol or x.min(initial=0) < -self.tol:
                    raise RuntimeError("Sistema interno instável; não usar pseudoinversa.")
                sol[self.vivos] = np.maximum(x, 0.0)
            if valor_fechado is not None:
                sol[self.fechados] = valor_fechado
            return sol

        h = np.zeros((self.n, len(self.alvos)))
        idx = np.flatnonzero(self.alvo >= 0)
        h[idx, self.alvo[idx]] = 1.0
        direto = self.r @ h
        continuar = (self.alvo < 0) & ~self.term
        r_cont = self.r.multiply(continuar[None, :]).tocsr()
        base_massas = np.zeros((self.n, 4))  # sem modelo, terminal, retenção, limite
        base_massas[:, 0] = self.sem_modelo
        base_massas[:, 1] = self.term.astype(float) + self.r @ self.term.astype(float)
        if len(self.fechados):
            base_massas[:, 2] = np.asarray(self.q[:, self.fechados].sum(axis=1)).ravel()
        self.exatas = np.zeros((self.k + 1, self.n, len(self.alvos)))
        self.massas = np.zeros((self.k + 1, self.n, 4))
        self.massas[0, :, 3] = 1.0
        self.max_erro_massa = 0.0
        acumulado = np.zeros_like(h)
        for passo in range(1, self.k + 1):
            rhs = direto if passo == 1 else r_cont @ self.exatas[passo - 1]
            self.exatas[passo] = resolver(rhs)
            self.massas[passo] = resolver(
                base_massas + r_cont @ self.massas[passo - 1], [0.0, 0.0, 1.0, 0.0]
            )
            acumulado += self.exatas[passo]
            erro = float(np.max(np.abs(acumulado.sum(axis=1)
                                      + self.massas[passo].sum(axis=1) - 1.0)))
            self.max_erro_massa = max(self.max_erro_massa, erro)
            if erro > self.tol:
                raise RuntimeError("Massa macro não fecha; não renormalizar para ocultar.")
        self.resumo = {"variante": self.variante, "n_nos": self.n,
                       "n_arestas_internas": int(self.q.nnz),
                       "n_arestas_macro": int(self.r.nnz),
                       "n_nos_retencao_interna": len(self.fechados),
                       "n_nos_sem_modelo": int(self.sem_modelo.sum()),
                       "max_erro_massa": self.max_erro_massa,
                       "max_residuo_sistema": self.max_residuo}

    def prever(self, estado: str, memoria: str, idade: float) -> dict:
        memoria = memoria or MS_SEM_MEMORIA
        if self.atualizar(memoria, estado) != memoria:
            raise ValueError("Memória atual incompatível com o estado observado.")
        i = self.indice[(estado, memoria)]
        m = self.modelos[i]
        if m is None or self.term[i]:
            return {"tem_previsao": False, "rota": self.rotas[i]}
        raw = ms_prever_raw(m, idade)
        contrib = np.zeros((len(self.alvos), self.k))
        massas = np.zeros(4)
        for d, p in zip(m["destinos"], raw["q"]):
            j = self.indice[(d, self.atualizar(memoria, d))]
            if self.macro(d) == self.macro(estado):
                contrib += p * self.exatas[1:self.k + 1, j, :].T
                massas += p * self.massas[self.k, j]
            elif self.alvo[j] >= 0:
                contrib[self.alvo[j], 0] += p
            elif self.term[j]:
                massas[1] += p
            else:
                if self.k > 1:
                    contrib[:, 1:] += p * self.exatas[1:self.k, j, :].T
                massas += p * self.massas[self.k - 1, j]
        scores = contrib.sum(axis=1)
        erro = abs(float(scores.sum() + massas.sum() - 1))
        if erro > self.tol or not np.isfinite(contrib).all():
            raise RuntimeError("Previsão não fecha massa no snapshot.")
        ordem = sorted(range(len(scores)), key=lambda j: (-scores[j], self.alvos[j]))
        util_top5 = [(r + 1, self.alvos[j], float(scores[j]),
                      int(np.argmax(contrib[j]) + 1))
                     for r, j in enumerate([j for j in ordem if scores[j] > 0][:5])]
        raw_top5 = [(r + 1, m["destinos"][int(j)], float(raw["q"][j]))
                    for r, j in enumerate(raw["ordem"][:5])]
        return {"tem_previsao": True, "rota": self.rotas[i], "raw_top5": raw_top5,
                "raw_probs": dict(zip(m["destinos"], map(float, raw["q"]))),
                "util_top5": util_top5,
                "util_probs": dict(zip(self.alvos, map(float, scores))),
                "util_ranks": {self.alvos[j]: r + 1 for r, j in enumerate(ordem)},
                "soma_quadrados_util": float(scores @ scores),
                "massa_acao_util": float(scores.sum()),
                "massa_sem_modelo": float(massas[0]),
                "massa_terminal": float(massas[1]),
                "massa_retencao_interna": float(massas[2]),
                "massa_limite_macro": float(massas[3]),
                "prob_sem_alvo_modelada": float(massas[1:].sum()),
                "alerta_limite": bool(m.get("alerta_limite", False)),
                "extrapolacao_tempo": bool(idade / 86400 > m["max_tempo_observado_dias"]),
                "erro_massa": erro}


def ms_bootstrap(pdf: pd.DataFrame, metricas: list, replicas: int,
                 seed: int, max_celulas: int) -> list[dict]:
    """Diferença pareada B2−A; ganho positivo é melhora em todas as métricas."""
    rng = np.random.default_rng(seed)
    resultados = []
    perdas = {"brier_acoes", "brier_algum_alvo", "logloss_conservador"}
    for (segmento, idade, coorte), sub in pdf.groupby(
            ["segmento", "idade_seg", "coorte"], sort=True):
        for metrica in metricas:
            p = sub.pivot(index="cd_bv", columns="variante", values=metrica)
            if set(p.columns) != {MS_A, MS_B2}:
                raise ValueError("Comparação incompleta entre variantes.")
            if not p[MS_A].isna().equals(p[MS_B2].isna()):
                raise ValueError("Métrica disponível em somente uma variante.")
            p = p.dropna()
            if p.empty:
                continue
            a, b = p[MS_A].to_numpy(float), p[MS_B2].to_numpy(float)
            ganho = a - b if metrica in perdas else b - a
            if len(p) * replicas > max_celulas:
                raise RuntimeError("Bootstrap excede proteção; não foi amostrado.")
            indices = rng.integers(0, len(p), (replicas, len(p)))
            lo, hi = np.quantile(ganho[indices].mean(axis=1), [0.025, 0.975])
            resultados.append({
                "segmento": str(segmento), "idade_seg": float(idade),
                "coorte": str(coorte), "metrica": metrica, "n_clientes": len(p),
                "n_casos": int(sub.loc[
                    sub["variante"].eq(MS_A) & sub["cd_bv"].isin(p.index), "n_casos"
                ].sum()),
                "media_a": float(a.mean()), "media_b2": float(b.mean()),
                "ganho_medio": float(ganho.mean()), "ganho_ic95_lo": float(lo),
                "ganho_ic95_hi": float(hi),
            })
    return resultados


def ms_autotestes_numericos() -> None:
    """Testes determinísticos de memória, relógio e conservação de massa."""
    def modelo(destinos, probs, mu=None):
        n = len(destinos)
        return {
            "formato": "sm_v23_lognormal_grupos", "unidade_tempo": "dias",
            "destinos": list(destinos), "pi_grupo": list(probs), "grupo": list(range(n)),
            "mu_grupo": mu if mu is not None else [-1.0] * n,
            "sigma_grupo": [1.0] * n, "r_destino_no_grupo": [1.0] * n,
            "p_destino": list(probs), "n_grupos": n, "alerta_limite": False,
            "parametros_no_limite": [], "max_tempo_observado_dias": 90.0,
            "estrutura": {"destinos": list(destinos), "grupo": list(range(n))},
        }

    config = {s: {"funil": f, "terminal_real": False} for s, f in (
        ("p", "pix"), ("b", "boleto"), ("l1", "login"), ("l2", "login"),
        ("ls", "login"), ("chat", None), ("m", "sem_modelo_real"),
    )}
    a = {"p": modelo(["l1"], [1.0]), "b": modelo(["l1"], [1.0]),
         "l1": modelo(["l2"], [1.0]), "l2": modelo(["ls"], [1.0]),
         "ls": modelo(["p", "b"], [0.5, 0.5]), "chat": modelo(["l1"], [1.0])}
    b2 = {("ls", "pix"): modelo(["p", "b"], [0.9, 0.1]),
          ("ls", "boleto"): modelo(["p", "b"], [0.2, 0.8])}
    original = ms_hash({"a": a, "b2": {str(k): v for k, v in b2.items()}})
    motor = MotorMacroMemoria(MS_B2, config, {"login"}, a, b2, {}, ["pix", "boleto"], 2)
    r = motor.prever("p", "pix", 0.0)
    assert np.isclose(r["util_probs"]["pix"], 0.9)
    assert r["util_top5"][0][3] == 2  # PIX -> login -> PIX: reentrada válida.
    assert np.isclose(motor.prever("b", "boleto", 0.0)["util_probs"]["boleto"], 0.8)
    assert motor.atualizar("boleto", "l2") == "boleto"
    assert motor.atualizar("boleto", "chat") == "boleto"
    assert motor.atualizar("boleto", "p") == "pix"
    assert not motor.prever("m", "sem_modelo_real", 0.0)["tem_previsao"]
    um = MotorMacroMemoria(MS_B2, config, {"login"}, a, b2, {}, ["pix", "boleto"], 1)
    assert np.isclose(um.prever("p", "pix", 0.0)["massa_limite_macro"], 1.0)
    assert um.prever("l1", "boleto", 0.0)["util_top5"][0][3] == 1
    # Alvo sem modelo pode ser alcançado. Modelo só é necessário para continuar.
    desconhecido = a | {"l1": modelo(["m"], [1.0])}
    um = MotorMacroMemoria(MS_A, config, {"login"}, desconhecido, {}, {}, ["pix"], 1)
    dois = MotorMacroMemoria(MS_A, config, {"login"}, desconhecido, {}, {}, ["pix"], 2)
    assert um.prever("l1", MS_SEM_MEMORIA, 0.0)["massa_limite_macro"] == 1.0
    assert dois.prever("l1", MS_SEM_MEMORIA, 0.0)["massa_sem_modelo"] == 1.0
    alvo_m = MotorMacroMemoria(MS_A, config, {"login"}, desconhecido, {}, {}, ["sem_modelo_real"], 2)
    assert alvo_m.prever("l1", MS_SEM_MEMORIA, 0.0)["massa_acao_util"] == 1.0
    fechado = a | {"l1": modelo(["l2"], [1.0]), "l2": modelo(["l1"], [1.0])}
    loop = MotorMacroMemoria(MS_A, config, {"login"}, fechado, {}, {}, ["pix"], 5)
    assert loop.prever("l1", MS_SEM_MEMORIA, 0.0)["massa_retencao_interna"] == 1.0
    geometrico = a | {"l1": modelo(["l1", "p"], [0.8, 0.2])}
    geo = MotorMacroMemoria(MS_A, config, {"login"}, geometrico, {}, {}, ["pix"], 1)
    assert np.isclose(geo.prever("l1", MS_SEM_MEMORIA, 0.0)["massa_acao_util"], 1.0)
    temporal = a | {"l1": modelo(["p", "b"], [0.5, 0.5], [-6.0, 0.0])}
    tempo = MotorMacroMemoria(MS_A, config, {"login"}, temporal, {}, {}, ["pix", "boleto"], 5)
    assert tempo.prever("l1", "pix", 86400.0)["util_probs"]["boleto"] > 0.99
    for idade in (0.0, 1.0, 1800.0, 86400.0, 604800.0):
        for s, mem in tempo.nos:
            r = tempo.prever(s, mem, idade)
            if r["tem_previsao"]:
                assert abs(r["massa_acao_util"] + r["prob_sem_alvo_modelada"]
                           + r["massa_sem_modelo"] - 1) < 1e-8
    assert original == ms_hash({"a": a, "b2": {str(k): v for k, v in b2.items()}})
    print("03: autotestes numéricos de memória, ciclos, reentrada, relógio e massa OK.")



def ms03_sql(tabela: str) -> str:
    partes = tabela.split(".")
    if len(partes) != 3 or not all(p.replace("_", "").isalnum() for p in partes):
        raise ValueError(f"Nome SQL inválido: {tabela}")
    return ".".join(f"`{p}`" for p in partes)


def ms03_cols(df: DataFrame, colunas: set, rotulo: str) -> None:
    faltam = colunas - set(df.columns)
    if faltam:
        raise RuntimeError(f"{rotulo}: colunas ausentes: {sorted(faltam)}")


def ms03_zero(nome: str, df: DataFrame) -> None:
    n = int(df.count())
    MS03_CHECKS.append((nome, n, "OK" if n == 0 else "FALHA"))
    if n:
        raise RuntimeError(f"Verificação {nome}: {n} divergências. Nada foi promovido.")


def ms03_unico(df: DataFrame, rotulo: str) -> dict:
    rows = df.limit(2).collect()
    if len(rows) != 1:
        raise RuntimeError(f"{rotulo}: esperado um registro; encontrados {len(rows)}.")
    return rows[0].asDict(recursive=True)


def ms03_cache(df: DataFrame) -> DataFrame:
    df = df.persist(StorageLevel.DISK_ONLY)
    df.count()
    MS03_CACHES.append(df)
    return df


def ms03_pequeno(df: DataFrame, limite: int, rotulo: str) -> list:
    rows = df.limit(limite + 1).collect()
    if len(rows) > limite:
        raise RuntimeError(f"{rotulo} excede proteção; nenhuma linha foi truncada.")
    return rows


def ms03_mostrar(nome: str, df: DataFrame, ordem=None) -> None:
    MS03_RELATORIOS[nome] = df
    print(f"\n{nome}", flush=True)
    vis = df.orderBy(*ordem) if ordem else df
    vis = vis.drop("cd_bv", "modelo_json", "raw_probs", "util_probs", "util_ranks")
    for campo in vis.schema.fields:
        if campo.dataType.simpleString() in ("double", "float"):
            vis = vis.withColumn(campo.name, F.round(F.col(campo.name), 6))
    vis.show(MS03_CFG["max_linhas_exibir"], truncate=False)


def ms03_com_meta(df: DataFrame) -> DataFrame:
    return (df.withColumn("id_experimento", F.lit(MS03_CFG["id_experimento"]))
            .withColumn("id_ajuste_fonte", F.lit(MS03_CFG["id_ajuste"]))
            .withColumn("id_02c_fonte", F.lit(MS03_CFG["id_02c"]))
            .withColumn("id_ms03", F.lit(MS03_ID))
            .withColumn("versao_ms03", F.lit(MS03_VERSAO))
            .withColumn("hash_ms03", F.lit(MS03_HASH)))


def ms03_guardar(df: DataFrame, nome: str) -> DataFrame:
    tabela = MS03_SAIDAS[nome]
    if tabela in RET_FONTES:
        raise RuntimeError("Uma saída coincide com uma fonte.")
    if not MS03_CFG["gravar"]:
        return ms03_cache(df)
    if spark.catalog.tableExists(tabela):
        ms03_cols(spark.table(tabela), {"id_ms03"}, tabela)
        if spark.table(tabela).filter(F.col("id_ms03") == MS03_ID).limit(1).count():
            raise RuntimeError("Etapa já escrita nesta execução; use Executar tudo para novo ID.")
    ms03_com_meta(df).write.format("delta").mode("append").saveAsTable(tabela)
    return spark.table(tabela).filter(F.col("id_ms03") == MS03_ID)

def ms03_rotular(base: DataFrame, alvos: list, k: int) -> DataFrame:
    """Primeira entrada em macro alvo diferente, em até k mudanças observadas.

    Não requer modelo para rotular. Continuação do macro atual não é alvo;
    reentrada nele depois de outro macro é. Futuro incompleto é censurado.
    """
    w = Window.partitionBy("cd_bv").orderBy("passo")
    atual_ok = F.col("estado").isNotNull() & ~F.coalesce(
        F.col("barreira_memoria"), F.lit(True)
    )
    df = base.withColumn("_ok", atual_ok).withColumn(
        "_macro", F.coalesce("funil", "estado")
    )
    for coluna in ("passo", "estado", "ts_estado", "_ok"):
        df = df.withColumn(f"_next{coluna}", F.lead(coluna).over(w))
    consecutivo = F.col("_nextpasso") == F.col("passo") + 1
    exata = F.col("elegivel_ajuste") & (F.col("tipo_censura") == "exata")
    delta = (F.expr("unix_micros(_nextts_estado)")
             - F.expr("unix_micros(ts_estado)")) / F.lit(1000000.0)
    erro_dest = ~F.col("destino").eqNullSafe(F.col("_nextestado"))
    erro_tempo = F.abs(delta - F.col("dur_min")) > MS03_CFG["tolerancia_duracao_seg"]
    ms03_zero("labels_arestas_exatas", df.filter(
        exata & consecutivo & F.col("_ok") & F.col("_next_ok")
        & (erro_dest | erro_tempo)
    ))
    df = df.withColumn("_link", F.coalesce(
        exata & consecutivo & F.col("_ok") & F.col("_next_ok")
        & ~erro_dest & ~erro_tempo, F.lit(False)
    )).withColumn("_prev_link", F.lag("_link").over(w))
    df = df.withColumn("_seg", F.sum(F.when(
        F.coalesce("_prev_link", F.lit(False)), 0
    ).otherwise(1)).over(w.rowsBetween(Window.unboundedPreceding, 0)))
    ws = Window.partitionBy("cd_bv", "_seg").orderBy("passo")
    df = df.withColumn("_prev_macro", F.lag("_macro").over(ws))
    df = df.withColumn("_bloco", F.sum(F.when(
        F.col("_macro").eqNullSafe(F.col("_prev_macro")), 0
    ).otherwise(1)).over(ws.rowsBetween(Window.unboundedPreceding, 0)))
    pontos = df.filter("_ok").select("cd_bv", "passo", "_seg", "_bloco")
    blocos = df.filter("_ok").groupBy("cd_bv", "_seg", "_bloco").agg(
        F.first("_macro").alias("_macro"),
        F.min("ts_estado").alias("_entrada"),
    )
    wb = Window.partitionBy("cd_bv", "_seg").orderBy("_bloco")
    for h in range(1, k + 1):
        blocos = blocos.withColumn(f"_acao_{h}", F.lead("_macro", h).over(wb))
        blocos = blocos.withColumn(f"_ts_{h}", F.lead("_entrada", h).over(wb))
    candidatos = [F.when(F.col(f"_acao_{h}").isin(alvos), F.struct(
        F.col(f"_acao_{h}").alias("acao"), F.lit(h).alias("mudanca"),
        F.col(f"_ts_{h}").alias("ts"),
    )) for h in range(1, k + 1)]
    blocos = blocos.withColumn("_primeira", F.coalesce(*candidatos))
    blocos = blocos.withColumn("status_label", F.when(
        F.col("_primeira").isNotNull(), "ALVO_OBSERVADO"
    ).when(F.col(f"_acao_{k}").isNotNull(), "HORIZONTE_OBSERVADO_SEM_ALVO")
        .otherwise("CENSURADO_ANTES_HORIZONTE"))
    labels = blocos.select(
        "cd_bv", "_seg", "_bloco", "status_label",
        F.col("_primeira.acao").alias("acao_real"),
        F.col("_primeira.mudanca").alias("passo_macro_real"),
        F.col("_primeira.ts").alias("ts_acao_real"),
        sum(F.col(f"_acao_{h}").isNotNull().cast("int") for h in range(1, k + 1))
        .alias("n_mudancas_futuras_observadas"),
    )
    return pontos.join(labels, ["cd_bv", "_seg", "_bloco"], "left").drop("_seg", "_bloco")


def ms03_segmentos(df: DataFrame) -> DataFrame:
    return df.withColumn("segmento", F.explode(F.when(
        F.col("origem_tecnica"), F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS"))
    ).otherwise(F.array(F.lit("TODAS")))))


def ms03_autoteste_spark() -> None:
    # Boleto atual -> etapas login -> boleto (nova ocorrência) -> consulta.
    sequencias = {
        "u1": [("b", "boleto"), ("l1", "login"), ("l2", "login"),
               ("b", "boleto"), ("q", "consulta")],
        "u2": [("l1", "login"), ("l2", "login")],
        "u3": [("q", "consulta"), (None, None), ("b", "boleto")],
        "u4": [("q", "consulta"), ("l1", "login"), ("q", "consulta")],
    }
    rows = []
    for u, seq in sequencias.items():
        for i, (s, f) in enumerate(seq):
            nxt = seq[i + 1][0] if i + 1 < len(seq) else None
            tipo = "exata" if s and nxt else "direita"
            rows.append((u, i + 1, s, f, nxt if tipo == "exata" else None,
                         1.0, tipo, s is not None, s is None))
    df = spark.createDataFrame(rows, (
        "cd_bv string, passo long, estado string, funil string, destino string, "
        "dur_min double, tipo_censura string, elegivel_ajuste boolean, barreira_memoria boolean"
    )).withColumn("ts_estado", F.expr("timestamp_micros(1700000000000000 + passo * 1000000)"))
    r = {(x.cd_bv, x.passo): x.asDict() for x in ms03_rotular(df, ["boleto"], 2).collect()}
    assert r[("u1", 1)]["acao_real"] == "boleto" and r[("u1", 1)]["passo_macro_real"] == 2
    assert r[("u1", 2)]["passo_macro_real"] == 1
    assert r[("u1", 3)]["passo_macro_real"] == 1
    assert r[("u2", 1)]["status_label"] == "CENSURADO_ANTES_HORIZONTE"
    assert r[("u3", 1)]["status_label"] == "CENSURADO_ANTES_HORIZONTE"
    assert r[("u4", 1)]["status_label"] == "HORIZONTE_OBSERVADO_SEM_ALVO"
    # Acesso seguro por ranking: mapa ausente retorna NULL, não erro ANSI.
    t = spark.createDataFrame([(1, {1: "pix"}), (5, {1: "pix"})], "r int, m map<int,string>")
    rr = t.select(F.element_at("m", F.col("r")).alias("v")).collect()
    assert rr[0].v == "pix" and rr[1].v is None
    print("03: autotestes Spark de labels, censura, reentrada e ranks OK.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Recuperar o experimento concluído e congelar entradas
# MAGIC Ignora IDs globais temporários. Primeiro tenta o pacote independente já salvo.
# MAGIC Na primeira retomada lê o registro concluído e as partições pelo ID original.
# MAGIC Para a base/catálogo exige histórico contínuo que prove ausência de mudanças
# MAGIC destrutivas desde a versão referenciada. Contagens, chaves, política, grupos e
# MAGIC snapshots são conferidos. ID e hash de configuração sozinhos NÃO provam
# MAGIC igualdade de conteúdo. Nenhuma proteção de retenção é desativada.
# MAGIC Se os dados foram substituídos/apagados e não há evidência suficiente, para
# MAGIC antes do Multi-step. Não inventa um experimento nem muda o treinamento.

# COMMAND ----------


def ret_escolher_manifesto(rows: list[dict], cfg: dict) -> dict:
    """Somente 02Cs concluídas do treinamento aprovado; nunca C02_ID em memória."""
    validos = [r for r in rows if (
        r.get("status") == "CONCLUIDO_COMPARACAO_02C"
        and r.get("id_experimento") == cfg["id_experimento"]
        and r.get("id_ajuste_fonte") == cfg["id_ajuste"]
    )]
    if not validos:
        raise RuntimeError(
            "Não existe 02C concluída para o treinamento aprovado. "
            "Nenhum outro treinamento será escolhido e nada será retreinado."
        )
    ids = [r["id_02c"] for r in validos]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Manifestos concluídos duplicados para o mesmo id_02c.")
    preferido = cfg.get("id_02c_preferido")
    encontrado = [r for r in validos if r["id_02c"] == preferido]
    if encontrado:
        return encontrado[0]
    # Só recomposições concluídas do MESMO ajuste, não o modelo mais recente.
    def ordem(r):
        fim = datetime.fromisoformat(r["fim_utc"].replace("Z", "+00:00"))
        if fim.tzinfo is None:
            raise ValueError("Manifesto sem timezone no instante de conclusão.")
        return fim.timestamp(), r["id_02c"]
    return max(validos, key=ordem)


def ret_provar_historico(historico: list[dict], referencia: int,
                         atual: int, permite_append: bool) -> list[str]:
    """Prova de preservação de linhas; não confunde hash de política com de dados."""
    if referencia > atual:
        raise RuntimeError("A versão atual é anterior à fonte registrada.")
    if referencia == atual:
        return []
    trecho = {int(r["version"]): r for r in historico
              if referencia <= int(r["version"]) <= atual}
    if set(trecho) != set(range(referencia, atual + 1)):
        raise RuntimeError(
            "O histórico de alterações não cobre todas as versões necessárias. "
            "Não é possível comprovar que o recorte foi preservado."
        )
    sem_mudar_linhas = {
        "SET TBLPROPERTIES", "UNSET TBLPROPERTIES", "OPTIMIZE",
        "VACUUM START", "VACUUM END", "COMPUTE STATS", "ANALYZE TABLE",
    }
    operacoes = []
    for v in range(referencia + 1, atual + 1):
        registro = trecho[v]
        op = str(registro.get("operation", "")).upper()
        params = registro.get("operationParameters") or {}
        append = (op == "WRITE" and str(params.get("mode", "")).lower() == "append")
        if op not in sem_mudar_linhas and not (permite_append and append):
            raise RuntimeError(
                f"Versão {v}: operação {op!r} não comprova preservação. "
                "Não será substituído o histórico por dados atuais diferentes."
            )
        operacoes.append(f"{v}:{op}" + (":Append" if append else ""))
    return operacoes


def ret_historia(tabela: str) -> list[dict]:
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Tabela necessária não encontrada: {tabela}")
    return [r.asDict(recursive=True) for r in spark.sql(
        f"DESCRIBE HISTORY {ms03_sql(tabela)}"
    ).collect()]


def ret_filtrar(df: DataFrame, filtros: dict) -> DataFrame:
    ms03_cols(df, set(filtros), "Filtro de execução")
    for coluna, valor in filtros.items():
        df = df.filter(F.col(coluna) == F.lit(valor))
    return df


def ret_ler_corrente(tabela: str, filtros: dict, referencia=None,
                     permite_append=True) -> DataFrame:
    """Leitura atual com prova de histórico, filtro prévio e materialização em disco.

    Não pede time travel. Se houve somente append, a igualdade do recorte depende
    também das contagens/chaves que são verificadas no preflight.
    """
    print(f"Verificando entrada: {tabela}", flush=True)
    historico = ret_historia(tabela)
    if not historico:
        raise RuntimeError(f"Histórico Delta vazio: {tabela}")
    antes = max(int(r["version"]) for r in historico)
    operacoes = []
    if referencia is not None:
        operacoes = ret_provar_historico(
            historico, int(referencia), antes, permite_append
        )
    df = ret_filtrar(spark.table(tabela), filtros)
    df = ms03_cache(df)
    depois = int(spark.sql(
        f"DESCRIBE HISTORY {ms03_sql(tabela)} LIMIT 1"
    ).first()["version"])
    if depois != antes:
        df.unpersist()
        raise RuntimeError(
            f"A fonte {tabela} mudou durante a leitura. "
            "Aguarde o outro escritor e reexecute somente este notebook."
        )
    RET_FONTES[tabela] = {
        "versao_referenciada": referencia, "versao_lida": antes,
        "filtros": filtros, "operacoes_verificadas": operacoes,
        "modo": "leitura_atual_verificada_por_id",
    }
    return df


def ret_assinatura(df: DataFrame) -> dict:
    """Checksum de transporte do NOVO pacote; não é prova do histórico anterior."""
    cols = sorted(df.columns)
    j = F.to_json(F.struct(*[F.col(c) for c in cols]), {
        "ignoreNullFields": "false", "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    })
    h = df.select(F.sha2(j, 256).alias("h"))
    agregados = [F.count("*").alias("n")]
    for i in range(4):
        agregados.append(F.sum(F.conv(F.substring("h", i * 16 + 1, 16), 16, 10)
                              .cast("decimal(38,0)")).alias(f"s{i}"))
    vals = h.agg(*agregados).first().asDict()
    return {"n": int(vals.pop("n")), "somas_sha256": [str(vals[f"s{i}"] or 0)
                                                      for i in range(4)]}


def ret_ler_pacote(meta: dict) -> dict:
    fuso = meta["fit_config"]["fonte_preparo"]["fuso_base"]
    spark.conf.set("spark.sql.session.timeZone", fuso)
    entradas = {}
    for tipo, info in meta["entradas"].items():
        tabela = info["tabela"]
        if not spark.catalog.tableExists(tabela):
            raise RuntimeError(f"Pacote concluído sem a tabela {tabela}.")
        df = spark.table(tabela).filter(F.col("id_pacote") == meta["id_pacote"])
        df = ms03_cache(df.select(*info["colunas"]))
        if df.schema.jsonValue() != info["schema"] or ret_assinatura(df) != info["assinatura"]:
            raise RuntimeError(f"O pacote materializado foi modificado: {tipo}.")
        entradas[tipo] = df
    print("Pacote independente recuperado:", meta["id_pacote"], flush=True)
    return entradas


def ret_gravar_pacote(entradas: dict, meta: dict) -> dict:
    """Somente tabelas novas deste notebook; substituição restrita ao pacote incompleto."""
    pacote = meta["id_pacote"]
    uuid.UUID(pacote)
    meta["entradas"] = {}
    for tipo, df in entradas.items():
        tabela = f"{MS03_CFG['prefixo_saida']}_fonte_{tipo}_hml"
        if tabela in RET_FONTES:
            raise RuntimeError("Destino do pacote coincide com fonte.")
        assinatura = ret_assinatura(df)
        saida = df.withColumn("id_pacote", F.lit(pacote))
        (saida.write.format("delta").mode("overwrite")
         .option("replaceWhere", f"id_pacote = '{pacote}'")
         .partitionBy("id_pacote").saveAsTable(tabela))
        copia = spark.table(tabela).filter(F.col("id_pacote") == pacote).select(*df.columns)
        # Comparação exata, inclusive multiplicidade de linhas e microssegundos.
        if df.exceptAll(copia).limit(1).count() or copia.exceptAll(df).limit(1).count():
            raise RuntimeError(f"Cópia independente diverge da entrada: {tipo}")
        meta["entradas"][tipo] = {
            "tabela": tabela, "colunas": df.columns, "schema": copia.schema.jsonValue(),
            "assinatura": assinatura,
        }
    meta["fontes_verificadas"] = dict(RET_FONTES)
    json_meta = json.dumps(meta, sort_keys=True, ensure_ascii=False, allow_nan=False)
    manifesto = spark.createDataFrame([(
        pacote, "CONCLUIDO_PACOTE", meta["id_experimento"], meta["id_ajuste"],
        meta["id_02c"], RET_CONTRATO, ms_hash(meta), json_meta,
        datetime.now(timezone.utc).isoformat(),
    )], "id_pacote string, status string, id_experimento string, id_ajuste string, "
        "id_02c string, contrato string, hash_pacote string, meta_json string, fim_utc string")
    (manifesto.write.format("delta").mode("overwrite")
     .option("replaceWhere", f"id_pacote = '{pacote}'")
     .partitionBy("id_pacote").saveAsTable(RET_PACOTES))
    print("Entradas congeladas em pacote independente:", pacote, flush=True)
    return meta


def ret_suporte_da_base(base: DataFrame, preparo: dict) -> DataFrame:
    # Mesma regra da Parte 01; não refaz eventos, pesos nem treinamento.
    p = json.loads(preparo["politica_json"])
    treino = base.filter(~F.col("validacao_cliente") & F.col("elegivel_ajuste"))
    exata = F.col("tipo_censura") == "exata"
    s = treino.groupBy("estado", "contexto_modelo", "origem_tecnica").agg(
        F.count("*").alias("n_observacoes_treino"),
        F.sum(exata.cast("long")).alias("n_saidas_exatas"),
        F.sum((F.col("tipo_censura") == "direita").cast("long")).alias("n_censuras_direita"),
        F.countDistinct("cd_bv").alias("n_clientes_treino"),
        F.countDistinct(F.when(exata, F.col("cd_bv"))).alias("n_clientes_exatas"),
        F.countDistinct(F.when(exata, F.col("destino"))).alias("n_destinos_exatos"),
        F.sum("peso_cliente_treino").alias("massa_pesos_cliente"),
    ).withColumn("contexto_tem_suporte_candidato", F.col("origem_tecnica")
        & (F.col("contexto_modelo") != MS_SEM_MEMORIA)
        & (F.col("n_saidas_exatas") >= p["min_eventos_contexto"])
        & (F.col("n_clientes_exatas") >= p["min_clientes_contexto"]))
    return s.withColumn("status_suporte", F.when(~F.col("origem_tecnica"), "MODELO_POR_ORIGEM")
        .when(F.col("contexto_modelo") == MS_SEM_MEMORIA, "SEM_MEMORIA_OBSERVADA")
        .when(F.col("contexto_tem_suporte_candidato"), "CANDIDATO_CONTEXTUAL")
        .otherwise("CONTEXTO_RARO_USAR_SUPORTE_ORIGEM"))


def ret_validar_entradas(entradas: dict, meta: dict) -> None:
    base = entradas["base"]
    p = meta["fit_config"]["fonte_preparo"]
    esperado = int(p["n_linhas_base"])
    if base.count() != esperado:
        raise RuntimeError("A base por ID não tem a contagem original; não foi aceita como equivalente.")
    ms03_zero("ret_base_chaves", base.groupBy("cd_bv", "passo").count().filter("count != 1"))
    ms03_zero("ret_politica", base.filter(
        ~F.col("politica_sha256").eqNullSafe(F.lit(p["politica_sha256"]))
    ))
    ms03_zero("ret_corte", base.filter(
        ~F.col("ts_corte_estado").eqNullSafe(F.lit(p["corte_estado_iso"]).cast("timestamp"))
    ))
    ms03_zero("ret_splits", base.groupBy("cd_bv").agg(
        F.countDistinct("validacao_cliente").alias("n")).filter("n != 1"))
    modelos = entradas["modelos_ab"]
    ms03_zero("ret_modelos_hash", modelos.filter(
        ~F.col("hash_ajuste").eqNullSafe(F.lit(meta["manifesto_fit"]["hash_ajuste"]))
    ))
    ms03_zero("ret_modelos_chaves", modelos.groupBy(
        "variante", "estado", "contexto_modelo").count().filter("count != 1"))
    treino = base.filter(~F.col("validacao_cliente") & F.col("elegivel_ajuste"))
    contagens = treino.groupBy("estado").agg(
        F.count("*").alias("n"), F.countDistinct("cd_bv").alias("clientes"),
        F.sum((F.col("tipo_censura") == "exata").cast("long")).alias("exatas"),
    )
    a = modelos.filter((F.col("variante") == MS_A) & ~F.col("reuso"))
    cmp = a.join(contagens, "estado", "full")
    ms03_zero("ret_contagens_origens", cmp.filter(
        ~F.col("n_observacoes").eqNullSafe(F.col("n"))
        | ~F.col("n_clientes").eqNullSafe(F.col("clientes"))
        | ~F.col("n_saidas_exatas").eqNullSafe(F.col("exatas"))
    ))
    b = modelos.filter((F.col("variante") == "B_MEMORIA") & ~F.col("reuso"))
    sc = entradas["suporte"].filter("contexto_tem_suporte_candidato").withColumnRenamed(
        "n_saidas_exatas", "exatas_contexto"
    )
    cmp_b = b.join(sc, ["estado", "contexto_modelo"], "full")
    ms03_zero("ret_contagens_contextos", cmp_b.filter(
        ~F.col("n_observacoes").eqNullSafe(F.col("n_observacoes_treino"))
        | ~F.col("n_clientes").eqNullSafe(F.col("n_clientes_treino"))
        | ~F.col("n_saidas_exatas").eqNullSafe(F.col("exatas_contexto"))
    ))
    # Conjunto exato de snapshots esperado; não aceita novos casos nem duplicados.
    val = entradas["validacao_02c"]
    ms03_zero("ret_validacao_pares", val.groupBy(*MS03_CHAVE).agg(
        F.count("*").alias("n"), F.countDistinct("variante").alias("v")
    ).filter("n != 2 OR v != 2"))
    casos = base.filter(F.col("validacao_cliente") & F.col("elegivel_ajuste")).select(
        "cd_bv", "passo", "dur_min"
    ).withColumn("idade_seg", F.explode(F.array(*[
        F.lit(float(a)) for a in MS03_CFG["idades_seg"]
    ]))).filter(F.col("dur_min") > F.col("idade_seg")).select(*MS03_CHAVE)
    vistos = val.filter(F.col("variante") == MS_A).select(*MS03_CHAVE)
    ms03_zero("ret_snapshots_faltando", casos.join(vistos, MS03_CHAVE, "left_anti"))
    ms03_zero("ret_snapshots_excedentes", vistos.join(casos, MS03_CHAVE, "left_anti"))
    # Catálogo atual só foi aceito após comprovar que as linhas não mudaram.
    cfg = entradas["config"]
    ms03_zero("ret_catalogo_chaves", cfg.groupBy("estado").count().filter("count != 1"))
    esp = base.filter(F.col("estado").isNotNull()).select("estado", "funil").distinct()
    chk = esp.alias("b").join(cfg.select("estado", "funil").alias("c"), "estado", "left")
    ms03_zero("ret_catalogo_funil", chk.filter(
        ~F.col("b.funil").eqNullSafe(F.col("c.funil"))
    ))


def retomar_entradas() -> tuple[dict, dict]:
    # O pacote já concluído permite retomar mesmo quando o log antigo foi limpo.
    if spark.catalog.tableExists(RET_PACOTES):
        p_df = ret_filtrar(spark.table(RET_PACOTES), {
            "status": "CONCLUIDO_PACOTE", "contrato": RET_CONTRATO,
            "id_experimento": MS03_CFG["id_experimento"],
            "id_ajuste": MS03_CFG["id_ajuste"],
        })
        rows = ms03_pequeno(p_df, 1000, "Pacotes de retomada")
        if rows:
            ds = [r.asDict(recursive=True) for r in rows]
            preferidos = [r for r in ds if r["id_02c"] == MS03_CFG["id_02c_preferido"]]
            escolhido = max(preferidos or ds, key=lambda r: (r["fim_utc"], r["id_pacote"]))
            if sum(r["id_pacote"] == escolhido["id_pacote"] for r in ds) != 1:
                raise RuntimeError("Pacote de retomada duplicado.")
            meta = json.loads(escolhido["meta_json"])
            if ms_hash(meta) != escolhido["hash_pacote"]:
                raise RuntimeError("Manifesto do pacote foi alterado.")
            MS03_CFG["id_02c"] = meta["id_02c"]
            return ret_ler_pacote(meta), meta

    candidatos = ret_ler_corrente(MS03_CFG["tabela_execucoes_02c"], {
        "id_experimento": MS03_CFG["id_experimento"],
        "id_ajuste_fonte": MS03_CFG["id_ajuste"],
        "status": "CONCLUIDO_COMPARACAO_02C",
    })
    rows = [r.asDict(recursive=True) for r in ms03_pequeno(candidatos, 1000, "Manifestos 02C")]
    man = ret_escolher_manifesto(rows, MS03_CFG)
    MS03_CFG["id_02c"] = man["id_02c"]
    if man["versao_02c"] != "v2.3_02c_memoria_tempo_compartilhado_v1":
        raise RuntimeError("A 02C concluída usa outro contrato.")
    cfg02 = json.loads(man["config_json"])
    if ms_hash(cfg02) != man["hash_02c"]:
        raise RuntimeError("Configuração 02C alterada.")
    if (cfg02["config"]["id_experimento"] != MS03_CFG["id_experimento"]
            or cfg02["config"]["id_ajuste"] != MS03_CFG["id_ajuste"]):
        raise RuntimeError("Manifesto 02C e configuração apontam para treinamentos diferentes.")
    manfit = ms03_unico(ret_ler_corrente(cfg02["config"]["tabela_ajustes"], {
        "id_experimento": MS03_CFG["id_experimento"], "id_ajuste": MS03_CFG["id_ajuste"],
        "status": "CONCLUIDO_TREINO_VALIDACAO",
    }), "Treinamento aprovado")
    fitcfg = json.loads(manfit["config_json"])
    assinatura = {"versao": fitcfg["versao_ajuste"], "numerico": fitcfg["numerico"],
                  "politica_preparo": fitcfg["fonte_preparo"]["politica_sha256"],
                  "idades": fitcfg["idades_seg"]}
    hfit = hashlib.sha256(json.dumps(assinatura, sort_keys=True).encode()).hexdigest()
    if hfit != manfit["hash_ajuste"] or hfit != cfg02["hash_ajuste_fonte"]:
        raise RuntimeError("Assinatura do treinamento não confere com a 02C.")
    preparo = fitcfg["fonte_preparo"]
    if hashlib.sha256(preparo["politica_json"].encode()).hexdigest() != preparo["politica_sha256"]:
        raise RuntimeError("Política de preparo alterada.")
    spark.conf.set("spark.sql.session.timeZone", preparo["fuso_base"])
    if MS03_CFG["idades_seg"] != cfg02["idades_seg"]:
        raise RuntimeError("Marcos temporais diferentes da comparação aprovada.")
    print("Retomando experimento:", MS03_CFG["id_experimento"], flush=True)
    print("Treinamento:", MS03_CFG["id_ajuste"], "| 02C concluída:", man["id_02c"], flush=True)
    saidas02 = json.loads(man["tabelas_json"])
    base = ret_ler_corrente(preparo["tabela_base_v23"], {
        "id_experimento": MS03_CFG["id_experimento"]
    }, fitcfg["versao_delta_base_v23"])
    config = ret_ler_corrente(preparo["tabela_config_v22"], {},
                              preparo["versao_delta_config_v22"], False)
    cfgcols = ["estado", "funil", "etapa", "tipo_estado"]
    if "terminal_real" in config.columns:
        cfgcols.append("terminal_real")
    config = config.select(*cfgcols)
    mt = fitcfg["tabelas_saida"]["tabela_modelos"]
    modelos = ret_ler_corrente(mt, {
        "id_experimento": MS03_CFG["id_experimento"], "id_ajuste": MS03_CFG["id_ajuste"],
    }, cfg02["fontes_delta"][mt]).filter(F.col("variante").isin(MS_A, "B_MEMORIA"))
    b2 = ret_ler_corrente(saidas02["modelos"], {"id_02c": man["id_02c"]})
    val = ret_ler_corrente(saidas02["validacao"], {"id_02c": man["id_02c"]})
    val = val.filter(F.col("variante").isin(MS_A, MS_B2))
    suporte = ret_suporte_da_base(base, preparo).withColumn(
        "id_experimento", F.lit(MS03_CFG["id_experimento"]))
    entradas = {"base": base, "config": config, "modelos_ab": modelos,
                "modelos_b2": b2, "validacao_02c": val, "suporte": suporte}
    pacote = str(uuid.uuid5(uuid.NAMESPACE_URL, ":".join((
        RET_CONTRATO, MS03_CFG["id_experimento"], MS03_CFG["id_ajuste"], man["id_02c"],
    ))))
    meta = {"id_pacote": pacote, "contrato": RET_CONTRATO,
            "id_experimento": MS03_CFG["id_experimento"], "id_ajuste": MS03_CFG["id_ajuste"],
            "id_02c": man["id_02c"], "manifesto_02c": man, "manifesto_fit": manfit,
            "fit_config": fitcfg, "config_02c": cfg02, "saidas_02c": saidas02,
            "fontes_verificadas": dict(RET_FONTES)}
    ret_validar_entradas(entradas, meta)
    return entradas, meta


if MS03_CFG["autotestes"]:
    ms_autotestes_numericos()
    ms03_autoteste_spark()
# Valores da sessão anterior não participam da seleção.
entradas, ret_meta = retomar_entradas()
MS03_CFG["id_02c"] = ret_meta["id_02c"]
manifesto_02c = ret_meta["manifesto_02c"]
fit_manifesto = ret_meta["manifesto_fit"]
config_02c = ret_meta["config_02c"]
fit = ret_meta["fit_config"]
preparo = fit["fonte_preparo"]
saida_02c = ret_meta["saidas_02c"]
MS03_PISO = float(config_02c["logloss_piso"])
MS03_FONTES = dict(ret_meta["fontes_verificadas"])
if not 0 < MS03_PISO < 1:
    raise RuntimeError("Piso de log loss inválido.")

config_df = entradas["config"]
ms03_cols(config_df, {"estado", "funil", "etapa", "tipo_estado"}, "Config")
colcfg = ["estado", "funil", "etapa", "tipo_estado"]
if "terminal_real" in config_df.columns:
    colcfg.append("terminal_real")
config_df = config_df.select(*colcfg)
ms03_zero("config_unica", config_df.groupBy("estado").count().filter("count != 1"))
config_rows = ms03_pequeno(config_df, 1000, "Catálogo de estados")
MS03_CONFIG = {r["estado"]: r.asDict() for r in config_rows}
if None in MS03_CONFIG:
    raise RuntimeError("Estado nulo no catálogo.")
# A base clean deste experimento não possui terminais reais.
if any(v.get("terminal_real") or v["tipo_estado"] in ("terminal", "silencio", "conversao")
       for v in MS03_CONFIG.values()):
    raise RuntimeError("Esta versão valida a cadeia clean sem terminais; revisar contrato.")
if not set(MS03_ALVOS) <= {v.get("funil") for v in MS03_CONFIG.values()}:
    raise RuntimeError("Whitelist não está integralmente presente no catálogo congelado.")
MS03_TECNICAS = set(fit["acoes_tecnicas"])
politica = json.loads(preparo["politica_json"])
if (politica["memoria"] != "ultimo_funil_nao_tecnico_observado_no_prefixo"
        or set(politica["acoes_tecnicas"]) != MS03_TECNICAS):
    raise RuntimeError("Política de memória diferente do contrato.")

base = entradas["base"].filter(
    F.col("id_experimento") == MS03_CFG["id_experimento"]
)
ms03_cols(base, {
    "cd_bv", "passo", "ts_estado", "ts_corte_estado", "estado", "destino", "dur_min",
    "tipo_censura", "elegivel_ajuste", "validacao_cliente", "barreira_memoria",
    "contexto_negocio", "contexto_modelo", "origem_tecnica", "funil",
    "passo_evidencia_memoria", "ts_evidencia_memoria", "politica_sha256",
}, "Base V2.3")
ms03_zero("politica_base", base.filter(
    ~F.col("politica_sha256").eqNullSafe(F.lit(preparo["politica_sha256"]))
))
ms03_zero("cliente_nos_dois_splits", base.groupBy("cd_bv").agg(
    F.countDistinct("validacao_cliente").alias("n")
).filter("n != 1"))
ms03_zero("base_cliente_passo_unico", base.groupBy("cd_bv", "passo").count().filter("count != 1"))
ms03_zero("memoria_prefixo", base.filter(
    (F.col("passo_evidencia_memoria") > F.col("passo"))
    | (F.col("ts_evidencia_memoria") > F.col("ts_estado"))
))
holdout = ms03_cache(base.filter("validacao_cliente").select(
    "cd_bv", "passo", "ts_estado", "ts_corte_estado", "estado", "destino", "dur_min",
    "tipo_censura", "elegivel_ajuste", "barreira_memoria", "contexto_negocio",
    "contexto_modelo", "origem_tecnica", "funil",
))

suporte_df = entradas["suporte"].filter(
    F.col("id_experimento") == MS03_CFG["id_experimento"]
).select("estado", "contexto_modelo", "status_suporte")
ms03_zero("suporte_unico", suporte_df.groupBy("estado", "contexto_modelo").count().filter("count != 1"))
MS03_SUPORTE = {(r.estado, r.contexto_modelo): r.status_suporte for r in ms03_pequeno(
    suporte_df, MS03_CFG["max_modelos_driver"], "Suporte"
)}

mt = fit["tabelas_saida"]["tabela_modelos"]
originais = entradas["modelos_ab"].filter(
    (F.col("id_experimento") == MS03_CFG["id_experimento"])
    & (F.col("id_ajuste") == MS03_CFG["id_ajuste"])
    & F.col("variante").isin(MS_A, "B_MEMORIA")
)
ms03_zero("hash_modelos_origem", originais.filter(
    ~F.col("hash_ajuste").eqNullSafe(F.lit(fit_manifesto["hash_ajuste"]))
))
originais = originais.filter((F.col("status_modelo") == "AJUSTADO") & ~F.col("reuso"))
ms03_zero("modelos_origem_unicos", originais.groupBy(
    "variante", "estado", "contexto_modelo"
).count().filter("count != 1"))
MS03_A, MS03_B_ORIGINAL = {}, {}
for row in ms03_pequeno(originais, MS03_CFG["max_modelos_driver"], "Modelos A/B"):
    m = json.loads(row["modelo_json"])
    ms_validar_modelo(m)
    if row["variante"] == MS_A:
        if row["nivel"] != "ORIGEM" or row["contexto_modelo"] != MS_BASE:
            raise RuntimeError("A fora do nível ORIGEM.")
        MS03_A[row["estado"]] = m
    else:
        if row["nivel"] != "CONTEXTO":
            raise RuntimeError("B não reutilizado fora do nível CONTEXTO.")
        MS03_B_ORIGINAL[(row["estado"], row["contexto_modelo"])] = m

b2_df = entradas["modelos_b2"].filter(F.col("id_02c") == MS03_CFG["id_02c"])
ms03_zero("hash_modelos_b2", b2_df.filter(
    ~F.col("hash_02c").eqNullSafe(F.lit(manifesto_02c["hash_02c"]))
))
ms03_zero("modelos_b2_unicos", b2_df.groupBy("estado", "contexto_modelo").count().filter("count != 1"))
MS03_B2 = {}
for row in ms03_pequeno(b2_df, MS03_CFG["max_modelos_driver"], "Modelos B2"):
    s, c = row["estado"], row["contexto_modelo"]
    if row["variante"] != MS_B2 or row["status_modelo"] != "COMPOSTO_SEM_RETREINO":
        raise RuntimeError("Contrato da tabela B2 diferente.")
    a, b = MS03_A.get(s), MS03_B_ORIGINAL.get((s, c))
    if a is None or b is None:
        raise RuntimeError("B2 sem componentes fonte.")
    if ms_hash(a) != row["hash_tempo_fonte"] or ms_hash(b) != row["hash_prob_fonte"]:
        raise RuntimeError("Componentes de B2 não correspondem ao ajuste fixado.")
    m = json.loads(row["modelo_json"])
    ms_validar_modelo(m)
    mapa_a = {d: (a["mu_grupo"][g], a["sigma_grupo"][g])
              for d, g in zip(a["destinos"], a["grupo"])}
    if set(mapa_a) != set(m["destinos"]) or m["p_destino"] != b["p_destino"]:
        raise RuntimeError("B2 alterou catálogo ou p_B.")
    for d, g in zip(m["destinos"], m["grupo"]):
        if mapa_a[d] != (m["mu_grupo"][g], m["sigma_grupo"][g]):
            raise RuntimeError("Tempo de B2 não alinhado com A.")
    MS03_B2[(s, c)] = m
if not MS03_A or not MS03_B2:
    raise RuntimeError("Catálogo A/B2 vazio.")

validacao02c = ms03_cache(entradas["validacao_02c"].filter(
    (F.col("id_02c") == MS03_CFG["id_02c"])
    & F.col("variante").isin(MS_A, MS_B2)
))
ms03_cols(validacao02c, {
    "cd_bv", "passo", "idade_seg", "estado", "contexto_modelo", "origem_tecnica",
    "tipo_censura", "destino_real", "variante", "tem_previsao", "rota_modelo_original",
    "prob_destino_real", "top5_previsto", "hash_02c",
}, "Validação 02C")
ms03_zero("hash_validacao02c", validacao02c.filter(
    ~F.col("hash_02c").eqNullSafe(F.lit(manifesto_02c["hash_02c"]))
))
ms03_zero("pares_validacao02c", validacao02c.groupBy(*MS03_CHAVE).agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v")
).filter("n != 2 OR v != 2"))
print("03:", MS03_ID, "| 02C fonte:", MS03_CFG["id_02c"])
print("Corte histórico:", preparo["corte_estado_iso"], "| nenhum modelo será treinado.")


# Somente após conferir modelos A/B/B2 e os pares da validação.
if MS03_CFG["gravar"] and "entradas" not in ret_meta:
    ret_meta = ret_gravar_pacote(entradas, ret_meta)
print("Entradas conferidas. Iniciando o Multi-step histórico A × B2.", flush=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Construir os mesmos snapshots históricos e avaliar A × B2
# MAGIC Nenhum retreino. Censura não vira resultado negativo.
# MAGIC A memória acompanha cada trajetória e as massas continuam explícitas.

# COMMAND ----------

labels = ms03_cache(ms03_rotular(holdout, MS03_ALVOS, MS03_CFG["max_passos_macro"]))
ms03_zero("labels_unicos", labels.groupBy("cd_bv", "passo").count().filter("count != 1"))
entrada_a = validacao02c.filter(F.col("variante") == MS_A).select(
    *MS03_CHAVE, F.col("estado").alias("_estado_02c"),
    F.col("contexto_modelo").alias("_contexto_02c"),
    F.col("tipo_censura").alias("_tipo_02c"), F.col("destino_real").alias("_destino_02c"),
)
ms03_zero("snapshots_presentes_no_holdout", entrada_a.join(
    holdout.select("cd_bv", "passo"), ["cd_bv", "passo"], "left_anti"
))
snapshots = entrada_a.join(holdout, ["cd_bv", "passo"], "inner")
ms03_zero("snapshot_asof_identico", snapshots.filter(
    ~F.col("estado").eqNullSafe(F.col("_estado_02c"))
    | ~F.col("contexto_modelo").eqNullSafe(F.col("_contexto_02c"))
    | ~F.col("tipo_censura").eqNullSafe(F.col("_tipo_02c"))
    | ~F.col("destino").eqNullSafe(F.col("_destino_02c"))
    | ~F.col("elegivel_ajuste") | F.col("barreira_memoria")
    | (F.col("dur_min") <= F.col("idade_seg"))
))
snapshots = snapshots.drop("_estado_02c", "_contexto_02c", "_tipo_02c", "_destino_02c").join(
    labels, ["cd_bv", "passo"], "left"
).withColumn("memoria_grafo", F.coalesce("contexto_negocio", F.lit(MS_SEM_MEMORIA)))
snapshots = snapshots.withColumn("ts_snapshot", F.expr(
    "timestamp_micros(unix_micros(ts_estado) + cast(round(idade_seg * 1000000) as bigint))"
)).withColumn("id_snapshot", F.sha2(F.to_json(F.struct(*MS03_CHAVE)), 256))
ms03_zero("labels_e_tempos", snapshots.filter(
    F.col("status_label").isNull() | (F.col("ts_snapshot") >= F.col("ts_corte_estado"))
    | ((F.col("status_label") == "ALVO_OBSERVADO") & (
        F.col("ts_acao_real").isNull() | (F.col("ts_acao_real") <= F.col("ts_snapshot"))
    ))
))
ms03_zero("snapshots_unicos", snapshots.groupBy(*MS03_CHAVE).count().filter("count != 1"))
if not snapshots.limit(1).count():
    raise RuntimeError("Nenhum snapshot do holdout disponível.")
MS03_KKEY = ["estado", "memoria_grafo", "idade_seg"]
MS03_CONFIG_EXEC = {
    "versao": MS03_VERSAO, "config": MS03_CFG, "fontes_verificadas": dict(MS03_FONTES), "pacote_entradas": ret_meta["id_pacote"],
    "hash_02c_fonte": manifesto_02c["hash_02c"], "alvos": MS03_ALVOS,
    "acoes_tecnicas": sorted(MS03_TECNICAS), "logloss_piso": MS03_PISO,
    "horizonte": "primeira_entrada_alvo_em_ate_5_mudancas_macro",
    "mesmo_funil": "custo_zero_sem_alvo; reentrada_apos_outro_funil_valida",
    "memoria": "ultimo_funil_nao_tecnico; tecnico_e_atendimento_preservam",
    "snapshot": "mesmos_cd_bv_passo_idade_da_02c; futuro_somente_no_label",
    "censura": "antes_de_5_mudancas_sem_alvo_nao_recebe_label_negativo",
    "estatistica": "media_por_cliente; bootstrap_pareado; IC95_exploratorio",
    "efeito_temporal": "somente_1a_transicao_usa_idade_atual; demais_idade_zero",
    "publicavel": False, "sem_retreino": True, "uso": "EXPERIMENTO_HISTORICO",
}
MS03_HASH = ms_hash(MS03_CONFIG_EXEC)
for nome, tabela in MS03_SAIDAS.items():
    ms03_sql(tabela)
    if tabela in MS03_FONTES:
        raise RuntimeError("Uma saída coincide com fonte.")
snapshots = ms03_guardar(snapshots, "snapshots")
ms03_mostrar("V23_MS03_01_LABELS", ms03_segmentos(snapshots).groupBy(
    "segmento", "idade_seg", "status_label"
).agg(F.count("*").alias("n_snapshots"), F.countDistinct("cd_bv").alias("n_clientes")),
    ["segmento", "idade_seg", "status_label"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1. Grafos A/B2 e kernels por estado + memória + idade
# MAGIC Soma analítica das transições internas. Não cria uma matriz macro média
# MAGIC que apaga a memória. Ciclos fechados são retenção interna; estado sem
# MAGIC modelo é massa não resolvida; nenhuma dessas massas é renormalizada.
# MAGIC Só catálogos pequenos vão ao driver. Os eventos continuam em Spark.

# COMMAND ----------

kernel_keys = ms03_pequeno(snapshots.select(*MS03_KKEY).distinct(),
                          MS03_CFG["max_kernels_driver"] // 2, "Kernels por variante")
ms_kernel_rows, ms_graph_rows = [], []
MS03_KERNEL_SCHEMA = (
    "variante string, estado string, memoria_grafo string, idade_seg double, "
    "tem_previsao boolean, rota_modelo string, "
    "raw_top5 array<struct<ranking:int,estado:string,score:double>>, "
    "util_top5 array<struct<ranking:int,acao:string,score:double,passo_macro_maior_contribuicao:int>>, "
    "raw_probs map<string,double>, util_probs map<string,double>, util_ranks map<string,int>, "
    "soma_quadrados_util double, massa_acao_util double, massa_sem_modelo double, "
    "massa_terminal double, massa_retencao_interna double, massa_limite_macro double, "
    "prob_sem_alvo_modelada double, alerta_limite boolean, extrapolacao_tempo boolean, "
    "erro_massa double"
)
for variante in (MS_A, MS_B2):
    motor = MotorMacroMemoria(
        variante, MS03_CONFIG, MS03_TECNICAS, MS03_A, MS03_B2, MS03_SUPORTE,
        MS03_ALVOS, MS03_CFG["max_passos_macro"], MS03_CFG["max_nos_expandidos"],
        MS03_CFG["max_arestas"], MS03_CFG["tolerancia_massa"],
    )
    if motor.exatas.size + motor.massas.size > MS03_CFG["max_celulas_numericas"]:
        raise RuntimeError("Proteção de memória numérica excedida.")
    ms_graph_rows.append(motor.resumo)
    for key in kernel_keys:
        r = motor.prever(key.estado, key.memoria_grafo, float(key.idade_seg))
        campos = (variante, key.estado, key.memoria_grafo, float(key.idade_seg),
                  bool(r["tem_previsao"]), r["rota"])
        if not r["tem_previsao"]:
            ms_kernel_rows.append(campos + ([], [], {}, {}, {}, None, None, None,
                                           None, None, None, None, False, False, None))
        else:
            ms_kernel_rows.append(campos + (
                r["raw_top5"], r["util_top5"], r["raw_probs"], r["util_probs"], r["util_ranks"],
                r["soma_quadrados_util"], r["massa_acao_util"], r["massa_sem_modelo"],
                r["massa_terminal"], r["massa_retencao_interna"], r["massa_limite_macro"],
                r["prob_sem_alvo_modelada"], r["alerta_limite"], r["extrapolacao_tempo"],
                r["erro_massa"],
            ))
    del motor
kernels = spark.createDataFrame(ms_kernel_rows, MS03_KERNEL_SCHEMA)
ms03_zero("kernels_unicos", kernels.groupBy("variante", *MS03_KKEY).count().filter("count != 1"))
ms03_zero("massa_kernels", kernels.filter(F.col("tem_previsao") & (
    (F.abs(F.col("massa_acao_util") + F.col("massa_sem_modelo")
           + F.col("prob_sem_alvo_modelada") - 1) > MS03_CFG["tolerancia_massa"])
    | F.isnan("massa_acao_util")
)))
kernels = ms03_guardar(kernels, "kernels")
ms03_mostrar("V23_MS03_02_GRAFOS", spark.createDataFrame(ms_graph_rows), ["variante"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2. Reproduzir RAW e avaliar a ação útil
# MAGIC Top1/5/MRR: somente snapshots com primeira ação-alvo observada.
# MAGIC Brier por ações: soma de 15 erros quadráticos (um indicador por alvo),
# MAGIC incluindo horizontes observados sem alvo. Não requer renormalizar alvos.
# MAGIC Log loss conservador: usa massa modelada do desfecho real; perda sem
# MAGIC modelo NÃO é transformada em 'nenhuma ação'. Um limite otimista que
# MAGIC aloca a massa perdida ao rótulo real também é reportado como diagnóstico.
# MAGIC Métricas não são diretamente comparáveis ao Brier/Top5 RAW da Parte 02.

# COMMAND ----------

kernel_cols = [c for c in kernels.columns if c not in (
    "id_experimento", "id_ajuste_fonte", "id_02c_fonte", "id_ms03", "versao_ms03", "hash_ms03"
)]
snapshot_cols = [c for c in snapshots.columns if c not in (
    "id_experimento", "id_ajuste_fonte", "id_02c_fonte", "id_ms03", "versao_ms03", "hash_ms03"
)]
pred = snapshots.select(*snapshot_cols).join(kernels.select(*kernel_cols), MS03_KKEY, "inner")
ms03_zero("duas_variantes_por_snapshot", pred.groupBy(*MS03_CHAVE).agg(
    F.count("*").alias("n"), F.countDistinct("variante").alias("v")
).filter("n != 2 OR v != 2"))
original = validacao02c.select(
    *MS03_CHAVE, "variante", F.col("tem_previsao").alias("_tem_original"),
    F.col("rota_modelo_original").alias("_rota_original"),
    F.col("prob_destino_real").alias("_raw_prob_original"),
    F.col("top5_previsto").alias("_raw_top5_original"),
)
rep = pred.join(original, MS03_CHAVE + ["variante"], "inner").withColumn(
    "_raw_prob", F.when(F.col("tem_previsao") & (F.col("tipo_censura") == "exata"),
                        F.coalesce(F.element_at("raw_probs", F.col("destino")), F.lit(0.0)))
).withColumn("_raw_top5", F.expr("transform(raw_top5, x -> x.estado)"))
ms03_zero("reproducao_RAW02c", rep.filter(
    ~F.col("tem_previsao").eqNullSafe(F.col("_tem_original"))
    | ~F.col("rota_modelo").eqNullSafe(F.col("_rota_original"))
    | ~F.col("_raw_top5").eqNullSafe(F.col("_raw_top5_original"))
    | (F.col("_raw_prob").isNull() != F.col("_raw_prob_original").isNull())
    | (F.abs(F.col("_raw_prob") - F.col("_raw_prob_original")) > MS03_CFG["tolerancia_prob"])
))
conhecido = F.col("status_label") != "CENSURADO_ANTES_HORIZONTE"
positivo = F.col("status_label") == "ALVO_OBSERVADO"
medir = F.col("tem_previsao") & conhecido
pred = pred.withColumn("y_algum_alvo", F.when(conhecido, positivo.cast("double")))
pred = pred.withColumn("prob_acao_real", F.when(positivo & F.col("tem_previsao"),
    F.coalesce(F.element_at("util_probs", F.col("acao_real")), F.lit(0.0))))
pred = pred.withColumn("prob_desfecho_real", F.when(medir,
    F.when(positivo, F.col("prob_acao_real")).otherwise(F.col("prob_sem_alvo_modelada"))))
pred = pred.withColumn("_rank_real", F.element_at("util_ranks", F.col("acao_real")))
pred = pred.withColumn("top1", F.when(medir & positivo,
    ((F.col("_rank_real") == 1) & (F.col("prob_acao_real") > 0)).cast("double")))
pred = pred.withColumn("top5", F.when(medir & positivo,
    ((F.col("_rank_real") <= 5) & (F.col("prob_acao_real") > 0)).cast("double")))
pred = pred.withColumn("mrr", F.when(medir & positivo,
    F.when(F.col("prob_acao_real") > 0, 1.0 / F.col("_rank_real")).otherwise(0.0)))
pred = pred.withColumn("brier_acoes", F.when(medir,
    F.col("soma_quadrados_util") + F.when(positivo, 1.0 - 2.0 * F.col("prob_acao_real"))
    .otherwise(0.0)))
pred = pred.withColumn("brier_algum_alvo", F.when(medir,
    F.pow(F.col("massa_acao_util") - F.col("y_algum_alvo"), 2)))
pred = pred.withColumn("logloss_conservador", F.when(medir,
    -F.log(F.greatest(F.col("prob_desfecho_real"), F.lit(MS03_PISO)))))
pred = pred.withColumn("logloss_limite_otimista", F.when(medir,
    -F.log(F.greatest(F.least(F.lit(1.0), F.col("prob_desfecho_real")
                             + F.col("massa_sem_modelo")), F.lit(MS03_PISO)))))
pred = pred.withColumn("nll_clip_aplicado", F.when(medir,
    F.col("prob_desfecho_real") < MS03_PISO))
pred = pred.withColumn("status_previsao", F.when(~F.col("tem_previsao"), "SEM_MODELO_ORIGEM")
    .when(F.size("util_top5") == 0, "SEM_ALVO_ALCANCAVEL_MODELO")
    .otherwise("PREVISAO_MULTISTEP_HISTORICA"))
pred = pred.withColumn("max_passos_macro", F.lit(MS03_CFG["max_passos_macro"]))
pred = pred.drop("_rank_real", "raw_probs", "util_ranks")
ms03_zero("metricas_finitas", pred.filter(medir & (
    F.isnan("brier_acoes") | F.isnan("logloss_conservador")
    | (F.col("brier_acoes") < -1e-10)
)))
pred = ms03_guardar(pred, "predicoes")
ms03_mostrar("V23_MS03_03_INTEGRIDADE", spark.createDataFrame(
    MS03_CHECKS, "verificacao string, n_divergencias long, status string"
))
ms03_mostrar("V23_MS03_04_COBERTURA", ms03_segmentos(pred).groupBy(
    "segmento", "variante", "idade_seg", "status_label", "status_previsao"
).agg(F.count("*").alias("n_snapshots"), F.countDistinct("cd_bv").alias("n_clientes")),
    ["segmento", "idade_seg", "variante", "status_label"])
MS03_METRICAS = ["top1", "top5", "mrr", "brier_acoes", "brier_algum_alvo", "logloss_conservador"]
metricas_evento = ms03_segmentos(pred).filter(medir).groupBy(
    "segmento", "variante", "idade_seg"
).agg(F.count("*").alias("n_desfechos_conhecidos"),
      F.sum(positivo.cast("long")).alias("n_alvos_observados"),
      *[F.avg(c).alias(c) for c in MS03_METRICAS],
      F.sum(F.col("nll_clip_aplicado").cast("long")).alias("n_logloss_clip"))
ms03_mostrar("V23_MS03_05_METRICAS", metricas_evento.filter("segmento = 'ORIGENS_TECNICAS'")
    .drop("segmento"), ["idade_seg", "variante"])
MS03_RELATORIOS["V23_MS03_05_METRICAS_TODAS"] = metricas_evento

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3. Pares por cliente e diagnóstico de concentração
# MAGIC Mesmos snapshots com modelo em A/B2. Clientes sem histórico suficiente
# MAGIC para conhecer o label continuam na cobertura, mas não viram negativos.
# MAGIC Mais listas diferentes é diagnóstico; não é critério de ganho preditivo.

# COMMAND ----------

wcaso = Window.partitionBy(*MS03_CHAVE)
pares = pred.withColumn("_n_previsoes", F.sum(F.col("tem_previsao").cast("int")).over(wcaso))
pares = ms03_segmentos(pares.filter((F.col("_n_previsoes") == 2) & conhecido))
pares = pares.withColumn("coorte", F.explode(F.when(positivo,
    F.array(F.lit("TODOS_DESFECHOS_CONHECIDOS"), F.lit("SOMENTE_ALVO_OBSERVADO"))
).otherwise(F.array(F.lit("TODOS_DESFECHOS_CONHECIDOS")))))
por_cliente = pares.groupBy("segmento", "idade_seg", "coorte", "cd_bv", "variante").agg(
    F.count("*").alias("n_casos"), *[F.avg(c).alias(c) for c in MS03_METRICAS]
)
pc_rows = ms03_pequeno(por_cliente, MS03_CFG["max_agregados_driver"], "Agregados por cliente")
if not pc_rows:
    raise RuntimeError("Nenhum desfecho conhecido com previsão comum; consultar cobertura.")
pares_rows = ms_bootstrap(pd.DataFrame([r.asDict() for r in pc_rows]), MS03_METRICAS,
    MS03_CFG["bootstrap_replicas"], MS03_CFG["semente"], MS03_CFG["max_celulas_bootstrap"])
comparacoes = spark.createDataFrame(pares_rows, (
    "segmento string, idade_seg double, coorte string, metrica string, n_clientes long, n_casos long, "
    "media_a double, media_b2 double, ganho_medio double, ganho_ic95_lo double, ganho_ic95_hi double"
))
comparacoes = ms03_guardar(comparacoes, "comparacoes")
# Cada métrica é impressa na coorte relevante, sem duplicar Top1/Top5.
ms03_mostrar("V23_MS03_06_PARES_B2_VS_A", comparacoes.filter(
    (F.col("segmento") == "ORIGENS_TECNICAS") & (
        ((F.col("coorte") == "SOMENTE_ALVO_OBSERVADO") & F.col("metrica").isin("top1", "top5", "mrr"))
        | ((F.col("coorte") == "TODOS_DESFECHOS_CONHECIDOS")
           & ~F.col("metrica").isin("top1", "top5", "mrr"))
    )
).select("idade_seg", "metrica", "coorte", "n_clientes", "n_casos", "media_a", "media_b2",
         "ganho_medio", "ganho_ic95_lo", "ganho_ic95_hi"), ["idade_seg", "metrica"])
MS03_RELATORIOS["V23_MS03_06_PARES_COMPLETOS"] = comparacoes.drop(
    "id_experimento", "id_ajuste_fonte", "id_02c_fonte", "id_ms03", "versao_ms03", "hash_ms03"
)

p = pred.filter("tem_previsao").withColumn("_acao_top1", F.expr(
    "element_at(map_from_entries(transform(util_top5, x -> struct(x.ranking, x.acao))), 1)"
))
ms03_mostrar("V23_MS03_07_TOP1_UTIL", p.groupBy("variante", "idade_seg", "_acao_top1").agg(
    F.count("*").alias("n_snapshots"), F.countDistinct("cd_bv").alias("n_clientes")
).withColumnRenamed("_acao_top1", "acao_top1"), ["idade_seg", "variante", F.desc("n_snapshots")])
p = p.withColumn("assinatura_top5", F.expr(
    "to_json(transform(util_top5, x -> struct(x.ranking as ranking, x.acao as acao)))"
))
assinaturas = p.groupBy("variante", "idade_seg", "assinatura_top5").count()
concentracao = assinaturas.groupBy("variante", "idade_seg").agg(
    F.count("*").alias("n_listas_distintas"), F.max("count").alias("n_lista_dominante"),
    F.sum("count").alias("n_snapshots_previstos"),
).withColumn("fracao_lista_dominante", F.col("n_lista_dominante") / F.col("n_snapshots_previstos"))
ms03_mostrar("V23_MS03_07_CONCENTRACAO", concentracao, ["idade_seg", "variante"])
MS03_RELATORIOS["V23_MS03_07_ASSINATURAS"] = assinaturas
massas = ms03_segmentos(pred).filter("tem_previsao").groupBy("segmento", "variante", "idade_seg").agg(
    *[F.avg(c).alias(c) for c in ("massa_acao_util", "massa_sem_modelo", "massa_terminal",
                               "massa_retencao_interna", "massa_limite_macro")],
    F.max("erro_massa").alias("max_erro_massa"),
)
ms03_mostrar("V23_MS03_08_MASSAS", massas.filter("segmento = 'ORIGENS_TECNICAS'")
    .drop("segmento"), ["idade_seg", "variante"])
MS03_RELATORIOS["V23_MS03_08_MASSAS_TODAS"] = massas
# Recuperação por alvo com o mesmo denominador de casos positivos observados.
por_acao = ms03_segmentos(pred).filter(medir & positivo).groupBy(
    "segmento", "variante", "idade_seg", "acao_real"
).agg(F.count("*").alias("n_casos"), F.countDistinct("cd_bv").alias("n_clientes"),
      F.avg("top1").alias("top1"), F.avg("top5").alias("top5"),
      F.avg("prob_acao_real").alias("score_real_medio"))
MS03_RELATORIOS["V23_MS03_08_POR_ACAO"] = por_acao

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Saída LONG histórica e encerramento
# MAGIC Até 5 linhas por (cd_bv, snapshot, variante), NÃO 5 por cliente no total:
# MAGIC há vários snapshots históricos. RAW e útil na mesma posição têm rankings
# MAGIC independentes. Esta tabela é para comparação, não disparo de campanha.

# COMMAND ----------

negocio = pred.withColumn("_raw_map", F.expr(
    "map_from_entries(transform(raw_top5, x -> struct(x.ranking, x)))"
)).withColumn("_util_map", F.expr(
    "map_from_entries(transform(util_top5, x -> struct(x.ranking, x)))"
)).withColumn("_n", F.greatest(F.size("raw_top5"), F.size("util_top5")))
negocio = negocio.withColumn("_rank", F.explode(F.sequence(F.lit(1), F.greatest("_n", F.lit(1)))))
negocio = negocio.withColumn("_raw", F.element_at("_raw_map", F.col("_rank")))
negocio = negocio.withColumn("_util", F.element_at("_util_map", F.col("_rank")))
negocio = negocio.select(
    F.to_date("ts_snapshot").alias("data_referencia"), "ts_snapshot", "id_snapshot", "cd_bv",
    "passo", "idade_seg", "variante", F.coalesce("funil", "estado").alias("acao_atual"),
    "contexto_negocio", F.when(F.col("_n") > 0, F.col("_rank")).alias("ranking"),
    F.col("_raw.estado").alias("proxima_acao_raw"), F.col("_raw.score").alias("score_raw"),
    F.col("_util.acao").alias("proxima_acao_util"), F.col("_util.score").alias("score_util"),
    F.col("_util.passo_macro_maior_contribuicao").alias("passo_macro_maior_contribuicao"),
    "status_previsao", "status_label", "acao_real", "passo_macro_real",
    F.lit("EXPERIMENTO_HISTORICO_NAO_DISPARAR").alias("uso_saida"),
)
ms03_zero("long_unico", negocio.groupBy("id_snapshot", "variante", "ranking").count().filter("count != 1"))
negocio = ms03_guardar(negocio, "negocio_historico")
ms03_mostrar("V23_MS03_09_SAIDA_LONG", negocio.groupBy("variante").agg(
    F.count("*").alias("n_linhas"), F.countDistinct("cd_bv").alias("n_clientes"),
    F.countDistinct("id_snapshot").alias("n_snapshots"),
))
resumos = None
for nome, df in MS03_RELATORIOS.items():
    lote = df.select(F.lit(nome).alias("relatorio"), F.to_json(F.struct(
        *[F.col(c) for c in df.columns]), {"ignoreNullFields": "false"}).alias("dados_json"))
    resumos = lote if resumos is None else resumos.unionByName(lote)
ms03_guardar(resumos, "resumos")
if MS03_CFG["gravar"]:
    fim = spark.createDataFrame([(
        "CONCLUIDO_MULTISTEP_HISTORICO_03", False, False, MS03_INICIO,
        datetime.now(timezone.utc).isoformat(),
        json.dumps(MS03_CONFIG_EXEC, sort_keys=True, ensure_ascii=False),
        json.dumps(MS03_SAIDAS, sort_keys=True),
    )], "status string, publicavel boolean, houve_retreino boolean, inicio_utc string, "
        "fim_utc string, config_json string, tabelas_json string")
    ms03_guardar(fim, "execucoes")
print("\nV23_MS03_10_CONCLUSAO")
print("id_ms03:", MS03_ID)
print("Saídas:", json.dumps(MS03_SAIDAS, indent=2))
print("Nenhum modelo retreinado; nenhuma tabela anterior sobrescrita.")
print("Validação histórica concluída. A tabela de negócio anterior permanece intacta.")
print("Consuma somente ID com CONCLUIDO_MULTISTEP_HISTORICO_03 no manifesto.")
for df in MS03_CACHES:
    df.unpersist()
