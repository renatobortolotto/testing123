# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — NBA V2.3: referência, treinamento e validação
# MAGIC
# MAGIC **Homologação, não promoção automática.** Dois modos explícitos:
# MAGIC
# MAGIC - `PRESERVAR_REFERENCIA` (padrão): lê o pacote físico aprovado dos 17 alvos,
# MAGIC   copia suas entradas, recalcula a validação histórica e exige equivalência dos
# MAGIC   kernels, rótulos e métricas com a referência persistida. **Não retreina.**
# MAGIC - `REAJUSTAR_CONGELADO`: exige confirmação explícita; usa a MESMA base preparada,
# MAGIC   mesmos clientes, splits, pesos, corte e hiperparâmetros do pacote. Reexecuta
# MAGIC   o ajuste original A/B/C/D, compõe B2 e valida uma candidata separada.
# MAGIC   C/D são diagnósticos preservados do código original; não são o motor promovido.
# MAGIC
# MAGIC Não reconstrói eventos brutos, não amplia a amostra e não adiciona features.
# MAGIC A entrada de treino é a base preparada congelada, parte do pacote já aprovado.
# MAGIC Os cálculos numéricos dos arquivos anteriores foram conservados; a consolidação
# MAGIC altera configuração e encadeamento, não a família estatística.
# MAGIC
# MAGIC Importe em notebook NOVO. As saídas usam prefixo `nba_sm_v23_consolidada`;
# MAGIC nenhuma view atual de negócio é substituída. A equivalência Spark/Delta ainda
# MAGIC precisa ser executada no cluster. Não execute apenas células intermediárias.

# COMMAND ----------

import copy
import json
import platform
import uuid
from datetime import datetime, timezone

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

CONS_CFG = {
    "modo": "PRESERVAR_REFERENCIA",
    "confirmar_reajuste": False,
    "id_ms03_referencia": "92a92d57-ce93-487c-bdf1-369e3dbd542f",
    "tabela_execucoes_referencia": (
        "ctg_dsti.renato_nba.nba_sm_v23_estavel_execucoes_hml"
    ),
    "tabela_pacotes_referencia": (
        "ctg_dsti.renato_nba.nba_sm_v23_estavel_pacotes_hml"
    ),
    "prefixo_saida": "ctg_dsti.renato_nba.nba_sm_v23_consolidada",
    "tolerancia_reproducao": 1e-9,
}
ALVOS_APROVADOS = ['debito_automatico',
 'emprestimos_pessoal_cartao',
 'opf_iniciador_pagamentos',
 'opf_recepcao_dados',
 'pagamentos_boleto',
 'pix_cadastro_chave',
 'pix_excluir_chave',
 'pix_trazer_chave',
 'portabilidade_salario',
 'tag_veicular_alterar_placa',
 'tag_veicular_ativar',
 'tag_veicular_bloquear',
 'tag_veicular_cancelar',
 'tag_veicular_contestacao',
 'tag_veicular_desbloquear',
 'tag_veicular_segunda_via',
 'tag_veicular_solicitar']
CONS_ID = str(uuid.uuid4())
CONS_INICIO = datetime.now(timezone.utc).isoformat()
CONS_CFG["modo"] = str(CONS_CFG["modo"]).strip().upper()
if CONS_CFG["modo"] not in {"PRESERVAR_REFERENCIA", "REAJUSTAR_CONGELADO"}:
    raise ValueError("Modo de consolidação inválido.")
if CONS_CFG["modo"] == "REAJUSTAR_CONGELADO" and not CONS_CFG["confirmar_reajuste"]:
    raise RuntimeError("Reajuste exige confirmar_reajuste=True; não é automático.")
if "spark" not in globals():
    raise RuntimeError("Execute no Databricks com sessão Spark.")
if CONS_CFG["prefixo_saida"] != "ctg_dsti.renato_nba.nba_sm_v23_consolidada":
    raise RuntimeError("Nesta homologação, conserve o prefixo isolado de saída.")
print("Consolidação:", CONS_ID, "| modo:", CONS_CFG["modo"])

# COMMAND ----------

import uuid
from datetime import datetime, timezone

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

# Parâmetros históricos preservados. IDs fonte serão preenchidos pelo manifesto.
MS03_CFG = {
    "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
    "id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
    "id_02c_preferido": "53d097ef-c176-485b-9848-4fa06b653064",
    "tabela_execucoes_02c": "ctg_dsti.renato_nba.nba_sm_v23_02c_execucoes_hml",
    "prefixo_saida": CONS_CFG["prefixo_saida"] + "_ms",
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
MS03_ALVOS = list(ALVOS_APROVADOS)
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
# MAGIC ## 1. Núcleo histórico preservado
# MAGIC
# MAGIC Motor de primeira chegada, probabilidades temporais, rótulos, censura e métricas.
# MAGIC Estas funções são as mesmas do notebook histórico anteriormente enviado.

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
# MAGIC ## 2. Carregar exclusivamente a referência aprovada
# MAGIC
# MAGIC Usa o pacote físico por ID, não versões expiradas do treino.
# MAGIC Confere assinatura, contagem, política, corte, splits e composição A/B2.
# MAGIC Nenhum modelo ou execução alternativa é escolhido automaticamente.

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

# COMMAND ----------

def cons_ler_atual(tabela: str) -> DataFrame:
    ms03_sql(tabela)
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(f"Fonte necessária não encontrada: {tabela}")
    return spark.table(tabela)


if MS03_CFG["autotestes"]:
    ms_autotestes_numericos()
    ms03_autoteste_spark()

CONS_REF_HIST = ms03_unico(
    cons_ler_atual(CONS_CFG["tabela_execucoes_referencia"]).filter(
        (F.col("id_ms03") == CONS_CFG["id_ms03_referencia"])
        & (F.col("status") == "CONCLUIDO_MULTISTEP_HISTORICO_03")
    ),
    "Execução histórica de referência",
)
CONS_REF_HIST_CFG = json.loads(CONS_REF_HIST["config_json"])
if ms_hash(CONS_REF_HIST_CFG) != CONS_REF_HIST["hash_ms03"]:
    raise RuntimeError("Hash do manifesto de referência divergente.")
if sorted(CONS_REF_HIST_CFG["alvos"]) != ALVOS_APROVADOS:
    raise RuntimeError("A referência não usa exatamente os 17 alvos aprovados.")
if CONS_REF_HIST_CFG["config"]["max_passos_macro"] != 5:
    raise RuntimeError("Horizonte da referência diferente de cinco mudanças.")
if CONS_REF_HIST_CFG["config"]["idades_seg"] != MS03_CFG["idades_seg"]:
    raise RuntimeError("Marcos de idade diferentes da referência.")
CONS_REF_TABELAS = json.loads(CONS_REF_HIST["tabelas_json"])
CONS_REF_PACOTE = ms03_unico(
    cons_ler_atual(CONS_CFG["tabela_pacotes_referencia"]).filter(
        (F.col("id_pacote") == CONS_REF_HIST_CFG["pacote_entradas"])
        & (F.col("status") == "CONCLUIDO_PACOTE")
    ),
    "Pacote físico de referência",
)
CONS_REF_META = json.loads(CONS_REF_PACOTE["meta_json"])
if CONS_REF_PACOTE["contrato"] != RET_CONTRATO:
    raise RuntimeError("Contrato de pacote incompatível.")
if ms_hash(CONS_REF_META) != CONS_REF_PACOTE["hash_pacote"]:
    raise RuntimeError("Assinatura do pacote divergente.")
for pacote_campo, hist_campo in (
    ("id_experimento", "id_experimento"),
    ("id_ajuste", "id_ajuste_fonte"),
    ("id_02c", "id_02c_fonte"),
):
    if CONS_REF_META[pacote_campo] != CONS_REF_HIST[hist_campo]:
        raise RuntimeError(f"IDs da referência incompatíveis: {pacote_campo}")
for campo in ("id_experimento", "id_ajuste", "id_02c"):
    MS03_CFG[campo] = CONS_REF_META[campo]
MS03_CFG["id_02c_preferido"] = CONS_REF_META["id_02c"]
CONS_REF_ENTRADAS = ret_ler_pacote(CONS_REF_META)
ret_validar_entradas(CONS_REF_ENTRADAS, CONS_REF_META)
print("Referência conferida:", CONS_CFG["id_ms03_referencia"])
print("Pacote:", CONS_REF_META["id_pacote"])
print("Clientes e amostra não serão alterados nesta consolidação.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reajuste opcional da mesma base congelada
# MAGIC
# MAGIC **Ignorado integralmente no modo padrão.** Esta seção não amplia a amostra e não
# MAGIC recria uma base a partir das fontes atuais. No modo de reajuste, primeiro produz
# MAGIC cópias físicas isoladas da base preparada, suporte e catálogo. Mantém a otimização
# MAGIC original A/B/C/D e a composição B2. O resultado é candidata, nunca promoção.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    CONS_FIT_PREFIX = CONS_CFG["prefixo_saida"] + "_fit"
    CONS_INPUT_PREFIX = CONS_CFG["prefixo_saida"] + "_entrada_" + CONS_ID.replace("-", "")
    CONS_PREPARO = copy.deepcopy(CONS_REF_META["fit_config"]["fonte_preparo"])
    CONS_INPUT_TABLES = {}
    for tipo in ("base", "suporte", "config"):
        destino = CONS_INPUT_PREFIX + "_" + tipo
        fonte = CONS_REF_ENTRADAS[tipo]
        if spark.catalog.tableExists(destino):
            raise RuntimeError("Cópia de entrada já existe; use uma execução nova.")
        fonte.write.format("delta").mode("errorifexists").saveAsTable(destino)
        copia = spark.table(destino)
        if fonte.exceptAll(copia).limit(1).count() or copia.exceptAll(fonte).limit(1).count():
            raise RuntimeError(f"Cópia de treino divergente: {tipo}")
        CONS_INPUT_TABLES[tipo] = destino
    CONS_PREPARO["tabela_base_v23"] = CONS_INPUT_TABLES["base"]
    CONS_PREPARO["tabela_suporte_v23"] = CONS_INPUT_TABLES["suporte"]
    CONS_PREPARO["tabela_config_v22"] = CONS_INPUT_TABLES["config"]
    CONS_PREPARO["versao_delta_config_v22"] = int(spark.sql(
        f"DESCRIBE HISTORY {ms03_sql(CONS_INPUT_TABLES['config'])} LIMIT 1"
    ).first()["version"])
    CONS_PREPARO_SCHEMA = (
        "id_experimento string, status string, versao_preparo string, "
        "politica_json string, politica_sha256 string, tabela_base_v22 string, "
        "versao_delta_base_v22 long, id_execucao_base_v22 string, "
        "tabela_config_v22 string, versao_delta_config_v22 long, "
        "tabela_base_v23 string, tabela_suporte_v23 string, corte_estado_iso string, "
        "n_linhas_base long, fuso_base string, registrado_em_utc string"
    )
    CONS_PREPARO_TABELA = CONS_INPUT_PREFIX + "_preparo"
    spark.createDataFrame([CONS_PREPARO], CONS_PREPARO_SCHEMA).write.format(
        "delta"
    ).mode("errorifexists").saveAsTable(CONS_PREPARO_TABELA)

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    import hashlib
    import json
    import math
    import uuid
    from collections.abc import Iterator
    from datetime import datetime, timezone

    import numpy as np
    import pandas as pd
    from scipy import optimize, special
    from pyspark import StorageLevel
    from pyspark.sql import DataFrame, Window
    from pyspark.sql import functions as F


    SM23_FIT_CFG = {
        "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
        "tabela_preparo": "ctg_dsti.renato_nba.nba_sm_v23_experimentos_hml",
        "tabela_modelos": "ctg_dsti.renato_nba.nba_sm_v23_modelos_hml",
        "tabela_validacao": "ctg_dsti.renato_nba.nba_sm_v23_validacao_destino_hml",
        "tabela_metricas": "ctg_dsti.renato_nba.nba_sm_v23_metricas_destino_hml",
        "tabela_pares": "ctg_dsti.renato_nba.nba_sm_v23_comparacao_pareada_hml",
        "tabela_ajustes": "ctg_dsti.renato_nba.nba_sm_v23_ajustes_hml",
        "idades_seg": [0.0, 1800.0, 86400.0, 604800.0],
        "gravar": True,
        "executar_autotestes": True,
        "max_linhas_por_origem": 150000,  # Proteção: para, NÃO amostra/trunca.
        "max_modelos_driver": 5000,
        "max_agregados_bootstrap": 100000,
        "bootstrap_replicas": 1000,
        "semente": 20261007,
        "max_linhas_exibir": 60,
    }
    SM23_NUM = {
        "min_eventos_origem": 80, "min_clientes_origem": 20,
        "min_eventos_grupo": 100, "min_clientes_grupo": 15,
        "max_grupos_proprios": 8,
        "sigma_min": 0.15, "sigma_max": 4.5, "maxiter": 800,
        "lambda_prob_base": 0.0001,
        "lambda_prob_contexto": 0.02,
        "lambda_tempo_contexto": 0.02,
    }
    VERSAO_AJUSTE = "v2.3_temporal_memoria_pesos_02_v1"
    LOG_FLOOR = math.log(1e-15)  # Só para log loss REPORTADA; não altera q.


    # Configuração encadeada; especificação numérica vem da referência, não de defaults novos.
    SM23_NUM = copy.deepcopy(CONS_REF_META["fit_config"]["numerico"])
    SM23_FIT_CFG.update({
        "id_experimento": CONS_REF_META["id_experimento"],
        "tabela_preparo": CONS_PREPARO_TABELA,
        "idades_seg": CONS_REF_META["fit_config"]["idades_seg"],
        "semente": CONS_REF_META["fit_config"]["semente"],
        "bootstrap_replicas": CONS_REF_META["fit_config"]["bootstrap_replicas"],
    })
    for chave, sufixo in {
        "tabela_modelos": "modelos_hml",
        "tabela_validacao": "validacao_destino_hml",
        "tabela_metricas": "metricas_destino_hml",
        "tabela_pares": "comparacao_pareada_hml",
        "tabela_ajustes": "ajustes_hml",
    }.items():
        SM23_FIT_CFG[chave] = CONS_FIT_PREFIX + "_" + sufixo

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 1. Modelo e regularização
# MAGIC
# MAGIC Para origem i e contexto c, p(j)=pi(grupo(j))*r(j|grupo(j)).
# MAGIC Duração Lognormal por grupo. A contribuição exata é log[p(j) f_g(t)],
# MAGIC e a censurada é log[sum_g pi_g S_g(censura)]. O peso da Parte 01
# MAGIC multiplica TODA essa contribuição. A função média divide por sum(w)
# MAGIC para não confundir a escala dos pesos com a força da penalidade.
# MAGIC
# MAGIC Grupos/vocabulário definidos por contagens do treino sem peso são os
# MAGIC mesmos para A/C e seus filhos B/D. Contextos herdam todos os destinos
# MAGIC da origem: não removemos destinos ausentes no contexto.
# MAGIC B regulariza em direção a A; D em direção a C (probabilidade e duração).
# MAGIC Os lambdas são uma especificação inicial fixa, não hiperparâmetros
# MAGIC otimizados no holdout. Alertas de limite são expostos, não ocultados.
# MAGIC
# MAGIC Contexto raro, não visto, desconhecido ou com falha numérica usa o pai
# MAGIC TEMPORAL da mesma família de peso, com rota explícita. Pai sem ajuste:
# MAGIC SEM_MODELO_ORIGEM, sem inventar um ranking estático.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    VARIANTES = (
        "A_REFERENCIA", "B_MEMORIA", "C_PONDERACAO", "D_MEMORIA_PONDERACAO",
    )
    BASE_CTX = "__BASE__"
    SEM_MEMORIA = "__SEM_MEMORIA__"
    LOG_2PI = math.log(2.0 * math.pi)


    def criar_estrutura(pdf, cfg):
        """Vocabulário/grupos definidos só no treino, iguais entre A/B/C/D."""
        ex = pdf.loc[pdf["tipo_censura"].eq("exata")]
        if len(ex) < cfg["min_eventos_origem"]:
            raise ValueError("SUPORTE_EVENTOS_ORIGEM")
        if ex["cd_bv"].nunique() < cfg["min_clientes_origem"]:
            raise ValueError("SUPORTE_CLIENTES_ORIGEM")
        suporte = ex.groupby("destino").agg(
            n=("cd_bv", "size"), clientes=("cd_bv", "nunique")
        ).reset_index().sort_values(
            ["n", "clientes", "destino"], ascending=[False, False, True]
        )
        proprios = suporte.loc[
            (suporte["n"] >= cfg["min_eventos_grupo"])
            & (suporte["clientes"] >= cfg["min_clientes_grupo"]), "destino"
        ].head(cfg["max_grupos_proprios"]).tolist()
        destinos = sorted(suporte["destino"].astype(str))
        nomes = list(proprios)
        mapa = {d: k for k, d in enumerate(proprios)}
        if any(d not in mapa for d in destinos):
            nomes.append("__GRUPO_COMPARTILHADO__")
            for d in destinos:
                mapa.setdefault(d, len(nomes) - 1)
        log_t = np.log(pdf["dur_min"].to_numpy(float) / 86400.0)
        return {
            "destinos": destinos, "grupo": [mapa[d] for d in destinos],
            "grupos_nomes": nomes,
            "limite_mu": [float(log_t.min() - 3), float(log_t.max() + 3)],
        }


    def validar_modelo(m):
        pi = np.asarray(m["pi_grupo"], float)
        r = np.asarray(m["r_destino_no_grupo"], float)
        g = np.asarray(m["grupo"], int)
        sigma = np.asarray(m["sigma_grupo"], float)
        mu = np.asarray(m["mu_grupo"], float)
        if not (len(m["destinos"]) == len(r) == len(g)):
            raise ValueError("CONTRATO_DESTINOS")
        if len(set(m["destinos"])) != len(g) or g.min() < 0 or g.max() >= len(pi):
            raise ValueError("CONTRATO_GRUPOS")
        if len(pi) != len(sigma) or len(pi) != len(mu):
            raise ValueError("CONTRATO_PARAMETROS")
        if not all(np.isfinite(x).all() for x in (pi, r, sigma, mu)):
            raise ValueError("MODELO_NAO_FINITO")
        if min(pi.min(), r.min(), sigma.min()) <= 0:
            raise ValueError("PARAMETRO_NAO_POSITIVO")
        if not np.isclose(pi.sum(), 1.0, atol=1e-10):
            raise ValueError("MASSA_GRUPOS")
        if not np.allclose(np.bincount(g, weights=r, minlength=len(pi)), 1):
            raise ValueError("MASSA_DENTRO_GRUPO")
        if not np.allclose(pi[g] * r, m["p_destino"], atol=1e-10):
            raise ValueError("P_DESTINO_INCONSISTENTE")
        return True


    def preparar_objetivo(pdf, peso_col, estrutura, cfg, pai=None):
        """L = -log-verossimilhança média ponderada + regularização explícita.

        Exata: log(pi_g * r_j|g * f_g(t)). Direita: log(sum_g pi_g S_g(c)).
        Mesmos pesos da Parte 01: não renormaliza por cliente/contexto.
        Dividir a FUNÇÃO por sum(w) estabiliza sua escala para a penalidade.
        """
        w = pdf[peso_col].to_numpy(float)
        t = pdf["dur_min"].to_numpy(float) / 86400.0
        exata = pdf["tipo_censura"].eq("exata").to_numpy()
        if not (np.isfinite(w).all() and np.isfinite(t).all()):
            raise ValueError("DADOS_NAO_FINITOS")
        if np.any(w <= 0) or np.any(t <= 0) or not exata.any():
            raise ValueError("PESO_TEMPO_OU_SAIDAS_INVALIDAS")
        destinos = estrutura["destinos"]
        jmap = {d: j for j, d in enumerate(destinos)}
        j = np.array([jmap[str(d)] for d in pdf.loc[exata, "destino"]], int)
        grupos = np.asarray(estrutura["grupo"], int)
        gex = grupos[j]
        ng = len(estrutura["grupos_nomes"])
        ne_logits = ng - 1
        we, xe = w[exata], np.log(t[exata])
        wc, xc = w[~exata], np.log(t[~exata])
        wg = np.bincount(gex, weights=we, minlength=ng)
        sx = np.bincount(gex, weights=we * xe, minlength=ng)
        sx2 = np.bincount(gex, weights=we * xe**2, minlength=ng)
        wj = np.bincount(j, weights=we, minlength=len(destinos))
        total_w = float(w.sum())
        if pai is not None:
            pi_prior = np.asarray(pai["pi_grupo"], float)
            r_prior = np.asarray(pai["r_destino_no_grupo"], float)
            mu_prior = np.asarray(pai["mu_grupo"], float)
            sigma_prior = np.asarray(pai["sigma_grupo"], float)
            lam_p = float(cfg["lambda_prob_contexto"])
            lam_t = float(cfg["lambda_tempo_contexto"])
        else:
            tamanhos = np.bincount(grupos, minlength=ng)
            pi_prior = tamanhos / len(destinos)  # Prior uniforme por destino.
            r_prior = 1.0 / tamanhos[grupos]
            mu_global = float(np.average(xe, weights=we))
            sigma_global = max(float(np.sqrt(np.average(
                (xe - mu_global)**2, weights=we
            ))), 0.5)
            mu_prior = np.full(ng, mu_global)
            sigma_prior = np.full(ng, sigma_global)
            for k in range(ng):
                mask = gex == k
                if mask.any():
                    mu_prior[k] = np.average(xe[mask], weights=we[mask])
                    sigma_prior[k] = np.sqrt(np.average(
                        (xe[mask] - mu_prior[k])**2, weights=we[mask]
                    ))
            sigma_prior = np.clip(
                sigma_prior, cfg["sigma_min"] * 1.2, cfg["sigma_max"] * 0.8
            )
            lam_p = float(cfg["lambda_prob_base"])
            lam_t = 0.0
        eta_prior = np.log(sigma_prior)
        # MAP das proporções dentro dos grupos: o mesmo prior probabilístico
        # da função abaixo, em escala média ponderada (sem pseudocontagem bruta).
        suavizado = wj / total_w + lam_p * pi_prior[grupos] * r_prior
        soma_g = np.bincount(grupos, weights=suavizado, minlength=ng)
        r = suavizado / soma_g[grupos]
        pi0 = (wg / total_w + lam_p * pi_prior)
        pi0 /= pi0.sum()
        if pai is not None:
            pi0 = pi_prior.copy()
        theta0 = np.r_[np.log(pi0[:-1] / pi0[-1]), mu_prior, eta_prior]
        bounds = (
            [(-25.0, 25.0)] * ne_logits
            + [tuple(estrutura["limite_mu"])] * ng
            + [(math.log(cfg["sigma_min"]), math.log(cfg["sigma_max"]))] * ng
        )
        theta0 = np.clip(theta0, np.array(bounds)[:, 0] + 1e-8,
                         np.array(bounds)[:, 1] - 1e-8)
        const_r = float(np.dot(wj, np.log(r)))
        const_prior_r = -lam_p * float(np.dot(pi_prior[grupos] * r_prior, np.log(r)))

        def unpack(theta):
            logits = np.r_[theta[:ne_logits], 0.0]
            log_pi = logits - special.logsumexp(logits)
            mu = theta[ne_logits:ne_logits + ng]
            eta = theta[ne_logits + ng:]
            return log_pi, np.exp(log_pi), mu, eta, np.exp(eta)

        def objetivo(theta):
            log_pi, pi, mu, eta, sigma = unpack(theta)
            rss = np.maximum(sx2 - 2 * mu * sx + mu**2 * wg, 0.0)
            ll = float(np.sum(wg * log_pi - sx - wg * (eta + 0.5 * LOG_2PI)
                              - 0.5 * rss / sigma**2)) + const_r
            contagem_latente = wg.copy()
            grad_mu = (sx - mu * wg) / sigma**2
            grad_eta = -wg + rss / sigma**2
            if len(wc):
                z = (xc[:, None] - mu) / sigma
                log_sf = special.log_ndtr(-z)
                log_mix = special.logsumexp(log_pi + log_sf, axis=1)
                resp_w = wc[:, None] * np.exp(log_pi + log_sf - log_mix[:, None])
                mills = np.exp(-0.5 * z**2 - 0.5 * LOG_2PI - log_sf)
                ll += float(np.dot(wc, log_mix))
                contagem_latente += resp_w.sum(axis=0)
                grad_mu += (resp_w * mills / sigma).sum(axis=0)
                grad_eta += (resp_w * mills * z).sum(axis=0)
            grad_logits = (contagem_latente - total_w * pi)[:ne_logits]
            prob_pen = -lam_p * float(np.dot(pi_prior, log_pi)) + const_prior_r
            dm = (mu - mu_prior) / sigma_prior
            de = eta - eta_prior
            tempo_pen = 0.5 * lam_t * float(np.dot(pi_prior, dm**2 + de**2))
            grad = -np.r_[grad_logits, grad_mu, grad_eta] / total_w
            grad[:ne_logits] += lam_p * (pi - pi_prior)[:ne_logits]
            grad[ne_logits:ne_logits + ng] += lam_t * pi_prior * dm / sigma_prior
            grad[ne_logits + ng:] += lam_t * pi_prior * de
            valor = -ll / total_w + prob_pen + tempo_pen
            if not (np.isfinite(valor) and np.isfinite(grad).all()):
                raise FloatingPointError("OBJETIVO_NAO_FINITO")
            return float(valor), grad

        return objetivo, theta0, bounds, unpack, r, total_w


    def ajustar_modelo(pdf, peso_col, estrutura, cfg, pai=None):
        objetivo, inicial, bounds, unpack, r, massa = preparar_objetivo(
            pdf, peso_col, estrutura, cfg, pai
        )
        resultados = []
        sementes = [inicial.copy()]
        alternativo = inicial.copy()
        ng = len(estrutura["grupos_nomes"])
        alternativo[-ng:] = np.clip(
            alternativo[-ng:] + 0.25, math.log(cfg["sigma_min"]) + 1e-5,
            math.log(cfg["sigma_max"]) - 1e-5
        )
        sementes.append(alternativo)
        for s in sementes:
            res = optimize.minimize(
                objetivo, s, method="L-BFGS-B", jac=True, bounds=bounds,
                options={"maxiter": cfg["maxiter"], "ftol": 1e-10,
                         "gtol": 1e-6, "maxls": 40},
            )
            resultados.append(res)
            if res.success and np.isfinite(res.fun):
                break
        bons = [x for x in resultados if x.success and np.isfinite(x.fun)]
        if not bons:
            raise ValueError("OTIMIZACAO_NAO_CONVERGIU: " + str(resultados[-1].message))
        res = min(bons, key=lambda x: x.fun)
        _, pi, mu, _, sigma = unpack(res.x)
        g = np.asarray(estrutura["grupo"], int)
        nomes_parametros = (
            [f"logit_{k}" for k in range(ng - 1)]
            + [f"mu_{k}" for k in range(ng)] + [f"log_sigma_{k}" for k in range(ng)]
        )
        limites = [nomes_parametros[k] for k, (v, (lo, hi)) in enumerate(
            zip(res.x, bounds)
        ) if min(abs(v - lo), abs(v - hi)) < 1e-4]
        m = {
            "formato": "sm_v23_lognormal_grupos", "unidade_tempo": "dias",
            "estrutura": estrutura, "destinos": estrutura["destinos"],
            "grupo": estrutura["grupo"], "grupos_nomes": estrutura["grupos_nomes"],
            "pi_grupo": pi.tolist(), "mu_grupo": mu.tolist(),
            "sigma_grupo": sigma.tolist(), "r_destino_no_grupo": r.tolist(),
            "p_destino": (pi[g] * r).tolist(), "n_grupos": ng,
            "peso_utilizado": peso_col, "massa_pesos": massa,
            "n_observacoes": len(pdf), "n_clientes": int(pdf["cd_bv"].nunique()),
            "max_tempo_observado_dias": float(pdf["dur_min"].max() / 86400),
            "alerta_limite": bool(limites), "parametros_no_limite": limites,
            "objetivo_medio_penalizado": float(res.fun), "n_iteracoes": int(res.nit),
            "regularizacao_prob": cfg["lambda_prob_contexto"] if pai else cfg["lambda_prob_base"],
            "regularizacao_tempo": cfg["lambda_tempo_contexto"] if pai else 0.0,
            "usa_pai": pai is not None,
        }
        validar_modelo(m)
        json.dumps(m, allow_nan=False)
        return m


    def prever_probabilidades(m, idades_seg):
        """Retorna q(j|i,c,T>a), log q e variação total contra p(j|i,c)."""
        idade = np.asarray(idades_seg, float).reshape(-1)
        if np.any(idade < 0) or not np.isfinite(idade).all():
            raise ValueError("IDADE_INVALIDA")
        pi = np.asarray(m["pi_grupo"], float)
        mu = np.asarray(m["mu_grupo"], float)
        sigma = np.asarray(m["sigma_grupo"], float)
        g = np.asarray(m["grupo"], int)
        log_r = np.log(np.asarray(m["r_destino_no_grupo"], float))
        sf = np.zeros((len(idade), len(pi)), float)
        pos = idade > 0
        if pos.any():
            z = (np.log(idade[pos, None] / 86400) - mu) / sigma
            sf[pos] = special.log_ndtr(-z)
        log_mass = np.log(pi) + sf
        log_grupos = log_mass - special.logsumexp(log_mass, axis=1)[:, None]
        log_q = log_grupos[:, g] + log_r
        q = np.exp(log_q)
        if not np.allclose(q.sum(axis=1), 1, atol=1e-9):
            raise ValueError("MASSA_DESTINOS_NAO_FECHA")
        tv = 0.5 * np.abs(q - np.asarray(m["p_destino"])).sum(axis=1)
        return q, log_q, tv


    def selecionar_modelo(registro, suporte, variante, origem, contexto, tecnica):
        """Fallback sempre para o modelo TEMPORAL da mesma família de peso."""
        familia = "C_PONDERACAO" if variante in (
            "C_PONDERACAO", "D_MEMORIA_PONDERACAO"
        ) else "A_REFERENCIA"
        pai = registro.get((familia, origem, BASE_CTX))
        if pai is None:
            return None, "SEM_MODELO_ORIGEM"
        if variante in ("A_REFERENCIA", "C_PONDERACAO"):
            return pai, "ORIGEM_REFERENCIA"
        if not tecnica:
            return pai, "ORIGEM_NAO_TECNICA"
        if contexto in (None, SEM_MEMORIA):
            return pai, "ORIGEM_SEM_MEMORIA"
        filho = registro.get((variante, origem, contexto))
        if filho is not None:
            return filho, "CONTEXTO_AJUSTADO"
        status = suporte.get((origem, contexto), "NAO_VISTO_NO_TREINO")
        if status == "CANDIDATO_CONTEXTUAL":
            return pai, "ORIGEM_FALHA_AJUSTE_CONTEXTO"
        if status == "NAO_VISTO_NO_TREINO":
            return pai, "ORIGEM_CONTEXTO_NAO_VISTO"
        return pai, "ORIGEM_CONTEXTO_RARO"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 2. Autotestes numéricos e de roteamento
# MAGIC Executam antes das fontes reais. Não testam acesso Delta/permissões.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def sm23_testes_numericos():
        rng = np.random.default_rng(2718)
        dados = []
        for u in range(36):
            contexto = "pix" if u < 18 else "boleto"
            for _ in range(12):
                pix = rng.random() < (0.85 if contexto == "pix" else 0.15)
                censura = rng.random() < 0.1
                t = float(rng.lognormal(-4 if pix else -1, 0.7) * 86400)
                dados.append({"cd_bv": str(u), "estado": "login:::topo",
                              "contexto_modelo": contexto, "dur_min": t,
                              "tipo_censura": "direita" if censura else "exata",
                              "destino": None if censura else ("pix" if pix else "boleto"),
                              "peso_evento_treino": 1.0, "peso_cliente_treino": 1 / 12})
        pdf = pd.DataFrame(dados)
        cfg = dict(SM23_NUM, min_eventos_origem=20, min_clientes_origem=5,
                   min_eventos_grupo=20, min_clientes_grupo=5)
        estrutura = criar_estrutura(pdf, cfg)
        obj, theta, _, _, _, _ = preparar_objetivo(
            pdf, "peso_cliente_treino", estrutura, cfg
        )
        erro = optimize.check_grad(lambda x: obj(x)[0], lambda x: obj(x)[1],
                                   theta, epsilon=1e-6)
        if erro > 1e-4:
            raise RuntimeError(f"Gradiente incorreto no autoteste: {erro}")
        pai = ajustar_modelo(pdf, "peso_cliente_treino", estrutura, cfg)
        q, _, tv = prever_probabilidades(pai, [0, 1800, 86400])
        assert np.allclose(q.sum(axis=1), 1.0)
        assert np.allclose(q[0], pai["p_destino"]) and tv[0] < 1e-10
        filho = ajustar_modelo(pdf.loc[pdf.contexto_modelo.eq("boleto")],
                               "peso_cliente_treino", estrutura, cfg, pai)
        j = estrutura["destinos"].index("boleto")
        assert filho["p_destino"][j] > pai["p_destino"][j]
        registro = {("A_REFERENCIA", "login", BASE_CTX): pai,
                    ("B_MEMORIA", "login", "boleto"): filho}
        assert selecionar_modelo(registro, {}, "B_MEMORIA", "login", "boleto", True)[0] is filho
        assert selecionar_modelo(registro, {}, "B_MEMORIA", "login", SEM_MEMORIA, True)[0] is pai
        assert selecionar_modelo(registro, {}, "B_MEMORIA", "login", "raro", True)[0] is pai
        print("V2.3 Parte 02 — gradiente, censura, massa e roteamento: OK.")


    if SM23_FIT_CFG["executar_autotestes"]:
        sm23_testes_numericos()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 3. Consumir somente o experimento de preparo concluído
# MAGIC Seleção explícita; nenhuma leitura das fontes brutas ou customers_query.
# MAGIC As versões Delta lidas nesta Parte 02 também ficam registradas.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def sql_nome(nome):
        partes = nome.split(".")
        if len(partes) != 3 or not all(partes) or any("`" in p for p in partes):
            raise ValueError("Esperado catalogo.schema.tabela, sem crase.")
        return ".".join(f"`{p}`" for p in partes)


    def ler_snapshot(tabela):
        if not spark.catalog.tableExists(tabela):
            raise RuntimeError(f"Tabela não encontrada: {tabela}")
        versao = int(spark.sql(f"DESCRIBE HISTORY {sql_nome(tabela)} LIMIT 1").first()["version"])
        return spark.read.option("versionAsOf", versao).table(tabela), versao


    def exigir_colunas(df, colunas, origem):
        faltantes = set(colunas) - set(df.columns)
        if faltantes:
            raise ValueError(f"{origem}: colunas ausentes: {sorted(faltantes)}")


    def ha_linhas(df):
        return bool(df.limit(1).count())


    def mostrar(nome, df):
        print(f"\n{nome}")
        df.show(SM23_FIT_CFG["max_linhas_exibir"], truncate=False)


    if "spark" not in globals():
        raise RuntimeError("Execute no Databricks com sessão Spark.")
    ID_EXPERIMENTO = str(SM23_FIT_CFG["id_experimento"]).strip()
    uuid.UUID(ID_EXPERIMENTO)
    ID_AJUSTE = str(uuid.uuid4())
    preparos, V_MANIFESTO = ler_snapshot(SM23_FIT_CFG["tabela_preparo"])
    registros = preparos.filter(
        (F.col("id_experimento") == ID_EXPERIMENTO)
        & (F.col("status") == "CONCLUIDO_PREPARO")
    ).limit(2).collect()
    if len(registros) != 1:
        raise RuntimeError("O ID deve possuir exatamente um CONCLUIDO_PREPARO.")
    PREPARO = registros[0].asDict()
    POLITICA = json.loads(PREPARO["politica_json"])
    if hashlib.sha256(PREPARO["politica_json"].encode()).hexdigest() != PREPARO["politica_sha256"]:
        raise RuntimeError("Hash da política de preparo não confere.")
    if spark.conf.get("spark.sql.session.timeZone") != PREPARO["fuso_base"]:
        raise RuntimeError("Fuso diferente do preparo. Não altere datas para contornar o teste.")
    if POLITICA["min_eventos_contexto"] != 80 or POLITICA["min_clientes_contexto"] != 20:
        print("Atenção: suporte contextual da Parte 01 tem limites personalizados.")

    base_snapshot, V_BASE = ler_snapshot(PREPARO["tabela_base_v23"])
    suporte_snapshot, V_SUPORTE = ler_snapshot(PREPARO["tabela_suporte_v23"])
    base_v23 = base_snapshot.filter(F.col("id_experimento") == ID_EXPERIMENTO)
    suporte_v23 = suporte_snapshot.filter(F.col("id_experimento") == ID_EXPERIMENTO)
    exigir_colunas(base_v23, {
        "cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
        "validacao_cliente", "elegivel_ajuste", "origem_tecnica", "contexto_modelo",
        "peso_evento_treino", "peso_cliente_treino", "politica_sha256",
    }, "Base V2.3")
    exigir_colunas(suporte_v23, {
        "estado", "contexto_modelo", "status_suporte", "contexto_tem_suporte_candidato",
    }, "Suporte V2.3")
    if base_v23.count() != int(PREPARO["n_linhas_base"]):
        raise RuntimeError("Contagem da base diverge do manifesto de preparo.")
    if ha_linhas(base_v23.filter(
        F.col("politica_sha256").isNull()
        | (F.col("politica_sha256") != PREPARO["politica_sha256"])
    )):
        raise RuntimeError("A base mistura políticas de memória/pesos.")
    if ha_linhas(base_v23.groupBy("cd_bv", "passo").count().filter("count != 1")):
        raise RuntimeError("Cliente/passo duplicado. Não deduplicar silenciosamente.")
    if ha_linhas(suporte_v23.groupBy("estado", "contexto_modelo").count().filter("count != 1")):
        raise RuntimeError("Contexto duplicado na tabela de suporte.")
    if ha_linhas(base_v23.groupBy("cd_bv").agg(
        F.countDistinct("validacao_cliente").alias("n")
    ).filter("n != 1")):
        raise RuntimeError("Cliente presente em ambos os splits.")

    COLS_TREINO = ["cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
                   "contexto_modelo", "origem_tecnica", "peso_evento_treino",
                   "peso_cliente_treino"]
    treino_v23 = base_v23.filter(
        ~F.col("validacao_cliente") & F.col("elegivel_ajuste")
    ).select(*COLS_TREINO).persist(StorageLevel.DISK_ONLY)
    holdout_v23 = base_v23.filter(
        F.col("validacao_cliente") & F.col("elegivel_ajuste")
    ).select("cd_bv", "passo", "estado", "destino", "dur_min", "tipo_censura",
             "contexto_modelo", "origem_tecnica")
    for df, nome in ((treino_v23, "treino"), (holdout_v23, "holdout")):
        if not ha_linhas(df):
            raise RuntimeError(f"{nome} vazio.")
        invalidas = (F.col("estado").isNull() | F.col("contexto_modelo").isNull()
                     | F.col("dur_min").isNull() | F.isnan("dur_min")
                     | (F.col("dur_min") <= 0) | (F.abs("dur_min") == float("inf"))
                     | ~F.col("tipo_censura").isin("exata", "direita")
                     | F.col("tipo_censura").isNull()
                     | ((F.col("tipo_censura") == "exata") & F.col("destino").isNull())
                     | ((F.col("tipo_censura") == "direita") & F.col("destino").isNotNull()))
        if ha_linhas(df.filter(invalidas)):
            raise RuntimeError(f"Observação elegível inválida no {nome}.")
    for col in ("peso_evento_treino", "peso_cliente_treino"):
        if ha_linhas(treino_v23.filter(
            F.col(col).isNull() | F.isnan(col) | (F.col(col) <= 0)
            | (F.abs(F.col(col)) == float("inf"))
        )):
            raise RuntimeError(f"Peso inválido: {col}")
    if ha_linhas(treino_v23.groupBy("cd_bv", "estado").agg(
        F.sum("peso_cliente_treino").alias("s")
    ).filter(F.abs(F.col("s") - 1) > 1e-8)):
        raise RuntimeError("Pesos não somam 1 por cliente-origem.")
    max_grupo = treino_v23.groupBy("estado").count().agg(F.max("count")).first()[0]
    if max_grupo > SM23_FIT_CFG["max_linhas_por_origem"]:
        raise RuntimeError(
            f"Maior origem tem {max_grupo} linhas; excede proteção de memória. "
            "Revisar capacidade, não truncar nem amostrar silenciosamente."
        )
    print("Experimento:", ID_EXPERIMENTO, "| Ajuste:", ID_AJUSTE)
    print("Corte preservado:", PREPARO["corte_estado_iso"])
    print("Linhas de treino:", treino_v23.count(), "| maior origem:", max_grupo)
    mostrar("V23_20_DADOS_AJUSTE", treino_v23.groupBy("tipo_censura").agg(
        F.count("*").alias("n_linhas"), F.countDistinct("cd_bv").alias("n_clientes"),
        F.sum("peso_cliente_treino").alias("massa_cliente_origem"),
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 4. Ajustar pais A/C e persistir antes de ajustar os contextos
# MAGIC Spark distribui por origem. Pandas recebe SOMENTE um grupo, não toda a base.
# MAGIC O limite de memória é uma proteção, não um parâmetro de amostragem.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    SCHEMA_MODELO = (
        "variante string, nivel string, estado string, contexto_modelo string, "
        "status_modelo string, n_observacoes long, n_saidas_exatas long, "
        "n_clientes long, massa_pesos double, n_grupos long, alerta_limite boolean, "
        "modelo_json string, detalhe string, reuso boolean"
    )


    def linha_modelo(pdf, variante, nivel, contexto, peso):
        return dict(variante=variante, nivel=nivel, estado=str(pdf["estado"].iloc[0]),
                    contexto_modelo=contexto, status_modelo="NAO_AJUSTADO",
                    n_observacoes=int(len(pdf)),
                    n_saidas_exatas=int(pdf["tipo_censura"].eq("exata").sum()),
                    n_clientes=int(pdf["cd_bv"].nunique()),
                    massa_pesos=float(pdf[peso].sum()), n_grupos=0,
                    alerta_limite=False, modelo_json=None, detalhe="", reuso=False)


    def preencher_modelo(linha, modelo):
        linha.update(status_modelo="AJUSTADO", n_grupos=int(modelo["n_grupos"]),
                     alerta_limite=bool(modelo["alerta_limite"]),
                     modelo_json=json.dumps(modelo, allow_nan=False),
                     detalhe=",".join(modelo["parametros_no_limite"]))
        return linha


    def ajustar_pais(pdf):
        pdf = pdf.sort_values(["cd_bv", "passo"], kind="mergesort")
        linhas = []
        try:
            estrutura = criar_estrutura(pdf, SM23_NUM)
            erro_suporte = None
        except ValueError as erro:
            estrutura, erro_suporte = None, str(erro)
        for variante, peso in (("A_REFERENCIA", "peso_evento_treino"),
                               ("C_PONDERACAO", "peso_cliente_treino")):
            linha = linha_modelo(pdf, variante, "ORIGEM", BASE_CTX, peso)
            if erro_suporte:
                linha["detalhe"] = erro_suporte
            else:
                try:
                    preencher_modelo(linha, ajustar_modelo(pdf, peso, estrutura, SM23_NUM))
                except (ValueError, FloatingPointError) as erro:
                    linha["detalhe"] = f"{type(erro).__name__}: {erro}"[:1000]
            linhas.append(linha)
        return pd.DataFrame(linhas)


    HASH_AJUSTE = hashlib.sha256(json.dumps(
        {"versao": VERSAO_AJUSTE, "numerico": SM23_NUM,
         "politica_preparo": PREPARO["politica_sha256"],
         "idades": SM23_FIT_CFG["idades_seg"]}, sort_keys=True
    ).encode()).hexdigest()


    def guardar(df, tabela, etapa):
        """Append isolado; não sobrescreve nenhuma execução. Releitura materializa."""
        saida = (df.withColumn("id_experimento", F.lit(ID_EXPERIMENTO))
                 .withColumn("id_ajuste", F.lit(ID_AJUSTE))
                 .withColumn("versao_ajuste", F.lit(VERSAO_AJUSTE))
                 .withColumn("hash_ajuste", F.lit(HASH_AJUSTE))
                 .withColumn("etapa_artefato", F.lit(etapa)))
        if SM23_FIT_CFG["gravar"]:
            if tabela in {PREPARO["tabela_base_v23"], PREPARO["tabela_suporte_v23"],
                          PREPARO["tabela_base_v22"], PREPARO["tabela_config_v22"],
                          SM23_FIT_CFG["tabela_preparo"]}:
                raise ValueError("Uma tabela de saída coincide com uma fonte.")
            if spark.catalog.tableExists(tabela):
                existente = spark.table(tabela)
                exigir_colunas(existente, {"id_ajuste", "etapa_artefato"}, tabela)
                if ha_linhas(existente.filter(
                    (F.col("id_ajuste") == ID_AJUSTE) & (F.col("etapa_artefato") == etapa)
                )):
                    raise RuntimeError(
                        "Etapa já escrita neste ID. Reexecute a Parte 02 "
                        "inteira para novo ID."
                    )
            saida.write.format("delta").mode("append").saveAsTable(tabela)
            return spark.table(tabela).filter(
                (F.col("id_ajuste") == ID_AJUSTE) & (F.col("etapa_artefato") == etapa)
            )
        saida = saida.persist(StorageLevel.DISK_ONLY)
        saida.count()
        return saida


    pais_brutos = treino_v23.groupBy("estado").applyInPandas(
        ajustar_pais, schema=SCHEMA_MODELO
    )
    pais = guardar(pais_brutos, SM23_FIT_CFG["tabela_modelos"], "PAIS")
    mostrar("V23_21_MODELOS_ORIGEM", pais.groupBy(
        "variante", "status_modelo", "alerta_limite"
    ).agg(F.count("*").alias("n_modelos"), F.sum("n_observacoes").alias("n_linhas")))
    mostrar("V23_21_FALHAS_ORIGEM", pais.filter("status_modelo != 'AJUSTADO'").groupBy(
        "variante", "detalhe"
    ).count().orderBy(F.desc("count")))
    linhas_pais = pais.filter("status_modelo = 'AJUSTADO'").select(
        "variante", "estado", "modelo_json"
    ).limit(SM23_FIT_CFG["max_modelos_driver"] + 1).collect()
    if len(linhas_pais) > SM23_FIT_CFG["max_modelos_driver"]:
        raise RuntimeError("Muitos modelos para broadcast. Rever dimensionamento.")
    PAIS = {(r["variante"], r["estado"]): json.loads(r["modelo_json"]) for r in linhas_pais}
    if not all(any(v == familia for v, _ in PAIS) for familia in ("A_REFERENCIA", "C_PONDERACAO")):
        raise RuntimeError("Nenhum pai ajustado em uma das duas famílias. Rever falhas.")
    BC_PAIS = spark.sparkContext.broadcast(PAIS)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 5. Ajustar somente contextos candidatos B/D
# MAGIC Sem memória/raros/não vistos continuam usando o pai temporal na avaliação.
# MAGIC Não utilizamos o holdout para criar grupos, suporte, prior ou pesos.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def ajustar_contextos(pdf):
        pdf = pdf.sort_values(["cd_bv", "passo"], kind="mergesort")
        origem, contexto = str(pdf["estado"].iloc[0]), str(pdf["contexto_modelo"].iloc[0])
        linhas = []
        for variante, familia, peso in (
            ("B_MEMORIA", "A_REFERENCIA", "peso_evento_treino"),
            ("D_MEMORIA_PONDERACAO", "C_PONDERACAO", "peso_cliente_treino"),
        ):
            linha = linha_modelo(pdf, variante, "CONTEXTO", contexto, peso)
            pai = BC_PAIS.value.get((familia, origem))
            if pai is None:
                linha["detalhe"] = "PAI_SEM_AJUSTE"
            else:
                try:
                    preencher_modelo(linha, ajustar_modelo(
                        pdf, peso, pai["estrutura"], SM23_NUM, pai
                    ))
                except (ValueError, FloatingPointError) as erro:
                    linha["detalhe"] = f"{type(erro).__name__}: {erro}"[:1000]
            linhas.append(linha)
        return pd.DataFrame(linhas)


    candidatos = suporte_v23.filter(F.col("contexto_tem_suporte_candidato")).select(
        "estado", "contexto_modelo"
    )
    base_contextos = treino_v23.join(candidatos, ["estado", "contexto_modelo"], "inner")
    if ha_linhas(base_contextos):
        contextos_brutos = base_contextos.groupBy("estado", "contexto_modelo").applyInPandas(
            ajustar_contextos, schema=SCHEMA_MODELO
        )
    else:
        contextos_brutos = spark.createDataFrame([], SCHEMA_MODELO)
    contextos = guardar(contextos_brutos, SM23_FIT_CFG["tabela_modelos"], "CONTEXTOS")
    mostrar("V23_22_MODELOS_CONTEXTO", contextos.groupBy(
        "variante", "status_modelo", "alerta_limite"
    ).agg(F.count("*").alias("n_modelos"), F.sum("n_observacoes").alias("n_linhas")))
    mostrar("V23_22_FALHAS_CONTEXTO", contextos.filter("status_modelo != 'AJUSTADO'").groupBy(
        "variante", "detalhe"
    ).count().orderBy(F.desc("count")))

    # Registrar explicitamente que B reutiliza A e D reutiliza C nas origens.
    reusos = pais.select(*spark.createDataFrame([], SCHEMA_MODELO).columns).withColumn(
        "variante", F.when(F.col("variante") == "A_REFERENCIA", "B_MEMORIA")
        .otherwise("D_MEMORIA_PONDERACAO")
    ).withColumn("reuso", F.lit(True))
    reusos = guardar(reusos, SM23_FIT_CFG["tabela_modelos"], "PAIS_REUTILIZADOS")
    REGISTRO = {(v, origem, BASE_CTX): m for (v, origem), m in PAIS.items()}
    ctx_coletados = contextos.filter("status_modelo = 'AJUSTADO'").select(
        "variante", "estado", "contexto_modelo", "modelo_json"
    ).limit(SM23_FIT_CFG["max_modelos_driver"] + 1).collect()
    if len(ctx_coletados) > SM23_FIT_CFG["max_modelos_driver"]:
        raise RuntimeError("Muitos contextos para broadcast. Rever dimensionamento.")
    for r in ctx_coletados:
        REGISTRO[(r["variante"], r["estado"], r["contexto_modelo"])] = json.loads(r["modelo_json"])
    SUPORTE = {(r["estado"], r["contexto_modelo"]): r["status_suporte"]
               for r in suporte_v23.select("estado", "contexto_modelo", "status_suporte").collect()}
    BC_REGISTRO = spark.sparkContext.broadcast(REGISTRO)
    BC_SUPORTE = spark.sparkContext.broadcast(SUPORTE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 6. Validação nos mesmos marcos e observações
# MAGIC O marco a considera dur_min > a. Censuras continuam na tabela técnica,
# MAGIC mas não viram rótulo de "nenhuma ação" e não entram no acerto do destino.
# MAGIC A comparação de destino usa saídas exatas observadas antes do corte;
# MAGIC não certifica probabilidade de agir em N dias nem valida multi-step.
# MAGIC Log loss usa piso 1e-15 explícito e contagem de clipping. Destino não
# MAGIC aprendido tem p=0 e é penalizado: não o excluímos silenciosamente.
# MAGIC Registros sem modelo ficam nas contagens de cobertura.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    SCHEMA_VALIDACAO = (
        "cd_bv string, passo long, estado string, contexto_modelo string, "
        "origem_tecnica boolean, idade_seg double, tipo_censura string, "
        "destino_real string, variante string, rota_modelo string, "
        "tem_previsao boolean, suporte_destino boolean, alerta_limite boolean, "
        "extrapolacao boolean, top1_previsto string, top5_previsto array<string>, "
        "acerto_top1 double, acerto_top5 double, brier double, logloss_clip double, "
        "logloss_sem_tempo_clip double, acerto_top1_sem_tempo double, "
        "nll_clip_aplicado boolean, prob_destino_real double, variacao_temporal_tv double"
    )
    VALID_COLS = [campo.strip().split()[0] for campo in SCHEMA_VALIDACAO.split(",")]


    def validar_lotes(iterator: Iterator[pd.DataFrame]):
        registro, suporte = BC_REGISTRO.value, BC_SUPORTE.value
        for pdf in iterator:
            saidas = []
            for (origem, contexto, tecnica), grupo_df in pdf.groupby(
                ["estado", "contexto_modelo", "origem_tecnica"], sort=False, dropna=False
            ):
                for idade in SM23_FIT_CFG["idades_seg"]:
                    casos = grupo_df.loc[grupo_df["dur_min"] > idade]
                    if casos.empty:
                        continue
                    for variante in VARIANTES:
                        m, rota = selecionar_modelo(registro, suporte, variante,
                                                   str(origem), str(contexto), bool(tecnica))
                        q = logq = p0 = None
                        top5, ordem0 = [], []
                        tv, extrap = None, False
                        if m is not None:
                            try:
                                q2, lq2, tv2 = prever_probabilidades(m, [idade])
                                q, logq, tv = q2[0], lq2[0], float(tv2[0])
                                p0 = np.asarray(m["p_destino"], float)
                                ordem = np.argsort(-q, kind="mergesort")
                                top5 = [m["destinos"][int(j)] for j in ordem[:5]]
                                ordem0 = np.argsort(-p0, kind="mergesort")
                                jmap = {d: j for j, d in enumerate(m["destinos"])}
                                extrap = idade / 86400 > m["max_tempo_observado_dias"]
                            except (ValueError, FloatingPointError):
                                m, q, logq = None, None, None
                                rota = "FALHA_NUMERICA_INFERENCIA"
                        for row in casos.itertuples(index=False):
                            exata = row.tipo_censura == "exata"
                            destino = str(row.destino) if exata else None
                            out = dict(cd_bv=str(row.cd_bv), passo=int(row.passo),
                                       estado=str(origem), contexto_modelo=str(contexto),
                                       origem_tecnica=bool(tecnica), idade_seg=float(idade),
                                       tipo_censura=str(row.tipo_censura), destino_real=destino,
                                       variante=variante, rota_modelo=rota,
                                       tem_previsao=m is not None, suporte_destino=None,
                                       alerta_limite=bool(m["alerta_limite"]) if m else False,
                                       extrapolacao=bool(extrap),
                                       top1_previsto=top5[0] if m else None,
                                       top5_previsto=top5 if m else [],
                                       acerto_top1=None, acerto_top5=None, brier=None,
                                       logloss_clip=None, logloss_sem_tempo_clip=None,
                                       acerto_top1_sem_tempo=None, nll_clip_aplicado=None,
                                       prob_destino_real=None, variacao_temporal_tv=tv if m else None)
                            if m is not None and exata:
                                j = jmap.get(destino)
                                prob = float(q[j]) if j is not None else 0.0
                                lq = float(logq[j]) if j is not None else -np.inf
                                lp0 = float(np.log(p0[j])) if j is not None else -np.inf
                                out.update(
                                    suporte_destino=j is not None,
                                    acerto_top1=float(destino == top5[0]),
                                    acerto_top5=float(destino in top5),
                                    brier=float(np.dot(q, q) - 2 * prob + 1),
                                    logloss_clip=float(-max(lq, LOG_FLOOR)),
                                    logloss_sem_tempo_clip=float(-max(lp0, LOG_FLOOR)),
                                    acerto_top1_sem_tempo=float(
                                        destino == m["destinos"][int(ordem0[0])]
                                    ),
                                    nll_clip_aplicado=bool(lq < LOG_FLOOR),
                                    prob_destino_real=prob,
                                )
                            saidas.append(out)
            result = pd.DataFrame(saidas, columns=VALID_COLS)
            if not result.empty:
                result["passo"] = pd.array(result["passo"], dtype="Int64")
            yield result


    arrow_anterior = spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")
    try:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "3000")
        validacao_bruta = holdout_v23.mapInPandas(validar_lotes, SCHEMA_VALIDACAO)
        validacao_v23 = guardar(validacao_bruta, SM23_FIT_CFG["tabela_validacao"], "VALIDACAO")
    finally:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", arrow_anterior)
    if ha_linhas(validacao_v23.groupBy("cd_bv", "passo", "idade_seg").agg(
        F.count("*").alias("n"), F.countDistinct("variante").alias("v")
    ).filter("n != 4 OR v != 4")):
        raise RuntimeError("Validação não preservou quatro variantes por observação/marco.")
    mostrar("V23_23_ROTAS_HOLDOUT", validacao_v23.groupBy(
        "variante", "idade_seg", "rota_modelo"
    ).agg(F.count("*").alias("n_observacoes"), F.countDistinct("cd_bv").alias("n_clientes"))
     .orderBy("idade_seg", "variante", F.desc("n_observacoes")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 7. Cobertura, desempenho e comparação pareada por cliente
# MAGIC Todos os destinos fora do suporte são penalizados com p=0; as métricas
# MAGIC de qualidade ficam condicionadas à existência de previsão, cuja
# MAGIC cobertura aparece ao lado. Pares usam INTERSEÇÃO de casos modelados
# MAGIC pelas quatro variantes. NLL/Brier menores são melhores; Top-1/5 maiores.
# MAGIC Bootstrap reamostra clientes e não linhas individuais. Um cliente conta
# MAGIC uma vez após calcular sua média nos mesmos casos do marco/segmento.
# MAGIC Os ICs descrevem este holdout de desenvolvimento, não garantem produção.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def adicionar_segmento(df):
        return df.withColumn("segmento", F.explode(F.when(
            F.col("origem_tecnica"), F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS"))
        ).otherwise(F.array(F.lit("TODAS")))))


    exatas_v23 = validacao_v23.filter(F.col("tipo_censura") == "exata")
    seg = adicionar_segmento(exatas_v23)
    metricas = seg.groupBy("segmento", "variante", "idade_seg").agg(
        F.count("*").alias("n_eventos"),
        F.sum(F.col("tem_previsao").cast("long")).alias("n_previstos"),
        F.sum(F.coalesce(F.col("suporte_destino").cast("long"), F.lit(0))).alias("n_suportados"),
        F.sum(F.coalesce(F.col("nll_clip_aplicado").cast("long"), F.lit(0))).alias("n_logloss_clip"),
        F.avg("acerto_top1").alias("top1_por_evento_previsto"),
        F.avg("acerto_top5").alias("top5_por_evento_previsto"),
        F.avg("logloss_clip").alias("logloss_por_evento_previsto"),
        F.avg("brier").alias("brier_por_evento_previsto"),
        F.avg("variacao_temporal_tv").alias("variacao_temporal_media"),
        F.avg(F.col("logloss_sem_tempo_clip") - F.col("logloss_clip")).alias("ganho_logloss_tempo"),
    ).withColumn("cobertura_modelo", F.col("n_previstos") / F.col("n_eventos"))
    metricas = guardar(metricas, SM23_FIT_CFG["tabela_metricas"], "METRICAS")
    mostrar("V23_24_METRICAS_DESTINO", metricas.select(
        "segmento", "variante", "idade_seg", "n_eventos", "n_previstos", "n_suportados",
        "cobertura_modelo", "top1_por_evento_previsto", "top5_por_evento_previsto",
        "logloss_por_evento_previsto", "brier_por_evento_previsto", "n_logloss_clip",
        "ganho_logloss_tempo",
    ).orderBy("segmento", "idade_seg", "variante"))

    w_caso = Window.partitionBy("cd_bv", "passo", "idade_seg")
    pares = exatas_v23.withColumn("_n_prev", F.sum(
        F.col("tem_previsao").cast("long")
    ).over(w_caso)).filter(F.col("_n_prev") == 4)
    pares_seg = adicionar_segmento(pares)
    por_cliente = pares_seg.groupBy("segmento", "idade_seg", "cd_bv", "variante").agg(
        F.count("*").alias("n_casos"),
        F.avg("logloss_clip").alias("logloss"), F.avg("brier").alias("brier"),
        F.avg("acerto_top1").alias("top1"), F.avg("acerto_top5").alias("top5"),
    )
    if por_cliente.count() > SM23_FIT_CFG["max_agregados_bootstrap"]:
        raise RuntimeError("Muitos agregados para bootstrap no driver. Rever dimensão.")
    # Somente médias por cliente (~506 clientes), nunca eventos individuais.
    pdf_clientes = por_cliente.toPandas()
    if pdf_clientes.empty:
        raise RuntimeError("Nenhum caso comum modelado nas quatro variantes.")
    rng = np.random.default_rng(SM23_FIT_CFG["semente"])
    linhas_pares = []
    for (segmento, idade), grupo in pdf_clientes.groupby(["segmento", "idade_seg"], sort=True):
        ids = sorted(grupo["cd_bv"].unique())
        indices = rng.integers(0, len(ids), size=(SM23_FIT_CFG["bootstrap_replicas"], len(ids)))
        for metrica in ("logloss", "brier", "top1", "top5"):
            tabela = grupo.pivot(index="cd_bv", columns="variante", values=metrica).reindex(ids)
            if set(tabela.columns) != set(VARIANTES) or tabela.isna().any().any():
                raise RuntimeError("Pares incompletos por cliente no bootstrap.")
            a = tabela["A_REFERENCIA"].to_numpy(float)
            for variante in VARIANTES[1:]:
                b = tabela[variante].to_numpy(float)
                ganho = a - b if metrica in ("logloss", "brier") else b - a
                boot = ganho[indices].mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                ncasos = int(grupo.loc[grupo.variante.eq("A_REFERENCIA"), "n_casos"].sum())
                linhas_pares.append(dict(
                    segmento=str(segmento), idade_seg=float(idade), variante=variante,
                    metrica=metrica, n_clientes=len(ids), n_casos=ncasos,
                    media_a=float(a.mean()), media_variante=float(b.mean()),
                    ganho_medio=float(ganho.mean()), ganho_ic95_lo=float(lo), ganho_ic95_hi=float(hi),
                ))
    SCHEMA_PARES = (
        "segmento string, idade_seg double, variante string, metrica string, "
        "n_clientes long, n_casos long, media_a double, media_variante double, "
        "ganho_medio double, ganho_ic95_lo double, ganho_ic95_hi double"
    )
    comparacao = guardar(spark.createDataFrame(linhas_pares, SCHEMA_PARES),
                        SM23_FIT_CFG["tabela_pares"], "COMPARACAO_PAREADA")
    mostrar("V23_25_COMPARACAO_PAREADA", comparacao.select(
        "segmento", "idade_seg", "variante", "metrica", "n_clientes", "n_casos",
        "media_a", "media_variante", "ganho_medio", "ganho_ic95_lo", "ganho_ic95_hi",
    ).filter("segmento = 'ORIGENS_TECNICAS'").orderBy("idade_seg", "metrica", "variante"))
    print("Ganho positivo = melhora sobre A, tanto para perdas quanto para acertos.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 8. Conclusão da etapa e manifesto
# MAGIC Cada tabela é escrita separadamente: não é uma transação conjunta.
# MAGIC Somente CONCLUIDO_TREINO_VALIDACAO autoriza consumir este ID de ajuste.
# MAGIC Falhas/modelos nos limites ficam registrados. Não há promoção automática.
# MAGIC O resultado NÃO é a nova tabela de negócio nem mede Top-5 macro acionável.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    config_execucao = {
        "versao_ajuste": VERSAO_AJUSTE, "numerico": SM23_NUM,
        "rotas": "B->contexto ou A; D->contexto ou C; sempre temporal",
        "sem_nova_amostragem": True, "idades_seg": SM23_FIT_CFG["idades_seg"],
        "validacao": "proximo_estado_raw_condicionado_a_duracao_maior_que_marco",
        "pares": "casos_modelados_nas_quatro_variantes; media_por_cliente",
        "logloss_piso": 1e-15, "bootstrap_replicas": SM23_FIT_CFG["bootstrap_replicas"],
        "semente": SM23_FIT_CFG["semente"], "fonte_preparo": PREPARO,
        "versao_delta_preparo": V_MANIFESTO, "versao_delta_base_v23": V_BASE,
        "versao_delta_suporte_v23": V_SUPORTE,
        "acoes_tecnicas": POLITICA["acoes_tecnicas"],
        "tabelas_saida": {k: v for k, v in SM23_FIT_CFG.items() if k.startswith("tabela_")},
    }
    meta_final = dict(
        id_experimento=ID_EXPERIMENTO, id_ajuste=ID_AJUSTE,
        status="CONCLUIDO_TREINO_VALIDACAO", versao_ajuste=VERSAO_AJUSTE,
        hash_ajuste=HASH_AJUSTE, politica_sha256=PREPARO["politica_sha256"],
        config_json=json.dumps(config_execucao, ensure_ascii=False, sort_keys=True),
        registrado_em_utc=datetime.now(timezone.utc).isoformat(), publicavel=False,
    )
    SCHEMA_MANIFESTO = (
        "id_experimento string, id_ajuste string, status string, versao_ajuste string, "
        "hash_ajuste string, politica_sha256 string, config_json string, "
        "registrado_em_utc string, publicavel boolean"
    )
    if SM23_FIT_CFG["gravar"]:
        # Todas as escritas anteriores já foram materializadas pela releitura.
        if spark.catalog.tableExists(SM23_FIT_CFG["tabela_ajustes"]):
            if ha_linhas(spark.table(SM23_FIT_CFG["tabela_ajustes"]).filter(
                F.col("id_ajuste") == ID_AJUSTE
            )):
                raise RuntimeError("Manifesto já concluído para este ID de ajuste.")
        spark.createDataFrame([meta_final], SCHEMA_MANIFESTO).write.format("delta").mode(
            "append"
        ).saveAsTable(SM23_FIT_CFG["tabela_ajustes"])
        print("V23_26_CONCLUSAO: treino e validação persistidos.")
    else:
        print("V23_26_CONCLUSAO: resultados apenas em memória; sem manifesto persistido.")
    print("id_experimento:", ID_EXPERIMENTO)
    print("id_ajuste:", ID_AJUSTE)
    print("Não houve alteração da V2.2 nem da tabela de negócio.")
    print("Próxima etapa: analisar comparação; depois propagar com memória por trajetória.")
    treino_v23.unpersist()

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    import copy
    import hashlib
    import json
    import math
    import uuid
    from datetime import datetime, timezone
    from typing import Any

    import numpy as np
    import pandas as pd
    import scipy
    from scipy import special, stats
    from pyspark import StorageLevel
    from pyspark.sql import DataFrame, Window
    from pyspark.sql import functions as F


    C02_CFG = {
        "id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
        "id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
        "tabela_ajustes": "ctg_dsti.renato_nba.nba_sm_v23_ajustes_hml",
        "prefixo_saida": "ctg_dsti.renato_nba.nba_sm_v23_02c",
        "gravar": True,
        "autotestes": True,
        "bootstrap_replicas": 2000,
        "semente": 20261008,
        "prob_baixa_triagem": 1e-6,  # Só relatório; não altera os scores.
        "max_modelos_driver": 5000,
        "max_entradas_kernels": 2000000,
        "max_agregados_driver": 100000,
        "max_celulas_bootstrap": 10000000,
        "max_linhas_exibir": 40,
        "casos_criticos_por_idade": 10,
        "tol_prob": 1e-9,
        "tol_metrica": 1e-7,
    }
    C02_VERSAO = "v2.3_02c_memoria_tempo_compartilhado_v1"
    C02_AJUSTE_SUPORTADO = "v2.3_temporal_memoria_pesos_02_v1"
    C02_BASE = "__BASE__"
    C02_A = "A_REFERENCIA"
    C02_B = "B_MEMORIA"
    C02_B2 = "B2_MEMORIA_TEMPO_COMPARTILHADO"
    C02_CHAVE = ["cd_bv", "passo", "idade_seg"]
    C02_INICIO = datetime.now(timezone.utc).isoformat()
    C02_ID = str(uuid.uuid4())
    C02_FONTES = {}
    C02_CHECKS = []
    C02_RELATORIOS = {}
    C02_CACHES = []


    C02_CFG.update({
        "id_experimento": ID_EXPERIMENTO,
        "id_ajuste": ID_AJUSTE,
        "tabela_ajustes": SM23_FIT_CFG["tabela_ajustes"],
        "prefixo_saida": CONS_CFG["prefixo_saida"] + "_02c",
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 1. Núcleo numérico e alinhamento
# MAGIC Conserva p_B, pi_B e r_B. Só troca mu/sigma por A.
# MAGIC O alinhamento compara a partição de DESTINOS, nunca assume que o índice
# MAGIC do grupo seja igual nos dois JSONs. Partições incompatíveis interrompem.
# MAGIC A/B usam a mesma fórmula da Parte 02, incluindo o desempate estável.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def c02_hash(objeto: Any) -> str:
        texto = json.dumps(objeto, sort_keys=True, ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()


    def c02_validar_modelo(m: dict) -> None:
        if m.get("formato") != "sm_v23_lognormal_grupos":
            raise ValueError("Formato de modelo diferente do contrato da Parte 02.")
        if m.get("unidade_tempo") != "dias":
            raise ValueError("Unidade do modelo deve ser dias.")
        pi, mu, sigma, r, p0 = [np.asarray(m[k], float) for k in (
            "pi_grupo", "mu_grupo", "sigma_grupo", "r_destino_no_grupo", "p_destino"
        )]
        g = np.asarray(m["grupo"], int)
        nd, ng = len(m["destinos"]), len(pi)
        if int(m["n_grupos"]) != ng:
            raise ValueError("n_grupos incompatível com os parâmetros.")
        if not nd or not ng or len(set(m["destinos"])) != nd:
            raise ValueError("Destinos vazios ou duplicados.")
        if any(len(x) != nd for x in (g, r, p0)):
            raise ValueError("Dimensões dos destinos incompatíveis.")
        if any(len(x) != ng for x in (mu, sigma, m["grupos_nomes"])):
            raise ValueError("Dimensões dos grupos incompatíveis.")
        if not np.array_equal(np.unique(g), np.arange(ng)):
            raise ValueError("Grupos inválidos ou sem destinos.")
        if not all(np.isfinite(x).all() for x in (pi, mu, sigma, r, p0)):
            raise ValueError("Parâmetro não finito.")
        if min(pi.min(), sigma.min(), r.min(), p0.min()) <= 0:
            raise ValueError("Massa ou sigma não positivo.")
        if not np.isclose(pi.sum(), 1, atol=1e-10, rtol=0):
            raise ValueError("A massa dos grupos não fecha.")
        if not np.allclose(np.bincount(g, weights=r), 1, atol=1e-10, rtol=0):
            raise ValueError("A massa dentro dos grupos não fecha.")
        if not np.allclose(pi[g] * r, p0, atol=1e-10, rtol=0):
            raise ValueError("p_destino incompatível com pi e r.")
        if m["estrutura"]["destinos"] != m["destinos"]:
            raise ValueError("Estrutura/destinos divergentes.")
        if bool(m["alerta_limite"]) != bool(m["parametros_no_limite"]):
            raise ValueError("Flag e lista de limites divergem.")
        if m["estrutura"]["grupo"] != m["grupo"]:
            raise ValueError("Estrutura/grupos divergentes.")
        if not (np.isfinite(m["max_tempo_observado_dias"])
                and m["max_tempo_observado_dias"] > 0):
            raise ValueError("Máximo temporal inválido.")


    def c02_compor_modelo(b: dict, a: dict) -> tuple[dict, list[int]]:
        """B2 = p_B × S_A; sem otimização, sem suavização ou novo piso."""
        c02_validar_modelo(a)
        c02_validar_modelo(b)
        if set(a["destinos"]) != set(b["destinos"]):
            raise ValueError("A e B precisam ter o mesmo vocabulário de destinos.")
        grupos_a = dict(zip(a["destinos"], a["grupo"]))
        grupos_b = np.asarray(b["grupo"], int)
        alinhamento = []
        for gb in range(b["n_grupos"]):
            membros = {d for d, g in zip(b["destinos"], grupos_b) if g == gb}
            candidatos = {grupos_a[d] for d in membros}
            if len(candidatos) != 1:
                raise ValueError("Um grupo de B cruza grupos temporais de A.")
            ga = next(iter(candidatos))
            if membros != {d for d, g in grupos_a.items() if g == ga}:
                raise ValueError("A partição temporal de A difere da de B.")
            alinhamento.append(int(ga))
        if len(set(alinhamento)) != len(a["pi_grupo"]):
            raise ValueError("Não há correspondência bijetiva entre grupos.")
        novo = copy.deepcopy(b)
        novo["mu_grupo"] = [a["mu_grupo"][g] for g in alinhamento]
        novo["sigma_grupo"] = [a["sigma_grupo"][g] for g in alinhamento]
        # Remapeia alertas temporais de A aos índices dos grupos de B.
        mapa_inverso = {ga: gb for gb, ga in enumerate(alinhamento)}
        limites = [p for p in b["parametros_no_limite"] if p.startswith("logit_")]
        for p in a["parametros_no_limite"]:
            if p.startswith(("mu_", "log_sigma_")):
                prefixo, ga = p.rsplit("_", 1)
                limites.append(f"{prefixo}_{mapa_inverso[int(ga)]}")
        novo["parametros_no_limite"] = limites
        novo["alerta_limite"] = bool(limites)
        novo["max_tempo_observado_dias"] = a["max_tempo_observado_dias"]
        novo["max_tempo_prob_fonte_dias"] = b["max_tempo_observado_dias"]
        novo["especificacao"] = "p_B_com_sobrevivencia_A_sem_reajuste"
        # Não atribui à recombinação objetivo/iterações de outro ajuste.
        for campo in ("objetivo_medio_penalizado", "n_iteracoes",
                      "regularizacao_prob", "regularizacao_tempo"):
            novo.pop(campo, None)
        novo["treinado_nesta_etapa"] = False
        c02_validar_modelo(novo)
        if not np.array_equal(novo["p_destino"], b["p_destino"]):
            raise RuntimeError("B2 alterou indevidamente p_B.")
        return novo, alinhamento


    def c02_prever(m: dict, idade_seg: float) -> dict:
        if not np.isfinite(idade_seg) or idade_seg < 0:
            raise ValueError("Idade inválida.")
        pi, mu, sigma = [np.asarray(m[k], float) for k in (
            "pi_grupo", "mu_grupo", "sigma_grupo"
        )]
        g = np.asarray(m["grupo"], int)
        log_sf = np.zeros(len(pi), float)
        if idade_seg > 0:
            t = idade_seg / 86400.0
            log_sf = special.log_ndtr(-(np.log(t) - mu) / sigma)
            independente = stats.lognorm.logsf(t, s=sigma, scale=np.exp(mu))
            if not np.allclose(log_sf, independente, rtol=1e-10, atol=1e-8):
                raise ValueError("Sobrevivência difere da verificação independente.")
        log_mass = np.log(pi) + log_sf
        log_s_mix = float(special.logsumexp(log_mass))
        log_q = (log_mass - log_s_mix)[g] + np.log(m["r_destino_no_grupo"])
        q = np.exp(log_q)
        if not np.isfinite(log_q).all() or not np.isclose(q.sum(), 1, atol=1e-9):
            raise ValueError("Distribuição condicional inválida.")
        p0 = np.asarray(m["p_destino"], float)
        if idade_seg == 0 and not np.allclose(q, p0, atol=1e-10, rtol=0):
            raise ValueError("q(0) diferente de p_destino.")
        ordem = np.argsort(-q, kind="mergesort")
        ordem0 = np.argsort(-p0, kind="mergesort")
        return {
            "q": q, "log_q": log_q, "log_s_mistura": log_s_mix,
            "soma_quadrados_q": float(np.dot(q, q)),
            "variacao_temporal_tv": float(np.abs(q - p0).sum() / 2),
            "top5": [m["destinos"][int(j)] for j in ordem[:5]],
            "top1_sem_tempo": m["destinos"][int(ordem0[0])],
        }


    def c02_bootstrap(pdf: pd.DataFrame, replicas: int, seed: int) -> list[dict]:
        """Mesmos clientes/casos em A/B/B2; reamostra clientes, não eventos."""
        rng = np.random.default_rng(seed)
        linhas = []
        comparacoes = (("B_VS_A", C02_A, C02_B), ("B2_VS_A", C02_A, C02_B2),
                       ("B2_VS_B", C02_B, C02_B2))
        for (segmento, idade), sub in pdf.groupby(["segmento", "idade_seg"], sort=True):
            ids = sorted(sub["cd_bv"].unique())
            if replicas * len(ids) > C02_CFG["max_celulas_bootstrap"]:
                raise RuntimeError("Bootstrap excede a proteção de memória do driver.")
            indices = rng.integers(0, len(ids), size=(replicas, len(ids)))
            for metrica in ("top1", "top5", "brier", "logloss"):
                p = sub.pivot(index="cd_bv", columns="variante", values=metrica).reindex(ids)
                if set(p.columns) != {C02_A, C02_B, C02_B2} or p.isna().any().any():
                    raise ValueError("Bootstrap com pares incompletos.")
                n_casos = int(sub.loc[sub.variante.eq(C02_A), "n_casos"].sum())
                for nome, ref, var in comparacoes:
                    a, b = p[ref].to_numpy(float), p[var].to_numpy(float)
                    ganho = a - b if metrica in {"brier", "logloss"} else b - a
                    boot = ganho[indices].mean(axis=1)
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    linhas.append({
                        "segmento": str(segmento), "idade_seg": float(idade),
                        "comparacao": nome, "metrica": metrica, "n_clientes": len(ids),
                        "n_casos": n_casos, "media_ref": float(a.mean()),
                        "media_variante": float(b.mean()), "ganho_medio": float(ganho.mean()),
                        "ganho_ic95_lo": float(lo), "ganho_ic95_hi": float(hi),
                    })
        return linhas


    def c02_autotestes_numericos() -> None:
        a = {
            "formato": "sm_v23_lognormal_grupos", "unidade_tempo": "dias",
            "destinos": ["x", "y", "z"], "grupo": [0, 1, 1],
            "grupos_nomes": ["x", "compartilhado"], "n_grupos": 2,
            "pi_grupo": [0.4, 0.6], "r_destino_no_grupo": [1.0, 0.25, 0.75],
            "p_destino": [0.4, 0.15, 0.45], "mu_grupo": [-3.0, 1.0],
            "sigma_grupo": [0.8, 1.2], "max_tempo_observado_dias": 30.0,
            "parametros_no_limite": ["log_sigma_0"], "alerta_limite": True,
            "estrutura": {"destinos": ["x", "y", "z"], "grupo": [0, 1, 1]},
        }
        # Reordena destinos E os índices de grupo de propósito.
        b = copy.deepcopy(a)
        b.update(destinos=["z", "x", "y"], grupo=[0, 1, 0],
                 grupos_nomes=["compartilhado", "x"], pi_grupo=[0.5, 0.5],
                 r_destino_no_grupo=[0.6, 1.0, 0.4], p_destino=[0.3, 0.5, 0.2],
                 mu_grupo=[-5.0, -8.0], sigma_grupo=[0.3, 0.2],
                 parametros_no_limite=[], alerta_limite=False)
        b["estrutura"] = {"destinos": b["destinos"], "grupo": b["grupo"]}
        antes = c02_hash(b)
        b2, mapa = c02_compor_modelo(b, a)
        assert mapa == [1, 0] and c02_hash(b) == antes
        assert b2["parametros_no_limite"] == ["log_sigma_1"]
        for idade in (0.0, 1800.0, 86400.0, 604800.0):
            r2 = c02_prever(b2, idade)
            t = idade / 86400.0
            sf_a = np.ones(2) if not idade else stats.lognorm.sf(
                t, s=a["sigma_grupo"], scale=np.exp(a["mu_grupo"])
            )
            por_destino = dict(zip(a["destinos"], np.asarray(a["grupo"])))
            massa = np.asarray(b["p_destino"]) * np.array([
                sf_a[por_destino[d]] for d in b["destinos"]
            ])
            assert np.allclose(r2["q"], massa / massa.sum(), atol=1e-12)
            if idade == 0:
                rb = c02_prever(b, idade)
                assert np.array_equal(r2["q"], rb["q"])
                assert r2["top5"] == rb["top5"]
        copia_a, _ = c02_compor_modelo(a, a)
        assert np.allclose(c02_prever(copia_a, 1800)["q"], c02_prever(a, 1800)["q"])
        ruim = copy.deepcopy(a)
        ruim["destinos"] = ["x", "y", "outro"]
        ruim["estrutura"]["destinos"] = ruim["destinos"]
        try:
            c02_compor_modelo(b, ruim)
        except ValueError:
            pass
        else:
            raise AssertionError("Vocabulário incompatível não foi bloqueado.")
        if not np.isfinite(c02_prever(b, 1e6)["log_q"]).all():
            raise AssertionError("Log-space falhou na cauda.")
        print("02C: autotestes numéricos, alinhamento e identidade em idade zero OK.")


    if C02_CFG["autotestes"]:
        c02_autotestes_numericos()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 2. Fontes fixadas e contrato do ajuste
# MAGIC Lê manifesto, modelos e validação; não relê eventos nem reconstrói memória.
# MAGIC A seleção de casos/rotas é a persistida em A/B. A referência de auditoria
# MAGIC não é usada para selecionar clientes, excluir erros ou calibrar parâmetros.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def c02_nome_sql(tabela: str) -> str:
        partes = tabela.split(".")
        if len(partes) != 3 or not all(p.replace("_", "").isalnum() for p in partes):
            raise ValueError("Use nome catalogo.schema.tabela sem crases.")
        return ".".join(f"`{p}`" for p in partes)


    def c02_ler(tabela: str) -> DataFrame:
        if not spark.catalog.tableExists(tabela):
            raise RuntimeError(f"Tabela não encontrada: {tabela}")
        if tabela not in C02_FONTES:
            C02_FONTES[tabela] = int(spark.sql(
                f"DESCRIBE HISTORY {c02_nome_sql(tabela)} LIMIT 1"
            ).first()["version"])
        return spark.read.option("versionAsOf", C02_FONTES[tabela]).table(tabela)


    def c02_colunas(df: DataFrame, nomes: set, rotulo: str) -> None:
        faltam = nomes - set(df.columns)
        if faltam:
            raise RuntimeError(f"{rotulo}: colunas ausentes: {sorted(faltam)}")


    def c02_cache(df: DataFrame) -> DataFrame:
        df = df.persist(StorageLevel.DISK_ONLY)
        df.count()
        C02_CACHES.append(df)
        return df


    def c02_exigir_zero(nome: str, erros: DataFrame) -> None:
        n = erros.count()
        C02_CHECKS.append((nome, int(n), "OK" if n == 0 else "FALHA"))
        if n:
            raise RuntimeError(f"02C interrompida: {nome}; {n} divergências. Sem promoção.")


    def c02_filtro_id(df: DataFrame) -> DataFrame:
        c02_colunas(df, {"id_experimento", "id_ajuste", "hash_ajuste"}, "Artefato")
        return df.filter(
            (F.col("id_experimento") == C02_CFG["id_experimento"])
            & (F.col("id_ajuste") == C02_CFG["id_ajuste"])
        )


    def c02_mostrar(nome: str, df: DataFrame, ordem: list | None = None) -> None:
        C02_RELATORIOS[nome] = df
        print(f"\n{nome}")
        mostrar = df.orderBy(*ordem) if ordem else df
        mostrar = mostrar.drop("cd_bv", "modelo_json")
        # Compacta só a exibição. As tabelas conservam a precisão original.
        for campo in mostrar.schema.fields:
            if campo.dataType.simpleString() in {"double", "float"}:
                valor = F.col(campo.name)
                if campo.name.endswith("_prob"):
                    valor = F.when(valor.isNotNull(), F.format_string("%.6g", valor))
                else:
                    valor = F.round(valor, 6)
                mostrar = mostrar.withColumn(campo.name, valor)
        mostrar.show(C02_CFG["max_linhas_exibir"], truncate=False)


    def c02_autotestes_spark() -> None:
        """Verifica mapas, destino desconhecido e censura no Spark do cluster."""
        dados = [
            ("conhecido", "x", "exata", {"x": (0.25, math.log(0.25))}),
            ("desconhecido", "z", "exata", {"x": (0.25, math.log(0.25))}),
            ("censurado", None, "direita", {"x": (0.25, math.log(0.25))}),
        ]
        teste = spark.createDataFrame(dados, (
            "caso string, destino string, tipo string, "
            "massas map<string,struct<prob:double,log_prob:double>>"
        )).withColumn("real", F.element_at(F.col("massas"), F.col("destino")))
        teste = teste.withColumn("ll", F.when(F.col("tipo") == "exata",
            -F.greatest(F.coalesce(F.col("real.log_prob"), F.lit(math.log(1e-15))),
                        F.lit(math.log(1e-15)))))
        valores = {r["caso"]: r.asDict(recursive=True) for r in teste.collect()}
        assert np.isclose(valores["conhecido"]["ll"], -math.log(0.25))
        assert np.isclose(valores["desconhecido"]["ll"], -math.log(1e-15))
        assert valores["desconhecido"]["real"] is None
        assert valores["censurado"]["real"] is None and valores["censurado"]["ll"] is None
        print("02C: autotestes Spark de mapas, destino sem suporte e censura OK.")


    if C02_CFG["autotestes"]:
        c02_autotestes_spark()


    c02_manifestos = c02_ler(C02_CFG["tabela_ajustes"]).filter(
        (F.col("id_experimento") == C02_CFG["id_experimento"])
        & (F.col("id_ajuste") == C02_CFG["id_ajuste"])
        & (F.col("status") == "CONCLUIDO_TREINO_VALIDACAO")
    ).limit(2).collect()
    if len(c02_manifestos) != 1:
        raise RuntimeError("Exige exatamente um manifesto de treino concluído para este ID.")
    C02_MANIFESTO = c02_manifestos[0].asDict()
    if C02_MANIFESTO["versao_ajuste"] != C02_AJUSTE_SUPORTADO:
        raise RuntimeError("Versão de ajuste não suportada; não adaptar silenciosamente.")
    C02_FIT = json.loads(C02_MANIFESTO["config_json"])
    C02_IDADES = [float(a) for a in C02_FIT["idades_seg"]]
    C02_PISO = float(C02_FIT["logloss_piso"])
    if C02_IDADES != [0.0, 1800.0, 86400.0, 604800.0] or not 0 < C02_PISO < 1:
        raise RuntimeError("Contrato de marcos/piso diferente do ajuste esperado.")
    C02_LOG_PISO = math.log(C02_PISO)
    C02_TAB_MODELOS = C02_FIT["tabelas_saida"]["tabela_modelos"]
    C02_TAB_VALIDACAO = C02_FIT["tabelas_saida"]["tabela_validacao"]
    c02_modelos_raw = c02_filtro_id(c02_ler(C02_TAB_MODELOS)).filter(
        F.col("variante").isin(C02_A, C02_B)
    )
    c02_valid_raw = c02_cache(c02_filtro_id(c02_ler(C02_TAB_VALIDACAO)).filter(
        F.col("variante").isin(C02_A, C02_B)
    ))
    c02_colunas(c02_modelos_raw, {
        "variante", "estado", "contexto_modelo", "nivel", "reuso", "status_modelo",
        "modelo_json", "alerta_limite",
    }, "Modelos")
    C02_VAL_COLS = {
        "cd_bv", "passo", "estado", "contexto_modelo", "origem_tecnica", "idade_seg",
        "tipo_censura", "destino_real", "variante", "rota_modelo", "tem_previsao",
        "suporte_destino", "alerta_limite", "extrapolacao", "top1_previsto", "top5_previsto",
        "acerto_top1", "acerto_top5", "brier", "logloss_clip", "logloss_sem_tempo_clip",
        "acerto_top1_sem_tempo", "nll_clip_aplicado", "prob_destino_real", "variacao_temporal_tv",
    }
    c02_colunas(c02_valid_raw, C02_VAL_COLS, "Validação")
    for df, nome in ((c02_modelos_raw, "modelos"), (c02_valid_raw, "validacao")):
        c02_exigir_zero(f"hash_{nome}", df.filter(
            ~F.col("hash_ajuste").eqNullSafe(F.lit(C02_MANIFESTO["hash_ajuste"]))
        ))
    c02_exigir_zero("duplicidade_modelos", c02_modelos_raw.groupBy(
        "variante", "estado", "contexto_modelo"
    ).count().filter("count != 1"))
    c02_exigir_zero("pares_A_B", c02_valid_raw.groupBy(*C02_CHAVE).agg(
        F.count("*").alias("n"), F.countDistinct("variante").alias("v")
    ).filter("n != 2 OR v != 2"))
    c02_exigir_zero("chaves_invalidas", c02_valid_raw.filter(
        F.col("cd_bv").isNull() | F.col("passo").isNull() | F.col("idade_seg").isNull()
        | ~F.col("idade_seg").isin(C02_IDADES) | F.col("estado").isNull()
        | F.col("contexto_modelo").isNull() | F.col("origem_tecnica").isNull()
        | F.col("tem_previsao").isNull() | F.col("tipo_censura").isNull()
        | ~F.col("tipo_censura").isin("exata", "direita")
        | ((F.col("tipo_censura") == "exata") & F.col("destino_real").isNull())
        | ((F.col("tipo_censura") == "direita") & F.col("destino_real").isNotNull())
    ))
    if not c02_valid_raw.limit(1).count():
        raise RuntimeError("Validação vazia para os IDs informados.")
    print("02C:", C02_ID, "| ajuste fonte:", C02_CFG["id_ajuste"])
    print("Corte preservado:", C02_FIT["fonte_preparo"]["corte_estado_iso"])
    print("Nenhum modelo será retreinado. Piso de log loss preservado:", C02_PISO)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 3. Catálogo B2 e distribuições por modelo/idade
# MAGIC Somente o catálogo pequeno vai ao driver. Casos de validação ficam no Spark.
# MAGIC Limites temporais de A continuam sinalizados; não são removidos.
# MAGIC Sem modelo na validação original continua sem modelo nesta comparação.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    c02_rows = c02_modelos_raw.filter(
        ~F.col("reuso") & (F.col("status_modelo") == "AJUSTADO")
    ).limit(C02_CFG["max_modelos_driver"] + 1).collect()
    if not c02_rows or len(c02_rows) > C02_CFG["max_modelos_driver"]:
        raise RuntimeError("Catálogo vazio ou maior que a proteção do driver.")
    C02_MODELOS = {}
    for row in c02_rows:
        m = json.loads(row["modelo_json"])
        c02_validar_modelo(m)
        if bool(m["alerta_limite"]) != bool(row["alerta_limite"]):
            raise RuntimeError("Alerta externo/JSON divergente.")
        chave = (row["variante"], row["estado"], row["contexto_modelo"])
        if chave[0] == C02_A and (chave[2] != C02_BASE or row["nivel"] != "ORIGEM"):
            raise RuntimeError("Referência A fora do nível ORIGEM.")
        if chave[0] == C02_B and row["nivel"] != "CONTEXTO":
            raise RuntimeError("B não reutilizado deve ser CONTEXTO.")
        C02_MODELOS[chave] = m

    c02_candidatos = []
    C02_COMPOSTOS = {}
    for (familia, estado, contexto), b in C02_MODELOS.items():
        if familia != C02_B:
            continue
        a = C02_MODELOS.get((C02_A, estado, C02_BASE))
        if a is None:
            raise RuntimeError(f"Contexto sem pai A: {estado} / {contexto}")
        b2, mapa = c02_compor_modelo(b, a)
        b2["componentes_02c"] = {
            "id_ajuste_fonte": C02_CFG["id_ajuste"], "estado": estado,
            "contexto_prob": contexto, "familia_prob": C02_B,
            "contexto_tempo": C02_BASE, "familia_tempo": C02_A,
            "hash_modelo_prob": c02_hash(b), "hash_modelo_tempo": c02_hash(a),
            "grupo_B_para_A": mapa,
        }
        r0, rb0 = c02_prever(b2, 0.0), c02_prever(b, 0.0)
        if not np.array_equal(r0["q"], rb0["q"]) or r0["top5"] != rb0["top5"]:
            raise RuntimeError("B2 não reproduz B exatamente em idade zero.")
        C02_COMPOSTOS[(C02_B2, estado, contexto)] = b2
        c02_candidatos.append((
            C02_B2, estado, contexto, "CONTEXTO", "COMPOSTO_SEM_RETREINO",
            c02_hash(b), c02_hash(a), mapa, b2["alerta_limite"],
            b2["parametros_no_limite"], json.dumps(b2, ensure_ascii=False, allow_nan=False),
        ))
    C02_MODELOS.update(C02_COMPOSTOS)
    C02_MODELOS_SCHEMA = (
        "variante string, estado string, contexto_modelo string, nivel string, "
        "status_modelo string, hash_prob_fonte string, hash_tempo_fonte string, "
        "grupo_B_para_A array<int>, alerta_limite boolean, "
        "parametros_no_limite array<string>, modelo_json string"
    )
    c02_modelos_b2 = spark.createDataFrame(c02_candidatos, C02_MODELOS_SCHEMA)
    C02_CHECKS.append(("catalogo_B2_p0_e_alinhamento", 0, "OK"))

    C02_KERNEL_SCHEMA = (
        "familia_kernel string, estado string, contexto_kernel string, idade_seg double, "
        "kernel_ok boolean, alerta_limite_calc boolean, alerta_prob_fonte boolean, "
        "alerta_tempo_fonte boolean, parametros_limite_calc array<string>, "
        "max_tempo_prob_fonte_dias double, max_tempo_tempo_fonte_dias double, "
        "top5_calc array<string>, top1_sem_tempo_calc string, soma_quadrados_q double, "
        "variacao_temporal_calc double, log_s_mistura double, "
        "destinos_calc map<string,struct<prob:double,log_prob:double,log_p0:double>>"
    )
    entradas = sum(len(m["destinos"]) for m in C02_MODELOS.values()) * len(C02_IDADES)
    if entradas > C02_CFG["max_entradas_kernels"]:
        raise RuntimeError("Catálogo de distribuições excede a proteção; não foi truncado.")
    c02_kernels_rows = []
    for (familia, estado, contexto), m in C02_MODELOS.items():
        prob_fonte = C02_MODELOS[(C02_B, estado, contexto)] if familia == C02_B2 else m
        tempo_fonte = C02_MODELOS[(C02_A, estado, C02_BASE)] if familia == C02_B2 else m
        alerta_prob = any(p.startswith("logit_") for p in prob_fonte["parametros_no_limite"])
        alerta_tempo = any(p.startswith(("mu_", "log_sigma_"))
                           for p in tempo_fonte["parametros_no_limite"])
        for idade in C02_IDADES:
            r = c02_prever(m, idade)
            destinos = {d: (float(r["q"][j]), float(r["log_q"][j]),
                            float(np.log(m["p_destino"][j])))
                        for j, d in enumerate(m["destinos"])}
            c02_kernels_rows.append((
                familia, estado, contexto, idade, True, m["alerta_limite"],
                alerta_prob, alerta_tempo, m["parametros_no_limite"],
                float(prob_fonte["max_tempo_observado_dias"]),
                float(tempo_fonte["max_tempo_observado_dias"]),
                r["top5"], r["top1_sem_tempo"], r["soma_quadrados_q"],
                r["variacao_temporal_tv"], r["log_s_mistura"], destinos,
            ))
    c02_kernels = spark.createDataFrame(c02_kernels_rows, C02_KERNEL_SCHEMA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 4. Reavaliar A/B e B2 sem mudar casos, rotas ou cobertura
# MAGIC A validação contém exatas e censuras. Censura recebe previsão, mas NÃO
# MAGIC um rótulo de destino; só as exatas entram nas métricas de classificação.
# MAGIC Destino desconhecido: p=0 e log loss no mesmo piso de antes.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    c02_base_a = c02_valid_raw.filter(F.col("variante") == C02_A)
    c02_base_b = c02_valid_raw.filter(F.col("variante") == C02_B)
    c02_meta = ["estado", "contexto_modelo", "origem_tecnica", "tipo_censura", "destino_real"]
    c02_meta_pares = c02_base_a.select(
        *C02_CHAVE, *[F.col(c).alias(f"a_{c}") for c in c02_meta]
    ).join(c02_base_b.select(*C02_CHAVE, *c02_meta), C02_CHAVE, "inner")
    c02_dif = F.lit(False)
    for c in c02_meta:
        c02_dif = c02_dif | ~F.col(c).eqNullSafe(F.col(f"a_{c}"))
    c02_exigir_zero("mesmas_observacoes_A_B", c02_meta_pares.filter(c02_dif))
    c02_exigir_zero("rotas_invalidas_A", c02_base_a.filter(
        F.col("rota_modelo").isNull()
        | ~F.col("rota_modelo").isin("ORIGEM_REFERENCIA", "SEM_MODELO_ORIGEM")
    ))
    C02_ROTAS_B = ["CONTEXTO_AJUSTADO", "ORIGEM_NAO_TECNICA", "ORIGEM_SEM_MEMORIA",
                   "ORIGEM_CONTEXTO_RARO", "ORIGEM_CONTEXTO_NAO_VISTO",
                   "ORIGEM_FALHA_AJUSTE_CONTEXTO", "SEM_MODELO_ORIGEM"]
    c02_exigir_zero("rotas_invalidas_B", c02_base_b.filter(
        F.col("rota_modelo").isNull() | ~F.col("rota_modelo").isin(C02_ROTAS_B)
        | ((F.col("rota_modelo") == "CONTEXTO_AJUSTADO") & ~F.col("origem_tecnica"))
    ))
    c02_entrada = c02_valid_raw.select(*sorted(C02_VAL_COLS)).unionByName(
        c02_base_b.select(*sorted(C02_VAL_COLS)).withColumn("variante", F.lit(C02_B2))
    ).withColumn("familia_kernel", F.when(
        F.col("rota_modelo") == "CONTEXTO_AJUSTADO", F.col("variante")
    ).otherwise(F.lit(C02_A))).withColumn("contexto_kernel", F.when(
        F.col("rota_modelo") == "CONTEXTO_AJUSTADO", F.col("contexto_modelo")
    ).otherwise(F.lit(C02_BASE)))
    c02_join = c02_entrada.join(
        F.broadcast(c02_kernels), ["familia_kernel", "estado", "contexto_kernel", "idade_seg"], "left"
    )
    c02_exigir_zero("disponibilidade_modelos", c02_join.filter(
        F.col("tem_previsao") != F.coalesce(F.col("kernel_ok"), F.lit(False))
    ))
    c02_join = c02_join.withColumn("_real", F.element_at(
        F.col("destinos_calc"), F.col("destino_real")
    )).withColumn("_prob", F.coalesce(F.col("_real.prob"), F.lit(0.0)))
    c02_exata = F.col("tem_previsao") & (F.col("tipo_censura") == "exata")
    c02_join = (
        c02_join
        .withColumn("calc_prob", F.when(c02_exata, F.col("_prob")))
        .withColumn("calc_suporte", F.when(c02_exata, F.col("_real").isNotNull()))
        .withColumn("calc_top1", F.when(c02_exata,
            (F.col("destino_real") == F.element_at("top5_calc", 1)).cast("double")))
        .withColumn("calc_top5", F.when(c02_exata,
            F.expr("array_contains(top5_calc, destino_real)").cast("double")))
        .withColumn("calc_brier", F.when(c02_exata,
            F.col("soma_quadrados_q") - 2 * F.col("_prob") + 1))
        .withColumn("calc_logloss", F.when(c02_exata,
            -F.greatest(F.coalesce(F.col("_real.log_prob"), F.lit(C02_LOG_PISO)), F.lit(C02_LOG_PISO))))
        .withColumn("calc_logloss_0", F.when(c02_exata,
            -F.greatest(F.coalesce(F.col("_real.log_p0"), F.lit(C02_LOG_PISO)), F.lit(C02_LOG_PISO))))
        .withColumn("calc_clip", F.when(c02_exata,
            F.col("_real").isNull() | (F.col("_real.log_prob") < C02_LOG_PISO)))
        .withColumn("calc_top1_0", F.when(c02_exata,
            (F.col("destino_real") == F.col("top1_sem_tempo_calc")).cast("double")))
        .withColumn("extrap_prob_fonte", F.coalesce(
            F.col("idade_seg") / 86400 > F.col("max_tempo_prob_fonte_dias"), F.lit(False)))
        .withColumn("extrap_tempo_fonte", F.coalesce(
            F.col("idade_seg") / 86400 > F.col("max_tempo_tempo_fonte_dias"), F.lit(False)))
    )
    c02_join = c02_cache(c02_join.drop("destinos_calc"))

    # Confere A/B contra TODOS os valores originais, antes de comparar B2.
    c02_replay = c02_join.filter(F.col("variante").isin(C02_A, C02_B))
    c02_metricas_replay = {
        "prob_destino_real": "calc_prob", "brier": "calc_brier", "logloss_clip": "calc_logloss",
        "logloss_sem_tempo_clip": "calc_logloss_0", "acerto_top1": "calc_top1",
        "acerto_top5": "calc_top5", "acerto_top1_sem_tempo": "calc_top1_0",
        "variacao_temporal_tv": "variacao_temporal_calc",
    }
    c02_erros_replay = []
    for original, calculado in c02_metricas_replay.items():
        tol = C02_CFG["tol_prob"] if original == "prob_destino_real" else C02_CFG["tol_metrica"]
        erro = (F.col(original).isNull() != F.col(calculado).isNull()) | (
            F.col(original).isNotNull() & ((F.abs(F.col(original) - F.col(calculado)) > tol)
                                           | F.isnan(calculado)))
        c02_erros_replay.append(F.sum(erro.cast("long")).alias(original))
    for original, calculado in (("suporte_destino", "calc_suporte"),
                                ("nll_clip_aplicado", "calc_clip")):
        c02_erros_replay.append(F.sum(
            (~F.col(original).eqNullSafe(F.col(calculado))).cast("long")
        ).alias(original))
    c02_erros_replay += [F.sum((F.col("tem_previsao") & (
        ~F.col("top5_previsto").eqNullSafe(F.col("top5_calc"))
        | ~F.col("top1_previsto").eqNullSafe(F.element_at("top5_calc", 1))
        | ~F.col("alerta_limite").eqNullSafe(F.col("alerta_limite_calc"))
        | ~F.col("extrapolacao").eqNullSafe(F.col("extrap_tempo_fonte"))
    )).cast("long")).alias("ranking_flags")]
    for nome, n in c02_replay.agg(*c02_erros_replay).first().asDict().items():
        C02_CHECKS.append((f"reproducao_{nome}", int(n or 0), "OK" if not n else "FALHA"))
        if n:
            raise RuntimeError(f"A/B não reproduzem a validação original: {nome}, n={n}")

    c02_validacao = c02_join.select(
        *C02_CHAVE, "estado", "contexto_modelo", "origem_tecnica", "tipo_censura", "destino_real",
        "variante", F.col("rota_modelo").alias("rota_modelo_original"),
        "familia_kernel", "contexto_kernel",
        "tem_previsao", F.col("calc_suporte").alias("suporte_destino"),
        F.when(F.col("tem_previsao"), F.col("familia_kernel") != C02_A).alias("usa_prob_contextual"),
        F.when(F.col("tem_previsao"), F.when(F.col("variante") == C02_B2, C02_A)
               .otherwise(F.col("familia_kernel"))).alias("familia_tempo_utilizada"),
        "alerta_prob_fonte", "alerta_tempo_fonte", "parametros_limite_calc",
        "extrap_prob_fonte", "extrap_tempo_fonte",
        F.element_at("top5_calc", 1).alias("top1_previsto"),
        F.coalesce("top5_calc", F.array().cast("array<string>")).alias("top5_previsto"),
        F.col("calc_top1").alias("acerto_top1"), F.col("calc_top5").alias("acerto_top5"),
        F.col("calc_brier").alias("brier"), F.col("calc_logloss").alias("logloss_clip"),
        F.col("calc_logloss_0").alias("logloss_sem_tempo_clip"),
        F.col("calc_clip").alias("nll_clip_aplicado"),
        F.col("calc_prob").alias("prob_destino_real"),
        F.when(c02_exata, F.col("_real.log_prob")).alias("log_prob_destino_real"),
        F.col("variacao_temporal_calc").alias("variacao_temporal_tv"), "log_s_mistura",
    )
    c02_validacao = c02_cache(c02_validacao)
    c02_exigir_zero("tres_variantes_mesma_cobertura", c02_validacao.groupBy(*C02_CHAVE).agg(
        F.count("*").alias("n"), F.countDistinct("variante").alias("v"),
        F.countDistinct("tem_previsao").alias("p")
    ).filter("n != 3 OR v != 3 OR p != 1"))
    # Identidade em idade zero inclui rankings, métricas e também casos censurados.
    c02_b_zero = c02_validacao.filter((F.col("variante") == C02_B) & (F.col("idade_seg") == 0))
    c02_b2_zero = c02_validacao.filter((F.col("variante") == C02_B2) & (F.col("idade_seg") == 0))
    c02_zero_cols = [
        "top5_previsto", "prob_destino_real", "brier", "logloss_clip", "acerto_top1", "acerto_top5",
    ]
    c02_zero = c02_b_zero.select(*C02_CHAVE, *c02_zero_cols).join(
        c02_b2_zero.select(*C02_CHAVE, *[F.col(c).alias(f"b2_{c}") for c in c02_zero_cols]),
        C02_CHAVE, "inner",
    )
    c02_zero_erro = F.lit(False)
    for c in c02_zero_cols:
        c02_zero_erro = c02_zero_erro | ~F.col(c).eqNullSafe(F.col(f"b2_{c}"))
    c02_exigir_zero("B2_igual_B_em_idade_zero", c02_zero.filter(c02_zero_erro))
    c02_b2_fallback = c02_validacao.filter(
        (F.col("variante") == C02_B2)
        & (F.col("rota_modelo_original") != "CONTEXTO_AJUSTADO")
    ).select(*C02_CHAVE, *[F.col(c).alias(f"b2_{c}") for c in c02_zero_cols])
    c02_a_fallback = c02_validacao.filter(F.col("variante") == C02_A).select(
        *C02_CHAVE, *c02_zero_cols
    ).join(c02_b2_fallback, C02_CHAVE, "inner")
    c02_exigir_zero("B2_igual_A_nas_rotas_de_reuso", c02_a_fallback.filter(c02_zero_erro))
    c02_checks = spark.createDataFrame(
        C02_CHECKS, "verificacao string, n_divergencias long, status string"
    )
    c02_mostrar("V23_02C_01_INTEGRIDADE", c02_checks)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 5. Métricas e comparação pareada
# MAGIC Sem previsões: ficam na cobertura, não nas médias de qualidade.
# MAGIC Sem suporte ao destino: penalização mantida, sem exclusão das métricas.
# MAGIC Brier/log loss menores são melhores; Top 1/5 maiores são melhores.
# MAGIC Ganho positivo nos pares = melhora. ICs não são corrigidos por múltiplas
# MAGIC comparações e descrevem este holdout de DESENVOLVIMENTO já examinado.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    def c02_segmentos(df: DataFrame) -> DataFrame:
        return df.withColumn("segmento", F.explode(F.when(
            F.col("origem_tecnica"), F.array(F.lit("TODAS"), F.lit("ORIGENS_TECNICAS"))
        ).otherwise(F.array(F.lit("TODAS")))))


    c02_rotas = c02_validacao.groupBy("variante", "idade_seg", "rota_modelo_original").agg(
        F.count("*").alias("n_observacoes"), F.countDistinct("cd_bv").alias("n_clientes")
    )
    c02_mostrar("V23_02C_02_ROTAS", c02_rotas, ["idade_seg", "variante", "rota_modelo_original"])
    c02_exatas = c02_cache(c02_validacao.filter(F.col("tipo_censura") == "exata"))
    c02_seg = c02_segmentos(c02_exatas)
    c02_metricas = c02_seg.groupBy("segmento", "variante", "idade_seg").agg(
        F.count("*").alias("n_eventos"), F.sum(F.col("tem_previsao").cast("long")).alias("n_previstos"),
        F.sum(F.coalesce(F.col("suporte_destino").cast("long"), F.lit(0))).alias("n_suportados"),
        F.avg("acerto_top1").alias("top1"), F.avg("acerto_top5").alias("top5"),
        F.avg("brier").alias("brier"), F.avg("logloss_clip").alias("logloss"),
        F.sum(F.coalesce(F.col("nll_clip_aplicado").cast("long"), F.lit(0))).alias("n_logloss_clip"),
        F.avg(F.col("logloss_sem_tempo_clip") - F.col("logloss_clip")).alias("ganho_logloss_tempo"),
    ).withColumn("cobertura_modelo", F.col("n_previstos") / F.col("n_eventos"))
    c02_mostrar("V23_02C_03_METRICAS", c02_metricas.filter(
        F.col("segmento") == "ORIGENS_TECNICAS"
    ).select(
        "variante", "idade_seg", "n_eventos", "n_previstos", "top1", "top5",
        "brier", "logloss", "n_logloss_clip",
    ),
        ["idade_seg", "variante"])
    C02_RELATORIOS["V23_02C_03_METRICAS_COMPLETAS"] = c02_metricas
    # Mesmos casos com A/B/B2; ausência numérica causaria interrupção acima.
    c02_pc = c02_segmentos(c02_exatas.filter("tem_previsao")).groupBy(
        "segmento", "idade_seg", "cd_bv", "variante"
    ).agg(F.count("*").alias("n_casos"), F.avg("acerto_top1").alias("top1"),
          F.avg("acerto_top5").alias("top5"), F.avg("brier").alias("brier"),
          F.avg("logloss_clip").alias("logloss"))
    c02_pc_rows = c02_pc.limit(C02_CFG["max_agregados_driver"] + 1).collect()
    if not c02_pc_rows or len(c02_pc_rows) > C02_CFG["max_agregados_driver"]:
        raise RuntimeError("Bootstrap vazio ou acima da proteção; nenhum evento foi amostrado.")
    # Apenas médias por cliente vão ao Pandas. Não exige Arrow/mapInPandas.
    c02_pdf_pc = pd.DataFrame([r.asDict() for r in c02_pc_rows])
    c02_boot = c02_bootstrap(c02_pdf_pc, C02_CFG["bootstrap_replicas"], C02_CFG["semente"])
    C02_PARES_SCHEMA = (
        "segmento string, idade_seg double, comparacao string, metrica string, n_clientes long, "
        "n_casos long, media_ref double, media_variante double, ganho_medio double, "
        "ganho_ic95_lo double, ganho_ic95_hi double"
    )
    c02_pares = spark.createDataFrame(c02_boot, C02_PARES_SCHEMA)
    for comp in ("B_VS_A", "B2_VS_A", "B2_VS_B"):
        c02_mostrar(f"V23_02C_04_{comp}", c02_pares.filter(
            (F.col("segmento") == "ORIGENS_TECNICAS") & (F.col("comparacao") == comp)
        ).drop("segmento", "comparacao"), ["idade_seg", "metrica"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 6. Probabilidades extremas e casos críticos
# MAGIC Mesma triagem para todas as variantes. Não aumenta o piso da métrica.
# MAGIC Os casos críticos são escolhidos pela piora ORIGINAL B−A, não pelo
# MAGIC resultado de B2. Servem à investigação; a comparação global usa todos.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    c02_extremos = c02_seg.filter("tem_previsao").groupBy("segmento", "idade_seg", "variante").agg(
        F.count("*").alias("n_casos"), F.countDistinct("cd_bv").alias("n_clientes"),
        F.sum((~F.col("suporte_destino")).cast("long")).alias("n_destinos_sem_suporte"),
        F.sum(F.col("nll_clip_aplicado").cast("long")).alias("n_logloss_clip"),
        F.sum((F.col("prob_destino_real") < C02_CFG["prob_baixa_triagem"])
              .cast("long")).alias("n_prob_muito_baixa"),
        F.sum((F.col("suporte_destino") & (F.col("prob_destino_real") < C02_CFG["prob_baixa_triagem"]))
              .cast("long")).alias("n_baixas_com_suporte"),
        F.sum(F.col("extrap_tempo_fonte").cast("long")).alias("n_extrap_tempo_fonte"),
    )
    c02_mostrar("V23_02C_05_EXTREMOS", c02_extremos.filter(
        F.col("segmento") == "ORIGENS_TECNICAS"
    ).drop("segmento"), ["idade_seg", "variante"])
    C02_RELATORIOS["V23_02C_05_EXTREMOS_COMPLETOS"] = c02_extremos
    c02_wide = c02_exatas.filter("tem_previsao").groupBy(
        *C02_CHAVE, "estado", "contexto_modelo", "origem_tecnica", "destino_real"
    ).agg(*[
        F.max(F.when(F.col("variante") == variante, F.col(coluna))).alias(f"{curto}_{sufixo}")
        for variante, curto in ((C02_A, "a"), (C02_B, "b"), (C02_B2, "b2"))
        for coluna, sufixo in (("prob_destino_real", "prob"), ("log_prob_destino_real", "log_prob"),
                               ("logloss_clip", "logloss"))
    ]).withColumn("delta_b_a", F.col("b_logloss") - F.col("a_logloss")).withColumn(
        "delta_b2_a", F.col("b2_logloss") - F.col("a_logloss")
    ).withColumn("delta_b2_b", F.col("b2_logloss") - F.col("b_logloss"))
    w_critico = Window.partitionBy("idade_seg").orderBy(F.desc("delta_b_a"), "cd_bv", "passo")
    c02_criticos = c02_wide.filter(
        F.col("origem_tecnica") & (F.col("idade_seg") > 0) & (F.col("delta_b_a") > 0)
    ).withColumn("ranking_piora_original", F.row_number().over(w_critico)).filter(
        F.col("ranking_piora_original") <= C02_CFG["casos_criticos_por_idade"]
    )
    c02_mostrar("V23_02C_06_CASOS_CRITICOS", c02_criticos.select(
        "idade_seg", "ranking_piora_original", "estado", "contexto_modelo", "destino_real",
        "a_prob", "b_prob", "b2_prob", "delta_b2_a", "delta_b2_b"
    ), ["idade_seg", "ranking_piora_original"])
    # Esta tabela interna inclui IDs; não os imprime nos relatórios agregados.
    C02_RELATORIOS["V23_02C_06_CASOS_CRITICOS_COMPLETOS"] = c02_criticos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Etapa opcional — 7. Persistência isolada e conclusão
# MAGIC Novas tabelas 02C, append por id_02c; sem promoção automática.
# MAGIC As escritas são separadas. Consuma somente IDs com manifesto concluído.
# MAGIC Falha/reexecução: executar o notebook inteiro cria outro id_02c.

# COMMAND ----------

if CONS_CFG['modo'] == 'REAJUSTAR_CONGELADO':
    C02_CONFIG_EXEC = {
        "versao": C02_VERSAO, "config": C02_CFG, "fontes_delta": C02_FONTES,
        "hash_ajuste_fonte": C02_MANIFESTO["hash_ajuste"],
        "corte_estado_iso": C02_FIT["fonte_preparo"]["corte_estado_iso"],
        "logloss_piso": C02_PISO, "idades_seg": C02_IDADES,
        "coorte": "todos_os_casos_A_B_persistidos_sem_nova_amostragem",
        "regra_B2": "p_B_com_S_A; fallback_B_preservado; grupos_alinhados_por_destino",
        "estatistica": "media_por_cliente;bootstrap_pareado;IC95_nao_ajustado_multiplicidade",
        "spark": spark.version, "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sem_retreino": True,
    }
    C02_HASH = c02_hash(C02_CONFIG_EXEC)
    C02_SAIDAS = {nome: f"{C02_CFG['prefixo_saida']}_{nome}_hml" for nome in (
        "modelos", "validacao", "comparacoes", "resumos", "execucoes"
    )}
    if len(set(C02_SAIDAS.values())) != 5 or set(C02_SAIDAS.values()) & set(C02_FONTES):
        raise RuntimeError("Uma saída coincide com outra saída ou com uma fonte.")
    for tabela in C02_SAIDAS.values():
        c02_nome_sql(tabela)


    def c02_gravar(df: DataFrame, nome: str) -> None:
        tabela = C02_SAIDAS[nome]
        saida = (df.withColumn("id_experimento", F.lit(C02_CFG["id_experimento"]))
                 .withColumn("id_ajuste_fonte", F.lit(C02_CFG["id_ajuste"]))
                 .withColumn("id_02c", F.lit(C02_ID))
                 .withColumn("versao_02c", F.lit(C02_VERSAO))
                 .withColumn("hash_02c", F.lit(C02_HASH)))
        if spark.catalog.tableExists(tabela):
            existente = spark.table(tabela)
            c02_colunas(existente, {"id_02c"}, tabela)
            if existente.filter(F.col("id_02c") == C02_ID).limit(1).count():
                raise RuntimeError("Etapa já escrita neste id_02c; reexecute com novo ID.")
        saida.write.format("delta").mode("append").saveAsTable(tabela)


    c02_resumos = None
    for nome, df in C02_RELATORIOS.items():
        lote = df.select(F.lit(nome).alias("relatorio"), F.to_json(
            F.struct(*[F.col(c) for c in df.columns]), {"ignoreNullFields": "false"}
        ).alias("dados_json"))
        c02_resumos = lote if c02_resumos is None else c02_resumos.unionByName(lote)
    if C02_CFG["gravar"]:
        c02_gravar(c02_modelos_b2, "modelos")
        c02_gravar(c02_validacao, "validacao")
        c02_gravar(c02_pares, "comparacoes")
        c02_gravar(c02_resumos, "resumos")
        c02_fim = spark.createDataFrame([(
            "CONCLUIDO_COMPARACAO_02C", False, False, C02_INICIO,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(C02_CONFIG_EXEC, ensure_ascii=False, sort_keys=True, allow_nan=False),
            json.dumps(C02_SAIDAS, ensure_ascii=False, sort_keys=True),
        )], "status string, publicavel boolean, houve_retreino boolean, inicio_utc string, "
            "fim_utc string, config_json string, tabelas_json string")
        c02_gravar(c02_fim, "execucoes")
        print("\nV23_02C_07_CONCLUSAO: comparação persistida, sem promoção.")
    else:
        print("\nV23_02C_07_CONCLUSAO: apenas em memória, sem manifesto persistido.")
    print("id_02c:", C02_ID)
    print("id_ajuste_fonte:", C02_CFG["id_ajuste"])
    print("Tabelas 02C:", json.dumps(C02_SAIDAS, indent=2))
    print("Nenhum modelo foi treinado; nenhuma tabela anterior foi sobrescrita.")
    print("Próximo passo: analisar A/B/B2 antes de decidir sobre Multi-step.")
    for df in C02_CACHES:
        df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Pacote independente e avaliação Multi-step
# MAGIC
# MAGIC No modo de preservação, copia os artefatos aprovados sem alterar parâmetros.
# MAGIC No reajuste, recupera somente o ID que acabou de terminar, usando as fontes novas.

# COMMAND ----------

if CONS_CFG["modo"] == "PRESERVAR_REFERENCIA":
    entradas = CONS_REF_ENTRADAS
    ret_meta = copy.deepcopy(CONS_REF_META)
else:
    MS03_CFG["id_experimento"] = ID_EXPERIMENTO
    MS03_CFG["id_ajuste"] = ID_AJUSTE
    MS03_CFG["id_02c_preferido"] = C02_ID
    MS03_CFG["tabela_execucoes_02c"] = C02_SAIDAS["execucoes"]
    entradas, ret_meta = retomar_entradas()

# Nova cópia física; a referência anterior não é escrita nem substituída.
ret_meta["id_pacote"] = str(uuid.uuid4())
ret_meta["consolidacao"] = {
    "id_consolidacao": CONS_ID,
    "modo": CONS_CFG["modo"],
    "id_ms03_referencia": CONS_CFG["id_ms03_referencia"],
    "id_pacote_referencia": CONS_REF_META["id_pacote"],
    "amostra_preservada": True,
}
for campo in ("id_experimento", "id_ajuste", "id_02c"):
    MS03_CFG[campo] = ret_meta[campo]
ret_validar_entradas(entradas, ret_meta)
ret_meta = ret_gravar_pacote(entradas, ret_meta)

# COMMAND ----------

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
print("Corte histórico:", preparo["corte_estado_iso"], "| etapa Multi-step sem ajuste adicional.")


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
    "publicavel": False, "sem_retreino": CONS_CFG["modo"] == "PRESERVAR_REFERENCIA", "uso": "EXPERIMENTO_HISTORICO",
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
# MAGIC Brier por ações: soma de 17 erros quadráticos (um indicador por alvo),
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
# MAGIC ## 5. Equivalência com a referência — antes de concluir
# MAGIC
# MAGIC No modo padrão exige mesmas chaves, labels, ranks, mapas e probabilidades.
# MAGIC Tolerância numérica explícita; nenhuma renormalização é usada para passar no teste.
# MAGIC Uma falha deixa somente saídas parciais em prefixo isolado, sem conclusão.
# MAGIC No reajuste, diferenças são esperadas e não resultam em promoção automática.

# COMMAND ----------

def cons_comparar_frames(
    novo: DataFrame,
    referencia: DataFrame,
    chaves: list[str],
    exatas: list[str],
    numericas: list[str],
    rotulo: str,
) -> dict:
    campos = chaves + exatas + numericas
    ms03_cols(novo, set(campos), rotulo + " novo")
    ms03_cols(referencia, set(campos), rotulo + " referência")
    for df, nome in ((novo, "novo"), (referencia, "referencia")):
        ms03_zero(rotulo + "_chaves_" + nome,
                  df.groupBy(*chaves).count().filter("count != 1"))
    ms03_zero(rotulo + "_faltantes",
              referencia.select(*chaves).join(novo.select(*chaves), chaves, "left_anti"))
    ms03_zero(rotulo + "_excedentes",
              novo.select(*chaves).join(referencia.select(*chaves), chaves, "left_anti"))
    cmp = novo.select(*campos).alias("n").join(
        referencia.select(*campos).alias("r"), chaves, "inner"
    )
    diferente = F.lit(False)
    for campo in exatas:
        diferente = diferente | ~F.col("n." + campo).eqNullSafe(F.col("r." + campo))
    for campo in numericas:
        n, r = F.col("n." + campo), F.col("r." + campo)
        igual = n.eqNullSafe(r) | (
            n.isNotNull() & r.isNotNull() & ~F.isnan(n) & ~F.isnan(r)
            & (F.abs(n - r) <= F.lit(CONS_CFG["tolerancia_reproducao"]))
        )
        diferente = diferente | ~F.coalesce(igual, F.lit(False))
    ms03_zero(rotulo + "_valores", cmp.filter(diferente))
    return {"verificacao": rotulo, "status": "OK", "n_linhas": novo.count()}


CONS_EQUIVALENCIA = []
if CONS_CFG["modo"] == "PRESERVAR_REFERENCIA":
    ref_k = cons_ler_atual(CONS_REF_TABELAS["kernels"]).filter(
        F.col("id_ms03") == CONS_CFG["id_ms03_referencia"]
    )
    kkeys = ["variante", "estado", "memoria_grafo", "idade_seg"]
    # Rank e ação são exatos. Valores numéricos são conferidos nos mapas completos.
    def ranks_df(df):
        return df.withColumn("_raw_ranks", F.expr(
            "transform(raw_top5, x -> named_struct('ranking', x.ranking, 'estado', x.estado))"
        )).withColumn("_util_ranks", F.expr(
            "transform(util_top5, x -> named_struct('ranking', x.ranking, 'acao', x.acao, "
            "'passo', x.passo_macro_maior_contribuicao))"
        ))
    CONS_EQUIVALENCIA.append(cons_comparar_frames(
        ranks_df(kernels), ranks_df(ref_k), kkeys,
        ["tem_previsao", "rota_modelo", "_raw_ranks", "_util_ranks",
         "alerta_limite", "extrapolacao_tempo"],
        ["soma_quadrados_util", "massa_acao_util", "massa_sem_modelo", "massa_terminal",
         "massa_retencao_interna", "massa_limite_macro", "prob_sem_alvo_modelada"],
        "kernels_e_rankings",
    ))
    for mapa in ("raw_probs", "util_probs"):
        n = kernels.select(*kkeys, F.explode(mapa).alias("acao", "p"))
        r = ref_k.select(*kkeys, F.explode(mapa).alias("acao", "p"))
        CONS_EQUIVALENCIA.append(cons_comparar_frames(
            n, r, kkeys + ["acao"], [], ["p"], "scores_" + mapa
        ))
    ref_p = cons_ler_atual(CONS_REF_TABELAS["predicoes"]).filter(
        F.col("id_ms03") == CONS_CFG["id_ms03_referencia"]
    )
    CONS_EQUIVALENCIA.append(cons_comparar_frames(
        pred, ref_p, ["cd_bv", "passo", "idade_seg", "variante"],
        ["status_label", "acao_real", "estado", "memoria_grafo", "tem_previsao"],
        ["top1", "top5", "mrr", "brier_acoes", "brier_algum_alvo", "logloss_conservador"],
        "labels_e_metricas",
    ))
    print("Referência reproduzida nos kernels e nas previsões históricas.")
else:
    print("Reajuste concluído como CANDIDATA. Não houve promoção nem exigência de scores idênticos.")

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
        "CONCLUIDO_MULTISTEP_HISTORICO_03", False, CONS_CFG["modo"] == "REAJUSTAR_CONGELADO", MS03_INICIO,
        datetime.now(timezone.utc).isoformat(),
        json.dumps(MS03_CONFIG_EXEC, sort_keys=True, ensure_ascii=False),
        json.dumps(MS03_SAIDAS, sort_keys=True),
    )], "status string, publicavel boolean, houve_retreino boolean, inicio_utc string, "
        "fim_utc string, config_json string, tabelas_json string")
    ms03_guardar(fim, "execucoes")
print("\nV23_MS03_10_CONCLUSAO")
print("id_ms03:", MS03_ID)
print("Saídas:", json.dumps(MS03_SAIDAS, indent=2))
print("Modo:", CONS_CFG["modo"], "| Nenhuma tabela aprovada foi sobrescrita.")
print("Validação histórica concluída. A tabela de negócio anterior permanece intacta.")
print("Consuma somente ID com CONCLUIDO_MULTISTEP_HISTORICO_03 no manifesto.")
for df in MS03_CACHES:
    df.unpersist()

# COMMAND ----------

import scipy

CONS_RESULTADO = {
    "id_consolidacao": CONS_ID,
    "modo": CONS_CFG["modo"],
    "id_ms03_referencia": CONS_CFG["id_ms03_referencia"],
    "id_ms03_gerado": MS03_ID,
    "id_pacote_gerado": ret_meta["id_pacote"],
    "tabela_ms03_execucoes": MS03_SAIDAS["execucoes"],
    "tabela_pacotes": RET_PACOTES,
    "equivalencia": CONS_EQUIVALENCIA,
    "promovido_producao": False,
    "python": platform.python_version(),
    "spark": spark.version,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__,
}
resultado_json = json.dumps(CONS_RESULTADO, ensure_ascii=False, sort_keys=True)
status_consolidacao = (
    "REFERENCIA_REPRODUZIDA" if CONS_CFG["modo"] == "PRESERVAR_REFERENCIA"
    else "CANDIDATA_REAJUSTADA_NAO_PROMOVIDA"
)
spark.createDataFrame([(
    CONS_ID, status_consolidacao, MS03_ID, resultado_json,
    datetime.now(timezone.utc).isoformat(),
)], "id_consolidacao string, status string, id_ms03 string, resultado_json string, fim_utc string").write.format(
    "delta"
).mode("append").saveAsTable(CONS_CFG["prefixo_saida"] + "_consolidacoes_hml")
print("\nCONS_01_CONCLUSAO")
print(json.dumps(CONS_RESULTADO, ensure_ascii=False, indent=2))
print("Configuração para avaliar este pacote no notebook 02 (não promove produção):")
print(json.dumps({
    "id_ms03_validado": MS03_ID,
    "tabela_ms03_execucoes": MS03_SAIDAS["execucoes"],
    "tabela_pacotes": RET_PACOTES,
}, indent=2))
