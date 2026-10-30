# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — NBA V2.3 | Eventos → treino A/B → B2 → avaliação → MLflow
# MAGIC Este notebook **cria a base a partir dos eventos e treina de verdade**. Não requer um experimento antigo.
# MAGIC B2 e as regras estatísticas são preservados; reconstruir dados atualizados pode alterar scores e métricas legitimamente.
# MAGIC 
# MAGIC **Fluxo:** configuração → código do modelo → base → treinamento → avaliação → registro. Execute em ordem.
# MAGIC Os módulos são embutidos, legíveis nos arquivos auxiliares do ZIP e registrados junto do modelo. Não é necessário copiá-los para o workspace.
# MAGIC 
# MAGIC Somente homologação: nenhuma view anterior é substituída. Os defaults de período/amostra reproduzem o desenho conhecido, **não garantem equivalência dos dados históricos**.
# MAGIC Requer Spark >=3.5, MLflow >=2.12.2, NumPy, Pandas, SciPy e PyArrow disponíveis no driver e nos executores. Não instala bibliotecas durante o job.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração — único bloco a editar para período, amostra e avaliação

# COMMAND ----------

import copy
import hashlib
import inspect
import json
import math
import platform
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import scipy
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from packaging.version import Version
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

CFG = {
    'fonte_eventos': 'ctg_dsti.renato_nba.base_jornadas_onboarding',
    'versao_eventos': None,  # None: fixa a versão Delta atual no início desta execução.
    # Defaults históricos explícitos. Ajuste aqui para treinar outro período.
    'inicio_historico': '2026-06-01T00:00:00+00:00',
    'corte_historico_exclusivo': '2026-09-25T00:00:00+00:00',
    'fuso': 'Etc/UTC',
    'modulo_amostra_clientes': 1000,  # 1 = todos; hash seleciona CLIENTES, não linhas.
    'sal_amostra': 'sm_v2_amostra_historica',
    'modulo_validacao_clientes': 5,  # Uma fração ~1/5 dos clientes, mesmo sal da V2.2.
    'sal_validacao': 'sm_v2_validacao_clientes',
    'debounce_seg': 30.0,
    'min_eventos_contexto': 80,
    'min_clientes_contexto': 20,
    'idades_avaliacao_seg': [0.0, 1800.0, 86400.0, 604800.0],
    'max_passos_macro': 5,
    'acoes_avaliacao': ['pagamentos_boleto', 'opf_recepcao_dados', 'pix_cadastro_chave'],
    'experimento_mlflow': '/Shared/NBA/V23_B2',
    'tracking_uri': 'databricks',
    'registered_model_name': None,  # Opcional: catalogo.schema.modelo. Não altera alias.
    'prefixo_saida': 'ctg_dsti.renato_nba.nba_v23_mlflow_treino',
    'autotestes': True,
    'max_linhas_por_grupo': 150000,  # Proteção: PARA; não trunca/amostra os grupos.
    'max_modelos_driver': 5000,
    'max_condicoes_avaliacao': 30000,
    'max_nos_grafo': 6000,
    'max_payload_mb': 128,
    'max_agregados_driver': 100000,
    'bootstrap_replicas': 2000,
    'semente_avaliacao': 20261009,
    # Opcional: comparação da base reconstruída com uma referência já preparada.
    'tabela_base_referencia': None,
    'id_base_referencia': None,
    'coluna_id_base_referencia': 'id_execucao',
}
ALVOS = sorted({
    'opf_iniciador_pagamentos', 'opf_recepcao_dados',
    'tag_veicular_alterar_placa', 'tag_veicular_ativar', 'tag_veicular_bloquear',
    'tag_veicular_cancelar', 'tag_veicular_contestacao', 'tag_veicular_desbloquear',
    'tag_veicular_segunda_via', 'tag_veicular_solicitar', 'pix_cadastro_chave',
    'pix_excluir_chave', 'pix_trazer_chave', 'pagamentos_boleto', 'debito_automatico',
    'emprestimos_pessoal_cartao', 'portabilidade_salario',
})
TECNICAS = {'app_login', 'app_habilitacao_device', 'app_primeiro_acesso',
            'app_reset_senha', 'app_atualizacao_cadastral'}
NUM = {
    'min_eventos_origem': 80, 'min_clientes_origem': 20,
    'min_eventos_grupo': 100, 'min_clientes_grupo': 15, 'max_grupos_proprios': 8,
    'sigma_min': 0.15, 'sigma_max': 4.5, 'maxiter': 800,
    'lambda_prob_base': 0.0001, 'lambda_prob_contexto': 0.02,
    'lambda_tempo_contexto': 0.02,
}
VERSAO_PIPELINE = 'v23_b2_eventos_mlflow_1.0.0'
# NÃO há modo preservar/reajustar congelado. Este notebook prepara dados e treina A/B.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Funções preservadas e preparação
# MAGIC O núcleo é empacotado por código: não há pickle de uma sessão Spark ou de variáveis do notebook.
# MAGIC Os códigos abaixo também são fornecidos em `nba_engine.py`, `nba_fit.py` e `nba_data.py` para leitura e manutenção.
# MAGIC **A lista de avaliação detalhada não filtra o treino nem o catálogo.**

# COMMAND ----------

ENGINE_SOURCE = r'''
"""Núcleo NBA V2.3 B2; extraído do motor aprovado, sem novo ajuste estatístico."""
from collections import deque
from pathlib import Path
from typing import Any
import copy
import hashlib
import json
import math
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

def op_compilar_motor(motor: MotorMacroMemoria) -> dict:
    """Coeficientes de primeira passagem por grupo; nenhuma aproximação de idade."""
    payload = {"alvos": motor.alvos, "k": motor.k, "nos": {}}
    na = len(motor.alvos)
    for i, (estado, memoria) in enumerate(motor.nos):
        m = motor.modelos[i]
        if m is None or motor.term[i]:
            continue
        coef = np.zeros((len(m["destinos"]), na * motor.k + 4))
        for pos, destino in enumerate(m["destinos"]):
            j = motor.indice[(destino, motor.atualizar(memoria, destino))]
            contrib = np.zeros((na, motor.k))
            massas = np.zeros(4)
            if motor.macro(destino) == motor.macro(estado):
                contrib = motor.exatas[1:motor.k + 1, j, :].T
                massas = motor.massas[motor.k, j]
            elif motor.alvo[j] >= 0:
                contrib[motor.alvo[j], 0] = 1.0
            elif motor.term[j]:
                massas[1] = 1.0
            else:
                if motor.k > 1:
                    contrib[:, 1:] = motor.exatas[1:motor.k, j, :].T
                massas = motor.massas[motor.k - 1, j]
            coef[pos, :na * motor.k] = contrib.reshape(-1)
            coef[pos, na * motor.k:] = massas
        grupo = np.asarray(m["grupo"], dtype=np.int32)
        r = np.asarray(m["r_destino_no_grupo"], dtype=float)
        cg = np.zeros((m["n_grupos"], coef.shape[1]))
        np.add.at(cg, grupo, r[:, None] * coef)
        if not np.allclose(cg.sum(axis=1), 1.0, atol=1e-8):
            raise RuntimeError("Coeficientes do grupo não conservam massa.")
        payload["nos"][i] = {
            "estado": estado, "memoria": memoria, "rota": motor.rotas[i],
            "pi": np.asarray(m["pi_grupo"], dtype=float),
            "mu": np.asarray(m["mu_grupo"], dtype=float),
            "sigma": np.asarray(m["sigma_grupo"], dtype=float),
            "grupo": grupo, "r": r, "coef": cg,
            "destinos": list(m["destinos"]),
            "alerta_limite": bool(m.get("alerta_limite", False)),
            "max_dias": float(m["max_tempo_observado_dias"]),
        }
    return payload

def op_probabilidades_lote(no: dict, idades: np.ndarray, na: int, k: int) -> dict:
    idades = np.asarray(idades, dtype=float).reshape(-1)
    if not np.isfinite(idades).all() or np.any(idades < 0):
        raise ValueError("Idade negativa ou não finita no scoring.")
    log_s = np.zeros((len(idades), len(no["pi"])))
    pos = idades > 0
    if np.any(pos):
        z = (np.log(idades[pos] / 86400.0)[:, None] - no["mu"]) / no["sigma"]
        log_s[pos] = special.log_ndtr(-z)
    log_mass = np.log(no["pi"])[None, :] + log_s
    pesos = np.exp(log_mass - special.logsumexp(log_mass, axis=1)[:, None])
    raw = pesos[:, no["grupo"]] * no["r"][None, :]
    res = pesos @ no["coef"]
    contrib = res[:, :na * k].reshape(-1, na, k)
    massas = res[:, na * k:]
    scores = contrib.sum(axis=2)
    erro = np.abs(scores.sum(axis=1) + massas.sum(axis=1) - 1.0)
    if (not np.isfinite(raw).all() or not np.isfinite(res).all()
            or np.any(res < -1e-10) or np.any(erro > 1e-8)
            or not np.allclose(raw.sum(axis=1), 1.0, atol=1e-9)):
        raise RuntimeError("Cálculo operacional não conservou as probabilidades.")
    return {"raw": raw, "scores": scores, "contrib": contrib,
            "massas": massas, "erro": erro}

def op_testar_compilacao(motor: MotorMacroMemoria, payload: dict) -> None:
    ids = sorted(payload["nos"])
    selecionados = ids[::max(1, len(ids) // 60)][:60]
    for i in selecionados:
        no = payload["nos"][i]
        idades = np.array([0.0, 1.0, 1800.0, 86400.0, 604800.0])
        r = op_probabilidades_lote(no, idades, len(payload["alvos"]), payload["k"])
        for p, idade in enumerate(idades):
            ref = motor.prever(no["estado"], no["memoria"], float(idade))
            raw = np.array([ref["raw_probs"][d] for d in no["destinos"]])
            util = np.array([ref["util_probs"][d] for d in payload["alvos"]])
            mm = [ref[c] for c in ("massa_sem_modelo", "massa_terminal",
                                   "massa_retencao_interna", "massa_limite_macro")]
            if not (np.allclose(raw, r["raw"][p], atol=1e-10)
                    and np.allclose(util, r["scores"][p], atol=1e-10)
                    and np.allclose(mm, r["massas"][p], atol=1e-10)):
                raise AssertionError("O cálculo em lotes divergiu do motor validado.")
    print("Operacional: coeficientes em lotes reproduzem o motor de referência.")


OUTPUT_SCHEMA = (
    "status_modelo string, tem_previsao boolean, rota_modelo string, "
    "raw_top5_json string, util_top5_json string, raw_probs_json string, "
    "util_probs_json string, massa_acao_util double, massa_sem_modelo double, "
    "massa_terminal double, massa_retencao_interna double, massa_limite_macro double, "
    "erro_massa double, extrapolacao_tempo boolean, alerta_limite boolean"
)
OUTPUT_COLUMNS = [part.strip().split()[0] for part in OUTPUT_SCHEMA.split(',')]
ARRAY_FIELDS = ('pi', 'mu', 'sigma', 'grupo', 'r', 'coef')


def completar_payload(motor, payload):
    payload['pares'] = [list(pair) for pair in motor.nos]
    payload['rotas'] = list(motor.rotas)
    return payload


def salvar_payload(payload, metadata_path, arrays_path):
    """JSON + NPZ sem pickle; arrays carregados com allow_pickle=False."""
    meta = {k: v for k, v in payload.items() if k != 'nos' and not k.startswith('_')}
    meta['nos'] = {}
    arrays = {}
    for i, no in payload['nos'].items():
        meta['nos'][str(i)] = {k: v for k, v in no.items() if k not in ARRAY_FIELDS}
        for name in ARRAY_FIELDS:
            arrays[f'n{i}_{name}'] = np.asarray(no[name])
    Path(metadata_path).write_text(json.dumps(meta, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    np.savez_compressed(arrays_path, **arrays)


def carregar_payload(metadata_path, arrays_path):
    payload = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        nodes = {}
        for key, no in payload['nos'].items():
            i = int(key)
            for name in ARRAY_FIELDS:
                no[name] = arrays[f'n{i}_{name}'].copy()
            nodes[i] = no
    payload['nos'] = nodes
    payload['_index'] = {tuple(pair): i for i, pair in enumerate(payload['pares'])}
    return payload


def _sem_previsao(status, rota):
    out = {c: None for c in OUTPUT_COLUMNS}
    out.update(status_modelo=status, tem_previsao=False, rota_modelo=rota,
               raw_top5_json='[]', util_top5_json='[]', raw_probs_json='{}',
               util_probs_json='{}', extrapolacao_tempo=False, alerta_limite=False)
    return out


def prever_dataframe(payload, dados: pd.DataFrame, sub_lote: int = 2048):
    """Mesma inferência compilada; devolve linhas na ordem do input."""
    required = {'estado', 'memoria', 'idade_seg'}
    if not required.issubset(dados.columns):
        raise ValueError(f'Colunas ausentes: {sorted(required - set(dados.columns))}')
    if sub_lote < 1:
        raise ValueError('sub_lote deve ser positivo.')
    if dados.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    index = payload.get('_index')
    if index is None:
        index = {tuple(pair): i for i, pair in enumerate(payload['pares'])}
    conhecidos = {pair[0] for pair in payload['pares']}
    df = dados.reset_index(drop=True)
    outputs, grupos = [None] * len(df), {}
    for pos, row in enumerate(df[['estado', 'memoria', 'idade_seg']].itertuples(index=False)):
        if pd.isna(row.estado):
            outputs[pos] = _sem_previsao('SEM_ESTADO', 'NAO_APLICAVEL')
            continue
        if pd.isna(row.idade_seg) or not math.isfinite(float(row.idade_seg)) or float(row.idade_seg) < 0:
            raise ValueError('Idade ausente, negativa ou não finita.')
        s = str(row.estado)
        mem = MS_SEM_MEMORIA if pd.isna(row.memoria) or row.memoria == '' else str(row.memoria)
        if s not in conhecidos:
            outputs[pos] = _sem_previsao('ESTADO_NAO_CATALOGADO', 'NAO_APLICAVEL')
            continue
        node = index.get((s, mem))
        if node is None:
            raise ValueError(f'Memória incompatível com estado catalogado: {s!r}, {mem!r}')
        if node not in payload['nos']:
            outputs[pos] = _sem_previsao('SEM_MODELO_ORIGEM', payload['rotas'][node])
            continue
        grupos.setdefault(node, []).append(pos)
    alvos, k = payload['alvos'], payload['k']
    for node, positions in grupos.items():
        no = payload['nos'][node]
        for ini in range(0, len(positions), sub_lote):
            ids = positions[ini:ini + sub_lote]
            ages = df.loc[ids, 'idade_seg'].to_numpy(float)
            result = op_probabilidades_lote(no, ages, len(alvos), k)
            for j, pos in enumerate(ids):
                raw, util = result['raw'][j], result['scores'][j]
                ri = np.argsort(-raw, kind='mergesort')[:5]
                ui = [int(a) for a in np.argsort(-util, kind='mergesort') if util[a] > 0][:5]
                raw_top = [{'ranking': n, 'estado': no['destinos'][int(a)], 'score': float(raw[a])}
                           for n, a in enumerate(ri, 1)]
                util_top = [{'ranking': n, 'acao': alvos[a], 'score': float(util[a]),
                             'passo_macro_maior_contribuicao': int(np.argmax(result['contrib'][j, a]) + 1)}
                            for n, a in enumerate(ui, 1)]
                out = dict(status_modelo='PREVISAO_B2' if payload.get('variante', MS_B2) == MS_B2 else 'PREVISAO_A',
                           tem_previsao=True, rota_modelo=no['rota'],
                           raw_top5_json=json.dumps(raw_top, ensure_ascii=False, allow_nan=False),
                           util_top5_json=json.dumps(util_top, ensure_ascii=False, allow_nan=False),
                           raw_probs_json=json.dumps(dict(zip(no['destinos'], map(float, raw))), allow_nan=False),
                           util_probs_json=json.dumps(dict(zip(alvos, map(float, util))), allow_nan=False),
                           massa_acao_util=float(util.sum()), erro_massa=float(result['erro'][j]),
                           extrapolacao_tempo=bool(ages[j] / 86400 > no['max_dias']),
                           alerta_limite=bool(no['alerta_limite']))
                for z, name in enumerate(('massa_sem_modelo', 'massa_terminal', 'massa_retencao_interna', 'massa_limite_macro')):
                    out[name] = float(result['massas'][j, z])
                outputs[pos] = out
    return pd.DataFrame(outputs, columns=OUTPUT_COLUMNS)

'''
exec(compile(ENGINE_SOURCE, 'nba_engine.py', "exec"), globals())

# COMMAND ----------

FIT_SOURCE = r'''
"""Ajuste A/B original: mesma verossimilhança, priors, limites e otimizador."""
import copy
import hashlib
import json
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import optimize, special

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

'''
exec(compile(FIT_SOURCE, 'nba_fit.py', "exec"), globals())

# COMMAND ----------

DATA_SOURCE = r'''
"""Preparação comum ao treinamento e scoring. Regras da V2.3 preservadas."""
import math
import re
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

MS_SEM_MEMORIA = "__SEM_MEMORIA__"
MS_BASE = "__BASE__"
def op_preparar_acoes(eventos: DataFrame, config: DataFrame,
                     tecnicas: set, debounce_seg: float) -> DataFrame:
    # Recebe somente cd_bv, ts_evento, estado, profundidade_max, até o corte.
    por_estado = eventos.groupBy("cd_bv", "ts_evento", "estado").agg(
        F.max("profundidade_max").alias("profundidade_max"),
        F.count("*").alias("n_ev"),
    ).withColumn("_prof", F.coalesce("profundidade_max", F.lit(-1.0)))
    wi = Window.partitionBy("cd_bv", "ts_evento")
    candidatos = por_estado.withColumn("_max", F.max("_prof").over(wi))
    candidatos = candidatos.withColumn("_eh_max", F.col("_prof") == F.col("_max"))
    instantes = candidatos.groupBy("cd_bv", "ts_evento").agg(
        F.sum(F.col("_eh_max").cast("int")).alias("n_max"),
        F.max(F.when(F.col("_eh_max"), F.col("estado"))).alias("_estado_max"),
        F.sum("n_ev").alias("n_ev"),
    ).withColumn("ordem_ambigua", F.col("n_max") != 1)
    instantes = instantes.withColumn(
        "estado", F.when(~F.col("ordem_ambigua"), F.col("_estado_max"))
    )
    w = Window.partitionBy("cd_bv").orderBy("ts_evento")
    wa = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df = instantes.withColumn("_s_ant", F.lag("estado").over(w))
    df = df.withColumn("_t_ant", F.lag("ts_evento").over(w))
    df = df.withColumn("_a_ant", F.lag("ordem_ambigua").over(w))
    gap = (F.expr("unix_micros(ts_evento)") - F.expr("unix_micros(_t_ant)")) / 1e6
    repetir = (
        F.col("estado").isNotNull() & F.col("_s_ant").isNotNull()
        & ~F.col("ordem_ambigua") & ~F.coalesce("_a_ant", F.lit(False))
        & (F.col("estado") == F.col("_s_ant")) & (gap <= debounce_seg)
    )
    df = df.withColumn("_abriu", F.when(repetir, 0).otherwise(1))
    df = df.withColumn("_burst", F.sum("_abriu").over(wa))
    acoes = df.groupBy("cd_bv", "_burst").agg(
        F.max("estado").alias("estado"),
        F.max("ts_evento").alias("ts_estado"),
        F.max(F.col("ordem_ambigua").cast("int")).cast("boolean").alias("ordem_ambigua"),
        F.sum("n_ev").alias("n_eventos_consolidados"),
    )
    w = Window.partitionBy("cd_bv").orderBy("ts_estado", "_burst")
    wa = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    acoes = acoes.withColumn("passo", F.row_number().over(w)).drop("_burst")
    w = Window.partitionBy("cd_bv").orderBy("passo")
    wa = w.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    cat = config.select("estado", "funil", "etapa", "tipo_estado").withColumn(
        "estado_catalogado", F.lit(True)
    )
    acoes = acoes.join(F.broadcast(cat), "estado", "left")
    acoes = acoes.withColumn("estado_catalogado", F.coalesce("estado_catalogado", F.lit(False)))
    acoes = acoes.withColumn(
        "barreira_memoria",
        F.col("ordem_ambigua") | ~F.col("estado_catalogado"),
    ).withColumn(
        "origem_tecnica", F.coalesce(F.col("funil").isin(sorted(tecnicas)), F.lit(False))
    )
    acoes = acoes.withColumn("_barreira_ant", F.lag("barreira_memoria").over(w))
    acoes = acoes.withColumn("_novo_trecho", F.when(
        (F.col("passo") == 1) | F.col("barreira_memoria")
        | F.coalesce("_barreira_ant", F.lit(False)), 1
    ).otherwise(0))
    acoes = acoes.withColumn("_trecho", F.sum("_novo_trecho").over(wa))
    wp = Window.partitionBy("cd_bv", "_trecho").orderBy("passo").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    evidence = ~F.col("barreira_memoria") & F.col("funil").isNotNull() & ~F.col("origem_tecnica")
    acoes = acoes.withColumn("_evidencia", F.when(evidence, F.struct(
        F.col("funil").alias("acao"), F.col("passo").alias("passo"),
        F.col("ts_estado").alias("ts"),
    )))
    acoes = acoes.withColumn("_memoria", F.last("_evidencia", ignorenulls=True).over(wp))
    return acoes.select(
        "cd_bv", "passo", "ts_estado", "estado", "funil", "etapa", "tipo_estado",
        "ordem_ambigua", "estado_catalogado", "barreira_memoria", "origem_tecnica",
        "n_eventos_consolidados",
        F.col("_memoria.acao").alias("contexto_negocio"),
        F.col("_memoria.passo").alias("passo_evidencia_memoria"),
        F.col("_memoria.ts").alias("ts_evidencia_memoria"),
    )

def ms03_rotular(base: DataFrame, alvos: list, k: int, tolerancia: float = 1e-3) -> DataFrame:
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
    erro_tempo = F.abs(delta - F.col("dur_min")) > tolerancia
    exigir_vazio("labels_arestas_exatas", df.filter(
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


def exigir_vazio(nome: str, df: DataFrame) -> None:
    if df.limit(1).count():
        raise RuntimeError(f'Integridade: {nome}. Inspecione os registros; não foram descartados.')


def validar_eventos(eventos: DataFrame) -> None:
    needed = {'cd_bv', 'ts_evento', 'estado', 'profundidade_max'}
    if not needed.issubset(eventos.columns):
        raise ValueError(f'Eventos sem colunas: {sorted(needed - set(eventos.columns))}')
    if eventos.schema['ts_evento'].dataType.simpleString() != 'timestamp':
        raise ValueError('Timestamp de eventos deve ser timestamp, não string/NTZ/date.')
    bad = eventos.agg(
        F.sum((F.col('cd_bv').isNull() | F.col('estado').isNull() | F.col('ts_evento').isNull()
               | (F.length(F.trim('cd_bv')) == 0) | (F.length(F.trim('estado')) == 0)).cast('long')).alias('invalidos'),
        F.sum(F.lower('estado').rlike('sem_acao|conversao').cast('long')).alias('sinteticos'),
        F.sum((F.isnan('profundidade_max') | (F.abs('profundidade_max') == float('inf'))).cast('long')).alias('prof_invalida'),
    ).first().asDict()
    if any(int(x or 0) for x in bad.values()):
        raise ValueError(f'Fonte incompatível com a V2.3 clean: {bad}')


def construir_catalogo(eventos: DataFrame) -> DataFrame:
    """Mesmo contrato clean: jornadas topo/navegacao/sucesso e atendimento/canal."""
    cfg = eventos.select('estado').distinct().withColumn('_p', F.split('estado', ':::'))
    exigir_vazio('estado fora do formato família:::etapa/canal', cfg.filter(F.size('_p') != 2))
    cfg = cfg.withColumn('_familia', F.col('_p')[0]).withColumn('_parte', F.col('_p')[1])
    jornada = F.col('_parte').isin('topo', 'navegacao', 'sucesso')
    cfg = cfg.withColumn('funil', F.when(jornada, F.col('_familia')))
    cfg = cfg.withColumn('etapa', F.when(jornada, F.col('_parte')))
    cfg = cfg.withColumn('tipo_estado', F.when(F.col('_parte') == 'topo', 'funil_abertura')
        .when(F.col('_parte') == 'navegacao', 'funil_meio')
        .when(F.col('_parte') == 'sucesso', 'funil_conclusao')
        .when(F.col('_familia') == 'atendimento', 'atendimento').otherwise('nao_mapeado'))
    exigir_vazio('estado sem definição no catálogo clean', cfg.filter("tipo_estado = 'nao_mapeado'"))
    return cfg.select('estado', 'funil', 'etapa', 'tipo_estado',
                      F.when(~jornada, F.col('_parte')).alias('detalhe'),
                      F.coalesce('funil', 'estado').alias('acao_macro'),
                      F.lit(False).alias('terminal_real'))


def construir_transicoes(acoes: DataFrame, corte_iso: str, sal_validacao: str,
                          modulo_validacao: int = 5) -> DataFrame:
    """Dur_min é em SEGUNDOS. Último estado e saída antes de ambiguidade: censura."""
    if modulo_validacao < 2:
        raise ValueError('modulo_validacao deve ser >= 2.')
    w = Window.partitionBy('cd_bv').orderBy('passo')
    b = acoes.withColumn('_s_next', F.lead('estado').over(w))
    b = b.withColumn('_t_next', F.lead('ts_estado').over(w))
    b = b.withColumn('_b_next', F.lead('barreira_memoria').over(w))
    b = b.withColumn('ts_corte_estado', F.lit(corte_iso).cast('timestamp'))
    b = b.withColumn('_fim', F.coalesce('_t_next', 'ts_corte_estado'))
    b = b.withColumn('dur_min', (F.expr('unix_micros(_fim)') - F.expr('unix_micros(ts_estado)')) / 1e6)
    b = b.withColumn('tipo_censura', F.when(F.col('barreira_memoria'), 'origem_ambigua')
        .when(F.col('_t_next').isNull() | F.col('_b_next'), 'direita').otherwise('exata'))
    b = b.withColumn('destino', F.when(F.col('tipo_censura') == 'exata', F.col('_s_next')))
    b = b.withColumn('dur_max', F.when(F.col('tipo_censura') == 'exata', F.col('dur_min')))
    b = b.withColumn('status_observacao', F.when(F.col('barreira_memoria'), 'ORIGEM_AMBIGUA')
        .when(F.col('_t_next').isNull(), 'CENSURA_CORTE')
        .when(F.col('_b_next'), 'CENSURA_ANTES_AMBIGUIDADE').otherwise('SAIDA_OBSERVADA'))
    b = b.withColumn('elegivel_ajuste', ~F.col('barreira_memoria') & (F.col('dur_min') > 0))
    b = b.withColumn('contexto_modelo', F.when(F.col('barreira_memoria'), F.lit(None).cast('string'))
        .when(F.col('origem_tecnica'), F.coalesce('contexto_negocio', F.lit(MS_SEM_MEMORIA)))
        .otherwise(MS_BASE))
    b = b.withColumn('validacao_cliente', F.pmod(F.xxhash64('cd_bv', F.lit(sal_validacao)), F.lit(modulo_validacao)) == 0)
    b = b.withColumn('peso_evento_treino', F.when(~F.col('validacao_cliente') & F.col('elegivel_ajuste'), 1.0))
    exigir_vazio('duração não positiva', b.filter('dur_min <= 0 OR dur_min IS NULL'))
    return b.drop('_s_next', '_t_next', '_b_next', '_fim')


def preparar_estado_atual(eventos: DataFrame, catalogo: DataFrame, tecnicas: set,
                           debounce_seg: float, publico: DataFrame, corte_iso: str) -> DataFrame:
    a = op_preparar_acoes(eventos, catalogo, tecnicas, debounce_seg)
    w = Window.partitionBy('cd_bv').orderBy(F.desc('ts_estado'), F.desc('passo'))
    ultimo = a.withColumn('_rn', F.row_number().over(w)).filter('_rn = 1').drop('_rn')
    b = publico.join(ultimo, 'cd_bv', 'left')
    b = b.withColumn('status_dado', F.when(F.col('ts_estado').isNull(), 'SEM_HISTORICO_NA_FONTE')
        .when(F.col('ordem_ambigua'), 'ESTADO_ATUAL_AMBIGUO')
        .when(~F.coalesce('estado_catalogado', F.lit(False)), 'ESTADO_NAO_CATALOGADO').otherwise('OK'))
    b = b.withColumn('memoria', F.coalesce('contexto_negocio', F.lit(MS_SEM_MEMORIA)))
    b = b.withColumn('ts_corte_eventos', F.lit(corte_iso).cast('timestamp'))
    b = b.withColumn('idade_seg', (F.expr('unix_micros(ts_corte_eventos)') - F.expr('unix_micros(ts_estado)')) / 1e6)
    b = b.withColumn('acao_atual', F.coalesce('funil', 'estado'))
    b = b.withColumn('data_referencia', F.to_date('ts_corte_eventos'))
    exigir_vazio('idade negativa', b.filter('idade_seg < 0'))
    return b


def autoteste_preparo(spark):
    """Executado no cluster antes de ler dados reais; não exige tabelas."""
    catalogo = spark.createDataFrame([
        ('b:::sucesso', 'b', 'sucesso', 'funil_conclusao'),
        ('l:::topo', 'l', 'topo', 'funil_abertura'),
        ('l:::sucesso', 'l', 'sucesso', 'funil_conclusao'),
        ('p:::topo', 'p', 'topo', 'funil_abertura'),
    ], 'estado string, funil string, etapa string, tipo_estado string')
    rows = [('u1', 0, 'b:::sucesso', 1.), ('u1', 10, 'l:::topo', 0.),
            ('u1', 20, 'l:::topo', 0.), ('u1', 40, 'l:::sucesso', 1.),
            ('u1', 60, 'p:::topo', 0.), ('u2', 0, 'b:::sucesso', 1.),
            ('u2', 10, 'l:::topo', 1.), ('u2', 10, 'l:::sucesso', 1.),
            ('u2', 40, 'l:::topo', 0.)]
    e = spark.createDataFrame(rows, 'cd_bv string, seg long, estado string, profundidade_max double')
    e = e.withColumn('ts_evento', F.expr('timestamp_micros(1700000000000000 + seg*1000000)')).drop('seg')
    a = op_preparar_acoes(e, catalogo, {'l'}, 30.)
    corte = '2023-11-14T22:15:20+00:00'
    b = construir_transicoes(a, corte, 'test')
    r = { (x.cd_bv, x.passo): x.asDict() for x in b.collect() }
    assert len([x for x in r if x[0] == 'u1']) == 4
    assert r['u1', 2]['contexto_negocio'] == 'b'
    assert r['u1', 2]['dur_min'] == 20.0
    assert r['u1', 4]['tipo_censura'] == 'direita'
    assert r['u2', 1]['tipo_censura'] == 'direita'
    assert r['u2', 3]['contexto_negocio'] is None
    labels = ms03_rotular(b, ['b', 'p'], 5)
    l = {(x.cd_bv, x.passo): x.asDict() for x in labels.collect()}
    assert l['u1', 1]['acao_real'] == 'p'
    assert l['u1', 1]['passo_macro_real'] == 2
    assert l['u1', 4]['status_label'] == 'CENSURADO_ANTES_HORIZONTE'
    print('Autoteste Spark: debounce, memória, censura e rótulos aprovados.')

'''
exec(compile(DATA_SOURCE, 'nba_data.py', "exec"), globals())

# COMMAND ----------

MODEL_SOURCE = r'''
"""Modelo customizado MLflow: carregamento por código, sem pickle do motor."""
from pathlib import Path
import copy
import hashlib
import json
import mlflow
from nba_engine import carregar_payload, prever_dataframe


class NBA23Model(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        assinatura = json.loads(Path(context.artifacts['integridade']).read_text(encoding='utf-8'))
        for name, digest in assinatura.items():
            atual = hashlib.sha256(Path(context.artifacts[name]).read_bytes()).hexdigest()
            if atual != digest:
                raise RuntimeError(f'Artefato do modelo alterado: {name}')
        self.bundle = json.loads(Path(context.artifacts['bundle']).read_text(encoding='utf-8'))
        if self.bundle['contrato'] != 'nba_v23_b2_mlflow_v1':
            raise ValueError('Contrato de modelo não suportado.')
        self.payload = carregar_payload(context.artifacts['metadata'], context.artifacts['coeficientes'])
        if self.payload['alvos'] != self.bundle['alvos'] or self.payload['k'] != self.bundle['max_passos_macro']:
            raise ValueError('Kernel e contrato de alvos/horizonte divergentes.')

    def predict(self, context, model_input, params=None):
        return prever_dataframe(self.payload, model_input)

    def contrato(self):
        """Metadados para preparar o input; não depende do notebook de treino."""
        return copy.deepcopy({k: v for k, v in self.bundle.items() if k not in ('modelos_a', 'modelos_b', 'modelos_b2')})

    def preparar_estados(self, eventos, publico, corte_iso):
        # PySpark só é necessário para a preparação batch, não no predict tabular.
        from nba_data import preparar_estado_atual
        spark = eventos.sparkSession
        rows = [(r['estado'], r.get('funil'), r.get('etapa'), r['tipo_estado']) for r in self.bundle['catalogo']]
        cfg = spark.createDataFrame(rows, 'estado string, funil string, etapa string, tipo_estado string')
        return preparar_estado_atual(eventos, cfg, set(self.bundle['tecnicas']),
                                      self.bundle['preparo']['debounce_seg'], publico, corte_iso)

    def validar_eventos(self, eventos):
        from nba_data import validar_eventos
        return validar_eventos(eventos)

    def teste_preparo(self, spark):
        from nba_data import autoteste_preparo
        return autoteste_preparo(spark)


mlflow.models.set_model(NBA23Model())

'''

# COMMAND ----------

import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pyspark.sql import functions as F


def nome_sql(nome):
    partes = nome.split('.')
    if not 1 <= len(partes) <= 3 or not all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', p) for p in partes):
        raise ValueError(f'Nome SQL inválido: {nome}')
    return '.'.join(f'`{p}`' for p in partes)


def timestamp_utc(texto):
    dt = datetime.fromisoformat(texto.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('A data deve informar offset/timezone. Exemplo: 2026-09-25T00:00:00+00:00')
    return dt.astimezone(timezone.utc)


def ler_delta_fixo(spark, tabela, versao=None):
    """Captura a versão atual UMA vez. Nenhuma troca por versão nova no meio do job."""
    if not spark.catalog.tableExists(tabela):
        raise ValueError(f'Fonte não encontrada: {tabela}')
    detail = spark.sql(f'DESCRIBE DETAIL {nome_sql(tabela)}').first().asDict()
    if detail.get('format', '').lower() != 'delta':
        raise ValueError(f'Fonte de eventos deve ser Delta: {tabela}')
    if versao is None:
        versao = int(spark.sql(f'DESCRIBE HISTORY {nome_sql(tabela)} LIMIT 1').first()['version'])
    df = spark.read.option('versionAsOf', int(versao)).table(tabela)
    df.limit(1).collect()
    print(f'Fonte fixa: {tabela} | versão={versao}')
    return df, {'tabela': tabela, 'versao_delta': int(versao),
                'table_id': detail.get('id'), 'location': detail.get('location')}


def conferir_identidade(spark, meta):
    current = spark.sql(f"DESCRIBE DETAIL {nome_sql(meta['tabela'])}").first().asDict()
    if current.get('id') != meta['table_id'] or current.get('location') != meta['location']:
        raise RuntimeError('A identidade da fonte mudou durante a execução.')
    # Novos commits são permitidos: todas as leituras usam a versão fixada.


def salvar_etapa(spark, df, tabela, campo_id, valor_id):
    """Append por execução; reexecução parcial com mesmo ID é bloqueada."""
    nome_sql(tabela)
    if spark.catalog.tableExists(tabela):
        if spark.table(tabela).filter(F.col(campo_id) == valor_id).limit(1).count():
            raise RuntimeError(f'{tabela}: ID já gravado. Reinicie o notebook para nova execução.')
    out = df.withColumn(campo_id, F.lit(valor_id))
    out.write.format('delta').mode('append').partitionBy(campo_id).saveAsTable(tabela)
    return spark.table(tabela).filter(F.col(campo_id) == valor_id)


def exibir(nome, df, limite=30):
    print('\n' + nome)
    df.show(limite, truncate=False)


def exigir_colunas(df, cols, label):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'{label}: colunas ausentes {sorted(missing)}')

# COMMAND ----------

# As funções numéricas A/B acima são as originais. Estes wrappers distribuem por origem/contexto.
MODELO_SCHEMA = (
    'variante string, nivel string, estado string, contexto_modelo string, '
    'status_modelo string, n_observacoes long, n_saidas_exatas long, n_clientes long, '
    'massa_pesos double, n_grupos long, alerta_limite boolean, modelo_json string, detalhe string'
)


def criar_ajustador(codigo_fit, numerico, contextos=False, broadcast_pais=None):
    def worker(pdf):
        scope = {}
        exec(compile(codigo_fit, 'nba_fit_empacotado.py', 'exec'), scope)
        pdf = pdf.sort_values(['cd_bv', 'passo'], kind='mergesort')
        origem = str(pdf['estado'].iloc[0])
        contexto = str(pdf['contexto_modelo'].iloc[0]) if contextos else '__BASE__'
        linha = dict(variante='B_MEMORIA' if contextos else 'A_REFERENCIA',
                     nivel='CONTEXTO' if contextos else 'ORIGEM', estado=origem,
                     contexto_modelo=contexto, status_modelo='NAO_AJUSTADO',
                     n_observacoes=len(pdf), n_saidas_exatas=int(pdf.tipo_censura.eq('exata').sum()),
                     n_clientes=int(pdf.cd_bv.nunique()), massa_pesos=float(pdf.peso_evento_treino.sum()),
                     n_grupos=0, alerta_limite=False, modelo_json=None, detalhe='')
        try:
            pai = broadcast_pais.value.get(origem) if contextos else None
            if contextos and pai is None:
                raise ValueError('PAI_SEM_AJUSTE')
            estrutura = pai['estrutura'] if pai else scope['criar_estrutura'](pdf, numerico)
            m = scope['ajustar_modelo'](pdf, 'peso_evento_treino', estrutura, numerico, pai)
            linha.update(status_modelo='AJUSTADO', n_grupos=int(m['n_grupos']),
                         alerta_limite=bool(m['alerta_limite']),
                         modelo_json=json.dumps(m, allow_nan=False),
                         detalhe=','.join(m['parametros_no_limite']))
        except (ValueError, FloatingPointError) as exc:
            linha['detalhe'] = f'{type(exc).__name__}: {exc}'[:1500]
        return pd.DataFrame([linha])
    return worker


def metricas_avaliacao(pred, acoes):
    """Separa RAW (saída exata), ranking útil (alvo observado) e perdas (desfecho conhecido)."""
    mapa = 'map<string,double>'
    rt = 'array<struct<ranking:int,estado:string,score:double>>'
    ut = 'array<struct<ranking:int,acao:string,score:double,passo_macro_maior_contribuicao:int>>'
    p = pred.withColumn('raw_probs', F.from_json('raw_probs_json', mapa))
    p = p.withColumn('util_probs', F.from_json('util_probs_json', mapa))
    p = p.withColumn('raw_top5', F.from_json('raw_top5_json', rt))
    p = p.withColumn('util_top5', F.from_json('util_top5_json', ut))
    p = p.withColumn('prob_real_raw', F.when(F.col('tem_previsao') & (F.col('tipo_censura') == 'exata'),
        F.coalesce(F.element_at('raw_probs', F.col('destino')), F.lit(0.0))))
    p = p.withColumn('raw_top1', F.when(F.col('prob_real_raw').isNotNull(),
        (F.expr('try_element_at(raw_top5, 1)')['estado'] == F.col('destino')).cast('double')))
    p = p.withColumn('raw_top5_acerto', F.when(F.col('prob_real_raw').isNotNull(),
        F.expr('array_contains(transform(raw_top5, x -> x.estado), destino)').cast('double')))
    p = p.withColumn('raw_brier', F.when(F.col('prob_real_raw').isNotNull(),
        F.expr('aggregate(map_values(raw_probs), cast(0 as double), (a, v) -> a + v*v)')
        + 1.0 - 2.0 * F.col('prob_real_raw')))
    p = p.withColumn('raw_logloss', F.when(F.col('prob_real_raw').isNotNull(),
        -F.log(F.greatest('prob_real_raw', F.lit(1e-15)))))
    known = F.col('status_label').isin('ALVO_OBSERVADO', 'HORIZONTE_OBSERVADO_SEM_ALVO')
    positive = F.col('status_label') == 'ALVO_OBSERVADO'
    medir = F.col('tem_previsao') & known
    p = p.withColumn('prob_acao_real', F.when(positive & F.col('tem_previsao'),
        F.coalesce(F.element_at('util_probs', F.col('acao_real')), F.lit(0.0))))
    # Ordena a lista COMPLETA por (-score, nome), como o motor; MRR não usa só Top5.
    p = p.withColumn('_ordem', F.expr('transform(array_sort(transform(map_entries(util_probs), '
        'x -> named_struct("neg", -x.value, "acao", x.key))), x -> x.acao)'))
    p = p.withColumn('_rank_real', F.expr('array_position(_ordem, acao_real)'))
    p = p.withColumn('top1', F.when(medir & positive,
        ((F.col('_rank_real') == 1) & (F.col('prob_acao_real') > 0)).cast('double')))
    p = p.withColumn('top5', F.when(medir & positive,
        ((F.col('_rank_real') <= 5) & (F.col('prob_acao_real') > 0)).cast('double')))
    p = p.withColumn('mrr', F.when(medir & positive,
        F.when(F.col('prob_acao_real') > 0, 1.0 / F.col('_rank_real')).otherwise(0.0)))
    p = p.withColumn('brier_acoes', F.when(medir,
        F.expr('aggregate(map_values(util_probs), cast(0 as double), (a, v) -> a + v*v)')
        + F.when(positive, 1.0 - 2.0 * F.col('prob_acao_real')).otherwise(0.0)))
    p = p.withColumn('brier_algum_alvo', F.when(medir,
        F.pow(F.col('massa_acao_util') - positive.cast('double'), 2)))
    p = p.withColumn('_p_desfecho', F.when(medir, F.when(positive, F.col('prob_acao_real'))
        .otherwise(F.col('massa_limite_macro') + F.col('massa_terminal') + F.col('massa_retencao_interna'))))
    p = p.withColumn('logloss_conservador', F.when(medir,
        -F.log(F.greatest('_p_desfecho', F.lit(1e-15)))))
    return p.drop('_ordem', '_rank_real', '_p_desfecho', 'raw_probs_json', 'util_probs_json',
                  'raw_top5_json', 'util_top5_json')


def resumir_avaliacao(p):
    metrics = ['raw_top1', 'raw_top5_acerto', 'raw_brier', 'raw_logloss',
               'top1', 'top5', 'mrr', 'brier_acoes', 'brier_algum_alvo', 'logloss_conservador']
    seg = p.withColumn('segmento', F.explode(F.when(F.col('origem_tecnica'),
        F.array(F.lit('TODAS'), F.lit('ORIGENS_TECNICAS'))).otherwise(F.array(F.lit('TODAS')))))
    pc = seg.groupBy('segmento', 'idade_seg', 'cd_bv', 'variante').agg(
        *[F.avg(m).alias(m) for m in metrics])
    gerais = pc.groupBy('segmento', 'idade_seg', 'variante').agg(
        *[expr for m in metrics for expr in (F.avg(m).alias(m), F.count(m).alias('n_clientes_' + m))])
    return gerais, pc, seg, metrics


def resumo_por_acao(p, alvos):
    conhecidos = p.filter(F.col('tem_previsao') & F.col('status_label').isin(
        'ALVO_OBSERVADO', 'HORIZONTE_OBSERVADO_SEM_ALVO'))
    grade = p.select('variante', 'idade_seg').distinct().crossJoin(
        spark.createDataFrame([(a,) for a in alvos], 'acao string'))
    casos = conhecidos.crossJoin(F.broadcast(spark.createDataFrame([(a,) for a in alvos], 'acao string')))
    casos = casos.withColumn('y', (F.col('acao_real').eqNullSafe(F.col('acao'))).cast('double'))
    casos = casos.withColumn('score', F.coalesce(F.element_at('util_probs', F.col('acao')), F.lit(0.0)))
    casos = casos.withColumn('em_top5', F.expr('array_contains(transform(util_top5, x -> x.acao), acao)'))
    casos = casos.withColumn('em_top1', F.expr('try_element_at(util_top5, 1)')['acao'] == F.col('acao'))
    pc = casos.groupBy('variante', 'idade_seg', 'acao', 'cd_bv').agg(
        F.avg(F.pow(F.col('score') - F.col('y'), 2)).alias('brier_cliente'),
        F.avg(F.when(F.col('y') == 1, F.col('em_top1').cast('double'))).alias('recall1_cliente'),
        F.avg(F.when(F.col('y') == 1, F.col('em_top5').cast('double'))).alias('recall5_cliente'))
    agg = pc.groupBy('variante', 'idade_seg', 'acao').agg(
        F.avg('brier_cliente').alias('brier'), F.avg('recall1_cliente').alias('recall_top1_cliente'),
        F.avg('recall5_cliente').alias('recall_top5_cliente'))
    suporte = casos.groupBy('variante', 'idade_seg', 'acao').agg(
        F.sum('y').cast('long').alias('n_positivos_snapshots'),
        F.countDistinct(F.when(F.col('y') == 1, F.col('cd_bv'))).alias('n_clientes_positivos'),
        F.sum(F.col('em_top1').cast('long')).alias('n_recomendacoes_top1'),
        F.avg(F.when(F.col('em_top1'), F.col('y'))).alias('precisao_top1_snapshot'),
        F.avg('score').alias('score_medio'))
    return grade.join(agg, ['variante', 'idade_seg', 'acao'], 'left').join(
        suporte, ['variante', 'idade_seg', 'acao'], 'left')


def lift_acao(spark, p, alvos, seed):
    # Seleção ANTES de conhecer censura/desfecho. Uma observação por pessoa/idade.
    w = Window.partitionBy('cd_bv', 'idade_seg', 'variante').orderBy(
        F.xxhash64('cd_bv', 'passo', F.lit(str(seed))), 'passo')
    selected = p.withColumn('_rn_lift', F.row_number().over(w)).filter('_rn_lift = 1')
    selected = selected.filter(F.col('tem_previsao') & F.col('status_label').isin(
        'ALVO_OBSERVADO', 'HORIZONTE_OBSERVADO_SEM_ALVO'))
    long = selected.crossJoin(F.broadcast(spark.createDataFrame([(a,) for a in alvos], 'acao string')))
    long = long.withColumn('score', F.coalesce(F.element_at('util_probs', F.col('acao')), F.lit(0.0)))
    long = long.withColumn('y', F.col('acao_real').eqNullSafe(F.col('acao')).cast('double'))
    keys = ['variante', 'idade_seg', 'acao']
    empates = long.groupBy(*keys, 'score').agg(F.count('*').alias('n'), F.sum('y').alias('pos'))
    wt = Window.partitionBy(*keys)
    wr = Window.partitionBy(*keys).orderBy(F.desc('score')).rowsBetween(Window.unboundedPreceding, -1)
    empates = empates.withColumn('N', F.sum('n').over(wt)).withColumn('P', F.sum('pos').over(wt))
    empates = empates.withColumn('antes', F.coalesce(F.sum('n').over(wr), F.lit(0)))
    frac = spark.createDataFrame([(0.10,), (0.20,)], 'fracao double')
    r = empates.crossJoin(F.broadcast(frac)).withColumn('f',
        F.greatest(F.lit(0.0), F.least(F.lit(1.0), (F.col('fracao') * F.col('N') - F.col('antes')) / F.col('n'))))
    r = r.groupBy(*keys, 'fracao').agg(F.max('N').alias('n_clientes'), F.max('P').alias('n_positivos'),
        F.sum(F.col('f') * F.col('n')).alias('clientes_faixa_esperados'),
        F.sum(F.col('f') * F.col('pos')).alias('positivos_faixa_esperados'))
    return r.withColumn('taxa_base', F.col('n_positivos') / F.col('n_clientes')).withColumn(
        'taxa_faixa', F.col('positivos_faixa_esperados') / F.col('clientes_faixa_esperados')).withColumn(
        'lift', F.when(F.col('n_positivos') > 0, F.col('taxa_faixa') / F.col('taxa_base'))).withColumn(
        'suporte', F.when(F.col('n_positivos') == 0, 'SEM_POSITIVOS').when(
            F.col('n_positivos') < 30, 'POUCO_SUPORTE').otherwise('DESCRITIVO_SEM_IC'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Iniciar run e validar ambiente
# MAGIC Em erro, a etapa marca a run como FAILED. Depois de corrigir, reinicie desde a configuração para gerar uma nova run.

# COMMAND ----------

if 'spark' not in globals():
    raise RuntimeError('Execute com sessão Spark do Databricks.')
if Version(spark.version.split('-')[0]) < Version('3.5.0'):
    raise RuntimeError('Requer Spark >= 3.5 para os contratos de timestamp e arrays.')
if Version(mlflow.__version__) < Version('2.12.2'):
    raise RuntimeError('Este pacote usa Models from Code: requer MLflow >= 2.12.2. Ajuste o ambiente do job antes de executar.')
if not (CFG['modulo_amostra_clientes'] >= 1 and CFG['modulo_validacao_clientes'] >= 2):
    raise ValueError('Módulos de amostra/validação inválidos.')
if CFG['max_passos_macro'] != 5 or len(ALVOS) != 17 or not set(CFG['acoes_avaliacao']) <= set(ALVOS):
    raise ValueError('Revise horizonte, os 17 alvos e as ações escolhidas para avaliação.')
if timestamp_utc(CFG['inicio_historico']) >= timestamp_utc(CFG['corte_historico_exclusivo']):
    raise ValueError('Início deve ser anterior ao corte exclusivo.')
if timestamp_utc(CFG['corte_historico_exclusivo']) > datetime.now(timezone.utc):
    raise ValueError('Corte de treino está no futuro.')
if 'nba_v23_mlflow' not in CFG['prefixo_saida']:
    raise ValueError('Use o novo prefixo nba_v23_mlflow para não sobrescrever artefatos anteriores.')
if mlflow.active_run() is not None:
    raise RuntimeError('Já existe uma run ativa. Finalize/cancele a execução anterior antes de iniciar outra.')
spark.conf.set('spark.sql.session.timeZone', CFG['fuso'])
mlflow.set_tracking_uri(CFG['tracking_uri'])
mlflow.set_experiment(CFG['experimento_mlflow'])
START = datetime.now(timezone.utc)
RUN = mlflow.start_run(run_name=f"v23_B2__{START:%Y%m%d_%H%M%S_UTC}__corte_{timestamp_utc(CFG['corte_historico_exclusivo']):%Y%m%d}__mod{CFG['modulo_amostra_clientes']}")
RUN_ID = RUN.info.run_id
mlflow.set_tags({'arquitetura': 'NBA_V23_B2', 'estado_pacote': 'EM_TREINAMENTO',
                 'promovido_producao': 'false', 'versao_codigo': VERSAO_PIPELINE})
TMP = Path(tempfile.mkdtemp(prefix='nba_v23_mlflow_'))
TABELAS = {x: f"{CFG['prefixo_saida']}_{x}_hml" for x in (
    'base', 'catalogo', 'suporte', 'modelos_a', 'modelos_b', 'validacao', 'metricas', 'execucoes')}
ARTEFATOS = {}


@contextmanager
def etapa(nome):
    mlflow.set_tag('etapa', nome)
    print('\nETAPA:', nome)
    try:
        yield
    except Exception as exc:
        mlflow.set_tags({'estado_pacote': 'FALHOU', 'falha_etapa': nome,
                         'erro_resumo': f'{type(exc).__name__}: {exc}'[:3000]})
        mlflow.end_run(status='FAILED')
        raise


def salvar(df, nome):
    out = salvar_etapa(spark, df, TABELAS[nome], 'run_id_treinamento', RUN_ID)
    ARTEFATOS[nome] = TABELAS[nome]
    return out


def registrar_relatorio(nome, df, limit=10000):
    exibir(nome, df)
    rows = df.limit(limit + 1).collect()
    if len(rows) > limit:
        raise RuntimeError(f'Relatório {nome} excede limite de coleta. Não é uma tabela de eventos.')
    pd.DataFrame([r.asDict(recursive=True) for r in rows]).to_csv(TMP / f'{nome}.csv', index=False)
    mlflow.log_artifact(str(TMP / f'{nome}.csv'), 'relatorios')


with etapa('PRECHECKS'):
    import pyarrow  # Falha cedo se o ambiente de applyInPandas não estiver instalado.
    if CFG['autotestes']:
        ms_autotestes_numericos()
        autoteste_preparo(spark)
    mlflow.log_dict({'config': CFG, 'alvos': ALVOS, 'tecnicas': sorted(TECNICAS),
                     'numerico': NUM}, 'contrato/config_treinamento.json')
    mlflow.log_params({
        'inicio_historico': CFG['inicio_historico'], 'corte_historico': CFG['corte_historico_exclusivo'],
        'modulo_amostra': CFG['modulo_amostra_clientes'], 'debounce_seg': CFG['debounce_seg'],
        'holdout': 'CLIENTE_HASH', 'max_passos_macro': CFG['max_passos_macro'], 'n_alvos': len(ALVOS),
        **NUM,
    })
    mlflow.log_dict({'python': platform.python_version(), 'spark': spark.version,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'mlflow': mlflow.__version__, 'pyarrow': pyarrow.__version__}, 'contrato/ambiente.json')
    for filename, source in [('nba_engine.py', ENGINE_SOURCE), ('nba_fit.py', FIT_SOURCE),
                             ('nba_data.py', DATA_SOURCE), ('modelo_pyfunc.py', MODEL_SOURCE)]:
        (TMP / filename).write_text(source, encoding='utf-8')
        mlflow.log_artifact(str(TMP / filename), 'codigo')
    print('Run MLflow:', RUN_ID)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Construir a base — leitura real dos eventos
# MAGIC Eventos [início, corte), amostra determinística por cliente, simultaneidade por profundidade, debounce de 30 s, transições, censura e memória. Catálogo derivado da telemetria clean.
# MAGIC O split é por cliente no mesmo período, conforme V2.3; **não é um teste temporal independente**. Censura antes de ambiguidade não vira transição artificial.
# MAGIC A disponibilidade por horário de ingestão não pode ser certificada sem esse campo na fonte.

# COMMAND ----------

with etapa('CRIACAO_BASE_DOS_EVENTOS'):
    fonte, META_FONTE = ler_delta_fixo(spark, CFG['fonte_eventos'], CFG['versao_eventos'])
    exigir_colunas(fonte, {'cd_bv', 'dm_navegacao', 'estado', 'profundidade_max'}, 'Eventos')
    if fonte.schema['dm_navegacao'].dataType.simpleString() != 'timestamp':
        raise ValueError('dm_navegacao deve ser timestamp; não converter silenciosamente.')
    # Os filtros temporais antecedem a construção da sequência. Null é retido para validar.
    eventos_periodo = fonte.select(F.col('cd_bv').cast('string'), F.col('dm_navegacao').alias('ts_evento'),
        F.col('estado').cast('string'), F.col('profundidade_max').cast('double')).filter(
        ((F.col('ts_evento') >= F.lit(CFG['inicio_historico']).cast('timestamp'))
         & (F.col('ts_evento') < F.lit(CFG['corte_historico_exclusivo']).cast('timestamp')))
        | F.col('ts_evento').isNull())
    # Catálogo é taxonomia dos eventos do período, não estimação usando o holdout.
    catalogo = construir_catalogo(eventos_periodo.filter('estado IS NOT NULL'))
    rows_cfg = catalogo.limit(2001).collect()
    if len(rows_cfg) > 2000:
        raise RuntimeError('Mais de 2000 estados. Revise o dimensionamento antes de continuar.')
    if not set(ALVOS) <= {r.funil for r in rows_cfg}:
        raise ValueError(f'Alvos não presentes no catálogo do período: {sorted(set(ALVOS) - {r.funil for r in rows_cfg})}')
    eventos = eventos_periodo.filter(
        F.pmod(F.xxhash64('cd_bv', F.lit(CFG['sal_amostra'])), F.lit(CFG['modulo_amostra_clientes'])) == 0)
    validar_eventos(eventos)
    if not eventos.limit(1).count():
        raise ValueError('Amostra sem eventos: reveja fonte/período/módulo.')
    acoes = op_preparar_acoes(eventos, catalogo, TECNICAS, CFG['debounce_seg'])
    base = construir_transicoes(acoes, CFG['corte_historico_exclusivo'],
                               CFG['sal_validacao'], CFG['modulo_validacao_clientes'])
    exigir_vazio('cliente em treino e validação', base.groupBy('cd_bv').agg(
        F.countDistinct('validacao_cliente').alias('n')).filter('n != 1'))
    exigir_vazio('chave cliente/passo duplicada', base.groupBy('cd_bv', 'passo').count().filter('count != 1'))
    exigir_vazio('memória do futuro', base.filter(
        (F.col('passo_evidencia_memoria') > F.col('passo')) | (F.col('ts_evidencia_memoria') > F.col('ts_estado'))))
    # Persistir evita recalcular todos os windows ao ajustar cada grupo.
    catalogo = salvar(catalogo, 'catalogo')
    base = salvar(base, 'base')
    treino = base.filter(~F.col('validacao_cliente') & F.col('elegivel_ajuste'))
    holdout = base.filter('validacao_cliente')  # Preserva barreiras para rotular corretamente.
    if not treino.limit(1).count() or not holdout.filter('elegivel_ajuste').limit(1).count():
        raise RuntimeError('Treino ou validação vazios. Não foi alterado o split automaticamente.')
    resumo_base = base.groupBy('validacao_cliente', 'status_observacao').agg(
        F.count('*').alias('n_observacoes'), F.countDistinct('cd_bv').alias('n_clientes'))
    registrar_relatorio('TREINO_01_BASE', resumo_base)
    sizes = base.groupBy('validacao_cliente').agg(F.countDistinct('cd_bv').alias('n')).collect()
    for r in sizes:
        mlflow.log_metric('n_clientes_validacao' if r.validacao_cliente else 'n_clientes_treino', float(r.n))
    mlflow.log_dict(META_FONTE, 'contrato/fonte_eventos.json')
    conferir_identidade(spark, META_FONTE)
    mlflow.set_tag('equivalencia_base_referencia', 'NAO_EXECUTADA')
    if CFG['tabela_base_referencia']:
        ref = spark.table(CFG['tabela_base_referencia'])
        if CFG['id_base_referencia']:
            ref = ref.filter(F.col(CFG['coluna_id_base_referencia']) == CFG['id_base_referencia'])
        campos = ['cd_bv', 'passo', 'ts_estado', 'estado', 'destino', 'dur_min',
                  'tipo_censura', 'validacao_cliente', 'contexto_negocio', 'contexto_modelo']
        exigir_colunas(ref, campos, 'Base de referência')
        exigir_vazio('base reconstruída difere da referência', base.select(*campos).exceptAll(ref.select(*campos)))
        exigir_vazio('referência difere da base reconstruída', ref.select(*campos).exceptAll(base.select(*campos)))
        mlflow.set_tag('equivalencia_base_referencia', 'APROVADA')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Treinamento A/B e composição B2
# MAGIC Não ajusta C/D, pois não fazem parte da candidata operacional escolhida. A e B usam os mesmos parâmetros, verossimilhança e otimizador originais; B2 usa p_B e S_A.
# MAGIC Grupos de pandas são limitados antes do ajuste: ultrapassar a proteção exige dimensionamento explícito; nunca trunca a amostra silenciosamente.

# COMMAND ----------

with etapa('AJUSTE_A_B_E_COMPOSICAO_B2'):
    # Todas as observações elegíveis dos clientes amostrados. Sem undersampling por destino/origem.
    max_group = treino.groupBy('estado').count().agg(F.max('count').alias('m')).first()['m']
    if max_group > CFG['max_linhas_por_grupo']:
        raise RuntimeError(f'Maior origem tem {max_group} linhas. applyInPandas carrega um grupo inteiro; dimensione o cluster/limite, sem truncar silenciosamente.')
    colfit = ['cd_bv', 'passo', 'estado', 'contexto_modelo', 'destino', 'dur_min', 'tipo_censura', 'peso_evento_treino']
    pais = treino.select(*colfit).groupBy('estado').applyInPandas(
        criar_ajustador(FIT_SOURCE, NUM), MODELO_SCHEMA)
    pais = salvar(pais, 'modelos_a')
    registrar_relatorio('TREINO_02_ORIGENS', pais.groupBy('status_modelo', 'alerta_limite').count())
    registrar_relatorio('TREINO_02_FALHAS_ORIGENS', pais.filter("status_modelo != 'AJUSTADO'").groupBy('detalhe').count())
    fitted = pais.filter("status_modelo = 'AJUSTADO'").select('estado', 'modelo_json').limit(CFG['max_modelos_driver'] + 1).collect()
    if not fitted or len(fitted) > CFG['max_modelos_driver']:
        raise RuntimeError('Quantidade de origens ajustadas vazia/acima do limite.')
    MODELOS_A = {r.estado: json.loads(r.modelo_json) for r in fitted}
    suporte = treino.groupBy('estado', 'contexto_modelo', 'origem_tecnica').agg(
        F.sum((F.col('tipo_censura') == 'exata').cast('long')).alias('n_exatas'),
        F.countDistinct(F.when(F.col('tipo_censura') == 'exata', F.col('cd_bv'))).alias('n_clientes_exatas'),
        F.count('*').alias('n_observacoes'))
    suporte = suporte.withColumn('candidato', F.col('origem_tecnica')
        & (F.col('contexto_modelo') != MS_SEM_MEMORIA)
        & (F.col('n_exatas') >= CFG['min_eventos_contexto'])
        & (F.col('n_clientes_exatas') >= CFG['min_clientes_contexto']))
    suporte = suporte.withColumn('status_suporte', F.when(~F.col('origem_tecnica'), 'MODELO_POR_ORIGEM')
        .when(F.col('contexto_modelo') == MS_SEM_MEMORIA, 'SEM_MEMORIA_OBSERVADA')
        .when(F.col('candidato'), 'CANDIDATO_CONTEXTUAL').otherwise('CONTEXTO_RARO'))
    suporte = salvar(suporte, 'suporte')
    contexto_base = treino.select(*colfit).join(suporte.filter('candidato').select('estado', 'contexto_modelo'),
                                               ['estado', 'contexto_modelo'], 'inner')
    bc_pais = spark.sparkContext.broadcast(MODELOS_A)
    if contexto_base.limit(1).count():
        filhos = contexto_base.groupBy('estado', 'contexto_modelo').applyInPandas(
            criar_ajustador(FIT_SOURCE, NUM, contextos=True, broadcast_pais=bc_pais), MODELO_SCHEMA)
    else:
        filhos = spark.createDataFrame([], MODELO_SCHEMA)
    filhos = salvar(filhos, 'modelos_b')
    bc_pais.unpersist()
    registrar_relatorio('TREINO_03_CONTEXTOS', filhos.groupBy('status_modelo', 'alerta_limite').count())
    registrar_relatorio('TREINO_03_FALHAS_CONTEXTOS', filhos.filter("status_modelo != 'AJUSTADO'").groupBy('detalhe').count())
    fitted_b = filhos.filter("status_modelo = 'AJUSTADO'").select('estado', 'contexto_modelo', 'modelo_json').limit(CFG['max_modelos_driver'] + 1).collect()
    if len(fitted_b) > CFG['max_modelos_driver']:
        raise RuntimeError('Contextos ajustados excedem limite de coleta.')
    MODELOS_B = {(r.estado, r.contexto_modelo): json.loads(r.modelo_json) for r in fitted_b}
    MODELOS_B2 = {key: c02_compor_modelo(m, MODELOS_A[key[0]])[0] for key, m in MODELOS_B.items()}
    sup_rows = suporte.limit(CFG['max_modelos_driver'] + 1).collect()
    if len(sup_rows) > CFG['max_modelos_driver']:
        raise RuntimeError('Catálogo de suporte acima do limite.')
    SUPORTE = {(r.estado, r.contexto_modelo): r.status_suporte for r in sup_rows}
    CATALOGO = {r['estado']: r.asDict() for r in catalogo.drop('run_id_treinamento').collect()}
    mlflow.log_metrics({'n_modelos_a': len(MODELOS_A), 'n_modelos_b': len(MODELOS_B), 'n_modelos_b2': len(MODELOS_B2)})
    # B2 conserva p_B e curvas A; a recombinação não aplica novo otimizador.
    for key, b2 in MODELOS_B2.items():
        if not np.array_equal(b2['p_destino'], MODELOS_B[key]['p_destino']):
            raise RuntimeError('B2 alterou p_B.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Avaliação geral e ações escolhidas
# MAGIC Ranking útil somente com alvo observado. Perdas com desfecho conhecido. Censurados permanecem na cobertura e não viram negativos.
# MAGIC Reporta médias por cliente; precisão por recomendação usa outro denominador explícito. Lift é descritivo, com um snapshot por cliente/idade, selecionado antes de excluir censurados; empates recebem seleção proporcional.
# MAGIC Bootstrap é pareado por cliente, exploratório e sem correção de multiplicidade. Não estima efeito causal de campanhas nem certifica calibração populacional.

# COMMAND ----------

with etapa('AVALIACAO_HISTORICA_RAW_E_UTIL'):
    labels = ms03_rotular(holdout, ALVOS, CFG['max_passos_macro'])
    ages = spark.createDataFrame([(float(x),) for x in CFG['idades_avaliacao_seg']], 'idade_seg double')
    snapshots = holdout.filter('elegivel_ajuste').crossJoin(F.broadcast(ages)).filter(F.col('dur_min') > F.col('idade_seg'))
    snapshots = snapshots.join(labels, ['cd_bv', 'passo'], 'left')
    snapshots = snapshots.withColumn('memoria', F.coalesce('contexto_negocio', F.lit(MS_SEM_MEMORIA)))
    exigir_vazio('snapshot sem label', snapshots.filter(F.col('status_label').isNull()))
    snapshots = snapshots.persist(StorageLevel.DISK_ONLY)
    registrar_relatorio('TREINO_04_LABELS', snapshots.groupBy('idade_seg', 'status_label').agg(
        F.count('*').alias('n_snapshots'), F.countDistinct('cd_bv').alias('n_clientes')))
    conditions = snapshots.select('estado', 'memoria', 'idade_seg').distinct().limit(CFG['max_condicoes_avaliacao'] + 1).collect()
    if not conditions or len(conditions) > CFG['max_condicoes_avaliacao']:
        raise RuntimeError('Quantidade de condições vazia/acima da proteção do driver.')
    INPUT_KERNELS = pd.DataFrame([r.asDict() for r in conditions])
    payloads, graph_rows, kernel_rows = {}, [], []
    for variant in (MS_A, MS_B2):
        motor = MotorMacroMemoria(variant, CATALOGO, TECNICAS, MODELOS_A, MODELOS_B2,
            SUPORTE, ALVOS, CFG['max_passos_macro'], max_nos=CFG['max_nos_grafo'])
        payload = completar_payload(motor, op_compilar_motor(motor))
        payload['variante'] = variant
        if sum(no['coef'].nbytes for no in payload['nos'].values()) > CFG['max_payload_mb'] * 1024**2:
            raise RuntimeError('Coeficientes excedem a proteção de memória. Reveja o grafo; não foi truncado.')
        if CFG['autotestes']:
            op_testar_compilacao(motor, payload)
        graph_rows.append(motor.resumo)
        payloads[variant] = payload
        calculated = prever_dataframe(payload, INPUT_KERNELS)
        for i, values in enumerate(calculated.itertuples(index=False, name=None)):
            key = INPUT_KERNELS.iloc[i]
            kernel_rows.append((str(key.estado), str(key.memoria), float(key.idade_seg), variant,
                                *[None if pd.isna(v) else v for v in values]))
        del motor
    kernels = spark.createDataFrame(kernel_rows, 'estado string, memoria string, idade_seg double, variante string, ' + OUTPUT_SCHEMA)
    del kernel_rows
    pred = snapshots.join(kernels, ['estado', 'memoria', 'idade_seg'], 'left')
    pred = metricas_avaliacao(pred, ALVOS)
    exigir_vazio('predição não associada ao snapshot', pred.filter(F.col('tem_previsao').isNull()))
    pred = salvar(pred, 'validacao')
    registrar_relatorio('TREINO_05_COBERTURA', pred.groupBy('variante', 'idade_seg', 'status_modelo').agg(
        F.count('*').alias('n_snapshots'), F.countDistinct('cd_bv').alias('n_clientes')))
    gerais, pc, seg, metric_names = resumir_avaliacao(pred)
    gerais = salvar(gerais, 'metricas')
    registrar_relatorio('TREINO_06_METRICAS_CLIENTE', gerais.orderBy('segmento', 'idade_seg', 'variante'))
    por_acao = resumo_por_acao(pred, ALVOS)
    registrar_relatorio('TREINO_07_ACOES_ESCOLHIDAS', por_acao.filter(
        F.col('acao').isin(CFG['acoes_avaliacao'])).orderBy('idade_seg', 'acao', 'variante'))
    registrar_relatorio('TREINO_07_TODAS_ACOES', por_acao.orderBy('idade_seg', 'acao', 'variante'))
    lifts = lift_acao(spark, pred, CFG['acoes_avaliacao'], CFG['semente_avaliacao'])
    registrar_relatorio('TREINO_08_LIFT_DESCRITIVO', lifts.orderBy('idade_seg', 'acao', 'variante', 'fracao'))
    registrar_relatorio('TREINO_09_MASSAS', pred.filter('tem_previsao').groupBy('variante', 'idade_seg').agg(
        *[F.avg(c).alias(c) for c in ['massa_acao_util', 'massa_sem_modelo', 'massa_terminal', 'massa_retencao_interna', 'massa_limite_macro']],
        F.max('erro_massa').alias('max_erro_massa')))
    # Médias por cliente são pequenas. Nenhuma linha bruta de eventos vai ao bootstrap.
    rows_pc = pc.limit(CFG['max_agregados_driver'] + 1).collect()
    if len(rows_pc) > CFG['max_agregados_driver']:
        raise RuntimeError('Agregados para bootstrap acima da proteção.')
    pairs_pdf = pd.DataFrame([r.asDict() for r in rows_pc])
    paired = []
    rng = np.random.default_rng(CFG['semente_avaliacao'])
    for (segment, age), g in pairs_pdf.groupby(['segmento', 'idade_seg']):
        for metric in metric_names:
            piv = g.pivot(index='cd_bv', columns='variante', values=metric)
            if not {MS_A, MS_B2} <= set(piv.columns):
                continue
            v = piv[[MS_A, MS_B2]].dropna()
            if v.empty:
                continue
            x, y = v[MS_A].to_numpy(float), v[MS_B2].to_numpy(float)
            diff = x - y if 'brier' in metric or 'logloss' in metric else y - x
            means = []
            if len(diff) > 1:
                for begin in range(0, CFG['bootstrap_replicas'], 100):
                    size = min(100, CFG['bootstrap_replicas'] - begin)
                    indices = rng.integers(0, len(diff), (size, len(diff)))
                    means.extend(diff[indices].mean(axis=1).tolist())
            lo, hi = (map(float, np.quantile(means, [.025, .975])) if means else (None, None))
            paired.append((segment, float(age), metric, len(v), float(x.mean()), float(y.mean()),
                           float(diff.mean()), lo, hi))
    paired_df = spark.createDataFrame(paired, 'segmento string, idade_seg double, metrica string, n_clientes long, '
        'media_a double, media_b2 double, ganho_b2 double, ic95_lo double, ic95_hi double')
    registrar_relatorio('TREINO_10_COMPARACAO_PAREADA', paired_df.orderBy('segmento', 'idade_seg', 'metrica'))
    # Métricas de fácil comparação no MLflow. Logloss conservador não certifica calibração.
    for row in gerais.collect():
        d = row.asDict()
        prefix = f"{d['variante']}/{d['segmento']}/idade_{int(d['idade_seg'])}"
        for name in metric_names:
            value = d[name]
            if value is not None and math.isfinite(float(value)):
                mlflow.log_metric(f'{prefix}/{name}', float(value))
    mlflow.log_dict(graph_rows, 'relatorios/grafos.json')
    snapshots.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Registrar modelo MLflow e carregar novamente
# MAGIC Artefatos: código, JSON dos parâmetros, NPZ dos coeficientes, catálogo, regra de preparo, assinatura e dependências.
# MAGIC A run só é concluída se o pacote carregado reproduzir os testes de inferência. Registro opcional no Model Registry não altera aliases.
# MAGIC Ao concluir, copie **model_uri** para o notebook 02; não precisa copiar IDs de etapas antigas.

# COMMAND ----------

with etapa('EMPACOTAMENTO_MLFLOW_E_TESTE_DE_RELOAD'):
    BUNDLE = {
        'contrato': 'nba_v23_b2_mlflow_v1', 'versao_pipeline': VERSAO_PIPELINE,
        'run_id_treinamento': RUN_ID, 'alvos': ALVOS, 'tecnicas': sorted(TECNICAS),
        'max_passos_macro': CFG['max_passos_macro'],
        'catalogo': list(CATALOGO.values()),
        'preparo': {k: CFG[k] for k in ('fuso', 'inicio_historico', 'corte_historico_exclusivo',
                    'debounce_seg', 'sal_amostra', 'modulo_amostra_clientes',
                    'sal_validacao', 'modulo_validacao_clientes')},
        'fonte': META_FONTE, 'numerico': NUM,
        'modelo_alvo': 'primeira_acao_util_em_ate_5_mudancas_macro',
        'formato_scores': 'nao_renormalizados; sem prazo em dias; nao calibrados por esta entrega',
        'separacao_validacao': 'holdout_por_cliente_no_mesmo_periodo; nao_testetemporal_independente',
        'elegibilidade_produto': False, 'calendario_frequencia_adicional': False,
        'modelos_a': MODELOS_A,
        'modelos_b': [{'estado': k[0], 'contexto': k[1], 'modelo': m} for k, m in MODELOS_B.items()],
        'modelos_b2': [{'estado': k[0], 'contexto': k[1], 'modelo': m} for k, m in MODELOS_B2.items()],
        'suporte': [{'estado': k[0], 'contexto': k[1], 'status': v} for k, v in SUPORTE.items()],
    }
    # Exemplo contém só nomes de estados/memória, sem cd_bv ou informação pessoal.
    example_keys = [n for n in payloads[MS_B2]['nos'].values()][:4]
    example = pd.DataFrame([{'estado': n['estado'], 'memoria': n['memoria'], 'idade_seg': age}
                            for n in example_keys for age in (0.0, 1800.0)])
    expected = prever_dataframe(payloads[MS_B2], example)
    BUNDLE['exemplo_inferencia'] = example.to_dict(orient='records')
    (TMP / 'bundle.json').write_text(json.dumps(BUNDLE, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    salvar_payload(payloads[MS_B2], TMP / 'kernel_meta.json', TMP / 'coeficientes.npz')
    artifacts = {'bundle': str(TMP / 'bundle.json'), 'metadata': str(TMP / 'kernel_meta.json'),
                 'coeficientes': str(TMP / 'coeficientes.npz')}
    hashes = {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in artifacts.items()}
    (TMP / 'integridade.json').write_text(json.dumps(hashes, sort_keys=True), encoding='utf-8')
    artifacts['integridade'] = str(TMP / 'integridade.json')
    signature = ModelSignature(
        inputs=Schema([ColSpec('string', 'estado'), ColSpec('string', 'memoria'), ColSpec('double', 'idade_seg')]),
        outputs=Schema([ColSpec(part.strip().split()[1], part.strip().split()[0]) for part in OUTPUT_SCHEMA.split(',')]),
    )
    requirements = [f'mlflow=={mlflow.__version__}', f'numpy=={np.__version__}',
                    f'pandas=={pd.__version__}', f'scipy=={scipy.__version__}']
    info_modelo = mlflow.pyfunc.log_model(
        artifact_path='modelo', python_model=str(TMP / 'modelo_pyfunc.py'),
        code_paths=[str(TMP / 'nba_engine.py'), str(TMP / 'nba_data.py')],
        artifacts=artifacts, signature=signature, input_example=example,
        pip_requirements=requirements,
    )
    MODEL_URI = info_modelo.model_uri
    restored = mlflow.pyfunc.load_model(MODEL_URI)
    obtained = restored.predict(example)
    for col in OUTPUT_COLUMNS:
        if col.startswith('massa_') or col == 'erro_massa':
            if not np.allclose(expected[col], obtained[col], rtol=0, atol=1e-9, equal_nan=True):
                raise RuntimeError(f'Reload divergiu em {col}.')
        elif expected[col].tolist() != obtained[col].tolist():
            raise RuntimeError(f'Reload divergiu em {col}.')
    # Também exercita preparação Spark a partir do CÓDIGO que foi empacotado no modelo.
    if CFG['autotestes']:
        restored.unwrap_python_model().teste_preparo(spark)
    conferir_identidade(spark, META_FONTE)
    mlflow.log_dict({'run_id': RUN_ID, 'model_uri': MODEL_URI, 'tabelas': TABELAS,
                     'fonte': META_FONTE, 'reload': 'APROVADO', 'promocao': 'NAO_REALIZADA'}, 'entrega.json')
    if CFG['registered_model_name']:
        mv = mlflow.register_model(MODEL_URI, CFG['registered_model_name'])
        mlflow.set_tag('registered_model_version', str(mv.version))
        print('Modelo registrado sem alterar alias:', mv.name, mv.version)
    mlflow.set_tags({'estado_pacote': 'CONCLUIDO', 'teste_reload': 'APROVADO',
                     'model_uri': MODEL_URI, 'promovido_producao': 'false'})
    fim = spark.createDataFrame([(
        'CONCLUIDO_TREINO_VALIDACAO', MODEL_URI, START.isoformat(), datetime.now(timezone.utc).isoformat(),
        json.dumps({'config': CFG, 'fonte': META_FONTE, 'tabelas': TABELAS}, ensure_ascii=False),
    )], 'status string, model_uri string, inicio_utc string, fim_utc string, config_json string')
    salvar(fim, 'execucoes')
    mlflow.end_run(status='FINISHED')
print('\nTREINO_11_CONCLUSAO')
print('Experimento:', CFG['experimento_mlflow'])
print('run_id:', RUN_ID)
print('model_uri:', MODEL_URI)
print('COPIE model_uri para o notebook 02. Não houve promoção automática.')
print('Este treino usou dados reconstruídos da fonte. Datas/versão diferentes podem alterar resultados legitimamente.')
