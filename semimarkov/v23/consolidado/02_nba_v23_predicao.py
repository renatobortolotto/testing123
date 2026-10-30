# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — NBA V2.3: predição consolidada para homologação
# MAGIC
# MAGIC Notebook independente: não treina nem escolhe automaticamente uma candidata.
# MAGIC Carrega o ID explicitamente configurado, verifica A/B/B2 e reproduz kernels
# MAGIC históricos antes do scoring. Mantém os **17 alvos aprovados**, Top 5 e horizonte
# MAGIC original. Integra a correção de leitura Delta na própria célula, sem patch.
# MAGIC
# MAGIC O padrão aponta para o modelo já aprovado e escreve em um prefixo novo de
# MAGIC homologação. Para avaliar a cópia do notebook 01, use os três valores impressos
# MAGIC em `CONS_01_CONCLUSAO`. Não troque a view consumida pelo negócio antes da comparação.
# MAGIC
# MAGIC `customers_query` deve existir nesta sessão; em job, configure uma fonte permanente
# MAGIC ou uma etapa explícita de ingestão. Não inventa a consulta de elegibilidade.
# MAGIC Sem calendário, novas frequências, retreino ou regras de produto.

# COMMAND ----------

import json
import math
import pickle
import re
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import special
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# ÚNICO bloco de configuração. Não usa IDs globais das outras partes.
OP_CFG = {
    "id_ms03_validado": "92a92d57-ce93-487c-bdf1-369e3dbd542f",
    "tabela_ms03_execucoes": (
        "ctg_dsti.renato_nba.nba_sm_v23_estavel_execucoes_hml"
    ),
    "tabela_pacotes": "ctg_dsti.renato_nba.nba_sm_v23_estavel_pacotes_hml",
    "fonte_publico": "customers_query",
    "coluna_data_publico": "AUTO",  # AUTO procura dt_snapshot_publico.
    "fonte_eventos": "ctg_dsti.renato_nba.base_jornadas_onboarding",
    "modo_scoring": "AMOSTRA",  # AMOSTRA, TODOS ou IDS.
    "modulo_amostra": 1000,  # Aproximadamente 0,1% dos IDs elegíveis.
    "ids_validacao": [],  # Em AMOSTRA, acrescenta IDs elegíveis; em IDS, limita a eles.
    "sal_amostra": "nba_v23_operacional_v1",
    "corte_eventos_iso": None,  # Opcional: watermark confiável com timezone.
    "max_atraso_eventos_horas": 72.0,  # Guarda operacional; não parâmetro do modelo.
    "max_atraso_publico_dias": 7,  # Só verificável quando a fonte contém a data.
    "debounce_seg": 30.0,  # Contrato atual; não otimizar neste notebook.
    "arrow_linhas_lote": 1000,
    "sub_lote_numerico": 256,
    "max_payload_mb": 128,
    "max_linhas_modelos": 5000,
    "n_kernels_regressao": 120,
    "max_linhas_exibir": 20,
    "prefixo_saida": "ctg_dsti.renato_nba.nba_sm_v23_consolidada_operacional",
    "autotestes": True,
}
OP_VERSAO = "v2.3_operacional_b2_consolidada_v1"
OP_ID = str(uuid.uuid4())
OP_INICIO = datetime.now(timezone.utc)
OP_FONTES = {}
OP_RELATORIOS = {}
OP_CACHES = []
OP_SAIDAS = {
    k: f"{OP_CFG['prefixo_saida']}_{k}_hml"
    for k in ("publico", "estados", "predicoes", "negocio", "resumos", "execucoes")
}

if "spark" not in globals():
    raise RuntimeError("Execute no Databricks com Spark e acesso às fontes.")
try:
    import pyarrow  # Necessário nos workers para mapInPandas.
except ImportError as exc:
    raise RuntimeError("O runtime deve ter PyArrow, Pandas, NumPy e SciPy.") from exc

OP_CFG["modo_scoring"] = str(OP_CFG["modo_scoring"]).upper()
if OP_CFG["modo_scoring"] not in {"AMOSTRA", "TODOS", "IDS"}:
    raise ValueError("modo_scoring deve ser AMOSTRA, TODOS ou IDS.")
if OP_CFG["modo_scoring"] == "IDS" and not OP_CFG["ids_validacao"]:
    raise ValueError("Preencha ids_validacao quando modo_scoring='IDS'.")
if (OP_CFG["modulo_amostra"] < 1 or OP_CFG["arrow_linhas_lote"] < 1
        or OP_CFG["sub_lote_numerico"] < 1):
    raise ValueError("Tamanhos de amostra e lotes devem ser positivos.")
if OP_CFG["debounce_seg"] != 30.0:
    raise ValueError("Esta versão preserva o debounce validado de 30 segundos.")
for nome in ("max_atraso_eventos_horas", "max_atraso_publico_dias"):
    if OP_CFG[nome] <= 0:
        raise ValueError(f"{nome} deve ser positivo.")

spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", OP_CFG["arrow_linhas_lote"])
print("Execução operacional:", OP_ID, "| modo:", OP_CFG["modo_scoring"])


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

if OP_CFG["prefixo_saida"] != "ctg_dsti.renato_nba.nba_sm_v23_consolidada_operacional":
    raise RuntimeError("Mantenha o prefixo isolado durante esta homologação.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Motor de referência e cálculo em lotes
# MAGIC O núcleo abaixo é o mesmo utilizado no Multi-step histórico. Não aprende parâmetros.
# MAGIC A otimização operacional agrega os coeficientes de continuação por grupo temporal;
# MAGIC o resultado é verificado contra o núcleo, inclusive em idades positivas. Só o catálogo
# MAGIC pequeno e os coeficientes chegam ao driver. Os clientes são processados em lotes Spark.

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

# COMMAND ----------

OP_HASH_NUCLEO_REFERENCIA = "59eb0d888488e145c837767cd84088d4461f1a5a53ac09aa531334fcc392aac6"

# COMMAND ----------

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

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Leitura por ID e persistência operacional
# MAGIC O pacote aprovado é uma cópia física independente. Não usa versões expiradas do treino. As fontes operacionais Delta são fixadas na versão vigente ao iniciar a leitura.
# MAGIC Cada execução nova grava em seu próprio ID. Só a conclusão libera a view
# MAGIC de consumo; etapas incompletas não aparecem como execução válida.

# COMMAND ----------

def op_sql(nome: str) -> str:
    partes = nome.split(".")
    if not 1 <= len(partes) <= 3 or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p) for p in partes
    ):
        raise ValueError(f"Identificador SQL inválido: {nome!r}")
    return ".".join(f"`{p}`" for p in partes)


def op_colunas(df: DataFrame, campos: set, origem: str) -> None:
    faltam = campos - set(df.columns)
    if faltam:
        raise ValueError(f"{origem}: faltam as colunas {sorted(faltam)}.")


def op_zero(rotulo: str, df: DataFrame) -> None:
    if df.limit(1).count():
        raise RuntimeError(f"Verificação falhou: {rotulo}. Nenhuma view será atualizada.")


def op_unico(df: DataFrame, rotulo: str) -> dict:
    rows = df.limit(2).collect()
    if len(rows) != 1:
        raise RuntimeError(f"{rotulo}: esperado um único registro; encontrados {len(rows)}.")
    return rows[0].asDict(recursive=True)


def op_pequeno(df: DataFrame, limite: int, rotulo: str) -> list:
    rows = df.limit(limite + 1).collect()
    if len(rows) > limite:
        raise RuntimeError(f"{rotulo}: excede a proteção do driver ({limite} linhas).")
    return rows


def op_ler(tabela: str) -> DataFrame:
    op_sql(tabela)
    if not spark.catalog.tableExists(tabela):
        raise RuntimeError(
            f"Fonte indisponível: {tabela}. Atualize a ingestão ou o nome em OP_CFG. "
            "Não é necessário repetir o treinamento. Uma view temporária de outro "
            "notebook pode não existir nesta sessão; prefira uma tabela permanente."
        )
    return spark.table(tabela)


# Registra apenas DataFrames abertos nesta execução operacional.
# Não use este registro para reaproveitar uma execução parcial anterior.
_OP_FONTES_FIXADAS = {}
OP_CFG["politica_leitura_fontes"] = "snapshot_delta_inicio_v1"


def _op_meta_delta(tabela: str) -> dict | None:
    """Retorna identidade e versão atual, sem ler as linhas da tabela."""
    info = spark.catalog.getTable(tabela)
    if info.isTemporary or str(info.tableType).upper() == "VIEW":
        return None

    detalhe = spark.sql(
        f"DESCRIBE DETAIL {op_sql(tabela)}"
    ).first().asDict()
    if str(detalhe.get("format", "")).lower() != "delta":
        return None
    if not detalhe.get("id") or not detalhe.get("location"):
        raise RuntimeError(f"Identidade Delta incompleta: {tabela}")

    historico = spark.sql(
        f"DESCRIBE HISTORY {op_sql(tabela)} LIMIT 1"
    ).first()
    if historico is None:
        raise RuntimeError(f"Histórico Delta indisponível: {tabela}")

    return {
        "id": str(detalhe["id"]),
        "location": str(detalhe["location"]),
        "version": int(historico["version"]),
    }


def op_versao_delta(tabela: str):
    """Compatibilidade com o restante do operacional."""
    meta = _op_meta_delta(tabela)
    return None if meta is None else meta["version"]


def _op_mesma_tabela(tabela: str, inicio: dict, fim: dict | None) -> None:
    """Novos commits são permitidos; troca da tabela/localização, não."""
    if fim is None or any(
        inicio[campo] != fim[campo] for campo in ("id", "location")
    ):
        raise RuntimeError(
            f"A identidade/localização da fonte {tabela} mudou. "
            "Não publicar esta execução; investigar a substituição da tabela."
        )
    if fim["version"] < inicio["version"]:
        raise RuntimeError(
            f"O histórico de {tabela} retrocedeu. "
            "Não será usada outra tabela silenciosamente."
        )


def op_fonte_operacional(tabela: str) -> DataFrame:
    """Fixa a versão Delta vigente na primeira leitura de cada fonte."""
    if tabela in OP_FONTES:
        info = OP_FONTES[tabela]
        if (
            info.get("id_execucao_leitura") != OP_ID
            or tabela not in _OP_FONTES_FIXADAS
        ):
            raise RuntimeError(
                "Há fontes de uma execução anterior/não fixada na sessão. "
                "Reexecute desde a configuração do operacional, com novo OP_ID."
            )
        return _OP_FONTES_FIXADAS[tabela]

    df = op_ler(tabela)
    meta = _op_meta_delta(tabela)
    if meta is None:
        if tabela == OP_CFG["fonte_eventos"]:
            raise RuntimeError(
                "A fonte de eventos deve ser uma tabela Delta permanente "
                "para fixar a versão durante o scoring."
            )
        if tabela != OP_CFG["fonte_publico"]:
            raise RuntimeError(f"Fonte não Delta não prevista: {tabela}")
        # O público de uma view continua sendo materializado em _publico_hml
        # pela seção 6. Isto não certifica as versões das bases dessa view.
        info = {
            "id_execucao_leitura": OP_ID,
            "modo_leitura": "PUBLICO_SEM_VERSAO_DELTA",
            "versao_delta_inicio": None,
            "versao_delta_fim": None,
        }
        print(
            f"PÚBLICO SEM VERSÃO DELTA: {tabela}. "
            "Será persistido pela seção 6; não é um snapshot das bases da view.",
            flush=True,
        )
    else:
        versao = meta["version"]
        print(
            f"FONTE FIXADA: {tabela} | versão Delta={versao}",
            flush=True,
        )
        # Versão atual capturada nesta execução, não versão antiga do treino.
        # Falha de leitura NÃO aciona fallback para a versão mais recente.
        df = (
            spark.read.option("versionAsOf", versao)
            .table(tabela)
        )
        # Uma leitura pequena detecta bloqueios antes do processamento pesado.
        # Não comprova a disponibilidade de todos os arquivos.
        df.limit(1).collect()
        depois = _op_meta_delta(tabela)
        _op_mesma_tabela(tabela, meta, depois)
        info = {
            "id_execucao_leitura": OP_ID,
            "modo_leitura": "DELTA_SNAPSHOT_INICIO",
            "id_tabela_delta": meta["id"],
            "localizacao_delta": meta["location"],
            "versao_delta_inicio": versao,
            "versao_delta_lida": versao,
            "versao_delta_fim": depois["version"],
        }

    OP_FONTES[tabela] = info
    _OP_FONTES_FIXADAS[tabela] = df
    return df


def op_verificar_fontes() -> None:
    """Verifica a identidade sem abortar por novos commits após a leitura."""
    for tabela, info in OP_FONTES.items():
        if (
            info.get("id_execucao_leitura") != OP_ID
            or tabela not in _OP_FONTES_FIXADAS
        ):
            raise RuntimeError(
                f"{tabela} não foi aberta pelo leitor fixo desta execução. "
                "Reinicie apenas o operacional; não pule esta verificação."
            )

        modo = info.get("modo_leitura")
        if modo == "PUBLICO_SEM_VERSAO_DELTA":
            if tabela != OP_CFG["fonte_publico"]:
                raise RuntimeError("Fonte não Delta fora do contrato.")
            if _op_meta_delta(tabela) is not None:
                raise RuntimeError("O tipo da fonte de público mudou.")
            continue

        if modo != "DELTA_SNAPSHOT_INICIO":
            raise RuntimeError(f"Modo de leitura não reconhecido: {modo!r}")

        inicio = {
            "id": info["id_tabela_delta"],
            "location": info["localizacao_delta"],
            "version": info["versao_delta_lida"],
        }
        fim = _op_meta_delta(tabela)
        _op_mesma_tabela(tabela, inicio, fim)
        info["versao_delta_fim"] = fim["version"]
        info["fonte_avancou"] = fim["version"] != inicio["version"]

        if info["fonte_avancou"]:
            print(
                f"FONTE AVANÇOU: {tabela} | "
                f"versão usada={inicio['version']} | "
                f"versão atual={fim['version']}. "
                "Mantida a versão fixada; novos dados entram na próxima execução.",
                flush=True,
            )


def op_assinatura(df: DataFrame) -> dict:
    # Mesma assinatura usada ao gravar o pacote histórico, com o mesmo fuso.
    j = F.to_json(F.struct(*[F.col(c) for c in sorted(df.columns)]), {
        "ignoreNullFields": "false",
        "timestampFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSSXXX",
        "timestampNTZFormat": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS",
    })
    h = df.select(F.sha2(j, 256).alias("h"))
    a = [F.count("*").alias("n")]
    for i in range(4):
        a.append(F.sum(F.conv(F.substring("h", i * 16 + 1, 16), 16, 10)
                       .cast("decimal(38,0)")).alias(f"s{i}"))
    v = h.agg(*a).first().asDict()
    return {"n": int(v["n"]), "somas_sha256": [str(v[f"s{i}"] or 0) for i in range(4)]}


def op_gravar(df: DataFrame, nome: str) -> DataFrame:
    tabela = OP_SAIDAS[nome]
    proibidas = set(OP_FONTES) | set(OP_TABELAS_MODELO)
    if tabela in proibidas:
        raise RuntimeError("Saída operacional não pode coincidir com fonte.")
    saida = df.withColumn("id_execucao", F.lit(OP_ID))
    saida = saida.withColumn("id_ms03_validado", F.lit(OP_CFG["id_ms03_validado"]))
    saida = saida.withColumn("id_pacote_modelo", F.lit(OP_META["id_pacote"]))
    saida = saida.withColumn("versao_operacional", F.lit(OP_VERSAO))
    saida = saida.withColumn("modo_scoring", F.lit(OP_CFG["modo_scoring"]))
    if spark.catalog.tableExists(tabela):
        anterior = spark.table(tabela)
        op_colunas(anterior, {"id_execucao"}, tabela)
        if anterior.filter(F.col("id_execucao") == OP_ID).limit(1).count():
            raise RuntimeError(
                f"{nome} já foi escrito neste ID. Use Executar tudo para nova execução."
            )
    (saida.write.format("delta").mode("append")
     .partitionBy("id_execucao").saveAsTable(tabela))
    return spark.table(tabela).filter(F.col("id_execucao") == OP_ID)


def op_relatorio(nome: str, df: DataFrame, ordem=None) -> None:
    OP_RELATORIOS[nome] = df
    print("\n" + nome)
    if ordem:
        df = df.orderBy(*ordem)
    for campo in df.schema.fields:
        if campo.dataType.simpleString() in {"double", "float"}:
            df = df.withColumn(campo.name, F.round(F.col(campo.name), 6))
    df.show(OP_CFG["max_linhas_exibir"], truncate=False)


# Verifica disponibilidade das duas entradas antes dos cálculos do motor.
# Apenas checagem do contrato; não utiliza views como histórico congelado.
for _fonte in (OP_CFG["fonte_publico"], OP_CFG["fonte_eventos"]):
    op_ler(_fonte)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Carregar somente o motor aprovado
# MAGIC Um único `id_ms03_validado` encadeia automaticamente o pacote e os modelos.
# MAGIC Nenhuma consulta às versões antigas das tabelas de treino é necessária.

# COMMAND ----------

if OP_CFG["autotestes"]:
    ms_autotestes_numericos()

hist = op_unico(op_ler(OP_CFG["tabela_ms03_execucoes"]).filter(
    (F.col("id_ms03") == OP_CFG["id_ms03_validado"])
    & (F.col("status") == "CONCLUIDO_MULTISTEP_HISTORICO_03")
), "Validação Multi-step concluída")
hist_cfg = json.loads(hist["config_json"])
if ms_hash(hist_cfg) != hist["hash_ms03"]:
    raise RuntimeError("Hash da execução histórica não confere.")
hist_tabelas = json.loads(hist["tabelas_json"])
id_pacote = hist_cfg["pacote_entradas"]
pacote = op_unico(op_ler(OP_CFG["tabela_pacotes"]).filter(
    (F.col("id_pacote") == id_pacote) & (F.col("status") == "CONCLUIDO_PACOTE")
), "Pacote de modelos")
OP_META = json.loads(pacote["meta_json"])
if pacote["contrato"] != "pacote_logico_verificado_v1":
    raise RuntimeError("Contrato do pacote incompatível.")
if ms_hash(OP_META) != pacote["hash_pacote"] or OP_META["id_pacote"] != id_pacote:
    raise RuntimeError("Hash ou ID do pacote divergente.")
for campo, campo_hist in (("id_experimento", "id_experimento"),
                           ("id_ajuste", "id_ajuste_fonte"),
                           ("id_02c", "id_02c_fonte")):
    if OP_META[campo] != hist[campo_hist]:
        raise RuntimeError(f"O pacote não corresponde ao {campo} validado.")

OP_FUSO = OP_META["fit_config"]["fonte_preparo"]["fuso_base"]
spark.conf.set("spark.sql.session.timeZone", OP_FUSO)
OP_ALVOS = sorted(hist_cfg["alvos"])
OP_TECNICAS = set(hist_cfg["acoes_tecnicas"])
OP_PASSOS = int(hist_cfg["config"]["max_passos_macro"])
if OP_ALVOS != ALVOS_APROVADOS or OP_PASSOS != 5:
    raise RuntimeError("O pacote deve conter os 17 alvos aprovados e cinco mudanças de macroação.")
OP_TABELAS_MODELO = {
    OP_CFG["tabela_pacotes"], OP_CFG["tabela_ms03_execucoes"],
    *[v["tabela"] for v in OP_META["entradas"].values()],
}
OP_ENTRADAS = {}
for tipo in ("config", "modelos_ab", "modelos_b2", "suporte"):
    info = OP_META["entradas"][tipo]
    df = op_ler(info["tabela"]).filter(F.col("id_pacote") == id_pacote).select(*info["colunas"])
    if df.schema.jsonValue() != info["schema"] or op_assinatura(df) != info["assinatura"]:
        raise RuntimeError(f"O conteúdo do pacote foi alterado: {tipo}.")
    OP_ENTRADAS[tipo] = df

cfg_cols = ["estado", "funil", "etapa", "tipo_estado"]
if "terminal_real" in OP_ENTRADAS["config"].columns:
    cfg_cols.append("terminal_real")
OP_CONFIG_DF = OP_ENTRADAS["config"].select(*cfg_cols)
op_zero("config duplicada", OP_CONFIG_DF.groupBy("estado").count().filter("count != 1"))
OP_CONFIG = {
    r["estado"]: r.asDict()
    for r in op_pequeno(OP_CONFIG_DF, 1000, "Catálogo do motor")
}
if None in OP_CONFIG or any(
    v.get("terminal_real") or v["tipo_estado"] in {"terminal", "silencio", "conversao"}
    for v in OP_CONFIG.values()
):
    raise RuntimeError("Pacote incompatível com a cadeia clean sem conversão sintética.")
OP_SUPORTE = {
    (r.estado, r.contexto_modelo): r.status_suporte
    for r in op_pequeno(OP_ENTRADAS["suporte"].select(
        "estado", "contexto_modelo", "status_suporte"
    ), OP_CFG["max_linhas_modelos"], "Suporte")
}
originais = OP_ENTRADAS["modelos_ab"].filter(
    (F.col("status_modelo") == "AJUSTADO") & ~F.col("reuso")
    & F.col("variante").isin(MS_A, "B_MEMORIA")
)
OP_A, OP_B, OP_B2 = {}, {}, {}
for row in op_pequeno(originais, OP_CFG["max_linhas_modelos"], "Modelos A/B"):
    m = json.loads(row.modelo_json)
    ms_validar_modelo(m)
    if row.variante == MS_A:
        if row.estado in OP_A or row.nivel != "ORIGEM":
            raise RuntimeError("Modelo A duplicado ou fora de nível.")
        OP_A[row.estado] = m
    else:
        if (row.estado, row.contexto_modelo) in OP_B or row.nivel != "CONTEXTO":
            raise RuntimeError("Modelo B duplicado ou fora de nível.")
        OP_B[row.estado, row.contexto_modelo] = m
for row in op_pequeno(OP_ENTRADAS["modelos_b2"], OP_CFG["max_linhas_modelos"], "Modelos B2"):
    key = (row.estado, row.contexto_modelo)
    if key in OP_B2 or row.variante != MS_B2 or row.status_modelo != "COMPOSTO_SEM_RETREINO":
        raise RuntimeError("Contrato B2 inválido ou duplicado.")
    a, b = OP_A[row.estado], OP_B[key]
    if ms_hash(a) != row.hash_tempo_fonte or ms_hash(b) != row.hash_prob_fonte:
        raise RuntimeError("Componentes fonte de B2 divergentes.")
    m = json.loads(row.modelo_json)
    ms_validar_modelo(m)
    tempos_a = {d: (a["mu_grupo"][g], a["sigma_grupo"][g])
                for d, g in zip(a["destinos"], a["grupo"])}
    if m["p_destino"] != b["p_destino"]:
        raise RuntimeError("B2 alterou as probabilidades de entrada de B.")
    for d, g in zip(m["destinos"], m["grupo"]):
        if tempos_a[d] != (m["mu_grupo"][g], m["sigma_grupo"][g]):
            raise RuntimeError("B2 alterou as curvas temporais compartilhadas.")
    OP_B2[key] = m

OP_MOTOR = MotorMacroMemoria(
    MS_B2, OP_CONFIG, OP_TECNICAS, OP_A, OP_B2, OP_SUPORTE, OP_ALVOS,
    passos=OP_PASSOS, max_nos=6000, max_arestas=1000000, tol=1e-8,
)
OP_PAYLOAD = op_compilar_motor(OP_MOTOR)
if OP_CFG["autotestes"]:
    op_testar_compilacao(OP_MOTOR, OP_PAYLOAD)

# Regressão contra scores REALMENTE persistidos no teste histórico aprovado.
regressao = op_ler(hist_tabelas["kernels"]).filter(
    (F.col("id_ms03") == OP_CFG["id_ms03_validado"])
    & (F.col("variante") == MS_B2) & F.col("tem_previsao")
).orderBy("idade_seg", "estado", "memoria_grafo")
# Amostra só de verificação do motor; não amostra nem modifica o treinamento.
reg_rows = []
por_idade = max(1, OP_CFG["n_kernels_regressao"] // 4)
for idade in (0.0, 1800.0, 86400.0, 604800.0):
    reg_rows.extend(regressao.filter(F.col("idade_seg") == idade).limit(por_idade).collect())
if not reg_rows:
    raise RuntimeError("Sem kernels históricos para verificar a reprodução operacional.")
for row in reg_rows:
    i = OP_MOTOR.indice[(row.estado, row.memoria_grafo)]
    no = OP_PAYLOAD["nos"][i]
    r = op_probabilidades_lote(no, np.array([row.idade_seg]), len(OP_ALVOS), OP_PASSOS)
    er = np.array([row.raw_probs[d] for d in no["destinos"]])
    eu = np.array([row.util_probs[d] for d in OP_ALVOS])
    if not (np.allclose(er, r["raw"][0], atol=1e-9)
            and np.allclose(eu, r["scores"][0], atol=1e-9)):
        raise RuntimeError("O motor operacional não reproduz o kernel histórico salvo.")

OP_DIM_NOS = spark.createDataFrame([
    (s, c, int(i), bool(i in OP_PAYLOAD["nos"]), OP_MOTOR.rotas[i])
    for i, (s, c) in enumerate(OP_MOTOR.nos)
], "estado_atual string, memoria_grafo string, id_no int, tem_modelo boolean, rota_modelo string")
tamanho_payload = len(pickle.dumps(OP_PAYLOAD, protocol=pickle.HIGHEST_PROTOCOL))
if tamanho_payload > OP_CFG["max_payload_mb"] * 1024**2:
    raise RuntimeError("Catálogo operacional excede o limite de broadcast.")
print("Pacote B2:", id_pacote, "| regressões históricas:", len(reg_rows))
print("Grafo:", OP_MOTOR.resumo)
print("Payload numérico:", round(tamanho_payload / 1024**2, 2), "MB")
# Somente arrays/dicionários pequenos nos workers; nenhum SparkSession na função.
OP_BC = spark.sparkContext.broadcast(OP_PAYLOAD)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Estado atual e memória, sem olhar o futuro
# MAGIC Preserva maior profundidade no mesmo instante e debounce de 30 segundos.
# MAGIC Um empate permanece ambíguo. Estado novo fora do catálogo vira barreira,
# MAGIC nunca é removido para recuperar uma ação antiga como se fosse a atual.
# MAGIC O histórico é o disponível na fonte; não se limita silenciosamente a 30/90 dias.

# COMMAND ----------

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


def op_autotestes_estado() -> None:
    config = spark.createDataFrame([
        ("p", "pix", "sucesso", "funil_conclusao"),
        ("b", "boleto", "sucesso", "funil_conclusao"),
        ("l", "login", "topo", "funil_abertura"),
        ("ln", "login", "navegacao", "funil_meio"),
        ("ls", "login", "sucesso", "funil_conclusao"),
        ("chat", None, None, "atendimento"),
    ], "estado string, funil string, etapa string, tipo_estado string")
    rows = [
        ("u1", 0, "b", 1.0), ("u1", 10, "l", 0.0),
        ("u1", 20, "l", 0.0), ("u1", 25, "ln", 0.5),
        ("u1", 30, "ls", 1.0), ("u1", 40, "p", 1.0),
        ("u2", 0, "p", 1.0), ("u2", 20, "l", 1.0),
        ("u2", 20, "b", 1.0), ("u2", 40, "l", 0.0),
        ("u3", 0, "b", 1.0), ("u3", 20, "chat", None),
        ("u3", 40, "l", 0.0),
        ("u4", 0, "b", 1.0), ("u4", 20, "novo_estado", 1.0),
        ("u4", 40, "l", 0.0),
        ("u5", 0, "l", 0.0), ("u5", 0, "ls", 1.0),
    ]
    df = spark.createDataFrame(rows, "cd_bv string, seg long, estado string, profundidade_max double")
    df = df.withColumn("ts_evento", F.expr("timestamp_micros(1700000000000000 + seg * 1000000)"))
    r = op_preparar_acoes(df.drop("seg"), config, {"login"}, 30.0)
    vals = r.select("cd_bv", "estado", "contexto_negocio", "ordem_ambigua",
                    F.expr("unix_micros(ts_estado)").alias("us")).collect()
    u1 = sorted([x for x in vals if x.cd_bv == "u1"], key=lambda x: x.us)
    assert [x.estado for x in u1] == ["b", "l", "ln", "ls", "p"]
    assert [x.contexto_negocio for x in u1] == ["boleto"] * 4 + ["pix"]
    assert u1[1].us == 1700000020000000  # Última evidência do burst, não início.
    ultimos = {u: max([x for x in vals if x.cd_bv == u], key=lambda x: x.us)
               for u in ("u2", "u3", "u4", "u5")}
    assert ultimos["u2"].contexto_negocio is None
    assert ultimos["u3"].contexto_negocio == "boleto"
    assert ultimos["u4"].contexto_negocio is None
    assert ultimos["u5"].estado == "ls"
    assert any(x.cd_bv == "u2" and x.ordem_ambigua for x in vals)
    prefix = op_preparar_acoes(df.filter("seg <= 30").drop("seg"), config, {"login"}, 30.0)
    got = prefix.filter("cd_bv = 'u1'").orderBy(F.desc("ts_estado")).first()
    assert got.contexto_negocio == "boleto"  # PIX futuro não influencia o prefixo.
    print("Operacional: estado, debounce, ambiguidade e memória Spark OK.")


if OP_CFG["autotestes"]:
    op_autotestes_estado()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Função distribuída de scoring
# MAGIC Não envia milhões de clientes ao driver. O cálculo usa grupos dentro de
# MAGIC pequenos lotes; não utiliza applyInPandas por cliente nem monta um grafo por pessoa.

# COMMAND ----------

OP_SCORE_COLS = ["cd_bv", "alerta_limite", "extrapolacao_tempo", "erro_massa"]
OP_SCORE_SCHEMA = ["cd_bv string", "alerta_limite boolean", "extrapolacao_tempo boolean", "erro_massa double"]
for rank in range(1, 6):
    for col, tipo in ((f"raw_{rank}", "string"), (f"raw_p_{rank}", "double"),
                      (f"util_{rank}", "string"), (f"util_p_{rank}", "double"),
                      (f"passo_{rank}", "int")):
        OP_SCORE_COLS.append(col)
        OP_SCORE_SCHEMA.append(f"{col} {tipo}")
OP_MASSAS = ["massa_acao_util", "massa_sem_modelo", "massa_terminal", "massa_retencao_interna", "massa_limite_macro"]
OP_SCORE_COLS += OP_MASSAS
OP_SCORE_SCHEMA += [f"{c} double" for c in OP_MASSAS]
OP_SCORE_SCHEMA = ", ".join(OP_SCORE_SCHEMA)


def op_scoring_iterator(iterator, payload: dict, sub_lote: int, colunas: list):
    alvos, k = payload["alvos"], payload["k"]
    na = len(alvos)
    for pdf in iterator:
        if pdf.empty:
            continue
        pdf = pdf.reset_index(drop=True)
        saida = []
        for no_id, sub in pdf.groupby("id_no", sort=False):
            no = payload["nos"].get(int(no_id))
            if no is None:
                raise RuntimeError("Um nó sem modelo chegou ao cálculo numérico.")
            for inicio in range(0, len(sub), sub_lote):
                lote = sub.iloc[inicio:inicio + sub_lote]
                idades = lote["idade_seg"].to_numpy(dtype=float)
                r = op_probabilidades_lote(no, idades, na, k)
                raw_order = np.argsort(-r["raw"], axis=1, kind="stable")[:, :5]
                util_order = np.argsort(-r["scores"], axis=1, kind="stable")[:, :5]
                for row_index, cliente in enumerate(lote["cd_bv"]):
                    out = {c: None for c in colunas}
                    out.update({"cd_bv": str(cliente), "alerta_limite": no["alerta_limite"],
                                "extrapolacao_tempo": bool(idades[row_index] / 86400.0 > no["max_dias"]),
                                "erro_massa": float(r["erro"][row_index]),
                                "massa_acao_util": float(r["scores"][row_index].sum())})
                    for rank, dest in enumerate(raw_order[row_index], start=1):
                        out[f"raw_{rank}"] = no["destinos"][int(dest)]
                        out[f"raw_p_{rank}"] = float(r["raw"][row_index, dest])
                    for rank, alvo in enumerate(util_order[row_index], start=1):
                        score = float(r["scores"][row_index, alvo])
                        if score <= 0:
                            continue
                        out[f"util_{rank}"] = alvos[int(alvo)]
                        out[f"util_p_{rank}"] = score
                        out[f"passo_{rank}"] = int(np.argmax(r["contrib"][row_index, alvo]) + 1)
                    for pos, nome in enumerate(OP_MASSAS[1:]):
                        out[nome] = float(r["massas"][row_index, pos])
                    saida.append(out)
        out_pdf = pd.DataFrame(saida, columns=colunas)
        for rank in range(1, 6):
            out_pdf[f"passo_{rank}"] = pd.array(out_pdf[f"passo_{rank}"], dtype="Int32")
        yield out_pdf


# Só fecha sobre broadcast e valores simples, não sobre DataFrames/SparkSession.
def op_funcao_worker(broadcast, sub_lote: int, colunas: list):
    def executar(iterator):
        yield from op_scoring_iterator(iterator, broadcast.value, sub_lote, colunas)
    return executar


OP_WORKER = op_funcao_worker(OP_BC, OP_CFG["sub_lote_numerico"], OP_SCORE_COLS)
# Verifica transporte Arrow e bibliotecas nos workers antes de ler eventos em escala.
no_probe = sorted(OP_PAYLOAD["nos"])[0]
probe = spark.createDataFrame([
    ("__teste_0__", no_probe, 0.0), ("__teste_1__", no_probe, 86400.0),
], "cd_bv string, id_no int, idade_seg double")
probe_out = probe.mapInPandas(OP_WORKER, OP_SCORE_SCHEMA).collect()
if len(probe_out) != 2 or any(x.erro_massa > 1e-8 for x in probe_out):
    raise RuntimeError("O autoteste de transporte/scoring nos workers falhou.")
print("Operacional: transporte Arrow e scoring nos workers OK.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Público e corte operacional
# MAGIC A elegibilidade define apenas QUEM pode receber NBA. Seu snapshot não é o
# MAGIC relógio do motor. Sem watermark explícito, o corte será o maior timestamp
# MAGIC observado na fonte até o início da execução; isso NÃO comprova completude.
# MAGIC O pipeline interrompe quando a fonte excede a tolerância de atraso configurada.

# COMMAND ----------

# Registra o instante no Spark e utiliza inteiros de microssegundos nas contas.
OP_AGORA_US = int(spark.sql("SELECT unix_micros(current_timestamp()) AS us").first()["us"])
publico_fonte = op_fonte_operacional(OP_CFG["fonte_publico"])
op_colunas(publico_fonte, {"cd_bv"}, "Público elegível")
col_data = OP_CFG["coluna_data_publico"]
if col_data == "AUTO":
    col_data = "dt_snapshot_publico" if "dt_snapshot_publico" in publico_fonte.columns else None
if col_data and col_data not in publico_fonte.columns:
    raise ValueError(f"Público: a coluna {col_data!r} não existe.")

publico_fonte = publico_fonte.select(
    F.col("cd_bv").cast("string").alias("cd_bv"),
    (F.to_date(F.col(col_data)) if col_data else F.lit(None).cast("date"))
    .alias("data_snapshot_publico"),
)
op_zero("IDs inválidos no público", publico_fonte.filter(
    F.col("cd_bv").isNull() | (F.length(F.trim("cd_bv")) == 0)
))
if col_data:
    op_zero("data nula no snapshot do público", publico_fonte.filter("data_snapshot_publico IS NULL"))
    data_publico = publico_fonte.agg(F.max("data_snapshot_publico").alias("d")).first()["d"]
    publico_fonte = publico_fonte.filter(F.col("data_snapshot_publico") == F.lit(data_publico))
    atraso_pub = spark.range(1).select(
        F.datediff(F.to_date(F.expr(f"timestamp_micros({OP_AGORA_US})")), F.lit(data_publico)).alias("d")
    ).first()["d"]
    if atraso_pub is None or atraso_pub < 0 or atraso_pub > OP_CFG["max_atraso_publico_dias"]:
        raise RuntimeError("Snapshot do público futuro/vazio/desatualizado. Atualize a fonte.")
else:
    data_publico = None
    print("ATENÇÃO: a fonte do público só informa IDs. Data real do snapshot permanece NULL.")
publico = publico_fonte.dropDuplicates(["cd_bv"])
if OP_CFG["modo_scoring"] == "AMOSTRA":
    amostra = F.pmod(F.xxhash64("cd_bv", F.lit(OP_CFG["sal_amostra"])),
                    F.lit(OP_CFG["modulo_amostra"])) == 0
    if OP_CFG["ids_validacao"]:
        amostra = amostra | F.col("cd_bv").isin([str(x) for x in OP_CFG["ids_validacao"]])
    publico = publico.filter(amostra)
elif OP_CFG["modo_scoring"] == "IDS":
    publico = publico.filter(F.col("cd_bv").isin([str(x) for x in OP_CFG["ids_validacao"]]))
publico = op_gravar(publico, "publico")
OP_N_PUBLICO = publico.count()
if OP_N_PUBLICO == 0:
    raise RuntimeError("Nenhum cliente no escopo. Confira amostra/IDs/elegibilidade.")
print("Público selecionado:", OP_N_PUBLICO, "| data informada pela fonte:", data_publico)

fonte = op_fonte_operacional(OP_CFG["fonte_eventos"])
op_colunas(fonte, {"cd_bv", "dm_navegacao", "estado", "profundidade_max"}, "Eventos reais")
if fonte.schema["dm_navegacao"].dataType.simpleString() != "timestamp":
    raise ValueError("dm_navegacao deve ser timestamp, como no preparo validado. Não converter silenciosamente.")
filtro_nao_futuro = F.col("dm_navegacao") <= F.expr(f"timestamp_micros({OP_AGORA_US})")
limites = fonte.agg(
    F.min(F.when(filtro_nao_futuro, F.expr("unix_micros(dm_navegacao)"))).alias("min_us"),
    F.max(F.when(filtro_nao_futuro, F.expr("unix_micros(dm_navegacao)"))).alias("max_us"),
    F.sum((~filtro_nao_futuro).cast("long")).alias("n_eventos_futuros"),
).first().asDict()
if limites["max_us"] is None:
    raise RuntimeError("Fonte de eventos vazia ou sem timestamp válido até o início da execução.")
if OP_CFG["corte_eventos_iso"]:
    corte = datetime.fromisoformat(OP_CFG["corte_eventos_iso"].replace("Z", "+00:00"))
    if corte.tzinfo is None:
        raise ValueError("corte_eventos_iso deve informar timezone/offset.")
    OP_CORTE_US = int(round(corte.timestamp() * 1_000_000))
    OP_REGRA_CORTE = "WATERMARK_INFORMADO"
else:
    OP_CORTE_US = int(limites["max_us"])
    OP_REGRA_CORTE = "MAXIMO_OBSERVADO_NAO_COMPROVA_COMPLETUDE"
if OP_CORTE_US > OP_AGORA_US or OP_CORTE_US < int(limites["min_us"]):
    raise ValueError("Corte futuro ou anterior ao histórico disponível.")
# Em watermark informado, mede a disponibilidade de eventos até esse corte.
if OP_CORTE_US < int(limites["max_us"]):
    ultimo_ate_corte = fonte.filter(
        F.col("dm_navegacao") <= F.expr(f"timestamp_micros({OP_CORTE_US})")
    ).agg(F.max(F.expr("unix_micros(dm_navegacao)")).alias("us")).first()["us"]
else:
    ultimo_ate_corte = limites["max_us"]
atraso_horas = (OP_AGORA_US - int(ultimo_ate_corte)) / 3_600_000_000.0
if atraso_horas > OP_CFG["max_atraso_eventos_horas"]:
    raise RuntimeError(
        f"Fonte de eventos está {atraso_horas:.1f} horas atrás do início desta execução. "
        "Atualize a ingestão para avaliar o público atual; não repita o treino."
    )
print("Corte:", datetime.fromtimestamp(OP_CORTE_US / 1e6, timezone.utc).isoformat(),
      "| regra:", OP_REGRA_CORTE, "| atraso horas:", round(atraso_horas, 2))
print("Eventos futuros fora do corte:", limites["n_eventos_futuros"] or 0)

eventos = fonte.select(
    F.col("cd_bv").cast("string").alias("cd_bv"),
    F.col("dm_navegacao").alias("ts_evento"),
    F.col("estado").cast("string").alias("estado"),
    F.col("profundidade_max").cast("double").alias("profundidade_max"),
).join(publico.select("cd_bv"), "cd_bv", "left_semi")
# Retém timestamp nulo para que a verificação de qualidade não o esconda.
eventos = eventos.filter(
    (F.col("ts_evento") <= F.expr(f"timestamp_micros({OP_CORTE_US})"))
    | F.col("ts_evento").isNull()
)
qualidade = eventos.agg(
    F.sum((F.col("ts_evento").isNull() | F.col("estado").isNull()
           | (F.length(F.trim("estado")) == 0)).cast("long")).alias("invalidos"),
    F.sum(F.lower("estado").rlike("sem_acao|conversao").cast("long")).alias("sinteticos"),
    F.sum((F.isnan("profundidade_max") | (F.abs("profundidade_max") == float("inf")))
          .cast("long")).alias("profundidade_invalida"),
).first().asDict()
if any(int(v or 0) for v in qualidade.values()):
    raise RuntimeError(f"Qualidade da fonte incompatível com o contrato clean: {qualidade}")

acoes = op_preparar_acoes(eventos, OP_CONFIG_DF, OP_TECNICAS, OP_CFG["debounce_seg"])
wlast = Window.partitionBy("cd_bv").orderBy(F.desc("ts_estado"), F.desc("passo"))
ultimo = acoes.withColumn("_rn", F.row_number().over(wlast)).filter("_rn = 1").drop("_rn")
estados = publico.select("cd_bv", "data_snapshot_publico").join(ultimo, "cd_bv", "left")
estados = estados.withColumn("status_dado", F.when(
    F.col("ts_estado").isNull(), "SEM_HISTORICO_NA_FONTE"
).when(F.col("ordem_ambigua"), "ESTADO_ATUAL_AMBIGUO").when(
    ~F.coalesce("estado_catalogado", F.lit(False)), "ESTADO_NAO_CATALOGADO"
).otherwise("OK"))
estados = estados.withColumn("memoria_grafo", F.when(
    F.col("status_dado") == "OK", F.coalesce("contexto_negocio", F.lit(MS_SEM_MEMORIA))
)).withColumn("contexto_modelo", F.when(
    F.col("status_dado") == "OK", F.when(F.col("origem_tecnica"), F.col("memoria_grafo"))
    .otherwise(F.lit(MS_BASE))
))
estados = estados.withColumnRenamed("estado", "estado_atual").withColumnRenamed("ts_estado", "ts_ultima_acao")
estados = estados.withColumn("acao_atual", F.coalesce("funil", "estado_atual"))
estados = estados.withColumn("ts_corte_eventos", F.expr(f"timestamp_micros({OP_CORTE_US})"))
estados = estados.withColumn("ts_execucao", F.expr(f"timestamp_micros({OP_AGORA_US})"))
estados = estados.withColumn("data_referencia", F.to_date("ts_corte_eventos"))
estados = estados.withColumn("idade_seg", F.when(
    F.col("ts_ultima_acao").isNotNull(),
    (F.lit(OP_CORTE_US) - F.expr("unix_micros(ts_ultima_acao)")) / F.lit(1e6)
))
estados = estados.join(F.broadcast(OP_DIM_NOS), ["estado_atual", "memoria_grafo"], "left")
op_zero("estado/memória válidos sem nó no grafo", estados.filter(
    (F.col("status_dado") == "OK") & F.col("id_no").isNull()
))
estados = estados.withColumn("tem_modelo", F.coalesce("tem_modelo", F.lit(False)))
estados = estados.withColumn("status_previsao", F.when(
    F.col("status_dado") != "OK", F.col("status_dado")
).when(~F.col("tem_modelo"), "SEM_MODELO_ORIGEM").otherwise("PREVISAO_B2_OPERACIONAL"))
estados = op_gravar(estados, "estados")
op_zero("memória do futuro", estados.filter(
    (F.col("passo_evidencia_memoria") > F.col("passo"))
    | (F.col("ts_evidencia_memoria") > F.col("ts_ultima_acao"))
))
op_zero("estado duplicado por cliente", estados.groupBy("cd_bv").count().filter("count != 1"))
if estados.count() != OP_N_PUBLICO:
    raise RuntimeError("O scoring perdeu ou duplicou clientes elegíveis.")
op_verificar_fontes()
op_relatorio("V23_OP_01_ESTADOS", estados.groupBy("status_dado", "status_previsao").agg(
    F.count("*").alias("n_clientes")
), [F.desc("n_clientes")])
op_relatorio("V23_OP_02_MEMORIA", estados.filter("status_dado = 'OK'").groupBy("origem_tecnica").agg(
    F.count("*").alias("n_clientes"),
    F.sum(F.col("contexto_negocio").isNotNull().cast("long")).alias("n_memoria_conhecida"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Scoring B2 e verificações
# MAGIC Mesma memória no estado atual e em todas as trajetórias. Sem ajustes de
# MAGIC elegibilidade dos produtos nem de calendário. Probabilidades não são renormalizadas.

# COMMAND ----------

entrada_scores = estados.filter("status_previsao = 'PREVISAO_B2_OPERACIONAL'").select(
    "cd_bv", "id_no", "idade_seg"
)
scores = entrada_scores.mapInPandas(OP_WORKER, OP_SCORE_SCHEMA)
pred = estados.join(scores, "cd_bv", "left")
raw_array = F.array(*[F.struct(
    F.lit(r).cast("int").alias("ranking"), F.col(f"raw_{r}").alias("estado"),
    F.col(f"raw_p_{r}").alias("score"),
) for r in range(1, 6)])
util_array = F.array(*[F.struct(
    F.lit(r).cast("int").alias("ranking"), F.col(f"util_{r}").alias("acao"),
    F.col(f"util_p_{r}").alias("score"),
    F.col(f"passo_{r}").alias("passo_macro_maior_contribuicao"),
) for r in range(1, 6)])
pred = pred.withColumn("_raw", raw_array).withColumn("_util", util_array)
pred = pred.withColumn("raw_top5", F.expr("filter(_raw, x -> x.estado IS NOT NULL)"))
pred = pred.withColumn("util_top5", F.expr("filter(_util, x -> x.acao IS NOT NULL)"))
pred = pred.withColumn("status_temporal", F.when(
    F.col("status_previsao") != "PREVISAO_B2_OPERACIONAL", "NAO_CALCULADO"
).when(F.col("extrapolacao_tempo"), "EXTRAPOLACAO_TEMPO_COMPARTILHADO")
    .otherwise("SEMIMARKOV_TEMPO_COMPARTILHADO"))
descartar = ["_raw", "_util", *[f"{p}_{r}" for r in range(1, 6)
                                for p in ("raw", "raw_p", "util", "util_p", "passo")]]
pred = pred.drop(*descartar)
pred = op_gravar(pred, "predicoes")
op_zero("previsão ausente após cálculo", pred.filter(
    (F.col("status_previsao") == "PREVISAO_B2_OPERACIONAL") & F.col("erro_massa").isNull()
))
op_zero("previsões duplicadas", pred.groupBy("cd_bv").count().filter("count != 1"))
if pred.count() != OP_N_PUBLICO:
    raise RuntimeError("O número de clientes mudou depois do cálculo distribuído.")
op_zero("massa inválida", pred.filter(F.col("massa_acao_util").isNotNull() & (
    (F.abs(sum(F.col(c) for c in OP_MASSAS) - 1.0) > 1e-8)
    | (F.col("erro_massa") > 1e-8) | F.isnan("massa_acao_util")
)))
prev = pred.filter("status_previsao = 'PREVISAO_B2_OPERACIONAL'")
OP_N_PREVISTOS = prev.count()
if OP_N_PREVISTOS == 0:
    raise RuntimeError("Nenhuma previsão calculada. Revise os status; não haverá publicação vazia.")
op_relatorio("V23_OP_03_ROTAS", pred.groupBy("status_previsao", "rota_modelo", "status_temporal").agg(
    F.count("*").alias("n_clientes")
), [F.desc("n_clientes")])
op_relatorio("V23_OP_04_MASSAS", prev.agg(
    *[F.avg(c).alias(c) for c in OP_MASSAS], F.max("erro_massa").alias("max_erro_massa"),
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Tabela de negócio LONG
# MAGIC RAW e útil são rankings independentes. A linha de rank 1 não diz que RAW 1
# MAGIC conduz especificamente a útil 1. Preserva RAW mesmo quando há menos ações úteis.
# MAGIC Clientes sem previsão ficam na tabela técnica, não na view de recomendações.

# COMMAND ----------

negocio = prev.withColumn("ranking", F.explode(F.sequence(F.lit(1), F.lit(5))))
negocio = negocio.withColumn("_raw_map", F.expr(
    "map_from_entries(transform(raw_top5, x -> struct(x.ranking, x)))"
)).withColumn("_util_map", F.expr(
    "map_from_entries(transform(util_top5, x -> struct(x.ranking, x)))"
))
negocio = negocio.withColumn("_r", F.element_at("_raw_map", F.col("ranking")))
negocio = negocio.withColumn("_u", F.element_at("_util_map", F.col("ranking")))
negocio = negocio.filter(F.col("_r.estado").isNotNull() | F.col("_u.acao").isNotNull()).select(
    "data_referencia", "cd_bv", "acao_atual", "ranking",
    F.col("_r.estado").alias("proxima_acao_raw"), F.col("_r.score").alias("score_raw"),
    F.col("_u.acao").alias("proxima_acao_util"), F.col("_u.score").alias("score_util"),
)
negocio = op_gravar(negocio, "negocio")
op_zero("rank duplicado na saída", negocio.groupBy("cd_bv", "ranking").count().filter("count != 1"))
n_negocio = negocio.count()
if negocio.select("cd_bv").distinct().count() != OP_N_PREVISTOS:
    raise RuntimeError("A tabela de negócio perdeu clientes com previsão.")

for tipo in ("raw", "util"):
    op_relatorio(f"V23_OP_05_TOP1_{tipo.upper()}", negocio.filter("ranking = 1").groupBy(
        f"proxima_acao_{tipo}"
    ).agg(F.count("*").alias("n_clientes"), F.avg(f"score_{tipo}").alias("score_medio")),
        [F.desc("n_clientes")])
assinaturas = prev.select(F.expr(
    "to_json(transform(util_top5, x -> struct(x.ranking AS ranking, x.acao AS acao)))"
).alias("assinatura")).groupBy("assinatura").count()
op_relatorio("V23_OP_06_CONCENTRACAO", assinaturas.agg(
    F.count("*").alias("n_listas_distintas"), F.max("count").alias("n_lista_dominante"),
    F.sum("count").alias("n_clientes_previstos"),
).withColumn("fracao_lista_dominante", F.col("n_lista_dominante") / F.col("n_clientes_previstos")))
op_relatorio("V23_OP_07_COBERTURA", spark.createDataFrame([
    (OP_N_PUBLICO, OP_N_PREVISTOS, n_negocio, OP_N_PREVISTOS / OP_N_PUBLICO)
], "n_clientes_elegiveis long, n_clientes_previstos long, n_linhas_negocio long, cobertura double"))

resumos = None
for nome, df in OP_RELATORIOS.items():
    lote = df.select(F.lit(nome).alias("relatorio"), F.to_json(
        F.struct(*[F.col(c) for c in df.columns]), {"ignoreNullFields": "false"}
    ).alias("dados_json"))
    resumos = lote if resumos is None else resumos.unionByName(lote)
op_gravar(resumos, "resumos")
op_verificar_fontes()

# Só a conclusão torna este ID consumível. Não há transação multi-tabela implícita.
run_config = {
    "versao": OP_VERSAO, "config": OP_CFG, "fontes": OP_FONTES,
    "id_pacote": id_pacote, "hash_pacote": pacote["hash_pacote"],
    "hash_nucleo_referencia": OP_HASH_NUCLEO_REFERENCIA,
    "alvos": OP_ALVOS, "tecnicas": sorted(OP_TECNICAS), "fuso": OP_FUSO,
    "ts_corte_micros": OP_CORTE_US, "regra_corte": OP_REGRA_CORTE,
    "data_snapshot_publico": str(data_publico) if data_publico else None,
    "atraso_eventos_horas": atraso_horas,
    "elegibilidade_por_produto": False, "calendario_frequencias": False,
    "confianca_calibrada": False, "uso": "HOMOLOGACAO_AVALIACAO_NEGOCIO_NAO_DISPARAR",
}
manifesto = spark.createDataFrame([(
    "CONCLUIDO_SCORING_B2", OP_INICIO.isoformat(), datetime.now(timezone.utc).isoformat(),
    OP_CORTE_US, OP_N_PUBLICO, OP_N_PREVISTOS, n_negocio,
    json.dumps(run_config, sort_keys=True, ensure_ascii=False), ms_hash(run_config),
    json.dumps(OP_SAIDAS, sort_keys=True), False,
)], "status string, inicio_utc string, fim_utc string, corte_micros long, n_clientes_elegiveis long, "
    "n_clientes_previstos long, n_linhas_negocio long, config_json string, hash_execucao string, "
    "tabelas_json string, autorizado_disparo boolean")
op_gravar(manifesto, "execucoes")

# Atualiza apenas a view do modo executado. AMOSTRA/IDS não alteram a view TODOS.
sufixo_view = {"TODOS": "negocio_atual", "AMOSTRA": "amostra_atual", "IDS": "ids_atual"}[OP_CFG["modo_scoring"]]
OP_VIEW = f"{OP_CFG['prefixo_saida']}_{sufixo_view}"
campos_negocio = ["data_referencia", "cd_bv", "acao_atual", "ranking", "proxima_acao_raw", "score_raw",
                  "proxima_acao_util", "score_util"]
spark.sql(f"""
    CREATE OR REPLACE VIEW {op_sql(OP_VIEW)} AS
    WITH ultima AS (
        SELECT id_execucao
        FROM {op_sql(OP_SAIDAS['execucoes'])}
        WHERE status = 'CONCLUIDO_SCORING_B2'
          AND modo_scoring = '{OP_CFG['modo_scoring']}'
          AND id_ms03_validado = '{OP_CFG['id_ms03_validado']}'
        ORDER BY corte_micros DESC, fim_utc DESC, id_execucao DESC
        LIMIT 1
    )
    SELECT {', '.join('n.' + op_sql(c) for c in campos_negocio)}
    FROM {op_sql(OP_SAIDAS['negocio'])} AS n
    INNER JOIN ultima AS u ON n.id_execucao = u.id_execucao
""")
OP_BC.unpersist(blocking=False)
print("\nV23_OP_08_CONCLUSAO")
print("id_execucao:", OP_ID)
print("id_ms03_validado:", OP_CFG["id_ms03_validado"])
print("id_pacote:", id_pacote)
print("Modo:", OP_CFG["modo_scoring"], "| clientes:", OP_N_PUBLICO, "| previstos:", OP_N_PREVISTOS)
print("View de avaliação:", OP_VIEW)
print("Tabelas:", json.dumps(OP_SAIDAS, indent=2))
print("Nenhum retreino. Nenhum disparo. Nenhuma tabela anterior foi sobrescrita.")
print("Para avaliar pessoa conhecida, filtre cd_bv na view e ordene por ranking.")
# Exemplo após a execução, usando um ID elegível e incluído no escopo:
# display(spark.table(OP_VIEW).filter(F.col('cd_bv') == 'ID_AQUI').orderBy('ranking'))
