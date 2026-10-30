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
