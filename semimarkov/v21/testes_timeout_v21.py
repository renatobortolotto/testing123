
"""Testes numericos locais; nao requerem Spark nem dados do banco.
Execute na pasta dos arquivos: python testes_timeout_v21.py
"""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats

ROOT = Path(__file__).resolve().parent
PATCH = ROOT / "nba_mvp_02d_corrigir_timeout_v21.py"
AUDIT = ROOT / "nba_mvp_03_auditar_timeout_v21.py"


def load_definitions(path, module_name, selected=None):
    tree = ast.parse(path.read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and (selected is None or node.name in selected)
    ]
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    from dataclasses import dataclass, asdict
    from typing import Any
    from collections.abc import Iterator
    module.__dict__.update(
        np=np, pd=pd, scipy=scipy, optimize=optimize, special=special, stats=stats,
        dataclass=dataclass, asdict=asdict, Any=Any, Iterator=Iterator,
        SMD_TOL_SEG=1e-9, SMD_FORMATO="semimarkov_timeout_misto_v21",
        SMD_TOP_K=5, SM_TOL=1e-8,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


M = load_definitions(PATCH, "smd_core_tests")
A = load_definitions(AUDIT, "smd_audit_tests",
                     {"sm_audit_log_s_grupos", "sm_auditar_temporal_pdf"})


def base(times, destinations, types_=None):
    times = np.asarray(times, float)
    if types_ is None:
        types_ = np.array(["exata"] * len(times))
    return pd.DataFrame({
        "cd_bv": [f"cliente_{i % 200}" for i in range(len(times))],
        "destino": destinations,
        "dur_min": times,
        "dur_max": np.where(np.asarray(types_) == "direita", np.nan, times),
        "tipo_censura": types_,
    })


def legacy_model():
    return {
        "formato": "semimarkov_lognormal_grupos_v2",
        "familia": "lognormal",
        "unidade": "dias",
        "destinos": ["A", "B"],
        "grupo": [0, 1],
        "r_destino_no_grupo": [1., 1.],
        "pi_grupo": [0.65, 0.35],
        "mu_grupo": [-3., 1.],
        "sigma_grupo": [0.9, 1.2],
        "p_destino": [0.65, 0.35],
        "n_grupos": 2,
        "max_tempo_observado_dias": 50.,
        "id_modelo_origem": "hash_legado",
    }


# Schema textual da saida; extraido sem executar as celulas Spark.
for _node in ast.parse(PATCH.read_text()).body:
    if isinstance(_node, ast.Assign):
        _names = [t.id for t in _node.targets if isinstance(t, ast.Name)]
        if "SMD_SCHEMA_OUTPUT" in _names:
            M.SMD_SCHEMA_OUTPUT = ast.literal_eval(_node.value)
M.SMD_COLUMNS_OUTPUT = [
    item.strip().split()[0] for item in M.SMD_SCHEMA_OUTPUT.split(",")
]

class TimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.atom_data = base(
            np.full(300, 1800.), ["sem_acao:::classe"] * 300
        )
        cls.atom = M.smd_ajustar_origem(cls.atom_data)
        cls.atom["id_modelo_origem"] = "hash_atomo"

        rng = np.random.default_rng(123)
        n = 1600
        dest = np.where(rng.random(n) < .55, "sem_acao:::classe", "NAVEGACAO")
        span = np.where(
            rng.random(n) < .8, 0., np.exp(rng.normal(np.log(130.), .8, n))
        )
        times = np.where(
            dest == "sem_acao:::classe",
            1800. + span,
            np.exp(rng.normal(np.log(75.), 1., n)),
        )
        censor = np.exp(rng.normal(np.log(4000.), 1.5, n))
        known = times < censor
        cls.mix_data = base(
            np.minimum(times, censor),
            np.where(known, dest, None),
            np.where(known, "exata", "direita"),
        )
        cls.mixed = M.smd_ajustar_origem(cls.mix_data)
        cls.mixed["id_modelo_origem"] = "hash_misto"

    def test_01_pure_atom_requires_no_sdlog(self):
        self.assertEqual(self.atom["tipo_grupo"], ["atomo_timeout"])
        self.assertEqual(self.atom["sigma_grupo"], [None])
        self.assertEqual(self.atom["n_iteracoes"], 0)

    def test_02_mass_at_timeout_recovered(self):
        group = self.mixed["tipo_grupo"].index("atomo_mais_lognormal")
        self.assertAlmostEqual(self.mixed["massa_atomo_grupo"][group], .8, delta=.1)
        self.assertEqual(self.mixed["atraso_grupo_seg"][group], 1800.)

    def test_03_cutoff_equality_uses_left_survival(self):
        q, qh, ns = M.smd_prever_destinos(self.atom, [1800. / 86400.])
        np.testing.assert_allclose(q, 1.)
        np.testing.assert_allclose(qh, 1.)
        np.testing.assert_allclose(ns, 0.)

    def test_04_atom_expired_is_not_static_fallback(self):
        with self.assertRaisesRegex(ValueError, "IDADE_FORA_SUPORTE"):
            M.smd_prever_destinos(self.atom, [1801. / 86400.])

    def test_05_horizon_exclusive_upper_boundary(self):
        _, qh, ns = M.smd_prever_destinos(self.atom, [0.], 1800. / 86400.)
        np.testing.assert_allclose(qh, 0.)
        np.testing.assert_allclose(ns, 1.)

    def test_06_normalization_and_time_dependence(self):
        q, qh, ns = M.smd_prever_destinos(
            self.mixed, np.array([0., 600., 1800., 1801., 3000.]) / 86400.
        )
        np.testing.assert_allclose(q.sum(1), 1., atol=1e-10)
        np.testing.assert_allclose(qh.sum(1) + ns, 1., atol=1e-10)
        self.assertGreater(np.abs(q[-1] - q[0]).max(), .05)
        self.assertTrue(np.all(qh <= q + 1e-12))

    def test_07_joint_gradient(self):
        cfg = M.AjusteTimeout()
        prep = M.smd_preparar_ajuste(self.mix_data.iloc[:800], cfg)
        theta = prep["initial"].copy()
        _, analytic = M.smd_objetivo(theta, prep, cfg)
        h = 1e-5
        numerical = np.empty_like(theta)
        for k in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[k] += h
            minus[k] -= h
            numerical[k] = (
                M.smd_objetivo(plus, prep, cfg)[0]
                - M.smd_objetivo(minus, prep, cfg)[0]
            ) / (2 * h)
        np.testing.assert_allclose(analytic, numerical, atol=1e-6, rtol=1e-5)

    def test_08_censored_likelihood_is_mixture(self):
        model = self.mixed
        age = 1000. / 86400.
        s = np.exp(M.smd_log_s_grupos(model, [age])[0])
        pi = np.asarray(model["pi_grupo"])
        self.assertTrue(0. < pi @ s <= 1.)
        # Mixture is not a product nor a replication per destination.
        self.assertNotAlmostEqual(pi @ s, np.prod(s))

    def test_09_interval_mass_formula(self):
        tau = 1800.
        omega = .7
        eta = np.log(omega / (1 - omega))
        mu, ls = np.log(100. / 86400.), np.log(.8)
        got = M.smd_termos_grupo(
            np.array([1700.]), np.array([1900.]), np.array([1]),
            "atomo_mais_lognormal", tau, mu, ls, eta,
        )[0][0]
        expected = omega + (1 - omega) * stats.lognorm.cdf(
            100. / 86400., s=.8, scale=np.exp(mu)
        )
        self.assertAlmostEqual(np.exp(got), expected, places=12)

    def test_10_silence_does_not_enter_rare_pool(self):
        rng = np.random.default_rng(6)
        df = base(
            np.r_[np.exp(rng.normal(np.log(60), 1., 300)), [1800.] * 3, [1., 3.]],
            ["A"] * 300 + ["sem_acao:::classe"] * 3 + ["RARO_B", "RARO_C"],
        )
        prep = M.smd_preparar_ajuste(df, M.AjusteTimeout())
        mapping = dict(zip(prep["destinos"], prep["grupo"]))
        self.assertNotEqual(mapping["sem_acao:::classe"], mapping["RARO_B"])
        self.assertEqual(mapping["RARO_B"], mapping["RARO_C"])

    def test_11_legacy_predictions_preserved(self):
        model = legacy_model()
        before = copy.deepcopy(model)
        ages = np.array([0., .02, 1., 10.])
        q, qh, ns = M.smd_prever_destinos(model, ages, 7.)
        lp = np.log(model["p_destino"])
        ls = stats.lognorm.logsf(
            ages[:, None], s=model["sigma_grupo"], scale=np.exp(model["mu_grupo"])
        )
        den = special.logsumexp(lp + ls, axis=1)
        expected = np.exp(lp + ls - den[:, None])
        np.testing.assert_allclose(q, expected, atol=1e-12)
        self.assertEqual(model, before)

    def test_12_no_top5_renormalization(self):
        model = legacy_model()
        model.update(
            destinos=[f"D{i}" for i in range(8)], grupo=[0] * 8,
            r_destino_no_grupo=[1 / 8] * 8, pi_grupo=[1.],
            mu_grupo=[0.], sigma_grupo=[1.], p_destino=[1 / 8] * 8,
            n_grupos=1,
        )
        q, _, _ = M.smd_prever_destinos(model, [1.])
        self.assertAlmostEqual(q[0, :5].sum(), 5 / 8)

    def test_13_bounds_remain_for_non_structural_constant(self):
        df = base(np.full(250, 25.), ["A"] * 250)
        with self.assertRaisesRegex(ValueError, "LIMITE_REMANESCENTE"):
            M.smd_ajustar_origem(df)

    def test_14_minimum_support_unchanged(self):
        with self.assertRaisesRegex(ValueError, "SUPORTE_INSUFICIENTE"):
            M.smd_ajustar_origem(self.atom_data.iloc[:15])

    def test_15_no_imputation_of_intervals(self):
        df = self.atom_data.copy()
        df.loc[0, "tipo_censura"] = "intervalo"
        df.loc[0, "dur_max"] = 1801.
        with self.assertRaisesRegex(ValueError, "PATCH_REQUER_EXATAS_DIREITA"):
            M.smd_ajustar_origem(df)

    def test_16_independent_audit_and_corruption_detection(self):
        models = {"ORIGEM": self.mixed}
        inp = pd.DataFrame({
            "cd_bv": ["c1", "c2"],
            "acao_atual": ["ORIGEM"] * 2,
            "status_input": ["OK"] * 2,
            "tempo_no_estado_seg": [10., 1900.],
            "data_referencia": [pd.Timestamp("2026-09-26").date()] * 2,
            "ts_corte_estado": [pd.Timestamp("2026-09-25")] * 2,
            "status_dados": ["CORTE_RETROSPECTIVO_NAO_D1"] * 2,
            "relogio": ["SILENCIO_OPERACIONAL_TIMEOUT_V2"] * 2,
        })
        out = M.smd_prever_pdf(inp, models, "teste_v21", 7.)
        good = A.sm_auditar_temporal_pdf(out, models, 7.)
        self.assertEqual(int(good["n_erros"].iloc[0]), 0)
        bad = out.copy()
        bad.loc[0, "prob_proxima_acao"] = bad.loc[0, "prob_na_entrada"]
        check = A.sm_auditar_temporal_pdf(bad, models, 7.)
        self.assertGreater(int(check["n_erros"].iloc[0]), 0)

    def test_17_unsupported_input_preserved(self):
        inp = pd.DataFrame({
            "cd_bv": ["sem_historico", "depois_atomo"],
            "acao_atual": [None, "ORIGEM"],
            "status_input": ["SEM_HISTORICO", "OK"],
            "tempo_no_estado_seg": [np.nan, 1801.],
            "data_referencia": [pd.Timestamp("2026-09-26").date()] * 2,
            "ts_corte_estado": [pd.Timestamp("2026-09-25")] * 2,
            "status_dados": ["RETROSPECTIVO"] * 2,
            "relogio": ["TESTE"] * 2,
        })
        out = M.smd_prever_pdf(inp, {"ORIGEM": self.atom}, "teste", 7.)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["status_previsao"]),
                         {"SEM_HISTORICO", "IDADE_FORA_SUPORTE_MODELO"})
        self.assertTrue(out["ranking"].isna().all())

    def test_18_json_has_no_nan(self):
        json.dumps(self.mixed, allow_nan=False)
        json.dumps(self.atom, allow_nan=False)

    def test_19_lower_support_silence_is_rejected(self):
        df = self.atom_data.copy()
        df.loc[0, ["dur_min", "dur_max"]] = 1799.
        with self.assertRaisesRegex(ValueError, "SAIDA_PARA_SILENCIO_ANTES_TIMEOUT"):
            M.smd_ajustar_origem(df)

    def test_20_shifted_tail_not_constant_1800(self):
        rng = np.random.default_rng(88)
        t = 1800. + np.exp(rng.normal(np.log(30), .75, 700))
        df = base(t, ["sem_acao:::classe"] * 700)
        fit = M.smd_ajustar_origem(df)
        self.assertEqual(fit["tipo_grupo"], ["lognormal"])
        self.assertEqual(fit["atraso_grupo_seg"], [1800.])
        s = np.exp(M.smd_log_s_grupos(fit, np.array([1700., 1800., 2000.]) / 86400.))
        np.testing.assert_allclose(s[:2], 1.)
        self.assertLess(s[-1, 0], .1)


if __name__ == "__main__":
    print(f"Python: {sys.version.split()[0]} | SciPy: {scipy.__version__}")
    unittest.main(verbosity=2)
