"""Testes locais: núcleo/contratos. Não executam Spark/Delta nem MLflow real."""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from scipy.optimize import approx_fprime

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nba_engine as eng
import nba_fit as fit


def modelo(destinos, probs, mu=0.0, sigma=1.0):
    n = len(destinos)
    return {'formato': 'sm_v23_lognormal_grupos', 'unidade_tempo': 'dias',
            'destinos': destinos, 'grupo': list(range(n)), 'n_grupos': n,
            'grupos_nomes': list(destinos),
            'estrutura': {'destinos': destinos, 'grupo': list(range(n)),
                          'grupos_nomes': list(destinos), 'limite_mu': [-20, 20]},
            'pi_grupo': probs, 'mu_grupo': [mu] * n, 'sigma_grupo': [sigma] * n,
            'r_destino_no_grupo': [1.0] * n, 'p_destino': probs,
            'alerta_limite': False, 'parametros_no_limite': [],
            'max_tempo_observado_dias': 100.0}


def sintetico17():
    alvos = [f'a{i:02d}' for i in range(17)]
    config = {a: {'funil': a, 'tipo_estado': 'funil_abertura', 'etapa': 'topo', 'terminal_real': False} for a in alvos}
    config['login'] = {'funil': 'app_login', 'tipo_estado': 'funil_abertura', 'etapa': 'topo', 'terminal_real': False}
    a = {s: modelo(['login'], [1.]) for s in alvos}
    weights = np.arange(1, 18, dtype=float); weights /= weights.sum()
    a['login'] = modelo(alvos, weights.tolist(), mu=-2, sigma=1.1)
    b = modelo(alvos, weights[::-1].tolist(), mu=-1, sigma=.7)
    b2 = fit.c02_compor_modelo(b, a['login'])[0]
    m = eng.MotorMacroMemoria(eng.MS_B2, config, {'app_login'}, a,
        {('login', 'a00'): b2}, {('login', 'a00'): 'CANDIDATO_CONTEXTUAL'}, alvos, 5)
    p = eng.completar_payload(m, eng.op_compilar_motor(m)); p['variante'] = eng.MS_B2
    return m, p, a, b, b2


def treino_sintetico():
    rng = np.random.default_rng(12)
    rows = []
    for i in range(120):
        rows.append((f'u{i % 30}', i, 'login', '__BASE__', 'a' if i % 3 else 'b',
                     float(np.exp(rng.normal(-1, .8)) * 86400), 'exata', 1.0))
    for i in range(15):
        rows.append((f'u{i}', 121+i, 'login', '__BASE__', None,
                     float(np.exp(rng.normal(0, .5)) * 86400), 'direita', 1.0))
    pdf = pd.DataFrame(rows, columns=['cd_bv', 'passo', 'estado', 'contexto_modelo', 'destino', 'dur_min', 'tipo_censura', 'peso_evento_treino'])
    cfg = dict(min_eventos_origem=10, min_clientes_origem=4, min_eventos_grupo=10,
               min_clientes_grupo=4, max_grupos_proprios=8, sigma_min=.15,
               sigma_max=4.5, maxiter=800, lambda_prob_base=.0001,
               lambda_prob_contexto=.02, lambda_tempo_contexto=.02)
    return pdf, cfg


class NucleoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.p, cls.a, cls.b, cls.b2 = sintetico17()
        cls.pdf, cls.cfg = treino_sintetico()
        cls.structure = fit.criar_estrutura(cls.pdf, cls.cfg)
        cls.parent = fit.ajustar_modelo(cls.pdf, 'peso_evento_treino', cls.structure, cls.cfg)

    def test_01_autotestes_referencia(self):
        eng.ms_autotestes_numericos()

    def test_02_massa_17_alvos(self):
        for pair in self.m.nos:
            for age in [0, 1, 1800, 86400, 604800]:
                out = self.m.prever(*pair, float(age))
                self.assertLess(out['erro_massa'], 1e-8)

    def test_03_compilacao_17_alvos(self):
        eng.op_testar_compilacao(self.m, self.p)

    def test_04_b2_preserva_destinos(self):
        self.assertEqual(self.b2['p_destino'], self.b['p_destino'])
        self.assertEqual(self.b2['mu_grupo'], self.a['login']['mu_grupo'])

    def test_05_idade_zero_b_igual_b2(self):
        np.testing.assert_allclose(fit.prever_probabilidades(self.b, [0])[0],
                                   fit.prever_probabilidades(self.b2, [0])[0], atol=1e-15)

    def test_06_batch_ordem_misturada(self):
        pdf = pd.DataFrame([('login','a00',1800.),('a03','a03',0.),('login','a04',86400.),('a00','a00',20.)], columns=['estado','memoria','idade_seg'])
        r = eng.prever_dataframe(self.p, pdf, 1)
        for i, row in enumerate(pdf.itertuples(index=False)):
            ref = self.m.prever(row.estado,row.memoria,row.idade_seg)
            self.assertEqual(json.loads(r.iloc[i].util_probs_json),ref['util_probs'])

    def test_07_top5_17_alvos(self):
        r = eng.prever_dataframe(self.p,pd.DataFrame([('login','a00',0.)], columns=['estado','memoria','idade_seg']))
        self.assertEqual(len(json.loads(r.iloc[0].util_probs_json)),17)
        self.assertEqual(len(json.loads(r.iloc[0].util_top5_json)),5)

    def test_08_roundtrip_json_npz(self):
        with tempfile.TemporaryDirectory() as d:
            meta, arr = Path(d)/'meta.json', Path(d)/'coef.npz'
            eng.salvar_payload(self.p,meta,arr)
            q = eng.carregar_payload(meta,arr)
            inp = pd.DataFrame([('login','a00',1800.)], columns=['estado','memoria','idade_seg'])
            pd.testing.assert_frame_equal(eng.prever_dataframe(self.p,inp),eng.prever_dataframe(q,inp))
            with np.load(arr,allow_pickle=False) as x:
                self.assertTrue(all(x[k].dtype != object for k in x.files))

    def test_09_desconhecido_nao_inventa(self):
        r=eng.prever_dataframe(self.p,pd.DataFrame([('novo','a00',0.)],columns=['estado','memoria','idade_seg']))
        self.assertFalse(r.iloc[0].tem_previsao)
        self.assertEqual(r.iloc[0].status_modelo,'ESTADO_NAO_CATALOGADO')

    def test_10_memoria_incompativel_bloqueia(self):
        with self.assertRaises(ValueError):
            eng.prever_dataframe(self.p,pd.DataFrame([('a00','a03',0.)],columns=['estado','memoria','idade_seg']))

    def test_11_idade_invalida_bloqueia(self):
        for a in [-1.,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):
                eng.prever_dataframe(self.p,pd.DataFrame([('login','a00',a)],columns=['estado','memoria','idade_seg']))

    def test_12_nao_fecha_top5_em_um(self):
        r = eng.prever_dataframe(self.p,pd.DataFrame([('login','a00',0.)],columns=['estado','memoria','idade_seg']))
        self.assertLess(sum(v['score'] for v in json.loads(r.iloc[0].util_top5_json)),1.)

    def test_13_gradiente_com_censura(self):
        fun, x, *_ = fit.preparar_objetivo(self.pdf,'peso_evento_treino',self.structure,self.cfg)
        val, analytic = fun(x)
        numeric = approx_fprime(x,lambda z:fun(z)[0],1e-6)
        np.testing.assert_allclose(analytic,numeric,rtol=1e-3,atol=2e-5)

    def test_14_ajuste_real(self):
        self.assertTrue(fit.validar_modelo(self.parent))
        self.assertFalse(self.parent['usa_pai'])

    def test_15_contexto_e_b2(self):
        child=fit.ajustar_modelo(self.pdf.iloc[:75], 'peso_evento_treino', self.structure,self.cfg,self.parent)
        b2=fit.c02_compor_modelo(child,self.parent)[0]
        self.assertTrue(child['usa_pai'])
        self.assertEqual(b2['p_destino'],child['p_destino'])
        self.assertEqual(b2['mu_grupo'],self.parent['mu_grupo'])

    def test_16_suporte_sem_alteracao_silenciosa(self):
        with self.assertRaises(ValueError):
            fit.criar_estrutura(self.pdf.iloc[:2],self.cfg)

    def test_17_predict_vazio(self):
        r=eng.prever_dataframe(self.p,pd.DataFrame(columns=['estado','memoria','idade_seg']))
        self.assertEqual(list(r.columns),eng.OUTPUT_COLUMNS)
        self.assertEqual(len(r),0)

    def test_18_pandas_indice_nao_contiguo(self):
        inp=pd.DataFrame([('login','a00',0.),('a00','a00',600.)],columns=['estado','memoria','idade_seg'],index=[22,51])
        self.assertEqual(len(eng.prever_dataframe(self.p,inp)),2)


class ArtifactTests(unittest.TestCase):
    def test_19_sintaxe_e_embutidos(self):
        for p in HERE.glob('*.py'):
            tree=ast.parse(p.read_text())
            if p.name.startswith('01_'):
                assigns={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
                for name,file in [('ENGINE_SOURCE','nba_engine.py'),('DATA_SOURCE','nba_data.py'),('FIT_SOURCE','nba_fit.py'),('MODEL_SOURCE','modelo_pyfunc.py')]:
                    self.assertEqual(ast.literal_eval(assigns[name]), '\n'+(HERE/file).read_text()+'\n')

    def test_20_notebooks_validos(self):
        import nbformat
        for p in HERE.glob('*.ipynb'):
            nb=nbformat.read(p,as_version=4); nbformat.validate(nb)
            self.assertTrue(all(not c.get('outputs') for c in nb.cells if c.cell_type=='code'))

    def test_21_sem_congelados_ou_ids_antigos_obrigatorios(self):
        for p in HERE.glob('0*.py'):
            s=p.read_text()
            self.assertNotIn('id_ms03_referencia',s)
            self.assertNotIn('PRESERVAR_REFERENCIA',s)
            self.assertNotIn('ac9ba1a5-e908-4d8b-958e-ce5536a6fa72',s)
            self.assertNotIn('.mode("overwrite")',s)
            self.assertNotIn(".mode('overwrite')",s)

    def test_22_selecao_explicita_modelo(self):
        s=(HERE/'02_nba_v23_predicao_mlflow.py').read_text()
        self.assertIn("if not CFG['model_uri'] and not CFG['run_id_treinamento']:",s)
        self.assertIn("r.info.status != 'FINISHED'",s)
        self.assertIn("env_manager='local'",s)
        self.assertIn('CENSURADO', (HERE/'01_nba_v23_treinamento_mlflow.py').read_text())

    def test_23_wrapper_com_artefatos_simulados(self):
        # Stub apenas para exercitar nosso load_context/predict. NÃO testa APIs MLflow.
        old=sys.modules.get('mlflow')
        stub=types.ModuleType('mlflow'); stub.pyfunc=types.SimpleNamespace(PythonModel=object)
        stub.models=types.SimpleNamespace(set_model=lambda m:None)
        sys.modules['mlflow']=stub
        try:
            scope={}; exec(compile((HERE/'modelo_pyfunc.py').read_text(),'modelo_pyfunc.py','exec'),scope)
            model=scope['NBA23Model']()
            _,p,*_=sintetico17()
            with tempfile.TemporaryDirectory() as td:
                d=Path(td); eng.salvar_payload(p,d/'meta.json',d/'coef.npz')
                (d/'bundle.json').write_text(json.dumps({'contrato':'nba_v23_b2_mlflow_v1','alvos':p['alvos'],'max_passos_macro':5}))
                artifacts={'bundle':str(d/'bundle.json'),'metadata':str(d/'meta.json'),'coeficientes':str(d/'coef.npz')}
                hashes={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in artifacts.items()}
                (d/'sha.json').write_text(json.dumps(hashes)); artifacts['integridade']=str(d/'sha.json')
                model.load_context(types.SimpleNamespace(artifacts=artifacts))
                inp=pd.DataFrame([('login','a00',50.)],columns=['estado','memoria','idade_seg'])
                pd.testing.assert_frame_equal(model.predict(None,inp),eng.prever_dataframe(p,inp))
                (d/'bundle.json').write_text('{}')
                with self.assertRaises(RuntimeError): model.load_context(types.SimpleNamespace(artifacts=artifacts))
        finally:
            if old is None:sys.modules.pop('mlflow',None)
            else:sys.modules['mlflow']=old

    def test_25_wrapper_treino_serializavel(self):
        source = (HERE/'01_nba_v23_treinamento_mlflow.py').read_text()
        tree = ast.parse(source)
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'criar_ajustador')
        ns = {'pd': pd, 'json': json}
        exec(compile(ast.Module(body=[func], type_ignores=[]), 'fit_wrapper', 'exec'), ns)
        pdf,cfg = treino_sintetico()
        fitter = ns['criar_ajustador']((HERE/'nba_fit.py').read_text(),cfg)
        cloned = cloudpickle.loads(cloudpickle.dumps(fitter))
        out=cloned(pdf)
        self.assertEqual(out.iloc[0].status_modelo,'AJUSTADO')
        self.assertEqual(out.iloc[0].variante,'A_REFERENCIA')

    def test_26_guardas_producao(self):
        train = (HERE/'01_nba_v23_treinamento_mlflow.py').read_text()
        score = (HERE/'02_nba_v23_predicao_mlflow.py').read_text()
        self.assertIn('Requer Spark >= 3.5',train)
        self.assertIn('try_element_at(util_top5, 1)',train)
        self.assertIn('try_element_at(raw_top5, ranking)',score)
        self.assertLess(score.index('PRED_00:'),score.index("pub = spark.table"))
        self.assertIn(".option('versionAsOf', int(versao))",score)
        self.assertNotIn('versao_delta_inicio !=',score)

    def test_27_config_preserva_alvos_parametros(self):
        source=(HERE/'01_nba_v23_treinamento_mlflow.py').read_text()
        nodes=ast.parse(source).body
        assigns={n.targets[0].id:n.value for n in nodes if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
        cfg=ast.literal_eval(assigns['CFG']); num=ast.literal_eval(assigns['NUM'])
        self.assertEqual(cfg['debounce_seg'],30.0)
        self.assertEqual(cfg['max_passos_macro'],5)
        self.assertEqual(cfg['modulo_amostra_clientes'],1000)
        self.assertEqual(num['lambda_prob_contexto'],.02)
        self.assertEqual(num['lambda_tempo_contexto'],.02)
        self.assertEqual(len(ast.literal_eval(assigns['ALVOS'].args[0])),17)

    def test_24_nucleo_AST_igual_ao_anterior(self):
        paths={
            '/mnt/data/nba_v23/nba_v23_02_treino_validacao_pareada.py':('nba_fit.py',['criar_estrutura','preparar_objetivo','ajustar_modelo']),
            '/mnt/data/nba_v23_estavel/nba_v23_retomada_estavel.py':('nba_engine.py',['MotorMacroMemoria','ms_prever_raw']),
            '/mnt/data/nba_v23_operacional/nba_v23_operacional_b2.py':('nba_data.py',['op_preparar_acoes']),
        }
        for origin,(dest,names) in paths.items():
            if not Path(origin).exists():
                self.skipTest('Fontes anteriores não estão neste diretório. Comparação já registrada na entrega.')
            def funcs(p):
                return {n.name:n for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
            a,b=funcs(origin),funcs(HERE/dest)
            for name in names:self.assertEqual(ast.dump(a[name]),ast.dump(b[name]),name)

if __name__=='__main__':
    unittest.main(verbosity=2)
