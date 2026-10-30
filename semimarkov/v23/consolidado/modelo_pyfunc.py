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
