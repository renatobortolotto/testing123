# NBA V2.3 — notebooks MLflow corrigidos

## Correção desta entrega

Esta revisão substitui **os dois notebooks MLflow** entregues anteriormente,
não os experimentos históricos nem a V2.4. Arquivos:

- `01_nba_v23_treinamento_mlflow.ipynb` ou `.py`;
- `02_nba_v23_predicao_mlflow.ipynb` ou `.py`.

Importe **um formato de cada notebook**. Os `.py` são exportações Databricks com
marcadores de célula; não são módulos auxiliares que devam ser criados/importados
pelo notebook. O `.ipynb` já apresenta as células separadas.

**Não há `ENGINE_SOURCE`, `FIT_SOURCE`, `DATA_SOURCE` ou `MODEL_SOURCE`. Não há
`exec`, `eval` ou `compile` de Python nos notebooks. Não há geração de módulos
`nba_engine.py`, `nba_fit.py`, `nba_data.py` ou `modelo_pyfunc.py`.**

## O que permanece igual

Mantidos em relação ao pacote `NBA_V23_Treino_Predicao_MLflow.zip` anterior:

- Fonte de eventos, período padrão, amostragem por cliente e split por cliente.
- Resolução de simultaneidade, debounce de 30 segundos, memória e censura.
- Verossimilhança, agrupamentos temporais, parâmetros, limites e otimizador.
- Ajuste A/B e composição B2: probabilidades contextuais de B + curvas de A.
- Os 17 alvos e o horizonte de cinco mudanças de macroação.
- Fórmulas das métricas, ações escolhidas para análise e regras dos relatórios.
- Saída técnica e LONG, Top 5 RAW e útil independentes, sem renormalização.
- Leitura Delta fixada por versão e gravação append por execução.
- Nenhuma nova feature, regra de elegibilidade, promoção ou campanha.

`verificacao_preservacao.json` documenta a comparação estrutural de 36 funções
ou classes. Nas oito funções de preparação Spark, somente imports foram
movidos para o corpo das funções e as anotações `DataFrame` foram colocadas em
aspas. As transformações Spark não foram alteradas. Isso evita carregar PySpark
apenas para desserializar a interface tabular; Spark continua necessário para
preparar a base e classificar em lotes no Databricks.

Essa equivalência é contra o **código MLflow entregue anteriormente**. Não é uma
certificação de equivalência com todos os resultados das primeiras experiências
V2.3, nem comprova execução nas tabelas do banco. Fonte, amostra, corte ou ambiente
diferentes ainda podem alterar os resultados de um novo treinamento.

## Organização do notebook 01

1. Configuração: período, amostra, ações de avaliação e experimento MLflow.
2. Funções de preparação: leitura, catálogo, sequências, memória e rótulos.
3. Funções de treinamento/inferência: A, B, B2, Multi-step e cálculo em lotes.
4. Funções de avaliação geral e por ação.
5. Classe `NBA23Model`, escrita normalmente na célula.
6. Abertura da run e verificações de ambiente/serialização.
7. Criação real da base a partir de `base_jornadas_onboarding`.
8. Treinamento de A/B e composição B2.
9. Avaliação geral e das ações escolhidas.
10. Registro no MLflow, recarga, teste de reprodução e conclusão.

As seções 2 a 5 são definições de funções/classes; só a partir da seção 6 começa
a execução da run. **Execute na ordem.** Não copie a configuração do pacote de
preservação antigo: não existem modos `PRESERVAR_REFERENCIA` ou
`REAJUSTAR_CONGELADO` neste treinamento a partir dos eventos.

## Como ficou o registro no MLflow

A seção final registra diretamente uma instância nova:

```python
info_modelo = mlflow.pyfunc.log_model(
    artifact_path="modelo",
    python_model=NBA23Model(),
    artifacts=artifacts,
    signature=signature,
    input_example=example,
    pip_requirements=requirements,
)
```

Não utiliza `code_paths` nem `mlflow.models.set_model`.

**O MLflow usa CloudPickle internamente para armazenar a classe e as funções
Python necessárias.** Isso produz artefatos de serialização do próprio MLflow;
não significa guardar a sessão inteira nem gerar arquivos Python pelo notebook.
A instância é registrada antes de carregar dados: não guarda SparkSession,
DataFrames, conexão ou broadcast como atributos. `load_context` recupera os
parâmetros de JSON e os coeficientes de NPZ, validando os hashes.

Os arquivos numéricos `bundle.json`, `kernel_meta.json`, `coeficientes.npz` e
`integridade.json`, bem como relatórios CSV/JSON, continuam sendo artefatos
normais. Eles são necessários para registrar e recuperar o modelo. O notebook
não precisa nem tenta eliminar toda gravação de arquivos.

A versão de CloudPickle é registrada junto das dependências. Driver/executores
e ambientes de carregamento devem ser compatíveis com o ambiente de treino.
**Carregue somente URIs/modelos de origem confiável**: serialização de objetos
Python não é um formato seguro para conteúdo de terceiros não confiáveis.

## Uso recomendado

### 01 — Treinamento

Edite `CFG`, `NUM`, `ALVOS` e `TECNICAS` nas células próprias somente quando
quiser mudar explicitamente a experiência. Para homologar esta refatoração,
conserve os mesmos dados/corte/amostra/parâmetros da execução comparada.

Os padrões foram mantidos, incluindo:

```python
"inicio_historico": "2026-06-01T00:00:00+00:00",
"corte_historico_exclusivo": "2026-09-25T00:00:00+00:00",
"modulo_amostra_clientes": 1000,
"debounce_seg": 30.0,
"experimento_mlflow": "/Shared/NBA/V23_B2",
```

Confira as datas. Não viram “hoje” automaticamente. Módulo 1000 é uma amostra
por hash de aproximadamente 0,1% dos clientes; módulo 1 inclui todos. Não altere
a amostra durante a comparação de equivalência do código.

A run recebe `versao_codigo = v23_b2_eventos_mlflow_1.0.1_notebook` e
`formato_empacotamento = pythonmodel_cloudpickle_notebook_v1`.

Ao concluir, copie `model_uri` de `TREINO_11_CONCLUSAO`.

### 02 — Predição

Informe somente `model_uri` OU `run_id_treinamento`. Sem escolha, o notebook
lista as runs e interrompe; não seleciona a mais recente por conta própria.

A preparação de estado/memória é obtida da classe carregada (`model_api`), que
contém as funções definidas no notebook 01. Não é necessário repetir o treino,
gerar módulos ou executar suas células em outra sessão para usar o modelo.

Mantenha `modo_scoring = "AMOSTRA"` na homologação. Após revisar os resultados,
use `TODOS`. `customers_query` precisa estar acessível nessa sessão; em produção,
configure fonte permanente ou etapa de ingestão explícita. A correção de código
não inventa uma fonte de público nem resolve permissões corporativas de upload.

As saídas continuam com prefixo `ctg_dsti.renato_nba.nba_v23_mlflow_scoring`.
Views: `..._amostra_atual` e `..._negocio_atual`. Não há promoção para as views
antigas da V2.3. As tabelas permanecem append por execução.

## Substituição das versões anteriores

Use estes arquivos como **substituição completa dos dois notebooks MLflow**,
não como células de patch colocadas sobre as definições antigas. Mantenha os
notebooks anteriores arquivados para referência.

Para o primeiro teste, execute o notebook 01 desde o início para criar uma run
com o empacotamento corrigido; depois aponte o notebook 02 para a nova URI. Essa
execução treina de verdade sobre os eventos configurados. Não sobrescreve os
modelos de runs anteriores nem troca automaticamente uma versão em produção.

## Verificações desta entrega

31 testes locais passaram. Incluem:

- Sintaxe e equivalência entre exportações Python e Jupyter.
- Ausência de execução de strings e de módulos auxiliares nos notebooks.
- Preservação estrutural de funções, classes e configurações do modelo.
- Ajuste numérico A/B, gradiente com censura e composição B2 em dados sintéticos.
- Equivalência de inferência, memória, reentrada, massa e Top 5.
- Serialização real com CloudPickle do worker de ajuste e da classe.
- Carregamento dessas serializações em **outro processo Python isolado**,
  sem os módulos locais anteriores e sem variáveis da sessão de treino.
- Recuperação dos artefatos JSON/NPZ e detecção de alteração de artefatos.

**Limites dos testes:** Spark, Delta, PyArrow e MLflow não estão instalados no
ambiente local desta entrega. Não foram executadas suas APIs reais, permissões,
juízos de desempenho, joins ou persistência nas suas tabelas. A classe-base do
MLflow foi simulada somente para exercitar nossa interface `load_context` e
`predict`; esse teste NÃO prova o funcionamento de `mlflow.pyfunc.log_model` ou
`load_model` no workspace. Os notebooks mantêm autotestes Spark, teste de reload
pelo MLflow real e teste nos executores para serem executados no seu cluster.

A comparação de AST não executa operações Spark. O worker contextual foi
validado localmente com um objeto simples de pais, não com um broadcast Spark
real. **Homologue antes de substituir o job atual.**

## Referências de implementação

- MLflow, API `pyfunc`: suporte a instância de `PythonModel`, CloudPickle,
  dependências e avisos de segurança.
  https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html
- MLflow, PythonModel Guide: definição de classe, `load_context` e `predict`.
  https://mlflow.org/docs/latest/ml/model/python_model/

As fontes foram consultadas para o mecanismo de registro. Não foram usadas
para substituir as fórmulas ou os critérios estatísticos da implementação.
