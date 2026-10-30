# NBA V2.3 — treinamento ponta a ponta e predição com MLflow

## Esta entrega substitui qual parte?

Substitui a consolidação anterior que partia de uma base congelada. O notebook 01
**lê os eventos, cria o catálogo/base, faz a amostragem por cliente, monta as
transições e censuras, calcula a memória, ajusta A/B e compõe B2**. Não contém
`PRESERVAR_REFERENCIA`, não exige IDs antigos e não depende de notebook aberto.
O notebook 02 usa exclusivamente a URI de um modelo concluído no MLflow e as
fontes operacionais.

Ainda é uma entrega para homologação no ambiente do usuário. Não foi executada
contra as tabelas reais nem contra o servidor MLflow/Delta do banco neste ambiente.

## Conteúdo

- `01_nba_v23_treinamento_mlflow.py` / `.ipynb`: notebook de treino completo.
- `02_nba_v23_predicao_mlflow.py` / `.ipynb`: seleção explícita de modelo e scoring.
- `nba_engine.py`, `nba_fit.py`, `nba_data.py`, `modelo_pyfunc.py`: módulos legíveis
  para revisão, auditoria e futura manutenção em Git. **Já estão embutidos no
  notebook 01; não é preciso importá-los manualmente no Databricks.**
- `testes_locais.py`, `resultado_testes.txt`, `origem_do_nucleo.json`, `SHA256SUMS.txt`.

Importe um formato de cada notebook, não ambos. Não acrescente os patches antigos.
Nenhuma chamada `%run`, instalação de bibliotecas, arquivo de outra sessão ou
artefato histórico aprovado é necessário para iniciar o treinamento.

## Ambiente e permissões

Spark >= 3.5, Python >= 3.10, MLflow >= 2.12.2, NumPy, Pandas, SciPy e PyArrow.
Use um runtime ML ou ambiente configurado com essas bibliotecas no driver E
nos executores. O código não faz `%pip install` durante o job.
O acesso de compute deve permitir SparkContext/broadcast e applyInPandas.
As bibliotecas devem corresponder ao ambiente de treino quando o scoring usa
`env_manager="local"`; requisitos exatos são gravados no MLflow. A predição executa
um smoke test distribuído pequeno ANTES de ler/processar o público real.

É necessário acesso de leitura à fonte Delta, permissão de criar/escrever tabelas
no schema escolhido e permissão no experimento do MLflow. Crie a pasta de workspace
`/Shared/NBA` ou ajuste a configuração para uma pasta existente/autorizada antes do
primeiro uso. Não são instaladas integrações, criadas credenciais ou concedidas
permissões pelo código.

## 01 — o que editar e executar

No bloco `CFG`:

| Campo | Default / finalidade |
|---|---|
| `fonte_eventos` | `ctg_dsti.renato_nba.base_jornadas_onboarding` |
| `inicio_historico` | `2026-06-01T00:00:00+00:00` |
| `corte_historico_exclusivo` | `2026-09-25T00:00:00+00:00` |
| `versao_eventos` | None: fixa o head Delta disponível ao iniciar a leitura |
| `modulo_amostra_clientes` | 1000: aproximadamente 0,1% dos clientes; 1: todos |
| `sal_amostra` / `sal_validacao` | Preservam os valores históricos conhecidos |
| `modulo_validacao_clientes` | 5: clientes de um bucket (~20%) no holdout |
| `acoes_avaliacao` | Boleto, recepção de dados OPF e cadastro de chave PIX |
| `experimento_mlflow` | `/Shared/NBA/V23_B2` |
| `registered_model_name` | None; registro opcional, sem alterar aliases |

As datas históricas são EXPLÍCITAS, não são atualizadas automaticamente para hoje.
Altere-as somente quando desejar treinar outra janela. O nome da run indica
arquitetura, início da execução UTC, corte e módulo da amostra. Quantidade real
de clientes, versão da fonte, parâmetros e ambiente ficam registrados.

A amostra seleciona IDs com hash e mantém as sequências completas dentro da janela.
Não utiliza `customers_query` no treino. A lista de 17 alvos não filtra os eventos.
`acoes_avaliacao` escolhe relatórios detalhados; também não filtra os exemplos.

Passos do notebook:
1. Validar ambiente, iniciar a run e executar autotestes.
2. Fixar a fonte Delta, construir catálogo e base, salvar por run.
3. Ajustar A por origem, B por contexto com suporte de treino; compor B2 com p_B e S_A.
4. Rotular holdout, calcular RAW/útil e métricas; avaliar as ações escolhidas.
5. Salvar código + JSON + NPZ no MLflow, carregar de volta e conferir previsões.
6. Concluir a run, imprimir `run_id` e `model_uri`. Nada é promovido automaticamente.

Saídas usam `ctg_dsti.renato_nba.nba_v23_mlflow_treino_*_hml`, append por
`run_id_treinamento`. Uma falha marca a run como FAILED (nas etapas envolvidas), e
artefatos parciais não devem ser consumidos. Corrija e reinicie o notebook para
uma run nova; não tente reaproveitar IDs de células intermediárias. Uma sessão
interrompida pode deixar uma run RUNNING; finalize-a explicitamente antes de
iniciar outra. O notebook não encerra uma run ativa que pertença a outro trabalho.

### Preparação preservada

Estado por timestamp: maior profundidade; empate entre estados de mesma profundidade
é ambiguidade, não ordenação inventada. Debounce consolida repetições consecutivas
do mesmo estado em até 30 segundos e usa o último timestamp do burst. A janela
é [início, corte). O último estado e a saída antes de ambiguidade são censurados
à direita. Origem ambígua não é elegível para ajuste. `dur_min` é em SEGUNDOS.
A memória é o último funil não técnico observado, preservada em login/etapas
técnicas e atendimento; barreiras a reiniciam. Não existem estados sintéticos
`sem_acao`/`conversao`. A entrada deve conter os eventos clean.

O catálogo é derivado dos estados do período: `familia:::topo/navegacao/sucesso`
ou `atendimento:::canal`, mesma taxonomia clean usada anteriormente. Estado fora
desse contrato gera erro, não uma inferência silenciosa de semântica. Para um
novo produto com outra nomenclatura, adaptar e revisar o catálogo explicitamente.

### O que está preservado e o que pode mudar

Foram copiados sem alteração da AST os métodos originais de verossimilhança,
gradientes, otimizador, preparação operacional e propagação. Os testes locais
comparam essas funções com as fontes disponíveis da V2.3. Parâmetros numéricos,
17 alvos, cinco mudanças e regra de memória seguem os valores anteriores.
C/D não são treinados: eram diagnósticos, não componentes do B2 escolhido.

NÃO há garantia de scores idênticos se a fonte, versão física, período ou amostra
forem diferentes. Treinar sobre fonte atual não é preservar bytes do modelo antigo.
Há uma comparação OPCIONAL com `tabela_base_referencia` e `id_base_referencia`
para verificar base reconstruída sob o mesmo recorte. Sem configuração, essa
verificação é registrada como não executada; não bloqueia a criação de candidata.
Para colocar a candidata em produção, compare-a com a versão aprovada e autorize
explicitamente seu consumo. O código não executa promoção automática.

### Escala e amostra

`applyInPandas` carrega cada grupo inteiro. O limite de 150 mil observações por
origem interrompe antes do ajuste; não descarta/trunca dados. Ao aumentar a amostra,
verifique esse limite, memória por executor, duração do ajuste e tamanho do grafo.
Não basta aumentar o limite sem dimensionar o compute. Uma curva de aprendizagem
(mais clientes mantendo período/split/parâmetros) é uma experiência distinta.

### Avaliação

Holdout por cliente, dentro da mesma janela, conforme a V2.3: NÃO é teste temporal
final independente. Os rótulos de ação útil usam primeira chegada em até cinco
mudanças de macro. Continuação do mesmo macro não é novo alvo; reentrada depois
de sair é permitida. Censurados não viram negativos.

RAW: métricas em saídas exatas. Ranking útil (Top1/Top5/MRR): alvos observados.
Brier das 17 ações, Brier de algum alvo e log loss conservador: todos os desfechos
conhecidos. A massa sem modelo não é renormalizada nem atribuída ao desfecho sem
alvo. O log loss conserva piso 1e-15 apenas na métrica. Não certifica calibração.
Principais médias primeiro por cliente; precisão por ação também é mostrada por
recomendação, explicitamente nomeada. Bootstrap de diferenças B2-A reamostra
clientes inteiros, é exploratório e sem ajuste por multiplicidade.
Lift Top10/Top20 usa um snapshot por cliente/idade, escolhido sem olhar o futuro,
antes de filtrar censurados. Empates na fronteira recebem seleção proporcional;
lift é descritivo, sem IC. Poucos positivos são sinalizados, não excluídos.

O tempo de evento é respeitado. Sem timestamp de disponibilidade/ingestão não é
possível certificar o histórico as-of de chegada da informação. Validar em período
posterior e revisar viés por censura antes de declarar ganho em produção.

## Registro no MLflow

Cada treinamento é uma run do mesmo experimento. O modelo customizado usa
`mlflow.pyfunc.log_model`, código legível e artefatos JSON/NPZ sem pickle do motor.
São incluídos catálogo, preparo, parâmetros A/B/B2, suporte e coeficientes.
O pacote não inclui a base com `cd_bv`; essa base fica em Delta com acesso governado.
O código também registra relatórios agregados, ambiente, configuração e tabela fonte.
MLflow ainda executa código ao carregar o modelo; carregue somente URIs confiáveis.

Registro no Model Registry é opcional, controlado por `registered_model_name`.
Não cria aliases de produção, não faz rollout e não autoriza disparos.

## 02 — seleção e predição

1. Importe o notebook 02 novo.
2. Preencha `model_uri` com o valor impresso pelo notebook 01, OU use
   `run_id_treinamento`. Não preencha ambos. Sem seleção, ele lista runs concluídas
   do experimento e interrompe solicitando a escolha; não pega a última sozinho.
3. Disponibilize a fonte de público e eventos; mantenha inicialmente AMOSTRA.
4. Execute tudo. Um teste de UDF nos executores ocorre antes da preparação pesada.
5. Valide e então use TODOS para classificar todo o público elegível.

A URI é resolvida/fixada antes do scoring. A preparação Spark de estado/memória
é carregada do próprio pacote (`unwrap_python_model()`), não de uma variável de
treino nem de um arquivo externo. Os executores usam `mlflow.pyfunc.spark_udf` com
schema explícito, vetorização e lotes; não coletam o público completo no driver.

`customers_query` é só o público. Em execução independente, ela precisa ser uma
fonte permanente ou ser criada pela ingestão do job, não por uma sessão alheia.
Se houver coluna de data, escolhe o snapshot máximo. Sem data, mantém NULL.
Os eventos Delta são fixados por versão no início: novo commit durante a execução
não é erro. Troca de identidade/localização da tabela é bloqueada. A fonte precisa
reter os arquivos desse snapshot durante o job; não há bypass de retenção.
O corte padrão é o máximo evento observado não futuro; não prova completude.
Prefira watermark explícito. Há tolerâncias configuráveis de atraso, sem usar
D-1 arbitrariamente. A idade e o estado usam o mesmo corte.

### Saídas

Prefixo novo: `ctg_dsti.renato_nba.nba_v23_mlflow_scoring`.
Tabelas físicas: `_publico_hml`, `_estados_hml`, `_predicoes_hml`, `_negocio_hml`,
`_execucoes_hml`. Append por execução, sem sobrescrever as tabelas anteriores.
A saída técnica inclui os scores dos 17 alvos, inclusive itens fora do Top5.

Views de avaliação:
- `ctg_dsti.renato_nba.nba_v23_mlflow_scoring_amostra_atual`
- `ctg_dsti.renato_nba.nba_v23_mlflow_scoring_negocio_atual`
- `ctg_dsti.renato_nba.nba_v23_mlflow_scoring_ids_atual`

As views selecionam somente uma execução concluída do modo e run de treinamento
utilizados na publicação: maior corte e conclusão como desempate. AMOSTRA não
atualiza a view TODOS. Não existe transação implícita entre todas as tabelas;
o manifesto final é o sinal de conclusão.

Colunas de negócio: `data_referencia`, `cd_bv`, `acao_atual`, `ranking`,
`proxima_acao_raw`, `score_raw`, `proxima_acao_util`, `score_util`.
Até cinco posições por cliente, RAW e útil em rankings independentes. Não há
normalização dos cinco scores para somarem 1. Não prevê prazo em dias ou conclusão
da operação. Elegibilidade individual e acionamento não são decididos neste código.

## O que foi efetivamente testado

`resultado_testes.txt` registra testes NUMÉRICOS/LOCAIS: ajuste com dados sintéticos,
gradiente/censura, composição B2, conservação de massa, inferência em lotes,
reentrada, 17 alvos, JSON+NPZ, serialização do wrapper de treinamento e contratos.
O carregador customizado foi exercitado com contexto simulado, **não com o servidor
MLflow**. As funções originais foram comparadas por AST.

Não foi possível instalar PySpark/MLflow real neste ambiente sem acesso à rede.
Portanto, testes Spark/Arrow/Delta, permissões e `log_model/spark_udf` reais são
executados no Databricks, não foram declarados aprovados aqui. Os notebooks possuem
smoke tests no cluster para detectar os problemas antes da carga pesada.

## Referências oficiais consultadas

- MLflow Models from Code: https://mlflow.org/docs/latest/ml/model/models-from-code/
- Pyfunc, load_context, unwrap_python_model e spark_udf:
  https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html
- Log/carregamento de modelos no Databricks:
  https://docs.databricks.com/aws/en/mlflow/models
- Memória de applyInPandas:
  https://spark.apache.org/docs/3.5.6/api/python/reference/pyspark.sql/api/pyspark.sql.GroupedData.applyInPandas.html
