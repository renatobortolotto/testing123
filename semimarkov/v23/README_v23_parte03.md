# NBA V2.3 — Parte 03: Multi-step contextual A × B2

## Execução

Importe **um** dos arquivos `nba_v23_03_multistep_contextual_historico.py` ou
`.ipynb` como um notebook separado no Databricks. Execute as células na ordem.
Os arquivos são autocontidos e equivalentes; não dependem de `%run`, módulos
locais, variáveis ou broadcasts das partes anteriores.

Os três IDs já estão preenchidos:

```python
"id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
"id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
"id_02c": "53d097ef-c176-485b-9848-4fa06b653064",
```

Exige um manifesto `CONCLUIDO_COMPARACAO_02C` e o ajuste original
`CONCLUIDO_TREINO_VALIDACAO`. Valida hashes, versões e contratos antes de
propagar. O fuso deve ser exatamente o do preparo. **Não altere o fuso nem
converta os timestamps para contornar uma divergência.**

**Não retreina, não recalibra, não seleciona parâmetros e não atualiza a tabela
de negócio. Não usa `customers_query`: esta é uma comparação histórica, não o
scoring do público elegível atual.**

## Escopo da comparação

São utilizados todos os snapshots A/B2 da 02C, nos marcos de idade já transcorrida
`0`, `1800`, `86400` e `604800` segundos. A seleção de clientes e eventos é a
persistida; não há amostragem nova. Entretanto, o label agora é a primeira ação
acionável futura, e não apenas o próximo estado RAW.

As rotas de inferência continuam as mesmas: A por origem; B2 por contexto nas
origens técnicas que possuem composição; reuso de A para origens não técnicas,
contextos desconhecidos, raros ou sem ajuste. Os dois motores usam o mesmo catálogo
completo e a mesma lista de 15 candidatas. Falta de modelo não gera fallback
estático nem probabilidade inventada.

Os modelos A e o histórico são lidos nas versões físicas associadas ao ajuste/02C.
O catálogo de estados é a versão utilizada na Parte 01. Os contextos compostos
B2 são lidos pelo ID da 02C e confrontados com hashes dos componentes A e B.
Não usa automaticamente o último modelo disponível.

## Memória e unidade dos passos

O estado de propagação é um par:

```text
(estado RAW, último funil não técnico observado)
```

Login, as demais etapas técnicas e os canais de atendimento preservam a memória.
Uma nova entrada em funil não técnico atualiza a memória. Consultas podem ser
memória, ainda que não pertençam à lista de ações acionáveis. Atendimento preserva
memória, mas seu próprio modelo continua por origem, como na Parte 01.

Cada ramo da trajetória possui sua memória. Dois caminhos que chegam ao mesmo
login por históricos diferentes não são fundidos numa distribuição populacional.

```text
boleto -> login:::topo -> login:::navegacao -> login:::sucesso -> PIX
```

Nesta sequência, existem duas mudanças de macro: boleto → login e login → PIX.
As transições internas de login permanecem no cálculo, com custo zero no contador.
O objetivo é a **primeira entrada em uma ação da whitelist, após mudar de macro,
em até cinco mudanças de macroação**. Qualquer etapa da ação pode ser a entrada;
não se exige `sucesso` e não se estima conclusão de uma transação.

A continuação do funil atual não é um novo alvo. Uma reentrada **depois de sair**
é permitida. Exemplo: `PIX -> login -> PIX` pode prever PIX na segunda mudança.
O funil inicial não é proibido durante todo o horizonte.

Blocos consecutivos do mesmo funil não têm timeout novo. Isso NÃO identifica uma
operação ou sessão real. Duas operações consecutivas do mesmo funil sem mudança
intermediária não são distinguidas por esse alvo. Foi preservada a unidade de
análise; não se criou uma regra de sessão para obter listas mais diversas.

## Componente temporal

A primeira transição usa a idade observada no snapshot:

```text
q_B2(j | i, contexto, a) ∝ p_B(j | i, contexto) * S_A,ij(a)
```

Nas chegadas futuras, a idade do novo estado é zero. A continuação usa sua
distribuição de entrada, com a memória daquela trajetória. **Não se reaplica a
idade da origem a todos os estados futuros.**

O horizonte é de mudanças de macroação, não minutos ou dias. Este notebook não
calcula probabilidade de agir em sete dias nem tempo esperado até a ação.
`passo_macro_maior_contribuicao` é o índice da mudança com maior contribuição ao
score, não uma média/esperança de passos.

## Como as etapas internas são resolvidas

O grafo separa transições internas ao mesmo macro (`Q`) das mudanças de macro (`R`).
A recorrência de primeira chegada soma as transições internas por resolução de
sistemas `(I - Q) X = RHS`, no conjunto que pode alcançar uma saída.

Componentes internas que não podem alcançar mudança de macro, terminal ou estado
sem modelo são classificadas como **retenção interna**. Não se usa pseudoinversa
nem se transforma esse resíduo em falta de modelo. O núcleo aceita terminais para
os autotestes, mas esta versão de execução exige o contrato clean sem terminais
reais utilizado na Parte 01.

A solução preserva, em cada horizonte:

```text
massa em ações-alvo
+ massa sem modelo
+ massa terminal
+ massa retida internamente
+ massa que consumiu cinco mudanças sem atingir alvo
= 1
```

Uma ação-alvo pode ser alcançada mesmo quando não existe modelo para sair dela:
primeira chegada não exige prever o que vem depois do alvo. Da mesma forma, se o
limite de cinco mudanças já foi consumido, não se exige um modelo para uma sexta
mudança. As verificações de massa e resíduo interrompem a execução em divergência;
não renormalizam silenciosamente.

## Construção do label histórico

O futuro é usado apenas no label. Estado, idade e memória de entrada vêm do prefixo
já persistido. A reprodução RAW contra a 02C verifica se o scoring preservou esses
inputs. O notebook não passa `acao_real` ou qualquer outro campo de futuro para o
motor.

Uma ligação de eventos só é seguida se a saída é exata e corresponde ao próximo
passo/timestamp/destino válidos. Lacunas, ambiguidades e censuras encerram a sequência
observável; não são puladas para encontrar um alvo conveniente.

Três status possíveis:

| Status | Significado |
|---|---|
| `ALVO_OBSERVADO` | Primeira ação da lista encontrada em até cinco mudanças; o restante do futuro não é necessário. |
| `HORIZONTE_OBSERVADO_SEM_ALVO` | Cinco mudanças foram realmente observadas, sem ação da lista. |
| `CENSURADO_ANTES_HORIZONTE` | Não apareceu alvo e ainda não observamos cinco mudanças completas. Não recebe label negativo. |

As métricas usam os mesmos labels nos dois motores. Snapshots censurados permanecem
na cobertura e no output, mas não recebem métricas supervisionadas de ação útil.
A censura pode ser informativa: os casos com label conhecido não devem ser tratados
como uma amostra automaticamente representativa de todos os snapshots operacionais.
Esta etapa não implementa IPCW nem certifica calibração populacional.

## Métricas e denominadores

### Ranking da ação útil

`top1`, `top5` e `mrr` são avaliados **somente onde um alvo foi observado**, usando
os mesmos casos modelados em A e B2. Um alvo com score zero não conta como acerto
por desempate de ações de massa zero. MRR usa o ranking da lista completa de 15
candidatas. O Top 5 continua sem renormalização.

### Qualidade probabilística nos desfechos conhecidos

- `brier_acoes`: soma dos 15 erros quadráticos entre os scores de primeira chegada
  e os 15 indicadores binários de qual alvo veio primeiro. Em um horizonte
  completamente observado sem alvo, todos os indicadores são zero. Não normaliza
  os scores para somar 1 e não é o Brier de destinos RAW da Parte 02.
- `brier_algum_alvo`: erro quadrático entre a soma dos scores e o indicador de
  ter atingido algum dos 15 alvos no horizonte.
- `logloss_conservador`: menos o log da probabilidade modelada do desfecho real,
  com o mesmo piso da 02C. Para alvo observado, usa seu score; para horizonte
  observado sem alvo, usa a soma de retenção interna, terminal e limite de passos.
  **Não distribui a massa sem modelo ao rótulo "sem alvo".** Havendo massa não
  resolvida, esta é uma avaliação conservadora da probabilidade modelada, não
  uma certificação de um classificador multinomial calibrado.
- `logloss_limite_otimista`: diagnóstico que aloca toda a massa sem modelo ao
  rótulo real. Não altera o score publicado. Ajuda a verificar quanto o resultado
  poderia depender da incerteza de cobertura.

O log loss conservador penaliza ausência de suporte sem inventar uma ação.
O piso limita somente a métrica. Falta de modelo inicial fica sem métricas e
continua na cobertura; massa perdida em estados posteriores é preservada.

A comparação principal calcula médias por cliente nos mesmos snapshots e usa
bootstrap pareado de clientes (2.000 réplicas). Ganho positivo é melhora, tanto
para acertos quanto para perdas. Intervalos de 95% são exploratórios, sem ajuste
por múltiplas comparações. Esse holdout já orientou decisões do desenvolvimento;
não é teste temporal final independente.

Existem duas coortes de avaliação: todos os desfechos conhecidos e somente os
alvos observados. Os prints escolhem a coorte relevante para cada métrica; os
artefatos guardam todas as comparações. Casos mudam entre marcos de idade.

## Relatórios principais

| Relatório | Uso |
|---|---|
| `V23_MS03_01_LABELS` | Observados versus censurados em cada marco. |
| `V23_MS03_02_GRAFOS` | Tamanho da expansão, componentes retidas e erro máximo de massa. |
| `V23_MS03_03_INTEGRIDADE` | Contratos, associação de snapshots e reprodução RAW. |
| `V23_MS03_04_COBERTURA` | Status por variante, idade e label. |
| `V23_MS03_05_METRICAS` | Métricas por evento em origens técnicas; versão completa também contém TODAS. |
| `V23_MS03_06_PARES_B2_VS_A` | Comparação principal com peso igual por cliente. |
| `V23_MS03_07_TOP1_UTIL` | Ações predominantes por marco/variante. |
| `V23_MS03_07_CONCENTRACAO` | Número de listas e frequência da lista dominante. |
| `V23_MS03_08_MASSAS` | Alvo, falta de modelo, retenção e limite de passos separados. |
| `V23_MS03_09_SAIDA_LONG` | Contagens do resultado histórico. |
| `V23_MS03_10_CONCLUSAO` | ID e tabelas da execução concluída. |

A concentração é calculada por snapshot, não por cliente único, e não é uma
métrica de acerto. A assinatura considera ações e posições, não os scores.
Os resumos completos incluem desempenho por alvo e todos os segmentos, mesmo
quando a tela mostra somente as primeiras linhas. Os prints ocultam IDs de clientes.

## Saídas e consumo

Prefixo novo: `ctg_dsti.renato_nba.nba_sm_v23_03`.

| Sufixo | Conteúdo |
|---|---|
| `_snapshots_hml` | Inputs históricos e labels, sem seleção por resultado do modelo. |
| `_kernels_hml` | Distribuições por variante/estado/memória/idade, para rastreabilidade. |
| `_predicoes_hml` | Duas linhas por snapshot, com todas as probabilidades úteis, massas e métricas. |
| `_comparacoes_hml` | Bootstrap pareado. |
| `_negocio_historico_hml` | Até cinco linhas rankeadas por cliente/snapshot/variante. |
| `_resumos_hml` | Relatórios completos em JSON. |
| `_execucoes_hml` | Manifesto com fontes, política, IDs e status de conclusão. |

Todos usam append com `id_ms03` e IDs fontes. Não sobrescrevem tabelas anteriores.
Reexecutar uma célula de escrita com o mesmo ID é bloqueado. Uma nova execução do
notebook gera outro ID. Snapshots/kernels são materializados antes das fases
seguintes para reduzir recomputação; falhas podem deixar saídas parciais.

**Consuma somente ID com `CONCLUIDO_MULTISTEP_HISTORICO_03`.** As escritas não são
uma transação conjunta. O manifesto conserva `publicavel=False`, `houve_retreino=False`.
`gravar=False` mantém relatórios em memória, sem manifesto persistido.

O output histórico não deve disparar campanha. Ele contém `variante`, `id_snapshot`,
`ts_snapshot` e o aviso `EXPERIMENTO_HISTORICO_NAO_DISPARAR`. Sua
`data_referencia` é a data do snapshot, não a data de hoje. RAW e ação útil de mesmo
ranking são colunas independentes, não uma cadeia causal entre essas duas ações.
Se um ranking não existir de um lado, seu campo fica nulo. Sem previsão, uma linha
com ranking nulo preserva o snapshot. Não há preenchimento artificial de cinco ações.

## Dimensionamento e testes

Eventos completos permanecem em Spark. Não utiliza UDF Python por evento/cliente,
`mapInPandas`, Arrow ou `toPandas` sobre os históricos. Catálogos de modelos, chaves
únicas dos kernels e médias por cliente vão ao driver com limites explícitos.
As proteções param a execução; não amostram nem truncam silenciosamente.

A expansão pode ser maior que a matriz RAW, mas usa matrizes esparsas. Não aloca
uma matriz densa de tamanho (estados × memórias) ao quadrado para todas as saídas.
Não promete latência NRT: esta etapa compara modelos históricos.

Testes locais efetivamente executados:

- Compilação do Python, de cada célula Jupyter e validação estrutural nbformat.
- Memória através de etapas técnicas/atendimento, atualização de memória,
  reentrada no alvo inicial, estados sem modelo e conservação de massa.
- Ciclos internos fechados e loops geométricos com saída.
- 180 comparações do cálculo analítico com enumeração de trajetórias RAW em
  cadeias sintéticas: erro máximo aproximadamente `9e-15`.
- Reprodução exata da função de inferência da 02C em 50 modelos sintéticos e
  seis idades (300 comparações); diferença máxima observada zero.
- Sinais e pareamento do bootstrap.

**Spark/Delta não foram executados neste ambiente.** Os autotestes Spark de labels,
censura, reentrada e acesso seguro por rank executam no cluster antes de consultar
as fontes reais. Erros de permissões, schema, versão ou integridade interrompem.

## Próxima decisão

Avaliar qualidade e cobertura da primeira ação útil antes de olhar apenas a
variedade do Top 5. Se o resultado for adequado, a etapa seguinte será scoring
atual da população elegível completa, reconstruindo memória somente com eventos
já disponíveis no corte operacional. Nenhuma promoção ou disparo está implementado.

## Referências de implementação

- Contratos internos: partes 01, 02 e 02C da V2.3 e seus manifestos persistidos.
- SciPy: `scipy.sparse.linalg.splu`, solução esparsa por LU:
  https://docs.scipy.org/doc/scipy-1.16.2/reference/generated/scipy.sparse.linalg.splu.html
- Apache Spark: funções SQL e janelas:
  https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html
- Apache Spark: `element_at` (acesso a mapa por chave, sem pressupor tamanho de array):
  https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.element_at.html
