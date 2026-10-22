# NBA V2.3 — referência por frequência × A × B2

## Entrega e execução

Importe **apenas um** destes arquivos em um notebook novo do Databricks:

- `nba_v23_frequencia_vs_a_b2.py`: notebook no formato source do Databricks.
- `nba_v23_frequencia_vs_a_b2.ipynb`: mesmo conteúdo em Jupyter.

Conecte ao cluster com acesso às tabelas persistidas e execute as células em
ordem. Não anexe ao pipeline operacional. Não execute treinamento, Multi-step
ou scoring antes deste estudo: ele recupera os resultados já gravados.

O bloco `CFG` já contém a execução operacional analisada nas imagens:

```python
"id_execucao": "52a577b9-4d54-41dd-9d24-7a871b675318"
```

O manifesto resolve automaticamente `id_ms03_validado`, o pacote físico, as
tabelas históricas, as idades e o piso de log loss. Não utiliza variáveis de
outras sessões. Para estudar outra execução concluída com a mesma lista de
17 alvos, altere somente esse ID. Não existe seleção automática de “último”
modelo ou substituição de uma entrada ausente por dados atuais.

O notebook é **somente leitura das tabelas/views**. Cria DataFrames, caches de
disco transitórios e relatórios na sessão; alinha o fuso da sessão ao pacote.
Não cria tabelas, não muda retenção, não modifica as views de negócio e não
calcula scores novos de A/B2. Estima apenas uma distribuição empírica de
referência, a partir dos rótulos de treino. Usa PySpark, NumPy e Pandas; não
contém `%pip`, `%run` nem importações de arquivos auxiliares.

## Pergunta avaliada

O alvo é **a primeira entrada em uma das 17 ações úteis em até cinco mudanças
de macroação**, conforme a definição histórica da V2.3. Continuação dentro do
mesmo funil não é nova entrada; reentrada depois de outro funil é permitida.

Não é uma previsão de sucesso/conversão, um prazo de cinco/sete dias, nem a
pergunta “essa ação apareceu em qualquer momento futuro?”. Se outro alvo
apareceu primeiro, o desfecho observado pertence a ele, mesmo que a ação de
interesse tenha ocorrido depois.

A distribuição das recomendações não é um objetivo de otimização. Boleto
pode permanecer em primeiro para a maioria dos clientes e ainda haver ganho
na qualidade ou priorização dos scores.

## As três alternativas

| Variante | Definição |
|---|---|
| `F_FREQUENCIA_TREINO` | Frequência empírica do primeiro desfecho no treino, sem estado ou memória, fixa por idade. |
| `A_REFERENCIA` | Scores históricos do Semi-Markov A, com estado/tempo, sem a memória contextual de B2. |
| `B2_MEMORIA_TEMPO_COMPARTILHADO` | Scores históricos de B2, com memória contextual e curvas temporais compartilhadas. |

F × B2 mede valor além das frequências gerais. A × B2 mede o ganho adicional
da memória neste experimento. F × A ajuda a decompor o ganho de estado/tempo.
Os três métodos são comparados nos mesmos snapshots suportados por A/B2.
Casos fora dessa interseção aparecem na cobertura, não entram como acertos
nem como erros zero.

## Como F é estimada, sem usar o holdout

1. Recupera a base completa preservada no pacote, contendo os dois splits.
2. Verifica chaves, política e separação dos clientes entre treino e holdout.
3. Reconstrói rótulos de primeiro alvo nas trajetórias, mantendo barreiras.
   A função de rotulagem é a do notebook estável, sem mudar sua lógica.
4. Confere que os rótulos reconstruídos do holdout reproduzem os persistidos.
   Essa conferência não estima probabilidades com dados de validação.
5. Usa **somente `validacao_cliente = false`** para construir os snapshots de
   treino. Em cada idade, usa origens elegíveis com duração maior que a idade
   e instante anterior ao corte histórico — o mesmo contrato de observação.
6. Exclui censurados da estimação e dá peso total **um por cliente/idade**
   entre seus snapshots com desfecho conhecido.
7. Calcula a distribuição dos 17 alvos e do desfecho auxiliar sem alvo.

Para uma pessoa com `n` snapshots conhecidos numa idade, cada um pesa `1/n`.
Assim, uma pessoa muito ativa não domina a referência apenas por volume.
Essa ponderação está alinhada à avaliação principal por cliente. O relatório
por snapshot é complementar e utiliza outro peso.

`__SEM_ALVO_NO_HORIZONTE__` é apenas um desfecho da avaliação: cinco mudanças
observadas sem um dos 17 alvos. **Não reinserimos o estado `sem_acao` na cadeia**
nem afirmamos que a pessoa ficou inativa.

Não há suavização ou balanceamento de classes. Uma ação nunca observada no
treino recebe probabilidade zero em F. O piso de log loss, recuperado do
manifesto histórico, é aplicado igualmente a todos os métodos.

## Censura e limites da avaliação

Desfechos conhecidos são:

- `ALVO_OBSERVADO`;
- `HORIZONTE_OBSERVADO_SEM_ALVO`.

`CENSURADO_ANTES_HORIZONTE` não é convertido em negativo. Essa é uma análise
de casos com resultado conhecido; ela não corrige censura informativa. As
frequências e curvas devem ser interpretadas no universo avaliável, não como
certificação da probabilidade de toda a população operacional.

As idades são analisadas separadamente. Os mesmos clientes podem aparecer em
mais de uma idade. O holdout já foi consultado no desenvolvimento, portanto
os resultados são diagnósticos/exploratórios. Uma confirmação independente
requer período posterior não usado nas decisões do modelo.

## Métricas e intervalos

### Brier das ações

Soma dos 17 erros quadráticos entre score e indicador do primeiro alvo.
Inclui horizontes conhecidos sem alvo (todos os 17 indicadores são zero).
Não normaliza os scores dos alvos para somarem um.

### Brier de algum alvo

Erro quadrático da soma dos 17 scores contra a presença de qualquer alvo
observado no horizonte. Não significa “qualquer evento no banco”.

### Log loss conservador

Usa a probabilidade do desfecho real e o piso numérico do histórico. Para
“sem alvo”, usa somente `prob_sem_alvo_modelada`. **A massa sem modelo não é
transferida para “sem alvo”**, e nenhuma probabilidade é renormalizada. Essa
métrica é secundária: preserva a penalização conservadora da implementação
existente e pode refletir massa não resolvida pelo motor.

### Top 1, Top 5 e MRR

Calculados apenas nos snapshots com uma ação-alvo observada. Não são
avaliados nos horizontes sem alvo nem nos censurados. Zero de score para o
alvo real não conta como acerto mesmo que um desempate entre zeros lhe desse
posição numérica baixa. Os ranks de A/B2 vêm dos kernels persistidos; F usa
probabilidade decrescente e nome de ação para desempatar, como o motor.

### Ponderação e comparações pareadas

O desempenho é calculado primeiro por cliente e depois agregado, dando peso
igual a cada pessoa dentro de segmento/idade. Para ranking, participam os
clientes com alvo observado; para perdas, todos os clientes com desfechos
conhecidos. Os denominadores aparecem nos relatórios.

O bootstrap reamostra **clientes inteiros**, usando a mesma reamostragem nas
duas alternativas comparadas. F e os modelos ficam fixos; o intervalo não
inclui a incerteza da estimação do treino. Há 2.000 réplicas em lotes por
padrão. Nenhum evento bruto é coletado para esse bootstrap, apenas médias
por cliente e grupo.

Ganho positivo sempre indica melhora:

- Perdas: erro da referência menos erro do candidato.
- Ranking: acerto do candidato menos acerto da referência.

Para perdas, `skill_relativo = 1 - media_candidato / media_referencia`.
O intervalo de skill usa a razão das médias reamostradas, não a média de
razões por cliente. Quando o denominador é zero, skill é indefinido (`NULL`).
Com um só cliente, não se apresenta um IC artificialmente preciso.

Os IC95 são exploratórios e sem correção por múltiplas comparações. Um ganho
positivo numa ação com pouquíssimos clientes positivos não certifica
utilidade nem generalização.

## Lift por ação e empates

Por padrão, seleciona um snapshot por cliente/idade usando hash determinístico
de chaves e semente. A seleção é **anterior ao filtro de censura**, não
favorece casos positivos nem utiliza score, ação real ou instante do evento
futuro. Depois, retém os desfechos conhecidos para os três métodos.

Esse recorte evita que a mesma pessoa apareça dezenas de vezes numa faixa.
É um recorte histórico, não uma simulação de todos os clientes num mesmo
corte operacional diário. A cobertura desse recorte é mostrada explicitamente.

A alternativa `coorte_lift = "TODOS_PESO_CLIENTE"` usa todos os snapshots
avaliáveis, com peso total um por cliente/idade/segmento. Nessa alternativa,
as faixas representam massa ponderada de snapshots, não um conjunto de
pessoas distintas. As perdas/ranking principais não mudam com essa opção.

Para cada ação, compara a taxa observada nos 10% e 20% de maiores scores com
a taxa do universo avaliável da mesma idade/segmento:

```text
lift = taxa observada da ação na faixa / taxa observada da ação na coorte
captura = positivos da faixa / positivos de toda a coorte
```

Se há empate no score na fronteira, seleciona uma fração igual de todo o
bloco empatado. A positividade é calculada depois de determinar essa fração,
sem olhar o alvo para escolher os registros. É a expectativa de desempate
aleatório, não uma seleção concreta para campanha.

Assim, F tem lift exatamente 1 onde há positivos. Empates não são resolvidos
por UUID/CPF, ordem física ou resultado futuro. `positivos_faixa_esperados`
pode ser fracionário por essa razão. Sem positivos, lift e captura ficam
`NULL`, nunca zero ou infinito.

**Lift e curvas são descritivos neste notebook; não têm IC.** O campo
`suporte_leitura` avisa quando há menos de 30 clientes positivos, por padrão.
É um aviso configurável de amostra, não regra de elegibilidade, filtro ou
limiar comercial. Todas as ações permanecem na análise.

## Relatórios principais

Todos os DataFrames completos ficam em `RELATORIOS`. `.show()` arredonda
somente a impressão; os dados guardados no dicionário não são arredondados.

| Relatório | Conteúdo |
|---|---|
| `FREQ_01_COBERTURA_TREINO` | Desfechos conhecidos/censurados por idade no treino. |
| `FREQ_02_REFERENCIA_F` | Probabilidades empíricas, suporte e denominadores da referência. |
| `FREQ_03_COBERTURA_COMPARACAO` | Cobertura dos modelos e labels antes das métricas. |
| `FREQ_04_METRICAS_CLIENTE` | Métricas principais F/A/B2, com peso igual por cliente. |
| `FREQ_04_METRICAS_SNAPSHOT` | Complemento ponderado por observação. |
| `FREQ_05_COMPARACAO_PAREADA` | Diferenças e IC no segmento geral. |
| `FREQ_05_COMPARACOES_COMPLETAS` | Diferenças em todas as idades e segmentos. |
| `FREQ_06_BRIER_POR_ACAO` | Brier e taxas por ação em idade zero. |
| `FREQ_06_BRIER_COMPLETO` | Todas as idades e segmentos. |
| `FREQ_06_PARES_BRIER_ACAO` | B2 × F por ação/idade zero, com IC e suporte. |
| `FREQ_06_PARES_ACAO_COMPLETOS` | Todas as três comparações por ação/idade/segmento. |
| `FREQ_07_COBERTURA_LIFT` | Universo escolhido antes da exclusão de censuras. |
| `FREQ_08_LIFT_TOP10_TOP20` | Lift de B2 por ação em idade zero. |
| `FREQ_08_LIFT_COMPLETO` | F/A/B2, todas as idades e segmentos. |
| `FREQ_09_CURVA_BOLETO` | Faixas de score de B2 para boleto em idade zero. |
| `FREQ_09_CURVAS_COMPLETAS` | Curvas das 17 ações e três variantes. |
| `FREQ_10_PAINEL_ACOES` | Brier/skill e lift juntos, com ambos os suportes explícitos. |

Exemplo para consultar outra idade sem mudar o estudo:

```python
from pyspark.sql import functions as F

RELATORIOS["FREQ_08_LIFT_COMPLETO"].filter(
    (F.col("variante") == "B2_MEMORIA_TEMPO_COMPARTILHADO")
    & (F.col("idade_seg") == 86400.0)
    & (F.col("segmento") == "TODAS")
).orderBy("acao", "faixa").show(40, truncate=False)
```

O notebook não grava os relatórios em tabelas. Ao final libera os caches
criados por ele; os DataFrames continuam disponíveis e podem recomputar
agregações quando consultados novamente. Não repete o motor A/B2.

## Verificações e testes

Foram executados **28 testes locais**, descritos em `testes_locais.py`, cobrindo
sintaxe, contrato de 17 alvos, regra de labels preservada por comparação AST,
natureza somente leitura, ausência de holdout na função de estimação de F,
Brier com 17 indicadores, massa não normalizada, sinal do bootstrap, ausência
de denominador, determinismo e oráculos NumPy para empates/lift/faixas.

As expressões Spark **não foram executadas neste ambiente local**. O pacote
PySpark não estava disponível. O notebook contém autotestes Spark pequenos
para o próprio cluster: reentrada, censura, barreiras, peso por cliente e
lift neutro sob empates. Também reproduz os rótulos e as métricas históricas
persistidas de A/B2 antes de aceitar as comparações.

Não se afirma validação end-to-end no Databricks ou nas tabelas reais.
`resultado_testes.txt` registra o resultado local.

## Referências técnicas

Documentação primária consultada para o desenho do estudo:

```text
scikit-learn — DummyClassifier (baseline por prior)
https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

scikit-learn — Probability calibration (Brier não mede apenas calibração)
https://scikit-learn.org/stable/modules/calibration.html

scikit-learn — Brier score loss
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html

Apache Spark — element_at (leitura dos mapas de scores/ranks)
https://spark.apache.org/docs/3.5.6/api/python/reference/pyspark.sql/api/pyspark.sql.functions.element_at.html
```

## O que não será feito

Não promove automaticamente B2, não muda o motor, não força diversidade,
não cria regras comerciais, não exclui cancelamento/bloqueio, não ajusta
calibração e não produz scores para acionamento. Elegibilidade continua
aguardando os responsáveis pelo negócio. Ganho preditivo não é ganho causal.
