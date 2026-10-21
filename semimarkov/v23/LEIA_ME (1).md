# NBA V2.3 — estudos dos 17 alvos úteis

## Executar

Importe **um** dos arquivos `nba_v23_estudo_17_acoes.py` ou `.ipynb` em um
notebook novo de análise no Databricks e execute as células na ordem.
Não acrescente células ao scoring operacional. Não reexecute treinamento,
Multi-step ou scoring apenas para realizar este estudo.

Configuração inicial:

```python
"view_negocio": "ctg_dsti.renato_nba.nba_sm_v23_operacional_negocio_atual",
"id_execucao": None,
"analisar_historico": True,
```

Com `id_execucao=None`, a análise lê a definição da view para identificar o modo
e o modelo e reproduz a seleção do manifesto concluído: maior corte, depois
conclusão mais recente. Não seleciona uma execução pela ordem de UUIDs nem pela
data do público. Um ID explícito pode ser informado para fixar outra execução
concluída; nesse caso a view não é usada para selecioná-la.

A lista de 17 alvos fornecida nesta conversa é conferida contra o manifesto.
O estudo interrompe se a execução tiver outra composição, mesmo que também tenha
17 itens. A lista está em `ALVOS_ESPERADOS`. Ela é uma conferência, não modifica o
modelo, os candidatos ou a tabela final.

O código lê as tabelas persistidas por IDs. Não usa variáveis do notebook anterior,
não procura o último modelo de forma independente, não solicita versões Delta
antigas e não lê a telemetria bruta. As partições de execuções concluídas devem
continuar imutáveis, conforme o contrato append do pipeline. Os hashes verificados
são de configuração/metadados, não uma prova de identidade de todas as linhas.

A análise usa Spark para as bases e percentis aproximados (`percentile_approx`).
Somente metadados pequenos e testes sintéticos são coletados no driver.
`persistir_em_disco=True` usa cache local `DISK_ONLY` para reduzir releituras; ele
é liberado ao final. Não há gravação ou alteração de tabelas/views de negócio.

## Relatórios principais

| Relatório | Interpretação |
|---|---|
| `EST17_01_COBERTURA` | Clientes com previsão e motivos sem previsão, no mesmo ID. |
| `EST17_02_COBERTURA_ALVOS` | Todos os 17 alvos, incluindo os ausentes, com contagens Top 1/Top 5. |
| `EST17_02_ALVOS_AUSENTES` | Quais alvos não apareceram no Top 5 nesta execução. |
| `EST17_03_RANKS` / `EST17_03_SCORES` | Posições e distribuição dos scores EXIBIDOS, não dos scores completos. |
| `EST17_04_CONCENTRACAO` / `EST17_04_LISTAS` | Repetição das listas ordenadas, sem confundir lista com scores idênticos. |
| `EST17_04_MARGENS_E_MASSA` | Separação Top 1–Top 2 e massa dos 17 alvos não mostrada no Top 5. |
| `EST17_05_CONTEXTO_TOP1` | Destinos por estado, memória e faixa de idade. Associação descritiva. |
| `EST17_06_SUPORTE_TREINO` | Presença na base preparada de treino e entradas observadas entre macros. |
| `EST17_07_RANKING_COMPLETO_HISTORICO` | Probabilidades/ranks de todos os 17 alvos nos kernels históricos. |
| `EST17_08_AUSENCIAS_E_EVIDENCIAS` | Junta ausência atual, suporte de treino e presença no histórico. |
| `EST17_10_DESEMPENHO_POR_ACAO` | Recuperação/precisão por ação no histórico B2 em idade zero. |

Os DataFrames completos ficam no dicionário `RELATORIOS`. Por exemplo:

```python
display(RELATORIOS["EST17_08_EVIDENCIAS_COMPLETAS"])

display(
    RELATORIOS["EST17_10_DESEMPENHO_HISTORICO_COMPLETO"]
    .filter("variante = 'B2_MEMORIA_TEMPO_COMPARTILHADO' AND idade_seg = 86400")
    .orderBy("acao")
)
```

As impressões usam até 40 linhas e arredondam apenas para exibição.
Os valores completos continuam nos DataFrames.

## Como interpretar os cinco alvos ausentes

Não aparecer no Top 5 **não implica score zero nem inexistência no treinamento**.
Há dois níveis separados:

1. **Operacional:** a tabela final e `util_top5` guardam somente cinco itens por
   cliente. Não há informação suficiente para reconstruir os scores/ranks atuais
   6 a 17 só com essas saídas.
2. **Histórico vinculado:** `util_probs` e `util_ranks` preservam os 17 alvos, por
   estado + memória + idade. O estudo usa esse material já salvo. Não executa o
   motor novamente, não arredonda idades atuais para encaixá-las nesses kernels
   e não transfere conclusões automaticamente do histórico para o público atual.

Um alvo ausente agora, mas positivo fora do Top 5 histórico, tem evidência de que
é alcançável em algumas condições já avaliadas. Isso não prova por que ele ficou
fora dos cinco primeiros de cada cliente atual. Zero nos kernels avaliados também
não prova impossibilidade em todos os estados/tempos possíveis.

O relatório de suporte conta observações preparadas, não operações financeiras.
`n_entradas_macro_exatas_treino` conta saídas exatas que entram no funil vindo de
outro macro. Não exige um modelo de saída do alvo para concluir que ele pode ser
alcançado: atingir um alvo e continuar a partir dele são problemas distintos.
Não estabelece mínimos de suporte ou políticas de aprovação por conta própria.

## Denominadores e métricas

- `fracao_clientes_top5`: clientes que receberam a ação / clientes com previsão.
  As frações das 17 ações podem somar mais de 100%, pois cada pessoa recebe até 5.
- Contagem ausente recebe **0**, mas estatísticas de score ausente ficam **NULL**.
  Não são imputados scores zero para os itens que não foram publicados.
- `recuperacao_top1/top5`: frequência de recuperar a ação real nos casos em que
  aquela ação foi o primeiro alvo observado. A média por cliente primeiro agrega
  dentro de cada pessoa e depois entre as pessoas.
- `precisao_top1/top5`: frequência de a ação ser o alvo real entre as suas
  recomendações com desfecho conhecido; inclui os horizontes completos sem alvo.
  Para Top 5, cada recomendação da ação é uma unidade; não é o hit rate global.
- Censurados não viram negativos. A cobertura desses casos é mostrada separadamente.
- Sem exemplos reais de uma ação, a recuperação é **NULL**, não 0%.
- Não há intervalos de confiança novos, calibração ou teste de efeito de campanha.
  O holdout já foi utilizado no desenvolvimento e possui snapshots repetidos.
- Margem entre primeiro e segundo score não é nível de confiança calibrado.
- O score atual é de primeira chegada entre os alvos definidos em até cinco
  mudanças de macroação; não é chance de sucesso nem probabilidade em sete dias.

## V2.4: fronteira de responsabilidade

Este notebook não implementa elegibilidade.

A proposta restrita para uma futura V2.4 é **integrar critérios oficiais de
cliente-ação**, somente quando forem fornecidos/aprovados pelos responsáveis.
O modelo prevê comportamento. O negócio e os canais decidem o uso, mensagem,
canal, campanha, prioridades e tratamento de ações como cancelamento.

Se a elegibilidade já for aplicada em CRM/outra camada, não é necessário duplicar
a lógica no modelo. A integração pode consumir as decisões/flags dessa camada.
Não se presume elegibilidade por um evento antigo, nem se bloqueia uma ação apenas
por conter `cancelar`, `bloquear` ou `excluir` no nome.

Quando houver regras aprovadas, o filtro deve considerar o conjunto de candidatos
completo para não perder uma alternativa elegível que estava além do Top 5.
Filtrar/reordenar não autoriza renormalizar scores nem apagar eventos da cadeia.
A primeira chegada entre TODOS os alvos e a primeira chegada entre SOMENTE os
alvos elegíveis são definições diferentes; eventual mudança de alvo precisa ser
explícita. Nada disso é modificado por este estudo.

## Testes

- Sintaxe Python verificada.
- 18 testes locais de contratos/metadados e verificações estáticas passaram:
  validação da lista, extração da definição da view, recusa de seleção ambígua,
  hashes determinísticos e ausência de operações de escrita/treinamento.
- Os testes locais não executam as transformações Spark. PySpark não estava
  disponível neste ambiente; a execução real permanece a validar no Databricks.
- O notebook inclui autotestes Spark pequenos, executados antes da leitura das
  bases, cobrindo alvos ausentes, denominadores e exclusão de censura.

Documentação primária consultada:
- Apache Spark: `pyspark.sql.functions.percentile_approx`.
  https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.percentile_approx.html
- Apache Spark: `pyspark.sql.functions.count_distinct`.
  https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.count_distinct.html
