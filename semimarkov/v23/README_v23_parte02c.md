# NBA V2.3 — Parte 02C: memória com tempo compartilhado

## Execução

Importe **um** dos arquivos `nba_v23_02c_memoria_tempo_compartilhado.py`
ou `.ipynb` como um notebook separado no Databricks. Execute as células na ordem.
As duas versões têm o mesmo conteúdo executável. Não é necessário manter objetos
em memória nem reexecutar as partes 01, 02 ou 02B.

IDs já preenchidos:

```python
"id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
"id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
```

O notebook exige exatamente um manifesto `CONCLUIDO_TREINO_VALIDACAO` da Parte 02
para esses IDs. Usa os nomes de tabelas indicados nesse manifesto e fixa as versões
Delta lidas nesta execução. Seleciona somente os modelos e as linhas de validação
do ajuste informado; não utiliza automaticamente o modelo mais recente.

**Não treina, não modifica parâmetros de A/B, não seleciona hiperparâmetros,
não executa Multi-step nem atualiza a tabela de negócio.** A gravação padrão está
habilitada apenas para novas tabelas da 02C. `gravar=False` executa os relatórios
sem escrita, inclusive sem manifesto persistido.

## Hipótese do experimento

A memória B melhorou a previsão na entrada dos estados, mas a auditoria identificou
probabilidades muito pequenas para algumas saídas tardias. O objetivo é isolar
a influência das curvas temporais específicas de contexto.

| Variante | Probabilidade na entrada | Curva de tempo |
|---|---|---|
| A_REFERENCIA | A, por origem | A, por grupo temporal da origem |
| B_MEMORIA | B, por contexto; A no reuso | B no contexto; A no reuso |
| B2_MEMORIA_TEMPO_COMPARTILHADO | **Exatamente a probabilidade utilizada por B** | **A da mesma origem** |

A fórmula de B2 é:

```text
q_B2(j | i, c, idade=a)
    = p_B(j | i, c) × S_A,ij(a)
      / soma_k [p_B(k | i, c) × S_A,ik(a)]
```

Cada destino conserva a sobrevivência de seu grupo em A. **Não é uma curva única
para todos os destinos.** A massa dos grupos na mistura continua vindo de B; não se
substitui o denominador simplesmente pela sobrevivência populacional de A.

Em idade zero, todas as sobrevivências são 1. B2 deve reproduzir B nas probabilidades,
nos rankings e nas métricas. Onde B já recorria a A, B2 permanece igual a A em todos
os marcos. Essas identidades são verificadas, não apenas presumidas.

### O que B2 não é

B2 é uma **recombinação dos parâmetros persistidos**, não uma nova solução de máxima
verossimilhança com o tempo fixo. As probabilidades de B foram estimadas originalmente
junto de suas próprias curvas; não são reestimadas aqui. Não atribuímos a B2 a função
objetivo ou o número de iterações do ajuste anterior.

Compartilhar tempo com A também não garante melhoria: A tem limitações próprias,
inclusive parâmetros em limites. A decisão depende dos relatórios, sem promoção
automática. Eventual melhoria nesta amostra não certifica calibração nem desempenho
futuro. Esse holdout já foi examinado e agora é de desenvolvimento.

## Segurança do alinhamento e reprodução

A função de composição alinha os grupos **pelo conjunto de destinos** de cada grupo,
não pela posição dos arrays. Suporta reordenação de destinos e renumeração dos grupos.
Vocabulários diferentes, grupos incompatíveis, parâmetros não finitos e massas
inválidas interrompem a execução. Não inventa destinos nem remove grupos.

A/B são reavaliados com a fórmula da Parte 02, em log-space. Antes da comparação,
o código confere sua reprodução contra os valores persistidos de:

- probabilidade do destino real, Brier e log loss;
- rankings e acertos Top 1/Top 5, distribuição sem tempo e variação temporal;
- suporte ao destino, clipping, alerta de limite e extrapolação.

Usa todos os destinos na normalização; nunca renormaliza apenas o Top 5.
A função `log_ndtr` evita calcular `log(1 - cdf)` por subtração na cauda.
A sobrevivência é também confrontada com `scipy.stats.lognorm.logsf`.

A rotulação e o roteamento A/B são os que já estavam persistidos. Uma falha
numérica de inferência não é silenciosamente transformada em nova previsão.
O vocabulário, os pesos de treino e o público não são alterados.

## Casos e métricas

Os marcos são os da Parte 02: `0`, `1800`, `86400` e `604800` segundos.
**São idades já transcorridas, não horizontes futuros.** O mesmo conjunto de casos
é usado dentro de cada marco, mas os casos mudam entre marcos: só chegaram aos marcos
maiores as observações que ainda não tinham terminado até aquele instante.

A 02C não reconstrói labels ou memória a partir das fontes atuais. Reutiliza
cada `cd_bv + passo + idade_seg` da validação original e confere A/B completos
sem duplicatas. Não amostra eventos ou clientes para avaliar.

Censuras continuam na tabela técnica com previsão quando existe modelo; não
recebem destino negativo e não entram nas métricas de classificação. O objetivo
desta parte continua sendo o próximo estado RAW entre saídas exatas observadas.
Não é uma avaliação da distribuição completa de tempos ou da chance de agir em N dias.

Sem modelo: o caso continua na cobertura, com métricas de qualidade nulas.
Destino fora do vocabulário: probabilidade zero, acerto zero, Brier penalizado
na distribuição completa e log loss no mesmo piso da Parte 02. Esses casos
não são removidos para melhorar o resultado.

O piso do log loss é lido do manifesto original (`1e-15` nesse ajuste).
Ele só limita a métrica reportada; não altera o score de B2. O log da probabilidade
sem piso é mantido para destinos suportados, inclusive quando exponenciar resulta
em zero numérico. `prob_baixa_triagem=1e-6` só conta casos no relatório.

### Comparação pareada

Para cada segmento/marco, calcula a média das métricas de cada cliente nos mesmos
casos. Depois calcula as médias entre clientes e os ganhos B−A, B2−A e B2−B.
O bootstrap reamostra clientes, usando os mesmos índices para as comparações
pareadas. São 2.000 réplicas por padrão, com semente fixa.

Nos relatórios de pares, **ganho positivo significa melhora**:

```text
Brier/log loss: ganho = referência − candidata
Top 1/Top 5:    ganho = candidata − referência
```

Os intervalos são percentis de 95%, descritivos, sem correção por múltiplas
comparações e sem interpretação como teste final independente.

Já nos casos críticos, `delta_b2_a` ou `delta_b2_b` mede a diferença de log loss:
**delta positivo é piora**. Os casos críticos foram escolhidos pela piora original
B−A, não pelo resultado favorável de B2. O relatório principal mantém todos os casos.

## Relatórios

| Nome | Uso |
|---|---|
| V23_02C_01_INTEGRIDADE | Checagens de contrato, reprodução A/B e identidade de B2 |
| V23_02C_02_ROTAS | Casos e clientes por marco e rota original |
| V23_02C_03_METRICAS | Top 1/5, Brier, log loss e clipping por evento, em origens técnicas |
| V23_02C_04_B_VS_A | Comparação por cliente para reproduzir o resultado anterior |
| V23_02C_04_B2_VS_A | Candidata contra a referência |
| V23_02C_04_B2_VS_B | Efeito isolado de trocar as curvas de tempo |
| V23_02C_05_EXTREMOS | Contagens de probabilidade muito baixa, clipping e suporte |
| V23_02C_06_CASOS_CRITICOS | O que aconteceu com os casos mais difíceis da auditoria |
| V23_02C_07_CONCLUSAO | IDs e tabelas efetivamente gravadas |

Os prints são arredondados para legibilidade. Probabilidades dos casos críticos
usam formato compacto/científico, evitando exibir `1e-100` como `0.000000`.
**A persistência conserva a precisão original.** Os resumos completos também incluem
`TODAS`; a exibição principal privilegia `ORIGENS_TECNICAS` para caber na tela.

A tabela técnica distingue a fonte das probabilidades da fonte temporal.
Por exemplo, em B2 um contexto pode estar além do maior tempo da fonte B e ainda
estar dentro da fonte A. Isso não demonstra que sua cauda ficou confiável;
apenas esclarece qual distribuição foi utilizada. Não se exige ausência de alertas
para incluir casos, nem se escolhe a melhor variante por cliente usando o destino real.

## Saídas e retomada

```text
ctg_dsti.renato_nba.nba_sm_v23_02c_modelos_hml
ctg_dsti.renato_nba.nba_sm_v23_02c_validacao_hml
ctg_dsti.renato_nba.nba_sm_v23_02c_comparacoes_hml
ctg_dsti.renato_nba.nba_sm_v23_02c_resumos_hml
ctg_dsti.renato_nba.nba_sm_v23_02c_execucoes_hml
```

A tabela de modelos contém apenas os contextos B2 compostos, com fontes e hashes;
quando não há contexto próprio, continua necessário reutilizar A. Seu status é
`COMPOSTO_SEM_RETREINO`. Ela não pode ser consumida pelo código de scoring antigo
como se fosse um novo ajuste convencional.

A validação contém três linhas por observação/marco (A, B, B2) e informação de
proveniência. Os resumos são gravados como `relatorio + dados_json`. Casos críticos
com IDs ficam somente no artefato interno; os prints não exibem `cd_bv`.

As gravações usam append e um novo `id_02c` por execução completa. Não sobrescrevem
as fontes. Reexecutar a célula de gravação com o mesmo ID é bloqueado para impedir
duplicação. Para uma nova tentativa, execute o notebook inteiro e obtenha outro ID.
Uma falha pode deixar artefatos parciais: **consuma apenas IDs com manifesto
`CONCLUIDO_COMPARACAO_02C`**. As cinco escritas não formam uma transação conjunta.
O manifesto sempre conserva `publicavel=False` e `houve_retreino=False`.

## Dimensionamento e testes

A inferência pré-calcula distribuições para um catálogo pequeno de modelos/idades
no driver. Os casos são associados a essas distribuições por join Spark, sem UDF
Python por cliente e sem `mapInPandas` ou dependência de Arrow para essa etapa.
Só os agregados por cliente vão ao Pandas para o bootstrap, com proteções explícitas
de tamanho. As observações completas não vão ao driver. As proteções interrompem,
não truncam ou amostram silenciosamente.

Testes locais realizados: compilação Python e das células Jupyter; 120 modelos
sintéticos em sete idades confrontados com a função original da Parte 02; equivalência
da fórmula de B2 em log-space; preservação exata de B em idade zero; reordenação de
destinos/grupos; bloqueio de partições incompatíveis; não alteração dos dicionários
fontes; casos de cauda e sinais do bootstrap.

**Spark e Delta não foram executados neste ambiente.** A validação no Databricks
ainda é necessária. O notebook inclui autotestes Spark de acesso a mapas, destino
fora de suporte e censura, executados antes das leituras reais. Falhas de acesso,
permissões, schema ou reprodução interrompem a execução.

## Critério para a próxima decisão

Conferir primeiro as identidades/reprodução. Depois comparar B2 com A e B nas idades
positivas, sem exigir melhora em todos os indicadores nem escolher pelo número de
listas diferentes. Redução de probabilidades extremas não substitui avaliar perda
média, cobertura e acertos. Não eliminar os clientes que pioraram.

B2 ainda não foi avaliada no Multi-step. Se o teste sustentar a recombinação, a etapa
seguinte deve comparar a propagação contextual em amostra controlada antes de
alterar o consumo de negócio. Não há avanço automático para essa etapa.

## Referências de implementação

- Contrato interno: `nba_v23_02_treino_validacao_pareada.py`, formato `sm_v23_lognormal_grupos`.
- SciPy, `log_ndtr`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html
- SciPy, `logsumexp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
- Spark, `element_at`: https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.element_at.html
- Delta Lake, leituras versionadas e append: https://docs.delta.io/delta-batch/
