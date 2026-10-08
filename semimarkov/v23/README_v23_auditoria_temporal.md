# NBA V2.3 — Parte 02B: auditoria temporal

## Execução

Importe **um** dos arquivos `nba_v23_02b_auditoria_temporal.py` ou
`nba_v23_02b_auditoria_temporal.ipynb` em um notebook de auditoria no Databricks.
Execute do início ao fim, depois da Parte 02 concluída e persistida.
Não é necessário reexecutar a Parte 01, o treinamento, a preparação bruta ou o
scoring completo. Não depende das variáveis desses notebooks em memória.

IDs preenchidos:

```python
"id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72",
"id_ajuste": "2a3fb103-cb9e-4862-ac61-32b5c288ff6a",
```

O ID de auditoria é gerado a cada execução completa. Se for auditar outro
ajuste, altere o ID explicitamente. Não existe seleção automática do último
modelo nem promoção automática de uma variante.

## O que é e o que não é

A auditoria localiza possíveis causas da deterioração de probabilidades nas
idades maiores. Ela verifica parâmetros, roteamento, scores, suporte temporal
e concentração dos erros. **Não retreina, não otimiza hiperparâmetros, não
altera os limites, não recalibra probabilidades e não executa Multi-step.**

Também não certifica ótimo global, calibração ou desempenho futuro. O holdout
já é um conjunto de desenvolvimento: subgrupos e intervalos usados nesta
investigação são exploratórios.

## Fontes e consistência

O notebook exige um único manifesto `CONCLUIDO_TREINO_VALIDACAO` no ajuste.
Lê as tabelas de modelos e validação indicadas no `config_json` desse manifesto
filtrando `id_experimento` e `id_ajuste`. As versões Delta dessas leituras são
fixadas e registradas na auditoria.

Para base de treino e suporte, utiliza as **versões Delta registradas na
Parte 02**, e não uma versão atual com dados possivelmente diferentes. Se
uma versão histórica não estiver mais disponível, a execução deve parar;
não há substituição silenciosa por uma versão recente. Histórico Delta requer
retenção dos arquivos correspondentes e não equivale a um backup permanente.

O fuso deve coincidir com o registrado no preparo. Não converta timestamps ou
mude a data de corte apenas para contornar essa verificação.

A auditoria verifica chaves, hashes, separação treino/holdout, quatro variantes
por caso, correspondência com a observação histórica, corte por idade e
recontagem do treino efetivamente usado em cada modelo.

## Perguntas respondidas

### 1. Quais parâmetros atingiram limites?

Reconstrói os limites da versão de ajuste utilizada:

- `logit_k`: log da razão entre a massa do grupo k e a massa do último grupo.
  Limites da especificação original: −25 a +25. O grupo de referência não
  tem um logit livre e não é inventado como novo parâmetro.
- `mu_k`: média do **logaritmo do tempo em dias**; usa os limites preservados
  em `estrutura.limite_mu`.
- `log_sigma_k`: log da dispersão do log-tempo; lê `sigma_min` e `sigma_max`
  do manifesto. Mostra o valor de sigma também na escala natural.

Identifica **lado inferior/superior**, valor ajustado, valor do pai e diferença.
Repete a tolerância `1e-4` da Parte 02 e confere `parametros_no_limite` e
`alerta_limite` já gravados. Parâmetros fora do domínio ou alertas impossíveis
de reproduzir são falhas estruturais; o notebook para.

Atingir uma restrição não prova sozinho que o modelo está errado. Um logit no
limite e um sigma no limite representam fenômenos distintos. A auditoria não
relaxa limites automaticamente nem interpreta convergência como ótimo global.

### 2. Esses modelos tinham suporte na cauda?

Conta as observações de treino de cada modelo com duração observada maior que
a idade avaliada, separando exatas e censuradas. Apresenta número de clientes,
número de observações e massa de pesos desse conjunto.

O limiar de 20 clientes é **somente triagem**, não um novo requisito de ajuste
ou de publicação. Uma idade pode estar abaixo do máximo global de treino e
mesmo assim ter poucos clientes de suporte. O inverso também importa:
`extrapolacao=True` é a definição global da Parte 02, não uma certificação de
que todos os demais casos têm bom suporte.

Nos grupos temporais, apresenta saídas exatas, clientes, quantidade de tempos
distintos, dispersão do log-tempo e quantis P10/P50/P90. As censuras não são
atribuídas a grupos/destinos desconhecidos. Ausência de saídas exatas num grupo
contextual pode coexistir com parâmetros herdados/regularizados pelo pai.

Contagens de risco não são uma curva Kaplan–Meier nem uma probabilidade de
sobrevivência empírica calibrada. Servem para examinar suporte.

### 3. Os scores persistidos são reproduzíveis?

Recalcula q a partir do JSON em log-space:

```text
log_mass_g(a) = log(pi_g) + log S_g(a)
log q_j(a) = log_mass_g(j)(a) − logsumexp(log_mass(a)) + log r_j|g(j)
```

Usa todas as probabilidades do catálogo; não renormaliza o Top 5. Confere a
sobrevivência também por `scipy.stats.lognorm.logsf`, com `s=sigma` e
`scale=exp(mu)` em dias. Verifica `q(0)=p`, massa, Brier multiclasse sem divisão
por dois, log loss, clipping, acertos e ordenação RAW.

A unidade de `idade_seg` é **segundos já transcorridos**, não um horizonte
futuro de 30 minutos ou sete dias.

Censuras permanecem sem rótulo de destino. Destinos não suportados recebem a
mesma penalização explícita da Parte 02, não desaparecem das métricas.
O piso de log loss é lido do manifesto original e não é alterado. Para destinos
suportados, o log(q) sem piso também fica disponível nos casos de triagem,
inclusive quando `exp(log(q))` é zero por underflow.

### 4. A piora se concentra em quais contextos, idades e clientes?

A comparação utiliza a mesma interseção de casos exatos com previsão nas
quatro variantes da Parte 02. Inclui faltas de suporte, clipping e extrapolações
nesses casos; não elimina os exemplos difíceis.

A auditoria distingue modelo contextual efetivamente utilizado de fallback de
origem. Por exemplo, B com `ORIGEM_CONTEXTO_RARO` deve ser associado ao pai A,
e não a um filho B inexistente. Os pais reutilizados em B/D não são contados
como novos ajustes de parâmetros.

Para localizar deterioração, os relatórios usam:

```text
delta_logloss = logloss_variante − logloss_referência
valor positivo = piora
```

A tabela por contexto apresenta uma contribuição **aditiva** à diferença
média por cliente:

```text
contribuição do grupo =
    soma(delta_caso / número_de_casos_do_cliente_no_recorte)
    / número_de_clientes_no_recorte
```

A soma das contribuições dos grupos reproduz a diferença média global por
cliente, conferida pelo notebook. Isso evita trocar o denominador ao comparar
contextos. Não é uma atribuição causal ao parâmetro no limite.

Os cruzamentos de flags são observacionais. Idades, contextos e volume de
clientes podem diferir entre categorias. Os clientes de categorias distintas
podem se sobrepor, pois a categoria pertence à observação.

### 5. Poucos clientes dominam o resultado? D realmente melhora B?

A concentração mostra a fração da **piora positiva** produzida pelos 1, 5 e 10
clientes com maior deterioração média. Essa parcela não é a fração do delta
líquido, porque melhorias de outros clientes podem compensar perdas.
Nenhum desses clientes é removido das métricas.

Repete as comparações A/B/C/D e acrescenta **D × B diretamente**. Usa médias
por cliente nos mesmos casos e bootstrap de clientes. Nos intervalos:

```text
ganho em perdas = referência − variante
ganho em acertos = variante − referência
valor positivo = melhora
```

Os intervalos são exploratórios, sem correção por testes múltiplos e sem
escolha automática de vencedor. Não representam uma nova validação externa.

## Relatórios principais para enviar

```text
V23_AUD_01_LIMITES_RESUMO
V23_AUD_02_PARAMETROS_LIMITES
V23_AUD_05_ERROS_POR_ALERTA
V23_AUD_06_CONTEXTOS_PRIORITARIOS
V23_AUD_07_CONCENTRACAO_CLIENTES
V23_AUD_08_PARES_D_VS_B
V23_AUD_12_CONCLUSAO
```

A versão compacta dos prints evita dezenas de colunas em cada imagem. Os
DataFrames completos ficam no dicionário `AUD_RELATORIOS` e nos resumos
persistidos. Não é necessário enviar todas as linhas ou dados identificáveis.

Relatórios auxiliares:

- `V23_AUD_03_INTEGRIDADE`: verificações de contratos e reprodução.
- `V23_AUD_04_COBERTURA`: casos antes da interseção, sem ocultar ausência de
  modelo/destino.
- `V23_AUD_06_CONTEXTOS_COMPLETO`: todos os grupos, sem corte por ranking.
- `V23_AUD_08_PARES_COMPLETO`: A/B/C/D contra A e D contra B, por segmento/idade.
- `V23_AUD_09_CASOS_TRIAGEM`: maiores pioras caso a caso em origens técnicas.
- `V23_AUD_10_EXPOSICAO_CAUDA`: volume realmente exposto a cada condição.
- `V23_AUD_11_CURVAS_MODELOS_PRIORITARIOS`: probabilidade do Top 1, variação
  frente a p e sobrevivência de mistura dos modelos prioritários de B.

Exemplo de inspeção no notebook, sem retreino:

```python
display(
    AUD_RELATORIOS["V23_AUD_06_CONTEXTOS_COMPLETO"]
    .filter("comparacao = 'B_VS_A' AND idade_seg = 86400")
    .orderBy(F.desc("contrib_delta_logloss_media_cliente"))
)
```

## Persistência e dados identificáveis

`gravar=True` grava **somente novas tabelas de auditoria**, por append e com
`id_experimento`, `id_ajuste`, `id_auditoria` e versão da auditoria:

```text
ctg_dsti.renato_nba.nba_sm_v23_auditoria_resumos_hml
ctg_dsti.renato_nba.nba_sm_v23_auditoria_parametros_hml
ctg_dsti.renato_nba.nba_sm_v23_auditoria_casos_hml
ctg_dsti.renato_nba.nba_sm_v23_auditoria_execucoes_hml
```

`resumos` guarda um JSON por linha de relatório, com o nome em `relatorio`,
permitindo esquemas distintos num único destino. `parametros` é uma tabela
relacional com todos os parâmetros e seu suporte, não só os alertas.

`casos` guarda no máximo 20 observações de piora por comparação/idade no
segmento técnico. Essa seleção é só para inspeção, **não altera a avaliação**.
A tabela interna conserva `cd_bv` para rastreamento, mas os prints não mostram
esse identificador. Não compartilhe a tabela individual fora do ambiente
permitido pelo banco.

As quatro escritas não são uma transação conjunta. Apenas
`CONCLUIDA_AUDITORIA` no manifesto indica conclusão de todas elas. Uma nova
execução completa gera outro ID; reexecutar a célula de gravação com o mesmo ID
não apaga nem duplica silenciosamente o resultado anterior.

O manifesto termina com:

```text
status_revisao = AGUARDA_REVISAO_DOS_RESULTADOS
autoriza_promocao = false
```

Isso é proposital. Ausência de falha estrutural não é autorização para mudar a
tabela de negócio ou seguir diretamente para produção.

## Como usar o resultado para decidir o próximo passo

1. **Divergência estrutural ou de reprodução:** corrigir o contrato/roteamento
   ou investigar o artefato antes de comparar novos modelos.
2. **Probabilidades tardias ruins concentradas em limites/baixo suporte:**
   investigar a especificação desses contextos; qualquer ajuste deve receber
   um novo `id_ajuste` e nova avaliação, sem alterar os artefatos atuais.
3. **Sem divergências e sem problema que exija correção antes de propagar:**
   seguir com um experimento Multi-step A × B carregando a memória em cada
   trajetória. Isso ainda é uma nova avaliação, não promoção automática.

Nenhuma dessas decisões é executada automaticamente pelo notebook. A auditoria
não relaxa bounds, não recorta probabilidades para melhorar as métricas, não
retira PIX, não remove clientes e não força diversidade.

## Escala e testes

Os eventos e joins ficam no Spark; novos caches usam `DISK_ONLY`. Só modelos
compactos e médias agregadas por cliente são levados ao driver, com proteções
explícitas de tamanho. O código não chama `clearCache()` nem limpa caches de
outros notebooks. Fontes/modelos/tabela de negócio nunca são sobrescritos.

Testes locais realizados:

- q(0)=p, normalização e comportamento de sobrevivência em caudas;
- reprodução da inferência da Parte 02 em **seis modelos sintéticos e 36
  combinações modelo-idade**, incluindo pais e contextos, com/sem pesos;
- detecção de limites inferiores/superiores de sigma, mu e logit;
- rejeição de distribuições corrompidas e flags de limite inconsistentes;
- sinais de ganho e ponderação igual de clientes no bootstrap;
- decomposição aditiva das diferenças por contexto;
- sintaxe Python, células do `.ipynb` e ausência de operações de retreino e
  sobrescrita de fontes.

**Spark/Arrow/Delta e os dados reais do banco não foram executados neste
ambiente.** Há autotestes numéricos e um pequeno autoteste Spark de roteamento
antes das fontes reais. Permissões, retenção Delta e execução distribuída são
validadas no Databricks.

## Referências técnicas

Documentação primária consultada para a implementação:

```text
SciPy 1.13.1 — lognorm (sigma e scale=exp(mu); logsf)
https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.lognorm.html

SciPy 1.13.1 — L-BFGS-B (limites e critérios de parada)
https://docs.scipy.org/doc/scipy-1.13.1/reference/optimize.minimize-lbfgsb.html

Databricks — histórico/versionAsOf e retenção de tabelas Delta
https://docs.databricks.com/aws/en/tables/history

scikit-learn — limitações de interpretar Brier como calibração isolada
https://scikit-learn.org/stable/modules/calibration.html
```
