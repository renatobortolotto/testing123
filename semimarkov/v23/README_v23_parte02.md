# NBA V2.3 — Parte 02: ajuste temporal e validação pareada

## O que executar

Importe **um** dos arquivos `nba_v23_02_treino_validacao_pareada.py` ou `.ipynb`
como o segundo notebook da V2.3 no Databricks. Os dois contêm as mesmas células.
Não é necessário reexecutar a Parte 01 ou manter seus objetos em memória.

O ID do preparo já está configurado:

```python
"id_experimento": "ac9ba1a5-e908-4d8b-958e-ce5536a6fa72"
```

O notebook exige uma única linha `CONCLUIDO_PREPARO` para esse ID. Lê as tabelas
de base/suporte indicadas pelo manifesto, filtra o ID e fixa as versões Delta
consumidas na execução. Conserva o corte histórico de 25/09/2026, os splits,
os pesos e as observações elegíveis. O fuso da sessão deve corresponder ao
preparo; não o altera automaticamente.

**Esta é uma etapa de treinamento e avaliação do próximo estado RAW. Não é
scoring de toda a população, não valida ainda o ranking macro acionável e
não substitui a tabela de negócio.**

## Quatro variantes

| Variante | Dados de ajuste | Memória em origens técnicas | Peso |
|---|---|---|---|
| A_REFERENCIA | Todas as observações elegíveis de treino | Não | 1 |
| B_MEMORIA | Mesmas observações; ajustes contextuais nos candidatos | Sim | 1 |
| C_PONDERACAO | Todas as observações elegíveis de treino | Não | 1/n do cliente-origem |
| D_MEMORIA_PONDERACAO | Mesmas observações; ajustes contextuais nos candidatos | Sim | 1/n do cliente-origem |

Não há nova amostragem ou balanceamento dos destinos. A utiliza a mesma
especificação estatística e mesma base desta comparação; **não deve coincidir
numericamente com a antiga V2.2**, que usava um limite amostral por origem e
outra regularização. A V2.2 persistida permanece intacta.

B reutiliza A nas origens não técnicas e nos contextos sem ajuste. D reutiliza C.
O compartilhamento é sempre com um **modelo temporal** da origem. Se não existe
pai ajustado, a previsão é marcada como ausente; não há fallback de frequências
estáticas. Os pesos de treino nunca são atribuídos ao holdout.

## Família estatística

Para cada origem, os destinos observados no treino são agrupados para estimar
durações Lognormais. O vocabulário e o agrupamento são definidos uma vez por
contagens sem peso e compartilhados entre A/C e seus filhos B/D. Até oito
destinos com suporte suficiente recebem grupos próprios; outros compartilham
um grupo temporal, **sem desaparecer como destinos**.

Escrevendo `g(j)` para o grupo temporal do destino:

```text
p_j = pi_g(j) * r_j|g(j)
exata:   log(pi_g(j) * r_j|g(j) * f_g(j)(t))
direita: log(sum_g pi_g * S_g(c))
```

O peso multiplica a contribuição inteira da observação: destino e duração nas
exatas; sobrevivência de mistura nas censuradas com destino desconhecido.
Não atribui um destino artificial às censuras.

A função minimizada é a **negativa da média ponderada da log-verossimilhança**,
mais penalidades. Dividir a função por `sum(w)` evita confundir a troca da
escala dos pesos com a força de regularização; **não recalcula o peso de cada
cliente dentro de cada contexto**.

### Regularização inicial explícita

- Base A/C: pequena suavização probabilística em direção a destinos uniformes
  (`lambda_prob_base=0.0001`); sem penalidade temporal adicional.
- Contexto B: prior probabilístico/temporal do pai A.
- Contexto D: prior probabilístico/temporal do pai C.
- Lambdas contextuais: `0.02` para probabilidade e `0.02` para duração.

Em notação resumida, a penalidade probabilística é a entropia cruzada da
distribuição de referência com a distribuição ajustada. A penalidade temporal
usa diferenças quadráticas das médias de log-tempo (padronizadas pelo sigma
do pai) e dos log-sigmas, ponderadas pela massa de grupo do pai.

Esses são **valores iniciais definidos antes de ler a comparação**, não
hiperparâmetros otimizados ou certificados. Contextos com destinos não observados
mantêm a possibilidade desses destinos pela suavização em direção ao pai.

A massa de pesos de um contexto não é o número de clientes distintos e não é
um tamanho amostral independente. Um cliente-origem distribui sua massa entre
os contextos observados; não recebe novamente massa 1 em cada contexto.

### Otimização

Usa SciPy L-BFGS-B com limites, gradiente analítico e uma segunda inicialização
em caso de falha. As saídas exatas usam estatísticas suficientes por grupo para
reduzir trabalho repetido. A solução continua sujeita a ótimo local e
especificação paramétrica inadequada; convergência não certifica o modelo.

Parâmetros que atingem limites ficam sinalizados em `alerta_limite` e
`parametros_no_limite`. Não são escondidos das métricas. Uma falha numérica de
contexto usa o pai temporal com rota explícita. Não reduz automaticamente os
mínimos de suporte nem os limites para produzir mais modelos.

## Memória: o que esta parte implementa

Utiliza `contexto_modelo` já calculado no prefixo histórico da Parte 01,
inclusive durante `topo`, `navegacao` e `sucesso` de login. Somente origens
técnicas recebem parâmetros contextuais próprios nesta candidata.

Roteamento de B/D:

| Situação | Modelo/rota |
|---|---|
| Origem não técnica | Pai temporal correspondente |
| Memória desconhecida | Pai temporal; `ORIGEM_SEM_MEMORIA` |
| Contexto ajustado | Filho contextual |
| Contexto raro | Pai temporal; `ORIGEM_CONTEXTO_RARO` |
| Contexto não visto no treino | Pai temporal; `ORIGEM_CONTEXTO_NAO_VISTO` |
| Contexto candidato cujo ajuste falhou | Pai temporal; `ORIGEM_FALHA_AJUSTE_CONTEXTO` |
| Pai sem ajuste | `SEM_MODELO_ORIGEM` |

A tabela de suporte considera apenas treino. O holdout não seleciona grupos,
contextos ou parâmetros. Contagens de clientes por status de contexto podem
se sobrepor: um cliente passa por vários contextos ao longo da jornada.

**Não use os novos modelos no antigo multi-step sem adaptar o roteamento.**
Na próxima etapa, cada caminho deverá carregar sua própria memória, preservá-la
nas etapas técnicas e atualizá-la ao entrar em outro funil não técnico. Usar o
contexto apenas no primeiro passo e voltar à matriz P0 compartilhada não atende
ao objetivo da V2.3.

## Avaliação incluída

Marcos: `0`, `1800`, `86400` e `604800` segundos. Só entram observações com
`dur_min > marco`: nesse instante a saída ainda não ocorreu.

A probabilidade prevista é:

```text
q_j(a) = p_j * S_g(j)(a) / sum_k p_k * S_g(k)(a)
```

Em `a=0`, q=p. Se todos os destinos compartilham a mesma distribuição de
tempo, a sobrevivência cancela no ranking, embora a duração continue modelada.

A tabela de validação mantém exatas e censuradas, mas as censuras **não recebem
rótulo de destino ou rótulo de não ação**. Acertos e perdas de destino usam
somente saídas exatas observadas até o corte. Essa seleção limita a inferência:
**não certifica probabilidade em horizonte de dias, calibração ou multi-step**.

Métricas por evento previsto:

- Top-1 e Top-5 do próximo estado RAW;
- Brier multiclasse, sem dividir por dois;
- log loss em log natural, com piso 1e-15 explicitamente reportado;
- cobertura do modelo e do destino;
- quantidade de log losses que exigiram o piso;
- diferença de log loss entre q(a) e p, nos mesmos casos, para examinar o tempo.

Destino que não apareceu no treino recebe p=0 e é penalizado, não excluído.
Ausência de modelo aparece na cobertura, sem receber uma probabilidade inventada.
As médias condicionadas à existência de previsão devem ser lidas junto da
cobertura. Uma média alta em outro marco não prova ganho temporal: os casos
que permanecem em risco mudam entre marcos.

### Comparação pareada principal

Mantém a interseção de observações com previsão nas quatro variantes e, em
cada marco/segmento, calcula a média por cliente nos mesmos casos. Compara
B/C/D contra A com bootstrap de clientes, não de linhas individuais.

**Ganho positivo sempre significa melhora:** A menos variante para log loss e
Brier; variante menos A para Top-1/Top-5. O relatório traz número de casos,
clientes e intervalo percentil de 95% sobre a diferença média. Os resultados
são de desenvolvimento: não promover automaticamente a variante com melhor
média. Esse holdout já orientou escolhas anteriores; uma avaliação temporal
posterior independente continua necessária.

São reportados `TODAS` as origens e `ORIGENS_TECNICAS`. Não esperamos que a
memória altere origens não técnicas nesta primeira candidata; por isso é
importante examinar o segmento afetado além da média global.

## Persistência e execução noturna

O padrão é `gravar=True`. Todas as tabelas são novas V2.3, com append e
`id_experimento` + `id_ajuste` (novo UUID em cada execução completa):

```text
ctg_dsti.renato_nba.nba_sm_v23_modelos_hml
ctg_dsti.renato_nba.nba_sm_v23_validacao_destino_hml
ctg_dsti.renato_nba.nba_sm_v23_metricas_destino_hml
ctg_dsti.renato_nba.nba_sm_v23_comparacao_pareada_hml
ctg_dsti.renato_nba.nba_sm_v23_ajustes_hml
```

A tabela de modelos contém pais A/C, filhos B/D e registros explícitos de
reutilização dos pais em B/D (`nivel=ORIGEM`, `reuso=True`). Use o nível e o
contexto na chave, não somente o estado de origem.

Os pais são salvos antes de começar os contextos; os contextos são salvos antes
da validação. Uma falha posterior não apaga os artefatos que já foram escritos,
mas não gera manifesto concluído. As tabelas não constituem uma transação
conjunta. Consuma somente o ID com `CONCLUIDO_TREINO_VALIDACAO` na última tabela.
Não há promoção automática; `publicavel=False` no manifesto.

O notebook não implementa retomada incremental de execução parcial. Reexecutar
inteiro gera outro ID de ajuste; não reexecutar células de append já concluídas
para o mesmo ID. Execuções antigas e a V2.2 ficam preservadas.

## Memória e paralelismo

O treino roda em `applyInPandas` por origem/contexto. Uma origem inteira precisa
caber no processo Python do executor. A maior origem informada nos relatórios
é muito menor que a proteção de 150 mil linhas do notebook; caso a proteção
seja excedida, a execução para, sem reduzir dados silenciosamente.

Não coleta a base de eventos inteira para o driver. Coleta modelos compactos
e médias agregadas por cliente para bootstrap, com limites explícitos.
As estruturas de trabalho Spark novas usam disco quando persistidas. Não usa
`clearCache()` nem interfere no cache de outros notebooks.

## Relatórios para devolver

```text
V23_20_DADOS_AJUSTE
V23_21_MODELOS_ORIGEM / V23_21_FALHAS_ORIGEM
V23_22_MODELOS_CONTEXTO / V23_22_FALHAS_CONTEXTO
V23_23_ROTAS_HOLDOUT
V23_24_METRICAS_DESTINO
V23_25_COMPARACAO_PAREADA
V23_26_CONCLUSAO + id_ajuste
```

## Testes realizados nesta entrega

Foram executados localmente testes com dados sintéticos de: gradiente analítico
(pai e contexto), normalização, q(0)=p, censura de mistura, invariância à escala
global dos pesos, herança de suporte, roteamento de fallback temporal,
efeito de cliente muito ativo e wrappers Pandas de treinamento/validação.
Os wrappers produziram as quatro variantes por caso, mantiveram ausências de
modelo e não rotularam censuras como destinos negativos.

Sintaxe Python e estrutura do `.ipynb` verificadas. **Acesso às suas tabelas,
execução distribuída Spark/Arrow, permissões Delta e resultados reais não foram
executados neste ambiente.** Os autotestes numéricos também rodam no Databricks
antes das fontes reais.

## Referências técnicas usadas na implementação

- SciPy 1.13.1: `rv_continuous.fit`, contribuição de dados censurados à
  verossimilhança e limitações de otimização.
- SciPy 1.13.1: `minimize(method='L-BFGS-B')`, gradiente, limites e convergência.
- PySpark 3.5: `GroupedData.applyInPandas`, agrupamento e memória por grupo.
