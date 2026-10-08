# NBA V2.3 — memória e ponderação: Parte 01

## Escopo da entrega

Este notebook prepara o experimento; **não é o treinamento completo da V2.3**.
Ele conserva os eventos da base V2.2 clean e acrescenta contexto de memória,
pesos por cliente e relatórios de suporte. O treino, a propagação contextual
multi-step e a publicação de recomendações serão etapas posteriores.

Importe `nba_v23_01_preparar_memoria_pesos.py` no Databricks como **novo notebook**.
O arquivo `.ipynb` contém as mesmas células; não execute ambos.

## Fontes já existentes

```text
ctg_dsti.renato_nba.nba_sm_v22_base_treino_clean_hml
ctg_dsti.renato_nba.nba_config_estados_v222_clean
```

Não lê `customers_query`, não chama as fontes brutas, não refaz o debounce,
não retreina, não altera o split nem a janela e não depende de objetos de outro
notebook em memória. O fuso esperado é `Etc/UTC`, igual ao experimento anterior.
Não altera o fuso da sessão automaticamente.

A leitura fixa as versões Delta das duas fontes. Se houver mais de uma
`id_execucao` na base, selecione `id_execucao_base` na configuração.

## Memória: significado e atualização

| Evento observado | Memória disponível |
|---|---|
| `pagamentos_boleto:::sucesso` | `pagamentos_boleto` |
| `app_login:::topo` | `pagamentos_boleto` |
| `app_login:::navegacao` | `pagamentos_boleto` |
| `app_login:::sucesso` | `pagamentos_boleto` |
| `pix_pagamento:::topo` | `pix_pagamento` |

Usa `funil`, não uma nova categorização por nome de estado. Consultas também
podem ser memória de negócio; a lista de ações acionáveis não define memória.

A política inicial memoriza o último funil não técnico. Os cinco funis técnicos
são login, habilitação do device, primeiro acesso, reset de senha e atualização
cadastral. Durante esses funis, a memória permanece disponível. Atendimento não
substitui a memória de negócio e não terá parâmetros contextuais nesta primeira
candidata. Essa é uma política configurada, não uma propriedade garantida dos dados.

Um estado ambíguo ou lacuna em `passo` reinicia a memória. No início do histórico,
ela pode estar desconhecida. `__SEM_MEMORIA__` é uma categoria de contexto, **não
um estado/evento artificial**. Não existe expiração temporal arbitrária.

`contexto_modelo` condicionará apenas as origens técnicas. Para outras origens,
usa `__BASE__`, preservando o modelo por estado. Quando o futuro multi-step entrar
em outro funil não técnico, a memória da trajetória deve ser atualizada. Aplicar
o contexto só ao primeiro passo e descartá-lo depois não cumpre o contrato.

Este preparo preserva a granularidade e o relógio anteriores. Não resolve o
problema separado de distinguir repetição da mesma ação em novas sessões.

## Ponderação

Para observações elegíveis de treino do cliente `u` na origem `i`:

```text
peso_evento_treino = 1
peso_cliente_treino = 1 / n_observacoes_cliente_origem
```

O denominador inclui exatas e censuradas elegíveis. Cada par cliente-origem soma
1. Não é peso por classe, por duração, por volume financeiro ou pelo total do
cliente em todas as origens. Não duplica nem descarta observações.

Os pesos são definidos pela origem RAW, **sem reparticionar por memória**.
Assim o experimento de memória não muda automaticamente a influência do cliente.
Holdout e linhas inelegíveis não recebem pesos de ajuste. Os pesos de avaliação
precisam ser calculados posteriormente sobre os snapshots efetivamente avaliados,
e não copiados destes pesos de treinamento.

No futuro ajuste, o peso deve multiplicar a contribuição inteira da observação
à verossimilhança, inclusive a sobrevivência de uma linha censurada. Não aplicar
ponderação apenas à matriz de destinos. A escala dos pesos também precisa ser
compatibilizada com a regularização (por exemplo, objetivo médio ponderado),
para que uma comparação não seja explicada somente por mudança na penalidade.

## Experimentos planejados

| Variante | Memória | Peso |
|---|---|---|
| A_REFERENCIA | Não | 1 por observação |
| B_MEMORIA | Sim | 1 por observação |
| C_PONDERACAO | Não | 1 / n do cliente-origem |
| D_MEMORIA_PONDERACAO | Sim | 1 / n do cliente-origem |

O artefato V2.2 permanece intacto. A referência A deverá ser ajustada na mesma
seleção de observações utilizada pelas outras variantes; não se deve presumir
que reproduzirá numericamente um ajuste anterior com outra amostra por origem.
Esta Parte 01 não faz nova amostragem nem ajusta nenhum dos quatro modelos.

Os limites de contexto (80 saídas exatas e 20 clientes com saídas exatas) servem
para identificar candidatos. Contextos sem suporte continuam na base. O futuro
estimador deverá compartilhar informação com a origem, não baixar limites só
para aumentar cobertura nem aprender suporte a partir da validação.

## Saídas novas e persistência

```text
ctg_dsti.renato_nba.nba_sm_v23_base_memoria_pesos_hml
ctg_dsti.renato_nba.nba_sm_v23_suporte_contexto_hml
ctg_dsti.renato_nba.nba_sm_v23_experimentos_hml
```

`SM23_CFG['gravar']` inicia em `True`. As saídas usam append e um novo
`id_experimento`. Não substituem a V2.2 nem a tabela de negócio. A última tabela
é o manifesto; somente `CONCLUIDO_PREPARO` autoriza consumo daquele experimento.

Se houver falha durante a escrita, partes podem ter sido salvas, mas o manifesto
não será concluído. Reexecutar o notebook inteiro cria um novo ID. Não interpretar
as tabelas parciais como uma execução completa e não apagar versões antigas.

## Relatórios para revisar antes de treinar

```text
V23_01_INVENTARIO
V23_02_COBERTURA_MEMORIA
V23_03_SUPORTE_CONTEXTOS
V23_04_AUDITORIA_PESOS
V23_05_PESOS_POR_ORIGEM
V23_06_CONTEXTO_HOLDOUT
```

O inventário deve preservar linhas/clientes da base selecionada. A auditoria dos
pesos deve mostrar soma 1 por cliente-origem, dentro da tolerância numérica.
Os autotestes Spark também verificam censura no denominador, memória através do
login, barreiras e invariância do contexto ao adicionar eventos futuros.

## Validação realizada nesta entrega

Sintaxe Python e estrutura Jupyter verificadas. A função pura de atualização da
memória foi testada, inclusive em 250 sequências para invariância do prefixo.
As identidades de pesos foram verificadas em exemplos sintéticos. **Spark e Delta
não foram executados localmente**; os autotestes Spark embutidos rodarão no cluster
antes da leitura das tabelas reais. Não foi validado o desempenho no seu Databricks.

## Referências técnicas

- Rosvall et al., *Memory in network flows and its effects on spreading dynamics
  and community detection*: https://arxiv.org/abs/1305.4807
- Spark `last`, utilizado em janela ordenada com limite até a linha atual:
  https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.last.html
- Databricks, histórico e leitura de versões Delta:
  https://docs.databricks.com/aws/en/tables/history
- SciPy 1.13.1, contribuições de dados censurados à verossimilhança:
  https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.rv_continuous.fit.html
