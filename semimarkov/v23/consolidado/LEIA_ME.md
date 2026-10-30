# NBA V2.3 — consolidação para homologação

## Por que este pacote foi criado

Os dois arquivos dos links da resposta anterior não estavam presentes no ambiente.
Esta entrega foi reconstruída a partir dos arquivos anteriores disponíveis. Não é
uma recuperação byte a byte daquela entrega indisponível.

O ZIP contém os dois notebooks em `.py` e `.ipynb`, estas instruções, o resultado
dos testes locais e um manifesto SHA-256. Importe **um formato de cada notebook**,
não execute `.py` e `.ipynb` em duplicidade.

## O que foi conservado

- Núcleo de probabilidades RAW, motor Multi-step, memória, ciclos e conservação de massa.
- B2: probabilidades de destino de B e curvas temporais de A.
- Os 17 alvos aprovados; cinco mudanças de macroação; até cinco posições de saída.
- A preparação operacional do estado e da memória, inclusive debounce de 30 segundos.
- Rótulos, censura, pesos, splits e base preparada da referência.
- Nenhuma regra nova de produto, calendário, frequência adicional ou balanceamento.

A correção de leitura Delta foi incorporada na seção de leitura do notebook 02.
As fontes operacionais Delta são fixadas na versão disponível ao início da leitura;
um novo commit não muda o snapshot escolhido. Troca da identidade da tabela
continua bloqueada. Não há fallback silencioso para outra versão.

## Arquivos

| Notebook | Responsabilidade |
|---|---|
| `01_nba_v23_treinamento_validacao` | Preservação/reprodução da referência ou reajuste explícito sobre a mesma base preparada congelada; composição B2; validação histórica; pacote independente. |
| `02_nba_v23_predicao` | Carregamento explícito do pacote, regressão contra kernels históricos, estado/memória, scoring e saída LONG. Não treina. |

## Primeiro uso: não retreine

No notebook 01, deixe:

```python
"modo": "PRESERVAR_REFERENCIA",
"confirmar_reajuste": False,
```

O ID histórico inicial é:

```text
92a92d57-ce93-487c-bdf1-369e3dbd542f
```

O notebook recupera a execução com 17 alvos e seu pacote físico. Confere os dados,
cria uma cópia independente em novas tabelas, reconstrói a validação histórica
com o motor preservado e compara chaves, labels, rankings, probabilidades e
métricas contra a referência. A tolerância numérica é `1e-9`.

Se uma comparação falhar, o novo histórico não recebe o marcador de conclusão.
Pode haver saídas parciais em tabelas isoladas; não são uma candidata aprovada.
Reexecute desde o início para obter novos identificadores.

## Como usar o pacote reproduzido na predição

Ao concluir o notebook 01, `CONS_01_CONCLUSAO` imprime:

```python
{
    "id_ms03_validado": "...",
    "tabela_ms03_execucoes": "...",
    "tabela_pacotes": "..."
}
```

Copie esses três valores para `OP_CFG` no notebook 02. Não copie somente o ID:
a nova execução e o pacote estão em tabelas da consolidação.

O notebook 02 também pode testar diretamente a referência anterior: os três
valores inicialmente configurados apontam para os artefatos aprovados já
existentes. Isso não exige rodar o notebook 01 na mesma sessão.

O padrão do scoring é `AMOSTRA`. A lista `ids_validacao` pode incluir clientes
conhecidos, desde que façam parte do público. Somente depois da homologação
execute `TODOS`.

## Fontes de predição

- `fonte_publico`: inicialmente `customers_query`, como no operacional anterior.
- `fonte_eventos`: `ctg_dsti.renato_nba.base_jornadas_onboarding`.

A ingestão não foi inventada ou incorporada. `customers_query` deve existir
nesta sessão. Para produção independente, configure uma tabela/view permanente
com `cd_bv` ou uma etapa de ingestão explícita no job. Não dependa de uma view
temporária criada manualmente em outro notebook.

## Saídas separadas

Todos os novos artefatos usam prefixos sob:

```text
ctg_dsti.renato_nba.nba_sm_v23_consolidada
```

A predição escreve em:

```text
ctg_dsti.renato_nba.nba_sm_v23_consolidada_operacional_...
```

Views novas:

```text
ctg_dsti.renato_nba.nba_sm_v23_consolidada_operacional_amostra_atual
ctg_dsti.renato_nba.nba_sm_v23_consolidada_operacional_negocio_atual
```

As tabelas físicas operacionais usam append por `id_execucao`; as views escolhem
somente execuções concluídas do mesmo modo/modelo, ordenadas por corte e conclusão.
O modelo e as views antigas de negócio não são promovidos, trocados ou sobrescritos.
Nesta entrega, os prefixos de homologação têm guardas explícitas para evitar que
sejam alterados inadvertidamente para os nomes anteriores.

## Caminho de treinamento

O modo `REAJUSTAR_CONGELADO`, junto com `confirmar_reajuste=True`, permite executar
novamente o otimizador sobre a MESMA base preparada congelada, com os parâmetros
numéricos recuperados do manifesto. Reexecuta a implementação original A/B/C/D
para preservar os diagnósticos, compõe B2 e executa a validação histórica.
As variantes ponderadas C/D não são selecionadas para o scoring B2.

Isso não prepara uma nova população a partir de telemetria atual. A base preparada
é um contrato de entrada explícito. Ampliar a amostra, mudar o período, refazer o
split ou introduzir features exige uma experiência distinta; não é feito nesta
consolidação. O reajuste não exige scores idênticos aos do modelo anterior e não
promove automaticamente a candidata resultante.

## Homologação operacional

A reprodução histórica verifica o motor e os rótulos. Para comprovar equivalência
fim a fim do scoring operacional, compare público, estados, memória, idade,
rotas, scores, ranks, massas e status usando o mesmo pacote, mesmos dados e mesmo
corte. Eventos novos podem alterar os resultados legitimamente; não compare uma
fonte atualizada com uma saída antiga como se fossem as mesmas entradas.

Não há confirmação de equivalência Spark/Delta executada neste ambiente. Não
substitua diretamente o job em produção somente porque os arquivos foram gerados.

## Testes locais realizados

17 testes passaram: sintaxe, estrutura dos notebooks, preservação das funções
estatísticas por comparação de AST (desconsiderando apenas espaços de docstrings),
integração exata do leitor corrigido, modos e prefixos seguros, autoteste de
verossimilhança/gradiente, composição B2, memória/ciclos/massa e equivalência do
cálculo em lotes num grafo sintético com 17 alvos.

O ambiente local não tem PySpark/PyArrow. Portanto, os testes locais não executam
leitura Delta, permissões, joins, persistência ou distribuição no cluster. Os
notebooks mantêm autotestes e verificações no Spark antes e durante a execução.
