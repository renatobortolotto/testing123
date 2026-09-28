# NBA — correção dirigida do timeout, V2.1

## O que executar agora
1. Não reexecute as partes 01 e 02 V2 nem o diagnóstico 02C.
2. No MESMO notebook onde o 02C terminou, acrescente e execute
   `nba_mvp_02d_corrigir_timeout_v21.py` em ordem.
3. Revise os quadros de resultado, cobertura, output e validação pareada.
4. A parte 03 antiga é incompatível com a nova família temporal.
   Sua substituta é `nba_mvp_03_auditar_timeout_v21.py`.
   Execute-a inicialmente com `GRAVAR_HOMOLOGACAO = False`.
5. Só uma decisão explícita posterior deve habilitar persistência em homologação.

## Escopo
- Reutiliza smc_dados (recorte de ajuste já usado no 02C).
- Reajusta somente origens cujo detalhe é LIMITE_PARAMETRICO_ATINGIDO.
- Preserva JSON e hash de cada modelo anteriormente AJUSTADO.
- Não mexe nas 118 origens com suporte insuficiente.
- Não altera base, relógio, público, unidade, cortes, sigma_min/sigma_max ou mínimos.
- Não usa validação para ajustar parâmetros.
- Não imputa durações, não acrescenta ruído e não volta ao ranking estático.
- Cria views candidatas V2.1 separadas das views V2. Não grava tabelas no 02D.
- Não há promessa de que as 29 origens passarão: limites remanescentes são recusados
  com identificação do parâmetro. Alterar a família não prova sua adequação.

## Modelo temporal da transição para silêncio
Na base operacional, T = span_do_bloco + timeout.
Quando span=0, há massa pontual em T=timeout. Quando span>0, modela-se o
excesso T-timeout com uma lognormal, mantendo as durações totais originais.

- Só massa observada: átomo no timeout, sem desvio-padrão artificial.
- Só excesso positivo observado: lognormal deslocada no timeout.
- Ambos: mistura de átomo e cauda deslocada, com peso ajustado conjuntamente.
- Destinos não-silêncio mantêm lognormais, com pooling dos raros.
- O silêncio não é incluído nesse pooling comportamental.

As probabilidades de destino, os tempos e a massa são ajustados por uma função
objetivo conjunta penalizada. Censuras sem destino entram pela soma das
sobrevivências dos grupos. O ajuste não é MLE puro por conter regularização
e pseudocontagens explícitas, mantidas do desenho anterior.

Caudas de silêncio que não possuem qualquer excesso positivo observado não são
inventadas. Um átomo puro pode tornar certa idade incompatível com o suporte;
esse cliente recebe status, e não uma previsão baseada em frequências.

## Fronteiras dos tempos
O estado foi construído em um corte exclusivo: eventos no próprio instante do
corte ainda não foram consumidos. Por isso, a aplicação usa P(T >= idade).
Com um átomo, >= e > diferem na fronteira. A janela de sete dias é [a,a+7).
Para componentes contínuos, essa diferença de fronteira tem probabilidade zero.

A validação pareada agora inclui eventos exatos no próprio marco (T>=marco).
Os cenários ANTES e DEPOIS usam as mesmas linhas e essa mesma definição.
NÃO compare diretamente o novo quadro de 30 minutos com o quadro antigo,
que excluía eventos iguais a 1800 segundos.
Não se compara a antiga NLL de densidades com contribuições de massa pontual.

## O que conferir
- Resultado da correção nas 29 origens: quantas recuperadas e motivos restantes.
- Modelos anteriormente AJUSTADO preservados byte a byte.
- Cobertura e top 1/top 5 ANTES/DEPOIS com o mesmo denominador.
- Estado, idade e versão de cada previsão.
- Para o silêncio atual, cujo modelo já estava ajustado, as probabilidades
  devem permanecer compatíveis com as anteriores; não foi reestimado.
- Clientes com entrada ambígua, sem histórico ou idade desconhecida permanecem
  com status e probabilidades nulas.
- Extrapolação temporal continua sendo sinalizada.
- Top 5 não é renormalizado; destinos com probabilidade zero não completam a lista.
- Cobertura dos dados, fuso e calibração seguem sem certificação de produção.

## Views criadas pelo 02D
- nba_sm_v21_modelos
- nba_sm_v21_parametros
- nba_sm_v21_previsoes_top5
- nba_sm_v21_validacao
- nba_sm_v21_configuracao

O conjunto recebe versão derivada da anterior com sufixo `_timeout_v21`.
Os ajustes preservados mantêm o hash de seu JSON.
As tabelas novas opcionais da parte 03 usam sufixo `_v21_hml`.
Nada altera as tabelas de origem nem os outputs permanentes antigos.

## Limite operacional dos checkpoints
O script usa localCheckpoint(eager=True) para materializar as pequenas amostras
de ajuste/validação e evitar novo sorteio a cada resumo.
São checkpoints temporários, não armazenamento confiável de produção.
Se forem perdidos no cluster, a execução precisa ser refeita.

## Testes locais
Na pasta dos arquivos, execute `python testes_timeout_v21.py`.
A suíte usa apenas NumPy, Pandas e SciPy. Extrai as funções puras dos notebooks;
não executa Spark nem consulta fontes do banco.

Executados nesta entrega: 20 testes, todos aprovados, em Python 3.13.5 e
SciPy 1.17.0. Incluem átomo puro, massa + cauda, gradientes, fronteiras,
censura, proibição de imputação, compatibilidade de previsões legadas,
auditoria independente e detecção de corrupção.

A sintaxe dos notebooks foi verificada. Não foi executada a integração no
Databricks. O ambiente do banco (Spark 3.5.2/SciPy 1.13.1) ainda precisa executar
os arquivos. Não instale ou atualize bibliotecas como parte deste teste.

## Referências das APIs
SciPy 1.13.1:
- https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.stats.lognorm.html
- https://docs.scipy.org/doc/scipy-1.13.1/reference/optimize.minimize-lbfgsb.html

As decisões de massa pontual, pooling e penalização são específicas desta
implementação. As referências de APIs não representam validação dos dados do banco.
