# NBA / Next Best Action — Evolução da Solução Semi-Markov

## 1. Objetivo do projeto

O objetivo é construir um motor capaz de responder, para cada cliente:

> **Dado o último comportamento observado e o tempo transcorrido desde ele, qual é a próxima ação mais provável?**

Além da próxima ação imediata, o modelo pode propagar a jornada por alguns passos para responder uma pergunta mais útil ao negócio:

> **Qual é a próxima ação relevante mais provável, mesmo que antes dela existam eventos intermediários como login ou navegação?**

O output esperado é um **Top 5 por cliente**, acompanhado de scores/probabilidades e informações de qualidade da previsão.

---

## 2. Evolução da solução

| Versão | Representação do cliente | Motor principal | Principal característica | Status |
|---|---|---|---|---|
| **Base / Legada** | Sequência extensa de estados + silêncio | Markov + modelos de duração | Arquitetura estatística extensa e difícil de auditar | Baseline |
| **V2.1 — MVP** | `estado atual + tempo no estado`, incluindo `sem_acao` | Semi-Markov | Primeira versão end-to-end auditável | Concluída |
| **V2.2 — Atual** | `última ação real + tempo desde a última ação` | Semi-Markov | Remove o colapso provocado pelo estado `sem_acao` | Em desenvolvimento |
| **V2.3 — Próxima** | V2.2 + contexto de calendário | Semi-Markov contextual | Capturar sazonalidade | Planejada |
| **V2.4 — Evolução** | V2.3 + histórico individual | Semi-Markov contextual/personalizado | Capturar hábito e recorrência do cliente | Planejada |

---

## 3. De onde saímos — solução base

A solução inicial já possuía conceitos estatísticos relevantes:

- Matriz de transição entre estados.
- Probabilidades `q(j|i)`.
- Análise de permanência.
- Censura.
- Kaplan-Meier.
- Distribuições Weibull, Lognormal e Exponencial.
- Suavização via Dirichlet / Empirical Bayes.
- Bootstrap e intervalos de confiança.

Fluxo conceitual:

```text
Histórico de jornadas
        ↓
Contagem origem → destino
        ↓
q(j | i)
        ↓
Modelo de duração do estado
        ↓
Potencial / transições futuras
```

### Limitações observadas

Apesar de estatisticamente rica, a implementação apresentava:

- grande quantidade de código customizado;
- dificuldade de auditoria;
- mistura entre componentes de transição e permanência;
- tratamento complexo de censura;
- estados sintéticos;
- dificuldade de chegar diretamente ao output de negócio.

O principal objetivo da reconstrução foi **simplificar a arquitetura sem perder o componente temporal**.

---

## 4. V2.1 — primeiro MVP Semi-Markov funcional

Na V2.1 definimos explicitamente:

\[
P(J=j \mid I=i,T\ge a)
\]

onde:

- `I` = estado atual;
- `J` = próximo estado;
- `a` = tempo que o cliente já permaneceu no estado.

Ou seja:

```text
Estado atual
      +
Tempo de permanência
      ↓
Probabilidade dos próximos estados
```

### Construção da jornada

A V2.1 utilizava:

```text
Evento
 ↓
Estado comportamental
 ↓
30 minutos sem atividade
 ↓
sem_acao:::classe
```

O silêncio passou a ser explicitamente modelado como um estado.

### Problema descoberto

Quando executamos o modelo para milhões de clientes, vimos aproximadamente:

```text
2,59 milhões de clientes
↓
sem_acao:::classe
↓
app_login:::topo
```

Ou seja, o modelo funcionava matematicamente, porém **quase todos os clientes estavam sendo colocados no mesmo contexto**.

Isso fazia com que as previsões fossem pouco diferenciadas.

---

## 5. Motor probabilístico da V2.1

Para cada origem `i`, aprendemos inicialmente:

\[
P(J=j \mid I=i)
\]

A essa probabilidade adicionamos o componente temporal.

Para cada destino/grupo de destinos modelamos uma função de sobrevivência:

\[
S_{ij}(t)=P(T>t \mid I=i,J=j)
\]

Então, sabendo que o cliente já permaneceu `a` unidades de tempo no estado:

\[
P(J=j \mid I=i,T>a)
\propto
P(J=j \mid I=i)\times S_{ij}(a)
\]

Após normalização:

\[
q_j(a)=
\frac{
p_{ij}S_{ij}(a)
}{
\sum_k p_{ik}S_{ik}(a)
}
\]

Esse é o núcleo do nosso Semi-Markov.

---

## 6. Tratamento especial descoberto na V2.1

A regra operacional de 30 minutos produzia grande quantidade de tempos exatamente iguais a:

\[
1800\text{ segundos}
\]

Uma Lognormal contínua sozinha não representava bem essa concentração.

A solução foi:

```text
Massa pontual em 1800 s
+
Lognormal para o excesso
```

Isso permitiu recuperar **27 das 29 origens** que inicialmente falhavam no ajuste temporal.

Após a correção, a cobertura de destinos em um dos principais cenários de validação passou de aproximadamente:

```text
28% → 99%
```

Esse ganho foi principalmente de **cobertura de modelo**, e não de accuracy.

---

## 7. Multi-step — da próxima ação técnica para a próxima ação relevante

Outro aprendizado importante surgiu ao testar um comportamento conhecido.

O modelo previa:

```text
sem_acao
    ↓
app_login:::topo
```

O resultado parecia pouco útil inicialmente.

Porém, ao propagar a cadeia:

```text
sem_acao
    ↓
login
    ↓
navegação
    ↓
...
    ↓
PIX
```

observamos probabilidade relevante de alcançar PIX depois de alguns passos.

Isso mostrou que existem duas perguntas diferentes.

### Próximo estado imediato

\[
P(X_{t+1})
\]

Exemplo:

```text
app_login:::topo
```

### Próxima ação relevante

\[
P(\text{primeira ação relevante em até N transições})
\]

Exemplo:

```text
pagamentos_boleto
PIX
seguro
...
```

Por isso foi criado o motor **multi-step**.

---

## 8. Ações relevantes do MVP

Inicialmente definimos:

```text
pagamentos_boleto:::topo
```

como alvo oficial de validação.

Depois adicionamos, para exploração:

```text
pagamentos_boleto:::topo
pix_cadastro_chave:::topo
pix_trazer_chave:::topo
seguros_auto_avulso_cotacao:::topo
gerar_boleto_financeira:::topo
```

Estados como:

```text
login
navegação
sucesso
```

continuam participando da cadeia, mas não precisam ser apresentados como recomendação final.

---

## 9. O que aprendemos com o MVP V2.1

O modelo passou pela auditoria independente:

```text
13.315 previsões auditadas
0 erros
erro máximo ≈ 2,55 × 10⁻¹⁵
```

Porém, a execução completa revelou um problema de representação:

> **O estado `sem_acao` eliminava grande parte da informação comportamental anterior.**

Dois clientes completamente diferentes podiam se transformar em:

```text
sem_acao + 2 dias
```

mesmo que um tivesse acabado de realizar PIX e outro estivesse vindo de financiamento.

Esse se tornou o principal motivador da V2.2.

---

## 10. V2.2 — última ação real + tempo

A V2.2 elimina completamente:

```text
sem_acao
```

e também elimina:

```text
timeout de 30 minutos como mudança de estado
```

Agora o estado do cliente é:

\[
\boxed{
\text{última ação real}
+
\text{tempo desde a última ação}
}
\]

Exemplo:

```text
ultima_acao:
pix_transferencia:::sucesso

tempo_desde_ultima_acao:
2 dias
```

Outro cliente pode estar:

```text
ultima_acao:
consulta_financiamento:::sucesso

tempo_desde_ultima_acao:
2 dias
```

Eles deixam de ser tratados como equivalentes.

---

## 11. Construção da base V2.2

O pipeline atualmente é:

```text
Eventos brutos
    ↓
Ordenação por cliente e timestamp
    ↓
Resolver múltiplos estados simultâneos
    usando profundidade_max
    ↓
Debounce de eventos técnicos repetidos
    ↓
Sequência de ações reais
    ↓
Tempo até a próxima ação
    ↓
Última ação sem próxima ação
    = censura à direita
```

---

## 12. Resolução de eventos simultâneos

Encontramos inicialmente cerca de:

```text
88 mil timestamps
```

com mais de um estado.

A informação `profundidade_max` permitiu resolver quase todos eles.

A regra ficou:

```text
mesmo cliente
+
mesmo timestamp
+
vários estados
        ↓
seleciona maior profundidade
```

Se existir empate real na maior profundidade, o registro continua marcado como ambíguo.

Após a correção, no público amostral:

```text
OK                       2.663
SEM_HISTORICO            1.655
ESTADO_ATUAL_AMBIGUO        36
```

A ambiguidade caiu drasticamente.

---

## 13. Debounce técnico

Outro problema encontrado foram autotransições extremamente rápidas.

Antes:

```text
112.845 autotransições
```

e:

```text
54% ≤ 5 segundos
90% ≤ 30 segundos
99% ≤ 5 minutos
```

Isso indicava telemetria repetida dentro da mesma interação.

Aplicamos então:

> **Mesmo estado consecutivo em até 30 segundos → mesma ação**

Exemplo:

```text
10:00:00 PIX navegação
10:00:03 PIX navegação
10:00:08 PIX navegação
```

vira:

```text
10:00:08 PIX navegação
```

Após o debounce:

```text
112.845 → 11.332 autotransições
```

Nenhuma das autotransições restantes ocorre em até 30 segundos.

---

## 14. Estado atual da base V2.2

Após limpeza:

```text
483.009 ações no treino
2.620 clientes
223 estados
11.332 autotransições
```

No público:

```text
OK                       2.680
SEM_HISTORICO            1.651
ESTADO_ATUAL_AMBIGUO        23
```

A distribuição das últimas ações agora é diversa:

```text
consulta_financiamento:::sucesso
consulta_cartao_credito:::sucesso
app_login:::sucesso
gerar_boleto_financeira:::sucesso
consulta_extrato_conta:::sucesso
atendimento:::whatsapp
pix_transferencia:::sucesso
pix_pagamento:::sucesso
...
```

Esse é exatamente o comportamento que queríamos obter.

---

## 15. Distribuição temporal atual

Após debounce, as durações exatas ficaram aproximadamente:

| Quantil | Tempo |
|---|---:|
| P50 | ~10 segundos |
| P90 | ~3,7 horas |
| P95 | ~30 horas |
| P99 | ~11,5 dias |

Essa distribuição representa melhor o tempo real entre ações do cliente.

---

## 16. Motor probabilístico da V2.2

A lógica permanece Semi-Markov:

\[
P(J=j \mid I=i,T>a)
\]

Porém agora:

\[
I = \text{última ação real}
\]

e:

\[
a = \text{tempo desde essa ação}
\]

Então:

\[
\boxed{
P(
\text{próxima ação}
\mid
\text{última ação},
\text{tempo desde ela}
)
}
\]

Para cada origem:

1. estimamos a frequência dos próximos destinos;
2. agrupamos destinos raros quando necessário;
3. modelamos a duração com Lognormal;
4. incorporamos censura à direita;
5. recalculamos a distribuição dos destinos conforme o tempo transcorrido.

---

## 17. Principais motores utilizados

### Motor 1 — Engenharia de jornadas

Tecnologias:

```text
Databricks
Spark
PySpark
SQL
Window Functions
```

Responsável por:

- ordenação temporal;
- construção das jornadas;
- resolução de timestamps simultâneos;
- debounce;
- duração;
- censura;
- definição do estado atual.

### Motor 2 — Probabilidade de transição

Aprende:

\[
P(J=j \mid I=i)
\]

a partir das transições históricas.

Estados/destinos com baixo suporte podem ser agrupados para evitar modelos excessivamente instáveis.

### Motor 3 — Sobrevivência / permanência

Modela:

\[
T_{ij}
\]

o tempo necessário para sair de `i` em direção a `j`.

Família principal da V2.2:

```text
Lognormal
```

A sobrevivência informa:

\[
S_{ij}(t)=P(T_{ij}>t)
\]

### Motor 4 — Semi-Markov

Combina:

```text
probabilidade do destino
+
tempo de permanência
```

para obter:

\[
P(J=j \mid I=i,T>a)
\]

Esse é o principal motor de inferência.

### Motor 5 — Multi-step

Propaga as probabilidades:

```text
ação atual
 ↓
passo 1
 ↓
passo 2
 ↓
passo 3
 ...
```

até encontrar uma ação relevante.

Permite ignorar no output etapas técnicas como:

```text
login
navegação
```

sem removê-las da cadeia.

---

## 18. Onde estamos agora

Estamos treinando a primeira versão da **V2.2**.

Ela está sendo comparada à V2.1 usando:

- amostra equivalente;
- holdout por cliente;
- Top-1;
- Top-5;
- cobertura;
- comportamento da probabilidade ao longo do tempo.

Nenhuma nova feature contextual foi adicionada ainda.

Assim conseguiremos medir isoladamente:

> **Quanto ganhamos simplesmente ao substituir `sem_acao` por `última ação + tempo`?**

---

## 19. V2.3 — contexto temporal

Depois de validar a V2.2, adicionaremos informações de calendário:

```text
dia_do_mes
dias_para_fim_mes
dia_da_semana
eh_fim_de_semana
```

O objetivo é capturar padrões como:

> **A probabilidade de determinadas ações aumenta no fim do mês.**

Passamos então de:

\[
P(J\mid I,T)
\]

para algo mais próximo de:

\[
P(J\mid I,T,X_{calendario})
\]

---

## 20. V2.4 — histórico individual

A próxima evolução adicionará comportamento do próprio cliente.

### PIX

```text
dias_desde_ultimo_pix
qtd_pix_7d
qtd_pix_30d
qtd_pix_90d
```

### Boleto

```text
dias_desde_ultimo_boleto
qtd_boleto_7d
qtd_boleto_30d
qtd_boleto_90d
```

### Uso geral

```text
dias_desde_ultima_atividade
qtd_eventos_7d
qtd_eventos_30d
```

A formulação passa a ser:

\[
\boxed{
P(
J
\mid
estado,
tempo,
calendário,
histórico\ individual
)
}
\]

Essa é a direção para um NBA realmente personalizado.

---

## 21. Roadmap

```text
BASE LEGADA
Markov + duração + potencial
        ↓
V2.1
Semi-Markov auditável
estado + tempo
        ↓
Problema identificado:
sem_acao domina a população
        ↓
V2.2
última ação real + tempo
        ↓
V2.3
+ contexto de calendário
        ↓
V2.4
+ comportamento individual
        ↓
NBA contextual / personalizado
        ↓
Policy Learning / Contextual Bandits
```

---

## 22. Resultado esperado

O objetivo final deixa de ser algo genérico como:

```text
Cliente está em silêncio
→ provavelmente fará login
```

e passa a ser:

```text
Última ação:
PIX transferência

Tempo:
27 dias

Calendário:
2 dias para o fim do mês

Histórico:
PIX recorrente próximo ao fechamento

        ↓

Top 5 ações relevantes

1. PIX transferência       score alto
2. Pagamento de boleto     score ...
3. ...
```

A evolução principal do projeto é:

> **Sair de um modelo populacional de transições para um motor contextual que combina jornada, tempo e comportamento individual para antecipar a próxima ação relevante do cliente.**
