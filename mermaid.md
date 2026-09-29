flowchart LR

    A[1. Histórico<br/>Interações e jornadas dos clientes]

    --> B[2. Aprendizado<br/>Identifica padrões de transição<br/>e tempo entre ações]

    --> C[3. Modelo Semi-Markov<br/>Combina estado atual<br/>e tempo de permanência]

    --> D[4. Execução diária<br/>Aplica o modelo aos clientes<br/>selecionados em D-1]

    --> E[5. Recomendação<br/>Top 5 próximas ações<br/>com probabilidade]





flowchart LR

    A[Histórico de comportamento]
    --> B[Jornadas e estados]

    B --> C[Modelo Semi-Markov]

    C --> C1[Probabilidade<br/>da próxima ação]
    C --> C2[Tempo entre<br/>as ações]

    C1 --> D[Estado atual +<br/>tempo de permanência]
    C2 --> D

    D --> E[Probabilidades atualizadas]

    E --> F[Top 5 próximas ações]

    F --> G[Cliente A<br/>1. Ação X — 0.70<br/>2. Ação Y — 0.15<br/>3. Ação Z — 0.08]