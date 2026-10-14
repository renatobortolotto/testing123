
df = spark.sql("""
WITH main AS (
    -- Teste atual
    SELECT
        REGEXP_REPLACE(
            TRIM(CAST(pp.cpf AS STRING)),
            '[^0-9]', ''
        ) AS cpf,

        COALESCE(
            CAST(
                TRY_CAST(
                    pp.proposta_ultimo_egv_alvo
                    AS DECIMAL(38, 0)
                ) AS STRING
            ),
            TRIM(CAST(
                pp.proposta_ultimo_egv_alvo AS STRING
            ))
        ) AS proposta_alvo,

        pp.TesteDigito,
        CAST(pp.TsInclusao AS TIMESTAMP) AS ts_inclusao,
        'atual' AS fase

    FROM ctg_dsti.`renato-egv`.predicoes_teste_pipeline_schedule pp
    WHERE pp.TipoControle IN (
        'teste_proporcional',
        'controle_proporcional'
    )
    AND pp.canais NOT IN ('Canal de Negócios VI')

    UNION ALL

    -- Primeiro teste
    SELECT
        REGEXP_REPLACE(
            TRIM(CAST(pp.cpf AS STRING)),
            '[^0-9]', ''
        ) AS cpf,

        COALESCE(
            CAST(
                TRY_CAST(
                    pp.proposta_ultimo_egv_alvo
                    AS DECIMAL(38, 0)
                ) AS STRING
            ),
            TRIM(CAST(
                pp.proposta_ultimo_egv_alvo AS STRING
            ))
        ) AS proposta_alvo,

        pp.TesteDigito,
        CAST(pp.TsInclusao AS TIMESTAMP) AS ts_inclusao,
        'primeiro_teste' AS fase

    FROM ctg_dsti.`renato-egv`.controle_ajustado_primeiros_testes pp
    WHERE pp.score > 0.3919
),

publico AS (
    SELECT DISTINCT *
    FROM main
    WHERE cpf IS NOT NULL
      AND cpf <> ''
      AND ts_inclusao IS NOT NULL
),

aprovadas AS (
    SELECT DISTINCT
        REGEXP_REPLACE(
            TRIM(CAST(nucpfcnpj AS STRING)),
            '[^0-9]', ''
        ) AS cpf,

        COALESCE(
            CAST(
                TRY_CAST(
                    NuProposta AS DECIMAL(38, 0)
                ) AS STRING
            ),
            TRIM(CAST(NuProposta AS STRING))
        ) AS proposta_aprovada,

        CAST(DtEntradaProposta AS TIMESTAMP) AS dt_entrada

    FROM propostas_aprovadas
    WHERE NuProposta IS NOT NULL
),

-- Identifica conversões ANTES de consolidar clientes
por_avaliacao AS (
    SELECT
        p.cpf,
        p.fase,
        p.TesteDigito,
        p.ts_inclusao,
        p.proposta_alvo,

        -- Verificação auxiliar
        MAX(
            CASE
                WHEN a.cpf IS NOT NULL THEN 1
                ELSE 0
            END
        ) AS tem_aprovacao_cpf,

        -- Mesma proposta: SEM restrição temporal
        MAX(
            CASE
                WHEN a.proposta_aprovada = p.proposta_alvo
                THEN 1
                ELSE 0
            END
        ) AS converteu_mesma_proposta,

        -- Proposta diferente: somente próximos 30 dias
        MAX(
            CASE
                WHEN (
                    p.proposta_alvo IS NULL
                    OR a.proposta_aprovada <> p.proposta_alvo
                )
                AND a.dt_entrada > p.ts_inclusao
                AND a.dt_entrada <=
                    p.ts_inclusao + INTERVAL 30 DAYS
                THEN 1
                ELSE 0
            END
        ) AS converteu_nova_proposta

    FROM publico p
    LEFT JOIN aprovadas a
        ON a.cpf = p.cpf

    GROUP BY
        p.cpf,
        p.fase,
        p.TesteDigito,
        p.ts_inclusao,
        p.proposta_alvo
),

-- Consolidação após identificação das conversões
por_cliente AS (
    SELECT
        cpf,
        fase,
        TesteDigito,
        MIN(ts_inclusao) AS ts_inclusao,
        COUNT(*) AS qtd_avaliacoes,
        MAX(tem_aprovacao_cpf) AS tem_aprovacao_cpf,
        MAX(converteu_mesma_proposta)
            AS converteu_mesma_proposta,
        MAX(converteu_nova_proposta)
            AS converteu_nova_proposta

    FROM por_avaliacao
    GROUP BY cpf, fase, TesteDigito
)

SELECT
    cpf,
    fase,
    TesteDigito,
    DATE(ts_inclusao) AS dtinclusao,
    qtd_avaliacoes,

    CASE
        WHEN converteu_mesma_proposta = 1
          OR converteu_nova_proposta = 1
        THEN 1
        ELSE 0
    END AS converteu,

    converteu_mesma_proposta,
    converteu_nova_proposta,
    tem_aprovacao_cpf,

    CASE
        WHEN converteu_mesma_proposta = 0
         AND converteu_nova_proposta = 1
        THEN 1
        ELSE 0
    END AS conversao_recuperada

FROM por_cliente
""")

display(df)


from pyspark.sql import functions as F

cpf_teste = "CPF_DO_CLIENTE"

display(
    df.filter(F.col("cpf") == cpf_teste).select(
        "cpf",
        "fase",
        "qtd_avaliacoes",
        "tem_aprovacao_cpf",
        "converteu_mesma_proposta",
        "converteu_nova_proposta",
        "converteu"
    )
)