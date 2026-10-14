
df = spark.sql("""
WITH main AS (
    -- TESTE ATUAL
    SELECT
        TRIM(CAST(pp.cpf AS STRING)) AS cpf,
        TRIM(CAST(pp.proposta_ultimo_egv_alvo AS STRING))
            AS proposta_ultimo_egv_alvo,
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

    -- PRIMEIRO TESTE
    SELECT
        TRIM(CAST(pp.cpf AS STRING)) AS cpf,
        TRIM(CAST(pp.proposta_ultimo_egv_alvo AS STRING))
            AS proposta_ultimo_egv_alvo,
        pp.TesteDigito,
        CAST(pp.TsInclusao AS TIMESTAMP) AS ts_inclusao,
        'primeiro_teste' AS fase
    FROM ctg_dsti.`renato-egv`.controle_ajustado_primeiros_testes pp
    WHERE pp.score > 0.3919
),

-- Deduplicação antes do JOIN com as aprovações
publico AS (
    SELECT
        cpf,
        proposta_ultimo_egv_alvo,
        TesteDigito,
        ts_inclusao,
        fase
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY cpf
                ORDER BY
                    ts_inclusao ASC,
                    proposta_ultimo_egv_alvo ASC,
                    fase ASC
            ) AS rn
        FROM main
        WHERE cpf IS NOT NULL
            AND ts_inclusao IS NOT NULL
    )
    WHERE rn = 1
),

-- Propostas com status AP
aprovadas AS (
    SELECT DISTINCT
        TRIM(CAST(nucpfcnpj AS STRING)) AS cpf,
        TRIM(CAST(NuProposta AS STRING)) AS proposta_aprovada,
        CAST(DtEntradaProposta AS TIMESTAMP) AS dt_entrada
    FROM propostas_aprovadas
    WHERE nucpfcnpj IS NOT NULL
        AND NuProposta IS NOT NULL
),

-- Identificação das duas formas de conversão
conversoes AS (
    SELECT
        p.cpf,
        p.proposta_ultimo_egv_alvo,
        p.TesteDigito,
        p.ts_inclusao,
        p.fase,

        -- Aprovação da proposta original
        MAX(
            CASE
                WHEN a.proposta_aprovada =
                     p.proposta_ultimo_egv_alvo
                THEN 1
                ELSE 0
            END
        ) AS converteu_mesma_proposta,

        -- Outra proposta criada nos 30 dias seguintes
        MAX(
            CASE
                WHEN (
                    p.proposta_ultimo_egv_alvo IS NULL
                    OR a.proposta_aprovada <>
                       p.proposta_ultimo_egv_alvo
                )
                AND a.dt_entrada > p.ts_inclusao
                AND a.dt_entrada <=
                    p.ts_inclusao + INTERVAL 30 DAYS
                THEN 1
                ELSE 0
            END
        ) AS converteu_nova_proposta,

        -- Primeira nova proposta aprovada encontrada
        MIN(
            CASE
                WHEN (
                    p.proposta_ultimo_egv_alvo IS NULL
                    OR a.proposta_aprovada <>
                       p.proposta_ultimo_egv_alvo
                )
                AND a.dt_entrada > p.ts_inclusao
                AND a.dt_entrada <=
                    p.ts_inclusao + INTERVAL 30 DAYS
                THEN a.dt_entrada
            END
        ) AS dt_primeira_nova_proposta

    FROM publico p
    LEFT JOIN aprovadas a
        ON a.cpf = p.cpf
        AND (
            a.proposta_aprovada =
                p.proposta_ultimo_egv_alvo
            OR (
                a.dt_entrada > p.ts_inclusao
                AND a.dt_entrada <=
                    p.ts_inclusao + INTERVAL 30 DAYS
            )
        )

    GROUP BY
        p.cpf,
        p.proposta_ultimo_egv_alvo,
        p.TesteDigito,
        p.ts_inclusao,
        p.fase
)

SELECT
    cpf,
    proposta_ultimo_egv_alvo,
    TesteDigito,
    DATE(ts_inclusao) AS dtinclusao,
    fase,

    CASE
        WHEN converteu_mesma_proposta = 1
          OR converteu_nova_proposta = 1
        THEN 1
        ELSE 0
    END AS converteu,

    converteu_mesma_proposta,
    converteu_nova_proposta,

    CASE
        WHEN converteu_mesma_proposta = 0
         AND converteu_nova_proposta = 1
        THEN 1
        ELSE 0
    END AS conversao_recuperada,

    dt_primeira_nova_proposta,

    CASE
        WHEN ts_inclusao <=
            CURRENT_TIMESTAMP() - INTERVAL 30 DAYS
        THEN 1
        ELSE 0
    END AS janela_30d_completa

FROM conversoes
""")

display(df)
