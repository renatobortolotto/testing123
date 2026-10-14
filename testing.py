
df = spark.sql("""
WITH main AS (
    SELECT
        pp.cpf,
        TRY_CAST(
            pp.proposta_ultimo_egv_alvo AS BIGINT
        ) AS proposta_ultimo_egv_alvo,
        pp.TesteDigito,
        CAST(pp.TsInclusao AS TIMESTAMP) AS ts_inclusao,
        DATE(pp.TsInclusao) AS dtinclusao
    FROM ctg_dsti.`renato-egv`.predicoes_teste_pipeline_schedule pp
    WHERE pp.TipoControle IN (
        'teste_proporcional',
        'controle_proporcional'
    )
    AND pp.canais NOT IN ('Canal de Negócios VI')

    UNION ALL

    SELECT
        pp.cpf,
        TRY_CAST(
            pp.proposta_ultimo_egv_alvo AS BIGINT
        ) AS proposta_ultimo_egv_alvo,
        pp.TesteDigito,
        CAST(pp.TsInclusao AS TIMESTAMP) AS ts_inclusao,
        DATE(pp.TsInclusao) AS dtinclusao
    FROM ctg_dsti.`renato-egv`.controle_ajustado_primeiros_testes pp
    WHERE pp.score > 0.3919
),

base AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY cpf
            ORDER BY ts_inclusao ASC,
                     proposta_ultimo_egv_alvo ASC
        ) AS rn
    FROM main
),

publico AS (
    SELECT *
    FROM base
    WHERE rn = 1
),

aprovadas AS (
    SELECT DISTINCT
        nucpfcnpj AS cpf,
        TRY_CAST(NuProposta AS BIGINT) AS proposta_aprovada,
        CAST(DtEntradaProposta AS TIMESTAMP) AS dt_entrada
    FROM propostas_aprovadas
),

resultado AS (
    SELECT
        p.cpf,
        p.proposta_ultimo_egv_alvo,
        p.TesteDigito,
        p.dtinclusao,

        MAX(
            CASE
                WHEN a.proposta_aprovada =
                     p.proposta_ultimo_egv_alvo
                THEN 1
                ELSE 0
            END
        ) AS converteu_mesma_proposta,

        MAX(
            CASE
                WHEN a.proposta_aprovada <>
                     p.proposta_ultimo_egv_alvo
                 AND a.dt_entrada > p.ts_inclusao
                THEN 1
                ELSE 0
            END
        ) AS converteu_nova_proposta,

        MIN(
            CASE
                WHEN a.proposta_aprovada <>
                     p.proposta_ultimo_egv_alvo
                 AND a.dt_entrada > p.ts_inclusao
                THEN a.dt_entrada
            END
        ) AS dt_primeira_nova_proposta

    FROM publico p
    LEFT JOIN aprovadas a
        ON a.cpf = p.cpf
    GROUP BY
        p.cpf,
        p.proposta_ultimo_egv_alvo,
        p.TesteDigito,
        p.dtinclusao
)

SELECT
    CASE
        WHEN converteu_mesma_proposta = 1
          OR converteu_nova_proposta = 1
        THEN 1
        ELSE 0
    END AS converteu,
    cpf,
    proposta_ultimo_egv_alvo,
    TesteDigito,
    dtinclusao,
    converteu_mesma_proposta,
    converteu_nova_proposta,
    dt_primeira_nova_proposta
FROM resultado
""")

display(df)
