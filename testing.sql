
%sql
WITH publico AS (
    SELECT DISTINCT
        cpf,
        canais,
        TipoControle
    FROM ctg_dsti.`renato-egv`.predicoes_teste_pipeline_schedule
    WHERE TipoControle IN (
        'teste_proporcional',
        'controle_proporcional'
    )
      AND cpf IS NOT NULL
),

aprovados AS (
    SELECT DISTINCT
        nucpfcnpj AS cpf
    FROM propostas_aprovadas
    WHERE nucpfcnpj IS NOT NULL
),

base AS (
    SELECT
        p.canais,
        p.TipoControle,
        CASE
            WHEN a.cpf IS NOT NULL THEN 1
            ELSE 0
        END AS converteu
    FROM publico p
    LEFT JOIN aprovados a
        ON p.cpf = a.cpf
),

contagem AS (
    SELECT
        canais,
        COUNT_IF(
            TipoControle = 'teste_proporcional'
        ) AS total_teste,
        COUNT_IF(
            TipoControle = 'controle_proporcional'
        ) AS total_controle,
        COUNT_IF(
            TipoControle = 'teste_proporcional'
            AND converteu = 1
        ) AS aprovados_teste,
        COUNT_IF(
            TipoControle = 'controle_proporcional'
            AND converteu = 1
        ) AS aprovados_controle
    FROM base
    GROUP BY canais
),

taxas AS (
    SELECT
        *,
        aprovados_teste / NULLIF(total_teste, 0)
            AS taxa_teste,
        aprovados_controle / NULLIF(total_controle, 0)
            AS taxa_controle
    FROM contagem
)

SELECT
    canais,
    total_teste,
    aprovados_teste,
    ROUND(taxa_teste * 100, 2) AS taxa_teste_pct,
    total_controle,
    aprovados_controle,
    ROUND(taxa_controle * 100, 2) AS taxa_controle_pct,

    ROUND(
        (taxa_teste - taxa_controle) * 100, 2
    ) AS delta_pp,

    ROUND(
        (taxa_teste / NULLIF(taxa_controle, 0) - 1) * 100,
        2
    ) AS lift_relativo_pct,

    ROUND(
        total_controle * (taxa_teste - taxa_controle),
        2
    ) AS incremento_projetado_controle

FROM taxas
ORDER BY delta_pp ASC;
