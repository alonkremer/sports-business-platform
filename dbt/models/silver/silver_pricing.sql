-- Silver: joined pricing table (face value + secondary market per section × game)
-- This is the core pricing fact table used by the ML pipeline.
{{ config(materialized='table') }}

WITH face AS (
    SELECT * FROM {{ ref('bronze_ticketmaster_prices') }}
),

secondary AS (
    SELECT * FROM {{ ref('bronze_secondary_market') }}
),

joined AS (
    SELECT
        f.game_id,
        f.season,
        f.date,
        f.opponent,
        f.section,
        f.tier,
        f.capacity,
        f.list_price                                AS face_price,
        f.base_price                                AS face_base_price,
        f.demand_multiplier,
        f.data_source                               AS face_data_source,

        -- Secondary market (primary demand truth)
        s.sold_price_avg,
        s.sold_price_p25,
        s.sold_price_p75,
        s.listing_price_avg,
        s.listing_price_min,
        s.secondary_premium_pct,
        s.market_health,
        s.revenue_opp_per_seat,
        s.n_sold_transactions_est,
        s.n_active_listings,
        s.data_source                               AS secondary_data_source,

        -- Price gap calculations
        s.sold_price_avg - f.list_price             AS price_gap_sold_vs_face,
        (s.sold_price_avg - f.list_price)
            / NULLIF(f.list_price, 0) * 100         AS price_gap_pct,

        -- STH resale margin: secondary sold − face − 10% healthy buffer
        s.sold_price_avg - (f.list_price * 1.10)   AS sth_resale_margin,

        -- Opportunity: how much can face be raised to capture value
        -- while keeping secondary at 10% premium (healthy band)?
        -- Optimal target = sold_price_avg / 1.10 (secondary stays at 10% above new face)
        s.sold_price_avg / 1.10 - f.list_price     AS optimal_price_increase,

        -- Market health category for alert system
        CASE
            WHEN s.secondary_premium_pct < 10 THEN 'cold'
            WHEN s.secondary_premium_pct <= 20 THEN 'healthy'
            WHEN s.secondary_premium_pct <= 35 THEN 'warm'
            ELSE 'hot'
        END AS market_health_computed,

        -- Revenue opportunity classification
        CASE
            WHEN s.sold_price_avg / 1.10 - f.list_price > 20 THEN 'high'
            WHEN s.sold_price_avg / 1.10 - f.list_price > 8  THEN 'medium'
            WHEN s.sold_price_avg / 1.10 - f.list_price > 0  THEN 'low'
            ELSE 'none'
        END AS opportunity_tier

    FROM face f
    LEFT JOIN secondary s
        ON f.game_id = s.game_id
        AND f.section = s.section
)

SELECT * FROM joined
WHERE game_id IS NOT NULL
ORDER BY season, date, tier, section
