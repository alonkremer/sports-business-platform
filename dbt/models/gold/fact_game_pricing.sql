-- Gold: fact_game_pricing — primary table for ML pipeline and dashboard
-- One row per game × section. Joins game context, pricing, inventory, and secondary market.
-- This is the single source of truth for the pricing recommendation engine.
{{ config(materialized='table') }}

WITH games AS (
    SELECT * FROM {{ ref('silver_games') }}
),

pricing AS (
    SELECT * FROM {{ ref('silver_pricing') }}
),

inventory_t7 AS (
    -- Most actionable snapshot: T-7 days (used for real-time pricing decisions)
    SELECT * FROM {{ ref('silver_inventory') }}
    WHERE days_before_game = 7
),

inventory_t30 AS (
    -- Early signal: T-30 days (for advance pricing strategy)
    SELECT * FROM {{ ref('silver_inventory') }}
    WHERE days_before_game = 30
),

final AS (
    SELECT
        -- ── Primary keys ──────────────────────────────────────────────────
        p.game_id,
        p.section,
        p.season,

        -- ── Game context ──────────────────────────────────────────────────
        g.date,
        g.day_of_week,
        g.opponent,
        g.opponent_clean,
        g.opponent_tier,
        g.is_rivalry,
        g.is_marquee,
        g.is_baja_cup,
        g.is_season_opener,
        g.is_decision_day,
        g.is_saturday,
        g.is_wednesday,
        g.is_sunday,
        g.is_weekend,
        g.is_fifa_adjacent,
        g.star_player_on_opponent,
        g.is_cross_border_event,
        g.season_half,
        g.home_odds_category,
        g.home_win_prob,
        g.home_xg,
        g.away_xg,
        g.temp_f,
        g.rain_prob,
        g.is_high_rain_risk,
        g.is_extreme_temp,
        g.result,
        g.goals_for,
        g.goals_against,
        g.goal_diff,
        g.attendance_or_projection,
        g.demand_index,

        -- ── Section / tier ────────────────────────────────────────────────
        p.tier,
        p.capacity,

        -- ── Face-value pricing ────────────────────────────────────────────
        p.face_price,
        p.face_base_price,
        p.demand_multiplier,
        p.face_data_source,

        -- ── Secondary market ──────────────────────────────────────────────
        p.sold_price_avg,                  -- PRIMARY demand truth
        p.sold_price_p25,
        p.sold_price_p75,
        p.listing_price_avg,
        p.secondary_premium_pct,
        p.market_health,
        p.market_health_computed,
        p.n_sold_transactions_est,
        p.n_active_listings,

        -- ── Price gap analysis ────────────────────────────────────────────
        p.price_gap_sold_vs_face,
        p.price_gap_pct,
        p.sth_resale_margin,
        p.optimal_price_increase,
        p.revenue_opp_per_seat,
        p.opportunity_tier,

        -- ── Inventory (T-7 snapshot) ──────────────────────────────────────
        i7.seats_sold                      AS seats_sold_t7,
        i7.seats_remaining                 AS seats_remaining_t7,
        i7.sell_through_pct                AS sell_through_t7,
        i7.velocity_per_day                AS velocity_t7,
        i7.velocity_category               AS velocity_cat_t7,
        i7.projected_final_sell_through    AS proj_sell_through_t7,
        i7.is_soldout                      AS is_soldout_t7,

        -- ── Inventory (T-30 snapshot) ─────────────────────────────────────
        i30.seats_sold                     AS seats_sold_t30,
        i30.sell_through_pct               AS sell_through_t30,
        i30.velocity_per_day               AS velocity_t30,

        -- ── Derived revenue opportunity ───────────────────────────────────
        -- Total revenue opportunity = per-seat gap × seats remaining
        COALESCE(p.revenue_opp_per_seat, 0)
            * COALESCE(i7.seats_remaining, p.capacity * 0.15) AS total_revenue_opportunity,

        -- Alert flags
        CASE
            WHEN p.secondary_premium_pct > 30 THEN TRUE ELSE FALSE
        END AS is_hot_market_alert,
        CASE
            WHEN i7.velocity_category = 'stalled'
            AND i7.sell_through_pct < 50
            THEN TRUE ELSE FALSE
        END AS is_cold_market_alert,
        CASE
            WHEN p.optimal_price_increase > 25 THEN TRUE ELSE FALSE
        END AS is_backlash_risk,

        -- Backlash risk score (0–10)
        LEAST(10, GREATEST(0,
            CASE WHEN p.optimal_price_increase > 25 THEN 4 ELSE 0 END +
            CASE WHEN g.is_rivalry THEN 2 ELSE 0 END +
            CASE WHEN g.is_marquee THEN 1 ELSE 0 END +
            CASE WHEN g.opponent_tier = 1 THEN 1 ELSE 0 END +
            CASE WHEN p.secondary_premium_pct > 50 THEN 2 ELSE 0 END
        )) AS backlash_risk_score

    FROM pricing p
    LEFT JOIN games g
        ON p.game_id = g.game_id
    LEFT JOIN inventory_t7 i7
        ON p.game_id = i7.game_id
        AND p.section = i7.section
    LEFT JOIN inventory_t30 i30
        ON p.game_id = i30.game_id
        AND p.section = i30.section
)

SELECT * FROM final
ORDER BY season, date, tier, section
