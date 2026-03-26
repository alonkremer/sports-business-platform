-- Gold: season-level summary for dashboard KPI strip (Season Overview view)
{{ config(materialized='table') }}

WITH fact AS (
    SELECT * FROM {{ ref('fact_game_pricing') }}
),

game_level AS (
    -- Aggregate to game level first
    SELECT
        game_id,
        season,
        date,
        opponent,
        opponent_tier,
        is_rivalry,
        is_marquee,
        is_baja_cup,
        is_season_opener,
        is_decision_day,
        is_saturday,
        day_of_week,
        demand_index,
        attendance_or_projection,

        -- Revenue totals at game level
        SUM(face_price * (capacity - COALESCE(seats_remaining_t7, capacity * 0.15))) AS est_primary_revenue,
        SUM(total_revenue_opportunity)      AS total_opp_this_game,
        AVG(secondary_premium_pct)          AS avg_secondary_premium,
        AVG(sell_through_t7)                AS avg_sell_through_t7,
        MAX(market_health_computed)         AS worst_market_health,
        MAX(is_hot_market_alert::INTEGER)   AS has_hot_alert,
        MAX(is_cold_market_alert::INTEGER)  AS has_cold_alert,
        MAX(backlash_risk_score)            AS max_backlash_risk,

        -- Count sections by opportunity tier
        COUNT(*) FILTER (WHERE opportunity_tier = 'high')   AS sections_high_opp,
        COUNT(*) FILTER (WHERE opportunity_tier = 'medium') AS sections_med_opp,
        COUNT(*) FILTER (WHERE market_health_computed = 'hot') AS sections_hot

    FROM fact
    GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
),

season_level AS (
    SELECT
        season,
        COUNT(DISTINCT game_id)                     AS total_games,
        SUM(total_opp_this_game)                    AS season_total_opportunity,
        AVG(total_opp_this_game)                    AS avg_opportunity_per_game,
        SUM(est_primary_revenue)                    AS est_season_primary_revenue,
        AVG(avg_secondary_premium)                  AS season_avg_secondary_premium,
        AVG(avg_sell_through_t7)                    AS season_avg_sell_through,
        MAX(max_backlash_risk)                      AS season_max_backlash_risk,
        COUNT(*) FILTER (WHERE has_hot_alert = 1)   AS games_with_hot_alert,
        COUNT(*) FILTER (WHERE has_cold_alert = 1)  AS games_with_cold_alert,
        SUM(sections_high_opp)                      AS total_high_opp_sections,
        SUM(sections_hot)                           AS total_hot_sections
    FROM game_level
    GROUP BY season
)

SELECT
    gl.*,
    sl.season_total_opportunity,
    sl.season_avg_secondary_premium,
    sl.season_avg_sell_through,
    sl.est_season_primary_revenue
FROM game_level gl
JOIN season_level sl ON gl.season = sl.season
ORDER BY gl.season, gl.date
