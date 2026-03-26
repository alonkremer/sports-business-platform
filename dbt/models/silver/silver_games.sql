-- Silver: cleaned and enriched game dimension
-- Adds derived fields, validates data quality, standardizes types.
{{ config(materialized='table') }}

WITH base AS (
    SELECT * FROM {{ ref('bronze_games') }}
),

enriched AS (
    SELECT
        game_id,
        season,
        date,
        day_of_week,
        opponent,
        -- Standardize opponent name (handle "TBD" for decision day)
        CASE
            WHEN opponent LIKE '%Decision Day%' OR opponent = 'TBD' THEN 'TBD'
            ELSE opponent
        END AS opponent_clean,
        opponent_tier,
        is_rivalry,
        is_marquee,
        is_baja_cup,
        is_season_opener,
        is_decision_day,
        is_saturday,
        is_wednesday,
        -- Derived flags
        CASE WHEN day_of_week = 'Sunday' THEN TRUE ELSE FALSE END AS is_sunday,
        CASE WHEN day_of_week IN ('Saturday', 'Sunday') THEN TRUE ELSE FALSE END AS is_weekend,
        -- Star player on opponent
        CASE
            WHEN opponent IN ('Inter Miami', 'LA Galaxy') THEN TRUE
            ELSE FALSE
        END AS star_player_on_opponent,
        -- Cross-border demand flag
        CASE WHEN is_baja_cup = TRUE THEN TRUE ELSE FALSE END AS is_cross_border_event,
        home_team,
        venue,
        result,
        goals_for,
        goals_against,
        CASE
            WHEN goals_for IS NOT NULL AND goals_against IS NOT NULL
            THEN goals_for - goals_against
            ELSE NULL
        END AS goal_diff,
        home_xg,
        away_xg,
        -- Use actual attendance if available, else projected
        COALESCE(attendance, projected_attendance) AS attendance_or_projection,
        attendance,
        projected_attendance,
        capacity,
        sell_through_pct,
        demand_index,
        home_win_prob,
        -- Implied odds category
        CASE
            WHEN home_win_prob >= 0.55 THEN 'heavy_favorite'
            WHEN home_win_prob >= 0.48 THEN 'slight_favorite'
            WHEN home_win_prob >= 0.40 THEN 'slight_underdog'
            ELSE 'underdog'
        END AS home_odds_category,
        temp_f,
        rain_prob,
        -- Weather flags
        CASE WHEN rain_prob > 0.30 THEN TRUE ELSE FALSE END AS is_high_rain_risk,
        CASE WHEN temp_f < 55 OR temp_f > 90 THEN TRUE ELSE FALSE END AS is_extreme_temp,
        -- Season half
        CASE WHEN MONTH(date) <= 6 THEN 'first_half' ELSE 'second_half' END AS season_half,
        -- FIFA World Cup pause adjacent (May 25 – Jul 16, 2026)
        CASE
            WHEN season = 2026 AND date BETWEEN '2026-05-11' AND '2026-05-24' THEN TRUE
            WHEN season = 2026 AND date BETWEEN '2026-07-17' AND '2026-07-30' THEN TRUE
            ELSE FALSE
        END AS is_fifa_adjacent,
        ingested_at
    FROM base
    WHERE game_id IS NOT NULL
      AND date IS NOT NULL
)

SELECT * FROM enriched
ORDER BY season, date
