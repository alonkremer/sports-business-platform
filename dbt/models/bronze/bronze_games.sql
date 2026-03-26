-- Bronze: raw game data from synthetic generator (2025 verified + 2026 projected)
-- This is a view over the CSV file — no transformations, just schema exposure.
{{ config(materialized='view') }}

SELECT
    game_id,
    season::INTEGER                    AS season,
    date::DATE                         AS date,
    day_of_week,
    opponent,
    opponent_tier::INTEGER             AS opponent_tier,
    is_rivalry::BOOLEAN                AS is_rivalry,
    is_marquee::BOOLEAN                AS is_marquee,
    is_baja_cup::BOOLEAN               AS is_baja_cup,
    is_season_opener::BOOLEAN          AS is_season_opener,
    is_decision_day::BOOLEAN           AS is_decision_day,
    is_saturday::BOOLEAN               AS is_saturday,
    is_wednesday::BOOLEAN              AS is_wednesday,
    home_team,
    venue,
    result,
    goals_for::INTEGER                 AS goals_for,
    goals_against::INTEGER             AS goals_against,
    home_xg::DOUBLE                    AS home_xg,
    away_xg::DOUBLE                    AS away_xg,
    attendance::INTEGER                AS attendance,
    projected_attendance::INTEGER      AS projected_attendance,
    capacity::INTEGER                  AS capacity,
    sell_through_pct::DOUBLE           AS sell_through_pct,
    demand_index::DOUBLE               AS demand_index,
    home_win_prob::DOUBLE              AS home_win_prob,
    temp_f::DOUBLE                     AS temp_f,
    rain_prob::DOUBLE                  AS rain_prob,
    CURRENT_TIMESTAMP                  AS ingested_at
FROM read_csv_auto('{{ env_var("DATA_RAW_DIR", "data/raw") }}/synthetic_games.csv')
