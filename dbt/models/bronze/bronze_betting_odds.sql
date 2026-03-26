-- Bronze: MLS historical betting odds (OddsPortal / synthetic fallback, 2020-2025)
{{ config(materialized='view') }}

SELECT
    season::INTEGER         AS season,
    date::DATE              AS date,
    home_team,
    away_team,
    home_ml::INTEGER        AS home_ml,
    draw_ml::INTEGER        AS draw_ml,
    away_ml::INTEGER        AS away_ml,
    ou_line::DOUBLE         AS ou_line,
    home_win_prob::DOUBLE   AS home_win_prob,
    draw_prob::DOUBLE       AS draw_prob,
    away_win_prob::DOUBLE   AS away_win_prob,
    data_source,
    CURRENT_TIMESTAMP       AS ingested_at
FROM read_csv_auto('{{ env_var("DATA_RAW_DIR", "data/raw") }}/oddsportal_mls_odds.csv')
