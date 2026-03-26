-- Bronze: MLS league-wide attendance history (2023-2025 verified)
{{ config(materialized='view') }}

SELECT
    season::INTEGER         AS season,
    team,
    avg_attendance::INTEGER AS avg_attendance,
    home_games::INTEGER     AS home_games,
    mls_rank::INTEGER       AS mls_rank,
    total_attendance::INTEGER AS total_attendance,
    CURRENT_TIMESTAMP       AS ingested_at
FROM read_csv_auto('{{ env_var("DATA_RAW_DIR", "data/raw") }}/mls_attendance_history.csv')
