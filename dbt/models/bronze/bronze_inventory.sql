-- Bronze: multi-snapshot seat inventory (T-30/14/7/3/1 days before game)
{{ config(materialized='view') }}

SELECT
    game_id,
    season::INTEGER             AS season,
    date::DATE                  AS date,
    opponent,
    section,
    tier,
    capacity::INTEGER           AS capacity,
    days_before_game::INTEGER   AS days_before_game,
    seats_sold::INTEGER         AS seats_sold,
    seats_remaining::INTEGER    AS seats_remaining,
    sell_through_pct::DOUBLE    AS sell_through_pct,
    velocity_per_day::DOUBLE    AS velocity_per_day,
    is_soldout::BOOLEAN         AS is_soldout,
    CURRENT_TIMESTAMP           AS ingested_at
FROM read_csv_auto('{{ env_var("DATA_RAW_DIR", "data/raw") }}/synthetic_inventory.csv')
