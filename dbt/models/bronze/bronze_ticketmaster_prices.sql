-- Bronze: raw Ticketmaster face-value prices (synthetic + any live scrape)
{{ config(materialized='view') }}

SELECT
    game_id,
    season::INTEGER         AS season,
    date::DATE              AS date,
    opponent,
    section,
    tier,
    capacity::INTEGER       AS capacity,
    list_price::DOUBLE      AS list_price,
    base_price::DOUBLE      AS base_price,
    demand_multiplier::DOUBLE AS demand_multiplier,
    data_source,
    CURRENT_TIMESTAMP       AS ingested_at
FROM read_csv_auto('{{ env_var("DATA_RAW_DIR", "data/raw") }}/ticketmaster_prices.csv')
