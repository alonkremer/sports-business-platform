-- Bronze: raw secondary market signals (consolidated across SeatGeek/StubHub/Vivid Seats)
-- sold_price_avg is the PRIMARY demand truth signal.
{{ config(materialized='view') }}

SELECT
    game_id,
    season::INTEGER              AS season,
    date::DATE                   AS date,
    opponent,
    section,
    tier,
    face_price::DOUBLE           AS face_price,
    sold_price_avg::DOUBLE       AS sold_price_avg,       -- PRIMARY: completed transactions
    sold_price_p25::DOUBLE       AS sold_price_p25,
    sold_price_p75::DOUBLE       AS sold_price_p75,
    listing_price_avg::DOUBLE    AS listing_price_avg,    -- supplementary: asking prices
    listing_price_min::DOUBLE    AS listing_price_min,
    secondary_premium_pct::DOUBLE AS secondary_premium_pct,
    market_health,
    revenue_opp_per_seat::DOUBLE AS revenue_opp_per_seat,
    n_sold_transactions_est::INTEGER AS n_sold_transactions_est,
    n_active_listings::INTEGER   AS n_active_listings,
    data_source,
    CURRENT_TIMESTAMP            AS ingested_at
FROM read_csv_auto('{{ env_var("DATA_RAW_DIR", "data/raw") }}/synthetic_secondary_market.csv')
