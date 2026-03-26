-- Silver: cleaned inventory snapshots with velocity and sell-through features
{{ config(materialized='table') }}

WITH base AS (
    SELECT * FROM {{ ref('bronze_inventory') }}
),

with_velocity AS (
    SELECT
        *,
        -- Sell-through velocity: seats sold per day in this snapshot window
        -- Computed by comparing to previous snapshot for same game+section
        LAG(seats_sold) OVER (
            PARTITION BY game_id, section
            ORDER BY days_before_game DESC
        ) AS prev_seats_sold,
        LAG(days_before_game) OVER (
            PARTITION BY game_id, section
            ORDER BY days_before_game DESC
        ) AS prev_days_before
    FROM base
),

final AS (
    SELECT
        game_id,
        season,
        date,
        opponent,
        section,
        tier,
        capacity,
        days_before_game,
        seats_sold,
        seats_remaining,
        sell_through_pct,
        velocity_per_day,
        is_soldout,

        -- Computed velocity between snapshots
        CASE
            WHEN prev_seats_sold IS NOT NULL AND prev_days_before IS NOT NULL
            THEN (seats_sold - prev_seats_sold)::DOUBLE / NULLIF(prev_days_before - days_before_game, 0)
            ELSE NULL
        END AS computed_velocity,

        -- Velocity category
        CASE
            WHEN velocity_per_day >= 15 THEN 'fast'
            WHEN velocity_per_day >= 5  THEN 'moderate'
            WHEN velocity_per_day >= 1  THEN 'slow'
            ELSE 'stalled'
        END AS velocity_category,

        -- Days-to-game urgency category
        CASE
            WHEN days_before_game <= 3  THEN 'last_minute'
            WHEN days_before_game <= 7  THEN 'near_term'
            WHEN days_before_game <= 14 THEN 'mid_term'
            ELSE 'early'
        END AS purchase_window,

        -- Projected final sell-through (linear extrapolation)
        CASE
            WHEN velocity_per_day > 0 AND days_before_game > 0
            THEN LEAST(100.0, sell_through_pct + (velocity_per_day * days_before_game / capacity * 100))
            ELSE sell_through_pct
        END AS projected_final_sell_through

    FROM with_velocity
)

SELECT * FROM final
ORDER BY season, date, section, days_before_game DESC
