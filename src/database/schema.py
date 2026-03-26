"""
DuckDB schema initializer.
Creates the Bronze/Silver/Gold schema layers and loads all raw CSV data.
Run this once before running dbt models.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
RAW_DIR  = DATA_DIR / "raw"
DB_PATH  = DATA_DIR / "sdfc_pricing.duckdb"


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def create_schemas(con: duckdb.DuckDBPyConnection) -> None:
    """Create medallion architecture schemas."""
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    logger.info("Schemas created: bronze, silver, gold")


def load_bronze_tables(con: duckdb.DuckDBPyConnection) -> None:
    """
    Load all Bronze layer tables from CSV files.
    Bronze = raw, no transformations applied.
    """
    tables = {
        "bronze.games": RAW_DIR / "synthetic_games.csv",
        "bronze.ticketmaster_prices": RAW_DIR / "ticketmaster_prices.csv",
        "bronze.secondary_market": RAW_DIR / "synthetic_secondary_market.csv",
        "bronze.mls_attendance": RAW_DIR / "mls_attendance_history.csv",
        "bronze.betting_odds": RAW_DIR / "oddsportal_mls_odds.csv",
        "bronze.inventory": RAW_DIR / "synthetic_inventory.csv",
        "bronze.fbref_attendance": RAW_DIR / "fbref_sdfc_2025.csv",
        "bronze.schedule_2026": RAW_DIR / "sdfc_schedule_2026.csv",
        "bronze.stubhub_sold": RAW_DIR / "stubhub_sold_transactions.csv",
        "bronze.seatgeek_listings": RAW_DIR / "seatgeek_market.csv",
        "bronze.transactions": RAW_DIR / "synthetic_transactions.csv",
    }

    for table_name, csv_path in tables.items():
        if not csv_path.exists():
            logger.warning(f"CSV not found, skipping: {csv_path.name}")
            continue
        try:
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.execute(f"""
                CREATE TABLE {table_name} AS
                SELECT * FROM read_csv_auto('{csv_path.as_posix()}')
            """)
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            logger.success(f"Loaded {table_name}: {count:,} rows")
        except Exception as exc:
            logger.error(f"Failed to load {table_name}: {exc}")


def build_silver_views(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build Silver layer as materialized views (cleaned, enriched).
    These replicate the dbt Silver models for direct Python use.
    """
    # silver.games
    con.execute("DROP VIEW IF EXISTS silver.games")
    con.execute("""
        CREATE VIEW silver.games AS
        SELECT
            game_id,
            season::INTEGER                   AS season,
            date::DATE                        AS date,
            day_of_week,
            opponent,
            CASE
                WHEN opponent LIKE '%Decision Day%' OR opponent = 'TBD' THEN 'TBD'
                ELSE opponent
            END                               AS opponent_clean,
            opponent_tier::INTEGER            AS opponent_tier,
            is_rivalry::BOOLEAN               AS is_rivalry,
            is_marquee::BOOLEAN               AS is_marquee,
            is_baja_cup::BOOLEAN              AS is_baja_cup,
            is_season_opener::BOOLEAN         AS is_season_opener,
            is_decision_day::BOOLEAN          AS is_decision_day,
            is_saturday::BOOLEAN              AS is_saturday,
            is_wednesday::BOOLEAN             AS is_wednesday,
            day_of_week = 'Sunday'            AS is_sunday,
            day_of_week IN ('Saturday','Sunday') AS is_weekend,
            opponent IN ('Inter Miami','LA Galaxy') AS star_player_on_opponent,
            is_baja_cup::BOOLEAN              AS is_cross_border_event,
            home_team,
            venue,
            result,
            goals_for,
            goals_against,
            home_xg,
            away_xg,
            COALESCE(attendance, projected_attendance) AS attendance_or_projection,
            attendance,
            projected_attendance,
            capacity,
            sell_through_pct,
            demand_index,
            home_win_prob,
            temp_f,
            rain_prob,
            rain_prob > 0.30                  AS is_high_rain_risk,
            CASE WHEN season = 2026
                 AND date BETWEEN '2026-05-11' AND '2026-07-30' THEN TRUE
                 ELSE FALSE END               AS is_fifa_adjacent
        FROM bronze.games
        WHERE game_id IS NOT NULL
    """)

    # silver.pricing
    con.execute("DROP VIEW IF EXISTS silver.pricing")
    con.execute("""
        CREATE VIEW silver.pricing AS
        SELECT
            f.game_id,
            f.season::INTEGER     AS season,
            f.date::DATE          AS date,
            f.opponent,
            f.section,
            f.tier,
            f.capacity::INTEGER   AS capacity,
            f.list_price::DOUBLE  AS face_price,
            f.base_price::DOUBLE  AS face_base_price,
            f.demand_multiplier::DOUBLE AS demand_multiplier,

            s.sold_price_avg::DOUBLE    AS sold_price_avg,
            s.sold_price_p25::DOUBLE    AS sold_price_p25,
            s.sold_price_p75::DOUBLE    AS sold_price_p75,
            s.listing_price_avg::DOUBLE AS listing_price_avg,
            s.secondary_premium_pct::DOUBLE AS secondary_premium_pct,
            s.market_health,
            s.revenue_opp_per_seat::DOUBLE AS revenue_opp_per_seat,
            s.n_sold_transactions_est::INTEGER AS n_sold_transactions_est,

            (s.sold_price_avg - f.list_price)   AS price_gap_sold_vs_face,
            (s.sold_price_avg - f.list_price) / NULLIF(f.list_price, 0) * 100 AS price_gap_pct,
            s.sold_price_avg - (f.list_price * 1.10) AS sth_resale_margin,
            s.sold_price_avg / 1.10 - f.list_price   AS optimal_price_increase,

            CASE
                WHEN s.secondary_premium_pct < 10  THEN 'cold'
                WHEN s.secondary_premium_pct <= 20 THEN 'healthy'
                WHEN s.secondary_premium_pct <= 35 THEN 'warm'
                ELSE 'hot'
            END AS market_health_computed,

            CASE
                WHEN s.sold_price_avg / 1.10 - f.list_price > 20 THEN 'high'
                WHEN s.sold_price_avg / 1.10 - f.list_price > 8  THEN 'medium'
                WHEN s.sold_price_avg / 1.10 - f.list_price > 0  THEN 'low'
                ELSE 'none'
            END AS opportunity_tier

        FROM bronze.ticketmaster_prices f
        LEFT JOIN bronze.secondary_market s
            ON f.game_id = s.game_id
            AND f.section = s.section
    """)

    logger.info("Silver views built: silver.games, silver.pricing")


def build_gold_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Build Gold layer materialized tables for ML pipeline and dashboard."""
    con.execute("DROP TABLE IF EXISTS gold.fact_game_pricing")
    con.execute("""
        CREATE TABLE gold.fact_game_pricing AS
        SELECT
            p.game_id,
            p.section,
            p.season,
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
            g.home_win_prob,
            g.home_xg,
            g.away_xg,
            g.temp_f,
            g.rain_prob,
            g.is_high_rain_risk,
            g.demand_index,
            g.attendance_or_projection,
            g.result,
            g.goals_for,
            g.goals_against,

            p.tier,
            p.capacity,
            p.face_price,
            p.face_base_price,
            p.demand_multiplier,

            p.sold_price_avg,
            p.sold_price_p25,
            p.sold_price_p75,
            p.listing_price_avg,
            p.secondary_premium_pct,
            p.market_health_computed AS market_health,
            p.n_sold_transactions_est,
            p.revenue_opp_per_seat,

            p.price_gap_sold_vs_face,
            p.price_gap_pct,
            p.sth_resale_margin,
            p.optimal_price_increase,
            p.opportunity_tier,

            -- Total revenue opportunity
            COALESCE(p.revenue_opp_per_seat, 0) * (p.capacity * 0.15) AS total_revenue_opportunity,

            -- Alert flags
            (p.secondary_premium_pct > 30)::BOOLEAN AS is_hot_market_alert,
            (p.optimal_price_increase > 25)::BOOLEAN AS is_backlash_risk,

            -- Backlash risk score (0–10)
            LEAST(10, GREATEST(0,
                CASE WHEN p.optimal_price_increase > 25 THEN 4 ELSE 0 END +
                CASE WHEN g.is_rivalry THEN 2 ELSE 0 END +
                CASE WHEN g.is_marquee THEN 1 ELSE 0 END +
                CASE WHEN g.opponent_tier = 1 THEN 1 ELSE 0 END +
                CASE WHEN p.secondary_premium_pct > 50 THEN 2 ELSE 0 END
            )) AS backlash_risk_score

        FROM silver.pricing p
        LEFT JOIN silver.games g ON p.game_id = g.game_id
        ORDER BY p.season, g.date, p.tier, p.section
    """)

    count = con.execute("SELECT COUNT(*) FROM gold.fact_game_pricing").fetchone()[0]
    logger.success(f"Built gold.fact_game_pricing: {count:,} rows")

    # Quick validation
    hot = con.execute(
        "SELECT COUNT(*) FROM gold.fact_game_pricing WHERE market_health = 'hot'"
    ).fetchone()[0]
    total_opp = con.execute(
        "SELECT SUM(total_revenue_opportunity) FROM gold.fact_game_pricing WHERE season = 2026"
    ).fetchone()[0]
    logger.info(f"  HOT sections: {hot}")
    logger.info(f"  2026 season total opp: ${total_opp:,.0f}" if total_opp else "  2026 opp: N/A")


def initialize(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Run full schema initialization."""
    logger.info(f"Initializing DuckDB at {db_path}")
    con = get_connection(db_path)
    create_schemas(con)
    load_bronze_tables(con)
    build_silver_views(con)
    build_gold_tables(con)
    logger.success("DuckDB schema initialization complete")
    return con


if __name__ == "__main__":
    con = initialize()
    # Spot-check
    print("\n── Gold table preview ──")
    print(con.execute("""
        SELECT game_id, section, tier, face_price, sold_price_avg,
               secondary_premium_pct, market_health, optimal_price_increase
        FROM gold.fact_game_pricing
        WHERE market_health = 'hot'
        ORDER BY optimal_price_increase DESC
        LIMIT 5
    """).df().to_string(index=False))
    con.close()
