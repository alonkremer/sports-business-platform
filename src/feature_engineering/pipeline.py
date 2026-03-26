"""
Feature engineering pipeline — 14 feature groups, 60+ variables.
Reads from DuckDB Gold layer, outputs demand_features.parquet.

Feature group naming follows SHAP-compatible conventions so explanations
can reference human-readable group labels in the dashboard.

Output: data/features/demand_features.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data"
DB_PATH   = DATA_DIR / "sdfc_pricing.duckdb"
FEAT_DIR  = DATA_DIR / "features"
OUT_FILE  = FEAT_DIR / "demand_features.parquet"

# ── Section tier → ordinal encoding ──────────────────────────────────────────
TIER_ORDINAL = {
    "upper_concourse":     1,
    "supporters_ga":       2,
    "upper_bowl":          3,
    "lower_bowl_corner":   4,
    "lower_bowl_goal":     5,
    "lower_bowl_midfield": 6,
    "west_club":           7,
    "field_club":          8,
}

# ── MLS market size benchmarks (for competitor context features) ──────────────
MLS_AVG_ATTENDANCE_2025 = 21_740
MLS_TOP_ATTENDANCE_2025 = 47_200  # Atlanta United


def _load_gold(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the Gold fact table from DuckDB."""
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute("SELECT * FROM gold.fact_game_pricing").df()
    con.close()
    logger.info(f"Loaded gold.fact_game_pricing: {len(df):,} rows")
    return df


def _add_group1_schedule_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Group 1: Game Schedule & Calendar Features."""
    df["date"] = pd.to_datetime(df["date"])

    df["g1_day_of_week_num"]    = df["date"].dt.dayofweek          # 0=Mon, 5=Sat
    df["g1_month"]              = df["date"].dt.month
    df["g1_week_of_year"]       = df["date"].dt.isocalendar().week.astype(int)
    df["g1_is_saturday"]        = df["is_saturday"].astype(int)
    df["g1_is_wednesday"]       = df["is_wednesday"].astype(int)
    df["g1_is_sunday"]          = df["is_sunday"].astype(int)
    df["g1_is_weekend"]         = df["is_weekend"].astype(int)
    df["g1_is_season_opener"]   = df["is_season_opener"].astype(int)
    df["g1_is_decision_day"]    = df["is_decision_day"].astype(int)
    df["g1_is_baja_cup"]        = df["is_baja_cup"].astype(int)
    df["g1_is_fifa_adjacent"]   = df["is_fifa_adjacent"].astype(int)
    df["g1_season"]             = df["season"].astype(int)

    # Season progress (0=start, 1=end)
    season_dates = df.groupby("season")["date"].agg(["min", "max"])
    def season_progress(row):
        s = season_dates.loc[row["season"]]
        total = (s["max"] - s["min"]).days or 1
        return (row["date"] - s["min"]).days / total
    df["g1_season_progress"] = df.apply(season_progress, axis=1).round(3)

    # Is second half of season
    df["g1_is_second_half"] = (df["season_half"] == "second_half").astype(int) if "season_half" in df.columns else 0

    return df


def _add_group2_team_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Group 2: Team Performance & Form Features."""
    df["g2_opponent_tier"]           = df["opponent_tier"].fillna(2).astype(int)
    df["g2_is_rivalry"]              = df["is_rivalry"].astype(int)
    df["g2_is_marquee"]              = df["is_marquee"].astype(int)
    df["g2_home_win_prob"]           = df["home_win_prob"].fillna(0.52)
    df["g2_home_xg"]                 = df["home_xg"].fillna(1.5)
    df["g2_away_xg"]                 = df["away_xg"].fillna(1.2)
    df["g2_xg_diff"]                 = df["g2_home_xg"] - df["g2_away_xg"]
    df["g2_is_heavy_favorite"]       = (df["home_win_prob"] >= 0.55).astype(int)
    df["g2_is_underdog"]             = (df["home_win_prob"] < 0.45).astype(int)

    # Opponent strength encoded
    df["g2_opponent_strength"] = df["g2_opponent_tier"].map({1: 3, 2: 2, 3: 1}).fillna(1)

    # Rivalry × tier interaction (most impactful single feature)
    df["g2_rivalry_tier1"] = ((df["is_rivalry"]) & (df["opponent_tier"] == 1)).astype(int)
    df["g2_marquee_tier1"] = ((df["is_marquee"]) & (df["opponent_tier"] == 1)).astype(int)

    # Western Conference rival flag (Galaxy, LAFC, Sounders)
    west_rivals = {"LA Galaxy", "LAFC", "Seattle Sounders", "Portland Timbers"}
    df["g2_is_west_rival"] = df["opponent"].isin(west_rivals).astype(int)

    return df


def _add_group3_star_players(df: pd.DataFrame) -> pd.DataFrame:
    """Group 3: Star Player & Match Significance Features."""
    df["g3_star_player_on_opp"]  = df["star_player_on_opponent"].astype(int)
    df["g3_is_cross_border"]     = df["is_cross_border_event"].astype(int)

    # High-profile match composite score
    df["g3_match_significance"] = (
        df["g2_is_rivalry"] * 2
        + df["g2_is_marquee"] * 2
        + df["g3_star_player_on_opp"] * 1.5
        + df["g1_is_baja_cup"] * 3
        + df["g1_is_season_opener"] * 2
        + df["g1_is_decision_day"] * 1.5
    ).round(2)

    return df


def _add_group4_market_betting(df: pd.DataFrame) -> pd.DataFrame:
    """Group 4: Market & Betting Signal Features."""
    df["g4_home_win_prob"]          = df["home_win_prob"].fillna(0.52)
    df["g4_secondary_premium_pct"]  = df["secondary_premium_pct"].fillna(12.0)
    df["g4_market_health_score"] = df["market_health"].map({
        "cold": 0, "healthy": 1, "warm": 2, "hot": 3
    }).fillna(1)

    # Secondary market health as binary flags
    df["g4_is_cold_market"]    = (df["market_health"] == "cold").astype(int)
    df["g4_is_healthy_market"] = (df["market_health"] == "healthy").astype(int)
    df["g4_is_warm_market"]    = (df["market_health"] == "warm").astype(int)
    df["g4_is_hot_market"]     = (df["market_health"] == "hot").astype(int)

    # Sold price available
    df["g4_sold_price_avg"]    = df["sold_price_avg"].fillna(df["face_price"] * 1.12)
    df["g4_sold_to_face_ratio"]= (df["g4_sold_price_avg"] / df["face_price"].replace(0, np.nan)).fillna(1.12)
    df["g4_n_transactions"]    = df["n_sold_transactions_est"].fillna(100).astype(int)

    return df


def _add_group5_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Group 5: Supply & Inventory Features."""
    df["g5_sell_through_t7"]     = df["sell_through_t7"].fillna(75.0)
    df["g5_seats_remaining_t7"]  = df["seats_remaining_t7"].fillna(df["capacity"] * 0.15)
    df["g5_velocity_t7"]         = df["velocity_t7"].fillna(2.0)
    df["g5_sell_through_t30"]    = df["sell_through_t30"].fillna(40.0)
    df["g5_velocity_t30"]        = df["velocity_t30"].fillna(0.5)
    df["g5_is_soldout"]          = df["is_soldout_t7"].fillna(False).astype(int)
    df["g5_pct_remaining"]       = (df["g5_seats_remaining_t7"] / df["capacity"].replace(0, 1) * 100).round(1)

    # Sell-through acceleration (T-7 vs T-30)
    df["g5_velocity_acceleration"] = (df["g5_velocity_t7"] - df["g5_velocity_t30"]).fillna(0)

    # Category
    df["g5_sell_through_cat"] = pd.cut(
        df["g5_sell_through_t7"],
        bins=[0, 40, 60, 80, 95, 100],
        labels=["low", "moderate", "good", "high", "sellout"],
        right=True
    ).astype(str)

    return df


def _add_group6_seat_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Group 6: Seat / Section Attribute Features."""
    df["g6_tier_ordinal"]        = df["tier"].map(TIER_ORDINAL).fillna(3)
    df["g6_capacity"]            = df["capacity"].fillna(2000).astype(int)
    df["g6_is_premium"]          = df["tier"].isin(["field_club", "west_club"]).astype(int)
    df["g6_is_ga"]               = (df["tier"] == "supporters_ga").astype(int)
    df["g6_is_lower_bowl"]       = df["tier"].str.startswith("lower_bowl").astype(int)
    df["g6_is_upper"]            = df["tier"].isin(["upper_bowl", "upper_concourse"]).astype(int)
    df["g6_is_midfield"]         = (df["tier"] == "lower_bowl_midfield").astype(int)
    df["g6_face_base_price"]     = df["face_base_price"].fillna(50.0)

    return df


def _add_group7_fan_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 7: Fan & Demographic Signal Features.
    Using static baseline estimates (Census / MTS research).
    Production: replace with live Census API + Google Maps API responses.
    """
    # San Diego demographics (static)
    df["g7_military_market_score"]    = 0.75  # SD has ~120K active military — high
    df["g7_tourist_market_score"]     = 0.60  # ~35M annual visitors
    df["g7_cross_border_base"]        = 0.45  # ~6K regular cross-border fans
    df["g7_transit_score"]            = 0.55  # MTS trolley to stadium

    # Cross-border demand boost for Baja Cup
    df["g7_cross_border_demand"] = np.where(
        df["g1_is_baja_cup"] == 1,
        df["g7_cross_border_base"] * 2.5,
        df["g7_cross_border_base"]
    )

    # Median income index (higher = more price-inelastic fans)
    df["g7_income_index"] = 1.05  # SD median HH income ~$92K vs MLS avg market

    return df


def _add_group8_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Group 8: Weather & Environment Features."""
    df["g8_temp_f"]          = df["temp_f"].fillna(70.0)
    df["g8_rain_prob"]       = df["rain_prob"].fillna(0.05)
    df["g8_is_high_rain"]    = df["is_high_rain_risk"].astype(int)

    # Temp comfort score (0=uncomfortable, 1=perfect)
    df["g8_temp_comfort"] = np.where(
        (df["g8_temp_f"] >= 60) & (df["g8_temp_f"] <= 78), 1.0,
        np.where(
            (df["g8_temp_f"] >= 55) & (df["g8_temp_f"] <= 85), 0.7,
            0.4
        )
    )

    # Weather demand impact (10°F variance = ~6% attendance swing; rain >30% = -17%)
    df["g8_weather_demand_impact"] = (
        df["g8_temp_comfort"] - df["g8_rain_prob"] * 0.55
    ).clip(0, 1).round(3)

    return df


def _add_group9_competing_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 9: Competition & Event Conflict Features.
    Static flags based on 2026 schedule knowledge.
    Production: integrate Ticketmaster Discovery API + MLB/NWSL schedules.
    """
    # Padres season: April–October; rough conflict probability
    df["g9_padres_conflict_prob"] = np.where(
        df["date"].dt.month.between(4, 10), 0.45, 0.0
    )

    # Comic-Con annually in July (high tourism weekend)
    df["g9_is_comic_con_weekend"] = (
        (df["date"].dt.month == 7) & (df["date"].dt.day.between(24, 28))
    ).astype(int)

    # Competing events score (composite)
    df["g9_competing_events_score"] = (
        df["g9_padres_conflict_prob"] * 0.4
        + df["g9_is_comic_con_weekend"] * 0.3
    ).round(3)

    return df


def _add_group10_promotions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 10: Promotions & Special Events.
    Static mapping based on 2026 schedule announced promotions.
    Production: scrape SD FC official schedule for promo flags.
    """
    # Known 2026 promotions (placeholder — enriched from SD FC schedule page)
    promo_games = {
        "2026_H01": {"has_giveaway": True,  "has_fireworks": False, "theme_night": "Opener"},
        "2026_H03": {"has_giveaway": False, "has_fireworks": False, "theme_night": "Rivalry"},
        "2026_H08": {"has_giveaway": True,  "has_fireworks": False, "theme_night": "Marquee"},
        "2026_H15": {"has_giveaway": True,  "has_fireworks": True,  "theme_night": "Baja Cup"},
    }

    df["g10_has_giveaway"]     = df["game_id"].map(lambda g: promo_games.get(g, {}).get("has_giveaway", False)).astype(int)
    df["g10_has_fireworks"]    = df["game_id"].map(lambda g: promo_games.get(g, {}).get("has_fireworks", False)).astype(int)
    df["g10_has_theme_night"]  = df["game_id"].map(lambda g: 1 if promo_games.get(g, {}).get("theme_night") else 0)
    df["g10_promo_score"]      = (df["g10_has_giveaway"] + df["g10_has_fireworks"] + df["g10_has_theme_night"]).clip(0, 3)

    return df


def _add_group11_pricing_comparables(df: pd.DataFrame) -> pd.DataFrame:
    """Group 11: Pricing Comparables (cross-entertainment)."""
    # Padres equivalent Lower Bowl ticket: ~$55–75; Upper Bowl ~$25–35
    PADRES_EQUIV = {"lower_bowl_midfield": 70, "upper_bowl": 30, "field_club": 120}
    df["g11_padres_equiv_price"] = df["tier"].map(PADRES_EQUIV).fillna(45.0)

    # Price relative to Padres
    df["g11_price_vs_padres"] = (df["face_price"] / df["g11_padres_equiv_price"].replace(0, np.nan)).fillna(1.0).round(3)

    # MLS average price context (~$50 league avg)
    df["g11_price_vs_mls_avg"] = (df["face_price"] / 50.0).round(3)

    return df


def _add_group12_historical(df: pd.DataFrame) -> pd.DataFrame:
    """Group 12: Historical Pattern Features."""
    # Historical average attendance same opponent (2025 season)
    hist_avg = {
        "St. Louis City SC":       34506,
        "Portland Timbers":        28800,
        "LAFC":                    32502,
        "Seattle Sounders":        28228,
        "Real Salt Lake":          25100,
        "Houston Dynamo":          24800,
        "Austin FC":               27200,
        "Inter Miami":             33100,
        "Vancouver Whitecaps":     25600,
        "Pachuca":                 21872,
        "LA Galaxy":               31000,
        "Minnesota United":        26300,
        "Nashville SC":            27500,
        "Club Tijuana":            30500,
        "Atlanta United":          29200,
        "New England Revolution":  25800,
        "FC Dallas":               27100,
    }
    df["g12_hist_avg_att_same_opp"] = df["opponent"].map(hist_avg).fillna(27000.0)

    # Historical demand index same opponent
    df["g12_hist_demand_idx_opp"] = (df["g12_hist_avg_att_same_opp"] / 35_000).round(4)

    # Prior game revenue opportunity (lagged — simplified as section avg)
    df["g12_face_price_lag"] = df.groupby(["tier", "season"])["face_price"].transform("mean").round(2)

    # Historical sell-through proxy (from 2025 calibration)
    att_to_sellthrough = {opp: att / 35_000 for opp, att in hist_avg.items()}
    df["g12_hist_sellthrough"] = df["opponent"].map(att_to_sellthrough).fillna(0.78)

    return df


def _add_group13_social_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group 13: Social Sentiment Features.
    Synthetic baseline — replace with Reddit API + VADER NLP in production.
    Social signals contribute ~2% MAPE improvement; dropped if below threshold.
    """
    # Game-level sentiment index (0=negative, 1=neutral, 0.5=positive)
    # Placeholder: derived from game type
    base_sentiment = 0.6
    df["g13_reddit_buzz_score"] = base_sentiment
    df["g13_social_sentiment"]  = (
        base_sentiment
        + df["g2_is_rivalry"] * 0.12
        + df["g2_is_marquee"] * 0.10
        + df["g1_is_baja_cup"] * 0.15
        - df["g4_is_cold_market"] * 0.08
    ).clip(0, 1).round(3)

    df["g13_is_high_buzz"]      = (df["g13_social_sentiment"] > 0.70).astype(int)

    return df


def _add_group14_cross_border(df: pd.DataFrame) -> pd.DataFrame:
    """Group 14: Cross-Border & Tijuana Market Features (SD FC unique)."""
    df["g14_is_baja_cup"]          = df["g1_is_baja_cup"]
    df["g14_is_club_tijuana"]      = (df["opponent"] == "Club Tijuana").astype(int)
    df["g14_cross_border_index"]   = np.where(
        df["g1_is_baja_cup"] == 1, 2.5,
        np.where(df["g14_is_club_tijuana"] == 1, 1.8, 0.45)
    )
    # Latino fan market share estimate (higher for Baja Cup)
    df["g14_latino_market_share"]  = np.where(
        df["g1_is_baja_cup"] == 1, 0.38, 0.24
    )

    return df


def _add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add target variables for ML training."""
    # Primary target: demand_index (normalized attendance)
    df["target_demand_index"] = df["demand_index"].fillna(0.80)

    # Secondary target: log sell-through at T-7 (for regression)
    df["target_log_sellthrough"] = np.log1p(df["g5_sell_through_t7"].fillna(75.0))

    # Price gap target (what we're trying to close)
    df["target_price_gap"]    = df["optimal_price_increase"].fillna(0.0)
    df["target_revenue_opp"]  = df["total_revenue_opportunity"].fillna(0.0)

    return df


def _select_ml_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select and order final ML feature columns."""
    feature_cols = [c for c in df.columns if c.startswith("g") and "_" in c and c[1].isdigit()]
    feature_cols.sort()

    # Add a few key raw fields that are useful directly
    extra_cols = [
        "game_id", "section", "tier", "season", "date",
        "face_price", "sold_price_avg", "secondary_premium_pct",
        "market_health", "opportunity_tier",
        "total_revenue_opportunity", "revenue_opp_per_seat",
        "optimal_price_increase", "sth_resale_margin",
        "is_hot_market_alert", "backlash_risk_score",
        "target_demand_index", "target_log_sellthrough",
        "target_price_gap", "target_revenue_opp",
    ]

    all_cols = extra_cols + [c for c in feature_cols if c not in extra_cols]
    available = [c for c in all_cols if c in df.columns]

    return df[available], feature_cols


def build_features(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Run the full 14-group feature pipeline."""
    logger.info("Starting feature engineering pipeline (14 groups)...")

    # Load gold table
    df = _load_gold(db_path)

    # Apply all 14 feature groups in sequence
    df = _add_group1_schedule_calendar(df)
    df = _add_group2_team_performance(df)
    df = _add_group3_star_players(df)
    df = _add_group4_market_betting(df)
    df = _add_group5_inventory(df)
    df = _add_group6_seat_attributes(df)
    df = _add_group7_fan_demographics(df)
    df = _add_group8_weather(df)
    df = _add_group9_competing_events(df)
    df = _add_group10_promotions(df)
    df = _add_group11_pricing_comparables(df)
    df = _add_group12_historical(df)
    df = _add_group13_social_sentiment(df)
    df = _add_group14_cross_border(df)
    df = _add_target_variables(df)

    feature_df, feature_cols = _select_ml_features(df)

    logger.info(f"Feature matrix: {len(feature_df):,} rows × {len(feature_df.columns)} columns")
    logger.info(f"  ML feature columns: {len(feature_cols)}")
    logger.info(f"  Feature groups populated: 14")
    logger.info(f"  Games covered: {feature_df['game_id'].nunique()}")
    logger.info(f"  Sections covered: {feature_df['section'].nunique()}")

    return feature_df


def run(db_path: Path = DB_PATH) -> pd.DataFrame:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_features(db_path)
    df.to_parquet(OUT_FILE, index=False, engine="pyarrow", compression="snappy")
    logger.success(f"Saved demand_features.parquet → {OUT_FILE}")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns[:10])} ...")
    logger.info(f"  NaN pct: {df.isnull().mean().mean() * 100:.1f}%")
    return df


if __name__ == "__main__":
    run()
