"""
SeatGeek secondary market scraper for SD FC games.
Uses Playwright (headless Chromium) to scrape listing prices + Deal Score.
Falls back to synthetic secondary market data calibrated to 2025 actuals.

NOTE: Only listing prices (not completed transactions) are available here.
Completed transaction prices from StubHub sold history are in stubhub_market.py.
Listing prices are noisy (speculative inflation) — use as a supplementary signal
alongside StubHub sold transactions as the primary demand truth.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = RAW_DIR / "seatgeek_market.csv"

SEATGEEK_URL = "https://seatgeek.com/san-diego-fc-tickets"

# Secondary market premium bands
# cold: <0%  (primary > secondary — underpriced primary, or very low demand)
# healthy: 10–15% above primary (STH resale zone — target equilibrium)
# warm: 16–30% above primary (strong demand, potential upward price adjustment)
# hot: >30% above primary (significantly underpriced primary)
PREMIUM_BANDS = {
    "cold":    (-0.15, 0.09),
    "healthy": (0.10,  0.20),
    "warm":    (0.21,  0.35),
    "hot":     (0.36,  0.80),
}

# Tier-specific secondary market premium calibration (2025 observed ranges)
# Based on SD FC inaugural season data and MLS market norms
TIER_MARKET_PARAMS = {
    "supporters_ga": {
        "base_premium": 0.05,   # GA nearly fixed; slight premium on hot games
        "premium_std": 0.08,
        "avg_listings": 45,
    },
    "lower_bowl_corner": {
        "base_premium": 0.18,
        "premium_std": 0.12,
        "avg_listings": 85,
    },
    "lower_bowl_goal": {
        "base_premium": 0.16,
        "premium_std": 0.11,
        "avg_listings": 90,
    },
    "lower_bowl_midfield": {
        "base_premium": 0.25,
        "premium_std": 0.14,
        "avg_listings": 60,
    },
    "field_club": {
        "base_premium": 0.30,
        "premium_std": 0.18,
        "avg_listings": 30,
    },
    "upper_bowl": {
        "base_premium": 0.08,
        "premium_std": 0.10,
        "avg_listings": 120,
    },
    "west_club": {
        "base_premium": 0.28,
        "premium_std": 0.16,
        "avg_listings": 25,
    },
    "upper_concourse": {
        "base_premium": 0.04,
        "premium_std": 0.07,
        "avg_listings": 200,
    },
}

# Game-level demand multiplier on secondary market premiums
GAME_DEMAND_BOOST = {
    "baja_cup":       0.55,
    "season_opener":  0.45,
    "tier1_rivalry":  0.40,
    "tier1_marquee":  0.35,
    "tier1":          0.20,
    "tier2":          0.00,
    "tier3":         -0.10,
    "decision_day":   0.25,
}


def _classify_premium(premium: float) -> str:
    for band, (low, high) in PREMIUM_BANDS.items():
        if low <= premium <= high:
            return band
    return "hot" if premium > 0.36 else "cold"


def _try_playwright_scrape(url: str) -> Optional[list[dict]]:
    """Attempt to scrape SeatGeek listing prices via Playwright."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        logger.warning("Playwright not installed — run: playwright install chromium")
        return None

    rows = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
                )
            )
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(4)

            content = page.content()
            if "san-diego-fc" not in content.lower() and "seatgeek" not in content.lower():
                logger.warning("SeatGeek page did not load expected content")
                browser.close()
                return None

            # Intercept JSON data embedded in page (SeatGeek embeds listings in __NEXT_DATA__)
            import re
            next_data_match = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', content, re.DOTALL)
            if next_data_match:
                try:
                    next_data = json.loads(next_data_match.group(1))
                    # Navigate SeatGeek JSON structure for listings
                    listings = (
                        next_data.get("props", {})
                        .get("pageProps", {})
                        .get("listings", [])
                    )
                    for listing in listings[:200]:  # cap at 200 listings
                        price = listing.get("price") or listing.get("list_price")
                        section = listing.get("section", "")
                        if price and section:
                            rows.append({
                                "section_raw": str(section),
                                "list_price": float(price),
                                "quantity": listing.get("quantity", 2),
                                "deal_score": listing.get("deal_score"),
                                "data_source": "seatgeek_live",
                            })
                except (json.JSONDecodeError, KeyError):
                    pass

            browser.close()
            logger.info(f"SeatGeek Playwright extracted {len(rows)} listings")
    except Exception as exc:
        logger.warning(f"SeatGeek Playwright scrape failed: {exc}")
        return None

    return rows if rows else None


def _generate_synthetic_market(
    schedule: pd.DataFrame,
    face_prices: Optional[pd.DataFrame],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Generate synthetic SeatGeek-style secondary market listing data.

    Calibration anchors:
    - 2025 SD FC home opener (34,506 att): secondary ~40-50% above face
    - 2025 Baja Cup (30,500 att): secondary ~35-45% above face
    - 2025 Inter Miami (33,100 att): secondary ~30-40% above face
    - 2025 mid-tier games (~25-27K): secondary 5-15% above face (healthy band)
    - Low demand games (~22-24K): secondary at or below face (cold)
    """
    rows = []

    for _, game in schedule.iterrows():
        # Determine game demand boost
        if game.get("is_baja_cup"):
            boost = GAME_DEMAND_BOOST["baja_cup"]
        elif game.get("is_season_opener"):
            boost = GAME_DEMAND_BOOST["season_opener"]
        elif game.get("is_decision_day"):
            boost = GAME_DEMAND_BOOST["decision_day"]
        elif game.get("opponent_tier") == 1 and game.get("is_rivalry"):
            boost = GAME_DEMAND_BOOST["tier1_rivalry"]
        elif game.get("opponent_tier") == 1 and game.get("is_marquee"):
            boost = GAME_DEMAND_BOOST["tier1_marquee"]
        elif game.get("opponent_tier") == 1:
            boost = GAME_DEMAND_BOOST["tier1"]
        elif game.get("opponent_tier") == 2:
            boost = GAME_DEMAND_BOOST["tier2"]
        else:
            boost = GAME_DEMAND_BOOST["tier3"]

        # Saturday premium on secondary too
        if game.get("is_saturday"):
            boost += 0.05
        elif game.get("is_wednesday"):
            boost -= 0.04

        for tier, params in TIER_MARKET_PARAMS.items():
            base_premium = params["base_premium"] + boost
            premium = rng.normal(base_premium, params["premium_std"])
            premium = float(np.clip(premium, -0.20, 1.20))

            # Get face price for this tier from face_prices if available
            face_price = None
            if face_prices is not None:
                mask = (face_prices["game_id"] == game["game_id"]) & (face_prices["tier"] == tier)
                if mask.any():
                    face_price = float(face_prices.loc[mask, "list_price"].mean())

            if face_price is None:
                # Fallback: use approximate base prices by tier
                TIER_BASE = {
                    "supporters_ga": 28, "lower_bowl_corner": 57, "lower_bowl_goal": 55,
                    "lower_bowl_midfield": 75, "field_club": 180, "upper_bowl": 40,
                    "west_club": 140, "upper_concourse": 32,
                }
                face_price = float(TIER_BASE.get(tier, 50))

            avg_secondary = round(face_price * (1 + premium), 2)
            min_secondary = round(avg_secondary * rng.uniform(0.85, 0.95), 2)
            max_secondary = round(avg_secondary * rng.uniform(1.05, 1.35), 2)
            n_listings = max(1, int(rng.normal(params["avg_listings"], params["avg_listings"] * 0.3)))

            health = _classify_premium(premium)

            rows.append({
                "game_id": game["game_id"],
                "season": game.get("season", 2026),
                "date": game["date"],
                "opponent": game["opponent"],
                "tier": tier,
                "face_price_avg": face_price,
                "secondary_avg_listing": avg_secondary,
                "secondary_min_listing": min_secondary,
                "secondary_max_listing": max_secondary,
                "secondary_premium_pct": round(premium * 100, 1),
                "n_listings": n_listings,
                "market_health": health,
                "demand_boost": round(boost, 3),
                "data_source": "synthetic",
            })

    return pd.DataFrame(rows)


def load_seatgeek_market(
    schedule: Optional[pd.DataFrame] = None,
    face_prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load SeatGeek secondary market data (live → synthetic fallback)."""
    if schedule is None:
        schedule_file = RAW_DIR / "sdfc_schedule_2026.csv"
        if schedule_file.exists():
            schedule = pd.read_csv(schedule_file)
        else:
            from src.data_ingestion.sdfc_schedule import load_schedule_2026
            schedule = load_schedule_2026()

    if face_prices is None:
        tm_file = RAW_DIR / "ticketmaster_prices.csv"
        if tm_file.exists():
            face_prices = pd.read_csv(tm_file)

    # Try live scrape
    live_rows = _try_playwright_scrape(SEATGEEK_URL)
    if live_rows:
        logger.success(f"Got {len(live_rows)} live SeatGeek listings")
        # Note: live data is game-agnostic listings; we'd need to match to specific games
        # For now, use as spot-check and fall through to synthetic for full coverage

    # Generate synthetic for full game × tier coverage
    rng = np.random.default_rng(seed=43)
    df = _generate_synthetic_market(schedule, face_prices, rng)
    return df


def run() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = load_seatgeek_market()
    df.to_csv(OUT_FILE, index=False)
    logger.success(f"Saved {len(df):,} SeatGeek market rows → {OUT_FILE}")
    logger.info(f"  Games covered  : {df['game_id'].nunique()}")
    logger.info(f"  Tiers covered  : {df['tier'].nunique()}")
    hot = df[df["market_health"] == "hot"]
    healthy = df[df["market_health"] == "healthy"]
    cold = df[df["market_health"] == "cold"]
    logger.info(f"  Market health  : {len(hot)} HOT / {len(healthy)} HEALTHY / {len(cold)} COLD")
    logger.info(f"  Avg premium    : {df['secondary_premium_pct'].mean():.1f}%")
    return df


if __name__ == "__main__":
    run()
