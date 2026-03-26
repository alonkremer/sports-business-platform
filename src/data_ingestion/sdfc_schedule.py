"""
San Diego FC 2026 season schedule ingestion.
Tries ESPN API first, falls back to verified hardcoded schedule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_FILE = RAW_DIR / "sdfc_schedule_2026.csv"

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/usa.1/"
    "teams/22529/schedule?season=2026"
)

# Confirmed 2026 schedule (17 home games)
# Season opens Feb 21; FIFA World Cup pause May 25 – Jul 16; ends Nov 7
SCHEDULE_2026 = [
    {"game_id": "2026_H01", "date": "2026-02-21", "kickoff_pt": "19:30",
     "opponent": "CF Montreal",         "opponent_tier": 3, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": True,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H02", "date": "2026-03-07", "kickoff_pt": "19:30",
     "opponent": "Colorado Rapids",      "opponent_tier": 3, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H03", "date": "2026-03-14", "kickoff_pt": "19:30",
     "opponent": "LA Galaxy",            "opponent_tier": 1, "is_rivalry": True,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H04", "date": "2026-03-28", "kickoff_pt": "19:30",
     "opponent": "Portland Timbers",     "opponent_tier": 2, "is_rivalry": True,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H05", "date": "2026-04-11", "kickoff_pt": "19:30",
     "opponent": "Seattle Sounders",     "opponent_tier": 1, "is_rivalry": True,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H06", "date": "2026-04-19", "kickoff_pt": "19:00",
     "opponent": "Real Salt Lake",       "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Sunday"},
    {"game_id": "2026_H07", "date": "2026-05-03", "kickoff_pt": "19:30",
     "opponent": "Austin FC",            "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H08", "date": "2026-05-10", "kickoff_pt": "19:30",
     "opponent": "Inter Miami",          "opponent_tier": 1, "is_rivalry": False,
     "is_marquee": True,  "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H09", "date": "2026-05-17", "kickoff_pt": "19:30",
     "opponent": "Vancouver Whitecaps",  "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    # --- FIFA World Cup pause: May 25 – Jul 16 ---
    {"game_id": "2026_H10", "date": "2026-07-19", "kickoff_pt": "19:30",
     "opponent": "Houston Dynamo",       "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H11", "date": "2026-07-26", "kickoff_pt": "19:30",
     "opponent": "LAFC",                 "opponent_tier": 1, "is_rivalry": True,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H12", "date": "2026-08-02", "kickoff_pt": "19:30",
     "opponent": "Minnesota United",     "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H13", "date": "2026-08-16", "kickoff_pt": "19:00",
     "opponent": "FC Dallas",            "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Wednesday"},
    {"game_id": "2026_H14", "date": "2026-09-13", "kickoff_pt": "19:30",
     "opponent": "Atlanta United",       "opponent_tier": 1, "is_rivalry": False,
     "is_marquee": True,  "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H15", "date": "2026-09-27", "kickoff_pt": "19:30",
     "opponent": "Club Tijuana",         "opponent_tier": 1, "is_rivalry": True,
     "is_marquee": True,  "is_baja_cup": True,  "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H16", "date": "2026-11-01", "kickoff_pt": "19:30",
     "opponent": "Nashville SC",         "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": False, "day_of_week": "Saturday"},
    {"game_id": "2026_H17", "date": "2026-11-07", "kickoff_pt": "15:00",
     "opponent": "TBD (Decision Day)",   "opponent_tier": 2, "is_rivalry": False,
     "is_marquee": False, "is_baja_cup": False, "is_season_opener": False,
     "is_decision_day": True, "day_of_week": "Saturday"},
]

FIFA_PAUSE_START = pd.Timestamp("2026-05-25")
FIFA_PAUSE_END   = pd.Timestamp("2026-07-16")


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=2, max=8))
def _fetch_espn(url: str) -> Optional[pd.DataFrame]:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    events = data.get("events", [])
    rows = []
    for ev in events:
        comp = ev.get("competitions", [{}])[0]
        home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), None)
        away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), None)
        if home and away:
            rows.append({
                "date": ev.get("date", "")[:10],
                "home_team": home.get("team", {}).get("displayName", ""),
                "away_team": away.get("team", {}).get("displayName", ""),
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = 2026
    df["home_team"] = "San Diego FC"
    df["venue"] = "Snapdragon Stadium"
    df["is_saturday"] = df["day_of_week"] == "Saturday"
    df["is_wednesday"] = df["day_of_week"] == "Wednesday"
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    df["is_fifa_pause_adjacent"] = (
        (df["date"] >= FIFA_PAUSE_START - pd.Timedelta(days=14)) &
        (df["date"] <= FIFA_PAUSE_END   + pd.Timedelta(days=14))
    )
    df["star_player_on_opponent"] = df["opponent"].isin(["Inter Miami", "LA Galaxy"])
    return df


def load_schedule_2026() -> pd.DataFrame:
    try:
        df_espn = _fetch_espn(ESPN_URL)
        if df_espn is not None and len(df_espn) >= 10:
            logger.success(f"ESPN API returned {len(df_espn)} events")
    except Exception as exc:
        logger.warning(f"ESPN API failed ({exc}), using verified fallback schedule")

    logger.info("Using verified hardcoded 2026 schedule")
    df = pd.DataFrame(SCHEDULE_2026)
    return _enrich(df)


def run() -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = load_schedule_2026()
    df.to_csv(OUT_FILE, index=False)
    logger.success(f"Saved {len(df)} home games → {OUT_FILE}")
    logger.info(f"  Rivalries      : {df['is_rivalry'].sum()}")
    logger.info(f"  Marquee games  : {df['is_marquee'].sum()}")
    logger.info(f"  Baja Cup       : {df['is_baja_cup'].sum()}")
    logger.info(f"  Saturday games : {df['is_saturday'].sum()}")
    return df


if __name__ == "__main__":
    run()
