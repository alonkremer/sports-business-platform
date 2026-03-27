"""
FastAPI application — SD FC Ticket Pricing Intelligence Platform.
All model inference goes through this API.
Streamlit calls this API, not the models directly.

Endpoints:
  GET  /health                     — Health check
  GET  /games                      — All games with demand forecasts
  GET  /games/{game_id}            — Single game detail
  POST /recommend/{game_id}        — Price recommendations (3 scenarios)
  GET  /market/{game_id}           — Secondary market health
  GET  /gap                        — Price gap analysis (all games)
  GET  /gap/{game_id}              — Price gap for specific game
  GET  /alerts                     — Active alerts
  GET  /retrospective/2025         — Revenue left on table in 2025

Run with: uvicorn api.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from loguru import logger

from api.schemas import (
    AlertResponse,
    GameSummary,
    GuardrailsConfig,
    MarketHealthResponse,
    PriceGapRow,
    RecommendationResponse,
    ScenarioDetail,
    SeasonOpportunityResponse,
)

ROOT = Path(__file__).resolve().parents[1]
FEAT_FILE = ROOT / "data" / "features" / "demand_features.parquet"

app = FastAPI(
    title="SD FC Pricing Intelligence API",
    description=(
        "AI-powered ticket pricing recommendations for San Diego FC. "
        "Provides 3-scenario EMSR-b optimizer output with SHAP explainability, "
        "secondary market analysis, and strategic guardrails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model state (loaded once at startup) ─────────────────────────────────────

_state: dict = {
    "features_df":    None,
    "gap_df":         None,
    "optimizer_df":   None,
    "demand_model":   None,
    "shap_explainer": None,
    "elasticity_df":  None,
    "retrospective":  None,
}


@app.on_event("startup")
async def startup_event():
    """Load models and data on startup."""
    logger.info("Starting SD FC Pricing API...")

    try:
        if FEAT_FILE.exists():
            _state["features_df"] = pd.read_parquet(FEAT_FILE)
            logger.info(f"Loaded features: {len(_state['features_df']):,} rows")
        else:
            logger.warning(f"Feature file not found: {FEAT_FILE} — run the pipeline first")
            # Load a minimal stub so API responds
            _state["features_df"] = pd.DataFrame()
    except Exception as exc:
        logger.error(f"Failed to load features: {exc}")
        _state["features_df"] = pd.DataFrame()

    # Lazy load: models and gap analysis computed on first request
    logger.info("API ready — models will load on first request")


def _ensure_models_loaded():
    """Lazy-load models and derived data on first request."""
    if _state["demand_model"] is not None:
        return

    if _state["features_df"] is None or _state["features_df"].empty:
        raise HTTPException(status_code=503, detail="Feature data not available. Run the data pipeline first.")

    logger.info("Loading ML models (first request)...")

    try:
        from src.pricing_model.demand_model import train as train_demand
        model, explainer, _ = train_demand(_state["features_df"], log_to_mlflow=False)
        _state["demand_model"]   = model
        _state["shap_explainer"] = explainer
        logger.info("Demand model loaded")
    except Exception as exc:
        logger.error(f"Demand model load failed: {exc}")

    try:
        from src.elasticity_model.causal_elasticity import run as run_elasticity
        _state["elasticity_df"] = run_elasticity(_state["features_df"])
        logger.info("Elasticity model loaded")
    except Exception as exc:
        logger.error(f"Elasticity model load failed: {exc}")

    try:
        from src.price_optimizer.emsr_optimizer import run_optimizer
        from src.price_gap.gap_calculator import run as run_gap, compute_retrospective_2025
        _state["optimizer_df"] = run_optimizer(_state["features_df"], _state["elasticity_df"])
        _state["gap_df"], _state["retrospective"] = run_gap(
            _state["features_df"], _state["optimizer_df"]
        )
        logger.info("Optimizer and gap analysis loaded")
    except Exception as exc:
        logger.error(f"Optimizer/gap load failed: {exc}")


# ── Helper functions ──────────────────────────────────────────────────────────

def _get_features(game_id: Optional[str] = None, season: int = 2026) -> pd.DataFrame:
    df = _state["features_df"]
    if df is None or df.empty:
        return pd.DataFrame()
    mask = df["season"] == season
    if game_id:
        mask &= df["game_id"] == game_id
    return df[mask]


def _build_recommendation(row: pd.Series, optimizer_row: Optional[pd.DataFrame]) -> dict:
    """Build recommendation response for a single section × game."""
    shap_top5 = []
    explanation = f"Section {row.get('section', '')}: pricing analysis based on demand signals."

    if _state["demand_model"] is not None:
        try:
            from src.pricing_model.demand_model import predict, generate_shap_explanation
            # Slice from features_df to preserve correct dtypes (row.to_frame().T casts to object)
            feat_row = _state["features_df"][
                (_state["features_df"]["game_id"] == row.get("game_id")) &
                (_state["features_df"]["section"] == row.get("section"))
            ]
            if not feat_row.empty:
                pred_df = predict(_state["demand_model"], _state["shap_explainer"], feat_row.iloc[[0]])
                if not pred_df.empty:
                    shap_top5 = pred_df.iloc[0]["shap_top5"]
                    # Use balanced optimizer price as optimal_price if available
                    face = float(row.get("face_price", 60))
                    optimal = face
                    if optimizer_row is not None and not optimizer_row.empty:
                        bal = optimizer_row[optimizer_row.get("scenario", pd.Series()) == "balanced"] if "scenario" in optimizer_row.columns else optimizer_row
                        if not bal.empty:
                            optimal = float(bal.iloc[0].get("price", face))
                    explanation = generate_shap_explanation(
                        shap_top5,
                        face_price=face,
                        optimal_price=optimal,
                        game_context={"section": row.get("section"), "opponent": row.get("opponent")},
                    )
        except Exception as exc:
            logger.warning(f"SHAP failed: {exc}")

    scenarios = {}
    if optimizer_row is not None and not optimizer_row.empty:
        for _, orow in optimizer_row.iterrows():
            sname = orow.get("scenario", "balanced")
            scenarios[sname] = ScenarioDetail(
                price=float(orow.get("price", row.get("face_price", 60))),
                price_change=float(orow.get("price_change", 0)),
                price_change_pct=float(orow.get("price_change_pct", 0)),
                expected_revenue=float(orow.get("expected_revenue", 0)),
                revenue_delta_pct=float(orow.get("revenue_delta_pct", 0)),
                expected_seats_sold=int(orow.get("expected_seats_sold", 0)),
                expected_sell_through=float(orow.get("expected_sell_through", 0)),
                sth_resale_margin=float(orow.get("sth_resale_margin", 0)),
                sth_resale_margin_pct=float(orow.get("sth_resale_margin_pct", 0)),
                sth_is_healthy=bool(orow.get("sth_is_healthy", True)),
                guardrails_applied=orow.get("guardrails_applied", []) or [],
                is_backlash_risk=bool(orow.get("is_backlash_risk", False)),
            )

    return {
        "game_id":                  str(row.get("game_id", "")),
        "section":                  str(row.get("section", "")),
        "tier":                     str(row.get("tier", "")),
        "face_price":               float(row.get("face_price", 0)),
        "sold_price_avg":           float(row.get("sold_price_avg", 0) or 0),
        "secondary_premium_pct":    float(row.get("secondary_premium_pct", 0) or 0),
        "market_health":            str(row.get("market_health", "healthy")),
        "demand_pred":              float(row.get("target_demand_index", 0.80)),
        "demand_pred_pct":          round(float(row.get("target_demand_index", 0.80)) * 100, 1),
        "optimal_price_increase":   float(row.get("optimal_price_increase", 0) or 0),
        "opportunity_tier":         str(row.get("opportunity_tier", "low")),
        "total_revenue_opportunity": float(row.get("total_revenue_opportunity", 0) or 0),
        "backlash_risk_score":      int(row.get("backlash_risk_score", 0) or 0),
        "shap_explanation":         explanation,
        "shap_top5":                shap_top5,
        "scenarios":                scenarios,
        "recommended_scenario":     "balanced",
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "features_loaded": _state["features_df"] is not None and not _state["features_df"].empty,
        "models_loaded": _state["demand_model"] is not None,
        "version": "1.0.0",
    }


@app.get("/games", response_model=list[GameSummary])
async def get_games(season: int = Query(default=2026, ge=2020, le=2030)):
    """Get all games with demand forecast and market health summary."""
    _ensure_models_loaded()
    df = _get_features(season=season)
    if df.empty:
        return []

    game_groups = df.groupby("game_id")
    result = []
    for game_id, gdf in game_groups:
        row = gdf.iloc[0]
        result.append(GameSummary(
            game_id=str(game_id),
            season=int(row.get("season", season)),
            date=str(row.get("date", ""))[:10],
            opponent=str(row.get("opponent", "")),
            opponent_tier=int(row.get("opponent_tier", 2) or 2),
            is_rivalry=bool(row.get("is_rivalry", False)),
            is_marquee=bool(row.get("is_marquee", False)),
            is_baja_cup=bool(row.get("is_baja_cup", False)),
            demand_index=float(row.get("target_demand_index", 0.80)),
            avg_secondary_premium_pct=float(gdf["secondary_premium_pct"].mean() or 0),
            total_revenue_opportunity=float(gdf["total_revenue_opportunity"].sum() or 0),
            market_health_dominant=str(gdf["market_health"].mode().iloc[0] if not gdf.empty else "healthy"),
            backlash_risk_max=int(gdf["backlash_risk_score"].max() or 0),
            is_hot_market=bool(gdf["is_hot_market_alert"].any()),
            sections_hot=int((gdf["market_health"] == "hot").sum()),
        ))
    return sorted(result, key=lambda x: x.date)


@app.post("/recommend/{game_id}", response_model=list[RecommendationResponse])
async def get_recommendations(
    game_id: str,
    section: Optional[str] = Query(default=None),
    guardrails: Optional[GuardrailsConfig] = None,
):
    """
    Get AI price recommendations (3 scenarios) for a game.
    Optionally filter to a specific section.
    """
    _ensure_models_loaded()
    df = _get_features(game_id=game_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    if section:
        df = df[df["section"] == section]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Section '{section}' not found for game '{game_id}'")

    results = []
    for _, row in df.iterrows():
        # Get optimizer scenarios
        opt_rows = None
        if _state["optimizer_df"] is not None:
            opt_rows = _state["optimizer_df"][
                (_state["optimizer_df"]["game_id"] == game_id) &
                (_state["optimizer_df"]["section"] == row["section"])
            ]
        rec = _build_recommendation(row, opt_rows)
        results.append(RecommendationResponse(**rec))

    return results


@app.get("/market/{game_id}", response_model=list[MarketHealthResponse])
async def get_market_health(game_id: str):
    """Get secondary market health signals for all sections of a game."""
    _ensure_models_loaded()
    df = _get_features(game_id=game_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    results = []
    for _, row in df.iterrows():
        sold_avg = float(row.get("sold_price_avg", 0) or 0)
        face = float(row.get("face_price", 60) or 60)
        results.append(MarketHealthResponse(
            game_id=str(row["game_id"]),
            section=str(row["section"]),
            tier=str(row["tier"]),
            face_price=face,
            sold_price_avg=sold_avg,
            listing_price_avg=float(row.get("listing_price_avg", 0) or 0) or None,
            secondary_premium_pct=float(row.get("secondary_premium_pct", 0) or 0),
            market_health=str(row.get("market_health", "healthy")),
            n_sold_transactions=int(row.get("n_sold_transactions_est", 0) or 0),
            n_active_listings=int(row.get("g4_n_transactions", 0) or 0),
            sth_resale_margin=float(row.get("sth_resale_margin", 0) or 0),
            sth_resale_margin_pct=float(row.get("secondary_premium_pct", 0) or 0),
            is_underpriced=bool((sold_avg / face) > 1.20 if face > 0 else False),
            revenue_opp_per_seat=float(row.get("revenue_opp_per_seat", 0) or 0),
        ))
    return results


@app.get("/gap", response_model=SeasonOpportunityResponse)
async def get_price_gap_season(
    season: int = Query(default=2026),
    tier: Optional[str] = Query(default=None),
    health_filter: Optional[str] = Query(default=None, description="cold|healthy|warm|hot"),
):
    """Season-wide price gap analysis with revenue opportunity ranking."""
    _ensure_models_loaded()

    gap_df = _state["gap_df"]
    if gap_df is None:
        from src.price_gap.gap_calculator import compute_price_gap
        gap_df = compute_price_gap(_state["features_df"])

    df = gap_df[gap_df["season"] == season].copy()
    if tier:
        df = df[df["tier"] == tier]
    if health_filter:
        df = df[df["market_health"] == health_filter]

    df = df.sort_values("revenue_opp_total", ascending=False)

    top_rows = []
    for _, row in df.head(20).iterrows():
        top_rows.append(PriceGapRow(
            game_id=str(row.get("game_id", "")),
            section=str(row.get("section", "")),
            tier=str(row.get("tier", "")),
            face_price=float(row.get("face_price", 0)),
            optimal_price=float(row.get("optimal_face_from_secondary", row.get("face_price", 0)) or 0),
            gap_abs=float(row.get("price_gap_abs", 0) or 0),
            gap_pct=float(row.get("price_gap_pct", 0) or 0),
            revenue_opportunity=float(row.get("revenue_opp_total", 0) or 0),
            opportunity_tier=str(row.get("opportunity_tier", "low")),
            market_health=str(row.get("market_health", "healthy")),
            sold_price_avg=float(row.get("sold_price_avg", 0) or 0),
            sth_health=str(row.get("sth_health_current", "healthy")),
            alert_hot=bool(row.get("alert_hot_market", False)),
            alert_cold=bool(row.get("alert_cold_market", False)),
            alert_backlash=bool(row.get("alert_backlash_risk", False)),
        ))

    return SeasonOpportunityResponse(
        season=season,
        total_opportunity=float(df["revenue_opp_total"].sum()),
        n_games=int(df["game_id"].nunique()),
        n_underpriced_sections=int(df["is_underpriced"].sum()),
        top_opportunities=top_rows,
        retrospective_2025=_state.get("retrospective"),
    )


@app.get("/gap/{game_id}", response_model=list[PriceGapRow])
async def get_price_gap_game(game_id: str):
    """Price gap analysis for all sections of a specific game."""
    _ensure_models_loaded()

    gap_df = _state["gap_df"]
    if gap_df is None:
        from src.price_gap.gap_calculator import compute_price_gap
        gap_df = compute_price_gap(_state["features_df"])

    df = gap_df[gap_df["game_id"] == game_id].sort_values("revenue_opp_total", ascending=False)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found")

    return [
        PriceGapRow(
            game_id=str(row.get("game_id", "")),
            section=str(row.get("section", "")),
            tier=str(row.get("tier", "")),
            face_price=float(row.get("face_price", 0)),
            optimal_price=float(row.get("optimal_face_from_secondary", row.get("face_price", 0)) or 0),
            gap_abs=float(row.get("price_gap_abs", 0) or 0),
            gap_pct=float(row.get("price_gap_pct", 0) or 0),
            revenue_opportunity=float(row.get("revenue_opp_total", 0) or 0),
            opportunity_tier=str(row.get("opportunity_tier", "low")),
            market_health=str(row.get("market_health", "healthy")),
            sold_price_avg=float(row.get("sold_price_avg", 0) or 0),
            sth_health=str(row.get("sth_health_current", "healthy")),
            alert_hot=bool(row.get("alert_hot_market", False)),
            alert_cold=bool(row.get("alert_cold_market", False)),
            alert_backlash=bool(row.get("alert_backlash_risk", False)),
        )
        for _, row in df.iterrows()
    ]


@app.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(season: int = Query(default=2026)):
    """Get all active pricing alerts for the season."""
    _ensure_models_loaded()

    df = _get_features(season=season)
    if df.empty:
        return []

    alerts = []

    # Hot market alerts
    hot = df[df["is_hot_market_alert"] == True]
    for _, row in hot.iterrows():
        alerts.append(AlertResponse(
            alert_type="hot_market",
            severity="warning",
            game_id=str(row["game_id"]),
            section=str(row["section"]),
            tier=str(row["tier"]),
            message=(
                f"Section {row['section']} vs {row.get('opponent', 'TBD')} is HOT — "
                f"secondary at {row.get('secondary_premium_pct', 0):.0f}% above face. "
                f"Estimated ${row.get('revenue_opp_per_seat', 0):.0f}/seat opportunity."
            ),
            recommended_action=(
                f"Raise price from ${row.get('face_price', 0):.0f} to "
                f"${row.get('optimal_price_increase', 0) + row.get('face_price', 0):.0f} "
                f"(Balanced scenario)."
            ),
            data={
                "face_price": float(row.get("face_price", 0)),
                "sold_price_avg": float(row.get("sold_price_avg", 0) or 0),
                "secondary_premium_pct": float(row.get("secondary_premium_pct", 0) or 0),
                "revenue_opp_per_seat": float(row.get("revenue_opp_per_seat", 0) or 0),
            }
        ))

    # Cold market alerts
    cold = df[df["market_health"] == "cold"] if "market_health" in df.columns else pd.DataFrame()
    for _, row in cold.iterrows():
        premium = float(row.get("secondary_premium_pct", 0) or 0)
        face = float(row.get("face_price", 0) or 0)
        alerts.append(AlertResponse(
            alert_type="cold_market",
            severity="info",
            game_id=str(row["game_id"]),
            section=str(row["section"]),
            tier=str(row["tier"]),
            message=(
                f"Section {row['section']} vs {row.get('opponent', 'TBD')} is COLD — "
                f"secondary at {premium:.0f}% vs face (${face:.0f}). "
                f"Demand below expectations. Consider promotional pricing or bundle."
            ),
            recommended_action="Hold price or apply targeted discount/bundle to drive sell-through.",
            data={
                "face_price": face,
                "secondary_premium_pct": premium,
                "market_health": str(row.get("market_health", "cold")),
            }
        ))

    # Backlash risk alerts
    backlash = df[df["backlash_risk_score"] >= 7]
    for _, row in backlash.iterrows():
        alerts.append(AlertResponse(
            alert_type="backlash_risk",
            severity="critical",
            game_id=str(row["game_id"]),
            section=str(row["section"]),
            tier=str(row["tier"]),
            message=(
                f"Section {row['section']} vs {row.get('opponent', 'TBD')}: "
                f"AI recommends >{row.get('optimal_price_increase', 0):.0f} increase. "
                f"Backlash Risk Score: {row.get('backlash_risk_score', 0)}/10. "
                f"Manual review required."
            ),
            recommended_action="Review carefully before implementing. Consider phased increase.",
            data={"backlash_risk_score": int(row.get("backlash_risk_score", 0) or 0)},
        ))

    # Sort by severity
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 2))
    return alerts


@app.get("/retrospective/2025")
async def get_retrospective_2025():
    """2025 Season Retrospective: revenue left on table in SD FC's inaugural season."""
    _ensure_models_loaded()
    retro = _state.get("retrospective")
    if not retro:
        from src.price_gap.gap_calculator import compute_retrospective_2025
        retro = compute_retrospective_2025(_state["features_df"])
        _state["retrospective"] = retro
    return retro
