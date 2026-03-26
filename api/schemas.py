"""
Pydantic schemas for FastAPI request/response validation.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ScenarioDetail(BaseModel):
    price: float
    price_change: float
    price_change_pct: float
    expected_revenue: float
    revenue_delta_pct: float
    expected_seats_sold: int
    expected_sell_through: float
    sth_resale_margin: float
    sth_resale_margin_pct: float
    sth_is_healthy: bool
    guardrails_applied: list[str]
    is_backlash_risk: bool


class RecommendationResponse(BaseModel):
    game_id: str
    section: str
    tier: str
    face_price: float
    sold_price_avg: float
    secondary_premium_pct: float
    market_health: str
    demand_pred: float
    demand_pred_pct: float
    optimal_price_increase: float
    opportunity_tier: str
    total_revenue_opportunity: float
    backlash_risk_score: int
    shap_explanation: str
    shap_top5: list[dict]
    scenarios: dict[str, ScenarioDetail]
    recommended_scenario: str = "balanced"


class GameSummary(BaseModel):
    game_id: str
    season: int
    date: str
    opponent: str
    opponent_tier: int
    is_rivalry: bool
    is_marquee: bool
    is_baja_cup: bool
    demand_index: float
    avg_secondary_premium_pct: float
    total_revenue_opportunity: float
    market_health_dominant: str
    backlash_risk_max: int
    is_hot_market: bool
    sections_hot: int


class MarketHealthResponse(BaseModel):
    game_id: str
    section: str
    tier: str
    face_price: float
    sold_price_avg: float
    listing_price_avg: Optional[float]
    secondary_premium_pct: float
    market_health: str
    n_sold_transactions: int
    n_active_listings: int
    sth_resale_margin: float
    sth_resale_margin_pct: float
    is_underpriced: bool
    revenue_opp_per_seat: float


class PriceGapRow(BaseModel):
    game_id: str
    section: str
    tier: str
    face_price: float
    optimal_price: float
    gap_abs: float
    gap_pct: float
    revenue_opportunity: float
    opportunity_tier: str
    market_health: str
    sold_price_avg: float
    sth_health: str
    alert_hot: bool
    alert_cold: bool
    alert_backlash: bool


class SeasonOpportunityResponse(BaseModel):
    season: int
    total_opportunity: float
    n_games: int
    n_underpriced_sections: int
    top_opportunities: list[PriceGapRow]
    retrospective_2025: Optional[dict] = None


class AlertResponse(BaseModel):
    alert_type: str  # hot_market | cold_market | backlash_risk | sth_margin_warning
    severity: str    # info | warning | critical
    game_id: str
    section: str
    tier: str
    message: str
    recommended_action: str
    data: dict[str, Any]


class GuardrailsConfig(BaseModel):
    strategic_mode: str = Field(default="revenue", pattern="^(acquisition|revenue|atmosphere)$")
    price_floors: Optional[dict[str, float]] = None
    price_ceilings: Optional[dict[str, float]] = None
    sth_resale_min_margin_pct: float = Field(default=10.0, ge=0, le=30)
    backlash_flag_pct_increase: float = Field(default=25.0, ge=5, le=100)
    military_discount_exempt: bool = True
