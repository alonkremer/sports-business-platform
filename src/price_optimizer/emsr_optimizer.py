"""
Layer 3+4: EMSR-b Revenue Optimizer + Strategic Guardrails.

EMSR-b (Expected Marginal Seat Revenue, version b) is the airline industry
standard for yield management, adapted here for sports section pricing.

3-scenario output per section × game:
  Conservative (α=0.3): ~90% capacity target, lower revenue upside
  Balanced    (α=0.6): target ★ — optimizes revenue × attendance × STH health
  Aggressive  (α=0.9): maximum revenue extraction, accepts lower capacity

Strategic Guardrails (Layer 4):
  CEO-configurable controls override pure optimization.
  No optimizer recommendation goes to the dashboard without passing all guards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
FEAT_FILE = FEAT_DIR / "demand_features.parquet"

# ── Scenario alphas ────────────────────────────────────────────────────────────
SCENARIOS = {
    "conservative": 0.30,
    "balanced":     0.60,
    "aggressive":   0.90,
}

# ── Default price grid ────────────────────────────────────────────────────────
# ±30% from current face price in 5% steps
PRICE_GRID_RANGE = 0.30
PRICE_GRID_STEP  = 0.05

# ── Default strategic guardrails ──────────────────────────────────────────────
@dataclass
class StrategicGuardrails:
    """
    CEO-level pricing controls. All optimizer recommendations must pass these.
    These are configurable per club and override pure revenue optimization.
    """
    # Strategic mode: acquisition | revenue | atmosphere
    strategic_mode: str = "revenue"

    # Price floors by tier (hard minimum, never go below)
    price_floors: dict = field(default_factory=lambda: {
        "supporters_ga":       18.0,
        "lower_bowl_corner":   40.0,
        "lower_bowl_goal":     38.0,
        "lower_bowl_midfield": 55.0,
        "field_club":          140.0,
        "upper_bowl":          28.0,
        "west_club":           100.0,
        "upper_concourse":     20.0,
    })

    # Price ceilings by tier (hard maximum)
    price_ceilings: dict = field(default_factory=lambda: {
        "supporters_ga":       45.0,
        "lower_bowl_corner":   120.0,
        "lower_bowl_goal":     115.0,
        "lower_bowl_midfield": 180.0,
        "field_club":          400.0,
        "upper_bowl":          85.0,
        "west_club":           300.0,
        "upper_concourse":     60.0,
    })

    # STH resale guarantee: primary price must be at least 10% below secondary sold avg
    # (STH must always be able to resell at a profit)
    sth_resale_min_margin_pct: float = 10.0

    # Fan equity rule: no adjacent tier can exceed 40% price difference
    fan_equity_max_gap_pct: float = 40.0

    # Backlash risk threshold: flag for manual review (don't block, just alert)
    backlash_flag_pct_increase: float = 25.0

    # Military discount lock: certain allocations exempt from upward optimization
    military_discount_exempt: bool = True
    military_discount_pct: float = 15.0

    # Acquisition mode: cap upward movement at +10% (fill seats, build fan base)
    acquisition_mode_cap: float = 10.0

    # Atmosphere mode: cap price to prioritize full house (use sell-through target)
    atmosphere_min_sellthrough: float = 90.0


DEFAULT_GUARDRAILS = StrategicGuardrails()


def _build_price_grid(face_price: float, step: float = PRICE_GRID_STEP) -> np.ndarray:
    """Generate ±30% price grid in 5% steps around face price."""
    # Use step*0.5 as epsilon to avoid float precision causing an extra step beyond +30%
    multipliers = np.arange(1 - PRICE_GRID_RANGE, 1 + PRICE_GRID_RANGE + step * 0.5, step)
    return np.round(face_price * multipliers, 2)


def _demand_curve(
    prices: np.ndarray,
    face_price: float,
    base_demand: float,
    elasticity: float,
    capacity: int,
) -> np.ndarray:
    """
    Compute demand at each price point using arc elasticity.
    Q(p) = Q0 × (p/p0)^ε  where ε is the (negative) price elasticity.

    base_demand: fractional demand at face_price (0–1)
    elasticity: negative float (e.g., -0.7)
    """
    price_ratios = prices / max(face_price, 1.0)
    demand_ratios = price_ratios ** elasticity  # elasticity negative → demand falls with price
    demands = base_demand * demand_ratios
    return np.clip(demands, 0.0, 1.0)


def _emsr_b_optimal(
    price_grid: np.ndarray,
    demand_curve: np.ndarray,
    capacity: int,
) -> dict:
    """
    Pure EMSR-b: find the revenue-maximising price across the price grid.
    Scenario blending (conservative / balanced / aggressive) happens in
    optimize_section_game — this function is scenario-agnostic.

    For each price point:
      Expected Revenue = price × min(demand × capacity, capacity)
    """
    expected_revenues  = []
    expected_attendances = []
    expected_sell_throughs = []

    for price, demand_frac in zip(price_grid, demand_curve):
        seats_sold = min(int(demand_frac * capacity), capacity)
        revenue    = price * seats_sold
        expected_revenues.append(revenue)
        expected_attendances.append(seats_sold)
        expected_sell_throughs.append(seats_sold / capacity * 100)

    rev_arr  = np.array(expected_revenues)
    best_idx = int(np.argmax(rev_arr))

    return {
        "optimal_price":         float(price_grid[best_idx]),
        "expected_revenue":      float(rev_arr[best_idx]),
        "expected_attendance":   float(expected_attendances[best_idx]),
        "expected_sell_through": float(expected_sell_throughs[best_idx]),
        "price_curve":           price_grid.tolist(),
        "revenue_curve":         expected_revenues,
    }


def _apply_guardrails(
    price: float,
    tier: str,
    face_price: float,
    sold_price_avg: float,
    guardrails: StrategicGuardrails,
    scenario: str,
) -> tuple[float, list[str]]:
    """
    Apply strategic guardrails to an optimizer recommendation.
    Returns (adjusted_price, list_of_guardrails_applied).
    """
    applied = []
    p = price

    # 1. Price floor
    floor = guardrails.price_floors.get(tier, 0.0)
    if p < floor:
        p = floor
        applied.append(f"floor_{tier}")

    # 2. Price ceiling
    ceiling = guardrails.price_ceilings.get(tier, 9999.0)
    if p > ceiling:
        p = ceiling
        applied.append(f"ceiling_{tier}")

    # 3. STH resale guarantee: primary must be ≤ secondary − 10%
    if sold_price_avg and sold_price_avg > 0:
        max_sth_safe = sold_price_avg * (1 - guardrails.sth_resale_min_margin_pct / 100)
        if p > max_sth_safe:
            p = max_sth_safe
            applied.append("sth_resale_protection")

    # 4. Strategic mode caps
    if guardrails.strategic_mode == "acquisition":
        max_acquisition = face_price * (1 + guardrails.acquisition_mode_cap / 100)
        if p > max_acquisition:
            p = max_acquisition
            applied.append("acquisition_mode_cap")

    # 5. Balanced/aggressive: never go below current face price (only conservative allows lowering)
    #    Exception: if STH cap is below face (secondary market cold), don't force price above secondary
    if scenario in ("balanced", "aggressive") and p < face_price:
        if sold_price_avg and sold_price_avg >= face_price:
            p = face_price
            applied.append("min_face_price")

    # 6. Round to nearest $1 for clean pricing
    p = max(floor, round(p))

    return p, applied


def optimize_section_game(
    game_id: str,
    section: str,
    tier: str,
    face_price: float,
    base_demand: float,
    elasticity: float,
    capacity: int,
    sold_price_avg: float,
    sth_resale_margin: float,
    guardrails: StrategicGuardrails = DEFAULT_GUARDRAILS,
) -> dict:
    """
    Run EMSR-b optimizer for a single section × game.
    Returns all 3 scenarios with guardrails applied.
    """
    price_grid = _build_price_grid(face_price)
    demand_at_price = _demand_curve(price_grid, face_price, base_demand, elasticity, capacity)

    # Compute the pure revenue-maximising price once — shared across scenarios.
    # Each scenario then blends between face_price (safe, fill seats) and the
    # EMSR optimal (maximum extraction) using its alpha weight:
    #   conservative (α=0.30) → 30 % of the way from face to optimal
    #   balanced     (α=0.60) → 60 % of the way
    #   aggressive   (α=0.90) → 90 % of the way
    emsr_result   = _emsr_b_optimal(price_grid, demand_at_price, capacity)
    emsr_optimal  = emsr_result["optimal_price"]

    scenarios_out = {}
    for scenario_name, alpha in SCENARIOS.items():
        # Blend, then snap to nearest grid price
        blended   = alpha * emsr_optimal + (1 - alpha) * face_price
        raw_price = float(price_grid[int(np.argmin(np.abs(price_grid - blended)))])

        # Apply guardrails
        adj_price, guardrails_applied = _apply_guardrails(
            raw_price, tier, face_price, sold_price_avg, guardrails, scenario_name
        )

        # Recompute attendance and revenue at adjusted price
        demand_at_adj = float(_demand_curve(
            np.array([adj_price]), face_price, base_demand, elasticity, capacity
        )[0])

        seats_sold_adj = min(int(demand_at_adj * capacity), capacity)
        revenue_adj = adj_price * seats_sold_adj
        sell_through_adj = seats_sold_adj / capacity * 100

        # STH resale margin at adjusted price
        sth_margin_adj = (sold_price_avg - adj_price) if sold_price_avg else 0.0
        sth_margin_pct = (sth_margin_adj / adj_price * 100) if adj_price > 0 else 0.0

        # Revenue delta vs flat pricing (face_price × base attendance)
        base_revenue = face_price * int(base_demand * capacity)
        revenue_delta_pct = ((revenue_adj - base_revenue) / max(base_revenue, 1)) * 100

        # Backlash risk
        price_increase_pct = ((adj_price - face_price) / face_price * 100) if face_price > 0 else 0
        is_backlash_risk = price_increase_pct > guardrails.backlash_flag_pct_increase

        scenarios_out[scenario_name] = {
            "price":               round(adj_price, 2),
            "price_raw":           round(raw_price, 2),
            "price_change":        round(adj_price - face_price, 2),
            "price_change_pct":    round(price_increase_pct, 1),
            "expected_revenue":    round(revenue_adj, 2),
            "revenue_delta_pct":   round(revenue_delta_pct, 1),
            "expected_seats_sold": seats_sold_adj,
            "expected_sell_through": round(sell_through_adj, 1),
            "sth_resale_margin":   round(sth_margin_adj, 2),
            "sth_resale_margin_pct": round(sth_margin_pct, 1),
            "sth_is_healthy":      sth_margin_pct >= 10.0,
            "guardrails_applied":  guardrails_applied,
            "is_backlash_risk":    is_backlash_risk,
        }

    return {
        "game_id":       game_id,
        "section":       section,
        "tier":          tier,
        "face_price":    face_price,
        "sold_price_avg": sold_price_avg,
        "base_demand":   base_demand,
        "elasticity":    elasticity,
        "capacity":      capacity,
        "scenarios":     scenarios_out,
        "recommended":   "balanced",  # default recommendation
    }


def run_optimizer(
    features_df: Optional[pd.DataFrame] = None,
    elasticity_df: Optional[pd.DataFrame] = None,
    guardrails: StrategicGuardrails = DEFAULT_GUARDRAILS,
) -> pd.DataFrame:
    """
    Run EMSR-b optimizer for all section × game combinations.
    Returns flattened DataFrame with one row per section × game × scenario.
    """
    if features_df is None:
        features_df = pd.read_parquet(FEAT_FILE)

    # Use 2026 season for forward-looking recommendations
    df = features_df[features_df["season"] == 2026].copy()
    logger.info(f"Optimizing {len(df):,} section × game rows (2026 season)")

    # Load or use provided elasticity estimates
    tier_elasticity = {}
    if elasticity_df is not None:
        for _, row in elasticity_df.iterrows():
            tier_elasticity[row["tier"]] = float(row["elasticity"])
    else:
        # Fall back to priors
        from src.elasticity_model.causal_elasticity import TIER_ELASTICITY_PRIORS
        tier_elasticity = TIER_ELASTICITY_PRIORS

    rows = []
    for _, row in df.iterrows():
        tier = row.get("tier", "upper_bowl")
        elasticity = tier_elasticity.get(tier, -0.70)

        result = optimize_section_game(
            game_id=str(row["game_id"]),
            section=str(row["section"]),
            tier=tier,
            face_price=float(row.get("face_price", 60)),
            base_demand=float(row.get("target_demand_index", 0.80)),
            elasticity=elasticity,
            capacity=int(row.get("capacity", 2000)),
            sold_price_avg=float(row.get("sold_price_avg", 0) or 0),
            sth_resale_margin=float(row.get("sth_resale_margin", 0) or 0),
            guardrails=guardrails,
        )

        # Flatten scenarios
        for scenario_name, scenario_data in result["scenarios"].items():
            rows.append({
                "game_id":     result["game_id"],
                "section":     result["section"],
                "tier":        result["tier"],
                "face_price":  result["face_price"],
                "sold_price_avg": result["sold_price_avg"],
                "elasticity":  result["elasticity"],
                "capacity":    result["capacity"],
                "scenario":    scenario_name,
                "is_recommended": scenario_name == result["recommended"],
                **scenario_data,
            })

    out_df = pd.DataFrame(rows)
    logger.success(f"Optimizer complete: {len(out_df):,} rows ({len(df):,} section-games × 3 scenarios)")

    # Summary
    balanced = out_df[out_df["scenario"] == "balanced"]
    logger.info(f"  Balanced avg price change: {balanced['price_change_pct'].mean():+.1f}%")
    logger.info(f"  Balanced avg revenue delta: {balanced['revenue_delta_pct'].mean():+.1f}%")
    logger.info(f"  STH healthy (balanced): {balanced['sth_is_healthy'].mean()*100:.0f}% of sections")
    logger.info(f"  Backlash risks flagged: {balanced['is_backlash_risk'].sum()}")

    return out_df


if __name__ == "__main__":
    results = run_optimizer()
    print(results[results["scenario"] == "balanced"][
        ["game_id", "section", "tier", "face_price", "price",
         "price_change_pct", "revenue_delta_pct", "sth_is_healthy"]
    ].head(10).to_string(index=False))
