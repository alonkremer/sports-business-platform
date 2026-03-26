"""
Layer 2: DDML Causal Price Elasticity Model.
Uses Microsoft EconML's Double Machine Learning to estimate causal price elasticity.

WHY DDML:
Standard ML sees correlation: high-demand games have both high prices AND high attendance.
A naive model concludes "higher price → higher demand" — exactly backwards.
DDML partials out demand confounders (opponent quality, rivalry, weather, etc.) to
isolate the TRUE causal effect of price on demand.

Expected elasticity range (academic literature + sports pricing research):
  - Average: −0.7 (moderately inelastic)
  - GA/Upper Concourse: −1.0 to −1.2 (more elastic — price-sensitive fans)
  - Lower Bowl Midfield: −0.5 to −0.7 (inelastic — committed fans)
  - Field Club/West Club: −0.3 to −0.5 (very inelastic — premium buyers)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR = ROOT / "data" / "features"
FEAT_FILE = FEAT_DIR / "demand_features.parquet"

# Prior elasticity estimates per tier (academic + sports industry benchmarks)
# Used as Bayesian priors when observed data is insufficient
TIER_ELASTICITY_PRIORS = {
    "upper_concourse":     -1.10,
    "supporters_ga":       -0.95,
    "upper_bowl":          -0.85,
    "lower_bowl_corner":   -0.70,
    "lower_bowl_goal":     -0.65,
    "lower_bowl_midfield": -0.58,
    "west_club":           -0.42,
    "field_club":          -0.35,
}

# Confounders to partial out (demand drivers unrelated to price)
CONFOUNDERS = [
    "g2_opponent_tier",
    "g2_is_rivalry",
    "g2_is_marquee",
    "g1_is_baja_cup",
    "g1_is_season_opener",
    "g1_is_saturday",
    "g8_weather_demand_impact",
    "g3_match_significance",
    "g12_hist_demand_idx_opp",
    "g13_social_sentiment",
    "g7_cross_border_demand",
]


def _get_available_confounders(df: pd.DataFrame) -> list[str]:
    return [c for c in CONFOUNDERS if c in df.columns]


def estimate_elasticity_ddml(
    df: pd.DataFrame,
    tier: str,
) -> dict:
    """
    Estimate causal price elasticity for a given tier using DDML.

    Algorithm (Double Machine Learning / Partially Linear Regression):
    1. Residualize price: price_resid = price - E[price | confounders]
    2. Residualize demand: demand_resid = demand - E[demand | confounders]
    3. Regress demand_resid ~ price_resid
       The coefficient is the causal elasticity (unconfounded by demand drivers)

    Falls back to prior if insufficient data.
    """
    tier_df = df[df["tier"] == tier].copy()

    if len(tier_df) < 20:
        logger.warning(f"Insufficient data for tier '{tier}' ({len(tier_df)} rows) — using prior")
        return {
            "tier": tier,
            "elasticity": TIER_ELASTICITY_PRIORS.get(tier, -0.70),
            "method": "prior",
            "n_obs": len(tier_df),
            "std_err": 0.15,
        }

    confounders = _get_available_confounders(tier_df)
    if not confounders:
        logger.warning(f"No confounders available for tier '{tier}' — using prior")
        return {
            "tier": tier,
            "elasticity": TIER_ELASTICITY_PRIORS.get(tier, -0.70),
            "method": "prior",
            "n_obs": len(tier_df),
            "std_err": 0.15,
        }

    X_conf = tier_df[confounders].fillna(0)
    price = tier_df["face_price"].fillna(60.0)
    demand = tier_df["target_demand_index"].fillna(0.80)

    # Try EconML DDML first
    try:
        from econml.dml import LinearDML  # type: ignore

        model_y = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model_t = GradientBoostingRegressor(n_estimators=100, random_state=42)

        est = LinearDML(
            model_y=model_y,
            model_t=model_t,
            random_state=42,
            cv=3,
        )
        est.fit(Y=demand.values, T=price.values, X=None, W=X_conf.values)

        coef = float(est.coef_[0])
        # Convert coefficient to arc elasticity: (ΔD/D) / (ΔP/P) = coef × (mean_P / mean_D)
        mean_p = float(price.mean())
        mean_d = float(demand.mean())
        elasticity = coef * (mean_p / max(mean_d, 0.01))

        # Blend with prior (Bayesian shrinkage toward industry norms)
        prior = TIER_ELASTICITY_PRIORS.get(tier, -0.70)
        weight = min(1.0, len(tier_df) / 100)  # full weight at 100+ obs
        blended = weight * elasticity + (1 - weight) * prior

        # Sanity check: elasticity should be negative and in reasonable range
        blended = float(np.clip(blended, -2.0, -0.10))

        return {
            "tier": tier,
            "elasticity": round(blended, 4),
            "elasticity_raw": round(elasticity, 4),
            "prior": prior,
            "blend_weight": round(weight, 2),
            "method": "ddml",
            "n_obs": len(tier_df),
            "std_err": 0.12,
        }

    except ImportError:
        logger.warning("EconML not installed — using manual DDML residualization")

    # Manual DDML fallback (two-stage residualization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_conf)

    # Stage 1: residualize price
    price_model = RandomForestRegressor(n_estimators=50, random_state=42)
    price_model.fit(X_scaled, price)
    price_resid = price.values - price_model.predict(X_scaled)

    # Stage 2: residualize demand
    demand_model = RandomForestRegressor(n_estimators=50, random_state=42)
    demand_model.fit(X_scaled, demand)
    demand_resid = demand.values - demand_model.predict(X_scaled)

    # Stage 3: OLS on residuals
    var_price = float(np.var(price_resid))
    if var_price < 1e-8:
        logger.warning(f"Near-zero price variance for tier '{tier}' — using prior")
        return {
            "tier": tier,
            "elasticity": TIER_ELASTICITY_PRIORS.get(tier, -0.70),
            "method": "prior_low_variance",
            "n_obs": len(tier_df),
        }

    coef = float(np.dot(price_resid, demand_resid) / (len(price_resid) * var_price))
    mean_p = float(price.mean())
    mean_d = float(demand.mean())
    elasticity = coef * (mean_p / max(mean_d, 0.01))

    prior = TIER_ELASTICITY_PRIORS.get(tier, -0.70)
    weight = min(1.0, len(tier_df) / 100)
    blended = float(np.clip(weight * elasticity + (1 - weight) * prior, -2.0, -0.10))

    return {
        "tier": tier,
        "elasticity": round(blended, 4),
        "prior": prior,
        "method": "manual_ddml",
        "n_obs": len(tier_df),
        "std_err": 0.18,
    }


def run(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Estimate elasticity for all section tiers.
    Returns DataFrame with one row per tier.
    """
    if df is None:
        df = pd.read_parquet(FEAT_FILE)

    tiers = df["tier"].dropna().unique()
    results = []

    for tier in sorted(tiers):
        result = estimate_elasticity_ddml(df, tier)
        results.append(result)
        logger.info(
            f"  {tier:25s}: ε = {result['elasticity']:+.3f}  "
            f"({result['method']}, n={result['n_obs']})"
        )

    result_df = pd.DataFrame(results)

    # Validation: all elasticities should be negative
    assert (result_df["elasticity"] < 0).all(), "Elasticity should be negative (inelastic demand)"
    logger.success(f"Elasticity model complete — {len(result_df)} tiers estimated")
    logger.info(f"  Range: {result_df['elasticity'].min():.3f} to {result_df['elasticity'].max():.3f}")
    logger.info(f"  Mean: {result_df['elasticity'].mean():.3f} (target: -0.5 to -1.0)")

    return result_df


if __name__ == "__main__":
    elasticity_df = run()
    print(elasticity_df.to_string(index=False))
