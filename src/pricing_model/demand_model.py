"""
Layer 1: XGBoost Demand Forecasting Model.
Predicts demand_index (normalized attendance) per section × game.

CV method: TimeSeriesSplit with 14-day purge gap to prevent data leakage.
Target: MAPE < 15% on holdout season.
SHAP values computed for every prediction → drives plain-English explanations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from loguru import logger
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[2]
FEAT_DIR  = ROOT / "data" / "features"
MODEL_DIR = ROOT / "mlflow"
FEAT_FILE = FEAT_DIR / "demand_features.parquet"

MLFLOW_TRACKING_URI = (ROOT / "mlflow").as_uri()
EXPERIMENT_NAME = "sdfc_demand_model"

# Feature columns for XGBoost (all g1–g14 prefixed columns)
TARGET_COL = "target_demand_index"

XGB_PARAMS = {
    "n_estimators":     400,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.85,
    "colsample_bytree": 0.80,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "reg:squarederror",
    "tree_method":      "hist",
    "random_state":     42,
    "n_jobs":           -1,
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all ML feature columns (g1–g14 prefixed, numeric only)."""
    cols = [c for c in df.columns if c.startswith("g") and c[1].isdigit() and c[2] == "_"]
    # Keep only numeric
    numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return sorted(numeric)


def _time_based_split(df: pd.DataFrame, n_splits: int = 3, gap_days: int = 14):
    """
    Time-series split with purge gap.
    Sorts by date, ensures no future data leakage.
    """
    df = df.sort_values("date").copy()
    dates = pd.to_datetime(df["date"])
    unique_dates = dates.sort_values().unique()
    n = len(unique_dates)
    fold_size = n // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_cutoff = unique_dates[(i + 1) * fold_size]
        gap_cutoff   = train_cutoff + pd.Timedelta(days=gap_days)
        test_cutoff  = unique_dates[min((i + 2) * fold_size, n - 1)]

        train_idx = df[dates < train_cutoff].index
        test_idx  = df[(dates > gap_cutoff) & (dates <= test_cutoff)].index

        if len(train_idx) > 10 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def train(
    df: Optional[pd.DataFrame] = None,
    log_to_mlflow: bool = True,
) -> tuple[xgb.XGBRegressor, shap.TreeExplainer, dict]:
    """
    Train XGBoost demand model with time-series CV.

    Returns:
        (model, shap_explainer, metrics)
    """
    if df is None:
        df = pd.read_parquet(FEAT_FILE)

    # Prefer 2025 actuals; fall back to all available data when 2025 is absent
    train_df = df[df["season"] == 2025].copy()
    if len(train_df) < 50:
        logger.warning(f"Only {len(train_df)} 2025 rows — training on all available seasons")
        train_df = df.copy()
    logger.info(f"Training on {len(train_df):,} rows")

    feature_cols = _get_feature_cols(train_df)
    logger.info(f"Feature count: {len(feature_cols)}")

    X = train_df[feature_cols].fillna(0)
    y = train_df[TARGET_COL].fillna(0.80)

    # Time-series cross-validation
    splits = _time_based_split(train_df)
    cv_maes, cv_mapes = [], []

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X.loc[train_idx], X.loc[val_idx]
        y_tr, y_val = y.loc[train_idx], y.loc[val_idx]

        model_cv = xgb.XGBRegressor(**XGB_PARAMS)
        model_cv.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model_cv.predict(X_val)
        mae  = float(np.mean(np.abs(preds - y_val)))
        mape = float(mean_absolute_percentage_error(y_val, preds)) * 100
        cv_maes.append(mae)
        cv_mapes.append(mape)
        logger.info(f"  Fold {fold_i + 1}: MAE={mae:.4f}, MAPE={mape:.1f}%")

    avg_mape = float(np.mean(cv_mapes))
    avg_mae  = float(np.mean(cv_maes))
    logger.info(f"CV MAPE: {avg_mape:.1f}% (target: <15%)")

    # Final model on full 2025 data
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X, y, verbose=False)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.head(100))
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_cols
    ).sort_values(ascending=False)

    metrics = {
        "cv_mape": avg_mape,
        "cv_mae":  avg_mae,
        "cv_folds": len(splits),
        "n_features": len(feature_cols),
        "n_train_rows": len(train_df),
        "top_features": mean_abs_shap.head(10).to_dict(),
    }

    logger.success(f"Model trained — CV MAPE: {avg_mape:.1f}%, Features: {len(feature_cols)}")
    logger.info(f"Top 5 features by SHAP: {list(mean_abs_shap.head(5).index)}")

    if log_to_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="xgboost_demand_v1"):
            mlflow.log_params(XGB_PARAMS)
            mlflow.log_metrics({
                "cv_mape": avg_mape,
                "cv_mae":  avg_mae,
            })
            mlflow.xgboost.log_model(model, artifact_path="demand_model")
            mlflow.log_dict(metrics, "metrics.json")
            logger.info("Logged to MLflow")

    return model, explainer, metrics


def predict(
    model: xgb.XGBRegressor,
    explainer: shap.TreeExplainer,
    df: pd.DataFrame,
    top_n_shap: int = 5,
) -> pd.DataFrame:
    """
    Generate demand predictions with SHAP explanations.

    Returns df with columns:
        demand_pred, demand_pred_pct, shap_top5 (list of (feature, value, impact) tuples)
    """
    feature_cols = _get_feature_cols(df)
    X = df[feature_cols].fillna(0)

    preds = model.predict(X)
    shap_vals = explainer.shap_values(X)

    results = df[["game_id", "section", "tier", "season", "date", "opponent"]].copy()
    results["demand_pred"]     = np.clip(preds, 0.0, 1.0).round(4)
    results["demand_pred_pct"] = (results["demand_pred"] * 100).round(1)

    # SHAP top-N per row
    shap_df = pd.DataFrame(shap_vals, columns=feature_cols, index=df.index)

    def top_shap(row_shap: pd.Series) -> list:
        abs_vals = row_shap.abs().nlargest(top_n_shap)
        return [
            {
                "feature": feat,
                "impact": round(float(row_shap[feat]), 4),
                "direction": "+" if row_shap[feat] > 0 else "-",
            }
            for feat in abs_vals.index
        ]

    results["shap_top5"] = [top_shap(shap_df.iloc[i]) for i in range(len(shap_df))]
    return results


def generate_shap_explanation(
    shap_top5: list[dict],
    face_price: float,
    optimal_price: float,
    game_context: dict,
) -> str:
    """
    Convert SHAP values into a plain-English explanation for the dashboard.
    This is the differentiating feature — no competitor does this.
    """
    section = game_context.get("section", "this section")
    opponent = game_context.get("opponent", "this opponent")
    gap = round(optimal_price - face_price, 0)

    # Build driver sentences
    driver_sentences = []
    feature_labels = {
        "g2_rivalry_tier1":       f"{opponent} rivalry",
        "g2_is_marquee":          f"marquee match ({opponent})",
        "g1_is_baja_cup":         "Baja Cup cross-border demand",
        "g1_is_saturday":         "Saturday premium",
        "g4_is_hot_market":       "secondary market demand (HOT)",
        "g4_secondary_premium_pct": "secondary market premium",
        "g5_sell_through_t7":     "high advance sell-through",
        "g3_star_player_on_opp":  "star player visit",
        "g8_weather_demand_impact": "favorable weather",
        "g2_opponent_tier":       "opponent quality",
        "g14_cross_border_index": "cross-border fan demand",
        "g1_is_season_opener":    "season opener demand",
        "g13_social_sentiment":   "strong fan buzz",
    }

    for item in shap_top5:
        feat = item["feature"]
        impact = item["impact"]
        direction = item["direction"]
        label = feature_labels.get(feat, feat.replace("g", "").replace("_", " ").split(" ", 1)[-1])

        if abs(impact) > 0.01:
            price_effect = round(abs(impact) * face_price * 0.3, 0)
            driver_sentences.append(
                f"{label} ({direction}${price_effect:.0f})"
            )

    if not driver_sentences:
        return f"Section priced ${gap:+.0f} vs optimal. Market signals suggest adjustment."

    drivers_str = ", ".join(driver_sentences[:4])

    if gap > 0:
        return (
            f"Section {section}: priced ${gap:.0f} below optimal. "
            f"Drivers: {drivers_str}. "
            f"Raising to ${optimal_price:.0f} adds revenue while keeping secondary market healthy."
        )
    elif gap < -5:
        return (
            f"Section {section}: priced ${abs(gap):.0f} above optimal. "
            f"Consider a modest reduction to improve sell-through."
        )
    else:
        return f"Section {section}: pricing is within optimal range. No change recommended."


def run() -> tuple[xgb.XGBRegressor, shap.TreeExplainer, dict]:
    model, explainer, metrics = train()
    logger.info(json.dumps(metrics, indent=2, default=str))
    return model, explainer, metrics


if __name__ == "__main__":
    run()
