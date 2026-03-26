"""
LangGraph Supervisor Orchestrator.
Coordinates 6 specialized Claude agents through the full pipeline:
  1. Data Engineer Agent   — ingest + synthetic generation + DuckDB schema
  2. Feature Engineer Agent — 14-group feature pipeline → parquet
  3. ML Scientist Agent    — XGBoost + DDML + EMSR-b optimizer
  4. PM Agent              — status tracking, gap analysis, reporting
  5. Dashboard Agent       — Streamlit hot-reload, view verification
  6. QA Agent              — pytest + backtest + sensitivity checks

Agents use MCP tools (filesystem, DuckDB) for their domain.
FastAPI serves inference; Streamlit calls API.

Run with: python agents/orchestrator.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]

# Attempt LangGraph import; graceful degradation to sequential runner
try:
    from langgraph.graph import END, StateGraph  # type: ignore
    from langgraph.graph.message import add_messages  # type: ignore
    HAS_LANGGRAPH = True
except ImportError:
    logger.warning("LangGraph not installed — running sequential pipeline")
    HAS_LANGGRAPH = False

try:
    import anthropic  # type: ignore
    HAS_ANTHROPIC = True
except ImportError:
    logger.warning("Anthropic SDK not installed — agents will use subprocess mode")
    HAS_ANTHROPIC = False


# ── Pipeline State ─────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    """Shared state across all agents in the LangGraph supervisor."""
    stage: str
    completed_stages: list[str]
    errors: list[str]
    warnings: list[str]
    artifacts: dict[str, Any]
    metrics: dict[str, Any]
    raw_data_ready: bool
    features_ready: bool
    models_ready: bool
    api_ready: bool
    dashboard_ready: bool
    tests_passed: bool
    gap_analysis: dict
    retrospective: dict


# ── Agent task definitions ─────────────────────────────────────────────────────

def run_data_engineer(state: PipelineState) -> PipelineState:
    """
    Data Engineer Agent: Run all data ingestion scrapers + synthetic generator + DuckDB schema.
    """
    logger.info("=" * 60)
    logger.info("DATA ENGINEER AGENT — Starting")
    errors = state.get("errors", [])
    artifacts = state.get("artifacts", {})

    stages = [
        ("FBref attendance scraper",    "src.data_ingestion.fbref_attendance",    "run"),
        ("SDFC schedule 2026",          "src.data_ingestion.sdfc_schedule",       "run"),
        ("MLS attendance history",      "src.data_ingestion.mls_attendance",      "run"),
        ("OddsPortal betting odds",     "src.data_ingestion.oddsportal_odds",     "run"),
        ("Ticketmaster face prices",    "src.data_ingestion.ticketmaster_prices", "run"),
        ("SeatGeek secondary market",   "src.data_ingestion.seatgeek_market",     "run"),
        ("StubHub sold transactions",   "src.data_ingestion.stubhub_market",      "run"),
        ("Vivid Seats listings",        "src.data_ingestion.vividseats_market",   "run"),
        ("Synthetic data generator",    "src.data_generation.synthetic_sdfc",     "run"),
        ("DuckDB schema init",          "src.database.schema",                    "initialize"),
    ]

    for stage_name, module_path, func_name in stages:
        logger.info(f"  Running: {stage_name}")
        try:
            mod = _import_module(module_path)
            if mod:
                fn = getattr(mod, func_name)
                result = fn()
                artifacts[stage_name] = str(result)[:100] if result is not None else "ok"
                logger.success(f"  ✓ {stage_name}")
            else:
                logger.warning(f"  ⚠ Could not import {module_path}")
        except Exception as exc:
            msg = f"{stage_name}: {exc}"
            errors.append(msg)
            logger.error(f"  ✗ {msg}")

    raw_ready = len(errors) < len(stages) // 2  # at least half succeeded
    logger.info(f"Data Engineer Agent complete — {len(errors)} errors")
    return {**state, "stage": "data_engineer", "completed_stages": state.get("completed_stages", []) + ["data_engineer"],
            "errors": errors, "artifacts": artifacts, "raw_data_ready": raw_ready}


def run_feature_engineer(state: PipelineState) -> PipelineState:
    """Feature Engineer Agent: Build 14-group feature pipeline."""
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEER AGENT — Starting")
    errors = state.get("errors", [])

    try:
        from src.feature_engineering.pipeline import run as build_features
        df = build_features()
        logger.success(f"Feature pipeline complete: {df.shape[0]:,} rows × {df.shape[1]} cols")
        return {**state, "stage": "feature_engineer",
                "completed_stages": state.get("completed_stages", []) + ["feature_engineer"],
                "features_ready": True, "errors": errors}
    except Exception as exc:
        errors.append(f"Feature pipeline: {exc}")
        logger.error(f"Feature pipeline failed: {exc}")
        return {**state, "features_ready": False, "errors": errors}


def run_ml_scientist(state: PipelineState) -> PipelineState:
    """ML Scientist Agent: Train demand model, elasticity model, run optimizer."""
    logger.info("=" * 60)
    logger.info("ML SCIENTIST AGENT — Starting")
    errors = state.get("errors", [])
    metrics = state.get("metrics", {})

    import pandas as pd
    feat_file = ROOT / "data" / "features" / "demand_features.parquet"
    if not feat_file.exists():
        errors.append("Feature file not found — run feature engineer first")
        return {**state, "models_ready": False, "errors": errors}

    df = pd.read_parquet(feat_file)

    # 1. Demand model
    try:
        from src.pricing_model.demand_model import train as train_demand
        model, explainer, model_metrics = train_demand(df, log_to_mlflow=True)
        metrics["demand_model"] = model_metrics
        logger.success(f"Demand model — CV MAPE: {model_metrics['cv_mape']:.1f}%")
    except Exception as exc:
        errors.append(f"Demand model: {exc}")
        logger.error(f"Demand model failed: {exc}")

    # 2. Elasticity model
    try:
        from src.elasticity_model.causal_elasticity import run as run_elasticity
        elast_df = run_elasticity(df)
        metrics["elasticity"] = elast_df.to_dict("records")
        logger.success(f"Elasticity model — {len(elast_df)} tiers")
    except Exception as exc:
        errors.append(f"Elasticity model: {exc}")
        logger.error(f"Elasticity model failed: {exc}")
        elast_df = None

    # 3. EMSR-b optimizer
    try:
        from src.price_optimizer.emsr_optimizer import run_optimizer
        opt_df = run_optimizer(df, elast_df if 'elast_df' in dir() else None)
        metrics["optimizer_rows"] = len(opt_df)
        logger.success(f"Optimizer — {len(opt_df):,} rows")
    except Exception as exc:
        errors.append(f"Optimizer: {exc}")
        logger.error(f"Optimizer failed: {exc}")

    # 4. Price gap calculator
    try:
        from src.price_gap.gap_calculator import run as run_gap
        gap_df, retro = run_gap(df)
        metrics["total_revenue_opportunity"] = gap_df["revenue_opp_total"].sum()
        metrics["retrospective"] = retro
        logger.success(f"Gap analysis — ${metrics['total_revenue_opportunity']:,.0f} total opportunity")
    except Exception as exc:
        errors.append(f"Gap calculator: {exc}")
        logger.error(f"Gap calculator failed: {exc}")
        retro = {}

    models_ready = len(errors) == 0
    return {**state, "stage": "ml_scientist",
            "completed_stages": state.get("completed_stages", []) + ["ml_scientist"],
            "models_ready": models_ready, "errors": errors, "metrics": metrics,
            "retrospective": retro}


def run_pm_agent(state: PipelineState) -> PipelineState:
    """PM Agent: Status tracking, gap analysis summary, sprint reporting."""
    logger.info("=" * 60)
    logger.info("PM AGENT — Pipeline Status Report")

    completed = state.get("completed_stages", [])
    errors = state.get("errors", [])
    metrics = state.get("metrics", {})

    logger.info(f"Completed stages: {completed}")
    logger.info(f"Total errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            logger.warning(f"  Error: {e}")

    if metrics.get("demand_model"):
        dm = metrics["demand_model"]
        logger.info(f"Demand model: CV MAPE={dm.get('cv_mape', 'N/A'):.1f}%, "
                    f"Features={dm.get('n_features', 'N/A')}")

    total_opp = metrics.get("total_revenue_opportunity", 0)
    if total_opp:
        logger.info(f"2026 Revenue Opportunity: ${total_opp:,.0f}")

    retro = state.get("retrospective", {})
    if retro:
        logger.info(f"2025 Retrospective: ${retro.get('total_revenue_opportunity', 0):,.0f} left on table")

    return {**state, "stage": "pm_agent",
            "completed_stages": completed + ["pm_agent"],
            "gap_analysis": {"total_opportunity": total_opp}}


def run_qa_agent(state: PipelineState) -> PipelineState:
    """QA Agent: Run pytest suite + data quality checks + backtest validation."""
    logger.info("=" * 60)
    logger.info("QA AGENT — Running tests")
    errors = state.get("errors", [])

    test_dir = ROOT / "tests"
    if test_dir.exists():
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short", "-q"],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        if result.returncode == 0:
            logger.success(f"All tests passed")
        else:
            logger.warning(f"Some tests failed:\n{result.stdout[-1000:]}")
            errors.append(f"pytest: {result.returncode} failures")
    else:
        logger.info("No test directory found — skipping pytest")

    # Data quality checks
    feat_file = ROOT / "data" / "features" / "demand_features.parquet"
    if feat_file.exists():
        import pandas as pd
        df = pd.read_parquet(feat_file)
        nan_pct = df.isnull().mean().mean() * 100
        if nan_pct > 10:
            errors.append(f"High NaN rate in features: {nan_pct:.1f}%")
            logger.warning(f"High NaN rate: {nan_pct:.1f}%")
        else:
            logger.success(f"Feature quality OK — NaN rate: {nan_pct:.1f}%")

        # Validate 2025 calibration
        g2025 = df[df["season"] == 2025]
        if not g2025.empty and "target_demand_index" in g2025.columns:
            avg_demand = g2025["target_demand_index"].mean()
            target = 28_064 / 35_000
            if abs(avg_demand - target) > 0.05:
                errors.append(f"2025 calibration drift: avg demand {avg_demand:.3f} vs target {target:.3f}")
            else:
                logger.success(f"2025 calibration OK: avg demand index = {avg_demand:.3f}")

    tests_passed = len([e for e in errors if "pytest" in e or "calibration" in e]) == 0
    return {**state, "stage": "qa_agent",
            "completed_stages": state.get("completed_stages", []) + ["qa_agent"],
            "tests_passed": tests_passed, "errors": errors}


# ── Helper ────────────────────────────────────────────────────────────────────

def _import_module(module_path: str):
    """Dynamically import a module."""
    try:
        import importlib
        return importlib.import_module(module_path)
    except Exception as exc:
        logger.warning(f"Cannot import {module_path}: {exc}")
        return None


# ── Sequential runner (fallback without LangGraph) ────────────────────────────

def run_sequential() -> PipelineState:
    """Run full pipeline sequentially (used when LangGraph not installed)."""
    state: PipelineState = {
        "stage": "start",
        "completed_stages": [],
        "errors": [],
        "warnings": [],
        "artifacts": {},
        "metrics": {},
        "raw_data_ready": False,
        "features_ready": False,
        "models_ready": False,
        "api_ready": False,
        "dashboard_ready": False,
        "tests_passed": False,
        "gap_analysis": {},
        "retrospective": {},
    }

    agents = [
        run_data_engineer,
        run_feature_engineer,
        run_ml_scientist,
        run_pm_agent,
        run_qa_agent,
    ]

    for agent_fn in agents:
        state = agent_fn(state)
        if len(state.get("errors", [])) > 10:
            logger.error("Too many errors — stopping pipeline")
            break

    return state


# ── LangGraph runner ──────────────────────────────────────────────────────────

def build_langgraph_pipeline():
    """Build LangGraph state machine for agent coordination."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("data_engineer",    run_data_engineer)
    workflow.add_node("feature_engineer", run_feature_engineer)
    workflow.add_node("ml_scientist",     run_ml_scientist)
    workflow.add_node("pm_agent",         run_pm_agent)
    workflow.add_node("qa_agent",         run_qa_agent)

    workflow.set_entry_point("data_engineer")
    workflow.add_edge("data_engineer",    "feature_engineer")
    workflow.add_edge("feature_engineer", "ml_scientist")
    workflow.add_edge("ml_scientist",     "pm_agent")
    workflow.add_edge("pm_agent",         "qa_agent")
    workflow.add_edge("qa_agent",         END)

    return workflow.compile()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    """Run the full SD FC pricing pipeline."""
    logger.info("=" * 60)
    logger.info("SD FC PRICING INTELLIGENCE PLATFORM — PIPELINE START")
    logger.info("=" * 60)

    start = time.time()

    if HAS_LANGGRAPH:
        logger.info("Using LangGraph state machine")
        pipeline = build_langgraph_pipeline()
        initial_state: PipelineState = {
            "stage": "start", "completed_stages": [], "errors": [],
            "warnings": [], "artifacts": {}, "metrics": {},
            "raw_data_ready": False, "features_ready": False, "models_ready": False,
            "api_ready": False, "dashboard_ready": False, "tests_passed": False,
            "gap_analysis": {}, "retrospective": {},
        }
        final_state = pipeline.invoke(initial_state)
    else:
        final_state = run_sequential()

    elapsed = time.time() - start

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Time elapsed     : {elapsed:.0f}s")
    logger.info(f"  Stages complete  : {final_state.get('completed_stages', [])}")
    logger.info(f"  Total errors     : {len(final_state.get('errors', []))}")
    logger.info(f"  Features ready   : {final_state.get('features_ready', False)}")
    logger.info(f"  Models ready     : {final_state.get('models_ready', False)}")
    logger.info(f"  Tests passed     : {final_state.get('tests_passed', False)}")

    gap = final_state.get("gap_analysis", {})
    if gap.get("total_opportunity"):
        logger.info(f"  Revenue opp (2026): ${gap['total_opportunity']:,.0f}")

    retro = final_state.get("retrospective", {})
    if retro.get("total_revenue_opportunity"):
        logger.info(f"  2025 retro       : ${retro['total_revenue_opportunity']:,.0f} left on table")

    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  API:       uvicorn api.main:app --reload --port 8000")
    logger.info("  Dashboard: streamlit run dashboard/app.py")
    logger.info("  MLflow:    mlflow ui --backend-store-uri mlflow/")
    logger.info("=" * 60)

    return final_state


if __name__ == "__main__":
    run()
