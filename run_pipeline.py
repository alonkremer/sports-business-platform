"""
SD FC Pricing Intelligence — Full Pipeline Runner.

Usage:
  python run_pipeline.py           # Full pipeline (data → features → models → gap)
  python run_pipeline.py --data    # Data ingestion only
  python run_pipeline.py --features # Feature engineering only
  python run_pipeline.py --models  # ML models only
  python run_pipeline.py --test    # Run test suite only
  python run_pipeline.py --api     # Start FastAPI server
  python run_pipeline.py --dash    # Start Streamlit dashboard

Full sequence:
  1. python run_pipeline.py           (build everything)
  2. python run_pipeline.py --api     (in one terminal)
  3. python run_pipeline.py --dash    (in another terminal)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent


def run_data():
    logger.info("Running data ingestion pipeline...")
    from src.data_ingestion.fbref_attendance import run as fbref_run
    from src.data_ingestion.sdfc_schedule import run as sched_run
    from src.data_ingestion.mls_attendance import run as mls_run
    from src.data_ingestion.oddsportal_odds import run as odds_run
    from src.data_ingestion.ticketmaster_prices import run as tm_run
    from src.data_ingestion.seatgeek_market import run as sg_run
    from src.data_ingestion.stubhub_market import run as sh_run
    from src.data_ingestion.vividseats_market import run as vs_run
    from src.data_generation.synthetic_sdfc import run as synth_run
    from src.database.schema import initialize

    fbref_run()
    sched_run()
    mls_run()
    odds_run()
    tm_run()
    sg_run()
    sh_run()
    vs_run()
    synth_run()
    initialize()
    logger.success("Data pipeline complete")


def run_features():
    logger.info("Running feature engineering pipeline...")
    from src.feature_engineering.pipeline import run
    df = run()
    logger.success(f"Features complete: {df.shape}")


def run_models():
    logger.info("Running ML model pipeline...")
    import pandas as pd
    feat_file = ROOT / "data" / "features" / "demand_features.parquet"

    if not feat_file.exists():
        logger.error("Feature file not found — run features first: python run_pipeline.py --features")
        sys.exit(1)

    df = pd.read_parquet(feat_file)

    from src.pricing_model.demand_model import train as train_demand
    model, explainer, metrics = train_demand(df)
    logger.success(f"Demand model: MAPE={metrics['cv_mape']:.1f}%")

    from src.elasticity_model.causal_elasticity import run as run_elasticity
    elast_df = run_elasticity(df)

    from src.price_optimizer.emsr_optimizer import run_optimizer
    opt_df = run_optimizer(df, elast_df)

    from src.price_gap.gap_calculator import run as run_gap
    gap_df, retro = run_gap(df, opt_df)
    logger.success(f"Gap analysis: ${gap_df['revenue_opp_total'].sum():,.0f} total opportunity")
    logger.info(f"2025 Retrospective: ${retro.get('total_revenue_opportunity', 0):,.0f}")


def run_tests():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v"],
        cwd=str(ROOT)
    )
    sys.exit(result.returncode)


def start_api():
    logger.info("Starting FastAPI server at http://localhost:8000")
    logger.info("API docs: http://localhost:8000/docs")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app", "--reload", "--port", "8000", "--host", "0.0.0.0"
    ], cwd=str(ROOT))


def start_dashboard():
    logger.info("Starting Streamlit dashboard...")
    logger.info("Requires API running: python run_pipeline.py --api")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py", "--server.port", "8501"
    ], cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="SD FC Pricing Intelligence Pipeline")
    parser.add_argument("--data",     action="store_true", help="Run data ingestion only")
    parser.add_argument("--features", action="store_true", help="Run feature engineering only")
    parser.add_argument("--models",   action="store_true", help="Run ML models only")
    parser.add_argument("--test",     action="store_true", help="Run test suite")
    parser.add_argument("--api",      action="store_true", help="Start FastAPI server")
    parser.add_argument("--dash",     action="store_true", help="Start Streamlit dashboard")
    args = parser.parse_args()

    if args.data:
        run_data()
    elif args.features:
        run_features()
    elif args.models:
        run_models()
    elif args.test:
        run_tests()
    elif args.api:
        start_api()
    elif args.dash:
        start_dashboard()
    else:
        # Full pipeline
        logger.info("Running full pipeline...")
        run_data()
        run_features()
        run_models()
        logger.success("Full pipeline complete!")
        logger.info("\nTo launch the platform:")
        logger.info("  Terminal 1: python run_pipeline.py --api")
        logger.info("  Terminal 2: python run_pipeline.py --dash")


if __name__ == "__main__":
    main()
