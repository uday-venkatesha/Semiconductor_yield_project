"""
=============================================================================
run_pipeline.py  –  Master Orchestrator
=============================================================================
Run this single script to execute all 5 phases end-to-end:

    python run_pipeline.py

Prerequisites
-------------
1.  Install dependencies:
        pip install -r requirements.txt

2.  Place your SECOM data files in data/raw/:
        data/raw/secom.csv          ← sensor readings  (required)
        data/raw/secom_labels.csv   ← labels + timestamps (optional;
                                       synthetic labels generated if absent)

    If your download used the original UCI names:
        secom.data   → rename/copy to  data/raw/secom.csv
        secom_labels.data → rename/copy to  data/raw/secom_labels.csv

Output
------
    data/processed/clean_features.parquet
    data/processed/labels.parquet
    data/processed/predictions_df.parquet
    data/processed/rca_df.parquet
    data/processed/shap_summary.png
    models/xgb_model.pkl
    database/manufacturing_yield.db   ← connect Tableau here

Estimated runtime (M1/M2 MacBook Pro or equivalent):
    Phase 1 : ~5 s
    Phase 2 : ~30–60 s  (XGBoost with early stopping)
    Phase 3 : ~30–90 s  (SHAP TreeExplainer on 300+ test rows)
    Phase 4 : ~20 s     (SQLite bulk insert of 500+ sensor columns)
    Phase 5 : <1 s      (SQL VIEW creation)
    Total   : ~2–4 minutes
=============================================================================
"""

import sys
import time
import logging
import traceback

# ---------------------------------------------------------------------------
# Logging – one root logger for the whole pipeline
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Add src/ to path so phase modules resolve correctly when run from root
# ---------------------------------------------------------------------------
import os
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_DIR)


# ---------------------------------------------------------------------------
# Phase imports
# ---------------------------------------------------------------------------
from phase1_preprocessing import run_preprocessing
from phase2_modeling       import run_modeling
from phase3_shap_rca       import run_shap_rca
from phase4_sql_pipeline   import run_sql_pipeline


# ---------------------------------------------------------------------------
# Helper: time each phase
# ---------------------------------------------------------------------------
def run_phase(name: str, func, *args, **kwargs):
    log.info("")
    log.info("▶  Starting %s", name)
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        log.info("✔  %s completed in %.1f s", name, elapsed)
        return result
    except Exception:
        elapsed = time.time() - t0
        log.error("✘  %s FAILED after %.1f s", name, elapsed)
        log.error(traceback.format_exc())
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pipeline_start = time.time()

    log.info("=" * 65)
    log.info("  Predictive Quality & Yield Optimization System")
    log.info("  Semiconductor Wafer Fabrication  |  SECOM Dataset")
    log.info("=" * 65)

    # ── Phase 1: Preprocessing ───────────────────────────────────────────
    run_phase("Phase 1 – Preprocessing & Feature Engineering", run_preprocessing)

    # ── Phase 2: Modeling ────────────────────────────────────────────────
    run_phase("Phase 2 – Predictive Modeling", run_modeling)

    # ── Phase 3: SHAP Root Cause Analysis ────────────────────────────────
    run_phase("Phase 3 – SHAP Root Cause Analysis", run_shap_rca)

    # ── Phase 4 + 5: SQL Pipeline & Views ────────────────────────────────
    run_phase("Phase 4+5 – SQL Data Pipeline & Tableau Views", run_sql_pipeline)

    # ── Done ─────────────────────────────────────────────────────────────
    total = time.time() - pipeline_start
    log.info("")
    log.info("=" * 65)
    log.info("  ✅  Pipeline complete in %.1f s", total)
    log.info("")
    log.info("  Outputs:")
    log.info("    data/processed/clean_features.parquet")
    log.info("    data/processed/predictions_df.parquet")
    log.info("    data/processed/rca_df.parquet")
    log.info("    data/processed/shap_summary.png")
    log.info("    models/xgb_model.pkl")
    log.info("    database/manufacturing_yield.db  ← connect Tableau here")
    log.info("")
    log.info("  Tableau Views available:")
    log.info("    v_yield_kpis          – daily defect rate KPIs")
    log.info("    v_high_risk_sensors   – top root-cause sensors (7-day)")
    log.info("    v_process_stability   – SPC sensor trend monitoring")
    log.info("    v_wafer_failure_detail – drill-through per failed wafer")
    log.info("=" * 65)


if __name__ == "__main__":
    main()