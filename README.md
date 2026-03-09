# Predictive Quality & Yield Optimization System
### Semiconductor Wafer Fabrication — SECOM Dataset

A production-grade, end-to-end ML pipeline that predicts wafer failures from
early-stage sensor data and delivers SHAP-powered root cause analysis into a
Tableau-ready SQLite database.

---

## Architecture

```
secom.csv (raw UCI data)
     │
     ▼
Phase 1 ─── Preprocessing & Feature Engineering
     │       Drop >50% missing cols, median impute, drop zero-variance
     ▼
Phase 2 ─── XGBoost Predictive Model
     │       Imbalance via scale_pos_weight, threshold-tuned for F1
     ▼
Phase 3 ─── SHAP Root Cause Analysis
     │       TreeExplainer, top-3 sensors per predicted failure
     ▼
Phase 4+5 ── SQLite Data Pipeline + Tableau Views
             3 normalised tables + 4 analytical SQL views
```

---

## Quick Start

### 1. Install dependencies

```bash
cd semiconductor_yield
pip install -r requirements.txt
```

### 2. Place your data

```
data/raw/secom.csv           ← rename your UCI secom.data file to this
data/raw/secom_labels.csv    ← rename secom_labels.data (optional)
```

**If you only have `secom.csv`** (sensor data only, no labels file):
The pipeline will automatically generate synthetic binary labels with a
realistic ~6.5% failure rate and 30-day mock timestamps.

### 3. Run the pipeline

```bash
python run_pipeline.py
```

That's it. All 5 phases execute sequentially.

---

## Outputs

| File | Description |
|---|---|
| `data/processed/clean_features.parquet` | Cleaned sensor matrix |
| `data/processed/labels.parquet` | Binary labels + timestamps |
| `data/processed/predictions_df.parquet` | Per-wafer model predictions |
| `data/processed/rca_df.parquet` | SHAP root cause attribution |
| `data/processed/shap_summary.png` | Global feature importance plot |
| `models/xgb_model.pkl` | Serialised XGBoost model + metadata |
| `database/manufacturing_yield.db` | **SQLite DB — connect Tableau here** |

---

## Tableau Connection

1. Install [SQLite ODBC Driver](http://www.ch-werner.de/sqliteodbc/)
2. Open Tableau Desktop → **Connect → Other Databases (ODBC)**
3. Select **SQLite3 ODBC Driver**, set path to `database/manufacturing_yield.db`
4. You will see 3 tables + 4 views available as data sources

### Recommended Dashboard Layout

| Sheet | View | Chart Type |
|---|---|---|
| Daily Yield KPI | `v_yield_kpis` | Dual-axis line: Actual vs Predicted defect rate |
| Root Cause Radar | `v_high_risk_sensors` | Horizontal bar chart by Total_SHAP_Impact |
| SPC Monitor | `v_process_stability` | Scatter + rolling avg failure probability |
| Failure Drill-Through | `v_wafer_failure_detail` | Table with SHAP waterfall detail |

---

## Key Technical Decisions

| Problem | Solution | Reason |
|---|---|---|
| Heavy class imbalance (~6.5% failures) | `scale_pos_weight` in XGBoost | Native, leak-free, no synthetic samples needed |
| Temporal data | Time-ordered train/test split (80/20) | Prevents data leakage from future into past |
| 591 noisy sensor features | Drop >50% missing → median impute → drop zero-variance | Reduces dimensionality without overfitting imputers |
| Model evaluation | Precision / Recall / F1 only | Accuracy is misleading on imbalanced data |
| Default 0.5 threshold suboptimal | Precision-Recall curve sweep → best F1 threshold | Balances false alarms vs missed failures |
| SHAP computation speed | `TreeExplainer` (not KernelExplainer) | 50–100× faster for XGBoost |

---

## Project Structure

```
semiconductor_yield/
├── data/
│   ├── raw/
│   │   └── secom.csv              ← place your UCI file here
│   └── processed/                 ← auto-generated outputs
├── models/
│   └── xgb_model.pkl
├── database/
│   └── manufacturing_yield.db
├── src/
│   ├── phase1_preprocessing.py
│   ├── phase2_modeling.py
│   ├── phase3_shap_rca.py
│   ├── phase4_sql_pipeline.py
│   └── phase5_sql_views.sql       ← standalone SQL reference
├── run_pipeline.py                ← entry point
├── requirements.txt
└── README.md
```

---

## Runtime Estimates

| Phase | M1 MacBook | AWS t3.medium |
|---|---|---|
| Phase 1 – Preprocessing | ~5 s | ~8 s |
| Phase 2 – XGBoost training | ~30–60 s | ~45–90 s |
| Phase 3 – SHAP values | ~30–90 s | ~60–120 s |
| Phase 4 – SQL insert | ~20 s | ~30 s |
| **Total** | **~2–4 min** | **~3–5 min** |