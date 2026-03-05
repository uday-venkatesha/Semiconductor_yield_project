# Predictive Quality & Yield Optimization System
### Semiconductor Wafer Fabrication — End-to-End ML Pipeline

---

## Project Structure

```
├── main.py                   # Orchestrator — runs all 4 phases in sequence
├── requirements.txt
├── data/
│   ├── raw/                  # ← PUT YOUR SECOM FILES HERE
│   │   ├── secom.data
│   │   └── secom_labels
│   ├── artifacts/            # Auto-created: parquet + pkl files between phases
│   └── db/
│       └── manufacturing_yield.db   # Auto-created SQLite database
├── src/
│   ├── 01_preprocess.py      # Phase 1: Clean, impute, feature-select
│   ├── 02_model_training.py  # Phase 2: XGBoost train + evaluate
│   ├── 03_root_cause.py      # Phase 3: SHAP root-cause analysis
│   └── 04_database_pipeline.py  # Phase 4: Load SQLite + apply views
├── sql/
│   └── 05_tableau_views.sql  # Phase 5: Analytical views for Tableau
└── notebooks/
    └── eda_secom_dataset.ipynb
```

---

## Step-by-Step Setup & Run Guide

### Step 1 — Download the SECOM Dataset

The UCI SECOM dataset ships as **two space-delimited files with no header row**.

1. Go to: https://archive.ics.uci.edu/ml/datasets/SECOM
2. Download `secom.data` and `secom_labels` (not a CSV — no extension)
3. Place both files in `data/raw/`:

```
data/raw/secom.data      ← 1567 rows × 591 sensor features
data/raw/secom_labels    ← 1567 rows × 2 cols (Pass/Fail, Timestamp)
```

> **File format note:** Both files are space-delimited, no header, missing values encoded as `NaN` strings.  
> `secom_labels` has two columns: column 0 = `-1` (Pass) or `1` (Fail), column 1 = UCI timestamp (ignored).

---

### Step 2 — Create a Virtual Environment

```bash
# From the project root directory
python -m venv venv

# Activate (Mac / Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> `pyarrow` is required for the `.parquet` inter-phase artifact format.  
> If you are on an M1/M2 Mac and XGBoost gives errors, try: `pip install xgboost --no-binary xgboost`

---

### Step 4 — Run the Full Pipeline

```bash
python main.py
```

This runs all four phases in order.  You will see progress output for each phase.  
Total runtime is typically **2–5 minutes** depending on your hardware.

---

### Step 4 (alternative) — Run Phases Individually

If you want to run or re-run a single phase:

```bash
python src/01_preprocess.py      # Must run first
python src/02_model_training.py  # Requires phase 1 artifacts
python src/03_root_cause.py      # Requires phase 1 + 2 artifacts
python src/04_database_pipeline.py  # Requires all prior artifacts
```

---

### Step 5 — Verify the Database

```bash
# Quick check with the sqlite3 CLI
sqlite3 data/db/manufacturing_yield.db

# Inside SQLite shell:
.tables
SELECT COUNT(*) FROM production_logs;
SELECT COUNT(*) FROM ml_predictions;
SELECT COUNT(*) FROM root_cause_analysis;
SELECT * FROM v_yield_kpis LIMIT 5;
SELECT * FROM v_high_risk_sensors LIMIT 10;
.quit
```

---

### Step 6 — Connect Tableau

1. Open Tableau Desktop → **Connect → To a File → SQLite**  
   *(If SQLite isn't listed, install the SQLite ODBC driver)*
2. Browse to `data/db/manufacturing_yield.db`
3. The three views (`v_yield_kpis`, `v_high_risk_sensors`, `v_process_stability`) will appear alongside the raw tables.

**Recommended dashboard sheets:**
| Sheet | View | Chart Type |
|---|---|---|
| Daily Defect Rate | `v_yield_kpis` | Dual-axis line (Actual vs Predicted) |
| High-Risk Sensors | `v_high_risk_sensors` | Horizontal bar chart |
| Sensor Drift | `v_process_stability` | Line chart (sensor value + failure prob) |

> **Note on `v_process_stability`:** This view references `Sensor_59` by default.  
> After running the pipeline, check `v_high_risk_sensors` for your actual top sensor and update the column name in `sql/05_tableau_views.sql`, then re-run `python src/04_database_pipeline.py`.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `FileNotFoundError: data/raw/secom.data` | Download the UCI files and place them in `data/raw/` |
| `ModuleNotFoundError: pyarrow` | Run `pip install pyarrow` |
| `OperationalError: no such column: Sensor_59` | Sensor_59 was dropped in feature selection. Update the column name in `sql/05_tableau_views.sql` to a sensor from `v_high_risk_sensors` |
| SHAP slow on large test set | Normal — TreeExplainer on 300+ features takes 1–2 min |

---

## Pipeline Architecture

```
secom.data + secom_labels
        │
        ▼
01_preprocess.py  ──────→  df_model.parquet
                            active_sensors.pkl
                                    │
                                    ▼
                   02_model_training.py  ──→  xgb_model.pkl
                                              final_predictions.parquet
                                              X_test.parquet
                                                      │
                                                      ▼
                              03_root_cause.py  ──→  root_cause_df.parquet
                                                              │
                                                              ▼
                                    04_database_pipeline.py
                                              │
                                              ▼
                               manufacturing_yield.db
                               ├── production_logs
                               ├── ml_predictions
                               ├── root_cause_analysis
                               ├── v_yield_kpis
                               ├── v_high_risk_sensors
                               └── v_process_stability
```