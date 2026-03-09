"""
Phase 1: Data Preprocessing & Feature Engineering
─────────────────────────────────────────────────
Loads the SECOM dataset from a SINGLE file (secom.data) using a manual
line-by-line parser — bypasses all pandas delimiter/quote inference issues
that occur with the UCI file's inconsistent whitespace formatting.
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DATA_PATH  = 'data/raw/secom.csv'
RAW_LABEL_PATH = 'data/raw/secom_labels'
ARTIFACTS_DIR  = 'data/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Manual Line-by-Line File Parser ───────────────────────────────────────
# pandas read_csv chokes on the UCI file because:
#   • multi-char regex sep triggers quote-handling bugs in the python engine
#   • the C engine doesn't support regex separators
# Solution: read raw lines ourselves, split on any whitespace, convert to float.
print("Loading raw SECOM data (manual parser)...")

rows = []
with open(RAW_DATA_PATH, 'r') as fh:
    for line in fh:
        line = line.strip()
        if not line:           # skip blank lines
            continue
        tokens = re.split(r'\s+', line)   # split on any whitespace run
        row = []
        for t in tokens:
            if t in ('NaN', 'nan', 'NA', 'na', ''):
                row.append(np.nan)
            else:
                try:
                    row.append(float(t))
                except ValueError:
                    row.append(np.nan)   # unexpected token → treat as missing
        rows.append(row)

# Pad any short rows to the most common row length (handles ragged last lines)
max_len = max(len(r) for r in rows)
rows    = [r + [np.nan] * (max_len - len(r)) for r in rows]

raw_df = pd.DataFrame(rows)

# Drop columns that are entirely NaN (trailing-delimiter artefact)
raw_df.dropna(axis=1, how='all', inplace=True)

print(f"  Raw file shape (after dropping all-NaN cols): {raw_df.shape}")

# ── 2. Separate Labels from Sensor Features ───────────────────────────────────
y_raw = None

if os.path.exists(RAW_LABEL_PATH):
    # ── Path A: companion secom_labels file exists ────────────────────────────
    print("  Found secom_labels file — loading labels from it.")
    label_rows = []
    with open(RAW_LABEL_PATH, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = re.split(r'\s+', line)
            label_rows.append(float(tokens[0]))   # first token is Pass/Fail
    y_raw = pd.Series(label_rows).reset_index(drop=True)
    X_raw = raw_df.reset_index(drop=True)

else:
    # ── Path B: label embedded as last column of secom.data ───────────────────
    last_col        = raw_df.iloc[:, -1]
    unique_non_null = set(last_col.dropna().unique())

    if unique_non_null.issubset({-1.0, 1.0}):
        print("  No secom_labels file found.")
        print("  Auto-detected label column as LAST column of secom.data (values: -1 / 1).")
        y_raw = last_col.reset_index(drop=True)
        X_raw = raw_df.iloc[:, :-1].reset_index(drop=True)
    else:
        raise FileNotFoundError(
            "\n\n❌  Cannot locate labels.\n"
            f"    Last column sample values: {list(last_col.dropna().unique()[:8])}\n"
            "    Please download 'secom_labels' from the UCI page and save it to:\n"
            "        data/raw/secom_labels\n"
            "    URL: https://archive.ics.uci.edu/ml/datasets/SECOM"
        )

# ── 3. Standardise Labels  (-1 = Pass → 0,  1 = Fail → 1) ───────────────────
y = y_raw.apply(lambda x: 0 if x == -1 else 1).reset_index(drop=True)

print(f"  Sensor matrix shape : {X_raw.shape[0]} wafers × {X_raw.shape[1]} features")
print(f"  Class distribution  : Pass={(y == 0).sum()}  Fail={(y == 1).sum()}\n")

# ── 4. Generate Synthetic Wafer IDs & Timestamps ─────────────────────────────
num_records = len(X_raw)
start_time  = datetime.now() - timedelta(days=30)
timestamps  = [start_time + timedelta(minutes=i * 27.5) for i in range(num_records)]
wafer_ids   = [f"WFR_{str(i).zfill(5)}" for i in range(num_records)]

X_raw = X_raw.copy()
X_raw.columns = [f"Sensor_{i}" for i in range(X_raw.shape[1])]
X_raw.insert(0, 'Timestamp', timestamps)
X_raw.insert(0, 'Wafer_ID',  wafer_ids)

# ── 5. Handle Missing Values ──────────────────────────────────────────────────
print("Handling missing values...")
sensor_cols = [c for c in X_raw.columns if c.startswith('Sensor_')]

# Drop columns where > 50 % of values are missing
col_threshold = int(len(X_raw) * 0.5)
X_cleaned     = X_raw.dropna(thresh=col_threshold, axis=1).copy()
sensor_cols   = [c for c in X_cleaned.columns if c.startswith('Sensor_')]
print(f"  Sensors after dropping >50 % NaN columns : {len(sensor_cols)}")

# Impute remaining NaNs with column median
imputer = SimpleImputer(strategy='median')
X_cleaned[sensor_cols] = imputer.fit_transform(X_cleaned[sensor_cols])

# ── 6. Drop Zero-Variance (Constant) Features ────────────────────────────────
print("Removing zero-variance features...")
selector = VarianceThreshold(threshold=0.0)
selector.fit(X_cleaned[sensor_cols])
active_sensors = [sensor_cols[i] for i, keep in enumerate(selector.get_support()) if keep]
print(f"  Active sensors after variance filter      : {len(active_sensors)}\n")

# ── 7. Build Final Modelling DataFrame ───────────────────────────────────────
df_model = X_cleaned[['Wafer_ID', 'Timestamp'] + active_sensors].copy()
df_model['Actual_Result'] = y.values

print(f"Final preprocessed shape : {df_model.shape}")
print(f"Failure rate             : {df_model['Actual_Result'].mean():.2%}\n")

# ── 8. Save Artifacts to Disk ─────────────────────────────────────────────────
df_model.to_parquet(f'{ARTIFACTS_DIR}/df_model.parquet', index=False)

with open(f'{ARTIFACTS_DIR}/active_sensors.pkl', 'wb') as f:
    pickle.dump(active_sensors, f)

print(f"✅ Phase 1 complete — artifacts saved to '{ARTIFACTS_DIR}/'")
print(f"   • df_model.parquet    ({df_model.shape[0]} rows × {df_model.shape[1]} cols)")
print(f"   • active_sensors.pkl  ({len(active_sensors)} features)")