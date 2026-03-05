"""
Phase 1: Data Preprocessing & Feature Engineering
─────────────────────────────────────────────────
Loads the SECOM dataset (two separate files), generates synthetic timestamps,
cleans missing values, drops zero-variance features, and saves artifacts to disk
so the next scripts in the pipeline can load them without re-running this phase.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# ── Paths ──────────────────────────────────────────────────────────────────────
# The SECOM dataset from UCI comes as TWO space-delimited files (no CSV headers):
#   secom.data   → 1567 rows × 591 sensor features
#   secom_labels → 1567 rows × 2 cols: [Pass/Fail(-1/1), Timestamp]
RAW_DATA_PATH  = 'data/raw/secom.data'
RAW_LABEL_PATH = 'data/raw/secom_labels'
ARTIFACTS_DIR  = 'data/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("Loading raw SECOM data...")
X_raw     = pd.read_csv(RAW_DATA_PATH,  sep=' ', header=None, na_values='NaN')
labels_df = pd.read_csv(RAW_LABEL_PATH, sep=' ', header=None, names=['Pass_Fail', 'UCI_Timestamp'])

# Standardise labels: SECOM convention -1 = Pass → 0, 1 = Fail → 1
y = labels_df['Pass_Fail'].apply(lambda x: 0 if x == -1 else 1).reset_index(drop=True)
X_raw = X_raw.reset_index(drop=True)

print(f"  Raw shape   : {X_raw.shape[0]} wafers × {X_raw.shape[1]} sensor features")
print(f"  Class split : {y.value_counts().to_dict()}\n")

# ── 2. Generate Synthetic Identifiers & Timestamps ────────────────────────────
# Spread 1 567 wafers evenly across the last 30 days (~one wafer every 27.5 min)
num_records = len(X_raw)
start_time  = datetime.now() - timedelta(days=30)
timestamps  = [start_time + timedelta(minutes=i * 27.5) for i in range(num_records)]
wafer_ids   = [f"WFR_{str(i).zfill(5)}" for i in range(num_records)]

# Name all sensor columns
X_raw.columns = [f"Sensor_{i}" for i in range(X_raw.shape[1])]

# Prepend metadata columns
X_raw.insert(0, 'Timestamp', timestamps)
X_raw.insert(0, 'Wafer_ID',  wafer_ids)

# ── 3. Handle Missing Values ───────────────────────────────────────────────────
print("Handling missing values...")
sensor_cols = [c for c in X_raw.columns if c.startswith('Sensor_')]

# Step 3a: Drop any feature where >50 % of values are missing
col_threshold = int(len(X_raw) * 0.5)
X_cleaned     = X_raw.dropna(thresh=col_threshold, axis=1).copy()
sensor_cols   = [c for c in X_cleaned.columns if c.startswith('Sensor_')]
print(f"  Sensors after dropping >50 % NaN columns : {len(sensor_cols)}")

# Step 3b: Impute remaining NaNs with column median (robust to outliers)
imputer = SimpleImputer(strategy='median')
X_cleaned[sensor_cols] = imputer.fit_transform(X_cleaned[sensor_cols])

# ── 4. Drop Zero-Variance (Constant) Features ─────────────────────────────────
print("Removing zero-variance features...")
selector = VarianceThreshold(threshold=0.0)
selector.fit(X_cleaned[sensor_cols])
active_sensors = [sensor_cols[i] for i, keep in enumerate(selector.get_support()) if keep]
print(f"  Active sensors after variance filter      : {len(active_sensors)}\n")

# ── 5. Build Final Modelling DataFrame ────────────────────────────────────────
df_model = X_cleaned[['Wafer_ID', 'Timestamp'] + active_sensors].copy()
df_model['Actual_Result'] = y.values

print(f"Final preprocessed shape : {df_model.shape}")
print(f"Failure rate             : {df_model['Actual_Result'].mean():.2%}\n")

# ── 6. Save Artifacts to Disk ─────────────────────────────────────────────────
# Parquet preserves dtypes (datetime, float32) and is fast to read/write
df_model.to_parquet(f'{ARTIFACTS_DIR}/df_model.parquet', index=False)

with open(f'{ARTIFACTS_DIR}/active_sensors.pkl', 'wb') as f:
    pickle.dump(active_sensors, f)

print(f"✅ Phase 1 complete — artifacts saved to '{ARTIFACTS_DIR}/'")
print(f"   • df_model.parquet       ({df_model.shape[0]} rows × {df_model.shape[1]} cols)")
print(f"   • active_sensors.pkl     ({len(active_sensors)} features)")