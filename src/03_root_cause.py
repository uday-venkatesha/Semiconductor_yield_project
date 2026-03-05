"""
Phase 3: Root Cause Analysis with SHAP
────────────────────────────────────────
Loads the trained model and test-set predictions from Phase 2, computes SHAP
values for every predicted failure, and extracts the top-3 contributing sensors
per wafer into a structured DataFrame saved to disk.
"""

import pandas as pd
import numpy as np
import pickle
import os
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = 'data/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Load Phase-1 & Phase-2 Artifacts ──────────────────────────────────────
print("Loading model and test-set artifacts...")

with open(f'{ARTIFACTS_DIR}/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open(f'{ARTIFACTS_DIR}/active_sensors.pkl', 'rb') as f:
    active_sensors = pickle.load(f)

X_test               = pd.read_parquet(f'{ARTIFACTS_DIR}/X_test.parquet')
final_predictions_df = pd.read_parquet(f'{ARTIFACTS_DIR}/final_predictions.parquet')

# Align column names (X_test was saved with reset index)
X_test.columns = active_sensors

print(f"  Test set shape       : {X_test.shape}")
print(f"  Predicted failures   : {(final_predictions_df['Predicted_Class'] == 1).sum()}\n")

# ── 2. Compute SHAP Values ────────────────────────────────────────────────────
print("Computing SHAP values (TreeExplainer — fast for XGBoost)...")
explainer   = shap.TreeExplainer(xgb_model)
# shap_values → ndarray of shape (n_samples, n_features)
shap_values = explainer.shap_values(X_test)
print(f"  SHAP matrix shape    : {shap_values.shape}\n")

# ── 3. Extract Top-3 Sensors for Each Predicted Failure ──────────────────────
# Positive SHAP value  → pushed the prediction toward Fail
# Negative SHAP value  → pushed the prediction toward Pass
failure_mask    = final_predictions_df['Predicted_Class'].values == 1
failure_indices = np.where(failure_mask)[0]   # row positions in X_test / SHAP matrix

print(f"Extracting top-3 root-cause sensors for {len(failure_indices)} predicted failures...")

rca_records = []
for idx in failure_indices:
    wafer_id      = final_predictions_df.iloc[idx]['Wafer_ID']
    wafer_shap    = shap_values[idx]                                # shape: (n_features,)

    # Indices of top-3 features by |SHAP| (descending)
    top3_indices  = np.argsort(np.abs(wafer_shap))[-3:][::-1]

    for rank, feat_idx in enumerate(top3_indices, start=1):
        rca_records.append({
            'Wafer_ID':    wafer_id,
            'Sensor_Name': active_sensors[feat_idx],
            'SHAP_Value':  float(wafer_shap[feat_idx]),   # + → drives failure
            'Rank':        rank,                           # 1 = most impactful
        })

root_cause_df = pd.DataFrame(rca_records)

print(f"Generated {len(root_cause_df)} RCA records for {len(failure_indices)} wafers.")
print(f"\nTop sensors by mean |SHAP| across all failures:")
top_sensors = (
    root_cause_df.groupby('Sensor_Name')['SHAP_Value']
    .apply(lambda s: np.abs(s).mean())
    .sort_values(ascending=False)
    .head(10)
)
print(top_sensors.to_string())
print()

# ── 4. Save Artifact ──────────────────────────────────────────────────────────
root_cause_df.to_parquet(f'{ARTIFACTS_DIR}/root_cause_df.parquet', index=False)

print("✅ Phase 3 complete — artifact saved to 'data/artifacts/'")
print("   • root_cause_df.parquet")