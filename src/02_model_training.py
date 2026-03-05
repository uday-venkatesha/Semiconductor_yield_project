"""
Phase 2: Predictive Modelling (XGBoost)
────────────────────────────────────────
Loads preprocessed artifacts from Phase 1, trains an XGBoostClassifier with
class-imbalance handling, evaluates on Recall / Precision / F1, then saves
the model and predictions to disk for the downstream phases.
"""

import pandas as pd
import numpy as np
import pickle
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = 'data/artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 1. Load Phase-1 Artifacts ─────────────────────────────────────────────────
print("Loading preprocessed data...")
df_model = pd.read_parquet(f'{ARTIFACTS_DIR}/df_model.parquet')

with open(f'{ARTIFACTS_DIR}/active_sensors.pkl', 'rb') as f:
    active_sensors = pickle.load(f)

print(f"  df_model shape   : {df_model.shape}")
print(f"  Active sensors   : {len(active_sensors)}\n")

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X_features = df_model[active_sensors]
y_target   = df_model['Actual_Result']

# stratify=y_target preserves the rare-failure class ratio in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target,
    test_size=0.2, random_state=42, stratify=y_target
)

print(f"Train size : {len(X_train)} ({y_train.mean():.2%} failures)")
print(f"Test size  : {len(X_test)}  ({y_test.mean():.2%} failures)\n")

# ── 3. Handle Class Imbalance via scale_pos_weight ───────────────────────────
# XGBoost's built-in mechanism: weight = count(negatives) / count(positives)
# This is equivalent to SMOTE but avoids synthetic sample generation artifacts.
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight : {scale_pos_weight:.1f}  (penalises missed failures)\n")

# ── 4. Train XGBoost Classifier ───────────────────────────────────────────────
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
)

print("Training XGBoost model...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── 5. Predict & Evaluate ─────────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]   # P(Fail)

print("\n─── Model Evaluation (focus: Recall / Precision / F1) ───────────────────")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Fail (1)']))

# ── 6. Build Predictions DataFrame ───────────────────────────────────────────
test_indices = X_test.index
final_predictions_df = pd.DataFrame({
    'Wafer_ID':           df_model.loc[test_indices, 'Wafer_ID'].values,
    'Timestamp':          df_model.loc[test_indices, 'Timestamp'].values,
    'Actual_Result':      y_test.values,
    'Predicted_Class':    y_pred,
    'Failure_Probability': y_prob,
})

print(f"\nPredictions dataframe : {final_predictions_df.shape}")
print(f"  Predicted failures  : {final_predictions_df['Predicted_Class'].sum()}")
print(f"  Actual failures     : {final_predictions_df['Actual_Result'].sum()}\n")

# ── 7. Save Artifacts ─────────────────────────────────────────────────────────
final_predictions_df.to_parquet(f'{ARTIFACTS_DIR}/final_predictions.parquet', index=False)

with open(f'{ARTIFACTS_DIR}/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Also save the test-set features (needed by Phase 3 SHAP)
X_test_reset = X_test.reset_index(drop=True)
X_test_reset.to_parquet(f'{ARTIFACTS_DIR}/X_test.parquet', index=False)

print("✅ Phase 2 complete — artifacts saved to 'data/artifacts/'")
print("   • xgb_model.pkl")
print("   • final_predictions.parquet")
print("   • X_test.parquet")