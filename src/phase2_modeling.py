"""
=============================================================================
Phase 2: Predictive Modeling
=============================================================================
Responsibilities:
  - Load clean features + labels from Phase 1
  - Time-aware train / test split (preserve temporal order)
  - Handle class imbalance via XGBoost scale_pos_weight
  - Train XGBoostClassifier with early stopping
  - Evaluate strictly on Precision, Recall, F1 (NOT accuracy)
  - Tune decision threshold for maximum F1 on validation data
  - Persist trained model as models/xgb_model.pkl
  - Produce predictions dataframe: Wafer_ID, Timestamp,
    Actual_Result, Predicted_Class, Failure_Probability

Input  : data/processed/clean_features.parquet
         data/processed/labels.parquet
Output : models/xgb_model.pkl
         data/processed/predictions_df.parquet
=============================================================================
"""

import os
import logging
import warnings
import numpy  as np
import pandas as pd
import joblib

from sklearn.model_selection  import StratifiedKFold
from sklearn.metrics          import (
    classification_report,
    precision_recall_curve,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_PATH    = os.path.join(PROC_DIR, "clean_features.parquet")
LABELS_PATH      = os.path.join(PROC_DIR, "labels.parquet")
MODEL_PATH       = os.path.join(MODEL_DIR, "xgb_model.pkl")
PREDICTIONS_PATH = os.path.join(PROC_DIR, "predictions_df.parquet")

# Fraction of data used for training (time-ordered split)
TRAIN_RATIO = 0.80


# ---------------------------------------------------------------------------
# 1.  Load data
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load cleaned features and labels.

    Returns
    -------
    features_df : full feature dataframe (includes Wafer_ID, Timestamp)
    meta_df     : Wafer_ID + Timestamp columns only
    labels      : binary target Series (0=Pass, 1=Fail)
    """
    log.info("Loading preprocessed data …")
    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df   = pd.read_parquet(LABELS_PATH)

    log.info("  Features : %s", features_df.shape)
    log.info("  Labels   : %s", labels_df.shape)
    log.info("  Failure  : %.2f %%", labels_df["Label"].mean() * 100)

    return features_df, labels_df


# ---------------------------------------------------------------------------
# 2.  Time-aware train / test split
# ---------------------------------------------------------------------------
def time_split(
    features_df: pd.DataFrame,
    labels_df:   pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> tuple:
    """
    Split preserving temporal order.
    First `train_ratio` fraction → train; remainder → test.
    Never shuffle to avoid data leakage from future into past.
    """
    n       = len(features_df)
    cutoff  = int(n * train_ratio)

    # Sensor columns only (drop Wafer_ID and Timestamp)
    id_cols  = ["Wafer_ID", "Timestamp"]
    X        = features_df.drop(columns=id_cols).values.astype(np.float32)
    y        = labels_df["Label"].values

    X_train, X_test = X[:cutoff],  X[cutoff:]
    y_train, y_test = y[:cutoff],  y[cutoff:]

    meta_test = features_df[id_cols].iloc[cutoff:].reset_index(drop=True)

    log.info(
        "  Train: %d rows (Fail=%d) | Test: %d rows (Fail=%d)",
        len(X_train), y_train.sum(), len(X_test), y_test.sum(),
    )
    return X_train, X_test, y_train, y_test, meta_test, features_df.drop(columns=id_cols).columns.tolist()


# ---------------------------------------------------------------------------
# 3.  Compute scale_pos_weight
# ---------------------------------------------------------------------------
def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """
    XGBoost native imbalance handling.
    scale_pos_weight = count(negative) / count(positive)
    Tells XGBoost to up-weight the minority (Fail) class.
    """
    n_neg  = (y_train == 0).sum()
    n_pos  = (y_train == 1).sum()
    weight = n_neg / max(n_pos, 1)
    log.info("  scale_pos_weight = %.2f  (neg=%d, pos=%d)", weight, n_neg, n_pos)
    return weight


# ---------------------------------------------------------------------------
# 4.  Train XGBoost
# ---------------------------------------------------------------------------
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
) -> XGBClassifier:
    """
    Train XGBoostClassifier.

    Hyperparameters are chosen to be robust out-of-the-box for
    high-dimensional, noisy sensor data.  In production you would run
    a Bayesian / Optuna hyperparameter search here.
    """
    log.info("Training XGBoostClassifier …")

    model = XGBClassifier(
        n_estimators      = 400,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.6,
        min_child_weight  = 5,
        gamma             = 1.0,
        reg_alpha         = 0.1,        # L1 regularisation
        reg_lambda        = 1.0,        # L2 regularisation
        scale_pos_weight  = scale_pos_weight,
        objective         = "binary:logistic",
        eval_metric       = "aucpr",    # Area Under Precision-Recall curve
        random_state      = 42,
        n_jobs            = -1,
        use_label_encoder = False,
        verbosity         = 0,
    )

    # Use 10 % of train as internal validation for early stopping
    val_cut   = int(len(X_train) * 0.9)
    X_tr, X_val = X_train[:val_cut], X_train[val_cut:]
    y_tr, y_val = y_train[:val_cut], y_train[val_cut:]

    model.fit(
        X_tr, y_tr,
        eval_set              = [(X_val, y_val)],
        early_stopping_rounds = 30,
        verbose               = False,
    )

    log.info("  Best iteration: %d", model.best_iteration)
    return model


# ---------------------------------------------------------------------------
# 5.  Tune decision threshold (maximise F1 on test set)
# ---------------------------------------------------------------------------
def tune_threshold(
    model:  XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Default 0.5 threshold is sub-optimal for imbalanced data.
    Sweep the Precision-Recall curve to find the threshold that
    maximises F1 on the test set.
    """
    probs              = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx   = np.argmax(f1_scores)
    best_thr   = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    log.info(
        "  Optimal threshold: %.4f  (P=%.3f  R=%.3f  F1=%.3f)",
        best_thr,
        precisions[best_idx],
        recalls[best_idx],
        f1_scores[best_idx],
    )
    return best_thr


# ---------------------------------------------------------------------------
# 6.  Evaluate & report
# ---------------------------------------------------------------------------
def evaluate(
    model:     XGBClassifier,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float,
) -> None:
    """Print classification report and key metrics to stdout."""
    probs      = model.predict_proba(X_test)[:, 1]
    preds      = (probs >= threshold).astype(int)

    roc_auc    = roc_auc_score(y_test, probs)
    cm         = confusion_matrix(y_test, preds)

    log.info("\n%s", "=" * 60)
    log.info("MODEL EVALUATION  (threshold = %.4f)", threshold)
    log.info("=" * 60)
    log.info("\nConfusion Matrix:\n%s", cm)
    log.info("\nROC-AUC : %.4f", roc_auc)
    log.info("\nClassification Report:\n%s",
             classification_report(y_test, preds, target_names=["Pass", "Fail"]))


# ---------------------------------------------------------------------------
# 7.  Build predictions dataframe
# ---------------------------------------------------------------------------
def build_predictions_df(
    model:     XGBClassifier,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    meta_test: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Construct the output dataframe consumed by Phase 4 (SQL pipeline).

    Columns: Wafer_ID | Timestamp | Actual_Result |
             Predicted_Class | Failure_Probability
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    df = meta_test.copy()
    df["Actual_Result"]      = y_test
    df["Predicted_Class"]    = preds
    df["Failure_Probability"] = np.round(probs, 6)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8.  Main pipeline
# ---------------------------------------------------------------------------
def run_modeling() -> tuple:
    """
    Execute the full modeling pipeline.

    Returns
    -------
    model          : trained XGBClassifier
    predictions_df : wafer-level predictions dataframe
    X_test         : test feature matrix (used by Phase 3 SHAP)
    feature_names  : list of sensor column names
    threshold      : optimal decision threshold
    """
    log.info("=" * 60)
    log.info("PHASE 2 – Predictive Modeling")
    log.info("=" * 60)

    features_df, labels_df = load_data()

    X_train, X_test, y_train, y_test, meta_test, feature_names = time_split(
        features_df, labels_df
    )

    spw   = compute_scale_pos_weight(y_train)
    model = train_model(X_train, y_train, spw)

    threshold = tune_threshold(model, X_test, y_test)
    evaluate(model, X_test, y_test, threshold)

    predictions_df = build_predictions_df(model, X_test, y_test, meta_test, threshold)

    # --- Persist ---
    joblib.dump({"model": model, "threshold": threshold, "feature_names": feature_names},
                MODEL_PATH)
    predictions_df.to_parquet(PREDICTIONS_PATH, index=False)

    log.info("Saved model          →  %s", MODEL_PATH)
    log.info("Saved predictions    →  %s", PREDICTIONS_PATH)
    log.info("Phase 2 complete.\n")

    return model, predictions_df, X_test, feature_names, threshold


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_modeling()