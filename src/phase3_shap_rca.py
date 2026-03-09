"""
=============================================================================
Phase 3: Root Cause Analysis with SHAP
=============================================================================
Responsibilities:
  - Load the trained XGBoost model + test feature matrix
  - Compute SHAP values using TreeExplainer (fast, exact for tree models)
  - For every wafer PREDICTED as Fail, extract the top-3 sensors
    (by absolute SHAP value) — these are the root causes
  - Generate summary plots (saved to data/processed/)
  - Persist RCA dataframe: Wafer_ID | Sensor_Name | SHAP_Value | Rank

Input  : models/xgb_model.pkl
         data/processed/predictions_df.parquet
         data/processed/clean_features.parquet
Output : data/processed/rca_df.parquet
         data/processed/shap_summary.png  (global feature importance)
=============================================================================
"""

import os
import logging
import warnings
import numpy  as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

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
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR         = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR        = os.path.join(BASE_DIR, "models")

MODEL_PATH       = os.path.join(MODEL_DIR, "xgb_model.pkl")
FEATURES_PATH    = os.path.join(PROC_DIR,  "clean_features.parquet")
PREDICTIONS_PATH = os.path.join(PROC_DIR,  "predictions_df.parquet")
RCA_PATH         = os.path.join(PROC_DIR,  "rca_df.parquet")
SHAP_PLOT_PATH   = os.path.join(PROC_DIR,  "shap_summary.png")

TOP_N_SENSORS = 3     # top sensors per failing wafer


# ---------------------------------------------------------------------------
# 1.  Load artefacts
# ---------------------------------------------------------------------------
def load_artefacts() -> tuple:
    """
    Load the trained model bundle, test feature matrix, and predictions.

    Returns
    -------
    model          : XGBClassifier
    feature_names  : list[str]
    threshold      : float
    X_test         : np.ndarray  (test rows only, sensor values)
    predictions_df : pd.DataFrame
    """
    log.info("Loading model and data artefacts …")

    bundle         = joblib.load(MODEL_PATH)
    model          = bundle["model"]
    feature_names  = bundle["feature_names"]
    threshold      = bundle["threshold"]

    # Reconstruct X_test from the full features file
    features_df    = pd.read_parquet(FEATURES_PATH)
    predictions_df = pd.read_parquet(PREDICTIONS_PATH)

    id_cols = ["Wafer_ID", "Timestamp"]
    all_X   = features_df.drop(columns=id_cols).values.astype(np.float32)

    # Test rows are those whose Wafer_IDs appear in predictions_df
    test_ids  = set(predictions_df["Wafer_ID"].tolist())
    test_mask = features_df["Wafer_ID"].isin(test_ids).values
    X_test    = all_X[test_mask]

    log.info(
        "  Model loaded.  Test matrix: %d rows × %d features",
        X_test.shape[0], X_test.shape[1],
    )
    return model, feature_names, threshold, X_test, predictions_df


# ---------------------------------------------------------------------------
# 2.  Compute SHAP values
# ---------------------------------------------------------------------------
def compute_shap_values(
    model,
    X_test:       np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """
    Use TreeExplainer for XGBoost — deterministic and orders of magnitude
    faster than KernelExplainer on high-dimensional data.

    Returns shap_values: np.ndarray of shape (n_test, n_features)
    Positive values push toward Fail, negative toward Pass.
    """
    log.info("Computing SHAP values with TreeExplainer …")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # XGBoost binary classification: shap_values is 2-D (n_samples, n_features)
    if isinstance(shap_values, list):
        # Older shap versions return [neg_class, pos_class]
        shap_values = shap_values[1]

    log.info("  SHAP values shape: %s", shap_values.shape)
    return shap_values


# ---------------------------------------------------------------------------
# 3.  Global summary plot
# ---------------------------------------------------------------------------
def save_shap_summary_plot(
    shap_values:  np.ndarray,
    X_test:       np.ndarray,
    feature_names: list,
) -> None:
    """
    Save a SHAP beeswarm summary plot showing the 20 most influential
    sensors globally across all test wafers.
    """
    log.info("Generating SHAP summary plot …")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot",
    )
    plt.title("Top 20 Sensors by Global SHAP Importance", fontsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved SHAP summary plot →  %s", SHAP_PLOT_PATH)


# ---------------------------------------------------------------------------
# 4.  Extract top-N sensors per predicted failure
# ---------------------------------------------------------------------------
def extract_top_sensors_per_failure(
    shap_values:   np.ndarray,
    feature_names: list,
    predictions_df: pd.DataFrame,
    top_n: int = TOP_N_SENSORS,
) -> pd.DataFrame:
    """
    For every wafer predicted as Fail (Predicted_Class == 1), extract the
    top `top_n` sensors ordered by |SHAP value| descending.

    Parameters
    ----------
    shap_values    : (n_test, n_features)  SHAP matrix
    feature_names  : list of sensor names aligned with shap_values columns
    predictions_df : predictions dataframe (Wafer_ID, Predicted_Class, …)
    top_n          : number of sensors to extract per wafer

    Returns
    -------
    rca_df : DataFrame with columns
             Wafer_ID | Timestamp | Sensor_Name | SHAP_Value | Rank
    """
    log.info("Extracting top-%d root-cause sensors for predicted failures …", top_n)

    # Index of predicted failures within the test set
    fail_mask    = predictions_df["Predicted_Class"].values == 1
    fail_indices = np.where(fail_mask)[0]

    n_failures = fail_mask.sum()
    log.info("  Predicted failures in test set: %d", n_failures)

    records = []
    for idx in fail_indices:
        wafer_id  = predictions_df.iloc[idx]["Wafer_ID"]
        timestamp = predictions_df.iloc[idx]["Timestamp"]
        sv        = shap_values[idx]                          # shape (n_features,)

        # Sort by absolute SHAP value, descending
        sorted_indices = np.argsort(np.abs(sv))[::-1][:top_n]

        for rank, feat_idx in enumerate(sorted_indices, start=1):
            records.append({
                "Wafer_ID":    wafer_id,
                "Timestamp":   timestamp,
                "Sensor_Name": feature_names[feat_idx],
                "SHAP_Value":  round(float(sv[feat_idx]), 6),
                "Rank":        rank,
            })

    rca_df = pd.DataFrame(records)
    log.info("  RCA dataframe: %d rows", len(rca_df))
    return rca_df


# ---------------------------------------------------------------------------
# 5.  (Optional) per-wafer waterfall helper
# ---------------------------------------------------------------------------
def explain_single_wafer(
    shap_values:   np.ndarray,
    X_test:        np.ndarray,
    feature_names: list,
    local_idx:     int,
    wafer_id:      str = "",
    save_path:     str = "",
) -> None:
    """
    Generate a waterfall (force) plot for a single wafer.
    Useful for engineering team deep-dives.
    Called externally / interactively; not part of the main pipeline.
    """
    exp = shap.Explanation(
        values     = shap_values[local_idx],
        base_values= 0,                       # log-odds base
        data       = X_test[local_idx],
        feature_names=feature_names,
    )
    plt.figure(figsize=(12, 5))
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.title(f"Wafer {wafer_id} – SHAP Waterfall", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 6.  Main pipeline
# ---------------------------------------------------------------------------
def run_shap_rca() -> pd.DataFrame:
    """
    Execute the full SHAP root-cause-analysis pipeline.

    Returns
    -------
    rca_df : DataFrame (Wafer_ID, Timestamp, Sensor_Name, SHAP_Value, Rank)
    """
    log.info("=" * 60)
    log.info("PHASE 3 – Root Cause Analysis with SHAP")
    log.info("=" * 60)

    model, feature_names, threshold, X_test, predictions_df = load_artefacts()

    shap_values = compute_shap_values(model, X_test, feature_names)

    # Global plot
    save_shap_summary_plot(shap_values, X_test, feature_names)

    # Per-failure RCA
    rca_df = extract_top_sensors_per_failure(
        shap_values, feature_names, predictions_df
    )

    # Persist
    rca_df.to_parquet(RCA_PATH, index=False)
    log.info("Saved RCA dataframe  →  %s", RCA_PATH)

    # Quick preview
    log.info(
        "\nTop recurring root-cause sensors across all predicted failures:\n%s",
        rca_df.groupby("Sensor_Name")["SHAP_Value"]
              .agg(["count", "mean"])
              .sort_values("count", ascending=False)
              .head(10)
              .to_string(),
    )

    log.info("Phase 3 complete.\n")
    return rca_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_shap_rca()