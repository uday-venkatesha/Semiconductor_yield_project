"""
=============================================================================
Phase 1: Data Preprocessing & Feature Engineering
=============================================================================
Responsibilities:
  - Load the SECOM dataset (UCI format: space-delimited, 'NaN' strings)
  - Parse or generate timestamps for time-series dashboarding
  - Parse binary Pass/Fail labels (-1 / +1  →  0 / 1)
  - Drop features with > 50% missing values
  - Apply median imputation for remaining missing values
  - Drop zero-variance (constant) features
  - Persist cleaned artefacts to data/processed/ as Parquet files

Input  : data/raw/secom.data          (1567 rows × 591 sensor columns)
         data/raw/secom_labels.data   (1567 rows: label + timestamp)
Output : data/processed/clean_features.parquet
         data/processed/labels.parquet

UCI File Format Notes
---------------------
  secom.data        : 591 space-separated float columns, 'NaN' for missing
  secom_labels.data : 2 columns — integer label (-1/+1) and datetime string
                      e.g.  -1  09:28:36  20-01-2008
  secom.names       : human-readable dataset description (not loaded)
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

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
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# File resolution: accept the original UCI .data filenames directly.
# Also support .csv in case the user renamed them — first match wins.
# ---------------------------------------------------------------------------
def _resolve_path(*candidates: str) -> str:
    """Return the first candidate path that exists on disk."""
    for p in candidates:
        if os.path.exists(p):
            return p
    # Return the first candidate as the "expected" path for the error message
    return candidates[0]

SECOM_DATA_PATH = _resolve_path(
    os.path.join(RAW_DIR, "secom.data"),        # original UCI name  ← preferred
    os.path.join(RAW_DIR, "secom.csv"),          # renamed by user
)
SECOM_LABELS_PATH = _resolve_path(
    os.path.join(RAW_DIR, "secom_labels.data"),  # original UCI name  ← preferred
    os.path.join(RAW_DIR, "secom_labels.csv"),   # renamed by user
)

# Thresholds
MISSING_DROP_THRESHOLD = 0.50   # Drop columns with >50 % missing values


# ---------------------------------------------------------------------------
# 1.  Load sensor data
# ---------------------------------------------------------------------------
def load_sensor_data(path: str) -> pd.DataFrame:
    """
    Load SECOM sensor data from secom.data (or secom.csv).

    secom.data format: 591 columns, single-space separated, no header,
    missing values encoded as the literal string 'NaN'.

    We always try the whitespace delimiter first because the .data file
    never uses commas.  The CSV fallback handles a user-renamed file that
    was genuinely saved as comma-separated.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Sensor data file not found: {path}\n"
            "Please place secom.data (or secom.csv) inside data/raw/"
        )

    log.info("Loading sensor data from: %s", path)

    # Primary: whitespace-delimited, no header (UCI native format)
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan", "NA", ""],
        engine="python",
    )

    # Fallback: if somehow only 1 column was parsed, retry as CSV
    if df.shape[1] < 100:
        log.warning("  Whitespace parse yielded too few columns — retrying as CSV.")
        df = pd.read_csv(path, na_values=["NaN", "nan", "NA", ""])

    log.info("  Loaded: %d rows × %d columns", *df.shape)

    # Always rename columns to Sensor_1 … Sensor_N
    df.columns = [f"Sensor_{i+1}" for i in range(df.shape[1])]
    return df


# ---------------------------------------------------------------------------
# 2.  Load or generate labels + timestamps
# ---------------------------------------------------------------------------
def load_labels_and_timestamps(n_rows: int) -> pd.DataFrame:
    """
    Load secom_labels.data and parse its two columns.

    UCI secom_labels.data format (space-separated, no header):
        Column 1 : integer label   — -1 = Pass,  1 = Fail
        Column 2 : datetime string — "HH:MM:SS  DD-MM-YYYY"
                   (time and date are separated by two spaces in the raw file,
                    so the full timestamp spans columns 2–3 when split naively;
                    we read cols 0/1 then reconstruct the datetime string)

    Example row:
        -1  09:28:36  20-01-2008

    Falls back to synthetic labels + timestamps when the file is absent.

    UCI label convention: -1 = Pass (+0), +1 = Fail (+1).
    """
    if os.path.exists(SECOM_LABELS_PATH):
        log.info("Loading labels from: %s", SECOM_LABELS_PATH)

        # Read raw text lines so we can handle the variable whitespace
        with open(SECOM_LABELS_PATH, "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        labels_raw  = []
        timestamps  = []

        for line in lines:
            parts = line.split()
            # parts[0] = label,  parts[1] = time,  parts[2] = date
            labels_raw.append(int(parts[0]))
            if len(parts) >= 3:
                # Combine "HH:MM:SS DD-MM-YYYY" → pandas datetime
                ts_str = f"{parts[1]} {parts[2]}"
                timestamps.append(
                    pd.to_datetime(ts_str, format="%H:%M:%S %d-%m-%Y", errors="coerce")
                )
            else:
                timestamps.append(pd.NaT)

        ldf = pd.DataFrame({"Label_Raw": labels_raw, "Timestamp": timestamps})

        # If all timestamps failed to parse, fall back to synthetic ones
        if ldf["Timestamp"].isna().all():
            log.warning("  Timestamp parsing failed — using synthetic timestamps.")
            ldf["Timestamp"] = _generate_timestamps(n_rows)
        else:
            # Fill any individual NaT values by interpolation
            ldf["Timestamp"] = ldf["Timestamp"].ffill().bfill()

        log.info(
            "  Parsed %d label rows from file.  "
            "Date range: %s → %s",
            len(ldf),
            ldf["Timestamp"].min(),
            ldf["Timestamp"].max(),
        )

    else:
        log.warning(
            "secom_labels.data not found at %s\n"
            "  Generating synthetic labels & timestamps.",
            SECOM_LABELS_PATH,
        )
        np.random.seed(42)
        # ~6.5 % failure rate mirrors the real SECOM dataset
        raw_labels = np.random.choice([-1, 1], size=n_rows, p=[0.935, 0.065])
        ldf = pd.DataFrame(
            {"Label_Raw": raw_labels, "Timestamp": _generate_timestamps(n_rows)}
        )

    # Remap: -1 (Pass) → 0,  +1 (Fail) → 1
    ldf["Label"] = ldf["Label_Raw"].map({-1: 0, 1: 1}).fillna(0).astype(int)
    ldf = ldf[["Label", "Timestamp"]].reset_index(drop=True)

    fail_pct = ldf["Label"].mean() * 100
    log.info("  Failure rate in dataset: %.2f %%", fail_pct)
    return ldf


def _generate_timestamps(n: int) -> pd.Series:
    """
    Create n evenly-spaced timestamps spanning the last 30 days.
    Simulates continuous wafer production (one wafer every ~27 minutes
    for 1 567 wafers over 30 days).
    """
    end   = pd.Timestamp.now().floor("min")
    start = end - pd.Timedelta(days=30)
    return pd.Series(pd.date_range(start=start, end=end, periods=n))


# ---------------------------------------------------------------------------
# 3.  Drop high-missing-value features
# ---------------------------------------------------------------------------
def drop_high_missing(df: pd.DataFrame, threshold: float = MISSING_DROP_THRESHOLD) -> pd.DataFrame:
    """Drop columns where the fraction of NaN exceeds `threshold`."""
    missing_frac = df.isna().mean()
    cols_to_drop  = missing_frac[missing_frac > threshold].index.tolist()
    log.info(
        "  Dropping %d / %d features with >%.0f%% missing values.",
        len(cols_to_drop), df.shape[1], threshold * 100,
    )
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# 4.  Median imputation for remaining NaN
# ---------------------------------------------------------------------------
def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NaN with column-wise median (robust to outliers)."""
    imputer = SimpleImputer(strategy="median")
    arr     = imputer.fit_transform(df)
    result  = pd.DataFrame(arr, columns=df.columns, index=df.index)
    remaining = result.isna().sum().sum()
    log.info("  Median imputation complete.  Remaining NaN: %d", remaining)
    return result


# ---------------------------------------------------------------------------
# 5.  Drop zero-variance (constant) features
# ---------------------------------------------------------------------------
def drop_zero_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Remove features that carry no information (std == 0)."""
    std          = df.std(numeric_only=True)
    const_cols   = std[std == 0].index.tolist()
    log.info("  Dropping %d zero-variance features.", len(const_cols))
    return df.drop(columns=const_cols)


# ---------------------------------------------------------------------------
# 6.  Main pipeline
# ---------------------------------------------------------------------------
def run_preprocessing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full preprocessing pipeline.

    Returns
    -------
    clean_df : pd.DataFrame
        Cleaned sensor features with Wafer_ID and Timestamp prepended.
    labels_df : pd.DataFrame
        DataFrame with columns [Wafer_ID, Timestamp, Label].
    """
    log.info("=" * 60)
    log.info("PHASE 1 – Preprocessing & Feature Engineering")
    log.info("=" * 60)

    # --- Load ---
    raw_sensors = load_sensor_data(SECOM_DATA_PATH)
    n_rows      = len(raw_sensors)
    labels_df   = load_labels_and_timestamps(n_rows)

    log.info("Raw sensor shape: %s", raw_sensors.shape)

    # --- Clean ---
    df = drop_high_missing(raw_sensors)
    df = impute_median(df)
    df = drop_zero_variance(df)

    log.info("Clean sensor shape after preprocessing: %s", df.shape)

    # --- Attach identifiers ---
    wafer_ids = [f"WAFER_{i+1:04d}" for i in range(n_rows)]
    df.insert(0, "Wafer_ID",  wafer_ids)
    df.insert(1, "Timestamp", labels_df["Timestamp"].values)

    labels_df.insert(0, "Wafer_ID", wafer_ids)

    # --- Persist ---
    features_path = os.path.join(PROC_DIR, "clean_features.parquet")
    labels_path   = os.path.join(PROC_DIR, "labels.parquet")

    df.to_parquet(features_path, index=False)
    labels_df.to_parquet(labels_path, index=False)

    log.info("Saved clean_features  →  %s", features_path)
    log.info("Saved labels          →  %s", labels_path)
    log.info("Phase 1 complete.\n")

    return df, labels_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_preprocessing()