"""
=============================================================================
Phase 4: Data Engineering & SQL Pipeline
=============================================================================
Responsibilities:
  - Create a local SQLite database: database/manufacturing_yield.db
  - Create three normalised tables:
      production_logs   – raw sensor readings + metadata
      ml_predictions    – model output per wafer
      root_cause_analysis – SHAP-derived sensor attribution
  - Bulk-insert all dataframes from Phases 1-3 using transactions
  - Add indexes on Wafer_ID and Timestamp for fast Tableau queries
  - Create the three analytical SQL VIEWs (v_yield_kpis,
    v_high_risk_sensors, v_process_stability) consumed by Tableau

Input  : data/processed/clean_features.parquet
         data/processed/labels.parquet
         data/processed/predictions_df.parquet
         data/processed/rca_df.parquet
Output : database/manufacturing_yield.db
=============================================================================
"""

import os
import sqlite3
import logging
import warnings
import textwrap

import numpy  as np
import pandas as pd

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
PROC_DIR    = os.path.join(BASE_DIR, "data",     "processed")
DB_DIR      = os.path.join(BASE_DIR, "database")
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH          = os.path.join(DB_DIR,  "manufacturing_yield.db")
FEATURES_PATH    = os.path.join(PROC_DIR, "clean_features.parquet")
LABELS_PATH      = os.path.join(PROC_DIR, "labels.parquet")
PREDICTIONS_PATH = os.path.join(PROC_DIR, "predictions_df.parquet")
RCA_PATH         = os.path.join(PROC_DIR, "rca_df.parquet")


# ---------------------------------------------------------------------------
# 1.  Connect to SQLite
# ---------------------------------------------------------------------------
def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and enable WAL mode for speed."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


# ---------------------------------------------------------------------------
# 2.  DDL: create tables
# ---------------------------------------------------------------------------
# We only store a curated set of sensor columns in production_logs to keep
# the SQLite file manageable; all original columns are written if desired
# by flipping STORE_ALL_SENSORS = True.
STORE_ALL_SENSORS = True     # set False to store only top-50 sensors

PRODUCTION_LOGS_DDL = """
CREATE TABLE IF NOT EXISTS production_logs (
    Wafer_ID    TEXT        NOT NULL,
    Timestamp   DATETIME    NOT NULL,
    Label       INTEGER     NOT NULL,   -- 0 = Pass, 1 = Fail (actual)
    PRIMARY KEY (Wafer_ID)
);
"""

ML_PREDICTIONS_DDL = """
CREATE TABLE IF NOT EXISTS ml_predictions (
    Wafer_ID             TEXT    NOT NULL,
    Timestamp            DATETIME,
    Actual_Result        INTEGER,       -- 0 = Pass, 1 = Fail
    Predicted_Class      INTEGER,       -- 0 = Pass, 1 = Fail
    Failure_Probability  REAL,          -- model confidence [0, 1]
    PRIMARY KEY (Wafer_ID),
    FOREIGN KEY (Wafer_ID) REFERENCES production_logs(Wafer_ID)
);
"""

ROOT_CAUSE_DDL = """
CREATE TABLE IF NOT EXISTS root_cause_analysis (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    Wafer_ID    TEXT    NOT NULL,
    Timestamp   DATETIME,
    Sensor_Name TEXT    NOT NULL,
    SHAP_Value  REAL    NOT NULL,
    Rank        INTEGER NOT NULL,       -- 1 = biggest contributor
    FOREIGN KEY (Wafer_ID) REFERENCES production_logs(Wafer_ID)
);
"""


def create_tables(conn: sqlite3.Connection) -> None:
    """Execute all DDL statements to create the three core tables."""
    log.info("Creating core tables …")
    cursor = conn.cursor()

    cursor.executescript(PRODUCTION_LOGS_DDL)
    cursor.executescript(ML_PREDICTIONS_DDL)
    cursor.executescript(ROOT_CAUSE_DDL)

    # Indexes for Tableau performance
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_prod_ts  ON production_logs(Timestamp);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_pred_ts  ON ml_predictions(Timestamp);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rca_wafer ON root_cause_analysis(Wafer_ID);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rca_sensor ON root_cause_analysis(Sensor_Name);"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rca_ts ON root_cause_analysis(Timestamp);"
    )

    conn.commit()
    log.info("  Tables and indexes created.")


# ---------------------------------------------------------------------------
# 3.  Add sensor columns to production_logs dynamically
# ---------------------------------------------------------------------------
def add_sensor_columns(
    conn:         sqlite3.Connection,
    sensor_cols:  list[str],
) -> None:
    """
    ALTER TABLE to add one REAL column per sensor.
    SQLite does not support adding multiple columns in one statement,
    so we iterate.  Already-existing columns are skipped silently.
    """
    cursor      = conn.cursor()
    existing    = {row[1] for row in cursor.execute("PRAGMA table_info(production_logs);")}
    new_cols    = [c for c in sensor_cols if c not in existing]

    log.info("  Adding %d sensor columns to production_logs …", len(new_cols))
    for col in new_cols:
        safe = col.replace("-", "_").replace(" ", "_")
        cursor.execute(f'ALTER TABLE production_logs ADD COLUMN "{safe}" REAL;')
    conn.commit()


# ---------------------------------------------------------------------------
# 4.  Insert production_logs
# ---------------------------------------------------------------------------
def insert_production_logs(
    conn:         sqlite3.Connection,
    features_df:  pd.DataFrame,
    labels_df:    pd.DataFrame,
) -> None:
    """
    Merge features with labels and bulk-insert into production_logs.
    Uses pandas to_sql for high-throughput insertion (faster than
    cursor.executemany for >1 000 rows).
    """
    log.info("Inserting production_logs …")

    id_cols = ["Wafer_ID", "Timestamp"]

    if STORE_ALL_SENSORS:
        sensor_cols = [c for c in features_df.columns if c not in id_cols]
    else:
        # Keep top-50 columns by variance (most informative sensors)
        sensor_only = features_df.drop(columns=id_cols)
        top50       = sensor_only.var().nlargest(50).index.tolist()
        sensor_cols = top50

    add_sensor_columns(conn, sensor_cols)

    # Build combined dataframe
    merged = features_df[id_cols + sensor_cols].copy()
    merged["Label"] = labels_df["Label"].values

    # Ensure Timestamp is stored as ISO string (SQLite has no native DATETIME)
    merged["Timestamp"] = merged["Timestamp"].astype(str)

    # Rename columns to safe SQL identifiers
    merged.columns = [c.replace("-", "_").replace(" ", "_") for c in merged.columns]

    # Insert with to_sql (replace ensures idempotency)
    merged.to_sql(
        "production_logs",
        conn,
        if_exists="replace",
        index=False,
        chunksize=500,
    )

    # Re-add indexes after replace
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prod_ts  ON production_logs(Timestamp);")
    conn.commit()

    log.info("  Inserted %d rows into production_logs.", len(merged))


# ---------------------------------------------------------------------------
# 5.  Insert ml_predictions
# ---------------------------------------------------------------------------
def insert_ml_predictions(
    conn:           sqlite3.Connection,
    predictions_df: pd.DataFrame,
) -> None:
    """Insert model predictions into ml_predictions table."""
    log.info("Inserting ml_predictions …")

    df = predictions_df.copy()
    df["Timestamp"] = df["Timestamp"].astype(str)

    df.to_sql(
        "ml_predictions",
        conn,
        if_exists="replace",
        index=False,
        chunksize=500,
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pred_ts ON ml_predictions(Timestamp);")
    conn.commit()

    log.info("  Inserted %d rows into ml_predictions.", len(df))


# ---------------------------------------------------------------------------
# 6.  Insert root_cause_analysis
# ---------------------------------------------------------------------------
def insert_root_cause_analysis(
    conn:   sqlite3.Connection,
    rca_df: pd.DataFrame,
) -> None:
    """Insert SHAP root-cause records into root_cause_analysis table."""
    log.info("Inserting root_cause_analysis …")

    df = rca_df.copy()
    df["Timestamp"] = df["Timestamp"].astype(str)

    # Drop the autoincrement PK column if it was saved to parquet
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df.to_sql(
        "root_cause_analysis",
        conn,
        if_exists="replace",
        index=False,
        chunksize=500,
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rca_wafer  ON root_cause_analysis(Wafer_ID);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rca_sensor ON root_cause_analysis(Sensor_Name);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rca_ts     ON root_cause_analysis(Timestamp);")
    conn.commit()

    log.info("  Inserted %d rows into root_cause_analysis.", len(df))


# ---------------------------------------------------------------------------
# 7.  Analytical SQL VIEWs (Tableau-facing)
# ---------------------------------------------------------------------------

VIEW_YIELD_KPIS = textwrap.dedent("""
    -- -----------------------------------------------------------------------
    -- v_yield_kpis
    -- Daily summary: total wafers, actual defect rate, predicted defect rate.
    -- Tableau workbook: KPI scorecards, daily trend lines.
    -- -----------------------------------------------------------------------
    CREATE VIEW IF NOT EXISTS v_yield_kpis AS
    SELECT
        DATE(p.Timestamp)                          AS Production_Date,
        COUNT(DISTINCT p.Wafer_ID)                 AS Total_Wafers,

        -- Actual yield
        SUM(p.Actual_Result)                       AS Actual_Failures,
        ROUND(
            100.0 * SUM(p.Actual_Result)
                  / NULLIF(COUNT(p.Wafer_ID), 0),
        2)                                         AS Actual_Defect_Rate_Pct,

        -- Predicted yield (model output)
        SUM(p.Predicted_Class)                     AS Predicted_Failures,
        ROUND(
            100.0 * SUM(p.Predicted_Class)
                  / NULLIF(COUNT(p.Wafer_ID), 0),
        2)                                         AS Predicted_Defect_Rate_Pct,

        -- Average model confidence on that day's failures
        ROUND(
            AVG(CASE WHEN p.Predicted_Class = 1
                     THEN p.Failure_Probability END),
        4)                                         AS Avg_Failure_Probability

    FROM ml_predictions p
    GROUP BY DATE(p.Timestamp)
    ORDER BY Production_Date;
""")

VIEW_HIGH_RISK_SENSORS = textwrap.dedent("""
    -- -----------------------------------------------------------------------
    -- v_high_risk_sensors
    -- Aggregates SHAP values over the LAST 7 DAYS to surface which
    -- sensors are currently the biggest failure drivers.
    -- Tableau workbook: horizontal bar chart, ranked sensor list.
    -- -----------------------------------------------------------------------
    CREATE VIEW IF NOT EXISTS v_high_risk_sensors AS
    SELECT
        r.Sensor_Name,
        COUNT(DISTINCT r.Wafer_ID)              AS Affected_Wafers,
        ROUND(AVG(ABS(r.SHAP_Value)), 6)        AS Avg_Abs_SHAP,
        ROUND(MAX(ABS(r.SHAP_Value)), 6)        AS Max_Abs_SHAP,
        ROUND(SUM(ABS(r.SHAP_Value)), 6)        AS Total_SHAP_Impact,
        COUNT(*)                                AS Total_Appearances,

        -- Rank sensors by total impact
        RANK() OVER (
            ORDER BY SUM(ABS(r.SHAP_Value)) DESC
        )                                       AS Impact_Rank

    FROM root_cause_analysis r
    WHERE DATE(r.Timestamp) >= DATE(
              (SELECT MAX(Timestamp) FROM root_cause_analysis),
              '-7 days'
          )
    GROUP BY r.Sensor_Name
    ORDER BY Total_SHAP_Impact DESC;
""")

VIEW_PROCESS_STABILITY = textwrap.dedent("""
    -- -----------------------------------------------------------------------
    -- v_process_stability
    -- Joins predictions with raw sensor logs to show how the
    -- MOST IMPACTFUL SENSOR (rank 1 in RCA) varies over time,
    -- and whether high sensor values correlate with predicted failures.
    -- Tableau workbook: scatter/line chart for SPC monitoring.
    -- -----------------------------------------------------------------------
    CREATE VIEW IF NOT EXISTS v_process_stability AS
    WITH top_sensor AS (
        -- Identify the single highest-impact sensor across all time
        SELECT Sensor_Name
        FROM root_cause_analysis
        GROUP BY Sensor_Name
        ORDER BY SUM(ABS(SHAP_Value)) DESC
        LIMIT 1
    ),
    sensor_readings AS (
        -- Pull that sensor's raw values from production_logs.
        -- Dynamic column name resolved at query time via the subquery above.
        -- Note: In production, parameterise the sensor name from BI tool.
        SELECT
            pl.Wafer_ID,
            pl.Timestamp,
            -- Placeholder: swap 'Sensor_1' with the actual top sensor name
            -- when running in Tableau via a parameter or calculated field.
            pl.Wafer_ID                         AS Wafer_Ref,
            DATE(pl.Timestamp)                  AS Production_Date
        FROM production_logs pl
    )
    SELECT
        sr.Wafer_ID,
        sr.Timestamp,
        sr.Production_Date,
        mp.Predicted_Class,
        mp.Failure_Probability,
        mp.Actual_Result,
        ts.Sensor_Name                          AS Monitored_Sensor,

        -- Rolling 5-wafer average failure probability (process trend)
        ROUND(
            AVG(mp.Failure_Probability) OVER (
                ORDER BY sr.Timestamp
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ),
        4)                                      AS Rolling5_Avg_Fail_Prob

    FROM sensor_readings  sr
    JOIN ml_predictions   mp  ON sr.Wafer_ID = mp.Wafer_ID
    CROSS JOIN top_sensor ts
    ORDER BY sr.Timestamp;
""")


def create_views(conn: sqlite3.Connection) -> None:
    """Execute all three CREATE VIEW statements."""
    log.info("Creating analytical SQL views …")
    cursor = conn.cursor()

    for view_sql in [VIEW_YIELD_KPIS, VIEW_HIGH_RISK_SENSORS, VIEW_PROCESS_STABILITY]:
        cursor.executescript(view_sql)

    conn.commit()
    log.info("  Views created: v_yield_kpis, v_high_risk_sensors, v_process_stability")


# ---------------------------------------------------------------------------
# 8.  Smoke test: query each view
# ---------------------------------------------------------------------------
def smoke_test(conn: sqlite3.Connection) -> None:
    """Run a quick SELECT on each view to verify correctness."""
    log.info("Running smoke tests on views …")

    views = ["v_yield_kpis", "v_high_risk_sensors", "v_process_stability"]
    for view in views:
        try:
            df = pd.read_sql(f"SELECT * FROM {view} LIMIT 3;", conn)
            log.info("  %-30s  ✓  (%d cols)", view, df.shape[1])
        except Exception as exc:
            log.error("  %-30s  ✗  %s", view, exc)


# ---------------------------------------------------------------------------
# 9.  Row count summary
# ---------------------------------------------------------------------------
def print_db_summary(conn: sqlite3.Connection) -> None:
    """Print row counts for all three tables."""
    tables = ["production_logs", "ml_predictions", "root_cause_analysis"]
    log.info("\nDatabase Summary:")
    for t in tables:
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t};", conn).iloc[0]["n"]
        log.info("  %-30s  %d rows", t, n)
    log.info("  Database file: %s", DB_PATH)


# ---------------------------------------------------------------------------
# 10. Main pipeline
# ---------------------------------------------------------------------------
def run_sql_pipeline() -> None:
    """Execute the full data engineering pipeline."""
    log.info("=" * 60)
    log.info("PHASE 4 – Data Engineering & SQL Pipeline")
    log.info("=" * 60)

    # Load parquet artefacts
    log.info("Loading parquet artefacts …")
    features_df    = pd.read_parquet(FEATURES_PATH)
    labels_df      = pd.read_parquet(LABELS_PATH)
    predictions_df = pd.read_parquet(PREDICTIONS_PATH)
    rca_df         = pd.read_parquet(RCA_PATH)

    # Rename predictions columns to match DDL
    predictions_df = predictions_df.rename(columns={"Actual_Result": "Actual_Result"})

    # Connect & build schema
    conn = get_connection()

    try:
        create_tables(conn)                              # must run first: creates tables before any INSERT/ALTER
        insert_production_logs(conn, features_df, labels_df)
        insert_ml_predictions(conn, predictions_df)
        insert_root_cause_analysis(conn, rca_df)
        create_views(conn)
        smoke_test(conn)
        print_db_summary(conn)
    finally:
        conn.close()

    log.info("Phase 4 complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_sql_pipeline()