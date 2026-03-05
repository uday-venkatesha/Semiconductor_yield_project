"""
Phase 4: Data Engineering & SQLite Pipeline
────────────────────────────────────────────
Loads all three DataFrames produced by Phases 1–3 and populates a local SQLite
database (manufacturing_yield.db) with three normalised tables, then creates
the five analytical views needed for the Tableau dashboard.
"""

import pandas as pd
import pickle
import sqlite3
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = 'data/artifacts'
DB_PATH       = 'data/db/manufacturing_yield.db'
SQL_VIEWS_FILE = 'sql/05_tableau_views.sql'
os.makedirs('data/db', exist_ok=True)

# ── 1. Load Artifacts from Previous Phases ────────────────────────────────────
print("Loading artifacts from Phases 1–3...")
df_model             = pd.read_parquet(f'{ARTIFACTS_DIR}/df_model.parquet')
final_predictions_df = pd.read_parquet(f'{ARTIFACTS_DIR}/final_predictions.parquet')
root_cause_df        = pd.read_parquet(f'{ARTIFACTS_DIR}/root_cause_df.parquet')

with open(f'{ARTIFACTS_DIR}/active_sensors.pkl', 'rb') as f:
    active_sensors = pickle.load(f)

print(f"  production_logs rows  : {len(df_model)}")
print(f"  ml_predictions rows   : {len(final_predictions_df)}")
print(f"  root_cause_analysis   : {len(root_cause_df)}\n")

# ── 2. Connect to SQLite ───────────────────────────────────────────────────────
print(f"Connecting to SQLite database at '{DB_PATH}'...")
conn   = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Enable foreign-key enforcement
cursor.execute("PRAGMA foreign_keys = ON;")

# ── 3. Create Normalised Tables ────────────────────────────────────────────────
# We use DROP IF EXISTS so the script is safely re-runnable
cursor.executescript("""
    DROP TABLE IF EXISTS root_cause_analysis;
    DROP TABLE IF EXISTS ml_predictions;
    DROP TABLE IF EXISTS production_logs;

    -- All cleaned wafer records + every active sensor reading
    CREATE TABLE production_logs (
        Wafer_ID      TEXT PRIMARY KEY,
        Timestamp     DATETIME NOT NULL,
        Actual_Result INTEGER  NOT NULL
        -- Sensor columns are appended dynamically by Pandas (to_sql)
    );

    -- One row per wafer in the test set
    CREATE TABLE ml_predictions (
        Wafer_ID            TEXT    PRIMARY KEY,
        Predicted_Class     INTEGER NOT NULL,
        Failure_Probability REAL    NOT NULL,
        FOREIGN KEY (Wafer_ID) REFERENCES production_logs(Wafer_ID)
    );

    -- Up to 3 root-cause sensor rows per predicted failure
    CREATE TABLE root_cause_analysis (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        Wafer_ID    TEXT    NOT NULL,
        Sensor_Name TEXT    NOT NULL,
        SHAP_Value  REAL    NOT NULL,
        Rank        INTEGER NOT NULL,
        FOREIGN KEY (Wafer_ID) REFERENCES ml_predictions(Wafer_ID)
    );
""")
conn.commit()
print("Tables created (or recreated).\n")

# ── 4. Insert DataFrames ───────────────────────────────────────────────────────
# production_logs → full df_model (all wafers + sensor columns)
# Pandas to_sql handles the dynamic sensor columns automatically
print("Inserting production_logs  (all wafers + sensor readings)...")
cols_for_logs = ['Wafer_ID', 'Timestamp', 'Actual_Result'] + active_sensors
df_model[cols_for_logs].to_sql(
    'production_logs', conn, if_exists='replace', index=False
)

# ml_predictions → test-set subset only
print("Inserting ml_predictions   (test-set predictions)...")
preds_to_insert = final_predictions_df[
    ['Wafer_ID', 'Predicted_Class', 'Failure_Probability']
]
preds_to_insert.to_sql('ml_predictions', conn, if_exists='append', index=False)

# root_cause_analysis → one row per (wafer, sensor) pair
print("Inserting root_cause_analysis (SHAP-based root causes)...")
root_cause_df[['Wafer_ID', 'Sensor_Name', 'SHAP_Value', 'Rank']].to_sql(
    'root_cause_analysis', conn, if_exists='append', index=False
)
conn.commit()

# ── 5. Create Analytical SQL Views from file ──────────────────────────────────
if os.path.exists(SQL_VIEWS_FILE):
    print(f"\nApplying SQL views from '{SQL_VIEWS_FILE}'...")
    with open(SQL_VIEWS_FILE, 'r') as f:
        sql_script = f.read()
    # SQLite doesn't support executescript for CREATE VIEW alongside other
    # statements cleanly if IF NOT EXISTS is absent, so we split manually
    for statement in sql_script.split(';'):
        stmt = statement.strip()
        if stmt:
            try:
                cursor.execute(stmt)
            except sqlite3.OperationalError as e:
                # Views already exist on re-run — safe to ignore
                if 'already exists' not in str(e):
                    raise
    conn.commit()
    print("Views applied successfully.")
else:
    print(f"⚠️  SQL views file not found at '{SQL_VIEWS_FILE}' — skipping.")

# ── 6. Quick Sanity-Check Queries ─────────────────────────────────────────────
print("\n─── Row counts ──────────────────────────────────────────────────────────")
for table in ['production_logs', 'ml_predictions', 'root_cause_analysis']:
    count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table:<25}: {count:,} rows")

conn.close()
print(f"\n✅ Phase 4 complete — database ready at '{DB_PATH}'")