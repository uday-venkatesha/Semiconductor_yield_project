import sqlite3

# Connect to local SQLite DB (creates it if it doesn't exist)
db_path = 'manufacturing_yield.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. Create normalized tables
cursor.executescript("""
    DROP TABLE IF EXISTS production_logs;
    CREATE TABLE production_logs (
        Wafer_ID TEXT PRIMARY KEY,
        Timestamp DATETIME,
        Actual_Result INTEGER
        -- Sensor columns will be appended dynamically by Pandas
    );

    DROP TABLE IF EXISTS ml_predictions;
    CREATE TABLE ml_predictions (
        Wafer_ID TEXT PRIMARY KEY,
        Predicted_Class INTEGER,
        Failure_Probability REAL,
        FOREIGN KEY (Wafer_ID) REFERENCES production_logs(Wafer_ID)
    );

    DROP TABLE IF EXISTS root_cause_analysis;
    CREATE TABLE root_cause_analysis (
        Wafer_ID TEXT,
        Sensor_Name TEXT,
        SHAP_Value REAL,
        FOREIGN KEY (Wafer_ID) REFERENCES ml_predictions(Wafer_ID)
    );
""")
conn.commit()

# 2. Insert DataFrames into SQLite
# We use the full df_model for production logs to simulate historical data
df_model.to_sql('production_logs', conn, if_exists='replace', index=False)

# Insert predictions (dropping timestamp/actual_result as they exist in production_logs)
preds_to_insert = final_predictions_df[['Wafer_ID', 'Predicted_Class', 'Failure_Probability']]
preds_to_insert.to_sql('ml_predictions', conn, if_exists='append', index=False)

# Insert RCA
root_cause_df.to_sql('root_cause_analysis', conn, if_exists='append', index=False)

print("Data successfully loaded into SQLite database.")