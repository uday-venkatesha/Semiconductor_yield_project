import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

# 1. Load the dataset (Assuming standard UCI format, which is often space-separated, but we'll default to CSV as requested)
# Note: Adjust 'sep' if your local copy is space or tab delimited.
df=pd.read_csv('data/raw/secom.csv', header=True, sep=' ') 
X_raw = df.drop(columns=['Pass/Fail'])  # Assuming 'Label' is the target column
y_raw = df['Pass/Fail']

# Standardize labels: SECOM uses -1 (Pass) and 1 (Fail). Convert to 0 (Pass) and 1 (Fail).
y = y_raw.apply(lambda x: 0 if x == -1 else 1)

# Generate mock identifiers and timestamps (spanning last 30 days)
num_records = len(X_raw)
start_time = datetime.now() - timedelta(days=30)
timestamps = [start_time + timedelta(minutes=(i * 27.5)) for i in range(num_records)] # ~1567 records over 30 days
wafer_ids = [f"WFR_{str(i).zfill(5)}" for i in range(num_records)]

# Assign column names to sensors
X_raw.columns = [f"Sensor_{i}" for i in range(X_raw.shape[1])]

# Insert Metadata
X_raw.insert(0, 'Timestamp', timestamps)
X_raw.insert(0, 'Wafer_ID', wafer_ids)

# 2. Handle Missing Values
# Drop features with > 50% missing values
threshold = len(X_raw) * 0.5
X_cleaned = X_raw.dropna(thresh=threshold, axis=1).copy()

# Impute remaining missing values with the median
imputer = SimpleImputer(strategy='median')
sensor_cols = [col for col in X_cleaned.columns if col.startswith('Sensor_')]

# We fit/transform only on the sensor columns, keeping metadata intact
X_cleaned[sensor_cols] = imputer.fit_transform(X_cleaned[sensor_cols])

# 3. Drop Zero-Variance Features (Constant columns)
selector = VarianceThreshold(threshold=0.0)
selector.fit(X_cleaned[sensor_cols])
active_sensors = [sensor_cols[i] for i in range(len(sensor_cols)) if selector.get_support()[i]]

# Final preprocessed DataFrame for modeling
df_model = X_cleaned[['Wafer_ID', 'Timestamp'] + active_sensors].copy()
df_model['Actual_Result'] = y