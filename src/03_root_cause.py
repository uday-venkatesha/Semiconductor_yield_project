import shap

# Initialize SHAP Tree Explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test set
# For XGBoost binary classification, shap_values is usually a 2D array: (n_samples, n_features)
shap_values = explainer.shap_values(X_test)

# Extract top 3 sensors for predicted failures
rca_records = []
failure_indices = np.where(y_pred == 1)[0] # get row indices in the test set where prediction is 1

for idx in failure_indices:
    # Map back to the original wafer ID
    wafer_id = final_predictions_df.iloc[idx]['Wafer_ID']
    
    # Get SHAP values for this specific wafer
    wafer_shap_vals = shap_values[idx]
    
    # Find the indices of the top 3 largest absolute SHAP values
    top_3_indices = np.argsort(np.abs(wafer_shap_vals))[-3:][::-1]
    
    for feature_idx in top_3_indices:
        rca_records.append({
            'Wafer_ID': wafer_id,
            'Sensor_Name': active_sensors[feature_idx],
            'SHAP_Value': float(wafer_shap_vals[feature_idx]) # positive SHAP = pushed towards failure
        })

# Create the Root Cause Analysis DataFrame
root_cause_df = pd.DataFrame(rca_records)
print(f"Generated {len(root_cause_df)} root cause records for {len(failure_indices)} predicted failures.")