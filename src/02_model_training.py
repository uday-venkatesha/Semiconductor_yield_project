import xgboost as xgb
from sklearn.metrics import classification_report

# Isolate features and target
X_features = df_model[active_sensors]
Y_target = df_model['Actual_Result']

# Train/Test Split (80/20) - Ensure we don't shuffle time-series blindly in a real scenario, 
# but for SECOM's baseline we'll do a standard split with stratify to preserve the rare minority class.
X_train, X_test, y_train, y_test = train_test_split(
    X_features, Y_target, test_size=0.2, random_state=42, stratify=Y_target
)

# Calculate scale_pos_weight to handle heavy class imbalance
# formula: count(negative examples) / count(positive examples)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost Classifier
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    max_depth=4, 
    learning_rate=0.05
)

xgb_model.fit(X_train, y_train)

# Predictions & Probabilities
y_pred = xgb_model.predict(X_test)
y_prob = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation focusing on Recall, Precision, and F1
print("--- Model Evaluation ---")
print(classification_report(y_test, y_pred, target_names=['Pass (0)', 'Fail (1)']))

# Generate Final Results DataFrame for the test set
# We need to pull the original Wafer_ID and Timestamp using the X_test indices
test_indices = X_test.index
final_predictions_df = pd.DataFrame({
    'Wafer_ID': df_model.loc[test_indices, 'Wafer_ID'],
    'Timestamp': df_model.loc[test_indices, 'Timestamp'],
    'Actual_Result': y_test,
    'Predicted_Class': y_pred,
    'Failure_Probability': y_prob
})