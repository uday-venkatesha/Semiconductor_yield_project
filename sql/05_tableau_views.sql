-- View 1: Yield KPIs (Daily Aggregation)
CREATE VIEW v_yield_kpis AS
SELECT 
    DATE(Timestamp) AS Production_Date,
    COUNT(Wafer_ID) AS Total_Wafers_Processed,
    SUM(Actual_Result) AS Actual_Defects,
    ROUND(CAST(SUM(Actual_Result) AS FLOAT) / COUNT(Wafer_ID) * 100, 2) AS Actual_Defect_Rate_Pct,
    
    -- Using LEFT JOIN because predictions only exist for the test set in our simulation
    SUM(COALESCE(p.Predicted_Class, 0)) AS Predicted_Defects,
    ROUND(CAST(SUM(COALESCE(p.Predicted_Class, 0)) AS FLOAT) / COUNT(Wafer_ID) * 100, 2) AS Predicted_Defect_Rate_Pct
FROM production_logs l
LEFT JOIN ml_predictions p ON l.Wafer_ID = p.Wafer_ID
GROUP BY DATE(Timestamp);

-- View 2: High-Risk Sensors (Last 7 Days)
-- Identifies which sensors are most frequently cited as root causes for failure
CREATE VIEW v_high_risk_sensors AS
SELECT 
    r.Sensor_Name,
    COUNT(r.Wafer_ID) as Frequency_as_Root_Cause,
    AVG(r.SHAP_Value) as Average_Impact_Score
FROM root_cause_analysis r
JOIN production_logs l ON r.Wafer_ID = l.Wafer_ID
WHERE DATE(l.Timestamp) >= DATE('now', '-7 days')
GROUP BY r.Sensor_Name
ORDER BY Average_Impact_Score DESC;

-- View 3: Process Stability (Sensor Variance)
-- A foundational view for a Tableau line chart to visualize a specific high-risk sensor drifting
CREATE VIEW v_process_stability AS
SELECT 
    l.Wafer_ID,
    l.Timestamp,
    l.Actual_Result,
    COALESCE(p.Predicted_Class, 0) AS Predicted_Class,
    p.Failure_Probability,
    -- We select Sensor_59 as a mock high-risk sensor example. 
    -- In Tableau, you'd unpivot the table or select the specific sensor dynamically.
    l.Sensor_59 AS Target_Sensor_Value 
FROM production_logs l
LEFT JOIN ml_predictions p ON l.Wafer_ID = p.Wafer_ID
ORDER BY l.Timestamp ASC;