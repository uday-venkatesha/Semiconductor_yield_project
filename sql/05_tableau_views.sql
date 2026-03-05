-- =============================================================================
-- Phase 5: Analytical SQL Views for Tableau
-- =============================================================================
-- Run order: after 04_database_pipeline.py has populated the three tables.
-- Each view is preceded by DROP VIEW IF EXISTS so this script is re-runnable.
-- =============================================================================


-- ── View 1: Daily Yield KPIs ──────────────────────────────────────────────────
-- Provides a daily time-series of actual vs. predicted defect rates.
-- Use in Tableau as a dual-axis line chart (date on X, rates on Y).
DROP VIEW IF EXISTS v_yield_kpis;
CREATE VIEW v_yield_kpis AS
SELECT
    DATE(l.Timestamp)                                                      AS Production_Date,
    COUNT(l.Wafer_ID)                                                      AS Total_Wafers_Processed,
    SUM(l.Actual_Result)                                                   AS Actual_Defects,
    ROUND(
        CAST(SUM(l.Actual_Result) AS FLOAT) / COUNT(l.Wafer_ID) * 100, 2
    )                                                                      AS Actual_Defect_Rate_Pct,
    -- Predictions only exist for the test set (≈20% of wafers); COALESCE to 0 elsewhere
    SUM(COALESCE(p.Predicted_Class, 0))                                    AS Predicted_Defects,
    ROUND(
        CAST(SUM(COALESCE(p.Predicted_Class, 0)) AS FLOAT) / COUNT(l.Wafer_ID) * 100, 2
    )                                                                      AS Predicted_Defect_Rate_Pct
FROM production_logs l
LEFT JOIN ml_predictions p ON l.Wafer_ID = p.Wafer_ID
GROUP BY DATE(l.Timestamp)
ORDER BY Production_Date ASC;


-- ── View 2: High-Risk Sensors (Rolling 7-Day Window) ──────────────────────────
-- Identifies which sensors appear most frequently as root causes and carry the
-- highest average SHAP impact over the last 7 days.
-- Use in Tableau as a horizontal bar chart (sorted by Average_Impact_Score).
DROP VIEW IF EXISTS v_high_risk_sensors;
CREATE VIEW v_high_risk_sensors AS
SELECT
    r.Sensor_Name,
    COUNT(r.Wafer_ID)        AS Frequency_as_Root_Cause,
    ROUND(AVG(r.SHAP_Value), 6) AS Average_Impact_Score,
    ROUND(AVG(r.Rank), 1)    AS Average_Rank          -- lower = higher priority
FROM root_cause_analysis r
JOIN production_logs l ON r.Wafer_ID = l.Wafer_ID
WHERE DATE(l.Timestamp) >= DATE('now', '-7 days')
GROUP BY r.Sensor_Name
ORDER BY Average_Impact_Score DESC;


-- ── View 3: Process Stability (Sensor Drift Over Time) ────────────────────────
-- Joins predictions with the raw sensor log so Tableau can plot a specific
-- sensor's value over time alongside failure probability.
-- NOTE: replace 'Sensor_59' below with whatever your top high-risk sensor is
--       after reviewing v_high_risk_sensors.  The column must exist in
--       production_logs (i.e. it must have survived the variance filter in
--       Phase 1).  If Sensor_59 was dropped, substitute a sensor from the
--       v_high_risk_sensors view.
DROP VIEW IF EXISTS v_process_stability;
CREATE VIEW v_process_stability AS
SELECT
    l.Wafer_ID,
    l.Timestamp,
    l.Actual_Result,
    COALESCE(p.Predicted_Class,     0)    AS Predicted_Class,
    COALESCE(p.Failure_Probability, 0.0)  AS Failure_Probability,
    -- ↓ Swap this column name with your actual top-risk sensor name
    l.Sensor_59                           AS Target_Sensor_Value
FROM production_logs l
LEFT JOIN ml_predictions p ON l.Wafer_ID = p.Wafer_ID
ORDER BY l.Timestamp ASC;