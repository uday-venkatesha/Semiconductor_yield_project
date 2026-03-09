-- =============================================================================
-- Phase 5: Analytical SQL Views for Tableau
-- =============================================================================
-- These views are created programmatically by phase4_sql_pipeline.py.
-- This file provides the STANDALONE SQL for:
--   (a) direct inspection / documentation
--   (b) porting to PostgreSQL / Snowflake / BigQuery in production
--
-- How to connect Tableau Desktop to the SQLite database:
--   1. Install the SQLite ODBC driver (http://www.ch-werner.de/sqliteodbc/)
--   2. In Tableau: Connect → Other Databases (ODBC) → SQLite ODBC Driver
--   3. Set the database path to: <project_root>/database/manufacturing_yield.db
--   4. Each view below will appear as a data source in Tableau.
-- =============================================================================


-- =============================================================================
-- VIEW 1: v_yield_kpis
-- Purpose : Daily production KPIs for executive scorecard and trend analysis.
-- Columns : Production_Date, Total_Wafers, Actual_Failures,
--           Actual_Defect_Rate_Pct, Predicted_Failures,
--           Predicted_Defect_Rate_Pct, Avg_Failure_Probability
-- Tableau : Line chart (date on X), dual-axis Actual vs Predicted defect rate.
--           KPI cards for today's yield.
-- =============================================================================

CREATE VIEW IF NOT EXISTS v_yield_kpis AS
SELECT
    DATE(p.Timestamp)                          AS Production_Date,
    COUNT(DISTINCT p.Wafer_ID)                 AS Total_Wafers,

    -- Actual yield metrics
    SUM(p.Actual_Result)                       AS Actual_Failures,
    ROUND(
        100.0 * SUM(p.Actual_Result)
              / NULLIF(COUNT(p.Wafer_ID), 0),
    2)                                         AS Actual_Defect_Rate_Pct,

    -- Predicted yield metrics (early warning signal)
    SUM(p.Predicted_Class)                     AS Predicted_Failures,
    ROUND(
        100.0 * SUM(p.Predicted_Class)
              / NULLIF(COUNT(p.Wafer_ID), 0),
    2)                                         AS Predicted_Defect_Rate_Pct,

    -- Average model confidence for that day's flagged wafers
    ROUND(
        AVG(CASE WHEN p.Predicted_Class = 1
                 THEN p.Failure_Probability END),
    4)                                         AS Avg_Failure_Probability

FROM ml_predictions p
GROUP BY DATE(p.Timestamp)
ORDER BY Production_Date;


-- =============================================================================
-- VIEW 2: v_high_risk_sensors
-- Purpose : Identify sensors most responsible for failures in the last 7 days.
--           Drives the "Root Cause" page of the Tableau dashboard.
-- Columns : Sensor_Name, Affected_Wafers, Avg_Abs_SHAP, Max_Abs_SHAP,
--           Total_SHAP_Impact, Total_Appearances, Impact_Rank
-- Tableau : Horizontal bar chart ranked by Total_SHAP_Impact.
--           Filter control to adjust the 7-day look-back window.
-- Note    : Adjust the date offset ('-7 days') as needed for your review cadence.
-- =============================================================================

CREATE VIEW IF NOT EXISTS v_high_risk_sensors AS
SELECT
    r.Sensor_Name,
    COUNT(DISTINCT r.Wafer_ID)              AS Affected_Wafers,
    ROUND(AVG(ABS(r.SHAP_Value)), 6)        AS Avg_Abs_SHAP,
    ROUND(MAX(ABS(r.SHAP_Value)), 6)        AS Max_Abs_SHAP,
    ROUND(SUM(ABS(r.SHAP_Value)), 6)        AS Total_SHAP_Impact,
    COUNT(*)                                AS Total_Appearances,

    -- Window function ranks sensors by cumulative impact
    RANK() OVER (
        ORDER BY SUM(ABS(r.SHAP_Value)) DESC
    )                                       AS Impact_Rank

FROM root_cause_analysis r
WHERE
    -- Rolling 7-day window anchored to the most recent record in the table.
    -- In a live pipeline, replace with:  WHERE DATE(r.Timestamp) >= DATE('now', '-7 days')
    DATE(r.Timestamp) >= DATE(
        (SELECT MAX(Timestamp) FROM root_cause_analysis),
        '-7 days'
    )
GROUP BY r.Sensor_Name
ORDER BY Total_SHAP_Impact DESC;


-- =============================================================================
-- VIEW 3: v_process_stability
-- Purpose : Statistical Process Control (SPC) monitoring.
--           Tracks how the most impactful sensor varies over time and
--           correlates raw sensor variance with predicted failure probability.
-- Columns : Wafer_ID, Timestamp, Production_Date, Predicted_Class,
--           Failure_Probability, Actual_Result, Monitored_Sensor,
--           Rolling5_Avg_Fail_Prob
-- Tableau : Dual-axis chart — sensor value on primary axis,
--           Rolling5_Avg_Fail_Prob on secondary axis.
--           Add reference lines for ±3σ control limits.
-- =============================================================================

CREATE VIEW IF NOT EXISTS v_process_stability AS

WITH top_sensor AS (
    -- Dynamically identify the single highest-impact sensor across ALL time.
    -- This drives the "Monitored_Sensor" label in the view.
    SELECT Sensor_Name
    FROM   root_cause_analysis
    GROUP  BY Sensor_Name
    ORDER  BY SUM(ABS(SHAP_Value)) DESC
    LIMIT  1
)

SELECT
    pl.Wafer_ID,
    pl.Timestamp,
    DATE(pl.Timestamp)                          AS Production_Date,

    mp.Predicted_Class,
    mp.Failure_Probability,
    mp.Actual_Result,

    ts.Sensor_Name                              AS Monitored_Sensor,

    -- Rolling 5-wafer average failure probability.
    -- Smooths noise and shows process drift before failures cluster.
    ROUND(
        AVG(mp.Failure_Probability) OVER (
            ORDER BY pl.Timestamp
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ),
    4)                                          AS Rolling5_Avg_Fail_Prob,

    -- Rolling standard deviation of failure probability (process spread).
    -- Rising σ indicates an unstable process — an early warning signal.
    -- SQLite lacks STDDEV; computed here as SQRT(variance) via window AVG.
    ROUND(
        SQRT(
            MAX(
                AVG(mp.Failure_Probability * mp.Failure_Probability) OVER (
                    ORDER BY pl.Timestamp
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                )
                -
                (AVG(mp.Failure_Probability) OVER (
                    ORDER BY pl.Timestamp
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                )) * (AVG(mp.Failure_Probability) OVER (
                    ORDER BY pl.Timestamp
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                )),
                0.0
            )
        ),
    6)                                          AS Rolling5_StdDev_Fail_Prob

FROM production_logs pl
JOIN ml_predictions  mp  ON pl.Wafer_ID = mp.Wafer_ID
CROSS JOIN top_sensor ts
ORDER BY pl.Timestamp;


-- =============================================================================
-- BONUS VIEW: v_wafer_failure_detail
-- Purpose : Row-level detail combining predictions + RCA for drill-through.
--           In Tableau: clicking a wafer in any chart navigates here.
-- =============================================================================

CREATE VIEW IF NOT EXISTS v_wafer_failure_detail AS
SELECT
    mp.Wafer_ID,
    mp.Timestamp,
    mp.Actual_Result,
    mp.Predicted_Class,
    mp.Failure_Probability,
    r.Sensor_Name,
    r.SHAP_Value,
    r.Rank                                       AS Sensor_Rank,

    -- Classify contribution direction
    CASE
        WHEN r.SHAP_Value > 0 THEN 'Pushes Toward Fail'
        ELSE                       'Pushes Toward Pass'
    END                                          AS SHAP_Direction

FROM ml_predictions       mp
LEFT JOIN root_cause_analysis r  ON mp.Wafer_ID = r.Wafer_ID
WHERE mp.Predicted_Class = 1      -- only predicted failures (drill-through focus)
ORDER BY mp.Timestamp DESC, mp.Failure_Probability DESC, r.Rank;