-- ============================================================
-- Disease Prediction System — Database Schema (PostgreSQL)
-- ============================================================

CREATE DATABASE IF NOT EXISTS diabetes_prediction;
USE diabetes_prediction;

-- Raw patient records
CREATE TABLE patients (
    patient_id      VARCHAR(10)    PRIMARY KEY,
    age             INT            NOT NULL CHECK (age BETWEEN 18 AND 120),
    gender          VARCHAR(10)    DEFAULT 'Female',
    pregnancies     INT            DEFAULT 0,
    glucose         DECIMAL(6,2),
    blood_pressure  DECIMAL(6,2),
    skin_thickness  DECIMAL(6,2),
    insulin         DECIMAL(8,2),
    bmi             DECIMAL(5,2),
    dpf             DECIMAL(6,3),   -- DiabetesPedigreeFunction
    outcome         INT            CHECK (outcome IN (0, 1)),
    age_group       VARCHAR(20)    GENERATED ALWAYS AS (
        CASE WHEN age < 30  THEN '<30'
             WHEN age < 45  THEN '30-45'
             WHEN age < 60  THEN '45-60'
             ELSE '60+' END
    ) STORED,
    created_at      TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
);

-- Engineered features (denormalized for BI)
CREATE TABLE patient_features (
    patient_id      VARCHAR(10)    PRIMARY KEY REFERENCES patients(patient_id),
    glucose_bmi     DECIMAL(8,4),
    glucose_age     DECIMAL(8,4),
    insulin_gluc    DECIMAL(8,4),
    bmi_age         DECIMAL(8,4),
    bmi_obese       BOOLEAN,
    high_glucose    BOOLEAN,
    risk_score      DECIMAL(5,2),
    bmi_category    VARCHAR(20)
);

-- Model predictions log
CREATE TABLE predictions (
    prediction_id   SERIAL         PRIMARY KEY,
    patient_id      VARCHAR(10)    REFERENCES patients(patient_id),
    model_name      VARCHAR(50)    NOT NULL,
    probability     DECIMAL(6,4)   NOT NULL,
    prediction      INT            NOT NULL,
    risk_level      VARCHAR(10),
    predicted_at    TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
);

-- Model metrics registry
CREATE TABLE model_metrics (
    metric_id       SERIAL         PRIMARY KEY,
    model_name      VARCHAR(50)    NOT NULL,
    accuracy        DECIMAL(6,4),
    auc_roc         DECIMAL(6,4),
    f1_score        DECIMAL(6,4),
    precision_score DECIMAL(6,4),
    recall_score    DECIMAL(6,4),
    avg_precision   DECIMAL(6,4),
    trained_at      TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
);

-- ── Useful analytical queries ───────────────────────────────
-- Q1: Diabetes prevalence by age group
SELECT age_group,
       COUNT(*) AS total_patients,
       SUM(outcome) AS diabetic,
       ROUND(100.0 * SUM(outcome) / COUNT(*), 2) AS prevalence_pct
FROM patients
GROUP BY age_group ORDER BY age_group;

-- Q2: Average biomarkers by outcome
SELECT outcome,
       ROUND(AVG(glucose),2)        AS avg_glucose,
       ROUND(AVG(bmi),2)            AS avg_bmi,
       ROUND(AVG(age),1)            AS avg_age,
       ROUND(AVG(insulin),1)        AS avg_insulin,
       ROUND(AVG(blood_pressure),2) AS avg_bp,
       COUNT(*)                     AS patient_count
FROM patients GROUP BY outcome;

-- Q3: High-risk patients (glucose > 140 AND bmi > 30)
SELECT p.patient_id, p.age, p.glucose, p.bmi, p.outcome,
       pr.probability, pr.risk_level
FROM patients p
JOIN predictions pr ON p.patient_id = pr.patient_id
WHERE p.glucose > 140 AND p.bmi > 30 AND pr.model_name = 'XGBoost'
ORDER BY pr.probability DESC LIMIT 20;

-- Q4: Model leaderboard
SELECT model_name, auc_roc, f1_score, accuracy,
       RANK() OVER (ORDER BY auc_roc DESC) AS auc_rank
FROM model_metrics ORDER BY auc_roc DESC;

-- Q5: Risk distribution
SELECT risk_level, COUNT(*) AS count,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM predictions WHERE model_name = 'XGBoost'
GROUP BY risk_level;
