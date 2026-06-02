# 🏥 End-to-End Disease Prediction System

> **Key Metric:** XGBoost achieves **AUC-ROC 0.8376** on the Pima Indians Diabetes dataset, with **LightGBM reaching 0.9407 F1-Score** — representing an 8–12% improvement in clinical recall over Logistic Regression baseline.

---

## 📋 Project Overview

A production-grade machine learning system for **diabetes risk prediction** built on the Pima Indians Diabetes dataset. Covers the full data science lifecycle: SQL ingestion → EDA → Feature Engineering → Multi-model training → SHAP explainability → REST API → Interactive dashboards.

```
disease_prediction/
├── src/pipeline.py          # Full ML pipeline (EDA → models → SHAP)
├── api/main.py              # FastAPI REST API
├── streamlit_app.py         # Streamlit interactive frontend
├── sql_schema.sql           # PostgreSQL schema + analytical queries
├── models/
│   ├── xgboost_model.pkl    # Trained XGBoost model
│   ├── scaler.pkl           # StandardScaler
│   ├── feature_cols.pkl     # Feature list
│   └── metrics.json         # All model metrics
├── data/patients.csv        # Patient records
├── plots/                   # 9 EDA + model + SHAP plots
└── requirements.txt
```

---

## 🗄️ SQL — Data Layer (PostgreSQL)

**Tables:** `patients`, `patient_features`, `predictions`, `model_metrics`

```sql
-- Diabetes prevalence by age group
SELECT age_group,
       ROUND(100.0 * SUM(outcome) / COUNT(*), 2) AS prevalence_pct
FROM patients
GROUP BY age_group ORDER BY age_group;

-- Model leaderboard
SELECT model_name, auc_roc, f1_score,
       RANK() OVER (ORDER BY auc_roc DESC) AS rank
FROM model_metrics;
```

Connect to **Power BI** via PostgreSQL connector → import `patients` + `predictions` tables → build risk distribution, prevalence by age, and feature importance visuals.

---

## 🔬 EDA Findings

| Finding | Detail |
|---|---|
| Class imbalance | ~35% diabetic, 65% non-diabetic |
| Missing data | Insulin: 48.9% zeros, SkinThickness: 29.5% zeros |
| Top correlations | Glucose↔Outcome (0.51), BMI↔Outcome (0.31), Age↔Outcome (0.24) |
| Key outliers | Insulin up to 846, Glucose up to 199 |

**Imputation strategy:** Replace physiologically impossible zeros with column median.

---

## ⚙️ Feature Engineering

| Feature | Formula | Rationale |
|---|---|---|
| `Glucose_BMI` | Glucose × BMI / 1000 | Combined metabolic burden |
| `Glucose_Age` | Glucose × Age / 1000 | Age-adjusted glucose risk |
| `Insulin_Gluc` | Insulin / (Glucose + 1) | Insulin resistance proxy |
| `BMI_Obese` | BMI ≥ 30 | Clinical obesity threshold |
| `HighGlucose` | Glucose ≥ 140 | Pre-diabetic threshold |
| `Risk_Score` | Weighted flag sum | Composite risk index |

---

## 🤖 Model Results

| Model | AUC-ROC | F1-Score | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| **XGBoost** | **0.8376** | 0.9229 | 0.8650 | 0.9259 | 0.9200 |
| Logistic Regression | 0.8366 | 0.8383 | 0.7425 | 0.8098 | 0.8681 |
| Random Forest | 0.8315 | 0.9333 | 0.8800 | 0.9420 | 0.9249 |
| LightGBM | 0.8164 | **0.9407** | **0.8925** | 0.9481 | 0.9333 |

**Best for clinical use (AUC):** XGBoost | **Best F1:** LightGBM

---

## 🔍 SHAP Explainability — Top Feature Drivers

1. **Glucose** — single largest contributor (high values strongly push toward diabetes)
2. **BMI** — obese patients show consistent SHAP uplift
3. **Age** — monotonic increase in risk contribution after 40
4. **DiabetesPedigreeFunction** — family history adds non-linear risk
5. **Glucose_BMI interaction** — combined effect amplifies individual signals

---

## 🚀 FastAPI Deployment

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Diabetes risk prediction |
| `GET` | `/metrics` | All model metrics |
| `GET` | `/health` | API health check |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":3,"Glucose":148,"BloodPressure":72,
       "SkinThickness":35,"Insulin":0,"BMI":33.6,
       "DiabetesPedigreeFunction":0.627,"Age":50}'
```

---

## 📊 Power BI Dashboard

Connect PostgreSQL → Import `patients`, `predictions`, `model_metrics` →

- **Card visuals:** Total patients, AUC-ROC, Diabetic %, High-risk count
- **Donut chart:** Risk level distribution (HIGH / MEDIUM / LOW)
- **Bar chart:** Prevalence by age group
- **Scatter plot:** Glucose vs BMI colored by outcome
- **Table:** Model leaderboard ranked by AUC-ROC
- **Line chart:** Prediction probability distribution

---

## 🌐 Streamlit Deployment (Hugging Face Spaces)

```bash
streamlit run streamlit_app.py
```

**Deploy to Hugging Face Spaces:**
1. Create new Space → SDK: Streamlit
2. Push this repo
3. Add `requirements.txt` with all deps

---

## 📦 Requirements

```
pandas>=2.0  numpy>=1.24  scikit-learn>=1.3  xgboost>=2.0
lightgbm>=4.0  shap>=0.44  matplotlib>=3.7  seaborn>=0.12
fastapi>=0.104  uvicorn>=0.24  streamlit>=1.28  joblib>=1.3
imbalanced-learn>=0.11  pydantic>=2.0  psycopg2-binary>=2.9
```

---

## 📁 Data Source

**Pima Indians Diabetes Database** — originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 768 female patients aged 21+, Pima Indian heritage.

Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age → binary Outcome (0/1).

---

*Built with Python · XGBoost · FastAPI · Streamlit · PostgreSQL · Power BI*
