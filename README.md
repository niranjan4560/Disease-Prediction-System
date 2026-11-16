# Disease Prediction System

A diabetes-risk prediction pipeline taken from data to a served model: feature engineering, four models compared on a held-out split, SHAP explanations for the chosen model, a FastAPI prediction endpoint that logs to PostgreSQL, and a Streamlit front end.

> **About the data.** The 2,000 patient records are **synthetic**. `pipeline.py` generates them with the same eight features as the Pima Indians Diabetes dataset and a logistic risk function, so the numbers below describe how the pipeline behaves, not clinical performance. This is a learning project and not a medical tool.

## What it does

| Stage | File | Detail |
|---|---|---|
| Data generation and EDA | `pipeline.py` | Generates the dataset, plots distributions, correlations and class balance |
| Feature engineering | `pipeline.py` | Median imputation of physiologically impossible zeros, plus seven engineered features |
| Training and evaluation | `pipeline.py` | Logistic Regression, Random Forest, XGBoost and LightGBM on a stratified 80/20 split |
| Explainability | `pipeline.py` | SHAP `TreeExplainer` on XGBoost: importance bar, beeswarm and a single-patient waterfall |
| Serving | `api_main.py` | FastAPI `POST /predict`, `GET /metrics`, `GET /health`; each prediction is written to PostgreSQL |
| Interface | `streamlit_app.py` | Five pages: prediction form, EDA, model comparison, SHAP plots, API reference |
| Data layer | `sql_schema.sql` | Tables for patients, features, predictions and model metrics, with analytical queries |

## Engineered features

| Feature | Formula | Rationale |
|---|---|---|
| `Glucose_BMI` | Glucose × BMI / 1000 | Combined metabolic load |
| `Glucose_Age` | Glucose × Age / 1000 | Age-adjusted glucose |
| `Insulin_Gluc` | Insulin / (Glucose + 1) | Insulin-resistance proxy |
| `BMI_Age` | BMI × Age / 100 | Age-adjusted BMI |
| `BMI_Obese` | BMI ≥ 30 | Clinical obesity threshold |
| `HighGlucose` | Glucose ≥ 140 | Pre-diabetic threshold |
| `Risk_Score` | Weighted sum of four flags | Composite index |

## Results

Held-out test split of 400 records, from `models/metrics.json`:

| Model | AUC-ROC | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.8366 | 0.8383 | 0.7425 | 0.9639 | 0.7417 |
| XGBoost (served) | 0.8339 | 0.9200 | 0.8600 | 0.9471 | 0.8944 |
| Random Forest | 0.8315 | 0.9333 | 0.8800 | 0.9333 | 0.9333 |
| LightGBM | 0.8164 | 0.9407 | 0.8925 | 0.9342 | 0.9472 |

How to read this:

- The generated dataset is about 90% positive, so F1, precision and recall are inflated by the class balance. A model that predicts "diabetic" for everyone would already score an F1 near 0.95. AUC-ROC is the more informative column.
- On AUC-ROC the four models are within 0.02 of each other, and logistic regression is marginally ahead. The tree models do not show a meaningful advantage on this data.

![ROC and precision-recall curves for all four models](plots/04_roc_pr.png)

## Explainability

SHAP values are computed for the XGBoost model on the test split. Glucose and BMI dominate, which matches the risk function used to generate the data.

![SHAP beeswarm plot showing feature impact direction and magnitude](plots/08_shap_beeswarm.png)

The API's `top_risk_factors` field uses fixed clinical thresholds (glucose above 140, BMI above 30 and so on). It does not compute SHAP values per request.

## Run it

```bash
pip install -r requirements.txt
```

Regenerate the data, plots and models (optional, the trained artefacts are committed):

```bash
python pipeline.py
```

Start the API. It needs a reachable PostgreSQL database for prediction logging:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@localhost:5432/diabetes_db"
uvicorn api_main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"Pregnancies":3,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

| Variable | Used by | Purpose |
|---|---|---|
| `DATABASE_URL` | `api_main.py` | PostgreSQL connection string for the `predictions` table |

## Power BI

The `predictions` and `patients` tables can be loaded into Power BI through its PostgreSQL connector for risk-distribution and prevalence reports. No `.pbix` file is included in this repository.

## Limitations and next steps

- Replace the synthetic data with the real Pima dataset (768 records) or a larger clinical dataset, and report cross-validated results.
- Rebalance the generated classes, or evaluate with balanced accuracy and calibration.
- Return per-request SHAP contributions from `/predict` in place of threshold rules.
- `sql_schema.sql` contains MySQL-style statements (`USE`, `CREATE DATABASE IF NOT EXISTS`) and needs adjusting to run unmodified on PostgreSQL.
- There are no automated tests yet.

## Stack

Python · scikit-learn · XGBoost · LightGBM · SHAP · FastAPI · Streamlit · PostgreSQL
