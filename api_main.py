"""
Disease Prediction REST API — FastAPI
POST /predict  → diabetes risk prediction
GET  /metrics  → model comparison metrics
GET  /health   → health check
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np, joblib, json, os

from sqlalchemy import create_engine
import pandas as pd

engine = create_engine(
    "postgresql://postgres:Niranjan%402004@localhost:5432/diabetes_db"
)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL  = joblib.load(f"{BASE}/models/xgboost_model.pkl")
SCALER = joblib.load(f"{BASE}/models/scaler.pkl")
FEATS  = joblib.load(f"{BASE}/models/feature_cols.pkl")
with open(f"{BASE}/models/metrics.json") as f:
    METRICS = json.load(f)

app = FastAPI(title="Disease Prediction API", version="1.0.0",
              description="XGBoost-powered diabetes risk prediction (Pima Indians dataset)")

class PatientInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20, example=3)
    Glucose: float = Field(..., ge=0, le=300, example=148)
    BloodPressure: float = Field(..., ge=0, le=200, example=72)
    SkinThickness: float = Field(..., ge=0, le=100, example=35)
    Insulin: float = Field(..., ge=0, le=900, example=0)
    BMI: float = Field(..., ge=0, le=80, example=33.6)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3, example=0.627)
    Age: int = Field(..., ge=18, le=100, example=50)

class PredictionResponse(BaseModel):
    patient_risk: str
    risk_probability: float
    prediction: int
    risk_level: str
    top_risk_factors: list
    model: str = "XGBoost"
    auc_roc: float = 0.8376

def engineer_features(data: dict) -> np.ndarray:
    g, b, a = data['Glucose'], data['BMI'], data['Age']
    ins, dpf = data['Insulin'], data['DiabetesPedigreeFunction']
    feats = {
        'Pregnancies': data['Pregnancies'],
        'Glucose': g, 'BloodPressure': data['BloodPressure'],
        'SkinThickness': data['SkinThickness'], 'Insulin': ins,
        'BMI': b, 'DiabetesPedigreeFunction': dpf, 'Age': a,
        'Glucose_BMI': g * b / 1000,
        'Glucose_Age': g * a / 1000,
        'Insulin_Gluc': ins / (g + 1),
        'BMI_Age': b * a / 100,
        'BMI_Obese': int(b >= 30),
        'HighGlucose': int(g >= 140),
        'Risk_Score': int(g > 140)*2 + int(b > 30)*1.5 + int(a > 45) + int(dpf > 0.5)
    }
    return np.array([[feats[f] for f in FEATS]])

@app.get("/health")
def health():
    return {"status": "ok", "model": "XGBoost", "features": len(FEATS)}

@app.get("/metrics")
def get_metrics():
    return {"model_comparison": METRICS, "best_model": "XGBoost", "best_auc": 0.8376}

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    data = patient.dict()
    X = engineer_features(data)
    prob = float(MODEL.predict_proba(X)[0][1])
    pred = int(prob >= 0.5)
    risk = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"
    # Feature contributions (simplified)
    factors = []
    if data['Glucose'] > 140: factors.append(f"High glucose ({data['Glucose']:.0f} mg/dL)")
    if data['BMI'] > 30: factors.append(f"Obese BMI ({data['BMI']:.1f})")
    if data['Age'] > 45: factors.append(f"Age risk group ({data['Age']} yrs)")
    if data['DiabetesPedigreeFunction'] > 0.5: factors.append(f"Family history (DPF={data['DiabetesPedigreeFunction']:.2f})")
    if data['Insulin'] > 200: factors.append(f"Elevated insulin ({data['Insulin']:.0f} μU/mL)")
    record = pd.DataFrame([{
    "glucose": data["Glucose"],
    "bmi": data["BMI"],
    "age": data["Age"],
    "pregnancies": data["Pregnancies"],
    "insulin": data["Insulin"],
    "blood_pressure": data["BloodPressure"],
    "dpf": data["DiabetesPedigreeFunction"],
    "risk_probability": prob,
    "prediction": pred,
    "risk_level": risk
     }])

    record.to_sql(
       "predictions",
        engine,
        if_exists="append",
        index=False
    )
    return PredictionResponse(
        patient_risk=f"{prob*100:.1f}% diabetes probability",
        risk_probability=round(prob, 4),
        prediction=pred,
        risk_level=risk,
        top_risk_factors=factors or ["No single dominant risk factor identified"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
