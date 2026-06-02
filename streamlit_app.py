"""
Disease Prediction System — Streamlit Frontend
Deploy: streamlit run streamlit_app.py
"""
import streamlit as st
import numpy as np, pandas as pd, json, joblib, os, requests
from pathlib import Path

BASE = Path(__file__).parent
st.set_page_config(page_title="Disease Prediction System", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:12px;padding:16px 20px;border:1px solid #e0e0e0;text-align:center}
.metric-val{font-size:2rem;font-weight:700;color:#2A6DD9}
.risk-high{background:#fdecea;color:#c62828;padding:6px 14px;border-radius:8px;font-weight:700}
.risk-medium{background:#fff8e1;color:#e65100;padding:6px 14px;border-radius:8px;font-weight:700}
.risk-low{background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:8px;font-weight:700}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
st.sidebar.title("🏥 Disease Prediction")
st.sidebar.markdown("**Pima Indians Diabetes Dataset**")
page = st.sidebar.radio("Navigate", ["🔍 Patient Prediction","📊 EDA Dashboard","🤖 Model Comparison","📈 SHAP Explainability","📋 API Reference"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Best Model:** XGBoost")
st.sidebar.markdown("**AUC-ROC:** 0.8376")
st.sidebar.markdown("**Dataset:** 2,000 patients")

# ── Load artifacts ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return (joblib.load(BASE/"models/xgboost_model.pkl"),
            joblib.load(BASE/"models/scaler.pkl"),
            joblib.load(BASE/"models/feature_cols.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv(BASE/"data/patients.csv")

@st.cache_data
def load_metrics():
    with open(BASE/"models/metrics.json") as f: return json.load(f)

model, scaler, feats = load_model()
df = load_data()
metrics = load_metrics()

def engineer(d):
    g,b,a,ins,dpf = d['Glucose'],d['BMI'],d['Age'],d['Insulin'],d['DiabetesPedigreeFunction']
    return {**d,'Glucose_BMI':g*b/1000,'Glucose_Age':g*a/1000,'Insulin_Gluc':ins/(g+1),
            'BMI_Age':b*a/100,'BMI_Obese':int(b>=30),'HighGlucose':int(g>=140),
            'Risk_Score':int(g>140)*2+int(b>30)*1.5+int(a>45)+int(dpf>0.5)}

# ══════════════════════════════════════════════════════════════
if page == "🔍 Patient Prediction":
    st.title("🔍 Patient Diabetes Risk Prediction")
    st.markdown("Enter patient vitals to get an instant XGBoost-powered risk assessment.")
    c1,c2,c3 = st.columns(3)
    with c1:
        pregnancies = st.slider("Pregnancies", 0, 17, 3)
        glucose     = st.slider("Glucose (mg/dL)", 44, 200, 120)
        bp          = st.slider("Blood Pressure (mmHg)", 24, 122, 72)
        skin        = st.slider("Skin Thickness (mm)", 0, 99, 20)
    with c2:
        insulin = st.slider("Insulin (μU/mL)", 0, 846, 80)
        bmi     = st.slider("BMI", 18.0, 67.0, 32.0, 0.1)
        dpf     = st.slider("Diabetes Pedigree Function", 0.08, 2.4, 0.47, 0.01)
        age     = st.slider("Age", 18, 81, 35)

    if st.button("🔮 Predict Risk", use_container_width=True):
        data = {'Pregnancies':pregnancies,'Glucose':glucose,'BloodPressure':bp,
                'SkinThickness':skin,'Insulin':insulin,'BMI':bmi,
                'DiabetesPedigreeFunction':dpf,'Age':age}
        eng  = engineer(data)
        X    = np.array([[eng[f] for f in feats]])
        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)
        risk = "HIGH" if prob>=0.7 else "MEDIUM" if prob>=0.4 else "LOW"
        with c3:
            st.markdown(f"### Risk Assessment")
            color = "#c62828" if risk=="HIGH" else "#e65100" if risk=="MEDIUM" else "#1b5e20"
            st.markdown(f"<h1 style='color:{color};font-size:3rem'>{prob*100:.1f}%</h1>", unsafe_allow_html=True)
            st.markdown(f"**Risk Level:** `{risk}`")
            st.markdown(f"**Prediction:** {'🔴 Diabetes' if pred else '🟢 No Diabetes'}")
            st.progress(prob)
            factors = []
            if glucose>140: factors.append(f"⚠️ High glucose: {glucose} mg/dL")
            if bmi>30: factors.append(f"⚠️ Obese BMI: {bmi:.1f}")
            if age>45: factors.append(f"⚠️ Age risk: {age} yrs")
            if dpf>0.5: factors.append(f"⚠️ Family history (DPF={dpf:.2f})")
            if factors:
                st.markdown("**Key Risk Factors:**")
                for f2 in factors: st.markdown(f2)

elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Patients", f"{len(df):,}")
    col2.metric("Diabetic", f"{df['Outcome'].sum():,} ({df['Outcome'].mean()*100:.1f}%)")
    col3.metric("Avg Glucose", f"{df['Glucose'].mean():.1f} mg/dL")
    col4.metric("Avg BMI", f"{df['BMI'].mean():.1f}")
    st.image(str(BASE/"plots/01_distributions.png"), caption="Feature Distributions by Outcome", use_column_width=True)
    c1,c2 = st.columns(2)
    c1.image(str(BASE/"plots/02_correlation.png"), caption="Correlation Matrix", use_column_width=True)
    c2.image(str(BASE/"plots/03_class_imbalance.png"), caption="Class Distribution", use_column_width=True)
    st.markdown("### Patient Data Sample")
    st.dataframe(df.head(20), use_container_width=True)

elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")
    rows = [{"Model":k,"Accuracy":v["Accuracy"],"AUC-ROC":v["AUC-ROC"],
             "F1":v["F1"],"Precision":v["Precision"],"Recall":v["Recall"]}
            for k,v in metrics.items()]
    mdf = pd.DataFrame(rows).sort_values("AUC-ROC",ascending=False).reset_index(drop=True)
    st.dataframe(mdf.style.highlight_max(subset=["AUC-ROC","F1","Accuracy"],color="#d4edda"), use_container_width=True)
    st.image(str(BASE/"plots/05_model_comparison.png"), caption="Metric Comparison", use_column_width=True)
    c1,c2 = st.columns(2)
    c1.image(str(BASE/"plots/04_roc_pr.png"), caption="ROC & PR Curves", use_column_width=True)
    c2.image(str(BASE/"plots/06_confusion_matrix.png"), caption="XGBoost Confusion Matrix", use_column_width=True)

elif page == "📈 SHAP Explainability":
    st.title("📈 SHAP — Model Explainability")
    st.info("SHAP (SHapley Additive exPlanations) shows which features drive each prediction.")
    st.image(str(BASE/"plots/07_shap_bar.png"), caption="Mean |SHAP| Feature Importance", use_column_width=True)
    st.image(str(BASE/"plots/08_shap_beeswarm.png"), caption="SHAP Beeswarm — Impact Direction & Magnitude", use_column_width=True)
    st.image(str(BASE/"plots/09_shap_waterfall.png"), caption="SHAP Waterfall — Single High-Risk Patient", use_column_width=True)

elif page == "📋 API Reference":
    st.title("📋 FastAPI Reference")
    st.code("""# Start the API
uvicorn api.main:app --reload --port 8000

# Predict
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"Pregnancies":3,"Glucose":148,"BloodPressure":72,
       "SkinThickness":35,"Insulin":0,"BMI":33.6,
       "DiabetesPedigreeFunction":0.627,"Age":50}'

# Get model metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health""", language="bash")
    st.markdown("### Response Example")
    st.json({"patient_risk":"73.4% diabetes probability","risk_probability":0.734,
             "prediction":1,"risk_level":"HIGH",
             "top_risk_factors":["High glucose (148 mg/dL)","Obese BMI (33.6)","Age risk group (50 yrs)"],
             "model":"XGBoost","auc_roc":0.8376})
