import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import seaborn as sns, warnings, json, os, joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,roc_auc_score,f1_score,
    precision_score,recall_score,confusion_matrix,roc_curve,
    precision_recall_curve,average_precision_score)
from sklearn.impute import SimpleImputer
import xgboost as xgb, lightgbm as lgb, shap

OUT="."
os.makedirs(f"{OUT}/plots",exist_ok=True); os.makedirs(f"{OUT}/models",exist_ok=True)
BLUE="#2A6DD9";CORAL="#E85D3A";TEAL="#1D9E75";AMBER="#EF9F27";PURPLE="#7F77DD";GRAY="#888780"
plt.rcParams.update({'figure.facecolor':'#FAFAF8','axes.facecolor':'#FAFAF8',
    'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#E8E6E0','grid.linewidth':0.6})

def generate_dataset(n=2000,seed=42):
    rng=np.random.RandomState(seed)
    age=rng.gamma(4,7,n).clip(21,81).astype(int)
    bmi=rng.normal(31.9,7.9,n).clip(18,67)
    gluc=np.where(rng.random(n)<0.35, rng.normal(155,25,n), rng.normal(100,18,n)).clip(44,199)
    bp=rng.normal(72,14,n).clip(40,120)
    ins=rng.exponential(85,n).clip(0,846)
    dpf=rng.exponential(0.47,n).clip(0.08,2.4)
    preg=rng.poisson(3.8,n).clip(0,17)
    sft=rng.normal(22,13,n).clip(0,90)
    for arr,r in [(gluc,0.005),(bp,0.045),(sft,0.295),(ins,0.489),(bmi,0.014)]:
        arr[rng.random(n)<r]=0
    # Strong realistic signal
    logit=(-8 + 0.055*gluc + 0.12*bmi + 0.022*age + 0.8*dpf + 0.005*ins
            - 0.01*bp + 0.15*preg)
    outcome=(rng.random(n)<1/(1+np.exp(-logit))).astype(int)
    return pd.DataFrame({'Pregnancies':preg,'Glucose':gluc,'BloodPressure':bp,
        'SkinThickness':sft,'Insulin':ins,'BMI':bmi,'DiabetesPedigreeFunction':dpf,
        'Age':age,'Outcome':outcome})

def run_eda(df):
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    fig.suptitle("Feature Distributions by Outcome",fontsize=14,fontweight='bold',y=1.01)
    features=[c for c in df.columns if c!='Outcome']
    for ax,feat in zip(axes.flat,features):
        for out,col,lbl in [(0,BLUE,'No Diabetes'),(1,CORAL,'Diabetes')]:
            ax.hist(df.loc[df['Outcome']==out,feat],bins=28,alpha=0.6,color=col,label=lbl,density=True)
        ax.set_title(feat,fontsize=10,fontweight='bold'); ax.set_ylabel("Density",fontsize=8)
    axes.flat[-1].axis('off'); axes.flat[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/01_distributions.png",dpi=130,bbox_inches='tight'); plt.close()

    fig,ax=plt.subplots(figsize=(9,7))
    sns.heatmap(df.corr(),mask=np.triu(np.ones_like(df.corr(),dtype=bool)),
        cmap=sns.diverging_palette(220,10,as_cmap=True),vmax=0.8,vmin=-0.8,
        annot=True,fmt=".2f",linewidths=0.5,ax=ax,annot_kws={"size":9},square=True)
    ax.set_title("Feature Correlation Matrix",fontsize=13,fontweight='bold',pad=12)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/02_correlation.png",dpi=130,bbox_inches='tight'); plt.close()

    fig,ax=plt.subplots(figsize=(5,4))
    counts=df['Outcome'].value_counts()
    bars=ax.bar(['No Diabetes','Diabetes'],counts.values,color=[BLUE,CORAL],width=0.5)
    for bar,v in zip(bars,counts.values):
        ax.text(bar.get_x()+bar.get_width()/2,v+5,f"{v}\n({v/len(df)*100:.1f}%)",ha='center',fontsize=11,fontweight='bold')
    ax.set_title("Class Distribution",fontsize=13,fontweight='bold'); ax.set_ylim(0,counts.max()*1.25)
    ax.yaxis.grid(True,alpha=0.4); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/03_class_imbalance.png",dpi=130,bbox_inches='tight'); plt.close()
    print("EDA plots saved ✓")

def feature_engineering(df):
    df=df.copy()
    clin=['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    for c in clin: df[c]=df[c].replace(0,np.nan)
    imp=SimpleImputer(strategy='median'); df[clin]=imp.fit_transform(df[clin])
    df['BMI_Obese']=(df['BMI']>=30).astype(int)
    df['HighGlucose']=(df['Glucose']>=140).astype(int)
    df['Age_bin']=pd.cut(df['Age'],bins=[0,30,45,60,100],labels=['<30','30-45','45-60','60+']).astype(str)
    df['Glucose_BMI']=df['Glucose']*df['BMI']/1000
    df['Glucose_Age']=df['Glucose']*df['Age']/1000
    df['Insulin_Gluc']=df['Insulin']/(df['Glucose']+1)
    df['BMI_Age']=df['BMI']*df['Age']/100
    df['Risk_Score']=((df['Glucose']>140).astype(int)*2+(df['BMI']>30).astype(int)*1.5+(df['Age']>45).astype(int)+(df['DiabetesPedigreeFunction']>0.5).astype(int))
    print(f"Feature engineering done. Shape: {df.shape} ✓")
    return df

def train_models(df):
    FEATS=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
           'DiabetesPedigreeFunction','Age','Glucose_BMI','Glucose_Age','Insulin_Gluc',
           'BMI_Age','BMI_Obese','HighGlucose','Risk_Score']
    X=df[FEATS]; y=df['Outcome']
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    sc=StandardScaler(); X_tr_sc=sc.fit_transform(X_tr); X_te_sc=sc.transform(X_te)
    sw=y_tr.map({0:1.0,1:(y_tr==0).sum()/(y_tr==1).sum()})
    MODELS={
        'Logistic Regression':LogisticRegression(C=1.0,max_iter=1000,random_state=42,class_weight='balanced'),
        'Random Forest':RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=3,class_weight='balanced',random_state=42),
        'XGBoost':xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.85,
            colsample_bytree=0.85,min_child_weight=3,gamma=0.1,reg_alpha=0.05,reg_lambda=1.5,
            scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),eval_metric='auc',random_state=42,verbosity=0),
        'LightGBM':lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.05,num_leaves=40,
            min_child_samples=15,class_weight='balanced',random_state=42,verbose=-1)}
    results={}; trained={}; roc_data={}; pr_data={}
    for name,model in MODELS.items():
        Xf,Xft=(X_tr_sc,X_te_sc) if name=='Logistic Regression' else (X_tr,X_te)
        model.fit(Xf,y_tr)
        yp=model.predict(Xft); ypr=model.predict_proba(Xft)[:,1]
        results[name]={'Accuracy':round(accuracy_score(y_te,yp),4),'AUC-ROC':round(roc_auc_score(y_te,ypr),4),
            'F1':round(f1_score(y_te,yp),4),'Precision':round(precision_score(y_te,yp),4),
            'Recall':round(recall_score(y_te,yp),4),'Avg Precision':round(average_precision_score(y_te,ypr),4)}
        trained[name]=model
        roc_data[name]=roc_curve(y_te,ypr)
        pr_data[name]=precision_recall_curve(y_te,ypr)
        print(f"  {name:<25} AUC={results[name]['AUC-ROC']:.4f}  F1={results[name]['F1']:.4f}  ACC={results[name]['Accuracy']:.4f}")

    COLS=[BLUE,TEAL,CORAL,AMBER]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
    for (nm,rd),col in zip(roc_data.items(),COLS):
        a1.plot(rd[0],rd[1],color=col,lw=2,label=f"{nm} (AUC={results[nm]['AUC-ROC']:.3f})")
    a1.plot([0,1],[0,1],'--',color=GRAY,lw=1); a1.set_xlabel("FPR"); a1.set_ylabel("TPR")
    a1.set_title("ROC Curves — All Models",fontweight='bold',fontsize=12); a1.legend(fontsize=9); a1.grid(True,alpha=0.3)
    for (nm,pd_),col in zip(pr_data.items(),COLS):
        a2.plot(pd_[1],pd_[0],color=col,lw=2,label=f"{nm} (AP={results[nm]['Avg Precision']:.3f})")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title("Precision-Recall Curves",fontweight='bold',fontsize=12)
    a2.legend(fontsize=9); a2.grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/04_roc_pr.png",dpi=130,bbox_inches='tight'); plt.close()

    metrics=['Accuracy','AUC-ROC','F1','Precision','Recall']
    fig,ax=plt.subplots(figsize=(13,5))
    x=np.arange(len(metrics)); w=0.18
    for i,(nm,col) in enumerate(zip(results.keys(),COLS)):
        ax.bar(x+i*w,[results[nm][m] for m in metrics],width=w,color=col,label=nm,alpha=0.88)
    ax.set_xticks(x+w*1.5); ax.set_xticklabels(metrics,fontsize=11); ax.set_ylim(0.5,1.05)
    ax.set_title("Model Performance Comparison",fontweight='bold',fontsize=13); ax.legend(fontsize=9)
    ax.yaxis.grid(True,alpha=0.35); ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/05_model_comparison.png",dpi=130,bbox_inches='tight'); plt.close()

    cm=confusion_matrix(y_te,trained['XGBoost'].predict(X_te))
    fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=['No DM','DM'],yticklabels=['No DM','DM'],linewidths=0.5,linecolor='white')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("XGBoost Confusion Matrix",fontweight='bold')
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/06_confusion_matrix.png",dpi=130,bbox_inches='tight'); plt.close()
    print("Model plots saved ✓")
    return results,trained,sc,X_tr,X_te,y_tr,y_te,FEATS

def run_shap(trained,X_tr,X_te,feats):
    model=trained['XGBoost']; exp=shap.TreeExplainer(model); sv=exp.shap_values(X_te)
    imp=pd.DataFrame({'Feature':feats,'SHAP':np.abs(sv).mean(axis=0)}).sort_values('SHAP',ascending=True)
    fig,ax=plt.subplots(figsize=(8,6))
    med=imp['SHAP'].median()
    ax.barh(imp['Feature'],imp['SHAP'],color=[CORAL if v>med else BLUE for v in imp['SHAP']],alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|"); ax.set_title("SHAP Feature Importance — XGBoost",fontweight='bold',fontsize=13)
    ax.xaxis.grid(True,alpha=0.3); ax.set_axisbelow(True)
    ax.legend(handles=[mpatches.Patch(color=CORAL,alpha=0.85,label='High importance'),mpatches.Patch(color=BLUE,alpha=0.85,label='Lower importance')],fontsize=9)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/07_shap_bar.png",dpi=130,bbox_inches='tight'); plt.close()

    fig,ax=plt.subplots(figsize=(9,6))
    shap.summary_plot(sv,X_te,feature_names=feats,show=False,plot_type="dot",max_display=12,plot_size=None)
    plt.title("SHAP Beeswarm — Feature Impact Direction",fontweight='bold',fontsize=13,pad=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/08_shap_beeswarm.png",dpi=130,bbox_inches='tight'); plt.close()

    idx=np.where(X_te['Glucose'].values>X_te['Glucose'].quantile(0.75))[0][0]
    shap_exp=shap.Explanation(values=sv[idx],base_values=exp.expected_value,data=X_te.iloc[idx].values,feature_names=feats)
    fig,ax=plt.subplots(figsize=(9,6))
    shap.waterfall_plot(shap_exp,max_display=10,show=False)
    plt.title("SHAP Waterfall — High-Risk Patient",fontweight='bold',fontsize=13,pad=10)
    plt.tight_layout(); plt.savefig(f"{OUT}/plots/09_shap_waterfall.png",dpi=130,bbox_inches='tight'); plt.close()
    print("SHAP plots saved ✓")
    return imp

if __name__=='__main__':
    df=generate_dataset(); run_eda(df)
    df2=feature_engineering(df)
    results,trained,sc,X_tr,X_te,y_tr,y_te,feats=train_models(df2)
    imp=run_shap(trained,X_tr,X_te,feats)
    joblib.dump(trained['XGBoost'],f"{OUT}/models/xgboost_model.pkl")
    joblib.dump(sc,f"{OUT}/models/scaler.pkl")
    joblib.dump(feats,f"{OUT}/models/feature_cols.pkl")
    with open(f"{OUT}/models/metrics.json",'w') as f: json.dump(results,f,indent=2)
    demo=df2[['Age','BMI','Glucose','BloodPressure','Insulin','DiabetesPedigreeFunction','Pregnancies','Outcome','Age_bin']].copy()
    demo['Patient_ID']=[f"P{i:04d}" for i in range(len(demo))]
    demo.to_csv(f"{OUT}/data/patients.csv",index=False)
    best=results['XGBoost']['AUC-ROC']; base=results['Logistic Regression']['AUC-ROC']
    print(f"\n✓ XGBoost AUC-ROC: {best:.4f} | Baseline (LR): {base:.4f} | Δ AUC: {best-base:+.4f}")
