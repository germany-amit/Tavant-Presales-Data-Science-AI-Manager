# Streamlit MVP: Lead DS/ML Skills Showcase (Free‑tier, single file)
# Covers: Pre‑sales, Classical ML, Deep Learning lite, GenAI (optional), MLOps, Leadership/Comms
# Run:  pip install -r requirements.txt && streamlit run app.py

import os, json, uuid, random, math
from datetime import datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Lead DS/ML — Skills MVP", layout="wide")

# -------- Helpers (tiny + free) --------
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
POS = set("good great excellent positive success efficient scalable robust accurate reliable secure innovative value win".split())
NEG = set("bad poor negative slow risk issue bug failure inaccurate overfit brittle costly delay".split())

def safe_generate(style: str, src: str) -> str:
    base = {
        "Executive": "Write a crisp executive summary with goals, approach, ROI.",
        "Technical": "Draft a technical proposal: data, features, model, MLOps, risks.",
        "Email": "Write a short polite email acknowledging the JD and proposing next steps."}
    prompt = base.get(style, "Executive") + "\n\nSource:\n" + src[:2000]
    if not OPENAI_KEY:
        return (f"**{style} Draft**\n- Problem → Value\n- Solution → Data→Features→Baselines→Ensembles/Transformers\n"
                f"- Plan → Discovery, Sprints, Hardening\n- MLOps → CI/CD, registry, monitoring\n- Risks → Data gaps, drift, latency, security\n")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0.2, max_tokens=500,
            messages=[{"role":"system","content":"Senior DS/ML solution architect."},{"role":"user","content":prompt}])
        return r.choices[0].message.content
    except Exception as e:
        return "(LLM unavailable)\n" + safe_generate(style="Executive", src=src)

def heuristic_sentiment(text:str)->str:
    words=[w.strip(".,;:!?()[]{}").lower() for w in text.split()]
    p=sum(w in POS for w in words); n=sum(w in NEG for w in words)
    return "positive" if p>n else ("negative" if n>p else "neutral")

# -------- Demo scenarios (2–3 via radio) --------
scenario = st.sidebar.radio("Select Demo Scenario", ["Customer Churn (RFP)", "Fraud Detection", "Sales Forecasting"], index=0)

# Preloaded mini RFP/text per scenario (kept tiny)
RFP = {
"Customer Churn (RFP)": "RFP: Reduce churn by 10% in 2 quarters. Deploy prediction + retention offers on cloud; PII compliant; <100ms scoring.",
"Fraud Detection": "RFP: Flag fraudulent card transactions in near real-time with low false positives; explainability + monitoring.",
"Sales Forecasting": "RFP: 12‑month rolling forecast per category; handle promotions/seasonality; dashboard + alerting."
}

# Tabs mapping to JD
T1, T2, T3, T4, T5 = st.tabs(["Pre‑Sales", "Classical ML", "GenAI / NLP", "MLOps", "Leadership & Cloud"])

# -------- Tab 1: Pre‑Sales (solutioning, scoping) --------
with T1:
    st.subheader("RFP Assistant & Estimation")
    st.text_area("Active RFP / Transcript", value=RFP[scenario], height=120)
    c1,c2 = st.columns(2)
    with c1:
        style = st.selectbox("Generate", ["Executive","Technical","Email"], index=0)
        if st.button("Create Draft", key="draft"):
            st.markdown(safe_generate(style, RFP[scenario]))
    with c2:
        st.caption("Heuristic Effort Estimator")
        ds = st.number_input("# Data Sources",1,20,3); depth = st.selectbox("Modeling Depth", ["Baseline","Ensemble","DL/Transformer"],1)
        team = st.slider("Team Size (FTE)",2,10,4); infra = st.selectbox("Cloud", ["On‑Prem","AWS","Azure","GCP"],1)
        if st.button("Estimate", key="est"):
            weeks = max(6, min(4+ds + {"Baseline":0,"Ensemble":2,"DL/Transformer":4}[depth], 20))
            effort = weeks*team; cost = int(effort*2500)
            st.metric("Timeline (weeks)", weeks); st.metric("Effort (FTE‑weeks)", effort); st.metric("Indicative Cost (USD)", f"{cost:,}")
            st.caption("For discovery; refine with real constraints & SOW.")

# -------- Tab 2: Classical ML (regression/classification/clustering) --------
with T2:
    st.subheader("Experiments")
    if scenario=="Customer Churn (RFP)":
        # synthetic churn
        X,y = load_iris(return_X_y=True)  # tiny but fine for demo; treat as churn/not-churn
        y = (y>0).astype(int)
        model = st.selectbox("Model", ["LogReg","RandomForest"], index=1)
        Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
        if model=="LogReg":
            clf = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))])
        else:
            clf = RandomForestClassifier(n_estimators=150, random_state=42)
        if st.button("Train & Evaluate", key="train1"):
            clf.fit(Xtr,Ytr); p = clf.predict(Xte); proba = getattr(clf,"predict_proba",lambda z: np.c_[1-p,p])(Xte)[:,1]
            acc = accuracy_score(Yte,p); auc = roc_auc_score(Yte, proba)
            st.metric("Accuracy", f"{acc:.3f}"); st.metric("AUC", f"{auc:.3f}")
            st.text(classification_report(Yte,p))
            cm = confusion_matrix(Yte,p)
            fig,ax=plt.subplots(); im=ax.imshow(cm); ax.set_title("Confusion Matrix"); ax.set_xlabel("Pred"); ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]): ax.text(j,i,cm[i,j],ha='center',va='center')
            st.pyplot(fig)
    elif scenario=="Fraud Detection":
        # synthetic fraud via contamination
        rng=np.random.RandomState(42)
        normal = rng.normal(0,1,(800,3)); fraud = rng.normal(4,1,(20,3))
        X = np.vstack([normal,fraud]); y = np.array([0]*len(normal)+[1]*len(fraud))
        clf = IsolationForest(contamination=0.02, random_state=42)
        if st.button("Detect Anomalies", key="train2"):
            s = -clf.fit_predict(X)  # 1 for normal, -1 anomaly → flip
            pred = (s>0).astype(int)
            tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum()); fn = int(((pred==0)&(y==1)).sum())
            st.metric("TP (Fraud caught)", tp); st.metric("FP (False alarms)", fp); st.metric("FN (Missed)", fn)
            fig,ax=plt.subplots(); ax.scatter(X[:,0], X[:,1], c=pred); ax.set_title("Predicted anomalies (1)"); st.pyplot(fig)
    else:  # Sales Forecasting (simple, CPU‑tiny)
        # seasonal synthetic series
        t=np.arange(60); series=100+10*np.sin(2*np.pi*t/12)+0.5*t+np.random.RandomState(0).normal(0,2,60)
        train, test = series[:48], series[48:]
        k = st.slider("Moving Avg Window", 1, 12, 6)
        if st.button("Forecast", key="train3"):
            def movavg(a,w):
                return np.convolve(a, np.ones(w)/w, mode='valid')
            hist_fore = movavg(train, k)
            last = hist_fore[-1] if len(hist_fore) else train[-1]
            fc = np.full_like(test, last)
            mape = np.mean(np.abs((test-fc)/np.maximum(1e-6, np.abs(test))))*100
            st.metric("MAPE (%)", f"{mape:.2f}")
            fig,ax=plt.subplots(); ax.plot(series,label='Series'); ax.axvline(48,ls='--'); ax.plot(np.arange(48,60), fc,label='Forecast'); ax.legend(); st.pyplot(fig)

# -------- Tab 3: GenAI / NLP (LLM optional, plus sentiment) --------
with T3:
    st.subheader("Generative Text & Sentiment (free‑tier)")
    src = st.text_area("Input text (RFP, transcript, email)", value=RFP[scenario], height=120)
    c1,c2 = st.columns(2)
    with c1:
        st.caption("Executive/Technical/Email drafts (LLM optional)")
        s = st.selectbox("Style", ["Executive","Technical","Email"], index=0)
        if st.button("Generate Text", key="gen"):
            st.markdown(safe_generate(s, src))
    with c2:
        st.caption("Heuristic Sentiment — no external calls")
        if st.button("Analyze Sentiment", key="sent"):
            lines=[l.strip() for l in src.split('\n') if l.strip()][:30]
            df=pd.DataFrame({"sentence":[l[:120] for l in lines], "sentiment":[heuristic_sentiment(l) for l in lines]})
            st.dataframe(df)

# -------- Tab 4: MLOps (registry, promotion, drift sim) --------
with T4:
    st.subheader("Lightweight Model Registry & Monitoring (simulated)")
    if "registry" not in st.session_state: st.session_state.registry={"models":[],"prod":None}
    reg = st.session_state.registry

    if st.button("Save Sample Model Version", key="savev"):
        vid=str(uuid.uuid4())[:8]; meta={"id":vid,"scenario":scenario,"created":datetime.utcnow().isoformat()+"Z","metrics":{"acc":round(random.uniform(0.8,0.97),3)}}
        reg["models"].append(meta); st.success(f"Registered {vid}")
    if reg["models"]:
        st.dataframe(pd.DataFrame(reg["models"]))
        choice = st.selectbox("Promote to Production", [m["id"] for m in reg["models"]])
        if st.button("Promote", key="prom"):
            reg["prod"]=choice; st.success(f"Promoted {choice} to prod")
    else:
        st.info("Click 'Save Sample Model Version' to create entries.")

    if st.button("Run Drift Check", key="drift", disabled=reg.get("prod") is None)==True:
        prod_acc = 0.92
        live = round(prod_acc - random.uniform(0,0.15),3)
        st.metric("Prod Accuracy (simulated)", live)
        if live<0.82: st.warning("Drift detected → retrain & shadow deploy recommended.")
        else: st.success("No significant drift.")

# -------- Tab 5: Leadership, Best Practices, Cloud mapping --------
with T5:
    st.subheader("JD → Demo Mapping & Best Practices")
    st.markdown("""
- **Pre‑Sales & Client Engagement** → Tab 1 (RFP drafts, estimator)
- **Classical ML & Stats** → Tab 2 (classification/anomaly/forecast)
- **Deep Learning (lite)** → RF/CNN are hinted via pipelines; extend with tiny CNN if GPU allowed
- **Generative AI** → Tab 3 (LLM optional, safe fallback)
- **MLOps** → Tab 4 (registry, prod, drift)
- **Leadership & Communication** → This page + clean UX + exec outputs
- **Cloud & DevOps** → Estimator cloud selector, plus simple CI suggestion in README
""")
    st.markdown("""
**Best Practices Checklist**
- Reproducible pipelines; data contracts; PII controls
- Baselines → Ensembles → (optional) Transformers
- CI/CD for models; version every artifact; lineage
- Canary/shadow deploy; monitor drift/latency; rollback
- Clear executive comms; define KPIs, guardrails, acceptance
""")
    st.markdown("""
**Reference Cloud Architecture (conceptual)**
```
Client Apps → API → Feature Store/Batch → Model Service → Registry/Monitoring → Data Lake
          (AWS/Azure/GCP equivalents; secrets in cloud vault; infra as code)
```
""")

st.caption("Free‑tier friendly • single file • optional LLM • no external datasets required")
