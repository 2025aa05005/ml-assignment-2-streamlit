import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

st.set_page_config(page_title="ML Assignment 2 Dashboard", layout="wide")

st.markdown("""
<style>
body {background-color: #f6f8fa;}
h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

st.title("🔬 ML Assignment 2 – Classification Models Dashboard")

@st.cache_data
def load_metrics():
    return pd.read_csv("metrics_results.csv")

metrics_df = load_metrics()

MODEL_PATHS = {
    "logistic": "model/saved_models/logistic.pkl",
    "decision_tree": "model/saved_models/decision_tree.pkl",
    "knn": "model/saved_models/knn.pkl",
    "naive_bayes": "model/saved_models/naive_bayes.pkl",
    "random_forest": "model/saved_models/random_forest.pkl",
    "xgboost": "model/saved_models/xgboost.pkl",
}

# -----------------------------
# Dataset Download Section
# -----------------------------
st.subheader("📥 Download Dataset")

with open("data/heart.csv", "rb") as f:
    st.download_button(
        label="⬇ Download Heart Disease Dataset (heart.csv)",
        data=f,
        file_name="heart.csv",
        mime="text/csv"
    )


tab2, tab3, tab4, tab1 = st.tabs(["📈 ROC Curves", "🔍 Confusion Matrix", "📄 Classification Report", "📊 Comparison"])

# ---- TAB 1 ----
with tab1:
    st.subheader("📊 Model Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        "⬇ Download Metrics CSV",
        data=metrics_df.to_csv(index=False),
        file_name="metrics_results.csv",
        mime="text/csv"
    )

    fig, ax = plt.subplots(1, 3, figsize=(18,5))
    sns.barplot(x="Model", y="Accuracy", data=metrics_df, ax=ax[0])
    sns.barplot(x="Model", y="AUC", data=metrics_df, ax=ax[1])
    sns.barplot(x="Model", y="F1", data=metrics_df, ax=ax[2])
    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=45)
    ax[0].set_title("Accuracy")
    ax[1].set_title("AUC")
    ax[2].set_title("F1 Score")
    st.pyplot(fig)

# ---- TAB 2 ----
with tab2:
    uploaded = st.file_uploader("Upload Test CSV (with target column)", type="csv", key="roc")
    if uploaded:
        df = pd.read_csv(uploaded)
        X = df.drop("target", axis=1)
        y = df["target"]

        fig, ax = plt.subplots(figsize=(8,6))
        for name, path in MODEL_PATHS.items():
            model = joblib.load(path)
            y_prob = model.predict_proba(X)[:,1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves Comparison")
        ax.legend()
        st.pyplot(fig)

# ---- TAB 3 ----
with tab3:
    model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
    uploaded2 = st.file_uploader("Upload Test CSV", type="csv", key="cm")
    if uploaded2:
        df = pd.read_csv(uploaded2)
        X = df.drop("target", axis=1)
        y = df["target"]

        model = joblib.load(MODEL_PATHS[model_name])
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"Confusion Matrix – {model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# ---- TAB 4 ----
with tab4:
    model_name = st.selectbox("Select Model for Report", list(MODEL_PATHS.keys()), key="report")
    uploaded3 = st.file_uploader("Upload Test CSV", type="csv", key="report_csv")
    if uploaded3:
        df = pd.read_csv(uploaded3)
        X = df.drop("target", axis=1)
        y = df["target"]

        model = joblib.load(MODEL_PATHS[model_name])
        y_pred = model.predict(X)

        report = classification_report(y, y_pred, output_dict=True)
        st.json(report)
