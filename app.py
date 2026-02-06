import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

st.set_page_config(page_title="ML Assignment 2 Dashboard", layout="wide")

st.title("🔬 ML Assignment 2 – Model Evaluation Dashboard")

# -----------------------------
# Load Metrics CSV
# -----------------------------
@st.cache_data
def load_metrics():
    return pd.read_csv("metrics_results.csv")

metrics_df = load_metrics()

# -----------------------------
# Load Models
# -----------------------------
MODEL_PATHS = {
    "logistic": "model/saved_models/logistic.pkl",
    "decision_tree": "model/saved_models/decision_tree.pkl",
    "knn": "model/saved_models/knn.pkl",
    "naive_bayes": "model/saved_models/naive_bayes.pkl",
    "random_forest": "model/saved_models/random_forest.pkl",
    "xgboost": "model/saved_models/xgboost.pkl",
}

# -----------------------------
# UI Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 ROC Curves", "🔍 Single Model Evaluation"])

# =============================
# TAB 1 – Model Comparison
# =============================
with tab1:
    st.subheader("📊 Metrics Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("📊 Bar Charts (Accuracy, AUC, F1)")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    sns.barplot(x="Model", y="Accuracy", data=metrics_df, ax=ax[0])
    sns.barplot(x="Model", y="AUC", data=metrics_df, ax=ax[1])
    sns.barplot(x="Model", y="F1", data=metrics_df, ax=ax[2])

    ax[0].set_title("Accuracy")
    ax[1].set_title("AUC")
    ax[2].set_title("F1 Score")

    for a in ax:
        a.set_xticklabels(a.get_xticklabels(), rotation=45)

    st.pyplot(fig)

# =============================
# TAB 2 – ROC Curves
# =============================
with tab2:
    st.subheader("📈 ROC Curves (Upload Test Data)")

    uploaded = st.file_uploader("Upload CSV with target column", type="csv", key="roc")

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

# =============================
# TAB 3 – Single Model Evaluation
# =============================
with tab3:
    st.subheader("🔍 Evaluate a Single Model")

    model_name = st.selectbox("Select Model", list(MODEL_PATHS.keys()))
    uploaded2 = st.file_uploader("Upload Test CSV with target column", type="csv", key="single")

    if uploaded2:
        df = pd.read_csv(uploaded2)
        X = df.drop("target", axis=1)
        y = df["target"]

        model = joblib.load(MODEL_PATHS[model_name])
        y_pred = model.predict(X)

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots(figsize=(4,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix – {model_name}")

        st.pyplot(fig)

        st.subheader("📄 Metrics from CSV")
        st.write(metrics_df[metrics_df["Model"] == model_name])
