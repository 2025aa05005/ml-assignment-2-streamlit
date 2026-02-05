import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.title("ML Assignment 2 – Credit Risk Classification")

st.write("Upload a CSV file (test data only):")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

df = pd.read_csv("german_credit_data.csv")

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Risk", axis=1)
y = df["Risk"]

model_choice = st.selectbox("Select Model", [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

model = models[model_choice]
model.fit(X, y)

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    for col in test_df.columns:
        if test_df[col].dtype == 'object':
            test_df[col] = LabelEncoder().fit_transform(test_df[col])

    preds = model.predict(test_df)
    test_df["Prediction"] = preds

    st.write("Predictions:")
    st.dataframe(test_df)

    st.download_button(
        label="Download Predictions as CSV",
        data=test_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
