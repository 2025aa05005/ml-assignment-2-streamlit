📌 Problem Statement

The objective of this assignment is to implement, evaluate, and compare multiple machine learning classification models on a real-world dataset and deploy them using an interactive Streamlit web application.

The project demonstrates an end-to-end ML workflow: data handling, model training, evaluation, visualization, and deployment.

📊 Dataset Description

The dataset used is the Heart Disease Dataset (UCI Machine Learning Repository).
It contains 13 input features and 1 target variable:

Target Column: target

0 → No Heart Disease

1 → Heart Disease Present

The dataset includes attributes such as age, sex, chest pain type, blood pressure, cholesterol, ECG results, maximum heart rate, exercise-induced angina, etc.

🤖 Models Used & Evaluation Metrics

The following 6 classification models were implemented on the same dataset:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

🔢 Metrics Calculated

Accuracy

AUC Score

Precision

Recall

F1 Score

Matthews Correlation Coefficient (MCC)

📈 Model Comparison Table
ML Model	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.86	0.91	0.87	0.84	0.85	0.72
Decision Tree	0.82	0.85	0.83	0.81	0.82	0.65
KNN	0.84	0.88	0.85	0.83	0.84	0.68
Naive Bayes	0.83	0.86	0.82	0.84	0.83	0.66
Random Forest	0.89	0.94	0.90	0.88	0.89	0.78
XGBoost	0.91	0.96	0.92	0.90	0.91	0.81
🧠 Model Performance Observations
Model	Observation
Logistic Regression	Strong baseline, works well on linearly separable data
Decision Tree	Easy to interpret but prone to overfitting
KNN	Sensitive to feature scaling, moderate performance
Naive Bayes	Assumes feature independence, reasonable results
Random Forest	Robust ensemble model, reduces overfitting
XGBoost	Best performing model with highest AUC and MCC
🌐 Streamlit App Features

Dataset upload option (CSV)

Model selection dropdown

Display of evaluation metrics

Confusion matrix

ROC Curves

Model comparison dashboard

Metrics CSV download button

Classification report tab

🚀 Deployment

The application is deployed on Streamlit Community Cloud and provides an interactive frontend for evaluation.
