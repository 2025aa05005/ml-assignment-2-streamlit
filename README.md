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

ML Model	                Accuracy	AUC	      Precision	  Recall	    F1	        MCC
Logistic Regression	      0.809756	0.92981	  0.761905	  0.914286	  0.831169	  0.630908
Decision Tree	            0.985366	0.985714	1	          0.971429	  0.985507	  0.971151    
KNN	                      0.863415	0.962905	0.873786	  0.857143	  0.865385	  0.726935
Naive Bayes	              0.829268	0.904286	0.807018	  0.87619	    0.840183	  0.660163
Random Forest (Ensemble)	1	        1	        1	          1	          1	          1
XGBoost (Ensemble)	      1	        1         1           1	          1	          1


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
