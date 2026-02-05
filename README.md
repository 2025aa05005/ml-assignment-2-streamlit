# ML Assignment 2 – Credit Risk Classification

## a. Problem Statement
Build and evaluate multiple machine learning classification models and deploy them using Streamlit.

## b. Dataset Description
We use the German Credit Risk Dataset. It contains customer financial and personal attributes to classify credit risk.

- Instances: 1000+
- Features: 20+
- Target: Risk (Good / Bad)

## c. Models Used

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | ... | ... | ... | ... | ... | ... |
| Decision Tree | ... | ... | ... | ... | ... | ... |
| KNN | ... | ... | ... | ... | ... | ... |
| Naive Bayes | ... | ... | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... | ... | ... |
| XGBoost | ... | ... | ... | ... | ... | ... |

### Observations

| Model | Observation |
|------|------------|
| Logistic Regression | Stable baseline performance |
| Decision Tree | Overfits slightly |
| KNN | Sensitive to scaling |
| Naive Bayes | Fast but simplistic |
| Random Forest | Strong ensemble performance |
| XGBoost | Best performance overall |

## Deployment
The app is deployed on Streamlit Cloud.

Features:
- CSV Upload
- Model Selection
- Evaluation Metrics
- Confusion Matrix
- CSV Download
