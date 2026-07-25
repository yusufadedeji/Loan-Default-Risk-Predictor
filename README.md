# Loan Default Risk Prediction Using Machine Learning

## Overview

This project develops and evaluates machine learning models for predicting loan default risk using borrower demographic, financial, and credit history information. The objective is to identify applicants who are more likely to default on their loans, enabling lenders to make informed credit decisions and minimize financial risk.

Five supervised machine learning algorithms were trained and compared, with XGBoost emerging as the best-performing model. The final model was further used to identify the most influential factors associated with loan default, generate borrower risk scores, and evaluate prediction performance using a confusion matrix.

---

## Problem Statement

Loan default remains one of the major challenges faced by financial institutions, particularly microfinance banks, fintech companies, and digital lenders. Traditional credit assessment methods are often time-consuming and may not effectively capture complex relationships within borrower data.

This project demonstrates how machine learning can improve loan approval decisions by accurately estimating the probability of borrower default.

---

## Objectives

The objectives of this project were to:

- Develop machine learning models for loan default prediction.
- Compare the performance of multiple classification algorithms.
- Identify the most important factors influencing loan default.
- Generate risk scores for individual borrowers.
- Recommend a suitable model for real-world credit risk assessment.

---

## Dataset

The project uses a publicly available loan dataset containing borrower demographic, financial, employment, and credit history information.

### Features include:

- Person Age
- Annual Income
- Employment Length
- Home Ownership
- Loan Amount
- Loan Intent
- Loan Grade
- Interest Rate
- Loan-to-Income Ratio
- Previous Credit Default
- Credit History Length

**Target Variable**

- `loan_status`
  - 0 = Non-default
  - 1 = Default

---

## Project Workflow

The project followed the standard machine learning pipeline:

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
   - One-Hot Encoding
   - Feature Scaling
5. Train-Test Split (70:30)
6. Model Training
7. Model Evaluation
8. Feature Importance Analysis
9. Risk Score Prediction
10. Confusion Matrix Visualization

---

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost

---

## Model Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification Report
- Confusion Matrix

---

## Best Performing Model

Among all models evaluated, **XGBoost** achieved the best overall performance and was selected as the final prediction model.

The model provides:

- High prediction accuracy
- Strong precision for identifying loan defaults
- Borrower risk scores (probability of default)
- Feature importance for model interpretability

---

## Key Outputs

### Feature Importance

The project identifies the variables that contribute most to loan default prediction using XGBoost feature importance.

Examples include:

- Loan Grade
- Interest Rate
- Loan-to-Income Ratio
- Annual Income
- Previous Credit Default
- Employment Length

---

### Risk Score

The model generates a probability score representing the likelihood that a borrower will default.

Example:

| Applicant | Risk Score | Interpretation |
|------------|-----------:|---------------|
| A | 0.08 | Low Risk |
| B | 0.74 | High Risk |

---

### Confusion Matrix

A confusion matrix was generated to evaluate the model's classification performance and illustrate correctly and incorrectly classified borrowers.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

## Repository Structure

```
├── Loan Default Risk Predictor_3MTT Capstone Project.ipynb
├── credit_risk_dataset.csv
├── 3MTT Capstone Project.pdf
├── README.md
```

---

## Applications

The developed model can be used by:

- Microfinance Banks
- Digital Lending Platforms
- FinTech Companies
- POS Agents offering credit facilities
- Cooperative Societies
- Credit Risk Analysts

It can support faster and more consistent credit decisions while helping lenders reduce loan default rates.

---

## Future Improvements

Potential improvements include:

- Hyperparameter optimization
- Cross-validation
- Deployment as a web application using Streamlit
- Integration of alternative credit data such as mobile transactions and repayment history
- Continuous model retraining using new loan records

---

## Author

**Adedeji, Yusuf Oluwapelumi**

---

## License

This project is intended for educational and research purposes.
