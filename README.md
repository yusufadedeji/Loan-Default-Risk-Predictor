# Loan Default Risk Prediction Using Machine Learning

## Overview

This project develops a machine learning system for predicting loan default risk using borrowers' demographic, financial, employment, and credit history information. Five classification models were developed and compared, with XGBoost selected as the final model and deployed as an interactive Streamlit application that generates borrower risk scores and default predictions.

## Project Objectives

The main objectives of this project were to:

- Predict the likelihood of loan default using machine learning.
- Compare the performance of different classification algorithms.
- Identify the key factors associated with loan default.
- Generate an individual risk score for borrowers.
- Develop a simple interactive application for practical use.
- Demonstrate how machine learning can support credit risk assessment.

## Dataset

The project uses a loan dataset containing demographic, financial, employment, and credit history information.

### Main Features

- Age
- Annual Income
- Home Ownership
- Employment Length
- Loan Intent
- Loan Grade
- Loan Amount
- Interest Rate
- Loan-to-Income Ratio
- Previous Credit Default
- Credit History Length

### Target Variable

`loan_status`

- `0` = Non-default
- `1` = Default

## Machine Learning Workflow

The project followed the following workflow:

1. Data loading and exploration
2. Data cleaning
3. Exploratory data analysis
4. Feature preprocessing
5. One-hot encoding of categorical variables
6. Min-Max scaling of numerical variables
7. Train-test split (70:30)
8. Model development and comparison
9. Model evaluation
10. XGBoost feature importance analysis
11. Borrower risk score generation
12. Interactive application development
13. Local testing and deployment preparation

## Models Developed

Five classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

XGBoost was selected as the final model based on its overall performance in the model comparison.

## Key Factors

XGBoost feature importance was used to identify the variables that contributed most to loan default prediction.

The analysis provides insight into important borrower characteristics such as:

- Loan grade
- Interest rate
- Loan-to-income ratio
- Annual income
- Previous credit default
- Employment characteristics

These factors can help lenders better understand the characteristics associated with higher default risk.

## Risk Score

The final XGBoost model generates a probability-based risk score representing the estimated likelihood of loan default.

For example:

```text
Risk Score: 18.45%
Prediction: Lower predicted risk of default
```

The risk score provides more information than a simple default/non-default classification because it allows borrowers to be ranked according to their estimated level of risk.

## Interactive Loan Default Risk Predictor

The machine learning model was developed into an interactive web application using Streamlit.

The application allows users to enter borrower information and receive:

- Estimated risk of default
- Default risk percentage
- Higher or lower predicted risk classification

### Application Workflow

```text
Borrower Information
        ↓
Data Preprocessing
        ↓
One-Hot Encoding
        ↓
Min-Max Scaling
        ↓
XGBoost Model
        ↓
Risk Score
        ↓
Default Risk Prediction
```

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Jupyter Notebook
- Streamlit
- Joblib
- GitHub

## Project Structure

```text
Loan-Default-Risk-Predictor/
│
├── app.py
├── requirements.txt
├── Loan_Default_XGBoost.pkl
├── OneHotEncoder.pkl
├── MinMaxScaler.pkl
└── Feature_Names.pkl
└── README.md
```

## Real-World Applications

The model can serve as a credit risk decision-support tool for:

- Microfinance banks
- Digital lending platforms
- FinTech companies
- POS agents offering credit
- Cooperative societies
- Small lending institutions

It can help lenders assess borrower risk more consistently and support faster lending decisions.

The model should be used as a decision-support tool rather than as a fully automated replacement for human credit assessment.

## Future Improvements

Future development could include:

- Hyperparameter tuning
- Cross-validation
- Model calibration
- Integration of additional credit and repayment information
- Integration of alternative financial data
- Continuous model retraining with new loan data
- Further testing of the model on external datasets
- Improved risk categories for lending decisions
- Full cloud deployment of the Streamlit application

## Deployment

The application was developed using Streamlit and tested locally before deployment.

The trained XGBoost model, encoder, scaler, and feature names are saved using Joblib so that the application uses the same preprocessing and model developed during training.

## Conclusion

This project demonstrates the use of machine learning to predict loan default risk and support data-driven credit assessment. The combination of XGBoost, feature importance, risk scoring, and an interactive application provides a practical approach for identifying and assessing borrowers at different levels of default risk.

## Author

**Yusuf Oluwapelumi Adedeji**

Data Science & Machine Learning Engineer

## Disclaimer

This project is developed for educational and demonstration purposes. The model should not be used as the sole basis for real-world lending decisions without further validation, monitoring, fairness assessment, and testing with appropriate real-world financial data.
