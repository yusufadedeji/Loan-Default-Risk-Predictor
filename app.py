
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load('Loan_Default_XGBoost.pkl')
feature_encoder = joblib.load('OneHotEncoder.pkl')
scaler = joblib.load('MinMaxScaler.pkl')
feature_names = joblib.load('Feature_Names.pkl')

st.title("Loan Default Risk Predictor")

st.write("Enter the borrower's information to estimate the risk of loan default.")

# User Inputs

person_age = st.number_input("Age", min_value=18, max_value=100, value=30)

person_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "MORTGAGE", "OWN", "OTHER"]
)

person_emp_length = st.number_input(
    "Employment Length (years)",
    min_value=0.0,
    max_value=60.0,
    value=5.0
)

loan_intent = st.selectbox(
    "Loan Intent",
    ["PERSONAL", "EDUCATION", "MEDICAL",
     "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

loan_int_rate = st.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.20
)

cb_person_default_on_file = st.selectbox(
    "Previous Default on File",
    ["Y", "N"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0.0,
    value=5.0
)


if st.button("Predict Risk"):

    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_income': [person_income],
        'person_home_ownership': [person_home_ownership],
        'person_emp_length': [person_emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [cb_person_default_on_file],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length]
    })

    # Separate categorical and numerical features

    cat_data = input_data[
        ['person_home_ownership',
         'loan_intent',
         'loan_grade',
         'cb_person_default_on_file']
    ]

    num_data = input_data[
        ['person_age',
         'person_income',
         'person_emp_length',
         'loan_amnt',
         'loan_int_rate',
         'loan_percent_income',
         'cb_person_cred_hist_length']
    ]

    # Encode categorical variables

    encoded_data = pd.DataFrame(
        feature_encoder.transform(cat_data),
        columns=feature_encoder.get_feature_names_out(cat_data.columns)
    )

    # Scale numerical variables

    scaled_data = pd.DataFrame(
        scaler.transform(num_data),
        columns=num_data.columns
    )

    # Combine the processed data

    processed_data = pd.concat(
        [scaled_data, encoded_data],
        axis=1
    )

    # Ensure correct feature order

    processed_data = processed_data[feature_names]

    # Predict risk

    risk_score = model.predict_proba(processed_data)[0, 1] * 100

    prediction = model.predict(processed_data)[0]

    # Display result

    st.subheader("Loan Default Risk")

    st.metric(
        "Estimated Risk of Default",
        f"{risk_score:.2f}%"
    )

    if prediction == 1:
        st.error("Higher predicted risk of default")
    else:
        st.success("Lower predicted risk of default")