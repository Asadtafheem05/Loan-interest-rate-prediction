import streamlit as st
import pandas as pd
import joblib

# ---------------- Load Pickles ----------------
ohe = joblib.load("ohe.pkl")
sc = joblib.load("sc.pkl")
model = joblib.load("ridge.pkl")

# ---------------- Indian Currency Formatter ----------------
def format_indian_currency(amount):
    amount = float(amount)
    s = f"{amount:.2f}"
    integer, decimal = s.split(".")

    if len(integer) > 3:
        last3 = integer[-3:]
        rest = integer[:-3]
        rest = ",".join([rest[max(i-2, 0):i] for i in range(len(rest), 0, -2)][::-1])
        integer = rest + "," + last3

    return f"₹{integer}.{decimal}"

# ---------------- Streamlit UI ----------------
st.set_page_config(
    page_title="Loan Interest Rate Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Interest Rate Prediction System")
st.markdown("### Enter Loan Details")

# ---------------- User Inputs ----------------
loan_amount = st.number_input("Loan Amount (₹)", min_value=1000.0, step=1000.0)
income = st.number_input("Annual Income (₹)", min_value=1000.0, step=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
emp_length = st.number_input("Employment Length (years)", min_value=0, step=1)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=1)
dti = st.slider("Debt-to-Income Ratio (0-1)", 0.0, 1.0, 0.3)

home = st.selectbox(
    "Home Ownership",
    ["rent", "own", "mortgage"]
)

purpose = st.selectbox(
    "Purpose of Loan",
    ["personal", "car", "business", "home", "education"]
)

# ---------------- Prediction Button ----------------
if st.button("Predict Interest Rate"):

    row_model = pd.DataFrame({
        'loan amount': [loan_amount],
        'annual income': [income],
        'credit score': [credit_score],
        'employment length': [emp_length],
        'loan term': [loan_term],
        'debt-to-income ratio (dti)': [dti],
        'home ownership': [home],
        'purpose of loan': [purpose]
    })

    # -------- One Hot Encoding --------
    cat_cols = ['home ownership', 'purpose of loan']
    row_ohe = ohe.transform(row_model[cat_cols])

    if hasattr(row_ohe, "toarray"):
        row_ohe = row_ohe.toarray()

    ohe_cols = ohe.get_feature_names_out(cat_cols)
    row_ohe = pd.DataFrame(row_ohe, columns=ohe_cols)

    # -------- Combine --------
    row_final = pd.concat(
        [row_model.drop(cat_cols, axis=1), row_ohe],
        axis=1
    )

    # -------- Match Scaler --------
    scaler_cols = sc.feature_names_in_
    row_final = row_final.reindex(columns=scaler_cols, fill_value=0)

    # -------- Scaling --------
    row_scaled = sc.transform(row_final)
    row_scaled = pd.DataFrame(row_scaled, columns=scaler_cols)

    # -------- Match Model --------
    model_cols = model.feature_names_in_
    row_scaled = row_scaled.reindex(columns=model_cols, fill_value=0)

    # -------- Prediction --------
    prediction = model.predict(row_scaled)[0]

    st.success(f"Predicted Interest Rate: {round(prediction, 2)} %")

    # -------- Financial Calculation --------
    interest_amount = loan_amount * (prediction / 100)
    total_payment = loan_amount + interest_amount

    st.markdown("### 💰 Financial Summary")

    st.write("Loan Amount:", format_indian_currency(loan_amount))
    st.write("Interest Amount:", format_indian_currency(interest_amount))
    st.write("Total Repayment:", format_indian_currency(total_payment))

    st.info(
        f"You will repay {format_indian_currency(total_payment)} "
        f"including {format_indian_currency(interest_amount)} as interest."
    )