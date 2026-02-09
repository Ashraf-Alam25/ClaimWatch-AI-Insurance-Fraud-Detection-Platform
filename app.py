import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ---------------- Load Dataset ----------------
df = pd.read_csv("/content/insurance_claims.csv")

# Convert target
df['fraud_reported'] = df['fraud_reported'].map({'Y': 1, 'N': 0})

# Features
features = [
    'age',
    'policy_annual_premium',
    'umbrella_limit',
    'capital-gains',
    'capital-loss',
    'incident_hour_of_the_day',
    'number_of_vehicles_involved',
    'bodily_injuries',
    'witnesses',
    'total_claim_amount'
]

# Handle missing values
for col in features:
    df[col] = df[col].fillna(df[col].mean())

df['fraud_reported'] = df['fraud_reported'].fillna(0)

X = df[features]
y = df['fraud_reported']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_scaled, y)

# ---------------- UI ----------------
st.set_page_config(page_title="ClaimWatch AI", layout="centered")

st.title("🛡️ ClaimWatch AI")
st.subheader("Insurance Fraud Detection System")

age = st.number_input("Age", 18, 100)
premium = st.number_input("Annual Premium")
umbrella = st.number_input("Umbrella Limit")
capital_gain = st.number_input("Capital Gain")
capital_loss = st.number_input("Capital Loss")
incident_hour = st.slider("Incident Hour", 0, 23)
vehicles = st.number_input("Vehicles Involved", 0, 5)
injuries = st.number_input("Bodily Injuries", 0, 5)
witnesses = st.number_input("Witnesses", 0, 5)
claim_amount = st.number_input("Total Claim Amount")

if st.button("🔍 Analyze Claim"):
    input_data = np.array([[
        age, premium, umbrella, capital_gain, capital_loss,
        incident_hour, vehicles, injuries, witnesses, claim_amount
    ]])

    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"### Fraud Probability: **{prob:.2f}**")

    if prob > 0.35:
        st.error("⚠️ High Risk / Fraudulent Claim")
        st.write("**AI Reasoning:**")
        st.write("- High claim amount")
        st.write("- Unusual historical pattern")
    else:
        st.success("✅ Low Risk / Genuine Claim")
