import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('energy_model.pkl')  # This is your XGBoost model
scaler = joblib.load('scaler.pkl')

st.title("Steel Industry Energy Consumption Prediction")

# Input fields
shift = st.selectbox("Shift", [0, 1])
furnace_temp = st.number_input("Furnace Temperature")
load = st.number_input("Load")
gas = st.number_input("Gas Consumption")

# Add other features here if needed
# Example:
# other_feature = st.number_input("Other Feature")

if st.button("Predict"):
    input_df = pd.DataFrame([[shift, furnace_temp, load, gas]],
                             columns=['Shift', 'Furnace_Temp', 'Load', 'Gas'])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f} MJ")