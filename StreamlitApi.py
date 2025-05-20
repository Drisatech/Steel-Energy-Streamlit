import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('energy_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Steel Industry Energy Consumption Prediction")

# Inputs
lag_kvarh = st.number_input("Lagging Current Reactive Power (kVarh)")
lead_kvarh = st.number_input("Leading Current Reactive Power (kVarh)")
lag_pf = st.number_input("Lagging Current Power Factor")
lead_pf = st.number_input("Leading Current Power Factor")
hour = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

# Prepare input
day_cols = ['Day_of_week_Monday', 'Day_of_week_Tuesday', 'Day_of_week_Wednesday',
            'Day_of_week_Thursday', 'Day_of_week_Friday', 'Day_of_week_Saturday']

day_values = [1 if day == f.split('_')[-1] else 0 for f in day_cols]

input_data = [[lag_kvarh, lead_kvarh, lag_pf, lead_pf, hour] + day_values]
columns = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
           'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'Hour'] + day_cols
input_df = pd.DataFrame(input_data, columns=columns)

# Predict
if st.button("Predict"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Energy Consumption: {prediction[0]:.2f} kW")