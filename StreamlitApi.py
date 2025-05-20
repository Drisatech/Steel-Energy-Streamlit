import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('energy_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Steel Energy Consumption Prediction Web App")

# Input fields
day = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
lagging = st.number_input('Lagging_Current_Reactive.Power_kVarh')
leading = st.number_input('Leading_Current_Reactive_Power_kVarh')
lag_pf = st.number_input('Lagging_Current_Power_Factor')
lead_pf = st.number_input('Leading_Current_Power_Factor')
hour = st.slider('Hour of Day', 0, 24)

# One-hot encode the selected day
day_cols = ['Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday',
            'Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday',
            'Day_of_week_Wednesday']
day_values = [1 if day == col.split('_')[-1] else 0 for col in day_cols]

# Form input data
input_data = [[
    lagging,
    leading,
    lag_pf,
    lead_pf,
    hour,
    *day_values
]]
input_df = pd.DataFrame(input_data, columns=[
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'Hour',
    *day_cols
])

# Match input column order with what scaler expects
input_df = input_df.reindex(columns=scaler.feature_names_in_)

# Prediction trigger
if st.button("Predict"):
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"Energy Consumption (Usage_KW) = {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
