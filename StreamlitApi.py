import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('energy_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit inputs
day = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
lagging = st.number_input('Lagging_Current_Reactive.Power_kVarh')
leading = st.number_input('Leading_Current_Reactive_Power_kVarh')
lag_pf = st.number_input('Lagging_Current_Power_Factor')
lead_pf = st.number_input('Leading_Current_Power_Factor')
hour = st.slider('Hour of Day', 0, 23)

# One-hot encode day
day_cols = ['Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday',
            'Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday',
            'Day_of_week_Wednesday']
day_values = [1 if day == col.split('_')[-1] else 0 for col in day_cols]

# Input DataFrame
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

# REINDEX TO MATCH TRAINING FEATURE ORDER
input_df = input_df.reindex(columns=scaler.feature_names_in_)

# Scale and Predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

st.write(f'Predicted Usage_Kw: {prediction[0]:.2f}')
