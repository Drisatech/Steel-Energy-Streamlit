import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('energy_model.pkl')
scaler = joblib.load('scaler.pkl')

# Page config
st.set_page_config(page_title="Steel Energy Prediction", layout="centered")

# App title
st.markdown("<h1 style='text-align: center; color: steelblue;'>Steel Energy Consumption Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Input Parameters")

    col1, col2 = st.columns(2)

    with col1:
        lagging = st.number_input('Lagging Reactive Power (kVarh)', value=0.0, format="%.2f")
        leading = st.number_input('Leading Reactive Power (kVarh)', value=0.0, format="%.2f")
        lag_pf = st.number_input('Lagging Power Factor', value=0.0, format="%.2f")

    with col2:
        lead_pf = st.number_input('Leading Power Factor', value=0.0, format="%.2f")
        hour = st.slider('Hour of Day', 0, 24, 0)
        day = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

    submitted = st.form_submit_button("Predict")

# One-hot encode the day of week
day_cols = ['Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday',
            'Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday',
            'Day_of_week_Wednesday']
day_values = [1 if day == col.split('_')[-1] else 0 for col in day_cols]

# Form the input data
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
input_df = input_df.reindex(columns=scaler.feature_names_in_)

# Prediction
if submitted:
    try:
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        st.success(f"**Energy Consumption (Usage_KW) = {prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
