import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("energy_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Steel Energy Consumption Predictor")

# User Inputs
lagging_reactive = st.number_input("Lagging Current Reactive Power (kVarh)", min_value=0.0)
leading_reactive = st.number_input("Leading Current Reactive Power (kVarh)", min_value=0.0)
lagging_pf = st.number_input("Lagging Current Power Factor", min_value=0.0, max_value=1.0)
leading_pf = st.number_input("Leading Current Power Factor", min_value=0.0, max_value=1.0)
hour = st.slider("Hour of Day (0-24)", min_value=0, max_value=23)

day = st.selectbox("Select Day of the Week", [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])

# One-hot encoding for day
day_cols = ['Day_of_week_Friday', 'Day_of_week_Monday', 'Day_of_week_Saturday',
            'Day_of_week_Sunday', 'Day_of_week_Thursday', 'Day_of_week_Tuesday',
            'Day_of_week_Wednesday']

day_values = [1 if day == f.split('_')[-1] else 0 for f in day_cols]

# Combine all inputs
input_data = [lagging_reactive, leading_reactive, lagging_pf, leading_pf, hour] + day_values

# Create DataFrame with correct feature names
feature_names = ['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh',
                 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'Hour'] + day_cols

input_df = pd.DataFrame([input_data], columns=feature_names)

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)

st.subheader("Predicted Energy Usage (kW):")
st.success(f"{prediction[0]:.2f}")