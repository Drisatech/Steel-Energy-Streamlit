# Install necessary libraries
!pip install fastapi uvicorn joblib pydantic

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load('energy_model.pkl')
scaler = joblib.load('scaler.pkl')

class EnergyInput(BaseModel):
    NSM: float
        Lagging_Current: float
            CO2: float
                # ... all other features (You will need to explicitly list all feature names here based on X.columns)



@app.post("/predict")
async def predict(data: EnergyInput):
    # Ensure the input data matches the order of columns used during training
    # It's highly recommended to load the feature names saved earlier
    # For now, we'll assume a specific order, but this is error-prone
    feature_order = ['NSM', 'Lagging_Current', 'CO2'] # This needs to be dynamically generated from feature_names.txt or X.columns
    input_values = [data.dict().get(feature, 0.0) for feature in feature_order] # Get values based on the order

    # For categorical features converted by get_dummies, you'll need to handle them specifically.
    # A better approach is to load the feature_names.txt and build the input array dynamically.
    # Example (assuming feature_names.txt contains comma-separated column names):
    # with open('feature_names.txt', 'r') as f:
    #     feature_names = f.read().split(',')
    # input_data_dict = data.dict()
    # ordered_input = [input_data_dict.get(name, 0.0) for name in feature_names]
    # input_data = np.array([ordered_input])

input_data = np.array([input_values]) # This line needs refinement based on all features

    scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
            return {"prediction": prediction[0]} 