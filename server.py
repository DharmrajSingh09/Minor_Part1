from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Import all feature utility functions from model_core.py
from model_core import (
    get_dish_type,
    get_base_shelf_life,
    is_nonveg,
    adjust_prediction,
    format_hours_minutes
)

# ------------------------------
# Load Model + Preprocessing
# ------------------------------

MODEL_FILE = "xgb_shelf_life_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"

xgb_model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
encoders = joblib.load(ENCODERS_FILE)

# label encoders inside encoders.pkl
le_dish = encoders["dish"]
le_storage = encoders["storage"]
le_type = encoders["type"]

# ------------------------------
# FastAPI App
# ------------------------------

app = FastAPI(title="Shelf Life Prediction API")

class PredictRequest(BaseModel):
    dish_name: str
    temperature: float
    humidity: float
    storage: str


# ------------------------------
# Prediction Route
# ------------------------------

@app.post("/predict")
def predict_shelf_life(data: PredictRequest):

    dish = data.dish_name
    temp = data.temperature
    humidity = data.humidity
    storage = data.storage

    # --------------------------
    # Feature Engineering
    # --------------------------
    dish_type = get_dish_type(dish)
    base_life = get_base_shelf_life(dish)
    nonveg_flag = is_nonveg(dish)
    temp_humidity = temp * humidity

    # --------------------------
    # Encoding
    # --------------------------
    try:
        dish_encoded = int(le_dish.transform([dish])[0])
    except:
        return {"error": f"Dish '{dish}' not found in training data."}

    try:
        storage_encoded = int(le_storage.transform([storage])[0])
    except:
        return {"error": f"Storage type '{storage}' not found."}

    try:
        dish_type_encoded = int(le_type.transform([dish_type])[0])
    except:
        return {"error": f"Dish type '{dish_type}' not recognized."}

    # Final input vector
    X = np.array([[
        dish_encoded,
        storage_encoded,
        temp,
        humidity,
        dish_type_encoded,
        base_life,
        nonveg_flag,
        temp_humidity
    ]])

    # Scale
    X_scaled = scaler.transform(X)

    # Raw model output
    raw_pred = float(xgb_model.predict(X_scaled)[0])

    # Adjusted output
    adj_pred = adjust_prediction(
        raw_pred,
        base_life,
        storage,
        temp,
        humidity,
        dish_type
    )

    hours, minutes = format_hours_minutes(adj_pred)

    return {
        "dish": dish,
        "storage": storage,
        "input_temperature": temp,
        "input_humidity": humidity,
        "raw_prediction_hours": round(raw_pred, 2),
        "adjusted_prediction_hours": adj_pred,
        "adjusted_prediction_formatted": f"{hours} hours {minutes} minutes"
    }


# ------------------------------
# Root endpoint
# ------------------------------

@app.get("/")
def root():
    return {"message": "Shelf Life Prediction API is running!"}
