# -----------------------------
# FastAPI app for shelf life prediction
# -----------------------------
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Config / paths
# -----------------------------
MODEL_PATH = "xgb_shelf_life_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"

# -----------------------------
# Load artifacts
# -----------------------------
xgb_model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
le_dish, le_storage, le_type = encoders["dish"], encoders["storage"], encoders["type"]

# -----------------------------
# Feature helpers
# -----------------------------
def get_dish_type(dish):
    dish = str(dish).lower()
    if any(x in dish for x in ["chicken", "fish", "mutton", "egg", "maach"]):
        return "NonVegCurry"
    elif any(x in dish for x in ["paneer", "dal", "curry", "korma", "butter"]):
        return "VegCurry"
    elif any(x in dish for x in ["roti", "naan", "chapati", "paratha", "puri"]):
        return "Bread"
    elif any(x in dish for x in ["biryani", "rice", "pulao", "khichdi"]):
        return "Rice"
    elif any(x in dish for x in ["laddu", "halwa", "jamun", "rasgulla", "cake", "sweet"]):
        return "Sweet"
    elif any(x in dish for x in ["lassi", "rabri", "milk", "cream", "ice_cream", "cheese"]):
        return "Dairy"
    elif any(x in dish for x in ["samosa", "kachori", "pakoda", "tikki", "fried"]):
        return "FriedSnack"
    else:
        return "Other"

def get_base_shelf_life(dish):
    dish = str(dish).lower()
    if "paneer" in dish: return 6
    if any(x in dish for x in ["chicken", "fish", "mutton"]): return 6
    if "rice" in dish: return 8
    if "aloo" in dish or "potato" in dish: return 10
    if any(x in dish for x in ["roti", "naan", "chapati", "bread"]): return 12
    if any(x in dish for x in ["laddu", "halwa", "jamun", "rasgulla", "cake", "sweet"]): return 3  # realistic open storage
    if "ice_cream" in dish: return 2
    if any(x in dish for x in ["fried", "samosa", "kachori", "pakoda", "tikki"]): return 24
    return 12

def is_nonveg(dish):
    dish = str(dish).lower()
    return 1 if any(x in dish for x in ["chicken", "fish", "mutton", "egg", "maach"]) else 0

def adjust_prediction(pred_hours, base_life, storage, temp_c, humidity, dish_type):
    pred = max(pred_hours, 0.01)
    type_multiplier = {
        "FriedSnack": 0.7,
        "NonVegCurry": 0.6,
        "VegCurry": 0.8,
        "Rice": 0.75,
        "Bread": 0.9,
        "Sweet": 1.0,
        "Dairy": 0.5,
        "Other": 0.85
    }
    pred *= type_multiplier.get(dish_type, 0.8)

    s = storage.lower()
    if "open" in s:
        pred *= 0.3 if temp_c >= 30 else 0.5 if temp_c >= 25 else 0.7
    elif "airtight" in s or "sealed" in s:
        pred *= 0.8 if temp_c >= 30 else 0.95
    elif "refrig" in s or "fridge" in s or "cold" in s:
        pred *= 1.5 if temp_c <= 4 else 1.2

    if humidity >= 80:
        pred *= 0.75
    elif humidity >= 60:
        pred *= 0.9

    pred = max(pred, base_life * 0.3)
    pred = min(pred, base_life * 3.5)

    return float(np.round(pred, 2))

def format_hours_minutes(pred_hours):
    hours = int(pred_hours)
    minutes = int(round((pred_hours - hours) * 60))
    if minutes >= 60:
        hours += 1
        minutes -= 60
    return hours, minutes

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="Shelf Life Prediction API")

class PredictionInput(BaseModel):
    dish: str
    storage: str
    temperature: float
    humidity: int

@app.post("/predict")
def predict(input: PredictionInput):
    try:
        dish_type = get_dish_type(input.dish)
        base_life = get_base_shelf_life(input.dish)
        nonveg_flag = is_nonveg(input.dish)
        temp_x_hum = input.temperature * input.humidity

        dish_encoded = le_dish.transform([input.dish])[0]
        storage_encoded = le_storage.transform([input.storage])[0]
        dish_type_encoded = le_type.transform([dish_type])[0] if dish_type in le_type.classes_ else 0

        X_input = np.array([[dish_encoded, storage_encoded, input.temperature, input.humidity,
                             dish_type_encoded, base_life, nonveg_flag, temp_x_hum]])
        X_input_scaled = scaler.transform(X_input)

        pred_raw = float(xgb_model.predict(X_input_scaled)[0])
        pred_adj = adjust_prediction(pred_raw, base_life, input.storage, input.temperature, input.humidity, dish_type)
        hours, minutes = format_hours_minutes(pred_adj)

        return {
            "predicted_hours_raw": pred_raw,
            "predicted_hours_adjusted": pred_adj,
            "hours": hours,
            "minutes": minutes
        }
    except Exception as e:
        return {"error": str(e)}

