import joblib
import numpy as np

# ============================
# Load Saved Model & Encoders
# ============================

MODEL_PATH = "xgb_shelf_life_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"

# Load model
xgb_model = joblib.load(MODEL_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load encoders
encoders = joblib.load(ENCODERS_PATH)
le_dish = encoders["dish"]
le_storage = encoders["storage"]
le_type = encoders["type"]


# ============================
# Feature Engineering Helpers
# ============================

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

    if "paneer" in dish:
        return 6
    if any(x in dish for x in ["chicken", "fish", "mutton"]):
        return 6
    if "rice" in dish:
        return 8
    if "aloo" in dish or "potato" in dish:
        return 10
    if any(x in dish for x in ["roti", "naan", "chapati", "bread"]):
        return 12
    if any(x in dish for x in ["laddu", "halwa", "jamun", "rasgulla", "cake", "sweet"]):
        return 72
    if "ice_cream" in dish:
        return 2
    if any(x in dish for x in ["fried", "samosa", "kachori", "pakoda", "tikki"]):
        return 24

    return 12


def is_nonveg(dish):
    dish = str(dish).lower()
    return 1 if any(x in dish for x in ["chicken", "fish", "mutton", "egg", "maach"]) else 0


# ============================
# Prediction Adjustment Logic
# ============================

def adjust_prediction(pred_hours, base_life, storage, temp_c, humidity, dish_type):
    pred = max(pred_hours, 0.01)  # avoid negative predictions

    # Dish type multipliers
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

    # Storage effect
    s = storage.lower()
    if "open" in s:
        pred *= 0.3 if temp_c >= 30 else 0.5 if temp_c >= 25 else 0.7
    elif "airtight" in s or "sealed" in s:
        pred *= 0.8 if temp_c >= 30 else 0.95
    elif "refrig" in s or "fridge" in s or "cold" in s:
        pred *= 1.5 if temp_c <= 4 else 1.2

    # Humidity effect
    if humidity >= 80:
        pred *= 0.75
    elif humidity >= 60:
        pred *= 0.9

    # Hard sanity limits
    pred = max(pred, base_life * 0.3)  # 30% min
    pred = min(pred, base_life * 3.5)  # 3.5x max

    return float(np.round(pred, 2))


# ============================
# Format output
# ============================

def format_hours_minutes(pred_hours):
    hours = int(pred_hours)
    minutes = int(round((pred_hours - hours) * 60))

    if minutes >= 60:
        hours += 1
        minutes -= 60

    return hours, minutes


# ============================
# Prediction Function
# ============================

def predict_shelf_life_from_input(dish_name, temperature, humidity, storage):
    """Returns raw prediction + adjusted prediction + formatted output."""

    dish_type = get_dish_type(dish_name)
    base_life = get_base_shelf_life(dish_name)
    nonveg_flag = is_nonveg(dish_name)
    temp_x_hum = temperature * humidity

    # Encode categorical values
    dish_encoded = le_dish.transform([dish_name])[0]
    storage_encoded = le_storage.transform([storage])[0]
    dish_type_encoded = le_type.transform([dish_type])[0]

    # Create feature row
    X_input = np.array([[dish_encoded, storage_encoded, temperature, humidity,
                         dish_type_encoded, base_life, nonveg_flag, temp_x_hum]])

    # Scale features
    X_scaled = scaler.transform(X_input)

    # Predict
    raw_pred = float(xgb_model.predict(X_scaled)[0])

    # Adjust
    adjusted_pred = adjust_prediction(
        raw_pred, base_life, storage, temperature, humidity, dish_type
    )

    hours, minutes = format_hours_minutes(adjusted_pred)

    return {
        "raw_hours": round(raw_pred, 2),
        "adjusted_hours": adjusted_pred,
        "formatted": f"{hours} hours {minutes} minutes"
    }
