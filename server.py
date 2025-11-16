                                                                                                                                                                              Now check for the final time is everything all right with the two files                                                                   # server.pyfrom fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_core import predict_shelf_life_from_input

app = FastAPI(title="Shelf Life Prediction API (minimal)")

class PredictRequest(BaseModel):
    dish_name: str
    temperature: float
    humidity: float
    storage: str

@app.post("/predict")
def predict(req: PredictRequest):
    result = predict_shelf_life_from_input(
        req.dish_name,
        req.temperature,
        req.humidity,
        req.storage
    )

    # If model_core returns an error dict, convert to HTTP 400
    if isinstance(result, dict) and result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/")
def root():
    return {"message": "Shelf Life Prediction API is running."}
                       
