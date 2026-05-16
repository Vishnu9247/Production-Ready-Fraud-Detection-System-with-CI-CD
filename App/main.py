from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd
app = FastAPI()

def load_model():
    current_folder = Path(".")
    pkl_file = next(current_folder.glob("*.pkl"))
    model = joblib.load(pkl_file)
    return model

class Transaction(BaseModel):
    amt: float
    lat: float
    long: float
    city_pop: int
    merch_lat: float
    merch_long: float
    hour: int
    day_of_week: int
    is_weekend: int
    is_late_night: int
    log_amt: float
    amt_zscore: float
    is_high_amt: int
    distance_from_home: float
    log_distance: float
    customer_age: int
    age_group_encoded: int
    category_encoded: float
    city_encoded: float
    state_encoded: float
    merchant_encoded: float
    job_encoded: float


@app.post("/predict")
def predict(transaction: Transaction):
    input_df = pd.DataFrame([transaction.model_dump()])

    model = load_model()
    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        fraud_probability = model.predict_proba(input_df)[0][1]
    else:
        fraud_probability = None

    return {
        "prediction": int(prediction),
        "fraud_probability": float(fraud_probability) if fraud_probability is not None else None
    }