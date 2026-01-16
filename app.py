import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
model = joblib.load("models/model.bin")

app = FastAPI(title="Bank Marketing Term Deposit Prediction")

class ClientData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str

class PredictResponse(BaseModel):
    subscription_probability: float
    will_subscribe: bool


@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.model_dump()])
    prob = model.predict_proba(df)[0][1]
    #prediction = int(prob >= 0.5)

    return PredictResponse(
        subscription_probability= prob,
        will_subscribe = prob >= 0.5
    )
