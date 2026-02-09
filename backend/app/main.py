from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import load_model

app = FastAPI(title="Fleet Predictive Maintenance API")

# ✅ Proper CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # ← VERY IMPORTANT
    allow_headers=["*"],   # ← VERY IMPORTANT
)

model = load_model()


class SensorInput(BaseModel):
    sensors: list[float]


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(data: SensorInput):
    sensors_array = np.array(data.sensors).reshape(1, -1)
    prediction = model.predict(sensors_array)[0]

    return {"predicted_RUL": float(round(prediction, 2))}
