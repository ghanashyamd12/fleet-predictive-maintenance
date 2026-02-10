from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Fleet Predictive Maintenance API")

# ✅ Proper CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # ← VERY IMPORTANT
    allow_headers=["*"],   # ← VERY IMPORTANT
)

from .model_loader import load_model_and_scaler

model, scaler = load_model_and_scaler()




class SensorInput(BaseModel):
    sensors: list[float]


@app.get("/")
def root():
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(data: SensorInput):
    import numpy as np

    sensors_array = np.array(data.sensors).reshape(1, -1)

    # apply scaler only if exists
    if scaler is not None:
        sensors_array = scaler.transform(sensors_array)

    prediction = model.predict(sensors_array)[0]

    return {"predicted_RUL": float(round(prediction, 2))}

