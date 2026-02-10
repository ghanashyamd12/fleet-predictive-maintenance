import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "ml", "model", "rul_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "ml", "model", "scaler.pkl")


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler
