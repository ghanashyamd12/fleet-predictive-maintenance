import joblib
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../ml/model/rul_model.pkl"
)


def load_model_and_scaler():
    data = joblib.load(MODEL_PATH)

    model = data["model"]
    scaler = data["scaler"]

    return model, scaler
