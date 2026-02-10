import joblib
import os


def load_model_and_scaler():
    # ✅ Always load the NEW trained model
    MODEL_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../ml/model/rul_model.pkl")
    )

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]

    return model, scaler
