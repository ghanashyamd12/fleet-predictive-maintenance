import joblib
import os


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../ml/model/rul_model.pkl"
)


def load_model_and_scaler():
    bundle = joblib.load(MODEL_PATH)

    # ✅ handle BOTH cases safely
    if isinstance(bundle, dict):
        model = bundle["model"]
        scaler = bundle["scaler"]
    else:
        # fallback (old model without scaler)
        model = bundle
        scaler = None

    return model, scaler
