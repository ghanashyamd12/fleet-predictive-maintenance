import joblib
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../ml/model/rul_model.pkl"
)

def load_model():
    return joblib.load(MODEL_PATH)
