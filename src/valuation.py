from pathlib import Path
import joblib
import pandas as pd
from src.schema import FEATURE_COLUMNS

def load_sample_data():
    path = Path("data/sample/demo_used_cars.csv")
    return pd.read_csv(path)

def load_model(path):
    path = Path(path)
    return joblib.load(path) if path.exists() else None

def predict_price(model, record):
    frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(frame)[0])
    return max(0.0, prediction)
