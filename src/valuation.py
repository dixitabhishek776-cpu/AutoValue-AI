from pathlib import Path

import joblib
import pandas as pd

from src.schema import FEATURE_COLUMNS
from src.model_training import train_model

DATA_PATH = Path("datasets/clean_used_cars.csv")
MODEL_PATH = Path("models/used_car_model.joblib")


def load_sample_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Ensure the cleaned dataset is present."
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def load_model(path=MODEL_PATH):
    path = Path(path)

    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            # If the saved artifact is incompatible, retrain from bundled data.
            pass

    model, selected_name = train_model(DATA_PATH)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    return model


def predict_price(model, record):
    missing = [column for column in FEATURE_COLUMNS if column not in record]
    if missing:
        raise ValueError(
            f"Prediction input is missing: {', '.join(missing)}"
        )

    frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
    prediction = float(model.predict(frame)[0])
    return max(0.0, prediction)
