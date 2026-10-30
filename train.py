from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.schema import (
    REQUIRED_COLUMNS,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    validate_dataframe,
)


def build_preprocessor():
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    return ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate used-car price models on real listing data."
    )
    parser.add_argument(
        "--data",
        default="datasets/clean_used_cars.csv",
        help="Path to cleaned used-car listing CSV",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    validate_dataframe(df)
    df = df[REQUIRED_COLUMNS].copy()

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    if len(df) < 30:
        raise ValueError(f"Not enough valid records to train: {len(df)}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=args.random_state,
        ),
    }

    results = []
    fitted_models = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("regressor", estimator),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = {
            "model": name,
            "mae_inr": round(float(mean_absolute_error(y_test, predictions)), 2),
            "rmse_inr": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }

        results.append(metrics)
        fitted_models[name] = pipeline
        print(json.dumps(metrics, indent=2))

    best_result = min(results, key=lambda item: item["mae_inr"])
    best_name = best_result["model"]
    best_model = fitted_models[best_name]

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(best_model, "models/used_car_model.joblib")

    report = {
        "dataset_path": str(data_path),
        "dataset_type": "Used-car listings",
        "target": "price",
        "target_disclosure": (
            "Prices are listed asking prices in INR, not verified final transaction prices."
        ),
        "records_used": int(len(df)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "split": "Random train/test split",
        "random_state": args.random_state,
        "model_comparison": results,
        "selected_model": best_name,
        "selected_model_metrics": best_result,
        "limitations": [
            "The dataset is relatively small.",
            "The source data covers older listings and may not represent current prices.",
            "Random split results may be optimistic for unseen models or cities.",
            "Metrics describe this dataset and split only; they are not a guarantee of market accuracy.",
        ],
    }

    with open("reports/model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nSelected model:", best_name)
    print("Saved models/used_car_model.joblib")
    print("Saved reports/model_metrics.json")


if __name__ == "__main__":
    main()
