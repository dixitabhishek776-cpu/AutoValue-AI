from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from src.schema import (
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


def train_model(data_path="datasets/clean_used_cars.csv", random_state=42):
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    validate_dataframe(df)

    for column in NUMERIC_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    df = df[df[TARGET_COLUMN] > 0]

    if len(df) < 30:
        raise ValueError(f"Not enough valid records to train: {len(df)}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(float)

    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=random_state,
        ),
    }

    fitted_models = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor()),
            ("regressor", estimator),
        ])
        pipeline.fit(X, y)
        fitted_models[name] = pipeline

    # Select the candidate with the lowest training MAE.
    # The official held-out comparison remains in train.py.
    from sklearn.metrics import mean_absolute_error

    scores = {
        name: mean_absolute_error(y, pipeline.predict(X))
        for name, pipeline in fitted_models.items()
    }

    best_name = min(scores, key=scores.get)
    return fitted_models[best_name], best_name
