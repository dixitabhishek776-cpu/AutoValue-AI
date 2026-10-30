from pathlib import Path
import json
import argparse
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import make_column_selector
from src.schema import REQUIRED_COLUMNS, FEATURE_COLUMNS, validate_dataframe

def main():
    parser = argparse.ArgumentParser(description="Train used-car price regression model")
    parser.add_argument("--data", default="data/sample/demo_used_cars.csv", help="CSV path")
    parser.add_argument("--target", default="selling_price")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    validate_dataframe(df)
    df = df[REQUIRED_COLUMNS].copy()
    df = df.dropna(subset=[args.target])
    df = df[df[args.target] > 0]
    X, y = df[FEATURE_COLUMNS], df[args.target].astype(float)
    categorical = ["brand", "fuel", "transmission"]
    numeric = [c for c in FEATURE_COLUMNS if c not in categorical]
    pre = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric)
    ])
    model = Pipeline([("preprocess", pre),
                      ("regressor", RandomForestRegressor(n_estimators=350, min_samples_leaf=2,
                                                         random_state=42, n_jobs=-1))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "dataset_path": args.data,
        "records_used": int(len(df)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "mae_inr": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse_inr": round(float(mean_squared_error(y_test, pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "disclosure": "Metrics describe only the supplied dataset. Demo data is synthetic and metrics are not real-market evidence."
    }
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(model, "models/used_car_model.joblib")
    with open("reports/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print("\\nSaved: models/used_car_model.joblib and reports/model_metrics.json")

if __name__ == "__main__":
    main()
