REQUIRED_COLUMNS = [
    "brand", "year", "km_driven", "fuel", "transmission",
    "owner", "mileage", "engine_cc", "seats", "selling_price"
]
FEATURE_COLUMNS = REQUIRED_COLUMNS[:-1]

def validate_dataframe(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    return True
