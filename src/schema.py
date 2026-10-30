REQUIRED_COLUMNS = [
    "brand",
    "model_name",
    "full_model_name",
    "year",
    "distance_travelled(kms)",
    "fuel_type",
    "city",
    "price",
]

FEATURE_COLUMNS = [
    "brand",
    "model_name",
    "full_model_name",
    "year",
    "distance_travelled(kms)",
    "fuel_type",
    "city",
]

CATEGORICAL_COLUMNS = [
    "brand",
    "model_name",
    "full_model_name",
    "fuel_type",
    "city",
]

NUMERIC_COLUMNS = [
    "year",
    "distance_travelled(kms)",
]

TARGET_COLUMN = "price"


def validate_dataframe(df):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    return True
