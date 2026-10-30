import pandas as pd
import pytest

from src.schema import (
    REQUIRED_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_dataframe,
)


def valid_dataframe():
    return pd.DataFrame([{
        "brand": "Honda",
        "model_name": "Brio",
        "full_model_name": "Honda Brio S Mt",
        "year": 2016,
        "distance_travelled(kms)": 9680.0,
        "fuel_type": "Petrol",
        "city": "Mumbai",
        "price": 425000.0,
    }])


def test_valid_real_listing_schema():
    df = valid_dataframe()
    assert validate_dataframe(df)


def test_missing_required_column():
    df = valid_dataframe().drop(columns=["city"])

    with pytest.raises(ValueError, match="city"):
        validate_dataframe(df)


def test_empty_dataframe_is_rejected():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)

    with pytest.raises(ValueError, match="empty"):
        validate_dataframe(df)


def test_feature_columns_exclude_target():
    assert TARGET_COLUMN == "price"
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert set(FEATURE_COLUMNS).issubset(set(REQUIRED_COLUMNS))


def test_cleaned_dataset_has_expected_columns_and_records():
    df = pd.read_csv("datasets/clean_used_cars.csv", encoding="utf-8-sig")

    assert set(df.columns) == set(REQUIRED_COLUMNS)
    assert len(df) >= 1000
    assert df[REQUIRED_COLUMNS].notna().all().all()


def test_prices_and_years_are_valid():
    df = pd.read_csv("datasets/clean_used_cars.csv", encoding="utf-8-sig")

    assert (pd.to_numeric(df["price"], errors="coerce") > 0).all()
    assert pd.to_numeric(df["year"], errors="coerce").between(1990, 2026).all()
    assert (pd.to_numeric(df["distance_travelled(kms)"], errors="coerce") >= 0).all()
