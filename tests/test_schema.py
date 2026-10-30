import pandas as pd
import pytest
from src.schema import validate_dataframe, REQUIRED_COLUMNS

def test_valid_schema():
    df = pd.DataFrame([{c: 1 for c in REQUIRED_COLUMNS}])
    assert validate_dataframe(df)

def test_missing_column():
    df = pd.DataFrame([{"brand":"Maruti"}])
    with pytest.raises(ValueError):
        validate_dataframe(df)
