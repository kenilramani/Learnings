import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from encoding import encode_categoricals


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Airline": ["AA", "BA", "AA", "AA"],
        "Source": ["NY", "NY", "LA", "NY"],
        "Destination": ["LA", "LA", "NY", "LA"],
    })


def test_encode_categoricals_columns(sample_df):
    encoded = encode_categoricals(sample_df, drop_first=False)
    expected_columns = {
        "Airline_AA",
        "Airline_BA",
        "Source_LA",
        "Source_NY",
        "Destination_LA",
        "Destination_NY",
    }
    assert set(encoded.columns) == expected_columns


def test_encode_categoricals_drop_first(sample_df):
    encoded = encode_categoricals(sample_df, drop_first=True)
    expected_columns = {
        "Airline_BA",
        "Source_NY",
        "Destination_NY",
    }
    assert set(encoded.columns) == expected_columns
