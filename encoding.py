import pandas as pd


def encode_categoricals(df, drop_first=False):
    """Return dummy encoded DataFrame using pandas.get_dummies."""
    return pd.get_dummies(df, drop_first=drop_first)
