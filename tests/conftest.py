"""Make the repository root importable and load the dataset once per session."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import preprocessing as P  # noqa: E402  (import after sys.path fix)


@pytest.fixture(scope="session")
def data():
    """(df, X, y_reg, y3, y2, groups) loaded once for the whole test session."""
    df = P.load_raw()
    X, y_reg, y3, y2, groups = P.get_xy(df)
    return df, X, y_reg, y3, y2, groups
