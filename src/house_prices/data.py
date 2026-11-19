"""Load and validate the labeled Kaggle house sales."""

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from house_prices.features import ENGINEERING_COLUMNS

TARGET = "SalePrice"
IDENTIFIER = "Id"


def load_house_sales(path: Path) -> pd.DataFrame:
    """Read the training CSV and preserve the literal category `None`."""
    data = pd.read_csv(path, keep_default_na=False, na_values=["", "NA"])
    required = {IDENTIFIER, TARGET, *ENGINEERING_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[IDENTIFIER].isna().any() or not data[IDENTIFIER].is_unique:
        raise ValueError(f"{IDENTIFIER} must contain unique, nonmissing values.")
    nonnumeric = [
        name
        for name in (TARGET, *ENGINEERING_COLUMNS)
        if not is_numeric_dtype(data[name])
    ]
    if nonnumeric:
        raise ValueError(f"Expected numeric columns: {', '.join(sorted(nonnumeric))}")
    prices = data[TARGET]
    if prices.empty or not (np.isfinite(prices) & prices.gt(0)).all():
        raise ValueError(f"{TARGET} must contain finite, positive prices.")
    return data
