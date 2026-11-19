"""Load training and inference CSVs in the Kaggle house-price schema."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from house_prices.features import ENGINEERING_COLUMNS

TARGET = "SalePrice"
IDENTIFIER = "Id"


def _load_house_data(
    path: Path, *, required: set[str], numeric: tuple[str, ...]
) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False, na_values=["", "NA"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[IDENTIFIER].isna().any() or not data[IDENTIFIER].is_unique:
        raise ValueError(f"{IDENTIFIER} must contain unique, nonmissing values.")
    nonnumeric = [name for name in numeric if not is_numeric_dtype(data[name])]
    if nonnumeric:
        raise ValueError(f"Expected numeric columns: {', '.join(sorted(nonnumeric))}")
    return data


def load_house_sales(path: Path) -> pd.DataFrame:
    """Read the training CSV and preserve the literal category `None`."""
    data = _load_house_data(
        path,
        required={IDENTIFIER, TARGET, *ENGINEERING_COLUMNS},
        numeric=(TARGET, *ENGINEERING_COLUMNS),
    )
    prices = data[TARGET]
    if prices.empty or not (np.isfinite(prices) & prices.gt(0)).all():
        raise ValueError(f"{TARGET} must contain finite, positive prices.")
    return data


def load_house_inputs(path: Path, required_features: Iterable[str]) -> pd.DataFrame:
    """Load Kaggle-style input rows with every feature expected by the fitted model."""
    return _load_house_data(
        path,
        required={IDENTIFIER, *ENGINEERING_COLUMNS, *required_features},
        numeric=ENGINEERING_COLUMNS,
    )
