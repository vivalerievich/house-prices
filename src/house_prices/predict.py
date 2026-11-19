"""Predict Ames home prices from a feature CSV and a fitted pipeline."""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from house_prices.data import IDENTIFIER, TARGET, load_house_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Fitted joblib pipeline"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="Kaggle-style feature CSV"
    )
    args = parser.parse_args(argv)

    try:
        model = joblib.load(args.model)
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is None or not callable(getattr(model, "predict", None)):
            raise ValueError("Expected a fitted House Prices pipeline.")
        data = load_house_inputs(args.data, feature_names)
        prices = model.predict(data.loc[:, feature_names])
    except (OSError, ValueError) as error:
        parser.error(str(error))

    predictions = pd.DataFrame({IDENTIFIER: data[IDENTIFIER], TARGET: prices})
    predictions.to_csv(sys.stdout, index=False)
    return 0
