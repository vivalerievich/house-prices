"""Run the fixed Ames house-price training experiment."""

import argparse
import json
from pathlib import Path

import joblib

from house_prices.data import load_house_sales
from house_prices.training import fit_and_evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/train.csv"), help="Kaggle train.csv"
    )
    parser.add_argument(
        "--model-out", type=Path, help="Optional joblib path for the fitted pipeline"
    )
    args = parser.parse_args(argv)
    if args.model_out and args.model_out.resolve() == args.data.resolve():
        parser.error("--model-out must differ from the training CSV path")

    try:
        data = load_house_sales(args.data)
        model, report = fit_and_evaluate(data)
        if args.model_out:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, args.model_out)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(json.dumps(report, indent=2, allow_nan=False))
    return 0
