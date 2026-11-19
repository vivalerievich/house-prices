import json

import joblib
import numpy as np
import pandas as pd
import pytest

from house_prices.cli import main
from house_prices.data import load_house_sales
from house_prices.features import (
    add_house_features,
    build_engineered_pipeline,
    select_categorical,
)


@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "Id": range(1, 37),
            "SalePrice": [110_000 + 2_500 * row for row in range(36)],
            "PoolArea": [0, 100, np.nan] * 12,
            "GarageArea": [0, 300, np.nan] * 12,
            "GarageCars": [0, 1, np.nan] * 12,
            "GrLivArea": [900 + 10 * row for row in range(36)],
            "LotArea": [4_000 + 25 * row for row in range(36)],
            "TotalBsmtSF": [0, 500, 800] * 12,
            "YrSold": [2009, 2010] * 18,
            "YearBuilt": [1980 + row for row in range(36)],
            "OverallQual": [4, 5, 8] * 12,
            "YearRemodAdd": [1980 + row + row % 2 for row in range(36)],
            "MSSubClass": [20, 60, 120] * 12,
            "Neighborhood": ["None", "North", np.nan] * 12,
        }
    )


def test_load_preserves_literal_category_and_handles_missing(sales, tmp_path):
    path = tmp_path / "train.csv"
    sales.to_csv(path, index=False, na_rep="NA")

    loaded = load_house_sales(path)

    assert loaded.loc[0, "Neighborhood"] == "None"
    assert pd.isna(loaded.loc[2, "Neighborhood"])
    assert loaded["SalePrice"].tolist() == sales["SalePrice"].tolist()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"Id": 1}, "unique, nonmissing"),
        ({"SalePrice": 0}, "finite, positive prices"),
        ({"SalePrice": "invalid"}, "Expected numeric columns"),
    ],
)
def test_load_rejects_invalid_training_data(sales, tmp_path, changes, message):
    path = tmp_path / "train.csv"
    sales.assign(**changes).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_house_sales(path)


def test_engineered_features_distinguish_zero_from_missing(sales):
    features = add_house_features(sales)

    np.testing.assert_allclose(features["HasPool"].iloc[:3], [0, 1, np.nan])
    np.testing.assert_allclose(features["HasGarage"].iloc[:3], [0, 1, np.nan])
    assert features.loc[1, "TotalSF"] == 1_410
    assert features.loc[1, "HouseAge"] == 29
    assert features.loc[1, "IsRemodeled"] == 1
    assert "HasPool" not in sales


def test_preprocessing_accepts_unseen_categories_using_fitted_columns(sales):
    predictors = sales.drop(columns=["Id", "SalePrice"]).assign(Constant=1)
    assert "MSSubClass" in select_categorical(predictors)
    pipeline = build_engineered_pipeline().fit(predictors)
    holdout = predictors.iloc[:3].assign(Neighborhood="Unseen", MSSubClass=np.nan)

    transformed = pipeline.transform(holdout)

    preprocessor = pipeline.named_steps["preprocessor"]
    assert "Constant" not in preprocessor.transformers_[0][2]
    assert transformed.shape[1] == pipeline.transform(predictors).shape[1]
    assert np.isfinite(transformed).all()


def test_cli_trains_and_saves_pipeline_using_local_csv(sales, tmp_path, capsys):
    data_path = tmp_path / "train.csv"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(data_path, index=False)
    model_path = tmp_path / "artifacts" / "model.joblib"

    assert main(["--data", str(data_path), "--model-out", str(model_path)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["cv_folds"] == 5
    assert (report["train_rows"], report["holdout_rows"]) == (28, 8)
    assert set(report["models"]) == {"median_baseline", "catboost"}
    for results in report["models"].values():
        assert set(results["holdout"]) == {"mae", "rmse", "r2", "rmsle"}
        assert all(np.isfinite(list(results["holdout"].values())))

    model = joblib.load(model_path)
    assert not {"Id", "SalePrice"}.intersection(model.feature_names_in_)
    price = model.predict(sales.drop(columns=["Id", "SalePrice"]).iloc[:1])[0]
    assert np.isfinite(price) and price > 0
