"""Fold-local preprocessing shared by the CLI and the exploration notebook."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

DOMINANT_SHARE_THRESHOLD = 0.86
NOMINAL_CODE = "MSSubClass"
ENGINEERING_COLUMNS = (
    "PoolArea",
    "GarageArea",
    "GarageCars",
    "GrLivArea",
    "LotArea",
    "TotalBsmtSF",
    "YrSold",
    "YearBuilt",
    "OverallQual",
    "YearRemodAdd",
)


def add_house_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add nine house-level features while preserving unknown values."""
    result = data.copy()

    pool_area = result["PoolArea"]
    result["HasPool"] = np.select(
        [pool_area.gt(0).fillna(False), pool_area.eq(0).fillna(False)],
        [1.0, 0.0],
        default=np.nan,
    )

    garage_area, garage_cars = result["GarageArea"], result["GarageCars"]
    has_garage = garage_area.gt(0) | garage_cars.gt(0)
    no_garage = garage_area.eq(0) & garage_cars.eq(0)
    result["HasGarage"] = np.select(
        [has_garage.fillna(False), no_garage.fillna(False)],
        [1.0, 0.0],
        default=np.nan,
    )

    living_area = result["GrLivArea"].where(result["GrLivArea"].gt(0))
    result["LotToLivingRatio"] = result["LotArea"] / living_area
    result["TotalSF"] = result["GrLivArea"] + result["TotalBsmtSF"]
    result["HouseAge"] = result["YrSold"] - result["YearBuilt"]
    result["QualityArea"] = result["GrLivArea"] * result["OverallQual"]
    result["LogTotalSF"] = np.log1p(result["TotalSF"].where(result["TotalSF"].ge(0)))
    result["LogLotArea"] = np.log1p(result["LotArea"].where(result["LotArea"].ge(0)))
    remodeled, built = result["YearRemodAdd"], result["YearBuilt"]
    result["IsRemodeled"] = np.select(
        [remodeled.gt(built).fillna(False), remodeled.le(built).fillna(False)],
        [1.0, 0.0],
        default=np.nan,
    )
    return result


def select_numeric(data: pd.DataFrame) -> list[str]:
    """Keep variable numeric features, treating housing type as nominal."""
    return [
        column
        for column in data.select_dtypes(include="number")
        if column != NOMINAL_CODE
        and data[column].value_counts(dropna=False, normalize=True).max()
        < DOMINANT_SHARE_THRESHOLD
    ]


def select_categorical(data: pd.DataFrame) -> list[str]:
    """Keep variable categorical features, including the housing type code."""
    numeric = set(data.select_dtypes(include="number")) - {NOMINAL_CODE}
    return [
        column
        for column in data
        if column not in numeric
        and data[column].value_counts(dropna=False, normalize=True).max()
        < DOMINANT_SHARE_THRESHOLD
    ]


def prepare_categories(data: pd.DataFrame) -> pd.DataFrame:
    """Preserve numeric category ordering and use a reserved code for unknown type."""
    categories = data.astype(object)
    if NOMINAL_CODE in data and pd.api.types.is_numeric_dtype(data[NOMINAL_CODE]):
        categories[NOMINAL_CODE] = data[NOMINAL_CODE].fillna(-1).astype(object)
    return categories


def build_numeric_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def build_preprocessor() -> ColumnTransformer:
    categorical = Pipeline(
        [
            (
                "categorical_types",
                FunctionTransformer(prepare_categories, feature_names_out="one-to-one"),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value="Missing", keep_empty_features=True
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", build_numeric_pipeline(), select_numeric),
            ("categorical", categorical, select_categorical),
        ],
        sparse_threshold=0,
    )


def build_engineered_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("features", FunctionTransformer(add_house_features, validate=False)),
            ("preprocessor", build_preprocessor()),
        ]
    )
