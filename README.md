# House Prices

Predict Ames, Iowa home sale prices from property characteristics. The training CLI evaluates a median baseline and reproduces the CatBoost model selected in the [exploration notebook](notebooks/house-prices.ipynb).

## Data and setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.11–3.14. Sign in to Kaggle, accept the competition rules, and download the files from the official [House Prices competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data). Extract the labeled `train.csv` into `data/train.csv`. Kaggle's `data_description.txt` documents the features. The evaluation below uses a split of `train.csv` and needs no competition submission.

```bash
uv sync
uv run house-prices --data data/train.csv
```

The CLI prints JSON metrics. Pass `--model-out artifacts/house-prices.joblib` to save the fitted pipeline for predictions in dollars. Keep the Kaggle CSV, competition archive, search cache, and generated model files local.

## Results

The CLI holds out 292 of the 1,460 labeled houses using an 80/20 split with seed 42. It fits and scores each fixed configuration on the same five shuffled folds within the 1,168 training rows, then fits on those rows and evaluates on the holdout. CatBoost predicts `log1p(SalePrice)` with nine engineered features, fold-local imputation and scaling, and one-hot encoding. `Id` identifies properties and stays outside the feature matrix.

| Metric | Median baseline CV | Median baseline holdout | CatBoost CV | CatBoost holdout |
| --- | ---: | ---: | ---: | ---: |
| MAE, USD | 54,586.14 ± 2,992.48 | 59,568.25 | 14,391.54 ± 989.83 | 15,232.55 |
| RMSE, USD | 78,860.09 ± 4,596.93 | 88,667.17 | 25,054.01 ± 5,480.01 | 26,858.97 |
| R² | −0.0459 ± 0.0162 | −0.0250 | 0.8926 ± 0.0392 | 0.9059 |
| RMSLE | 0.3905 ± 0.0206 | 0.4323 | 0.1194 ± 0.0174 | 0.1304 |

CV values show the mean and sample standard deviation across the five folds. The CatBoost settings are 600 iterations, depth 4, learning rate 0.1, L2 leaf regularization 3, and a Bernoulli subsample of 0.8. CV also guided model selection in the original comparison, so these CV estimates can be optimistic. The holdout is an internal split that has already been used for reporting. Generalization to other housing markets or time periods remains untested.

`PageValues` is a session-level feature from the Online Shoppers Intention dataset. Kaggle house-sale records contain property and transaction characteristics. For a prediction before sale, check that sale-related inputs such as `SaleType` and `SaleCondition` are available at the intended prediction time.

## Full notebook

The notebook includes exploratory analysis, three feature-preparation branches, 30 model configurations, and comparisons of four hyperparameter search methods across 17 ensemble variants. It selects by training CV RMSLE, compares native CatBoost categories and one-hot encoding, and reports the fixed configurations on the historical holdout. Permutation importance and residual plots describe the resulting models. The best Part I model is Lasso with engineered features. CatBoost with one-hot encoding and engineered features improved mean CV RMSLE by 6.41% relative to it.

On macOS, the optional notebook's LightGBM and XGBoost imports require the OpenMP runtime (`brew install libomp`). Launch Jupyter from the repository root and select the environment's Python kernel.

```bash
uv sync --group notebook
uv run --group notebook jupyter lab notebooks/house-prices.ipynb
```

Run cells in order. The full experiment runs 10,880 ensemble CV fits and can take hours. Completed searches checkpoint under `.cache/ensemble-searches/`. The CLI runs the two fixed configurations directly.

## Checks

Unit tests generate local synthetic fixtures and run without network access or Kaggle credentials.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
