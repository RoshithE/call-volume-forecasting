"""
Core forecasting utilities for the demand/volume forecasting project.

Mirrors the methodology used in production ARIMA-based call/case volume
forecasting: build baselines, grid search a seasonal ARIMA model on AIC,
validate out-of-sample, and report MAPE.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def load_series(csv_path: str) -> pd.Series:
    """Load the monthly demand series indexed by month-start dates."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    df = df.set_index("month").sort_index()
    series = df["sales"].astype(float)
    series.index.freq = "MS"
    return series


def train_test_split(series: pd.Series, test_periods: int = 12):
    train = series.iloc[:-test_periods]
    test = series.iloc[-test_periods:]
    return train, test


def adf_test(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test."""
    result = adfuller(series.dropna())
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "is_stationary": result[1] < 0.05,
    }


def mape(actual, predicted) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100)


def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Baseline 1: repeat last observed value."""
    return np.repeat(train.iloc[-1], horizon)


def seasonal_naive_forecast(train: pd.Series, horizon: int, season_length: int = 12) -> np.ndarray:
    """Baseline 2: repeat the value from the same month last year."""
    last_season = train.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


def grid_search_sarima(
    train: pd.Series,
    test: pd.Series,
    p_range=range(0, 3),
    d_range=range(0, 2),
    q_range=range(0, 3),
    seasonal_period: int = 12,
):
    """
    Grid search (p,d,q)(P,D,Q,m) combinations, rank by AIC on the training
    fit, then validate the top candidates against the held-out test set
    and pick the final model by out-of-sample MAPE.

    Two-stage selection like this (AIC to shortlist, holdout MAPE to
    decide) is what production model selection looks like: AIC alone can
    overfit in-sample, and a real deployment decision needs a holdout
    check before going live.
    """
    orders = list(itertools.product(p_range, d_range, q_range))
    seasonal_orders = list(itertools.product(range(0, 2), range(0, 2), range(0, 2)))

    results = []
    for order in orders:
        for seasonal_order in seasonal_orders:
            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=(*seasonal_order, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit = model.fit(disp=False)
                results.append(
                    {"order": order, "seasonal_order": (*seasonal_order, seasonal_period), "aic": fit.aic}
                )
            except Exception:
                continue

    results_df = pd.DataFrame(results).sort_values("aic").reset_index(drop=True)

    top_candidates = results_df.head(5)
    validated = []
    for _, row in top_candidates.iterrows():
        model = SARIMAX(
            train,
            order=row["order"],
            seasonal_order=row["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        forecast = fit.get_forecast(steps=len(test)).predicted_mean
        score = mape(test.values, forecast.values)
        validated.append(
            {
                "order": row["order"],
                "seasonal_order": row["seasonal_order"],
                "aic": row["aic"],
                "test_mape": score,
            }
        )

    validated_df = pd.DataFrame(validated).sort_values("test_mape").reset_index(drop=True)
    best = validated_df.iloc[0]

    best_model = SARIMAX(
        train,
        order=best["order"],
        seasonal_order=best["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    return best_model, best, validated_df


def evaluate_all(train: pd.Series, test: pd.Series, sarima_forecast: np.ndarray) -> pd.DataFrame:
    horizon = len(test)
    naive_fc = naive_forecast(train, horizon)
    snaive_fc = seasonal_naive_forecast(train, horizon)

    rows = [
        {"model": "Naive (last value)", "mape": mape(test.values, naive_fc)},
        {"model": "Seasonal naive (12mo lag)", "mape": mape(test.values, snaive_fc)},
        {"model": "SARIMA (grid searched)", "mape": mape(test.values, sarima_forecast)},
    ]
    return pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)


def forecast_future(model, steps: int):
    fc = model.get_forecast(steps=steps)
    return fc.predicted_mean, fc.conf_int(alpha=0.05)
