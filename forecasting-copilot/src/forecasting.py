"""ARIMA model selection, training, and forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.evaluation import calculate_metrics

CANDIDATE_ORDERS = [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]


def train_test_split_series(df: pd.DataFrame, train_fraction: float = 0.8):
    """Chronological (never shuffled) train/test split."""
    split_idx = int(len(df) * train_fraction)
    split_idx = max(1, min(split_idx, len(df) - 1))
    train = df.iloc[:split_idx].reset_index(drop=True)
    test = df.iloc[split_idx:].reset_index(drop=True)
    return train, test


def select_best_model(train: pd.DataFrame, test: pd.DataFrame, orders=CANDIDATE_ORDERS):
    """Fit each candidate ARIMA order on train, evaluate on test, keep best RMSE.

    Returns dict with keys: order, fitted_model, test_predictions, metrics.
    Failed orders are skipped without raising.
    """
    best = None
    train_series = train["volume"].astype(float)

    for order in orders:
        try:
            model = ARIMA(train_series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=len(test))
            forecast = np.asarray(forecast)
            metrics = calculate_metrics(test["volume"].to_numpy(), forecast)

            if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
                best = {
                    "order": order,
                    "fitted_model": fitted,
                    "test_predictions": forecast,
                    "metrics": metrics,
                }
        except Exception:
            continue

    if best is None:
        raise RuntimeError(
            "All candidate ARIMA orders failed to fit. Try a longer or cleaner history."
        )
    return best


def refit_on_full_history(df: pd.DataFrame, order):
    """Refit the chosen order on the full cleaned history before forecasting forward."""
    model = ARIMA(df["volume"].astype(float), order=order)
    return model.fit()


def forecast_future(fitted_model, last_date, horizon: int, alpha: float = 0.05) -> pd.DataFrame:
    """Produce a future forecast with confidence bounds."""
    forecast_result = fitted_model.get_forecast(steps=horizon)
    mean_forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=alpha)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    lower_col, upper_col = conf_int.columns[0], conf_int.columns[1]
    result = pd.DataFrame(
        {
            "date": future_dates,
            "forecast": np.asarray(mean_forecast),
            "lower_bound": np.asarray(conf_int[lower_col]).clip(min=0),
            "upper_bound": np.asarray(conf_int[upper_col]),
        }
    )
    return result
