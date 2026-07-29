import numpy as np
import pytest

from src.evaluation import calculate_metrics
from src.forecasting import (
    forecast_future,
    refit_on_full_history,
    select_best_model,
    train_test_split_series,
)
from src.synthetic_data import generate_synthetic_data


def _sample_df():
    df = generate_synthetic_data(days=120, random_seed=1)
    return df[["date", "volume"]]


def test_train_test_split_is_chronological():
    df = _sample_df()
    train, test = train_test_split_series(df, train_fraction=0.8)
    assert len(train) + len(test) == len(df)
    assert train["date"].max() < test["date"].min()


def test_select_best_model_returns_valid_order():
    df = _sample_df()
    train, test = train_test_split_series(df)
    best = select_best_model(train, test)
    assert best["order"] in [(1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)]
    assert best["metrics"]["rmse"] >= 0


def test_forecast_future_length_matches_horizon():
    df = _sample_df()
    train, test = train_test_split_series(df)
    best = select_best_model(train, test)
    full_model = refit_on_full_history(df, best["order"])
    horizon = 14
    forecast_df = forecast_future(full_model, df["date"].max(), horizon)
    assert len(forecast_df) == horizon
    assert set(["date", "forecast", "lower_bound", "upper_bound"]).issubset(forecast_df.columns)


def test_metric_calculation():
    actual = np.array([100, 200, 300])
    predicted = np.array([110, 190, 310])
    metrics = calculate_metrics(actual, predicted)
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["rmse"] > 0
    assert metrics["mape"] > 0


def test_metric_calculation_handles_zero_actual():
    actual = np.array([0, 100, 200])
    predicted = np.array([5, 90, 210])
    metrics = calculate_metrics(actual, predicted)
    assert metrics["mape"] == metrics["mape"]  # not NaN, since not all zero


def test_metric_calculation_all_zero_actual_is_nan():
    actual = np.array([0, 0, 0])
    predicted = np.array([1, 2, 3])
    metrics = calculate_metrics(actual, predicted)
    assert metrics["mape"] != metrics["mape"]  # NaN check
