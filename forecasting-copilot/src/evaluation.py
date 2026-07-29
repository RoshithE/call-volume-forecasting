"""Forecast accuracy metrics with plain-English explanations."""

from __future__ import annotations

import numpy as np

METRIC_EXPLANATIONS = {
    "mae": (
        "Mean Absolute Error: on average, the forecast is off by this many "
        "calls/units per day, regardless of direction."
    ),
    "rmse": (
        "Root Mean Squared Error: similar to MAE but penalizes larger misses "
        "more heavily, useful for spotting occasional big errors."
    ),
    "mape": (
        "Mean Absolute Percentage Error: on average, the forecast deviates from "
        "actual volume by this percentage. Easier to compare across datasets."
    ),
}


def calculate_metrics(actual, predicted) -> dict:
    """Calculate MAE, RMSE, and MAPE, handling zero actual values safely."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    nonzero_mask = actual != 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}
