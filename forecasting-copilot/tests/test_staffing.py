import pandas as pd
import pytest

from src.staffing import estimate_staffing


def test_staffing_calculation_basic():
    forecast_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "forecast": [100, 200, 150, 300, 50],
    })
    result = estimate_staffing(forecast_df, available_staff=10, cases_per_employee_per_day=20, target_utilization_pct=100)
    assert result["average_required_staff"] == pytest.approx(160 / 20)
    assert result["peak_required_staff"] == pytest.approx(300 / 20)


def test_staffing_shortage_detected():
    forecast_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "forecast": [1000, 1000, 1000],
    })
    result = estimate_staffing(forecast_df, available_staff=5, cases_per_employee_per_day=20, target_utilization_pct=100)
    assert result["potential_shortage"] > 0


def test_staffing_zero_cases_per_employee_raises():
    forecast_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2), "forecast": [10, 20]})
    with pytest.raises(ValueError):
        estimate_staffing(forecast_df, available_staff=5, cases_per_employee_per_day=0)


def test_staffing_excess_capacity_detected():
    forecast_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "forecast": [10, 10, 10],
    })
    result = estimate_staffing(forecast_df, available_staff=100, cases_per_employee_per_day=20, target_utilization_pct=100)
    assert result["potential_excess_capacity"] > 0


def test_staffing_zero_forecast_volume_handled():
    forecast_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3),
        "forecast": [0, 0, 0],
    })
    result = estimate_staffing(forecast_df, available_staff=5, cases_per_employee_per_day=20, target_utilization_pct=100)
    assert result["average_required_staff"] == 0
    assert result["potential_shortage"] == 0
