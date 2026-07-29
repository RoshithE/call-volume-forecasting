"""Simplified staffing estimation based on forecasted demand."""

from __future__ import annotations

import pandas as pd


def estimate_staffing(
    forecast_df: pd.DataFrame,
    available_staff: float,
    cases_per_employee_per_day: float,
    target_utilization_pct: float = 85.0,
) -> dict:
    """Estimate staffing needs from a forecast dataframe with a 'forecast' column.

    This is a simplified planning estimate, not a full workforce-management
    model (it does not account for shift patterns, part-time staff, shrinkage,
    or intraday arrival curves).
    """
    if cases_per_employee_per_day <= 0:
        raise ValueError("cases_per_employee_per_day must be greater than zero")

    effective_capacity = cases_per_employee_per_day * (target_utilization_pct / 100.0)

    working = forecast_df.copy()
    working["required_staff"] = working["forecast"] / effective_capacity

    average_required = float(working["required_staff"].mean())
    peak_required = float(working["required_staff"].max())
    peak_row = working.loc[working["required_staff"].idxmax()]

    shortage = max(0.0, peak_required - available_staff)
    excess = max(0.0, available_staff - average_required)

    high_demand_dates = (
        working.sort_values("required_staff", ascending=False)
        .head(5)[["date", "forecast", "required_staff"]]
    )

    return {
        "average_required_staff": average_required,
        "peak_required_staff": peak_required,
        "peak_demand_date": peak_row["date"],
        "available_staff": available_staff,
        "potential_shortage": shortage,
        "potential_excess_capacity": excess,
        "high_demand_dates": high_demand_dates,
        "detail": working[["date", "forecast", "required_staff"]],
    }
