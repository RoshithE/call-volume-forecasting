"""Synthetic call-volume data generator.

Creates a realistic multi-month daily time series with weekly seasonality,
a gradual trend, random noise, occasional demand spikes, and periods of
unusually low volume. Used so the Streamlit app can be explored without
requiring the user to upload their own data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    days: int = 560,
    start_date: str = "2024-01-01",
    base_volume: float = 350.0,
    weekly_amplitude: float = 60.0,
    trend_per_day: float = 0.15,
    noise_std: float = 18.0,
    spike_probability: float = 0.02,
    spike_multiplier: float = 1.8,
    low_period_probability: float = 0.01,
    low_period_multiplier: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic daily call/demand volume dataset.

    Defaults produce at least 18 months (560 days) of data with weekly
    patterns, a gradual trend, random noise, demand spikes, and occasional
    low-volume periods.

    Returns a DataFrame with columns: date, volume, team, category.
    """
    if days < 30:
        raise ValueError("days must be at least 30 to produce a usable series")

    rng = np.random.default_rng(random_seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    day_of_week = dates.dayofweek  # Monday=0 ... Sunday=6
    weekly_pattern = np.array([1.15, 1.10, 1.05, 1.08, 1.20, 0.75, 0.60])
    weekly_component = weekly_amplitude * (weekly_pattern[day_of_week] - 1.0)

    trend_component = trend_per_day * np.arange(days)
    noise_component = rng.normal(loc=0.0, scale=noise_std, size=days)

    volume = base_volume + weekly_component + trend_component + noise_component

    spike_mask = rng.random(days) < spike_probability
    volume[spike_mask] *= spike_multiplier

    low_mask = rng.random(days) < low_period_probability
    volume[low_mask] *= low_period_multiplier

    volume = np.clip(volume, a_min=5, a_max=None)

    teams = rng.choice(["Fraud", "Billing", "Support"], size=days, p=[0.4, 0.35, 0.25])
    categories = rng.choice(["inbound", "outbound", "escalation"], size=days, p=[0.6, 0.3, 0.1])

    df = pd.DataFrame(
        {
            "date": dates,
            "volume": np.round(volume).astype(int),
            "team": teams,
            "category": categories,
        }
    )
    return df
