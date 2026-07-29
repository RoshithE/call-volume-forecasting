"""Exploratory analysis helpers: summary statistics and Plotly charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def summary_statistics(df: pd.DataFrame) -> dict:
    """Compute headline summary statistics from a clean date/volume dataframe."""
    volume = df["volume"]
    recent_window = min(14, len(df) // 2) if len(df) >= 4 else len(df)

    if recent_window > 0 and len(df) >= recent_window * 2:
        recent_avg = volume.tail(recent_window).mean()
        prior_avg = volume.iloc[-recent_window * 2 : -recent_window].mean()
        pct_change = ((recent_avg - prior_avg) / prior_avg * 100) if prior_avg else 0.0
    else:
        pct_change = 0.0

    return {
        "num_records": int(len(df)),
        "date_start": df["date"].min(),
        "date_end": df["date"].max(),
        "average_volume": float(volume.mean()),
        "median_volume": float(volume.median()),
        "max_volume": float(volume.max()),
        "min_volume": float(volume.min()),
        "highest_volume_date": df.loc[volume.idxmax(), "date"],
        "lowest_volume_date": df.loc[volume.idxmin(), "date"],
        "recent_pct_change": float(pct_change),
    }


def day_of_week_pattern(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["day_of_week"] = working["date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = working.groupby("day_of_week")["volume"].mean().reindex(order).reset_index()
    grouped.columns = ["day_of_week", "average_volume"]
    return grouped


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["month"] = working["date"].dt.to_period("M").astype(str)
    grouped = working.groupby("month")["volume"].mean().reset_index()
    grouped.columns = ["month", "average_volume"]
    return grouped


def historical_chart(df: pd.DataFrame):
    fig = px.line(df, x="date", y="volume", title="Historical Demand / Call Volume")
    fig.update_layout(xaxis_title="Date", yaxis_title="Volume")
    return fig


def day_of_week_chart(dow_df: pd.DataFrame):
    fig = px.bar(dow_df, x="day_of_week", y="average_volume", title="Average Volume by Day of Week")
    return fig


def monthly_trend_chart(month_df: pd.DataFrame):
    fig = px.bar(month_df, x="month", y="average_volume", title="Average Volume by Month")
    return fig
