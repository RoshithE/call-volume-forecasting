"""Helpers for loading uploaded CSV data and guessing column names."""

from __future__ import annotations

import pandas as pd


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV file (Streamlit UploadedFile or path) into a DataFrame."""
    df = pd.read_csv(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    return df


def guess_date_column(df: pd.DataFrame):
    for candidate in ["date", "Date", "DATE", "ds"]:
        if candidate in df.columns:
            return candidate
    return df.columns[0] if len(df.columns) else None


def guess_volume_column(df: pd.DataFrame):
    for candidate in ["volume", "Volume", "VOLUME", "y", "calls", "call_volume"]:
        if candidate in df.columns:
            return candidate
    return df.columns[1] if len(df.columns) > 1 else None
