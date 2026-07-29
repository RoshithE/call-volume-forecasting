"""Data validation and cleaning for uploaded demand/call-volume data.

The validator distinguishes between:
  * Blocking errors - problems severe enough that we refuse to silently
    "fix" them (e.g. negative volumes, missing required columns, too few
    records). The user must resolve these in the source data.
  * Auto-corrected changes - reasonable, well-understood fixes that are
    applied automatically (date parsing, sorting, de-duplication via
    aggregation, and interpolation of small gaps). Every change is logged
    and returned to the caller so it can be displayed to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

MIN_RECORDS_REQUIRED = 30
MAX_INTERPOLATION_GAP_DAYS = 3


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    changes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_and_clean(df, date_col, volume_col, min_records=MIN_RECORDS_REQUIRED):
    """Validate and clean a raw uploaded dataframe.

    Returns (cleaned_df, report). cleaned_df is None if a blocking error
    prevents further processing.
    """
    report = ValidationReport()

    if date_col not in df.columns:
        report.errors.append(f"Required date column '{date_col}' was not found.")
    if volume_col not in df.columns:
        report.errors.append(f"Required volume column '{volume_col}' was not found.")
    if not report.is_valid:
        return None, report

    working = df[[date_col, volume_col]].copy()
    working.columns = ["date", "volume"]

    parsed_dates = pd.to_datetime(working["date"], errors="coerce")
    n_bad_dates = int(parsed_dates.isna().sum())
    if n_bad_dates > 0:
        report.changes.append(f"Dropped {n_bad_dates} row(s) with unparseable date values.")
    working["date"] = parsed_dates
    working = working[working["date"].notna()]

    numeric_volume = pd.to_numeric(working["volume"], errors="coerce")
    n_bad_volume = int(numeric_volume.isna().sum())
    if n_bad_volume > 0:
        report.changes.append(f"Dropped {n_bad_volume} row(s) with non-numeric volume values.")
    working["volume"] = numeric_volume
    working = working[working["volume"].notna()]

    if working.empty:
        report.errors.append("No valid rows remained after removing bad dates/volumes.")
        return None, report

    n_negative = int((working["volume"] < 0).sum())
    if n_negative > 0:
        report.errors.append(
            f"{n_negative} row(s) contain negative volume values. Negative call/demand "
            "volume is not a value we will silently correct - please fix the source "
            "data and re-upload."
        )
        return None, report

    working = working.sort_values("date")

    n_duplicates = int(working["date"].duplicated().sum())
    if n_duplicates > 0:
        working = working.groupby("date", as_index=False)["volume"].sum()
        report.changes.append(f"Aggregated {n_duplicates} duplicate date row(s) by summing volume.")
    else:
        working = working[["date", "volume"]]

    if len(working) < min_records:
        report.errors.append(
            f"Only {len(working)} usable record(s) found; at least {min_records} are "
            "required for a reliable forecast."
        )
        return None, report

    full_range = pd.date_range(working["date"].min(), working["date"].max(), freq="D")
    working = working.set_index("date").reindex(full_range)
    working.index.name = "date"
    n_missing = int(working["volume"].isna().sum())

    if n_missing > 0:
        is_na = working["volume"].isna()
        group_id = (is_na != is_na.shift()).cumsum()
        run_length = working.groupby(group_id)["volume"].transform("size")
        small_gap_mask = is_na & (run_length <= MAX_INTERPOLATION_GAP_DAYS)
        large_gap_mask = is_na & (run_length > MAX_INTERPOLATION_GAP_DAYS)

        if small_gap_mask.any():
            working["volume"] = working["volume"].interpolate(
                method="linear", limit=MAX_INTERPOLATION_GAP_DAYS, limit_area="inside"
            )
            report.changes.append(
                f"Interpolated {int(small_gap_mask.sum())} missing day(s) found in gaps of "
                f"{MAX_INTERPOLATION_GAP_DAYS} days or fewer."
            )
        if large_gap_mask.any():
            report.warnings.append(
                f"{int(large_gap_mask.sum())} missing day(s) fall within gaps larger than "
                f"{MAX_INTERPOLATION_GAP_DAYS} days and were left blank rather than guessed "
                "at. Consider reviewing your source data for these dates."
            )

    working = working.reset_index()
    working = working.dropna(subset=["volume"])
    working["volume"] = working["volume"].astype(float)

    return working, report
