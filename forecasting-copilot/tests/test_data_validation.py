import pandas as pd

from src.data_validation import validate_and_clean


def make_df(dates, volumes):
    return pd.DataFrame({"date": dates, "volume": volumes})


def test_missing_columns_returns_error():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    cleaned, report = validate_and_clean(df, date_col="date", volume_col="volume")
    assert cleaned is None
    assert not report.is_valid
    assert any("date" in e for e in report.errors)


def test_invalid_dates_are_dropped():
    dates = ["2024-01-01", "not-a-date"] + [f"2024-01-{d:02d}" for d in range(3, 32)]
    volumes = [100] * len(dates)
    df = make_df(dates, volumes)
    cleaned, report = validate_and_clean(df, "date", "volume", min_records=10)
    assert cleaned is not None
    assert any("unparseable date" in c for c in report.changes)


def test_duplicate_dates_are_aggregated():
    dates = [f"2024-01-{d:02d}" for d in range(1, 32)] + ["2024-01-01"]
    volumes = [100] * 31 + [50]
    df = make_df(dates, volumes)
    cleaned, report = validate_and_clean(df, "date", "volume", min_records=10)
    assert cleaned is not None
    first_day_volume = cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-01"), "volume"].iloc[0]
    assert first_day_volume == 150
    assert any("duplicate" in c for c in report.changes)


def test_negative_values_are_not_silently_fixed():
    dates = [f"2024-01-{d:02d}" for d in range(1, 32)]
    volumes = [100] * 30 + [-5]
    df = make_df(dates, volumes)
    cleaned, report = validate_and_clean(df, "date", "volume", min_records=10)
    assert cleaned is None
    assert any("negative" in e.lower() for e in report.errors)


def test_insufficient_history_is_blocking():
    dates = [f"2024-01-{d:02d}" for d in range(1, 6)]
    volumes = [100] * 5
    df = make_df(dates, volumes)
    cleaned, report = validate_and_clean(df, "date", "volume", min_records=30)
    assert cleaned is None
    assert any("at least" in e for e in report.errors)


def test_nonnumeric_volume_dropped():
    dates = [f"2024-01-{d:02d}" for d in range(1, 32)]
    volumes = [100] * 30 + ["oops"]
    df = make_df(dates, volumes)
    cleaned, report = validate_and_clean(df, "date", "volume", min_records=10)
    assert cleaned is not None
    assert any("non-numeric" in c for c in report.changes)
