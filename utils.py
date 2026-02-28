"""
GridShield – Data Loading & Cleaning Utilities
Handles SAS-style datetime parsing, merging of load/weather/events data,
and basic preprocessing.
"""
import pandas as pd
import numpy as np
from config import (
    TRAIN_LOAD_FILE, TRAIN_WEATHER_FILE, EVENTS_FILE,
    TEST_LOAD_FILE, TEST_WEATHER_FILE,
    DATETIME_FORMAT, FREQ,
    COVID_LOCKDOWN_START, COVID_LOCKDOWN_END, COVID_EXTENDED_END,
    PEAK_START_HOUR, PEAK_END_HOUR,
)


def parse_sas_datetime(series: pd.Series) -> pd.Series:
    """Parse SAS-style datetimes like '01APR2013:00:15:00' to pd.Timestamp."""
    return pd.to_datetime(series, format=DATETIME_FORMAT)


def load_load_data(is_train: bool = True) -> pd.DataFrame:
    """Load electricity load data."""
    file_path = TRAIN_LOAD_FILE if is_train else TEST_LOAD_FILE
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["DATETIME"] = parse_sas_datetime(df["DATETIME"])
    df = df.sort_values("DATETIME").reset_index(drop=True)
    df = df.set_index("DATETIME")
    # Handle any missing intervals by reindexing
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=FREQ)
    df = df.reindex(full_idx)
    df.index.name = "DATETIME"
    # Forward-fill small gaps, then interpolate
    df["LOAD"] = df["LOAD"].interpolate(method="linear", limit=4)
    df["LOAD"] = df["LOAD"].ffill().bfill()
    return df


def load_weather_data(is_train: bool = True) -> pd.DataFrame:
    """Load external factor / weather data."""
    file_path = TRAIN_WEATHER_FILE if is_train else TEST_WEATHER_FILE
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["DATETIME"] = parse_sas_datetime(df["DATETIME"])
    df = df.sort_values("DATETIME").reset_index(drop=True)
    df = df.set_index("DATETIME")
    # Interpolate missing weather values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].interpolate(method="linear", limit=8)
        df[col] = df[col].ffill().bfill()
    return df


def load_events_data() -> pd.DataFrame:
    """Load events / holiday data.
    Handles messy date entries: date ranges, descriptive text, etc.
    Only keeps rows with well-formed dates.
    """
    import re
    df = pd.read_csv(EVENTS_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    parsed_dates = []
    holiday_flags = []
    event_names = []

    for _, row in df.iterrows():
        date_str = str(row.get("Date", "")).strip()
        h_ind = row.get("Holiday_Ind", 0)
        e_name = str(row.get("Event_Name", ""))

        # Match clean date formats: "01-Jan-11", "15/06/2020", "25-Dec-2020"
        # Skip anything with "to", "Phase", "Lockdown", etc.
        if any(skip in date_str for skip in ["Phase", "Lockdown", "onwards"]):
            continue

        # Handle "dd/mm/yyyy to dd/mm/yyyy" date ranges
        range_match = re.match(
            r"(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})",
            date_str
        )
        if range_match:
            try:
                start = pd.to_datetime(range_match.group(1), dayfirst=True)
                end = pd.to_datetime(range_match.group(2), dayfirst=True)
                for d in pd.date_range(start, end):
                    parsed_dates.append(d)
                    holiday_flags.append(h_ind)
                    event_names.append(e_name)
            except Exception:
                pass
            continue

        # Skip other "to" patterns (like "3rd to 5th Aug-20")
        if " to " in date_str.lower():
            continue

        # Try parsing single dates
        try:
            d = pd.to_datetime(date_str, dayfirst=True)
            parsed_dates.append(d)
            holiday_flags.append(h_ind)
            event_names.append(e_name)
        except Exception:
            continue

    result = pd.DataFrame({
        "Date": parsed_dates,
        "Event_Name": event_names,
        "Holiday_Ind": holiday_flags,
    })
    result = result.sort_values("Date").reset_index(drop=True)
    return result


def merge_all_data(is_train: bool = True) -> pd.DataFrame:
    """
    Merge load, weather, and events into a single dataframe.
    Returns a DataFrame indexed by DATETIME at 15-min resolution.
    """
    load_df = load_load_data(is_train)
    weather_df = load_weather_data(is_train)

    # Merge load and weather on index
    df = load_df.join(weather_df, how="left")

    # Merge events: create a date column for matching
    events_df = load_events_data()
    df["_date"] = df.index.date
    events_df["_date"] = events_df["Date"].dt.date

    # Create holiday flag
    holiday_dates = set(events_df["_date"].unique())
    df["is_holiday"] = df["_date"].apply(lambda d: 1 if d in holiday_dates else 0)
    df.drop(columns=["_date"], inplace=True)

    # Add basic time features
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Peak-hour flag
    df["is_peak"] = ((df["hour"] >= PEAK_START_HOUR) &
                     (df["hour"] < PEAK_END_HOUR)).astype(int)

    # COVID regime flags
    covid_start = pd.Timestamp(COVID_LOCKDOWN_START)
    covid_end = pd.Timestamp(COVID_LOCKDOWN_END)
    covid_ext = pd.Timestamp(COVID_EXTENDED_END)
    df["is_covid_lockdown"] = ((df.index >= covid_start) &
                               (df.index <= covid_end)).astype(int)
    df["is_covid_period"] = ((df.index >= covid_start) &
                             (df.index <= covid_ext)).astype(int)

    return df


def get_train_test_split(df: pd.DataFrame, test_start: str = "2020-04-01"):
    """
    Split data into train/test by date.
    Default: test from April 2020 onward (last year of data).
    """
    test_start_ts = pd.Timestamp(test_start)
    train = df[df.index < test_start_ts].copy()
    test = df[df.index >= test_start_ts].copy()
    return train, test
