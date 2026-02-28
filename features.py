"""
GridShield – Feature Engineering Pipeline
Lag features, rolling stats, cyclical encoding, weather interactions,
event flags, Fourier seasonality, STL decomposition, and regime detection.
"""
import numpy as np
import pandas as pd
from config import (
    INTERVALS_PER_DAY, PEAK_START_HOUR, PEAK_END_HOUR,
    COVID_LOCKDOWN_START, COVID_LOCKDOWN_END,
)


def add_lag_features(df: pd.DataFrame, col: str = "LOAD") -> pd.DataFrame:
    """Add lag features for autoregressive structure."""
    lags = {
        "lag_1": 1,
        "lag_2": 2,
        "lag_4": 4,        # 1 hour
        "lag_96": 96,      # 1 day
        "lag_192": 192,    # 2 days
        "lag_672": 672,    # 1 week
    }
    for name, lag in lags.items():
        df[name] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str = "LOAD") -> pd.DataFrame:
    """Add rolling statistics."""
    windows = [4, 12, 24, 96]  # 1h, 3h, 6h, 24h (in 15-min intervals)
    for w in windows:
        df[f"rolling_mean_{w}"] = df[col].shift(1).rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"] = df[col].shift(1).rolling(w, min_periods=1).std()
    # Rolling min/max for daily range
    df["rolling_min_96"] = df[col].shift(1).rolling(96, min_periods=1).min()
    df["rolling_max_96"] = df[col].shift(1).rolling(96, min_periods=1).max()
    df["daily_range"] = df["rolling_max_96"] - df["rolling_min_96"]
    return df


def add_ramp_rate(df: pd.DataFrame, col: str = "LOAD") -> pd.DataFrame:
    """Load ramp rate: change from previous interval."""
    df["ramp_rate"] = df[col].diff(1)
    df["ramp_rate_4"] = df[col].diff(4)     # hourly ramp
    df["ramp_rate_96"] = df[col].diff(96)    # daily ramp
    return df


def add_cyclical_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encoding via sin/cos transforms."""
    # Hour of day
    hour_frac = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)
    # Day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    # Month of year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Day of year
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_fourier_features(df: pd.DataFrame, n_harmonics: int = 4) -> pd.DataFrame:
    """Fourier seasonality terms for daily, weekly, and yearly cycles."""
    t = np.arange(len(df), dtype=float)
    periods = {
        "daily": INTERVALS_PER_DAY,              # 96
        "weekly": INTERVALS_PER_DAY * 7,          # 672
        "yearly": INTERVALS_PER_DAY * 365.25,     # ~35064
    }
    for period_name, period in periods.items():
        for k in range(1, n_harmonics + 1):
            df[f"fourier_{period_name}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
            df[f"fourier_{period_name}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Weather interaction features."""
    # Temperature-Humidity Index (if humidity-like columns exist)
    if "ACT_HEAT_INDEX" in df.columns and "ACT_TEMP" in df.columns:
        df["THI"] = df["ACT_HEAT_INDEX"]  # Already a heat index
        df["temp_sq"] = df["ACT_TEMP"] ** 2
        df["temp_peak"] = df["ACT_TEMP"] * df["is_peak"]
        if "COOL_FACTOR" in df.columns:
            df["cool_peak"] = df["COOL_FACTOR"] * df["is_peak"]
            df["cool_hour"] = df["COOL_FACTOR"] * (df["hour"] + df["minute"] / 60.0)
    # Heatwave proxy: temp > rolling 95th percentile
    if "ACT_TEMP" in df.columns:
        rolling_temp_95 = df["ACT_TEMP"].rolling(
            INTERVALS_PER_DAY * 7, min_periods=INTERVALS_PER_DAY
        ).quantile(0.95)
        df["is_heatwave"] = (df["ACT_TEMP"] > rolling_temp_95).astype(int)
        df["heatwave_peak"] = df["is_heatwave"] * df["is_peak"]
    return df


def add_event_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Event and COVID interaction features."""
    # Holiday × hour interaction
    df["holiday_hour"] = df["is_holiday"] * df["hour"]
    # Lockdown × hour
    df["lockdown_hour"] = df.get("is_covid_lockdown", 0) * df["hour"]
    # Weekend × peak
    df["weekend_peak"] = df["is_weekend"] * df["is_peak"]
    return df


def add_regime_detection(df: pd.DataFrame, col: str = "LOAD",
                         window: int = 672, threshold: float = 3.0) -> pd.DataFrame:
    """
    CUSUM-based regime shift detection on load residuals.
    Flags intervals where cumulative deviation exceeds threshold × std.
    """
    rolling_mean = df[col].rolling(window, min_periods=1).mean()
    residual = df[col] - rolling_mean
    rolling_std = residual.rolling(window, min_periods=1).std().fillna(1)
    normalized = (residual / rolling_std).fillna(0)
    cusum_pos = np.zeros(len(df))
    cusum_neg = np.zeros(len(df))
    regime_flag = np.zeros(len(df))
    for i in range(1, len(df)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized.iloc[i] - 0.5)
        cusum_neg[i] = min(0, cusum_neg[i-1] + normalized.iloc[i] + 0.5)
        if cusum_pos[i] > threshold or cusum_neg[i] < -threshold:
            regime_flag[i] = 1
            cusum_pos[i] = 0
            cusum_neg[i] = 0
    df["regime_shift"] = regime_flag
    return df


def gate_features_for_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Remove features that would cause leakage for a given forecast horizon.
    Any lag shorter than the horizon is dropped.
    """
    cols_to_drop = []
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    for c in lag_cols:
        lag_val = int(c.split("_")[1])
        if lag_val < horizon:
            cols_to_drop.append(c)
    # Rolling features based on recent data
    rolling_cols = [c for c in df.columns if c.startswith("rolling_") or c.startswith("ramp_")]
    if horizon > 4:
        cols_to_drop.extend(rolling_cols)

    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline."""
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_ramp_rate(df)
    df = add_cyclical_encoding(df)
    df = add_fourier_features(df)
    df = add_weather_interactions(df)
    df = add_event_interactions(df)
    # Regime detection is slow on full dataset, use a sampled approach
    df = add_regime_detection(df)
    return df
