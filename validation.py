"""
GridShield – Time-Aware Validation Framework
Expanding window backtest, rolling CV, leakage prevention,
and piecewise training splits.
"""
import pandas as pd
import numpy as np
from typing import Generator, Tuple, List
from config import (
    INTERVALS_PER_DAY, VALIDATION_TRAIN_YEARS,
    VALIDATION_TEST_DAYS, COVID_LOCKDOWN_START, COVID_EXTENDED_END,
)


def expanding_window_splits(
    df: pd.DataFrame,
    min_train_days: int = 365 * 2,
    test_days: int = 30,
    step_days: int = 30,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Expanding window backtest.
    Training grows from min_train_days, test is a fixed window.
    Strictly chronological — no future leakage.
    """
    n_intervals = INTERVALS_PER_DAY
    min_train = min_train_days * n_intervals
    test_size = test_days * n_intervals
    step_size = step_days * n_intervals

    start = min_train
    while start + test_size <= len(df):
        train = df.iloc[:start]
        test = df.iloc[start:start + test_size]
        yield train, test
        start += step_size


def rolling_cv_splits(
    df: pd.DataFrame,
    train_days: int = 365,
    test_days: int = 30,
    step_days: int = 30,
    gap_days: int = 1,
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Rolling cross-validation with a fixed training window.
    Gap prevents information leakage across the train-test boundary.
    """
    n_intervals = INTERVALS_PER_DAY
    train_size = train_days * n_intervals
    test_size = test_days * n_intervals
    gap_size = gap_days * n_intervals
    step_size = step_days * n_intervals

    start = 0
    while start + train_size + gap_size + test_size <= len(df):
        train_end = start + train_size
        test_start = train_end + gap_size
        test_end = test_start + test_size
        train = df.iloc[start:train_end]
        test = df.iloc[test_start:test_end]
        yield train, test
        start += step_size


def get_regime_splits(df: pd.DataFrame) -> dict:
    """
    Split data into Pre-COVID, COVID, and Post-COVID regimes.
    Returns dict of {regime_name: DataFrame}.
    """
    covid_start = pd.Timestamp(COVID_LOCKDOWN_START)
    covid_end = pd.Timestamp(COVID_EXTENDED_END)

    splits = {
        "pre_covid": df[df.index < covid_start],
        "covid": df[(df.index >= covid_start) & (df.index <= covid_end)],
        "post_covid": df[df.index > covid_end],
    }
    return splits


def verify_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """
    Verify that there is no temporal overlap between train and test.
    Returns True if splits are clean.
    """
    if len(train_df) == 0 or len(test_df) == 0:
        return True
    train_max = train_df.index.max()
    test_min = test_df.index.min()
    return train_max < test_min


def get_feature_columns(df: pd.DataFrame, target: str = "LOAD") -> List[str]:
    """Get feature columns, excluding target and non-feature columns."""
    exclude = {target, "DATETIME", "_date", "year"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [
        np.float64, np.float32, np.int64, np.int32, np.uint8,
        float, int,
    ]]
