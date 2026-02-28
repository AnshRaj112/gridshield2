"""
GridShield – Historical Backtest Engine
Computes all mandatory metrics and comparison against naive baselines.
"""
import numpy as np
import pandas as pd
from typing import Dict
from penalty import (
    compute_penalty_summary, compute_naive_penalty,
    compute_full_penalty, compute_deviation, compute_pct_deviation,
)
from config import FINANCIAL_CAP, INTERVALS_PER_DAY


def run_backtest(
    forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    regime: str = "tiered",
) -> Dict:
    """
    Run full backtest on forecast vs actual.
    Computes all mandatory Output B metrics.
    """
    # Model metrics
    model_summary = compute_penalty_summary(forecast, actual, is_peak, regime)

    # Naive baseline: previous-day same-time
    naive_summary = compute_naive_penalty(actual, is_peak, regime)

    # Rolling mean baseline
    rolling_forecast = pd.Series(actual).rolling(
        INTERVALS_PER_DAY, min_periods=1
    ).mean().shift(1).bfill().values
    rolling_summary = compute_penalty_summary(rolling_forecast, actual, is_peak, regime)

    # Penalty reduction vs baselines
    naive_reduction = (
        (naive_summary["total_penalty"] - model_summary["total_penalty"])
        / naive_summary["total_penalty"] * 100
        if naive_summary["total_penalty"] > 0 else 0
    )
    rolling_reduction = (
        (rolling_summary["total_penalty"] - model_summary["total_penalty"])
        / rolling_summary["total_penalty"] * 100
        if rolling_summary["total_penalty"] > 0 else 0
    )

    return {
        "model": model_summary,
        "naive_baseline": naive_summary,
        "rolling_baseline": rolling_summary,
        "penalty_reduction_vs_naive_pct": naive_reduction,
        "penalty_reduction_vs_rolling_pct": rolling_reduction,
    }


def format_backtest_report(results: Dict) -> str:
    """Format backtest results as a readable report."""
    m = results["model"]
    n = results["naive_baseline"]

    report = []
    report.append("=" * 70)
    report.append("GRIDSHIELD 2 – HISTORICAL BACKTEST PERFORMANCE (OUTPUT B)")
    report.append("=" * 70)
    report.append("")
    report.append("ABT REGIME PERFORMANCE SUMMARY")
    report.append("-------------------------------------------------------------")

    report.append("┌─────────────────────────────────────────────────────────────┐")
    report.append("│              MANDATORY OUTPUT B (SLDC COMPLIANCE)           │")
    report.append("├─────────────────────────────────┬───────────────────────────┤")
    report.append(f"│ Total Financial Exposure (ABT)  │ ₹{m['total_penalty']:>20,.2f} │")
    report.append(f"│ Peak-Hour Exposure (18h-22h)    │ ₹{m['peak_penalty']:>20,.2f} │")
    report.append(f"│ Off-Peak Exposure               │ ₹{m['offpeak_penalty']:>20,.2f} │")
    report.append(f"│ Forecast Bias                   │ {m['forecast_bias_pct']:>19.2f}% │")
    report.append(f"│ 95th Pctl Abs Deviation (kW)    │ {m['p95_abs_deviation_kw']:>20.2f} │")
    report.append(f"│ >5% Underforecast Violations    │ {m['reliability_violations']:>20d} │")
    report.append(f"│ Financial Cap                   │ ₹{m['financial_cap']:>20,.2f} │")
    report.append(f"│ Cap Compliant                   │ {'✓ YES' if m['cap_compliant'] else '✗ NO':>25s} │")
    report.append(f"│ Bias In Bounds [-2%, +3%]       │ {'✓ YES' if m['bias_in_bounds'] else '✗ NO':>25s} │")
    report.append(f"│ MAPE                            │ {m['mape_pct']:>19.2f}% │")
    report.append("├─────────────────────────────────┴───────────────────────────┤")
    report.append("│         FINANCIAL EXPOSURE REDUCTION VS BASELINES           │")
    report.append("├─────────────────────────────────┬───────────────────────────┤")
    report.append(f"│ Naive Baseline Exposure         │ ₹{n['total_penalty']:>20,.2f} │")
    report.append(f"│ Reduction vs Naive              │ {results['penalty_reduction_vs_naive_pct']:>19.2f}% │")
    report.append(f"│ Reduction vs Rolling Mean       │ {results['penalty_reduction_vs_rolling_pct']:>19.2f}% │")
    report.append("└─────────────────────────────────┴───────────────────────────┘")

    return "\n".join(report)


def compute_interval_penalties(
    forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    timestamps: pd.DatetimeIndex,
    regime: str = "tiered",
) -> pd.DataFrame:
    """Return per-interval penalty breakdown for dashboard visualization."""
    deviation = compute_deviation(forecast, actual)
    pct_deviation = compute_pct_deviation(forecast, actual)
    penalty = compute_full_penalty(forecast, actual, is_peak, regime)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "forecast": forecast,
        "actual": actual,
        "deviation": deviation,
        "pct_deviation": pct_deviation * 100,
        "penalty": penalty,
        "is_peak": is_peak,
        "cumulative_penalty": np.cumsum(penalty),
    })
    return df
