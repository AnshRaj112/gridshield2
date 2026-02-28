"""
GridShield – Constrained Optimization Engine
Finds optimal forecast bias offset to minimize penalty while
satisfying financial cap, reliability, and bias constraints.
"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
from typing import Tuple
from config import (
    FINANCIAL_CAP, MAX_RELIABILITY_VIOLATIONS,
    UNDERESTIMATION_THRESHOLD, BIAS_LOWER_BOUND, BIAS_UPPER_BOUND,
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE, PEAK_UNDER_MULTIPLIER,
)
from penalty import compute_full_penalty, compute_penalty_summary


def find_optimal_bias(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    regime: str = "tiered",
    bias_range: Tuple[float, float] = (-0.05, 0.10),
    n_points: int = 200,
) -> dict:
    """
    Grid search for the optimal constant bias offset b*
    that minimizes penalty subject to constraints.

    Adjusted forecast = base_forecast × (1 + b)

    Returns optimal bias, penalty, and constraint satisfaction.
    """
    best_result = None
    best_penalty = float("inf")
    results = []

    for b in np.linspace(bias_range[0], bias_range[1], n_points):
        adjusted = base_forecast * (1 + b)

        summary = compute_penalty_summary(adjusted, actual, is_peak, regime)

        # Check constraints
        cap_ok = summary["total_penalty"] <= FINANCIAL_CAP
        rel_ok = summary["reliability_violations"] <= MAX_RELIABILITY_VIOLATIONS
        bias_val = summary["forecast_bias_pct"] / 100.0
        bias_ok = BIAS_LOWER_BOUND <= bias_val <= BIAS_UPPER_BOUND

        all_ok = cap_ok and rel_ok and bias_ok

        result = {
            "bias_offset": b,
            "total_penalty": summary["total_penalty"],
            "reliability_violations": summary["reliability_violations"],
            "forecast_bias_pct": summary["forecast_bias_pct"],
            "cap_compliant": cap_ok,
            "reliability_compliant": rel_ok,
            "bias_compliant": bias_ok,
            "all_constraints_met": all_ok,
            "summary": summary,
        }
        results.append(result)

        if all_ok and summary["total_penalty"] < best_penalty:
            best_penalty = summary["total_penalty"]
            best_result = result

    # If no feasible solution, return the one with minimum penalty
    if best_result is None:
        best_result = min(results, key=lambda r: r["total_penalty"])
        best_result["warning"] = "No solution satisfies all constraints"

    return best_result, results


def optimize_quantile_buffer(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    regime: str = "tiered",
) -> dict:
    """
    Find optimal quantile buffer: scale the forecast unevenly for
    peak vs off-peak to minimize total penalty under constraints.

    Adjusts:
    - Peak forecast = base × (1 + b_peak)
    - Off-peak forecast = base × (1 + b_offpeak)
    """
    best_penalty = float("inf")
    best_config = None

    for b_peak in np.linspace(-0.02, 0.08, 50):
        for b_offpeak in np.linspace(-0.03, 0.05, 40):
            adjusted = base_forecast.copy()
            peak_mask = is_peak == 1
            adjusted[peak_mask] *= (1 + b_peak)
            adjusted[~peak_mask] *= (1 + b_offpeak)

            summary = compute_penalty_summary(adjusted, actual, is_peak, regime)

            bias_val = summary["forecast_bias_pct"] / 100.0
            cap_ok = summary["total_penalty"] <= FINANCIAL_CAP
            rel_ok = summary["reliability_violations"] <= MAX_RELIABILITY_VIOLATIONS
            bias_ok = BIAS_LOWER_BOUND <= bias_val <= BIAS_UPPER_BOUND

            if cap_ok and rel_ok and bias_ok:
                if summary["total_penalty"] < best_penalty:
                    best_penalty = summary["total_penalty"]
                    best_config = {
                        "peak_buffer": b_peak,
                        "offpeak_buffer": b_offpeak,
                        "total_penalty": summary["total_penalty"],
                        "reliability_violations": summary["reliability_violations"],
                        "forecast_bias_pct": summary["forecast_bias_pct"],
                        "summary": summary,
                        "all_constraints_met": True,
                    }

    if best_config is None:
        # Fallback: find minimum penalty regardless of constraints
        best_config = {
            "peak_buffer": 0.03,
            "offpeak_buffer": 0.0,
            "warning": "No fully feasible solution found",
        }

    return best_config


def pareto_frontier(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    regime: str = "tiered",
    n_points: int = 100,
) -> list:
    """
    Generate Pareto frontier: penalty vs reliability violations.
    Traces the set of efficient points where improving one objective
    requires worsening the other.
    """
    points = []

    for b in np.linspace(-0.03, 0.10, n_points):
        adjusted = base_forecast * (1 + b)
        summary = compute_penalty_summary(adjusted, actual, is_peak, regime)
        points.append({
            "bias_offset": b,
            "total_penalty": summary["total_penalty"],
            "reliability_violations": summary["reliability_violations"],
            "forecast_bias_pct": summary["forecast_bias_pct"],
            "mape_pct": summary["mape_pct"],
        })

    # Extract Pareto-optimal points
    pareto = []
    for p in sorted(points, key=lambda x: x["total_penalty"]):
        if not pareto or p["reliability_violations"] < pareto[-1]["reliability_violations"]:
            pareto.append(p)

    return pareto, points
