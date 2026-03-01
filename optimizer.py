"""
GridShield – Constrained Optimization Engine
Finds optimal forecast bias offset to minimize penalty while
satisfying financial cap, reliability, and bias constraints.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize
from typing import Tuple
from config import (
    MAX_RELIABILITY_VIOLATIONS,
    UNDERESTIMATION_THRESHOLD, BIAS_LOWER_BOUND, BIAS_UPPER_BOUND,
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE, PEAK_UNDER_MULTIPLIER,
)
from penalty import compute_full_penalty, compute_penalty_summary

# Stage 3 Binding Board Directive - Buffering limit
MAX_AVG_UPLIFT = 0.03   # average forecast uplift relative to unbiased baseline <= 3%


def find_optimal_bias(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    financial_cap: float,
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

        summary = compute_penalty_summary(adjusted, actual, is_peak, financial_cap, regime)

        # Check constraints
        cap_ok = summary["total_penalty"] <= financial_cap
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
    financial_cap: float,
    regime: str = "tiered",
) -> dict:
    """
    Constrained optimization under binding Board-Level Directives:
    - Expected Penalty <= Cap
    - VaR (95%) <= Cap
    - Violations <= 3
    - Bias in [-2%, +3%]

    Uses a Lagrangian barrier method: infeasible solutions are scaled by
    massive penalty multipliers rendering them unselectable unless no
    feasible options exist.
    """
    from risk_engine import monte_carlo_penalty_simulation

    best_score = float("inf")
    best_config = None
    
    # Track the absolute minimum penalty achieved (ignoring constraints)
    min_achievable_penalty = float("inf")

    for b_peak in np.linspace(0.12, -0.05, 20):
        for b_offpeak in np.linspace(-0.02, 0.08, 20):
            adjusted = base_forecast.copy()
            peak_mask = is_peak == 1
            adjusted[peak_mask] *= (1 + b_peak)
            adjusted[~peak_mask] *= (1 + b_offpeak)

            summary = compute_penalty_summary(adjusted, actual, is_peak, financial_cap, regime)
            expected_penalty = summary["total_penalty"]
            
            # Heuristic pruning: If expected penalty is already 25% over cap, skip expensive MC
            if expected_penalty > financial_cap * 1.25:
                var_95 = expected_penalty * 1.5 # dummy conservative estimate for pruning
                fast_mc = {"var_95": var_95}
            else:
                # Fast Monte Carlo to evaluate VaR (fewer paths for speed during grid search)
                fast_mc = monte_carlo_penalty_simulation(
                    adjusted, actual, is_peak, financial_cap, regime, n_simulations=100
                )
                var_95 = fast_mc["var_95"]

            violations = summary["reliability_violations"]
            bias_val = summary["forecast_bias_pct"] / 100.0
            
            if expected_penalty < min_achievable_penalty:
                min_achievable_penalty = expected_penalty

            # --- Lagrangian Multipliers (Barrier Penalties) ---
            penalty_score = expected_penalty
            constraint_violations = 0
            
            if expected_penalty > financial_cap:
                penalty_score += (expected_penalty - financial_cap) * 1000
                constraint_violations += 1
                
            if var_95 > financial_cap:
                penalty_score += (var_95 - financial_cap) * 1000
                constraint_violations += 1
                
            if violations > MAX_RELIABILITY_VIOLATIONS:
                penalty_score += (violations - MAX_RELIABILITY_VIOLATIONS) * 500000
                constraint_violations += 1
                
            if not (BIAS_LOWER_BOUND <= bias_val <= BIAS_UPPER_BOUND):
                bias_breach = max(0, bias_val - BIAS_UPPER_BOUND) + max(0, BIAS_LOWER_BOUND - bias_val)
                penalty_score += bias_breach * 10000000
                constraint_violations += 1

            # Stage 3 Constraint 4: Average uplift relative to unbiased base <= 3%
            avg_uplift = float(np.mean((adjusted - base_forecast) / np.where(base_forecast == 0, 1, base_forecast)))
            if avg_uplift > MAX_AVG_UPLIFT:
                uplift_breach = avg_uplift - MAX_AVG_UPLIFT
                penalty_score += uplift_breach * 5000000
                constraint_violations += 1

            # Tier 3 Intervals Soft Constraint (<= 15%)
            if regime in ["tiered", "stage2_shock"]:
                # Calculate percentage deviation for each interval
                # Avoid division by zero for actual values
                adj_pct_dev = np.abs((adjusted - actual) / np.where(actual == 0, 1, actual))
                tier3_mask = adj_pct_dev > 0.10 # 10% deviation threshold for Tier 3
                tier3_proportion = np.mean(tier3_mask)
                if tier3_proportion > 0.15: # 15% maximum proportion of Tier 3 intervals
                    penalty_score += (tier3_proportion - 0.15) * 1000000
                    constraint_violations += 1 # Consider this a constraint violation

            if penalty_score < best_score:
                best_score = penalty_score
                is_feasible = (constraint_violations == 0)
                
                best_config = {
                    "peak_buffer": b_peak,
                    "offpeak_buffer": b_offpeak,
                    "total_penalty": expected_penalty,
                    "var_95": var_95,
                    "reliability_violations": violations,
                    "forecast_bias_pct": summary["forecast_bias_pct"],
                    "cap_utilization_pct": (expected_penalty / financial_cap) * 100 if financial_cap > 0 else float('inf'),
                    "summary": summary,
                    "is_feasible": is_feasible,
                    "constraint_violations": constraint_violations,
                    "lagrangian_score": penalty_score
                }

    if not best_config["is_feasible"]:
        best_config["warning"] = "STRUCTURAL INFEASIBILITY: No solution satisfies all constraints."
        best_config["minimum_required_cap"] = min_achievable_penalty * 1.05  # Add 5% margin

    return best_config


def compute_risk_transparency_outputs(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    financial_cap: float,
    timestamps: pd.DatetimeIndex = None,
    regime: str = "tiered",
) -> dict:
    """
    Stage 3 Mandatory Risk Transparency Reporting:
    - Total penalty
    - Peak-hour penalty
    - Off-peak penalty
    - 95th percentile absolute deviation
    - Worst 5 deviation intervals
    - Financial impact of peak-hour volatility
    """
    summary = compute_penalty_summary(base_forecast, actual, is_peak, financial_cap, regime)

    act_safe = np.where(actual == 0, np.nan, actual)
    abs_dev = np.abs((base_forecast - actual) / act_safe) * 100  # in %

    p95_dev_pct = float(np.nanpercentile(abs_dev, 95))
    p95_dev_kw = float(np.percentile(np.abs(base_forecast - actual), 95))
    
    interval_penalties = compute_full_penalty(base_forecast, actual, is_peak, regime=regime)

    valid_idx = np.where(~np.isnan(abs_dev))[0]
    worst5_idx = valid_idx[np.argsort(abs_dev[valid_idx])[-5:][::-1]]
    worst5 = [
        {
            "interval": int(i),
            "timestamp": str(timestamps[i]) if timestamps is not None else None,
            "abs_dev_pct": float(abs_dev[i]),
            "forecast": float(base_forecast[i]),
            "actual": float(actual[i]),
            "is_peak": bool(is_peak[i]),
            "penalty_impact": float(interval_penalties[i]),
        }
        for i in worst5_idx
    ]

    peak_mask = is_peak == 1
    peak_forecast = base_forecast[peak_mask]
    peak_actual = actual[peak_mask]
    peak_act_safe = np.where(peak_actual == 0, 1, peak_actual)
    peak_vol = float(np.std((peak_forecast - peak_actual) / peak_act_safe))
    peak_vol_impact = peak_vol * summary["total_penalty"]

    return {
        "total_penalty":              summary["total_penalty"],
        "peak_penalty":               summary.get("peak_penalty", 0),
        "offpeak_penalty":            summary.get("offpeak_penalty", 0),
        "p95_abs_dev_pct":            p95_dev_pct,
        "p95_abs_deviation_kw":       p95_dev_kw,
        "worst5_intervals":           worst5,
        "worst_5_intervals":          worst5,
        "peak_vol_financial_impact":  peak_vol_impact,
        "peak_volatility_financial_impact": peak_vol_impact,
        "reliability_violations":     summary["reliability_violations"],
        "mape_pct":                   summary.get("mape_pct", 0),
        "forecast_bias_pct":          summary.get("forecast_bias_pct", 0),
    }


def pareto_frontier(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    financial_cap: float,
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
        summary = compute_penalty_summary(adjusted, actual, is_peak, financial_cap, regime)
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
