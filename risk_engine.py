"""
GridShield – Monte Carlo Risk Simulation Engine
Generates alternative reality paths to estimate VaR/CVaR
and expected penalty distribution under uncertainty.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from config import (
    MC_SIMULATIONS, MC_TEMP_NOISE_STD, MC_LOAD_NOISE_PCT,
    FINANCIAL_CAP, PEAK_UNDER_MULTIPLIER,
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE,
)
from penalty import compute_penalty_summary, compute_full_penalty


def monte_carlo_penalty_simulation(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    regime: str = "tiered",
    n_simulations: int = MC_SIMULATIONS,
    load_noise_pct: float = MC_LOAD_NOISE_PCT,
    seed: int = 42,
    financial_cap: float = FINANCIAL_CAP,
) -> Dict:
    """
    Monte Carlo simulation: perturb actuals with Gaussian noise
    to estimate the distribution of penalty exposure.

    Returns VaR, CVaR, mean penalty, percentile distribution.
    """
    rng = np.random.RandomState(seed)
    penalties = []
    violations_list = []
    bias_list = []

    for i in range(n_simulations):
        # Perturb actual load with noise
        noise = rng.normal(0, load_noise_pct, size=len(actual))
        simulated_actual = actual * (1 + noise)
        simulated_actual = np.maximum(simulated_actual, 0)

        summary = compute_penalty_summary(
            base_forecast, simulated_actual, is_peak, regime
        )
        penalties.append(summary["total_penalty"])
        violations_list.append(summary["reliability_violations"])
        bias_list.append(summary["forecast_bias_pct"])

    penalties = np.array(penalties)
    violations_arr = np.array(violations_list)
    bias_arr = np.array(bias_list)

    return {
        "mean_penalty": float(np.mean(penalties)),
        "std_penalty": float(np.std(penalties)),
        "median_penalty": float(np.median(penalties)),
        "var_95": float(np.percentile(penalties, 95)),
        "var_99": float(np.percentile(penalties, 99)),
        "cvar_95": float(np.mean(penalties[penalties >= np.percentile(penalties, 95)])),
        "var_95_cap_utilization_pct": float(np.percentile(penalties, 95) / financial_cap * 100),
        "cvar_95_cap_utilization_pct": float(np.mean(penalties[penalties >= np.percentile(penalties, 95)]) / financial_cap * 100),
        "max_penalty": float(np.max(penalties)),
        "min_penalty": float(np.min(penalties)),
        "cap_breach_prob": float(np.mean(penalties > financial_cap)),
        "mean_violations": float(np.mean(violations_arr)),
        "mean_bias_pct": float(np.mean(bias_arr)),
        "penalty_distribution": penalties.tolist(),
        "violation_distribution": violations_arr.tolist(),
        "percentiles": {
            "p5": float(np.percentile(penalties, 5)),
            "p25": float(np.percentile(penalties, 25)),
            "p50": float(np.percentile(penalties, 50)),
            "p75": float(np.percentile(penalties, 75)),
            "p95": float(np.percentile(penalties, 95)),
        },
    }


def scenario_simulation(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    scenario_name: str = "base",
    penalty_rate_multiplier: float = 1.0,
    peak_multiplier_override: float = None,
    cyclone_impact_pct: float = 0.0,
    heatwave_impact_pct: float = 0.0,
    regime: str = "tiered",
) -> Dict:
    """
    Simulate a specific scenario by modifying load and penalty parameters.

    - cyclone_impact_pct: increase actual load by this % (supply disruption)
    - heatwave_impact_pct: increase actual load by this % (AC demand surge)
    - penalty_rate_multiplier: scale all penalty rates
    """
    # Modify actual load for scenario
    modified_actual = actual.copy()
    if cyclone_impact_pct > 0:
        # Cyclone: random spike in load for 20% of intervals
        rng = np.random.RandomState(123)
        spike_mask = rng.random(len(actual)) < 0.2
        modified_actual[spike_mask] *= (1 + cyclone_impact_pct)
    if heatwave_impact_pct > 0:
        # Heatwave: sustained increase during peak hours
        peak_mask = is_peak == 1
        modified_actual[peak_mask] *= (1 + heatwave_impact_pct)

    summary = compute_penalty_summary(
        base_forecast, modified_actual, is_peak, regime
    )

    # Apply penalty rate multiplier
    summary["total_penalty"] *= penalty_rate_multiplier
    summary["peak_penalty"] *= penalty_rate_multiplier
    summary["offpeak_penalty"] *= penalty_rate_multiplier

    return {
        "scenario_name": scenario_name,
        "penalty_rate_multiplier": penalty_rate_multiplier,
        "cyclone_impact_pct": cyclone_impact_pct,
        "heatwave_impact_pct": heatwave_impact_pct,
        **summary,
    }


def sensitivity_analysis(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    regime: str = "tiered",
    financial_cap: float = FINANCIAL_CAP,
) -> pd.DataFrame:
    """
    Sensitivity analysis: how penalty changes across
    penalty rate × peak multiplier × bias offset matrix.
    """
    rows = []

    for rate_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for bias_offset in [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]:
            adjusted = base_forecast * (1 + bias_offset)
            summary = compute_penalty_summary(adjusted, actual, is_peak, regime)
            rows.append({
                "rate_multiplier": rate_mult,
                "bias_offset": bias_offset,
                "total_penalty": summary["total_penalty"] * rate_mult,
                "reliability_violations": summary["reliability_violations"],
                "forecast_bias_pct": summary["forecast_bias_pct"],
                "cap_compliant": summary["total_penalty"] * rate_mult <= financial_cap,
            })

    return pd.DataFrame(rows)
