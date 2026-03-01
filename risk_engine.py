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
    PEAK_UNDER_MULTIPLIER,
     PENALTY_UNDER_BASE, PENALTY_OVER_BASE,
)
from penalty import compute_penalty_summary, compute_decomposed_penalty


def monte_carlo_penalty_simulation(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    financial_cap: float,
    regime: str = "tiered",
    n_simulations: int = MC_SIMULATIONS,
    load_noise_pct: float = MC_LOAD_NOISE_PCT,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo simulation: Controlled Tail Simulation.
    Empirically clip top/bottom 0.5% of noise to prevent extreme distortion.
    Decompose linear vs tier-jump exposure.

    Returns VaR, CVaR, mean penalty, percentile distribution for total, linear, and jump.
    """
    rng = np.random.RandomState(seed)
    
    # 1. Fit conditional residual distribution
    residuals = (actual - base_forecast) / np.maximum(base_forecast, 1)
    
    peak_mask = (is_peak == 1)
    offpeak_mask = ~peak_mask
    
    peak_res_sorted = np.sort(residuals[peak_mask])
    offpeak_res_sorted = np.sort(residuals[offpeak_mask])
    
    if len(peak_res_sorted) > 0:
        p_p005 = np.percentile(peak_res_sorted, 0.5)
        p_p995 = np.percentile(peak_res_sorted, 99.5)
    else:
        p_p005, p_p995 = -0.5, 0.5
        
    if len(offpeak_res_sorted) > 0:
        op_p005 = np.percentile(offpeak_res_sorted, 0.5)
        op_p995 = np.percentile(offpeak_res_sorted, 99.5)
    else:
        op_p005, op_p995 = -0.5, 0.5
        
    totals, linears, jumps = [], [], []
    violations_list, bias_list = [], []

    for i in range(n_simulations):
        # 2. Sample conditionally and apply 99.5% EVT clipping
        sim_res = np.zeros_like(actual, dtype=float)
        
        sim_res[peak_mask] = rng.choice(peak_res_sorted, size=np.sum(peak_mask))
        sim_res[offpeak_mask] = rng.choice(offpeak_res_sorted, size=np.sum(offpeak_mask))
        
        sim_res[peak_mask] = np.clip(sim_res[peak_mask], p_p005, p_p995)
        sim_res[offpeak_mask] = np.clip(sim_res[offpeak_mask], op_p005, op_p995)
        
        simulated_actual = base_forecast * (1 + sim_res)
        simulated_actual = np.maximum(simulated_actual, 0)
        
        lin, jmp, tot = compute_decomposed_penalty(base_forecast, simulated_actual, is_peak, regime)
        
        totals.append(tot)
        linears.append(lin)
        jumps.append(jmp)
        
        # We don't bother recalculating full summary dictionary in 1000 loop to be fast, just quick metric approximation
        # Bias
        actual_sum = np.sum(simulated_actual)
        fc_sum = np.sum(base_forecast)
        bias_pct = ((fc_sum - actual_sum) / actual_sum * 100) if actual_sum > 0 else 0
        bias_list.append(bias_pct)
        # Violations
        under_pct = np.where(simulated_actual > 0, (simulated_actual - base_forecast)/simulated_actual, 0)
        v = np.sum((under_pct > 0.05) & is_peak)
        violations_list.append(v)
        

    totals = np.array(totals)
    linears = np.array(linears)
    jumps = np.array(jumps)
    violations_arr = np.array(violations_list)
    bias_arr = np.array(bias_list)
    
    def get_risk_metrics(arr):
        mean_val = float(np.mean(arr))
        var95 = float(np.percentile(arr, 95))
        cvar95 = float(np.mean(arr[arr >= var95])) if len(arr[arr >= var95]) > 0 else var95
        return mean_val, var95, cvar95

    t_mean, t_var, t_cvar = get_risk_metrics(totals)
    l_mean, l_var, l_cvar = get_risk_metrics(linears)
    j_mean, j_var, j_cvar = get_risk_metrics(jumps)

    return {
        "mean_penalty": t_mean,
        "var_95": t_var,
        "cvar_95": t_cvar,
        
        "linear_mean": l_mean,
        "linear_var_95": l_var,
        "linear_cvar_95": l_cvar,
        
        "jump_mean": j_mean,
        "jump_var_95": j_var,
        "jump_cvar_95": j_cvar,

        "var_95_cap_utilization_pct": float(t_var / financial_cap * 100) if financial_cap > 0 else 0,
        "cvar_95_cap_utilization_pct": float(t_cvar / financial_cap * 100) if financial_cap > 0 else 0,
        "cap_breach_prob": float(np.mean(totals > financial_cap)),
        "mean_violations": float(np.mean(violations_arr)),
        "mean_bias_pct": float(np.mean(bias_arr)),
    }


def scenario_simulation(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    financial_cap: float,
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
        base_forecast, modified_actual, is_peak, financial_cap, regime
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
    financial_cap: float,
    regime: str = "tiered",
) -> pd.DataFrame:
    """
    Sensitivity analysis: how penalty changes across
    penalty rate × peak multiplier × bias offset matrix.
    """
    rows = []

    for rate_mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        for bias_offset in [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]:
            adjusted = base_forecast * (1 + bias_offset)
            summary = compute_penalty_summary(adjusted, actual, is_peak, financial_cap, regime)
            rows.append({
                "rate_multiplier": rate_mult,
                "bias_offset": bias_offset,
                "total_penalty": summary["total_penalty"] * rate_mult,
                "reliability_violations": summary["reliability_violations"],
                "forecast_bias_pct": summary["forecast_bias_pct"],
                "cap_compliant": summary["total_penalty"] * rate_mult <= financial_cap,
            })

    return pd.DataFrame(rows)
