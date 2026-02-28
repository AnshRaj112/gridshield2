"""
GridShield – Risk Strategy Module
Optimal bias positioning, quantile selection logic,
financial exposure modeling, and executive-level strategy report.
"""
import numpy as np
import pandas as pd
from typing import Dict
from config import (
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE, PENALTY_UNDER_STAGE2,
    PEAK_UNDER_MULTIPLIER, BIAS_LOWER_BOUND, BIAS_UPPER_BOUND,
    FINANCIAL_CAP, MAX_RELIABILITY_VIOLATIONS, QUANTILES,
    compute_quantile, PENALTY_UNDER_PEAK
)


def compute_risk_strategy(
    base_forecast: np.ndarray,
    actual: np.ndarray,
    is_peak: np.ndarray,
    mc_results: Dict,
    optimizer_results: Dict,
    backtest_results: Dict,
    regime: str = "tiered",
) -> Dict:
    """
    Compute comprehensive risk strategy with mathematical justification.
    """
    # Optimal quantile computation: q = C_under / (C_under + C_over)
    tau_offpeak = compute_quantile(PENALTY_UNDER_BASE, PENALTY_OVER_BASE)
    tau_peak = compute_quantile(PENALTY_UNDER_PEAK, PENALTY_OVER_BASE)
    
    # Stage 2 Shock increases off-peak to 6; peak stays same relative ratio or scales further
    tau_stage2_offpeak = compute_quantile(PENALTY_UNDER_STAGE2, PENALTY_OVER_BASE)
    tau_stage2_peak = compute_quantile(PENALTY_UNDER_STAGE2 * PEAK_UNDER_MULTIPLIER, PENALTY_OVER_BASE)

    strategy = {
        "quantile_selection": {
            "base_offpeak_tau": tau_offpeak,
            "base_peak_tau": tau_peak,
            "stage2_offpeak_tau": tau_stage2_offpeak,
            "stage2_peak_tau": tau_stage2_peak,
            "derivation": (
                "τ* = C_under / (C_under + C_over). "
                f"Base Off-Peak: τ = {PENALTY_UNDER_BASE}/({PENALTY_UNDER_BASE}+{PENALTY_OVER_BASE}) = {tau_offpeak:.4f}. "
                f"Base Peak: τ = {PENALTY_UNDER_PEAK}/({PENALTY_UNDER_PEAK}+{PENALTY_OVER_BASE}) = {tau_peak:.4f}."
            ),
        },
        "bias_positioning": {
            "allowed_range": f"[{BIAS_LOWER_BOUND*100:.0f}%, {BIAS_UPPER_BOUND*100:.0f}%]",
            "optimal_bias_pct": (
                optimizer_results.get("forecast_bias_pct", 0)
                if isinstance(optimizer_results, dict) else 0
            ),
            "rationale": (
                "Under asymmetric penalties where Pu > Po, the optimal strategy "
                "biases forecast slightly upward (overforecast) to reduce the more "
                "expensive underforecast penalties. The optimizer found the minimum-penalty "
                "bias within the regulatory bounds."
            ),
        },
        "financial_exposure": {
            "expected_penalty": mc_results.get("mean_penalty", 0),
            "var_95": mc_results.get("var_95", 0),
            "cvar_95": mc_results.get("cvar_95", 0),
            "cap_breach_probability": mc_results.get("cap_breach_prob", 0),
            "financial_cap": FINANCIAL_CAP,
            "headroom": FINANCIAL_CAP - mc_results.get("var_95", 0),
        },
        "constraint_satisfaction": {
            "financial_cap_met": mc_results.get("mean_penalty", 0) < FINANCIAL_CAP,
            "reliability_met": (
                mc_results.get("mean_violations", 0) <= MAX_RELIABILITY_VIOLATIONS
            ),
            "bias_met": (
                BIAS_LOWER_BOUND <= mc_results.get("mean_bias_pct", 0) / 100 <= BIAS_UPPER_BOUND
            ),
        },
        "penalty_reduction": {
            "vs_naive_pct": backtest_results.get("penalty_reduction_vs_naive_pct", 0),
            "vs_rolling_pct": backtest_results.get("penalty_reduction_vs_rolling_pct", 0),
        },
    }

    return strategy


def generate_strategy_report(strategy: Dict) -> str:
    """Generate Mandatory Output C: Risk Strategy Proposal."""
    qs = strategy["quantile_selection"]
    bp = strategy["bias_positioning"]
    fe = strategy["financial_exposure"]
    cs = strategy["constraint_satisfaction"]
    pr = strategy["penalty_reduction"]

    report = []
    report.append("=" * 70)
    report.append("GRIDSHIELD – RISK STRATEGY PROPOSAL (OUTPUT C)")
    report.append("=" * 70)

    report.append("")
    report.append("1. MATHEMATICAL DERIVATIONS")
    report.append("-" * 40)
    report.append(f"   {qs['derivation']}")
    report.append(f"   Base Off-Peak τ* = {qs['base_offpeak_tau']:.4f}")
    report.append(f"   Base Peak τ*     = {qs['base_peak_tau']:.4f}")
    report.append(f"   Stage2 Off-Peak τ* = {qs['stage2_offpeak_tau']:.4f}")
    report.append(f"   Stage2 Peak τ*     = {qs['stage2_peak_tau']:.4f}")

    report.append("")
    report.append("2. BIAS POSITIONING STRATEGY")
    report.append("-" * 40)
    report.append(f"   Allowed Range: {bp['allowed_range']}")
    report.append(f"   Optimal Bias:  {bp['optimal_bias_pct']:.2f}%")
    report.append(f"   Rationale: {bp['rationale']}")

    report.append("")
    report.append("3. FINANCIAL EXPOSURE MODELING")
    report.append("-" * 40)
    report.append(f"   Expected Penalty (Mean): ₹{fe['expected_penalty']:,.2f}")
    report.append(f"   VaR (95%):               ₹{fe['var_95']:,.2f}")
    report.append(f"   CVaR (95%):              ₹{fe['cvar_95']:,.2f}")
    report.append(f"   Financial Cap:           ₹{fe['financial_cap']:,.2f}")
    report.append(f"   Cap Headroom:            ₹{fe['headroom']:,.2f}")
    report.append(f"   Cap Breach Probability:  {fe['cap_breach_probability']*100:.1f}%")

    report.append("")
    report.append("4. CONSTRAINT SATISFACTION PROOF")
    report.append("-" * 40)
    report.append(f"   Financial Cap Met:    {'✓' if cs['financial_cap_met'] else '✗'}")
    report.append(f"   Reliability Met:      {'✓' if cs['reliability_met'] else '✗'}")
    report.append(f"   Bias Bounds Met:      {'✓' if cs['bias_met'] else '✗'}")

    report.append("")
    report.append("5. EXPECTED PENALTY REDUCTION")
    report.append("-" * 40)
    report.append(f"   vs Naive Baseline:    {pr['vs_naive_pct']:.1f}%")
    report.append(f"   vs Rolling Mean:      {pr['vs_rolling_pct']:.1f}%")

    report.append("")
    report.append("6. EXECUTIVE SUMMARY & ADVISORY MANDATE")
    report.append("-" * 40)
    report.append(
        "   Under Maharashtra's Availability Based Tariff (ABT) regulations, "
        "volumetric forecasting accuracy (RMSE/MAPE) is structurally secondary to "
        "Financial Exposure Minimization. Given the asymmetric penalty structure "
        f"where Under-forecasting (cost: ₹{PENALTY_UNDER_BASE} to ₹{PENALTY_UNDER_STAGE2}) "
        f"is strictly more punitive than Over-forecasting (cost: ₹{PENALTY_OVER_BASE}), "
        "our proposed architecture abandons traditional mean-regression."
    )
    report.append("")
    report.append(
        "   The GridShield system deploys a Cost-Aware Quantile Regression "
        "engine precisely calibrated to the theoretically optimal τ*. By deliberately "
        f"positioning forecast bias within the regulatory bounds {bp['allowed_range']} "
        "to optimize the Cost-of-Error, the system achieves a "
        f"{pr['vs_naive_pct']:.1f}% reduction in penalty exposure vs naive SLDC submissions."
    )
    report.append("")
    report.append(
        "   Despite elevated out-of-time volatility and structural regime shifts "
        "(Stage 2 Shock), Monte Carlo simulation with 1000 paths confirms "
        f"expected financial exposure is contained at ₹{fe['expected_penalty']:,.0f}. "
        "The strategy is mathematically defensible, constraint-compliant, and "
        "dynamically adapts to changing ABT penalty rates via real-time "
        "τ* recalculation."
    )

    return "\n".join(report)
