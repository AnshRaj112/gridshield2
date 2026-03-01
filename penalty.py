"""
GridShield – Penalty Computation Engine
Tiered penalty structure, peak multiplier, Stage 2 shock,
and total financial exposure calculation.
"""
import numpy as np
import pandas as pd
from config import (
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE, PENALTY_UNDER_STAGE2,
    TIERED_PENALTIES, PEAK_UNDER_MULTIPLIER,
    PEAK_START_HOUR, PEAK_END_HOUR,
    MAX_RELIABILITY_VIOLATIONS,
    UNDERESTIMATION_THRESHOLD, BIAS_LOWER_BOUND, BIAS_UPPER_BOUND,
)


def compute_deviation(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Deviation_t = Forecast_t - Actual_t
    Negative = underforecast (Forecast < Actual) — more dangerous
    Positive = overforecast  (Forecast > Actual) — wasteful but safer
    """
    return forecast - actual


def compute_pct_deviation(forecast: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Percentage deviation relative to actual."""
    return np.where(actual != 0, (forecast - actual) / actual, 0.0)


def linear_penalty(deviation: np.ndarray, is_peak: np.ndarray,
                    pu: float = PENALTY_UNDER_BASE,
                    po: float = PENALTY_OVER_BASE,
                    peak_multiplier: float = PEAK_UNDER_MULTIPLIER) -> np.ndarray:
    """
    Linear penalty: flat rate for under/overforecast.
    Peak multiplier applied to underforecast during peak hours.
    """
    penalty = np.zeros_like(deviation, dtype=float)
    under_mask = deviation < 0
    over_mask = deviation > 0
    # Underforecast penalty
    penalty[under_mask] = np.abs(deviation[under_mask]) * pu
    # Apply peak multiplier
    peak_under = under_mask & (is_peak == 1)
    penalty[peak_under] *= peak_multiplier
    # Overforecast penalty
    penalty[over_mask] = np.abs(deviation[over_mask]) * po
    return penalty


def tiered_penalty(deviation: np.ndarray, actual: np.ndarray,
                   is_peak: np.ndarray,
                   tiers: list = None,
                   peak_multiplier: float = PEAK_UNDER_MULTIPLIER) -> np.ndarray:
    """
    Tiered penalty based on absolute % deviation.
    0–3%  → ₹2/unit
    3–7%  → ₹6/unit
    >7%   → ₹12/unit
    Peak multiplier applied to underforecast during peak hours.
    """
    if tiers is None:
        tiers = TIERED_PENALTIES

    abs_pct = np.abs(np.where(actual != 0, deviation / actual, 0.0))
    abs_dev = np.abs(deviation)

    penalty = np.zeros_like(deviation, dtype=float)
    for threshold, rate in tiers:
        mask = abs_pct <= threshold
        # Apply rate only to those not yet assigned
        unassigned = penalty == 0
        apply_mask = mask & unassigned
        penalty[apply_mask] = abs_dev[apply_mask] * rate

    # Anything still unassigned gets the last tier
    still_zero = (penalty == 0) & (abs_dev > 0)
    if np.any(still_zero):
        penalty[still_zero] = abs_dev[still_zero] * tiers[-1][1]

    # Apply peak multiplier to underforecast during peak
    under_mask = deviation < 0
    peak_under = under_mask & (is_peak == 1)
    penalty[peak_under] *= peak_multiplier

    return penalty


def stage2_shock_penalty(deviation: np.ndarray, actual: np.ndarray,
                         is_peak: np.ndarray,
                         pu_offpeak: float = PENALTY_UNDER_BASE,
                         pu_peak: float = PENALTY_UNDER_STAGE2,
                         po: float = PENALTY_OVER_BASE) -> np.ndarray:
    """
    Stage 2 Shock: Linear penalty structure.
    Underforecast (Actual > Forecast): Off-peak = ₹4, Peak = ₹6
    Overforecast (Forecast > Actual): ₹2 everywhere
    """
    penalty = np.zeros_like(deviation, dtype=float)
    under_mask = deviation < 0
    over_mask = deviation > 0
    abs_dev = np.abs(deviation)

    # Overforecast
    penalty[over_mask] = abs_dev[over_mask] * po

    # Underforecast off-peak
    under_offpeak = under_mask & (is_peak == 0)
    penalty[under_offpeak] = abs_dev[under_offpeak] * pu_offpeak

    # Underforecast peak
    under_peak = under_mask & (is_peak == 1)
    penalty[under_peak] = abs_dev[under_peak] * pu_peak

    return penalty


def compute_full_penalty(forecast: np.ndarray, actual: np.ndarray,
                         is_peak: np.ndarray,
                         regime: str = "tiered") -> np.ndarray:
    """
    Compute per-interval penalty.
    """
    deviation = compute_deviation(forecast, actual)
    if regime == "linear":
        return linear_penalty(deviation, is_peak)
    elif regime == "tiered":
        return tiered_penalty(deviation, actual, is_peak)
    elif regime == "stage2_shock":
        return stage2_shock_penalty(deviation, actual, is_peak)
    else:
        raise ValueError(f"Unknown penalty regime: {regime}")


def compute_decomposed_penalty(forecast: np.ndarray, actual: np.ndarray,
                               is_peak: np.ndarray,
                               regime: str = "tiered") -> tuple[float, float, float]:
    """
    Returns (linear_penalty, tier_jump_penalty, total_penalty).
    The linear_penalty is what the penalty WOULD be if the base linear rates were used.
    The tier_jump_penalty is the extra penalty incurred due to convex jumps.
    """
    from config import TIERED_PENALTIES, PEAK_UNDER_MULTIPLIER
    deviation = compute_deviation(forecast, actual)
    
    # Base rate for decomposition is the FIRST TIER (lowest rate)
    first_tier_rate = TIERED_PENALTIES[0][1]
    linear_penalties = np.abs(deviation) * first_tier_rate
    
    # Apply peak multiplier if applicable to base decomposition too
    peak_under = (deviation < 0) & (is_peak == 1)
    linear_penalties[peak_under] *= PEAK_UNDER_MULTIPLIER
    
    base_penalty_sum = linear_penalties.sum()
    base_penalty_sum = linear_penalties.sum()
    
    # Actual selected regime
    if regime == "linear":
        total_penalties = linear_penalties
    elif regime == "tiered":
        total_penalties = tiered_penalty(deviation, actual, is_peak)
    elif regime == "stage2_shock":
        total_penalties = stage2_shock_penalty(deviation, actual, is_peak)
    else:
        raise ValueError(f"Unknown penalty regime: {regime}")
        
    total_penalty_sum = total_penalties.sum()
    tier_jump_penalty = total_penalty_sum - base_penalty_sum
    
    # Floor at 0 just in case regimes are cheaper than linear (though mathematically impossible here)
    tier_jump_penalty = max(0, tier_jump_penalty)
    
    return base_penalty_sum, tier_jump_penalty, total_penalty_sum


def compute_penalty_summary(forecast: np.ndarray, actual: np.ndarray,
                            is_peak: np.ndarray,
                            financial_cap: float,
                            regime: str = "tiered") -> dict:
    """
    Compute comprehensive penalty summary for the backtest.
    Returns dict with all mandatory metrics.
    """
    deviation = compute_deviation(forecast, actual)
    pct_deviation = compute_pct_deviation(forecast, actual)
    penalty = compute_full_penalty(forecast, actual, is_peak, regime)

    peak_mask = is_peak == 1
    offpeak_mask = is_peak == 0

    # Reliability: count intervals with >5% underestimation ONLY during peak hours
    underest_pct = np.where(actual != 0, (actual - forecast) / actual, 0.0)
    violations = np.sum((underest_pct > UNDERESTIMATION_THRESHOLD) & peak_mask)

    # Forecast bias
    bias = np.mean(pct_deviation)

    return {
        "total_penalty": float(np.sum(penalty)),
        "peak_penalty": float(np.sum(penalty[peak_mask])),
        "offpeak_penalty": float(np.sum(penalty[offpeak_mask])),
        "forecast_bias_pct": float(bias * 100),
        "p95_abs_deviation_kw": float(np.percentile(np.abs(deviation), 95)),
        "reliability_violations": int(violations),
        "financial_cap": financial_cap,
        "cap_compliant": bool(np.sum(penalty) <= financial_cap),
        "bias_in_bounds": bool(BIAS_LOWER_BOUND <= bias <= BIAS_UPPER_BOUND),
        "mean_abs_deviation_kw": float(np.mean(np.abs(deviation))),
        "max_abs_deviation_kw": float(np.max(np.abs(deviation))),
        "mape_pct": float(np.mean(np.abs(pct_deviation)) * 100),
    }


def compute_naive_penalty(actual: np.ndarray, is_peak: np.ndarray,
                          financial_cap: float,
                          regime: str = "tiered") -> dict:
    """
    Compute penalty for naive baseline (previous-day same-time forecast).
    Uses lag-96 as the naive forecast.
    """
    naive_forecast = np.roll(actual, 96)
    naive_forecast[:96] = actual[:96]  # fill first day
    return compute_penalty_summary(naive_forecast, actual, is_peak, financial_cap, regime)
