# GRIDSHIELD 2 – FORECAST RISK ADVISORY (OUTPUT A)

## 1. Executive Modeling Architecture
The system deploys a **Cost-Aware Hybrid Ensemble** designed specifically for ABT 
regulatory compliance. Traditional RMSE-minimizing models (like standard OLS or LightGBM) 
fail under Maharashtra's asymmetric deviation tariff mechanism. Instead, GridShield 
optimizes for **Financial Exposure Minimization** using Quantile Regression targeted 
at the mathematically derived optimal τ*.

- **Base Engine**: LightGBM Quantile Regressor (calibrated to τ*)
- **Residual Correction**: XGBoost Regressor (correcting non-linear error distributions)
- **Horizon Strategy**: Distinct models trained per forecast horizon to eliminate temporal leakage.

## 2. Feature Engineering Logic
Features are dynamically gated by horizon to prevent future data leakage (SLDC compliance).
- **Autoregressive Lags**: Load at t-96, t-192, etc., providing baseline cyclicality.
- **Rolling Statistics**: 24h mean/std to capture near-term volatility trends.
- **Calendar/Time**: Sine/cosine encoding of Hour, Day of Week, and Month for circular continuity.
- **Regulatory Flags**: Binary indicators for Peak Hours (18:00–22:00) where exposure is elevated.

## 3. Structural Break & Volatility Handling
The model ingests Out-of-Time Test Data characterized by elevated volatility and amplified 
peak-hour variability (Stage 2 Shock). Tree-based ensembles intrinsically partition structural 
breaks without the rigidity of traditional ARIMA. The shift from Stage 1 to Stage 2 is handled 
not by retraining, but by dynamically shifting the target quantile τ* upward as Under-forecast 
costs scale from ₹4 to ₹6.

## 4. Time-Aware Validation & Leakage Controls
- Chronological train/validation splits (no random cross-validation).
- Target variables shifted explicitly by the forecast horizon length.
- Feature gating strictly drops any lag shorter than the target horizon.

### Multi-Horizon Support
Separate models trained for each forecast horizon:
- **t+1**: Avg Penalty=₹461,896.08, MAPE=0.45%, Bias=+0.18%, Reliability Violations=1
- **t+96**: Avg Penalty=₹28,464,179.07, MAPE=6.46%, Bias=+5.44%, Reliability Violations=133
- **t+288**: Avg Penalty=₹61,231,999.29, MAPE=12.02%, Bias=+10.86%, Reliability Violations=465

| Category | Features |
|----------|----------|
| Lag | lag_1, lag_2, lag_4, lag_96, lag_192, lag_672 |
| Rolling Stats | rolling_mean/std at 4/12/24/96 intervals, min/max/range |
| Ramp Rate | diff(1), diff(4), diff(96) |
| Cyclical | sin/cos of hour, day-of-week, month, day-of-year |
| Fourier | 4-harmonic sin/cos for daily, weekly, yearly periods |
| Weather | THI, temp², temp×peak, cool×peak, cool×hour, heatwave flag |
| Events | holiday flag, holiday×hour, weekend×peak |
| COVID | lockdown flag, extended COVID flag, lockdown×hour |
| Regime | CUSUM-based structural break detector |

## 3. Seasonality Handling

- **Fourier Harmonics**: sin/cos pairs at T=96 (daily), T=672 (weekly),
  T≈35064 (yearly) with 4 harmonics each (24 features total).
- **Cyclical Encoding**: Maps periodic features to continuous sin/cos
  space, preserving adjacency (23:00 → 00:00 is close).

## 4. Structural Break Handling

- **CUSUM Detector**: Cumulative sum control chart on normalized
  residuals. Flags regime shifts when CUSUM exceeds ±3σ threshold.
- **COVID Regime**: Explicit binary encoding for lockdown and
  extended COVID periods. Interaction terms capture
  lockdown-specific hourly demand patterns.
- **Piecewise Training**: data split into Pre-COVID / COVID /
  Post-COVID regimes for regime-specific evaluation.

## 5. Validation Methodology

- **Expanding Window**: Train on [0..t], test on [t+1..t+h],
  growing the training set by 30 days each iteration.
- **Rolling Cross-Validation**: Fixed 365-day training window
  with 1-day gap to prevent boundary leakage.
- **Strict Chronological**: No shuffling, no future data in training.

## 6. Leakage Controls

- **Horizon-Aware Gating**: For horizon h, all lag features
  with lag < h are automatically removed from the feature matrix.
- **Rolling Stats Shift**: All rolling statistics use `.shift(1)`
  to exclude the current interval from computation.
- **Train/Test Verification**: Automated assertion that
  max(train_index) < min(test_index).

## 7. Regime-Shift Robustness

- CUSUM detection flags regime boundaries automatically
- COVID-specific features allow model to learn pandemic behavior
- Quantile regression is inherently robust to outliers
- XGBoost residual correction adapts to persistent drift
- Monte Carlo simulation stress-tests under perturbed conditions

## 8. Post-Update Justification (Stage 2)

1. **Quantile Modification**: The baseline optimal quantile of `0.667` (from `4/(4+2)`)
   has been systematically overridden during Stage 2 Peak Hours (18:00–22:00) to `0.750`.
   This mathematical recalibration explicitly addresses the regulatory penalty hike
   for under-forecasting (₹4 → ₹6).
2. **Bias Modification**: An intentional systematic bias offset of `-0.08%` to `+0.17%`
   is applied post-prediction. This is not a model error, but a calculated regulatory
   strategy to position our predictions safely within the SLDC permitted bounds
   `[-2%, +3%]`, fundamentally prioritizing financial exposure over zero-mean symmetry.