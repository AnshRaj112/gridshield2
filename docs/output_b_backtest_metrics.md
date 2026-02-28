# GRIDSHIELD – HISTORICAL BACKTEST METRICS (OUTPUT B)

## Stage 1 (Historical Training)

### 1-3. Forecasts, Deviation Penalties, and Quantities
| SLDC Compliance Metric | Actual Financial Exposure |
|--------|-------|
| Total Financial Exposure (ABT) | ₹93,806.72 |
| Peak-Hour Exposure (18h-22h) | ₹14,981.42 |
| Off-Peak Exposure | ₹78,825.30 |
| Structured Bias Offset | 0.17% |
| 95th Pctl Grid Draw Deviation | 32.20 kW |
| >5% Underestimation Violations | 0 |
| Financial Cap Exceedance (₹50,000) | ✗ Cap Breached |
| Regulatory Bias Bounds [-2%, +3%] | ✓ Strict Compliance |
| Volumetric Error (MAPE) | 1.07% |

**Standard Forecasting (Naive) Exposure**: ₹1,519,301.24
**Financial Edge vs Naive Submission**: 93.8%
**Financial Edge vs Rolling Strategy**: 97.3%

## Stage 2 (Out-of-Time Test + Shock)

### 1-3. Forecasts, Deviation Penalties, and Quantities
| SLDC Compliance Metric | Actual Financial Exposure |
|--------|-------|
| Total Financial Exposure (ABT) | ₹116,463.71 |
| Peak-Hour Exposure (18h-22h) | ₹13,001.21 |
| Off-Peak Exposure | ₹103,462.51 |
| Structured Bias Offset | 0.17% |
| 95th Pctl Grid Draw Deviation | 32.20 kW |
| >5% Underestimation Violations | 0 |
| Financial Cap Exceedance (₹50,000) | ✗ Cap Breached |
| Regulatory Bias Bounds [-2%, +3%] | ✓ Strict Compliance |
| Volumetric Error (MAPE) | 1.07% |

**Standard Forecasting (Naive) Exposure**: ₹531,009.48
**Financial Edge vs Naive Submission**: 78.1%
**Financial Edge vs Rolling Strategy**: 88.2%

## 4. Compare: Historical vs Test Penalty Exposure

- **Stage 1 (Historical) Penalty Exposure**: ₹93,806.72
- **Stage 2 (Test+Shock) Penalty Exposure**: ₹116,463.71
- **Regime Shift Impact**: The transition from historical Base Tariff to Stage 2 
  elevated volatility + Peak-Hour Escalation represents a 24.2% 
  increase in absolute financial exposure.

## 5. Recalibration of Forecasting and Buffering Strategy

To mitigate the Stage 2 Peak-Hour Penalty Escalation (Underforecast Cost: ₹4 → ₹6), 
the following dynamic recalibration was executed systemically:

1. **Dynamic Quantile Recalibration**: The system continuously recalculates 
   the theoretically optimal target quantile `τ* = C_under / (C_under + C_over)`. 
   During standard periods, `τ* = 0.667`. During Stage 2 Peak Hours, the strategy 
   automatically shifts `τ*` to `0.750`, natively creating a robust safety buffer 
   without manual overrides.
2. **Bias Positioning**: We optimized the intentional forecast offset within 
   the bounds [-2%, +3%] using simulated annealing over the exact asymmetrical loss 
   function to find the mathematical minimum of financial penalty.
