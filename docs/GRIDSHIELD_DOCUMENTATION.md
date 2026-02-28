# GridShield – Complete System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Regulatory Scenario](#regulatory-scenario)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Modeling Architecture](#modeling-architecture)
6. [Penalty Engine](#penalty-engine)
7. [Peak-Hour Multiplier – Explained](#peak-hour-multiplier)
8. [Constraint Optimization](#constraint-optimization)
9. [Risk Engine](#risk-engine)
10. [Dashboard](#dashboard)
11. [Module Reference](#module-reference)
12. [How to Run](#how-to-run)

---

## System Overview

GridShield is a production-grade, regulatory-compliant electricity demand forecasting and penalty optimization system. It trains on **283,391 rows** of 15-minute interval load data (April 2013 – April 2021) and tests on **2,977 rows** (May – June 2021).

**Core Pipeline (7 Stages):**
```
Data Loading → Feature Engineering → Model Training → Backtesting →
Penalty Optimization → Monte Carlo Risk Simulation → Documentation
```

---

## Regulatory Scenario

The system operates under the **"GridShield – The Regulatory Squeeze"** scenario:

### Stage 2 Shock
- Underforecast penalty increases: **₹4 → ₹6** during peak hours

### Tiered Penalty Structure
| Deviation Range | Penalty Rate |
|----------------|-------------|
| 0–3% | ₹2 per kW |
| 3–7% | ₹6 per kW |
| >7% | ₹12 per kW |

### Peak-Hour Multiplier
- Underforecast penalty is **×2** during peak hours (18:00–22:00)

### Stage 3 Binding Constraints
| Constraint | Value |
|-----------|-------|
| Financial Cap | ₹50,000 |
| Max Reliability Violations | ≤3 intervals with >5% underestimation |
| Bias Bounds | [-2%, +3%] |

---

## Data Pipeline

### Data Sources
| File | Location | Records | Description |
|------|----------|---------|-------------|
| `Electric_Load_Data_Train.csv` | `train_data/` | 283,391 | 15-min load (kW), Apr 2013–Apr 2021 |
| `External_Factor_Data_Train.csv` | `train_data/` | 283,391 | Temperature, humidity, heat index, wind, rain |
| `Events_Data.csv` | `train_data/` | 645 | Holidays, lockdowns, cyclones |
| `Electric_Load_Data_Test.csv` | `test_data/` | 2,977 | 15-min load (kW), May–Jun 2021 |
| `External_Factor_Data_Test.csv` | `test_data/` | 2,977 | Weather for test period |

### Processing Steps
1. **Parse SAS datetimes** (`01APR2013:00:15:00` → standard timestamps)
2. **Reindex** to fill missing 15-min intervals (linear interpolation, max 4 gaps)
3. **Merge** load + weather on index; events on date
4. **Combine** train and test before feature engineering (ensures lags compute correctly across boundaries)
5. **Split** back into train/test after features are computed

---

## Feature Engineering

81 features are engineered across 9 categories:

| Category | Features | Count |
|----------|----------|-------|
| **Lag** | lag_1, lag_2, lag_4, lag_96, lag_192, lag_672 | 6 |
| **Rolling Stats** | mean/std at 4/12/24/96 intervals, min/max/range | 13 |
| **Ramp Rate** | diff(1), diff(4), diff(96) | 3 |
| **Cyclical** | sin/cos of hour, day-of-week, month, day-of-year | 8 |
| **Fourier** | 4-harmonic sin/cos for daily, weekly, yearly | 24 |
| **Weather** | THI, temp², temp×peak, cool×peak, heatwave flag | 7 |
| **Events** | holiday, holiday×hour, weekend×peak, lockdown×hour | 4 |
| **COVID** | lockdown flag, extended COVID flag | 2 |
| **Regime** | CUSUM-based structural break detector | 1 |

### Horizon-Aware Feature Gating
For horizon `h`, all lag features with `lag < h` are automatically removed to prevent data leakage. For horizons > 4 intervals, rolling features are also dropped.

---

## Modeling Architecture

### Primary: LightGBM Quantile Regression
- **Objective**: `quantile` at τ* = Pu / (Pu + Po)
- **Why quantile?** The regulatory penalty is asymmetric — underforecasting costs more than overforecasting. Instead of predicting the mean (which minimizes squared error), we predict the τ*-th quantile, which directly minimizes expected penalty.
- **Config**: 1000 estimators, 63 leaves, 0.05 learning rate, early stopping

### Secondary: XGBoost Residual Correction
- **Objective**: `reg:squarederror` on LightGBM residuals
- **Purpose**: Corrects persistent algorithmic bias
- **Blending**: `Final = LightGBM_pred + 0.5 × XGB_correction`

### Multi-Horizon Forecasting
Separate models are trained for each horizon:

| Horizon | Steps Ahead | Meaning |
|---------|------------|---------|
| t+1 | 1 | Next 15 minutes |
| t+96 | 96 | Next day (24h) |
| t+288 | 288 | 3 days ahead |

### Achieved Performance
| Horizon | MAE (kW) | RMSE (kW) | MAPE (%) | Bias (%) |
|---------|----------|-----------|----------|----------|
| t+1 | 4.9 | 9.5 | **0.47** | +0.17 |
| t+96 | 73.3 | 104.2 | 6.85 | +5.95 |
| t+288 | 142.0 | 190.9 | 13.17 | +12.31 |

---

## Penalty Engine

### How Penalties Are Computed

```
Deviation_t = Forecast_t – Actual_t
```

- **Negative deviation** (Forecast < Actual) = **Underforecast** → more expensive
- **Positive deviation** (Forecast > Actual) = **Overforecast** → less expensive

### Three Penalty Regimes

#### 1. Linear
```
if underforecast: penalty = |deviation| × ₹4
if overforecast:  penalty = |deviation| × ₹2
```

#### 2. Tiered (Default)
Penalty rate depends on the **percentage deviation**:
```
|deviation| / actual ≤ 3%  → ₹2/kW
|deviation| / actual ≤ 7%  → ₹6/kW
|deviation| / actual > 7%  → ₹12/kW
```

#### 3. Stage 2 Shock
Strictly linear penalty structure.
Underforecast (Actual > Forecast): ₹4/kW (Off-peak), ₹6/kW (Peak)
Overforecast (Forecast > Actual): ₹2/kW

---

## Peak-Hour Multiplier

### What is it?

The **Peak-Hour Multiplier** is a regulatory mechanism that **doubles the underforecast penalty during peak demand hours** (18:00 – 22:00).

### Why does it exist?

During peak hours, electricity demand is highest and supply is most constrained. If a utility underforecasts during these hours:
- **Grid instability**: the grid operator must procure emergency power at premium rates
- **Blackout risk**: insufficient generation capacity can cause rolling blackouts
- **Economic damage**: industrial production halts, hospitals lose power, etc.

To discourage underforecasting during these critical hours, regulators **multiply the underforecast penalty by 2×**. This creates a strong financial incentive to overforecast slightly during peak hours, which is safer for grid stability.

### How it works in the model

```
if (Forecast < Actual) AND (hour is 18:00–22:00):
    penalty = base_penalty × 2.0    ← peak multiplier applied
else:
    penalty = base_penalty
```

### Impact on optimal quantile (τ*)

The peak multiplier changes the optimal forecast quantile:

| Period | τ* Formula | τ* Value |
|--------|-----------|----------|
| Off-peak | 4/(4+2) | 0.667 |
| **Peak** | **(4×2)/(4×2+2)** | **0.800** |
| Stage 2 Off-peak | 6/(6+2) | 0.750 |
| Stage 2 Peak | (6×2)/(6×2+2) | 0.857 |

Higher τ* during peak = model forecasts higher = more overforecasting = fewer underforecast penalties.

### In the dashboard

The **Peak-Hour Multiplier slider** (default 2.0) lets you simulate different regulatory scenarios. Setting it to 3.0 would triple underforecast penalties during peak hours, and the total penalty and all charts instantly recalculate.

---

## Constraint Optimization

### Three Binding Constraints
1. **Financial Cap**: Total penalty ≤ ₹50,000
2. **Reliability**: ≤ 3 intervals with > 5% underestimation
3. **Bias Bounds**: Forecast bias must be within [-2%, +3%]

### Optimization Approach
- **Grid search** over bias offsets from -3% to +5%
- At each offset, recalculate penalties and check all constraints
- Select the offset that minimizes total penalty while satisfying constraints
- **Quantile buffer**: Separate peak/off-peak buffers to fine-tune

### Pareto Frontier
Generates 100 bias points and plots total penalty vs reliability violations, showing the optimal trade-off curve.

---

## Risk Engine

### Monte Carlo Simulation
- **1,000 simulation paths** with Gaussian noise:
  - Temperature: ±2°C standard deviation
  - Load: ±3% Gaussian perturbation
- Computes: **VaR (95%)**, **CVaR (95%)**, cap breach probability

### Scenario Simulation
| Scenario | Impact |
|----------|--------|
| Cyclone | +15% load spike |
| Heatwave | +10% AC demand |
| Penalty Hike ×1.5 | All rates × 1.5 |
| Extreme (Cyclone + Hike) | Both combined |

### Sensitivity Analysis
Matrix of penalties across different rate multipliers and bias offsets.

---

## Dashboard

4-page interactive Streamlit dashboard at `http://localhost:8504`:

### Page 1: Executive Summary
- KPI cards (Total Penalty, MAPE, Violations, Bias, VaR)
- Financial Cap utilization gauge
- Bias tracking gauge
- Model performance table by horizon

### Page 2: Forecast Analysis
- Forecast vs Actual with 95% confidence intervals
- Daily penalty breakdown (bar chart)
- Risk heatmap (hour × day-of-week)
- Cumulative penalty over time

### Page 3: Train vs Test Comparison
- Side-by-side dataset statistics (records, date range, load distribution)
- Performance comparison table (MAPE, bias, penalties for both sets)
- Load distribution histograms
- Box plots
- Forecast vs Actual for both train and test
- Error distribution comparison
- Daily MAPE comparison
- Hourly load profile comparison

### Page 4: Risk & Scenarios
- Pareto frontier (penalty vs violations)
- Scenario impact bar chart
- Monte Carlo percentile distribution

### Interactive Penalty Controls (All Pages)
| Control | Effect |
|---------|--------|
| **Under-forecast Penalty (₹/kW)** | Changes cost of underforecasting |
| **Over-forecast Penalty (₹/kW)** | Changes cost of overforecasting |
| **Peak-Hour Multiplier** | Scales peak underforecast penalty |
| **Financial Cap (₹)** | Adjusts regulatory cap |
| **Days to Display** | Controls chart date range |

All charts and metrics **recalculate instantly** when controls are adjusted.

---

## Module Reference

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.py` | 110 | All constants, paths, penalty rates, model params |
| `utils.py` | 170 | Data loading, datetime parsing, merging |
| `features.py` | 170 | 81 engineered features, horizon gating |
| `validation.py` | 85 | Expanding window, rolling CV, leakage checks |
| `models.py` | 240 | LightGBM quantile, XGBoost residual, multi-horizon |
| `penalty.py` | 197 | Tiered/linear/shock penalty functions |
| `optimizer.py` | 145 | Constrained optimization, Pareto frontier |
| `backtest.py` | 155 | Historical backtest, baseline comparisons |
| `risk_engine.py` | 150 | Monte Carlo, scenario simulation, sensitivity |
| `risk_strategy.py` | 180 | τ* derivation, executive report generation |
| `main.py` | 470 | Pipeline orchestrator, documentation output |
| `dashboard.py` | 350 | Multi-page Streamlit dashboard |

---

## How to Run

### 1. Run the full pipeline
```bash
/c/Python313/python.exe main.py
```
This trains on the **entire** `train_data/` dataset and tests on the **entire** `test_data/` dataset. Outputs are saved to `docs/`.

### 2. Launch the dashboard
```bash
/c/Python313/python.exe -m streamlit run dashboard.py --server.port 8504
```

### 3. Output files
| File | Description |
|------|-------------|
| `output_a_forecast_model.md` | Model architecture documentation |
| `output_b_backtest_metrics.md` | Backtest metrics under all regimes |
| `output_c_risk_strategy.md` | Risk strategy with mathematical derivations |
| `pipeline_state.json` | Full pipeline state for dashboard |
| `interval_penalties.csv` | Test interval-level penalties |
| `train_interval_penalties.csv` | Train interval-level penalties |
| `dataset_comparison.json` | Train vs test statistics |
| `sensitivity_matrix.csv` | Rate × bias penalty matrix |
