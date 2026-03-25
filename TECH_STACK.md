# GridShield2 – Tech Stack

This document provides a comprehensive overview of every technology, library, framework, and tool used in the **GridShield2** project.

---

## Table of Contents

1. [Programming Language](#1-programming-language)
2. [Core Data Libraries](#2-core-data-libraries)
3. [Machine Learning & Forecasting](#3-machine-learning--forecasting)
4. [Optimisation & Scientific Computing](#4-optimisation--scientific-computing)
5. [Web Dashboard & Visualisation](#5-web-dashboard--visualisation)
6. [Data Storage & File Formats](#6-data-storage--file-formats)
7. [Configuration & Project Structure](#7-configuration--project-structure)
8. [Documentation & Reporting](#8-documentation--reporting)
9. [Version Control](#9-version-control)

---

## 1. Programming Language

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.x | Sole language for all pipeline, modelling, and dashboard code |

Python is used end-to-end: data ingestion, feature engineering, model training, optimisation, risk simulation, and the interactive web dashboard.

---

## 2. Core Data Libraries

| Library | Version | Purpose |
|---|---|---|
| **NumPy** | 2.3.4 | Numerical arrays, vectorised arithmetic, Monte Carlo sample generation |
| **Pandas** | 2.3.3 | DataFrame-based data manipulation, SAS datetime parsing, CSV I/O |

### NumPy
Used for fast numerical operations across all modules — generating random samples for Monte Carlo simulation (`risk_engine.py`), array slicing in feature engineering (`features.py`), and penalty computations (`penalty.py`).

### Pandas
The primary data structure throughout the pipeline. All CSV datasets (training, test, events) are loaded and manipulated as Pandas DataFrames. Time-series indexing, resampling, and rolling-window calculations are handled via Pandas APIs.

---

## 3. Machine Learning & Forecasting

| Library | Version | Purpose |
|---|---|---|
| **LightGBM** | 4.6.0 | Primary forecasting engine — Quantile Gradient Boosted Regression Trees |
| **XGBoost** | 3.2.0 | Secondary residual-correction component of the Hybrid Ensemble |
| **scikit-learn** | 1.7.2 | Model-selection utilities, cross-validation helpers, metrics |

### LightGBM
The core forecasting model. A separate `LGBMRegressor` is trained per forecast horizon (t+1, t+96, t+192, t+288, 5-day, 15-day) as a **Quantile Regressor** (`objective='quantile'`). Key hyper-parameters (set in `config.py`):
- `n_estimators = 1000`
- `learning_rate = 0.05`
- `num_leaves = 63`
- L1 regularisation (`reg_alpha`) and L2 regularisation (`reg_lambda`)
- Optimal quantile: τ = 0.667 (off-peak hours), τ = 0.750 (peak hours 18:00–22:00)

### XGBoost
Acts as a **residual corrector** on top of the LightGBM base prediction. Configuration:
- `n_estimators = 500`
- `max_depth = 6`
- Similar L1/L2 regularisation penalties

### scikit-learn
Provides `TimeSeriesSplit`, `cross_val_score`, and error metrics used in the time-aware expanding-window validation (`validation.py`) and backtest module (`backtest.py`).

---

## 4. Optimisation & Scientific Computing

| Library | Version | Purpose |
|---|---|---|
| **SciPy** | 1.16.2 | Constrained continuous optimisation (`scipy.optimize.minimize`) |

### SciPy
Used in `optimizer.py` to solve the **constrained bias-offset optimisation** problem:
- **Objective**: Minimise total expected penalty under the ABT tariff structure
- **Constraints**:
  - Reliability: ≤ 3 intervals with > 5 % underestimation per day
  - Bias bounds: −2 % ≤ mean bias ≤ +3 %
  - Financial cap: Total penalty ≤ ₹50,000
- Pareto-frontier search across bias-offset and quantile-buffer space

---

## 5. Web Dashboard & Visualisation

| Library | Version | Purpose |
|---|---|---|
| **Streamlit** | 1.54.0 | Interactive web application framework |
| **Plotly** | 6.3.1 | Interactive, publication-quality charts and risk heatmaps |

### Streamlit
Powers the entire interactive dashboard (`dashboard.py`, ~988 lines). Key capabilities exposed through the UI:
- Live penalty recalculation with adjustable parameters
- Train vs. test dataset statistical comparison
- Strategy selection and bias-offset sliders
- Executive-level summary metrics

### Plotly
All visualisations inside the Streamlit dashboard are rendered with Plotly:
- Time-series forecast vs. actual load plots
- Penalty breakdown bar charts (peak / off-peak)
- Monte Carlo distribution histograms
- Risk heatmaps (sensitivity matrix)
- Pareto-frontier scatter plots

---

## 6. Data Storage & File Formats

| Format | Tool/Library | Usage |
|---|---|---|
| **CSV** | Pandas (`read_csv` / `to_csv`) | All input datasets and per-interval penalty outputs |
| **JSON** | Python `json` stdlib | Pipeline state persistence (`pipeline_state.json`), dataset comparison |
| **Markdown** | Plain text | Documentation outputs (`docs/*.md`) |
| **LaTeX** | Plain text | Technical appendix (`docs/Technical_Appendix.tex`) |
| **PDF** | External (pre-existing) | Case study guidelines (`Case 02 - Stage 2 Guidelines.pdf`) |

### Input Datasets
| File | Records | Description |
|---|---|---|
| `train_data/Electric_Load_Data_Train.csv` | 283,391 | 15-min electricity load (Apr 2013 – Apr 2021) |
| `train_data/External_Factor_Data_Train.csv` | — | Weather & external features (14 MB) |
| `train_data/Events_Data.csv` | — | Holiday, lockdown, and event flags |
| `test_data/Electric_Load_Data_Test.csv` | 2,977 | Out-of-time test set (May–Jun 2021) |
| `test_data/External_Factor_Data_Test.csv` | — | Corresponding weather features |

---

## 7. Configuration & Project Structure

| File | Purpose |
|---|---|
| `config.py` | Central configuration: paths, penalty rates, model hyper-parameters, simulation settings |
| `requirements.txt` | Pinned Python dependency list |
| `.gitignore` | Git exclusion rules |

### Module Overview

| Module | Lines | Role |
|---|---|---|
| `main.py` | 587 | End-to-end pipeline orchestrator |
| `config.py` | 114 | Centralised configuration |
| `models.py` | 306 | QuantileLGBM, XGBoost residual corrector, HybridEnsemble |
| `features.py` | 169 | 81-feature engineering pipeline |
| `penalty.py` | 215 | Tiered penalty computation engine |
| `validation.py` | 104 | Expanding-window cross-validation, leakage prevention |
| `backtest.py` | 124 | Historical backtesting metrics and naive baselines |
| `optimizer.py` | 294 | Constrained bias-offset optimisation, Pareto frontier |
| `risk_engine.py` | 203 | Monte Carlo simulation, VaR / CVaR, scenario analysis |
| `risk_strategy.py` | 229 | Risk strategy reporting and bias positioning |
| `utils.py` | 172 | SAS datetime parsing, data loading and merging |
| `dashboard.py` | 988 | Streamlit interactive web dashboard |

---

## 8. Documentation & Reporting

| Technology | Usage |
|---|---|
| **Markdown** | System documentation, forecast model write-ups, backtest reports, risk strategy summaries |
| **LaTeX** | Mathematical formulations in the Technical Appendix |
| **JSON** | Full pipeline-state serialisation (models, metrics, split indices) |

Generated documentation is written to the `docs/` directory by `main.py` at the end of each pipeline run:
- `GRIDSHIELD_DOCUMENTATION.md` – Complete system documentation
- `Technical_Appendix.md` / `.tex` – Mathematical proofs and formulations
- `output_a_forecast_model.md` – Modelling architecture and feature engineering rationale
- `output_b_backtest_metrics.md` – Historical penalty metrics and baseline comparisons
- `output_c_risk_strategy.md` – Optimal bias positioning and strategy recommendations

---

## 9. Version Control

| Technology | Usage |
|---|---|
| **Git** | Source code version control |
| **GitHub** | Remote repository hosting and collaboration |

---

## Summary

GridShield2 is a **pure-Python** project with no separate frontend framework or database layer. The full technology footprint is:

```
Python 3.x
├── Data layer       → NumPy, Pandas (CSV files)
├── ML layer         → LightGBM, XGBoost, scikit-learn
├── Optimisation     → SciPy
├── Visualisation    → Plotly, Streamlit
└── Docs/Reporting   → Markdown, LaTeX, JSON
```
