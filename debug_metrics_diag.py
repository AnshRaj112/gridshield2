
import pandas as pd
import numpy as np
import os
import json
from optimizer import compute_risk_transparency_outputs
from risk_engine import monte_carlo_penalty_simulation

DOCS_DIR = "docs"
interval_path = os.path.join(DOCS_DIR, "interval_penalties.csv")

if os.path.exists(interval_path):
    df = pd.read_csv(interval_path)
    print(f"Columns: {df.columns.tolist()}")
    
    if "base_forecast" in df.columns:
        base = df["base_forecast"].values
        actual = df["actual"].values
        is_peak = df["is_peak"].values
        cap = 50000.0
        
        rt = compute_risk_transparency_outputs(base, actual, is_peak, financial_cap=cap, regime="tiered")
        print("\nRisk Transparency:")
        print(f"  P95 KW: {rt.get('p95_abs_deviation_kw')}")
        print(f"  P95 Pct: {rt.get('p95_abs_dev_pct')}")
        print(f"  Vol Impact: {rt.get('peak_volatility_financial_impact')}")
        print(f"  Worst 5: {len(rt.get('worst_5_intervals', []))}")
        
        mc = monte_carlo_penalty_simulation(base, actual, is_peak, financial_cap=cap, regime="tiered", n_simulations=10)
        print("\nMonte Carlo (10 sims):")
        print(f"  Mean: {mc.get('mean_penalty')}")
        print(f"  Linear Mean: {mc.get('linear_mean')}")
        print(f"  Jump Mean: {mc.get('jump_mean')}")
    else:
        print("Error: base_forecast not in CSV")
else:
    print("Error: CSV not found")
