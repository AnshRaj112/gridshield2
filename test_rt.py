from dashboard import load_data, run_dynamic_rt
from optimizer import compute_risk_transparency_outputs
import pandas as pd

state, test_intervals, train_intervals, comparison = load_data()
test_iv = test_intervals.copy()
has_base = "base_forecast" in test_intervals.columns
print("has_base", has_base)

dyn_rt = run_dynamic_rt(test_iv["forecast"].values, test_iv["actual"].values, test_iv["is_peak"].values, test_iv["timestamp"].values, 50000.0)
print(dyn_rt['peak_penalty'])
print(dyn_rt['offpeak_penalty'])

