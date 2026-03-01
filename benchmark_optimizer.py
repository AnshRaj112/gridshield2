import numpy as np
import pandas as pd
import time
from optimizer import optimize_quantile_buffer

# Mock data
n = 2977
base_fc = np.random.uniform(100, 500, n)
actual = base_fc * np.random.uniform(0.9, 1.1, n)
is_peak = (np.random.uniform(0, 1, n) > 0.8).astype(int)
cap = 50000.0

print("Starting benchmark of optimize_quantile_buffer...")
start = time.time()
result = optimize_quantile_buffer(base_fc, actual, is_peak, cap, regime="tiered")
end = time.time()

print(f"Time taken: {end - start:.2f} seconds")
print(f"Feasible: {result.get('is_feasible')}")
print(f"Best Peak Buffer: {result.get('peak_buffer')}")
