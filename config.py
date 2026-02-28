"""
GridShield – Central Configuration
All penalty rates, thresholds, horizons, and system constants.
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "train_data")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

# Train files
TRAIN_LOAD_FILE = os.path.join(TRAIN_DATA_DIR, "Electric_Load_Data_Train.csv")
TRAIN_WEATHER_FILE = os.path.join(TRAIN_DATA_DIR, "External_Factor_Data_Train.csv")
EVENTS_FILE = os.path.join(TRAIN_DATA_DIR, "Events_Data.csv")

# Test files
TEST_LOAD_FILE = os.path.join(TEST_DATA_DIR, "Electric_Load_Data_Test.csv")
TEST_WEATHER_FILE = os.path.join(TEST_DATA_DIR, "External_Factor_Data_Test.csv")

# ─── Datetime ─────────────────────────────────────────────────────────────────
DATETIME_FORMAT = "%d%b%Y:%H:%M:%S"  # SAS-style: 01APR2013:00:15:00
FREQ = "15min"
INTERVALS_PER_DAY = 96
INTERVALS_PER_HOUR = 4

# ─── Peak Hours (As per Stage 1 & 2 Guidelines) ──────────────────────────────
PEAK_START_HOUR = 18   # 18:00 (6:00 PM)
PEAK_END_HOUR = 22     # 22:00 (10:00 PM) (exclusive end, last peak interval is 21:45)

# ─── Penalty Regimes ─────────────────────────────────────────────────────────
# Cost structure (₹ per unit deviation)
PENALTY_OVER_BASE = 2.0         # overforecast cost (Forecast > Actual)
PENALTY_UNDER_BASE = 4.0        # off-peak underforecast cost
PENALTY_UNDER_PEAK = 6.0        # peak-hour underforecast cost (Stage 2 Shock)
PENALTY_UNDER_STAGE2 = 6.0      # alias for Stage 2 shock rate

# Tiered penalty structure (based on absolute % deviation)
TIERED_PENALTIES = [
    (0.03, 2.0),   # 0–3%  deviation → ₹2
    (0.07, 6.0),   # 3–7%  deviation → ₹6
    (1e9,  12.0),  # >7%   deviation → ₹12
]

# Dynamic cost-based quantiles: q = C_under / (C_under + C_over)
def compute_quantile(c_under: float, c_over: float) -> float:
    return c_under / (c_under + c_over)

QUANTILE_OFFPEAK = compute_quantile(PENALTY_UNDER_BASE, PENALTY_OVER_BASE)  # 4/(4+2) = 0.667
QUANTILE_PEAK = compute_quantile(PENALTY_UNDER_PEAK, PENALTY_OVER_BASE)     # 6/(6+2) = 0.750

# Legacy alias (for penalty.py backward compat — derived, not hardcoded)
PEAK_UNDER_MULTIPLIER = PENALTY_UNDER_PEAK / PENALTY_UNDER_BASE  # 6/4 = 1.5

# ─── Stage 3 Binding Constraints ─────────────────────────────────────────────
FINANCIAL_CAP = 50000.0         # ₹ total penalty cap
MAX_RELIABILITY_VIOLATIONS = 3  # max intervals with >5% underestimation
UNDERESTIMATION_THRESHOLD = 0.05  # 5% threshold for reliability check
BIAS_LOWER_BOUND = -0.02       # -2%
BIAS_UPPER_BOUND = 0.03        # +3%

# ─── Forecast Horizons ───────────────────────────────────────────────────────
HORIZONS = {
    "t+1":   1,
    "t+96":  96,     # 1 day ahead
    "t+192": 192,    # 2 days ahead
    "t+288": 288,    # 3 days ahead
    "5-day": 480,    # 5 × 96
    "15-day": 1440,  # 15 × 96
}

# ─── COVID Regime ─────────────────────────────────────────────────────────────
COVID_LOCKDOWN_START = "2020-03-25"
COVID_LOCKDOWN_END = "2020-06-30"
COVID_EXTENDED_END = "2020-12-31"

# ─── Model Parameters ────────────────────────────────────────────────────────
QUANTILES = [0.50, QUANTILE_OFFPEAK, QUANTILE_PEAK]
DEFAULT_QUANTILE = QUANTILE_OFFPEAK  # off-peak quantile as default

LGBM_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": 0,
    "n_jobs": -1,
}

# ─── Monte Carlo ──────────────────────────────────────────────────────────────
MC_SIMULATIONS = 1000
MC_TEMP_NOISE_STD = 2.0       # °C noise on temperature
MC_LOAD_NOISE_PCT = 0.03      # 3% Gaussian noise on load

# ─── Validation ───────────────────────────────────────────────────────────────
VALIDATION_TRAIN_YEARS = 5     # expanding window minimum
VALIDATION_TEST_DAYS = 30      # size of each test fold
