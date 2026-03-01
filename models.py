"""
GridShield – Multi-Horizon Forecasting Models
LightGBM Quantile, XGBoost Residual Correction, Hybrid Ensemble.
Multi-horizon support with horizon-aware feature gating.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Optional, Tuple
from config import (
    LGBM_PARAMS, XGB_PARAMS, QUANTILES, DEFAULT_QUANTILE, HORIZONS,
    PENALTY_UNDER_BASE, PENALTY_OVER_BASE, PENALTY_UNDER_PEAK
)
from features import gate_features_for_horizon
from validation import get_feature_columns


class QuantileLGBM:
    """LightGBM with native quantile regression."""

    def __init__(self, quantile: float = DEFAULT_QUANTILE, **kwargs):
        self.quantile = quantile
        params = {**LGBM_PARAMS, **kwargs}
        self.model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=quantile,
            **params,
        )
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series,
            eval_set: Optional[List[Tuple]] = None,
            sample_weight: Optional[np.ndarray] = None):
        self.feature_names_ = list(X.columns)
        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        self.model.fit(X, y, sample_weight=sample_weight, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_


class ResidualXGB:
    """XGBoost model trained on residuals from the primary model."""

    def __init__(self, **kwargs):
        params = {**XGB_PARAMS, **kwargs}
        self.model = XGBRegressor(
            objective="reg:squarederror",
            **params,
        )

    def fit(self, X: pd.DataFrame, residuals: pd.Series,
            eval_set: Optional[List[Tuple]] = None,
            sample_weight: Optional[np.ndarray] = None):
        fit_params = {}
        if eval_set:
            fit_params["eval_set"] = eval_set
            fit_params["verbose"] = False
            self.model.set_params(early_stopping_rounds=50)
        self.model.fit(X, residuals, sample_weight=sample_weight, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class HybridEnsemble:
    """
    Hybrid Ensemble: LightGBM Quantile + XGBoost Residual Correction.
    Final forecast = LightGBM prediction + XGBoost residual correction.
    """

    def __init__(self, quantile: float = DEFAULT_QUANTILE,
                 residual_weight: float = 0.5):
        self.quantile = quantile
        self.residual_weight = residual_weight
        self.lgbm = QuantileLGBM(quantile=quantile)
        self.xgb_residual = ResidualXGB()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            sample_weight: Optional[np.ndarray] = None):
        # Step 1: Train LightGBM quantile model
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        self.lgbm.fit(X_train, y_train, eval_set=eval_set, sample_weight=sample_weight)

        # Step 2: Compute residuals
        lgbm_pred_train = self.lgbm.predict(X_train)
        residuals = y_train.values - lgbm_pred_train

        # Step 3: Train XGBoost on residuals
        xgb_eval = None
        if X_val is not None and y_val is not None:
            lgbm_pred_val = self.lgbm.predict(X_val)
            val_residuals = y_val.values - lgbm_pred_val
            xgb_eval = [(X_val, val_residuals)]
        self.xgb_residual.fit(X_train, pd.Series(residuals), eval_set=xgb_eval, sample_weight=sample_weight)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        lgbm_pred = self.lgbm.predict(X)
        xgb_correction = self.xgb_residual.predict(X)
        return lgbm_pred + self.residual_weight * xgb_correction

    @property
    def feature_importances_(self):
        return self.lgbm.feature_importances_


class MultiHorizonForecaster:
    """
    Multi-horizon forecaster: separate models for each horizon.
    Each model has horizon-aware feature gating to prevent leakage.
    """

    def __init__(self, quantile: float = DEFAULT_QUANTILE,
                 horizons: Dict[str, int] = None):
        self.quantile = quantile
        self.horizons = horizons or HORIZONS
        self.models: Dict[str, HybridEnsemble] = {}
        self.feature_sets: Dict[str, List[str]] = {}
        self.metrics: Dict[str, dict] = {}
        self.peak_buffers: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, target: str = "LOAD",
            test_frac: float = 0.15, financial_cap: float = 50000.0):
        """
        Train a model for each horizon.
        Automatically gates features to prevent leakage.
        """
        for name, horizon in self.horizons.items():
            print(f"\n{'='*60}")
            print(f"Training horizon: {name} (h={horizon})")
            print(f"{'='*60}")

            # Horizon-specific quantile tuning with decay weighting beyond 24 hours
            tau = self.quantile
            if horizon > 96:
                tau = 0.5 + (self.quantile - 0.5) * np.exp(-(horizon - 96) / 288)
                print(f"  Decayed target quantile: {tau:.4f}")

            # Gate features for this horizon
            df_h = gate_features_for_horizon(df.copy(), horizon)

            # Create target: shift by horizon
            df_h["target"] = df_h[target].shift(-horizon)
            if "is_peak" in df_h.columns:
                df_h["target_is_peak"] = df_h["is_peak"].shift(-horizon)
            df_h = df_h.dropna(subset=["target"])

            # Get feature columns
            feature_cols = get_feature_columns(df_h, target="target")
            feature_cols = [c for c in feature_cols if c not in [target, "target_is_peak"]]
            self.feature_sets[name] = feature_cols

            X = df_h[feature_cols].copy()
            y = df_h["target"].copy()

            # Handle remaining NaNs
            X = X.ffill().fillna(0)

            # Train/val split (chronological)
            split_idx = int(len(X) * (1 - test_frac))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Retrain using higher weights on recent out-of-time data to address distribution shift
            # Exponentially increasing weights: oldest=e^-2 (~0.13), newest=e^0 (1.0)
            sample_weight = np.exp(np.linspace(-2, 0, len(X_train)))
            sample_weight = sample_weight / sample_weight.mean()

            # Separate short-, medium-, and long-term models
            if horizon <= 96:
                # Short-term: standard complexity
                model = HybridEnsemble(quantile=tau)
            else:
                # Medium/Long-term: shallower trees, stronger regularization to stabilize multi-horizon error growth
                model = HybridEnsemble(quantile=tau)
                model.lgbm.model.set_params(max_depth=4, num_leaves=15, reg_lambda=5.0)
                model.xgb_residual.model.set_params(max_depth=3, reg_lambda=5.0)

            # Train model
            model.fit(X_train, y_train, X_val, y_val, sample_weight=sample_weight)
            self.models[name] = model

            # Compute metrics on validation set
            pred = model.predict(X_val)
            
            # Peak-hour adaptive quantile recalibration
            if "target_is_peak" in df_h.columns:
                target_is_peak_val = df_h["target_is_peak"].iloc[split_idx:].values
                peak_residuals = y_val.values[target_is_peak_val == 1] - pred[target_is_peak_val == 1]
                if len(peak_residuals) > 0:
                    # e.g. 75th percentile of residuals during peak hours to serve as buffer
                    peak_buffer = np.percentile(peak_residuals, 75)
                    self.peak_buffers[name] = peak_buffer
                    # Apply buffer for validation metrics computation
                    pred[target_is_peak_val == 1] += peak_buffer
                else:
                    self.peak_buffers[name] = 0.0
            else:
                self.peak_buffers[name] = 0.0

            # Compute penalty summary using integrated ABT financial metrics rather than RMSE alone
            from penalty import compute_penalty_summary
            is_peak_val = df_h["target_is_peak"].iloc[split_idx:].values if "target_is_peak" in df_h.columns else np.zeros(len(y_val))
            val_summary = compute_penalty_summary(pred, y_val.values, is_peak_val, financial_cap, regime="tiered")

            self.metrics[name] = {
                "financial_penalty": val_summary["total_penalty"],
                "reliability_violations": val_summary["reliability_violations"],
                "bias_pct": val_summary["forecast_bias_pct"],
                "mape": val_summary["mape_pct"],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_features": len(feature_cols),
                "peak_buffer": self.peak_buffers[name],
            }
            print(f"  Penalty: ₹{self.metrics[name]['financial_penalty']:,.2f}")
            print(f"  Reliability Violations: {self.metrics[name]['reliability_violations']}")
            print(f"  MAPE: {self.metrics[name]['mape']:.2f}%")
            print(f"  Bias: {self.metrics[name]['bias_pct']:+.2f}%")
            print(f"  Adaptive Peak Buffer applied: {self.peak_buffers[name]:.2f} kW")

    def predict(self, df: pd.DataFrame, horizon_name: str,
                target: str = "LOAD") -> np.ndarray:
        """Generate predictions for a specific horizon."""
        if horizon_name not in self.models:
            raise ValueError(f"No model trained for horizon: {horizon_name}")
        model = self.models[horizon_name]
        horizon = self.horizons[horizon_name]
        df_h = gate_features_for_horizon(df.copy(), horizon)
        feature_cols = self.feature_sets[horizon_name]
        available_cols = [c for c in feature_cols if c in df_h.columns]
        X = df_h[available_cols].copy()
        # Add missing columns as zeros
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[feature_cols]
        X = X.ffill().fillna(0)
        
        pred = model.predict(X)
        
        # Apply peak-hour adaptive recalibration based on horizon-shifted timestamp 
        if self.peak_buffers.get(horizon_name, 0.0) != 0.0:
            if isinstance(X.index, pd.DatetimeIndex):
                target_times = X.index + pd.Timedelta(minutes=15 * horizon)
                is_peak_pred = (target_times.hour >= 18) & (target_times.hour < 22)
                pred[is_peak_pred] += self.peak_buffers[horizon_name]
                
        return pred

    def get_quantile_predictions(self, df: pd.DataFrame,
                                 horizon_name: str,
                                 quantiles: List[float] = None,
                                 target: str = "LOAD") -> Dict[float, np.ndarray]:
        """
        Generate predictions at multiple quantiles for confidence intervals.
        Retrains lightweight models at each quantile.
        """
        if quantiles is None:
            quantiles = QUANTILES

        horizon = self.horizons[horizon_name]
        df_h = gate_features_for_horizon(df.copy(), horizon)
        df_h["target"] = df_h[target].shift(-horizon)
        df_h = df_h.dropna(subset=["target"])

        feature_cols = self.feature_sets.get(horizon_name)
        if feature_cols is None:
            feature_cols = get_feature_columns(df_h, target="target")
            feature_cols = [c for c in feature_cols if c != target]

        X = df_h[feature_cols].copy().ffill().fillna(0)
        y = df_h["target"]

        split_idx = int(len(X) * 0.85)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        results = {}
        for q in quantiles:
            q_model = QuantileLGBM(quantile=q)
            q_model.fit(X_train, y_train)
            results[q] = q_model.predict(X_test)

        return results, X_test.index



