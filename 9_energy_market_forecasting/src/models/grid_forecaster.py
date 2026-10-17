"""
Grid Supply and Demand Time Series Forecaster.
Predicts system load demand, wind generation, solar generation, and resulting net load ramp rates.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class GridForecaster:
    """Forecasting engine for grid electrical demand and renewable supply."""

    def __init__(self):
        self.targets = ("system_demand_mw", "solar_generation_mw", "wind_generation_mw")
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_cols: Dict[str, list] = {}

    def _get_target_features(self, df: pd.DataFrame, target: str) -> list:
        base_feats = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", 
            "is_weekend", "is_peak_hour"
        ]
        if target == "system_demand_mw":
            domain_feats = [
                "temperature_c", "cooling_degree", "heating_degree",
                "system_demand_mw_lag_1h", "system_demand_mw_lag_24h", "system_demand_mw_lag_168h",
                "system_demand_mw_roll_24h_mean", "system_demand_mw_roll_24h_std"
            ]
        elif target == "solar_generation_mw":
            domain_feats = [
                "solar_irradiance_wm2", "cloud_cover_pct", "solar_potential",
                "solar_generation_mw_lag_24h"
            ]
        elif target == "wind_generation_mw":
            domain_feats = [
                "wind_speed_ms", "wind_power_factor",
                "wind_generation_mw_lag_1h", "wind_generation_mw_lag_2h", "wind_generation_mw_lag_24h"
            ]
        else:
            domain_feats = []

        return [f for f in base_feats + domain_feats if f in df.columns]

    def fit(self, df: pd.DataFrame) -> "GridForecaster":
        """Fits gradient boosted regression models for grid demand and generation."""
        for target in self.targets:
            features = self._get_target_features(df, target)
            self.feature_cols[target] = features

            X = df[features]
            y = df[target]

            model = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=32,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbosity=-1
            )
            model.fit(X, y)
            self.models[target] = model
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for demand, solar, wind, and calculates implied net load.
        Net Load (MW) = System Demand - (Solar Generation + Wind Generation)
        """
        out = pd.DataFrame(index=df.index)
        if "timestamp" in df.columns:
            out["timestamp"] = df["timestamp"]

        for target in self.targets:
            model = self.models.get(target)
            if model is None:
                raise ValueError(f"Model for {target} not fitted.")
            features = self.feature_cols[target]
            pred = model.predict(df[features])

            # Non-negativity for power generation
            if target in ("solar_generation_mw", "wind_generation_mw"):
                pred = np.maximum(0.0, pred)
            out[f"{target}_pred"] = np.round(pred, 1)

        # Implied Net Load
        out["net_load_mw_pred"] = np.round(
            out["system_demand_mw_pred"] - (out["solar_generation_mw_pred"] + out["wind_generation_mw_pred"]), 1
        )
        return out

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluates metrics including WAPE (Weighted Absolute Percentage Error)."""
        preds = self.predict(df_test)
        metrics = {}
        eval_targets = list(self.targets) + ["net_load_mw"]

        for target in eval_targets:
            y_true = df_test[target].values
            y_pred = preds[f"{target}_pred"].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            wape = (np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) * 100.0

            metrics[target] = {
                "mae_mw": float(np.round(mae, 1)),
                "rmse_mw": float(np.round(rmse, 1)),
                "wape_pct": float(np.round(wape, 2))
            }
        return metrics
