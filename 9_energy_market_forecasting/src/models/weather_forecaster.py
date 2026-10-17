"""
Weather Time Series Forecaster for Energy Markets.
Forecasts temperature, wind speed, and solar irradiance across multi-hour horizons.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class WeatherForecaster:
    """Multi-target weather forecasting engine using gradient boosted trees."""

    def __init__(self, targets: Tuple[str, ...] = ("temperature_c", "wind_speed_ms", "solar_irradiance_wm2")):
        self.targets = targets
        self.models: Dict[str, lgb.LGBMRegressor] = {}
        self.feature_cols: Dict[str, list] = {}

    def _get_target_features(self, df: pd.DataFrame, target: str) -> list:
        # Base calendar & cyclical features
        time_feats = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
        # Autoregressive lags
        lags = [c for c in df.columns if c.startswith(f"{target}_lag_")]
        # Other cross-weather predictors
        other_weather = []
        if target != "temperature_c" and "temperature_c_lag_1h" in df.columns:
            other_weather.append("temperature_c_lag_1h")
        if target == "solar_irradiance_wm2" and "cloud_cover_pct" in df.columns:
            other_weather.append("cloud_cover_pct")
        return [f for f in time_feats + lags + other_weather if f in df.columns]

    def fit(self, df: pd.DataFrame) -> "WeatherForecaster":
        """Trains weather forecasting models for each meteorological target."""
        for target in self.targets:
            features = self._get_target_features(df, target)
            self.feature_cols[target] = features

            X = df[features]
            y = df[target]

            model = lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.06,
                max_depth=5,
                num_leaves=24,
                subsample=0.85,
                random_state=42,
                verbosity=-1
            )
            model.fit(X, y)
            self.models[target] = model
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates weather predictions for the input feature dataframe."""
        out = pd.DataFrame(index=df.index)
        if "timestamp" in df.columns:
            out["timestamp"] = df["timestamp"]

        for target in self.targets:
            model = self.models.get(target)
            if model is None:
                raise ValueError(f"Model for {target} has not been fitted.")
            features = self.feature_cols[target]
            pred = model.predict(df[features])

            # Apply physical non-negativity bounds
            if target in ("wind_speed_ms", "solar_irradiance_wm2"):
                pred = np.maximum(0.0, pred)
            out[f"{target}_pred"] = np.round(pred, 2)
        return out

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluates MAE and RMSE across all weather targets."""
        preds = self.predict(df_test)
        metrics = {}
        for target in self.targets:
            y_true = df_test[target].values
            y_pred = preds[f"{target}_pred"].values
            metrics[target] = {
                "mae": float(np.round(mean_absolute_error(y_true, y_pred), 3)),
                "rmse": float(np.round(root_mean_squared_error(y_true, y_pred), 3))
            }
        return metrics
