"""
Electricity Price Forecasting (EPF) Engine with Probabilistic Quantile Regression.
Predicts Day-Ahead and Real-Time Locational Marginal Prices (LMP) alongside P10, P50, P90 quantiles
and extreme price spike probabilities for algorithmic trading.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, List
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

class PriceForecaster:
    """
    Probabilistic wholesale power price forecaster.
    Employs quantile regression to quantify upside spike risk and tail downside.
    """

    def __init__(self, quantiles: List[float] = [0.10, 0.50, 0.90]):
        self.quantiles = quantiles
        # Models dict: target -> {quantile: model}
        self.quantile_models: Dict[str, Dict[float, lgb.LGBMRegressor]] = {}
        self.spike_classifier: lgb.LGBMClassifier = None
        self.feature_cols: Dict[str, list] = {}

    def _get_target_features(self, df: pd.DataFrame, target: str) -> list:
        base_feats = [
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
            "is_weekend", "is_peak_hour"
        ]
        grid_feats = [
            "net_load_mw", "net_load_mw_lag_1h", "net_load_mw_lag_24h", "net_load_mw_roll_24h_mean",
            "system_demand_mw", "solar_generation_mw", "wind_generation_mw",
            "reserve_margin_ratio"
        ]
        weather_feats = ["temperature_c", "cooling_degree", "heating_degree"]
        
        if target == "day_ahead_lmp":
            price_lags = ["day_ahead_lmp_lag_24h", "day_ahead_lmp_lag_48h", "day_ahead_lmp_lag_168h"]
        else: # real_time_lmp
            price_lags = [
                "day_ahead_lmp", "real_time_lmp_lag_1h", "real_time_lmp_lag_2h",
                "real_time_lmp_lag_24h", "real_time_lmp_roll_6h_mean", "real_time_lmp_roll_24h_mean"
            ]

        all_candidates = base_feats + grid_feats + weather_feats + price_lags
        return [f for f in all_candidates if f in df.columns]

    def fit(self, df: pd.DataFrame) -> "PriceForecaster":
        """Fits quantile regression models (P10, P50, P90) for DA and RT prices."""
        for target in ("day_ahead_lmp", "real_time_lmp"):
            features = self._get_target_features(df, target)
            self.feature_cols[target] = features
            self.quantile_models[target] = {}

            X = df[features]
            y = df[target]

            for q in self.quantiles:
                q_model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=32,
                    subsample=0.85,
                    random_state=42,
                    verbosity=-1
                )
                q_model.fit(X, y)
                self.quantile_models[target][q] = q_model

        # Train Price Spike Classifier: Target is whether RT price > $100/MWh
        rt_features = self.feature_cols["real_time_lmp"]
        y_spike = (df["real_time_lmp"] > 100.0).astype(int)
        self.spike_classifier = lgb.LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            scale_pos_weight=4.0, # Handle positive spike imbalance
            max_depth=5,
            random_state=42,
            verbosity=-1
        )
        self.spike_classifier.fit(df[rt_features], y_spike)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates probabilistic price forecasts: P10 (downside), P50 (median), P90 (spike risk),
        and price spike probability.
        """
        out = pd.DataFrame(index=df.index)
        if "timestamp" in df.columns:
            out["timestamp"] = df["timestamp"]

        for target in ("day_ahead_lmp", "real_time_lmp"):
            features = self.feature_cols[target]
            p10 = self.quantile_models[target][0.10].predict(df[features])
            p50 = self.quantile_models[target][0.50].predict(df[features])
            p90 = self.quantile_models[target][0.90].predict(df[features])

            # Monotonic quantile sort to prevent quantile crossing
            quantiles_stacked = np.sort(np.vstack([p10, p50, p90]), axis=0)
            prefix = "da" if "day_ahead" in target else "rt"

            out[f"{prefix}_price_p10"] = np.round(quantiles_stacked[0], 2)
            out[f"{prefix}_price_p50"] = np.round(quantiles_stacked[1], 2)
            out[f"{prefix}_price_p90"] = np.round(quantiles_stacked[2], 2)
            out[f"{prefix}_price_pred"] = out[f"{prefix}_price_p50"]

        # Price spike probability (> $100/MWh)
        rt_features = self.feature_cols["real_time_lmp"]
        spike_probs = self.spike_classifier.predict_proba(df[rt_features])[:, 1]
        out["spike_probability"] = np.round(spike_probs, 3)

        # Expected spread (RT - DA)
        out["expected_spread_rt_minus_da"] = np.round(out["rt_price_p50"] - out["da_price_p50"], 2)
        return out

    @staticmethod
    def compute_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
        """Calculates pinball (quantile) loss."""
        err = y_true - y_pred
        return float(np.mean(np.maximum(alpha * err, (alpha - 1) * err)))

    def evaluate(self, df_test: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates point metrics, quantile pinball loss, and coverage probability."""
        preds = self.predict(df_test)
        metrics = {}

        for target in ("day_ahead_lmp", "real_time_lmp"):
            prefix = "da" if "day_ahead" in target else "rt"
            y_true = df_test[target].values
            p10 = preds[f"{prefix}_price_p10"].values
            p50 = preds[f"{prefix}_price_p50"].values
            p90 = preds[f"{prefix}_price_p90"].values

            mae = mean_absolute_error(y_true, p50)
            rmse = root_mean_squared_error(y_true, p50)
            pinball_10 = self.compute_pinball_loss(y_true, p10, 0.10)
            pinball_50 = self.compute_pinball_loss(y_true, p50, 0.50)
            pinball_90 = self.compute_pinball_loss(y_true, p90, 0.90)

            # Empirical Coverage: fraction of actual prices inside [P10, P90] (nominal = 80%)
            coverage = np.mean((y_true >= p10) & (y_true <= p90)) * 100.0

            metrics[target] = {
                "mae_usd": float(np.round(mae, 2)),
                "rmse_usd": float(np.round(rmse, 2)),
                "pinball_loss_p10": float(np.round(pinball_10, 2)),
                "pinball_loss_p50": float(np.round(pinball_50, 2)),
                "pinball_loss_p90": float(np.round(pinball_90, 2)),
                "coverage_p10_p90_pct": float(np.round(coverage, 1))
            }
        return metrics
