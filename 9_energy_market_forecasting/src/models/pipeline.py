"""
Unified Forecasting Pipeline Orchestrator.
Loads data, executes feature transformations, fits hierarchical models (Weather -> Grid -> Price),
and provides high-throughput real-time prediction bundles for the trading engine.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

from ..features.engineer import prepare_feature_matrix
from .weather_forecaster import WeatherForecaster
from .grid_forecaster import GridForecaster
from .price_forecaster import PriceForecaster

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = DATA_DIR / "energy_grid.db"
CSV_PATH = DATA_DIR / "energy_dataset.csv"

class ForecastingPipeline:
    """End-to-end multi-target time series forecasting pipeline."""

    def __init__(self):
        self.weather_model = WeatherForecaster()
        self.grid_model = GridForecaster()
        self.price_model = PriceForecaster()
        self.is_fitted = False
        self.df_features: pd.DataFrame = None

    def load_data(self) -> pd.DataFrame:
        """Loads and joins telemetry data from SQLite or CSV fallback."""
        if DB_PATH.exists():
            conn = sqlite3.connect(DB_PATH)
            query = """
                SELECT 
                    m.timestamp, m.market_node, m.day_ahead_lmp, m.real_time_lmp, m.congestion_price, m.loss_price,
                    g.system_demand_mw, g.solar_generation_mw, g.wind_generation_mw, g.net_load_mw, g.spinning_reserves_mw,
                    w.temperature_c, w.wind_speed_ms, w.solar_irradiance_wm2, w.relative_humidity, w.cloud_cover_pct
                FROM market_prices m
                JOIN grid_telemetry g ON m.timestamp = g.timestamp
                JOIN weather_observations w ON m.timestamp = w.timestamp
                ORDER BY m.timestamp ASC
            """
            df = pd.read_sql(query, conn)
            conn.close()
        elif CSV_PATH.exists():
            df = pd.read_csv(CSV_PATH)
        else:
            raise FileNotFoundError("Neither SQLite DB nor CSV dataset found. Run data/generate_data.py first.")

        return df

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs feature engineering and splits into train (80%) and test (20%) sets."""
        raw_df = self.load_data()
        self.df_features = prepare_feature_matrix(raw_df)
        
        split_idx = int(len(self.df_features) * 0.8)
        df_train = self.df_features.iloc[:split_idx].reset_index(drop=True)
        df_test = self.df_features.iloc[split_idx:].reset_index(drop=True)
        return df_train, df_test

    def train(self) -> Dict[str, Any]:
        """Trains weather, grid, and price models hierarchically and evaluates on test split."""
        df_train, df_test = self.prepare_data()

        print("Fitting Weather Forecaster...")
        self.weather_model.fit(df_train)
        weather_metrics = self.weather_model.evaluate(df_test)

        print("Fitting Grid Demand & Supply Forecaster...")
        self.grid_model.fit(df_train)
        grid_metrics = self.grid_model.evaluate(df_test)

        print("Fitting Probabilistic Price Forecaster (Quantiles P10/P50/P90)...")
        self.price_model.fit(df_train)
        price_metrics = self.price_model.evaluate(df_test)

        self.is_fitted = True
        return {
            "weather": weather_metrics,
            "grid": grid_metrics,
            "price": price_metrics,
            "train_samples": len(df_train),
            "test_samples": len(df_test)
        }

    def generate_trading_feed(self, horizon_hours: int = 24) -> pd.DataFrame:
        """
        Produces a production forecast bundle for the trading platform across the requested horizon.
        Combines weather, net load ramp, quantile price intervals, and spike probabilities.
        """
        if not self.is_fitted:
            self.train()

        # Select latest horizon rows
        latest_df = self.df_features.tail(horizon_hours).copy().reset_index(drop=True)

        weather_preds = self.weather_model.predict(latest_df)
        grid_preds = self.grid_model.predict(latest_df)
        price_preds = self.price_model.predict(latest_df)

        feed = pd.DataFrame({
            "timestamp": latest_df["timestamp"],
            # Weather
            "temp_pred_c": weather_preds["temperature_c_pred"],
            "wind_speed_pred_ms": weather_preds["wind_speed_ms_pred"],
            "solar_irradiance_pred_wm2": weather_preds["solar_irradiance_wm2_pred"],
            # Grid
            "system_demand_pred_mw": grid_preds["system_demand_mw_pred"],
            "solar_gen_pred_mw": grid_preds["solar_generation_mw_pred"],
            "wind_gen_pred_mw": grid_preds["wind_generation_mw_pred"],
            "net_load_pred_mw": grid_preds["net_load_mw_pred"],
            # Prices & Risk Quantiles
            "da_price_p10": price_preds["da_price_p10"],
            "da_price_p50": price_preds["da_price_p50"],
            "da_price_p90": price_preds["da_price_p90"],
            "rt_price_p10": price_preds["rt_price_p10"],
            "rt_price_p50": price_preds["rt_price_p50"],
            "rt_price_p90": price_preds["rt_price_p90"],
            "expected_spread": price_preds["expected_spread_rt_minus_da"],
            "spike_probability": price_preds["spike_probability"],
            # Actual values for latency/drift verification
            "da_price_actual": latest_df["day_ahead_lmp"],
            "rt_price_actual": latest_df["real_time_lmp"]
        })
        return feed
