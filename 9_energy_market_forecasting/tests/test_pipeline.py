"""
Comprehensive Unit & Integration Test Suite for Energy Market Forecasting & Trading System.
Verifies feature engineering, model quantile monotonicity, trading strategies, and API schemas.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from src.features.engineer import (
    build_cyclical_features,
    build_weather_features,
    prepare_feature_matrix
)
from src.models.weather_forecaster import WeatherForecaster
from src.models.grid_forecaster import GridForecaster
from src.models.price_forecaster import PriceForecaster
from src.trading.strategy import EnergyTradingEngine, TradeSignal
from src.models.pipeline import ForecastingPipeline

class TestEnergyForecastingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a fast, lightweight 10-day dataset (240 hours) for rapid testing
        from data.generate_data import generate_energy_time_series
        cls.raw_df = generate_energy_time_series(start_date="2025-01-01", n_days=14, seed=101)
        cls.featured_df = prepare_feature_matrix(cls.raw_df)

    def test_01_feature_engineering_cyclical_and_physics(self):
        """Tests that cyclical and physics-informed features are bounded properly."""
        df = self.featured_df
        # Check cyclical bounds
        self.assertTrue(df["hour_sin"].between(-1.0, 1.0).all())
        self.assertTrue(df["hour_cos"].between(-1.0, 1.0).all())
        # Check non-negative cooling and heating degree days
        self.assertTrue((df["cooling_degree"] >= 0.0).all())
        self.assertTrue((df["heating_degree"] >= 0.0).all())
        # Check lag columns exist
        self.assertIn("system_demand_mw_lag_24h", df.columns)
        self.assertIn("real_time_lmp_lag_1h", df.columns)
        self.assertIn("temperature_c_lag_1h", df.columns)

    def test_02_weather_forecaster(self):
        """Verifies weather forecaster fits and produces bounded outputs."""
        wf = WeatherForecaster()
        train_df = self.featured_df.iloc[:150]
        test_df = self.featured_df.iloc[150:]

        wf.fit(train_df)
        preds = wf.predict(test_df)

        self.assertIn("temperature_c_pred", preds.columns)
        self.assertIn("wind_speed_ms_pred", preds.columns)
        self.assertIn("solar_irradiance_wm2_pred", preds.columns)
        # Solar irradiance must be non-negative
        self.assertTrue((preds["solar_irradiance_wm2_pred"] >= 0.0).all())

    def test_03_grid_supply_demand_forecaster(self):
        """Verifies grid demand, solar, wind, and implied net load calculation."""
        gf = GridForecaster()
        train_df = self.featured_df.iloc[:150]
        test_df = self.featured_df.iloc[150:]

        gf.fit(train_df)
        preds = gf.predict(test_df)

        self.assertIn("system_demand_mw_pred", preds.columns)
        self.assertIn("net_load_mw_pred", preds.columns)

        # Implied net load equation check: Net Load = Demand - (Solar + Wind)
        calc_net_load = preds["system_demand_mw_pred"] - (preds["solar_generation_mw_pred"] + preds["wind_generation_mw_pred"])
        np.testing.assert_allclose(preds["net_load_mw_pred"].values, calc_net_load.values, atol=0.2)

    def test_04_probabilistic_price_forecaster_quantile_monotonicity(self):
        """Verifies P10 <= P50 <= P90 quantile order without crossing."""
        pf = PriceForecaster()
        train_df = self.featured_df.iloc[:150]
        test_df = self.featured_df.iloc[150:]

        pf.fit(train_df)
        preds = pf.predict(test_df)

        p10 = preds["rt_price_p10"].values
        p50 = preds["rt_price_p50"].values
        p90 = preds["rt_price_p90"].values

        # Monotonicity test: P10 <= P50 <= P90
        self.assertTrue((p10 <= p50).all(), "Quantile crossing detected: P10 > P50")
        self.assertTrue((p50 <= p90).all(), "Quantile crossing detected: P50 > P90")

        # Spike probability between 0 and 1
        spike_prob = preds["spike_probability"].values
        self.assertTrue((spike_prob >= 0.0).all() and (spike_prob <= 1.0).all())

    def test_05_trading_engine_signals_and_bess_feasibility(self):
        """Verifies trading signals and physical battery constraints (SoC limits)."""
        engine = EnergyTradingEngine(min_spread_threshold=5.0)
        test_df = self.featured_df.iloc[150:].copy()
        test_df["da_price_p50"] = test_df["day_ahead_lmp"]
        test_df["rt_price_p50"] = test_df["real_time_lmp"]
        test_df["rt_price_p10"] = test_df["real_time_lmp"] - 10.0
        test_df["rt_price_p90"] = test_df["real_time_lmp"] + 15.0
        test_df["spike_probability"] = 0.08

        # 1. Virtual bidding signals
        signals = engine.generate_virtual_bids(test_df)
        self.assertEqual(len(signals), len(test_df))
        valid_actions = {"DEC_BUY_DA_SELL_RT", "INC_SELL_DA_BUY_RT", "HOLD"}
        for s in signals:
            self.assertIn(s.action, valid_actions)

        # 2. BESS Battery schedule
        bess_df = engine.optimize_bess_dispatch(test_df)
        # Verify battery State of Charge stays within 0 and 400 MWh
        self.assertTrue((bess_df["bess_soc_mwh"] >= 0.0).all())
        self.assertTrue((bess_df["bess_soc_mwh"] <= 400.0).all())

    def test_06_trading_backtest_metrics(self):
        """Verifies Sharpe and drawdown calculations execute cleanly."""
        engine = EnergyTradingEngine()
        test_df = self.featured_df.iloc[100:].copy()
        test_df["da_price_p50"] = test_df["day_ahead_lmp"]
        test_df["rt_price_p50"] = test_df["real_time_lmp"]
        perf = engine.backtest_strategies(test_df)

        self.assertIn("total_net_pnl_usd", perf)
        self.assertIn("annualized_sharpe_ratio", perf)
        self.assertIn("max_drawdown_usd", perf)
        self.assertIsInstance(perf["annualized_sharpe_ratio"], float)

if __name__ == "__main__":
    unittest.main()
