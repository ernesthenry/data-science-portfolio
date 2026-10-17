"""
Walk-Forward Time Series Backtest Engine & Quantitative Validation.
Executes expanding window cross-validation to assess out-of-sample forecast accuracy
and financial Sharpe performance without data leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from ..models.pipeline import ForecastingPipeline
from ..trading.strategy import EnergyTradingEngine

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculates Symmetric Mean Absolute Percentage Error (sMAPE)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    # Avoid zero division
    valid = denominator > 1e-5
    if not np.any(valid):
        return 0.0
    return float(np.mean(diff[valid] / denominator[valid]) * 100.0)

def run_walk_forward_evaluation(n_splits: int = 3, test_horizon_hours: int = 720) -> Dict[str, Any]:
    """
    Performs expanding-window walk-forward validation across chronological folds.
    Each fold simulates 30 days (720 hours) of out-of-sample trading operations.
    """
    pipeline = ForecastingPipeline()
    df_raw = pipeline.load_data()
    from ..features.engineer import prepare_feature_matrix
    df_features = prepare_feature_matrix(df_raw)

    total_samples = len(df_features)
    min_train = total_samples - (n_splits * test_horizon_hours)

    fold_metrics = []
    engine = EnergyTradingEngine()

    for fold in range(n_splits):
        train_end = min_train + fold * test_horizon_hours
        test_end = train_end + test_horizon_hours

        train_split = df_features.iloc[:train_end].copy().reset_index(drop=True)
        test_split = df_features.iloc[train_end:test_end].copy().reset_index(drop=True)

        # Fit models on expanding history
        pipeline.weather_model.fit(train_split)
        pipeline.grid_model.fit(train_split)
        pipeline.price_model.fit(train_split)
        pipeline.is_fitted = True

        # Generate out-of-sample predictions
        weather_eval = pipeline.weather_model.evaluate(test_split)
        grid_eval = pipeline.grid_model.evaluate(test_split)
        price_eval = pipeline.price_model.evaluate(test_split)

        price_preds = pipeline.price_model.predict(test_split)
        test_with_preds = test_split.copy()
        for col in price_preds.columns:
            if col != "timestamp":
                test_with_preds[col] = price_preds[col]

        trading_perf = engine.backtest_strategies(test_with_preds)

        fold_metrics.append({
            "fold": fold + 1,
            "train_hours": len(train_split),
            "test_hours": len(test_split),
            "start_time": test_split["timestamp"].iloc[0],
            "end_time": test_split["timestamp"].iloc[-1],
            "weather_temp_mae": weather_eval["temperature_c"]["mae"],
            "grid_demand_wape": grid_eval["system_demand_mw"]["wape_pct"],
            "price_rt_mae": price_eval["real_time_lmp"]["mae_usd"],
            "price_coverage_pct": price_eval["real_time_lmp"]["coverage_p10_p90_pct"],
            "fold_net_pnl_usd": trading_perf["total_net_pnl_usd"],
            "fold_sharpe": trading_perf["annualized_sharpe_ratio"]
        })

    summary = {
        "n_folds": n_splits,
        "avg_price_mae": float(np.mean([m["price_rt_mae"] for m in fold_metrics])),
        "avg_coverage_pct": float(np.mean([m["price_coverage_pct"] for m in fold_metrics])),
        "avg_sharpe_ratio": float(np.mean([m["fold_sharpe"] for m in fold_metrics])),
        "total_pnl_across_folds": float(np.sum([m["fold_net_pnl_usd"] for m in fold_metrics])),
        "fold_details": fold_metrics
    }
    return summary

if __name__ == "__main__":
    print("Running Walk-Forward Cross-Validation...")
    results = run_walk_forward_evaluation(n_splits=3, test_horizon_hours=720)
    print(f"Validation complete! Average Price MAE: ${results['avg_price_mae']:.2f}/MWh, Sharpe: {results['avg_sharpe_ratio']:.2f}")
