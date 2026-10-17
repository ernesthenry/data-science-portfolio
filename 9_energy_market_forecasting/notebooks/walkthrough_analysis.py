"""
Walkthrough Analysis & Visual Verification Script.
Executes end-to-end model training, backtesting, and quantitative reporting.
"""

import sys
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.pipeline import ForecastingPipeline
from src.trading.strategy import EnergyTradingEngine
from src.evaluation.backtest import run_walk_forward_evaluation
from agentic_workflow.agentic_quant_agent import AgenticQuantAnalyst

def main():
    print("=" * 80)
    print("  ⚡ ENERGY TIME SERIES FORECASTING & ALGORITHMIC TRADING SYSTEM")
    print("=" * 80)

    # 1. Pipeline Training & Evaluation
    pipeline = ForecastingPipeline()
    print("\n[STEP 1] Fitting Hierarchical Forecasters (Weather -> Grid -> Price)...")
    metrics = pipeline.train()

    print("\n--- Model Benchmark Performance ---")
    print(f"Weather Temp MAE:       {metrics['weather']['temperature_c']['mae']} °C")
    print(f"Weather Wind MAE:       {metrics['weather']['wind_speed_ms']['mae']} m/s")
    print(f"Grid Demand WAPE:       {metrics['grid']['system_demand_mw']['wape_pct']} %")
    print(f"Grid Net Load WAPE:     {metrics['grid']['net_load_mw']['wape_pct']} %")
    print(f"Day-Ahead LMP MAE:      ${metrics['price']['day_ahead_lmp']['mae_usd']}/MWh")
    print(f"Real-Time LMP MAE:      ${metrics['price']['real_time_lmp']['mae_usd']}/MWh")
    print(f"P10-P90 Coverage (Nom 80%): {metrics['price']['real_time_lmp']['coverage_p10_p90_pct']} %")

    # 2. Production Trading Signals & Battery Arbitrage
    print("\n[STEP 2] Generating Production Trading Signals for Trading Platform Feed...")
    feed = pipeline.generate_trading_feed(horizon_hours=24)
    engine = EnergyTradingEngine()
    signals = engine.generate_virtual_bids(feed)
    bess_sched = engine.optimize_bess_dispatch(feed)

    print(f"Generated {len(signals)} Virtual Bidding signals across 24-hour horizon.")
    active_bids = [s for s in signals if s.action != "HOLD"]
    print(f"Active Discretionary Bids: {len(active_bids)}")
    for b in active_bids[:4]:
        print(f"  • {b.timestamp} | {b.action:<20} | Vol: {b.volume_mw:4.1f} MW | Spread: ${b.expected_spread:5.2f} | Rationale: {b.rationale}")

    # 3. Strategy Financial Performance
    print("\n[STEP 3] Evaluating Out-of-Sample Financial Backtest...")
    split_idx = int(len(pipeline.df_features) * 0.8)
    test_split = pipeline.df_features.iloc[split_idx:].copy().reset_index(drop=True)
    perf = engine.backtest_strategies(test_split)

    print(f"Total Cumulative Net PnL:  ${perf['total_net_pnl_usd']:,.2f}")
    print(f"Virtual Bidding PnL:       ${perf['virtual_bidding_pnl_usd']:,.2f}")
    print(f"BESS Storage PnL:          ${perf['bess_arbitrage_pnl_usd']:,.2f}")
    print(f"Annualized Sharpe Ratio:   {perf['annualized_sharpe_ratio']:.2f}")
    print(f"Annualized Sortino Ratio:  {perf['annualized_sortino_ratio']:.2f}")
    print(f"Daily Win Rate:            {perf['daily_win_rate_pct']:.1f}%")
    print(f"Max Drawdown:              ${perf['max_drawdown_usd']:,.2f}")

    # 4. Agentic Briefing
    print("\n[STEP 4] Autonomous Agentic Briefing Generation...")
    agent = AgenticQuantAnalyst()
    briefing = agent.generate_trader_briefing(feed, signals)
    print("\n" + briefing)
    print("=" * 80)

if __name__ == "__main__":
    main()
