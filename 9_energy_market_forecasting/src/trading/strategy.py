"""
Algorithmic Energy Trading Engine & Strategy Backtester.
Translates multi-horizon price and grid forecasts into production trading decisions:
1. Virtual Bidding (DA-RT Convergence Spread Arbitrage)
2. BESS (Battery Energy Storage System) 4-Hour Arbitrage Dispatch Optimizer
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class TradeSignal:
    timestamp: str
    asset_id: str
    action: str  # 'DEC_BUY_DA_SELL_RT', 'INC_SELL_DA_BUY_RT', 'BESS_CHARGE', 'BESS_DISCHARGE', 'HOLD'
    volume_mw: float
    expected_spread: float
    da_price_forecast: float
    rt_price_forecast: float
    spike_risk: float
    rationale: str

class EnergyTradingEngine:
    """Production algorithmic trading strategy engine for energy commodities."""

    def __init__(
        self,
        min_spread_threshold: float = 6.50,
        max_position_mw: float = 100.0,
        transaction_fee: float = 0.45,
        bess_power_mw: float = 100.0,
        bess_capacity_mwh: float = 400.0,
        bess_rte: float = 0.88,
        bess_cycling_cost: float = 3.50
    ):
        self.min_spread_threshold = min_spread_threshold
        self.max_position_mw = max_position_mw
        self.transaction_fee = transaction_fee
        self.bess_power_mw = bess_power_mw
        self.bess_capacity_mwh = bess_capacity_mwh
        self.bess_rte = bess_rte
        self.bess_cycling_cost = bess_cycling_cost

    def generate_virtual_bids(self, forecast_df: pd.DataFrame) -> List[TradeSignal]:
        """
        Generates Virtual Bidding orders based on predicted Day-Ahead vs Real-Time price divergence.
        - DEC (Buy DA, Sell RT): When RT forecast > DA forecast + threshold
        - INC (Sell DA, Buy RT): When DA forecast > RT forecast + threshold
        """
        signals = []
        for _, row in forecast_df.iterrows():
            ts = str(row["timestamp"])
            da_pred = row.get("da_price_p50", row.get("day_ahead_lmp", 40.0))
            rt_pred = row.get("rt_price_p50", row.get("real_time_lmp", 40.0))
            spread = rt_pred - da_pred
            uncertainty_band = row.get("rt_price_p90", rt_pred + 10) - row.get("rt_price_p10", rt_pred - 10)
            spike_prob = row.get("spike_probability", 0.05)

            # Volatility risk adjustment: scale down position when forecast dispersion is high
            risk_scaling = np.clip(25.0 / (uncertainty_band + 1e-4), 0.25, 1.0)
            pos_size = float(np.round(self.max_position_mw * risk_scaling, 1))

            if spread > self.min_spread_threshold:
                # Expecting Real-Time to clear higher: Buy DA, Sell RT (DEC bid)
                action = "DEC_BUY_DA_SELL_RT"
                rationale = f"Forecasted RT premium: +${spread:.2f}/MWh (Spike risk {spike_prob*100:.1f}%)"
            elif spread < -self.min_spread_threshold:
                # Expecting Day-Ahead to clear higher: Sell DA, Buy RT (INC offer)
                action = "INC_SELL_DA_BUY_RT"
                rationale = f"Forecasted DA premium: +${abs(spread):.2f}/MWh"
            else:
                action = "HOLD"
                pos_size = 0.0
                rationale = "Spread within neutral transaction cost buffer"

            signals.append(TradeSignal(
                timestamp=ts,
                asset_id="VIRTUAL_HUB_NORTH",
                action=action,
                volume_mw=pos_size,
                expected_spread=float(np.round(spread, 2)),
                da_price_forecast=float(np.round(da_pred, 2)),
                rt_price_forecast=float(np.round(rt_pred, 2)),
                spike_risk=float(np.round(spike_prob, 3)),
                rationale=rationale
            ))
        return signals

    def optimize_bess_dispatch(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimal daily dispatch schedule for a 100MW / 400MWh battery.
        Charges during lowest price / negative price solar hours, discharges during peak net load ramps.
        """
        df = forecast_df.copy()
        n = len(df)
        charge_mw = np.zeros(n)
        discharge_mw = np.zeros(n)
        soc_mwh = np.zeros(n)  # State of Charge
        current_soc = self.bess_capacity_mwh * 0.20 # Start at 20% SoC

        price_series = df.get("rt_price_p50", df.get("day_ahead_lmp", pd.Series(np.ones(n)*40.0))).values

        # Quantile thresholds for charge and discharge
        q_charge = np.quantile(price_series, 0.25)
        q_discharge = np.quantile(price_series, 0.75)

        for t in range(n):
            price = price_series[t]
            # Charging opportunity (price below 25th percentile or negative)
            if price <= q_charge and current_soc < self.bess_capacity_mwh:
                max_charge = min(self.bess_power_mw, (self.bess_capacity_mwh - current_soc) / self.bess_rte)
                charge_mw[t] = max_charge
                current_soc += max_charge * self.bess_rte
            # Discharging opportunity (price above 75th percentile and battery has energy)
            elif price >= q_discharge and current_soc > 20.0:
                max_discharge = min(self.bess_power_mw, current_soc - 20.0)
                discharge_mw[t] = max_discharge
                current_soc -= max_discharge

            soc_mwh[t] = current_soc

        df["bess_charge_mw"] = np.round(charge_mw, 1)
        df["bess_discharge_mw"] = np.round(discharge_mw, 1)
        df["bess_soc_mwh"] = np.round(soc_mwh, 1)
        df["bess_soc_pct"] = np.round((soc_mwh / self.bess_capacity_mwh) * 100.0, 1)

        # Revenue = Discharge * Price - Charge * Price / RTE - cycling costs
        revenue = (
            discharge_mw * price_series - 
            charge_mw * price_series - 
            (discharge_mw + charge_mw) * (self.bess_cycling_cost + self.transaction_fee)
        )
        df["bess_hourly_pnl_usd"] = np.round(revenue, 2)
        return df

    def backtest_strategies(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs complete quantitative backtest on out-of-sample data with actual settlement prices.
        Computes Sharpe Ratio, Sortino Ratio, Cumulative PnL, Win Rate, and Max Drawdown.
        """
        virtual_signals = self.generate_virtual_bids(test_df)
        bess_results = self.optimize_bess_dispatch(test_df)

        # 1. Virtual Bidding PnL
        vb_pnls = []
        da_actual = test_df["day_ahead_lmp"].values
        rt_actual = test_df["real_time_lmp"].values

        for idx, sig in enumerate(virtual_signals):
            da_act = da_actual[idx]
            rt_act = rt_actual[idx]
            vol = sig.volume_mw

            if sig.action == "DEC_BUY_DA_SELL_RT":
                # Buy DA, Sell RT -> Gross Spread = RT - DA
                gross_pnl = vol * (rt_act - da_act)
                fee = vol * (self.transaction_fee * 2)
                net = gross_pnl - fee
            elif sig.action == "INC_SELL_DA_BUY_RT":
                # Sell DA, Buy RT -> Gross Spread = DA - RT
                gross_pnl = vol * (da_act - rt_act)
                fee = vol * (self.transaction_fee * 2)
                net = gross_pnl - fee
            else:
                net = 0.0
            vb_pnls.append(net)

        vb_series = pd.Series(vb_pnls, index=test_df.index)
        bess_pnl_series = bess_results["bess_hourly_pnl_usd"]
        total_pnl_series = vb_series + bess_pnl_series

        # Quantitative Performance Metrics
        daily_pnl = total_pnl_series.groupby(test_df["timestamp"].astype(str).str.slice(0, 10)).sum()
        days_count = max(len(daily_pnl), 1)
        mean_daily = daily_pnl.mean()
        std_daily = daily_pnl.std() + 1e-6
        downside_std = daily_pnl[daily_pnl < 0].std() + 1e-6

        # Annualized metrics (assuming 365 trading days for 24/7 power markets)
        sharpe = (mean_daily / std_daily) * np.sqrt(365)
        sortino = (mean_daily / downside_std) * np.sqrt(365)

        # Maximum Drawdown
        cum_pnl = daily_pnl.cumsum()
        peak = cum_pnl.cummax()
        drawdown = cum_pnl - peak
        max_drawdown = float(drawdown.min())

        win_days = (daily_pnl > 0).sum()
        win_rate = (win_days / days_count) * 100.0

        return {
            "total_net_pnl_usd": float(np.round(total_pnl_series.sum(), 2)),
            "virtual_bidding_pnl_usd": float(np.round(vb_series.sum(), 2)),
            "bess_arbitrage_pnl_usd": float(np.round(bess_pnl_series.sum(), 2)),
            "annualized_sharpe_ratio": float(np.round(sharpe, 2)),
            "annualized_sortino_ratio": float(np.round(sortino, 2)),
            "max_drawdown_usd": float(np.round(max_drawdown, 2)),
            "daily_win_rate_pct": float(np.round(win_rate, 1)),
            "total_trading_days": int(days_count),
            "daily_pnl": daily_pnl.to_dict()
        }
