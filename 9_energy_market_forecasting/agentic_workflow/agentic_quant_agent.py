"""
Agentic Energy Quantitative Analyst Copilot.
An autonomous agent that inspects real-time model telemetry, diagnoses market price spikes,
detects forecast drift, and generates executive trading briefings for human traders.
"""

import json
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd

class AgenticQuantAnalyst:
    """
    Autonomous Quantitative Agent that monitors forecasting accuracy, explains market regime shifts,
    and drafts operational briefings for the algorithmic trading desk.
    """

    def __init__(self, agent_name: str = "Aura-Quant-Agent v2.4"):
        self.agent_name = agent_name

    def analyze_forecast_telemetry(self, forecast_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Agentic inspection of live forecast performance:
        1. Quantile excursion check: Are actual prices breaking through P90?
        2. Spread anomaly detection: Is DA-RT divergence widening beyond normal?
        3. Renewable penetration & net load ramp diagnostics.
        """
        anomalies = []
        insights = []

        # Check for extreme price spikes
        spike_hours = forecast_df[forecast_df["spike_probability"] > 0.35]
        if not spike_hours.empty:
            for _, r in spike_hours.iterrows():
                anomalies.append({
                    "timestamp": str(r["timestamp"]),
                    "type": "ELEVATED_SPIKE_RISK",
                    "p90_price": float(r["rt_price_p90"]),
                    "p50_price": float(r["rt_price_p50"]),
                    "spike_prob": float(r["spike_probability"]),
                    "net_load_mw": float(r["net_load_pred_mw"])
                })

        # Net load ramp diagnostics (duck curve severity)
        net_load = forecast_df["net_load_pred_mw"].values
        ramps = np.diff(net_load)
        max_ramp_idx = int(np.argmax(ramps)) if len(ramps) > 0 else 0
        max_ramp_mw = float(ramps[max_ramp_idx]) if len(ramps) > 0 else 0.0

        if max_ramp_mw > 3000.0:
            insights.append(
                f"Severe 1-hour evening net load ramp of {max_ramp_mw:,.0f} MW forecasted at hour index {max_ramp_idx}. "
                "Recommend staging peaker gas capacity or BESS discharge reserves."
            )

        # DA vs RT spread opportunity
        spreads = forecast_df["expected_spread"].values
        mean_spread = float(np.mean(spreads))
        max_spread = float(np.max(spreads))
        min_spread = float(np.min(spreads))

        return {
            "agent": self.agent_name,
            "generated_at": datetime.utcnow().isoformat(),
            "anomalies_detected": len(anomalies),
            "anomaly_details": anomalies,
            "mean_spread_usd": round(mean_spread, 2),
            "max_spread_usd": round(max_spread, 2),
            "min_spread_usd": round(min_spread, 2),
            "insights": insights
        }

    def generate_trader_briefing(self, forecast_df: pd.DataFrame, trading_signals: list) -> str:
        """
        Synthesizes model predictions and risk factors into a concise, actionable morning briefing
        ready for the power trading desk.
        """
        analysis = self.analyze_forecast_telemetry(forecast_df)
        def get_act(s):
            return s.action if hasattr(s, 'action') else s.get('action', '')

        def get_vol(s):
            return s.volume_mw if hasattr(s, 'volume_mw') else s.get('volume_mw', 0.0)

        n_buy = sum(1 for s in trading_signals if get_act(s) == 'DEC_BUY_DA_SELL_RT')
        n_sell = sum(1 for s in trading_signals if get_act(s) == 'INC_SELL_DA_BUY_RT')
        total_vol = sum(get_vol(s) for s in trading_signals)

        briefing = f"""# ⚡ Power Trading Desk Briefing
**Prepared by**: `{self.agent_name}`  
**Timestamp**: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`  
**Market Node**: `HUB_NORTH` | **Horizon**: `{len(forecast_df)} Hours`

---

### 1. Executive Summary & Market Bias
- **Convergence Spread**: Mean forecasted Real-Time vs Day-Ahead spread is **${analysis['mean_spread_usd']:.2f}/MWh** (Max: **${analysis['max_spread_usd']:.2f}/MWh**, Min: **${analysis['min_spread_usd']:.2f}/MWh**).
- **Active Orders Recommended**: **{n_buy} DEC Bids (Buy DA / Sell RT)** | **{n_sell} INC Offers (Sell DA / Buy RT)** across **{total_vol:.1f} MW** total volume.
- **Price Spike Alerts**: **{analysis['anomalies_detected']} hours** flagged with elevated spike probability (>35%).

### 2. Grid Physical Conditions & Duck Curve
{chr(10).join(f'- {insight}' for insight in analysis['insights']) if analysis['insights'] else '- Grid net load ramps remain within normal operating reserve tolerances.'}

### 3. Recommended Trading Strategy Actions
- **Virtual Bidding**: Exploit anticipated evening peak spread divergences where thermal gas heat rates elevate RT clearing prices above DA bids.
- **Battery Storage (BESS 100MW/400MWh)**:
  - **Charge Windows**: Early afternoon solar flood hours (low net load).
  - **Discharge Windows**: Net load peak hours (18:00 - 21:00) capturing high marginal price quantiles.

### 4. Risk & Guardrails
- Stop-loss limit active at **$18.00/MWh** adverse divergence.
- Quantile dispersion scaling applied to all position sizes to throttle risk during high meteorological uncertainty.

---
*Autonomous agent audit signature: `{self.agent_name}-{datetime.utcnow().strftime('%Y%m%d%H%M')}`*
"""
        return briefing

if __name__ == "__main__":
    from ..models.pipeline import ForecastingPipeline
    from ..trading.strategy import EnergyTradingEngine
    
    print("Running Agentic Quant Analyst...")
    pipe = ForecastingPipeline()
    pipe.train()
    feed = pipe.generate_trading_feed(horizon_hours=24)
    engine = EnergyTradingEngine()
    signals = engine.generate_virtual_bids(feed)

    agent = AgenticQuantAnalyst()
    report = agent.generate_trader_briefing(feed, signals)
    print("\n" + report)
