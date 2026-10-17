"""
Interactive Streamlit Dashboard for Energy Market Time Series Forecasting & Algorithmic Trading.
Provides visual trading desk analytics, quantile price bands, Duck Curve ramps, and agentic briefings.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Energy Quant | Market Forecasting & Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-buy {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-sell {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.models.pipeline import ForecastingPipeline
from src.trading.strategy import EnergyTradingEngine
from agentic_workflow.agentic_quant_agent import AgenticQuantAnalyst

@st.cache_resource
def get_pipeline():
    pipe = ForecastingPipeline()
    pipe.train()
    return pipe

# Load pipeline
with st.spinner("Initializing Energy Time Series Forecaster & Trading Engine..."):
    pipeline = get_pipeline()
    engine = EnergyTradingEngine()
    agent = AgenticQuantAnalyst()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/electricity.png", width=64)
st.sidebar.title("⚡ Energy Quant Desk")
st.sidebar.markdown("**ISO Node:** `HUB_NORTH` | **Zone:** `ZONE_NORTH`")

horizon_hours = st.sidebar.slider("Forecast Horizon (Hours)", min_value=12, max_value=72, value=24, step=6)
spread_threshold = st.sidebar.slider("Virtual Bidding Spread Threshold ($/MWh)", min_value=2.0, max_value=15.0, value=6.5, step=0.5)
bess_power = st.sidebar.slider("BESS Storage Power (MW)", min_value=25, max_value=200, value=100, step=25)

# Generate Feed
feed_df = pipeline.generate_trading_feed(horizon_hours=horizon_hours)
engine.min_spread_threshold = spread_threshold
engine.bess_power_mw = float(bess_power)
signals = engine.generate_virtual_bids(feed_df)
bess_df = engine.optimize_bess_dispatch(feed_df)

# Top Bar KPIs
st.title("Wholesale Power Market Forecasting & Algorithmic Trading Feed")
st.markdown("*Real-time production time series forecasts feeding algorithmic execution gateways and battery dispatch.*")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    avg_da = feed_df["da_price_p50"].mean()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg DA Price (P50)</div>
        <div class="metric-val">${avg_da:.2f} <span style="font-size:14px;color:#94a3b8">/MWh</span></div>
    </div>""", unsafe_allow_html=True)
with col2:
    avg_rt = feed_df["rt_price_p50"].mean()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Avg RT Price (P50)</div>
        <div class="metric-val">${avg_rt:.2f} <span style="font-size:14px;color:#94a3b8">/MWh</span></div>
    </div>""", unsafe_allow_html=True)
with col3:
    max_spike = feed_df["rt_price_p90"].max()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Peak Spike Risk (P90)</div>
        <div class="metric-val" style="color:#f59e0b">${max_spike:.2f} <span style="font-size:14px;color:#94a3b8">/MWh</span></div>
    </div>""", unsafe_allow_html=True)
with col4:
    max_net_load = feed_df["net_load_pred_mw"].max()
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Peak Net Load</div>
        <div class="metric-val">{max_net_load:,.0f} <span style="font-size:14px;color:#94a3b8">MW</span></div>
    </div>""", unsafe_allow_html=True)
with col5:
    active_bids = sum(1 for s in signals if s.action != "HOLD")
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Active Orders</div>
        <div class="metric-val" style="color:#10b981">{active_bids} <span style="font-size:14px;color:#94a3b8">Signals</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dashboard Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Probabilistic Price Quantiles",
    "🦆 Grid Supply & Duck Curve",
    "🌤️ Weather Predictors",
    "💰 Trading Strategies & PnL",
    "⚙️ Production Feed & Go Gateway",
    "🤖 Agentic Copilot Briefing"
])

# ----------------- TAB 1: Probabilistic Price Quantiles -----------------
with tab1:
    st.subheader("Day-Ahead & Real-Time Price Forecasting with Quantile Risk Bands")
    st.caption("P10 (downside floor), P50 (median clearing forecast), and P90 (scarcity spike risk) generated via LightGBM Quantile Regression.")

    fig = go.Figure()

    # RT Quantile Ribbon (P10 to P90)
    fig.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["rt_price_p90"],
        line=dict(width=0), showlegend=False, name="RT P90 (Upside Risk)"
    ))
    fig.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["rt_price_p10"],
        fill='tonexty', fillcolor='rgba(56, 189, 248, 0.15)',
        line=dict(width=0), name="RT 80% Confidence Band [P10, P90]"
    ))
    # Median Forecasts
    fig.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["rt_price_p50"],
        line=dict(color="#38bdf8", width=3), name="Real-Time LMP (P50 Forecast)"
    ))
    fig.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["da_price_p50"],
        line=dict(color="#f59e0b", width=2.5, dash="dash"), name="Day-Ahead LMP (P50 Forecast)"
    ))
    # Actual Settlement Prices
    fig.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["rt_price_actual"],
        mode="markers", marker=dict(color="#ef4444", size=6), name="Actual RT Settlement Price"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title="Timestamp (Hourly)",
        yaxis_title="Price ($/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Spike Risk Indicator
    spike_df = feed_df[["timestamp", "rt_price_p50", "rt_price_p90", "expected_spread", "spike_probability"]].copy()
    spike_df["spike_alert"] = spike_df["spike_probability"].apply(lambda p: "🚨 HIGH RISK" if p > 0.35 else "NORMAL")
    st.dataframe(spike_df.style.highlight_max(subset=["spike_probability"], color="#7f1d1d"), use_container_width=True)

# ----------------- TAB 2: Grid Supply & Duck Curve -----------------
with tab2:
    st.subheader("Grid Supply/Demand Dynamics & The Evening Ramp (Duck Curve)")
    st.caption("Net Load = System Demand - (Solar Generation + Wind Generation). High evening ramps trigger thermal peakers.")

    fig_grid = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                             subplot_titles=("Power Generation Stack vs Total Demand (MW)", "Net Load & 3-Hour Ramp Rate (MW)"))

    # Stacked generation
    fig_grid.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["solar_gen_pred_mw"],
        fill='tozeroy', line=dict(color="#eab308"), name="Solar Generation (MW)"
    ), row=1, col=1)
    fig_grid.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["wind_gen_pred_mw"],
        fill='tozeroy', line=dict(color="#06b6d4"), name="Wind Generation (MW)"
    ), row=1, col=1)
    fig_grid.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["system_demand_pred_mw"],
        line=dict(color="#f43f5e", width=3), name="Total System Demand (MW)"
    ), row=1, col=1)

    # Net load
    fig_grid.add_trace(go.Scatter(
        x=feed_df["timestamp"], y=feed_df["net_load_pred_mw"],
        line=dict(color="#a855f7", width=3), name="Net Load (MW)"
    ), row=2, col=1)

    fig_grid.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig_grid, use_container_width=True)

# ----------------- TAB 3: Weather Predictors -----------------
with tab3:
    st.subheader("Numerical Weather Predictions (Exogenous Model Drivers)")
    st.caption("Coupled multi-target weather forecasts for temperature, wind speed, and solar irradiance.")

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=feed_df["timestamp"], y=feed_df["temp_pred_c"], line=dict(color="#f97316", width=2.5)))
        fig_temp.update_layout(template="plotly_dark", height=280, title="Temperature (°C)", margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_temp, use_container_width=True)
    with col_w2:
        fig_wind = go.Figure()
        fig_wind.add_trace(go.Scatter(x=feed_df["timestamp"], y=feed_df["wind_speed_pred_ms"], line=dict(color="#38bdf8", width=2.5)))
        fig_wind.update_layout(template="plotly_dark", height=280, title="Wind Speed (m/s)", margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_wind, use_container_width=True)
    with col_w3:
        fig_solar = go.Figure()
        fig_solar.add_trace(go.Scatter(x=feed_df["timestamp"], y=feed_df["solar_irradiance_pred_wm2"], line=dict(color="#eab308", width=2.5)))
        fig_solar.update_layout(template="plotly_dark", height=280, title="Solar Irradiance (W/m²)", margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig_solar, use_container_width=True)

# ----------------- TAB 4: Trading Strategies & PnL -----------------
with tab4:
    st.subheader("Algorithmic Trading Performance & Battery Storage Dispatch")
    st.caption("Quantitative performance backtest across out-of-sample settlement prices.")

    # Run quick backtest on test slice
    split_idx = int(len(pipeline.df_features) * 0.8)
    test_slice = pipeline.df_features.iloc[split_idx:].copy().reset_index(drop=True)
    perf = engine.backtest_strategies(test_slice)

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    pcol1.metric("Cumulative Net PnL", f"${perf['total_net_pnl_usd']:,.2f}")
    pcol2.metric("Annualized Sharpe Ratio", f"{perf['annualized_sharpe_ratio']:.2f}")
    pcol3.metric("Daily Win Rate", f"{perf['daily_win_rate_pct']:.1f}%")
    pcol4.metric("Max Drawdown", f"${perf['max_drawdown_usd']:,.2f}")

    # BESS Battery Schedule Chart
    st.markdown("#### Optimal Battery (100MW / 400MWh) Dispatch & State of Charge")
    fig_bess = make_subplots(specs=[[{"secondary_y": True}]])
    fig_bess.add_trace(go.Bar(
        x=bess_df["timestamp"], y=bess_df["bess_charge_mw"],
        marker_color="#22c55e", name="Charge (MW)"
    ), secondary_y=False)
    fig_bess.add_trace(go.Bar(
        x=bess_df["timestamp"], y=-bess_df["bess_discharge_mw"],
        marker_color="#ef4444", name="Discharge (MW)"
    ), secondary_y=False)
    fig_bess.add_trace(go.Scatter(
        x=bess_df["timestamp"], y=bess_df["bess_soc_pct"],
        line=dict(color="#a855f7", width=3), name="State of Charge (%)"
    ), secondary_y=True)

    fig_bess.update_layout(template="plotly_dark", height=380, margin=dict(l=40,r=40,t=20,b=40))
    st.plotly_chart(fig_bess, use_container_width=True)

# ----------------- TAB 5: Production Feed & Go Gateway -----------------
with tab5:
    st.subheader("Production Trading Platform Interface & Go Execution Bridge")
    st.caption("Demonstrating high-performance low-latency architecture connecting Python ML models to a Go order routing microservice.")

    st.code("""
// Go Execution Gateway Snippet (go_trading_client/main.go)
func (h *TradingFeedHandler) RouteOrder(sig TradingSignal, idx int) FIXOrderMessage {
    approved, reason := h.Risk.Validate(sig)
    return FIXOrderMessage{
        MsgType:       "35=D (NewOrderSingle)",
        ClOrdID:       fmt.Sprintf("ORD-%s-%03d", time.Now().Format("20060102150405"), idx),
        Symbol:        "HUB_NORTH_DA_RT",
        OrderQty:      sig.VolumeMW,
        Price:         sig.DAPriceForecast,
        RiskStatus:    reason,
        LatencyMicros: time.Since(start).Microseconds(),
    }
}
    """, language="go")

    st.markdown("#### Real-Time Algorithmic Order Tickets")
    signal_table = []
    for s in signals:
        signal_table.append({
            "Timestamp": s.timestamp,
            "Action": s.action,
            "Volume (MW)": s.volume_mw,
            "Expected Spread": f"${s.expected_spread:.2f}",
            "DA Forecast": f"${s.da_price_forecast:.2f}",
            "RT Forecast": f"${s.rt_price_forecast:.2f}",
            "Spike Risk": f"{s.spike_risk*100:.1f}%",
            "Rationale": s.rationale
        })
    st.dataframe(pd.DataFrame(signal_table), use_container_width=True)

# ----------------- TAB 6: Agentic Copilot Briefing -----------------
with tab6:
    st.subheader("Autonomous Agentic Trader Morning Briefing")
    st.caption("Generated automatically by the Agentic Quantitative Copilot via model telemetry diagnostics.")
    briefing_md = agent.generate_trader_briefing(feed_df, signals)
    st.markdown(briefing_md)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 13px;'>Built by Kato Ernest Henry | Data Science & Algorithmic Trading Systems</p>", unsafe_allow_html=True)
