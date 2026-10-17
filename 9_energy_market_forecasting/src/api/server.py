"""
Production Trading Platform API Feed.
High-throughput REST and streaming endpoint server feeding wholesale market forecasts,
quantile risk intervals, and algorithmic trading execution signals to trading desks and execution gateways.
"""

import time
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..models.pipeline import ForecastingPipeline
from ..trading.strategy import EnergyTradingEngine, TradeSignal
from ..config import config

app = FastAPI(
    title="Energy Market Time Series Forecasting & Trading Feed API",
    description="Feeds real-time weather, grid supply/demand, and LMP price quantiles to algorithmic trading platforms.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline & trading engine instances
pipeline = ForecastingPipeline()
trading_engine = EnergyTradingEngine()

# Pre-train / warm up models on startup
@app.on_event("startup")
def startup_event():
    print("Pre-training & initializing forecasting pipeline models on startup...")
    pipeline.train()
    print("Forecasting pipeline initialized and ready to serve trading desk!")

# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_fitted: bool
    train_samples: int
    test_samples: int
    zone_id: str
    market_node: str

class QuantilePricePoint(BaseModel):
    timestamp: str
    p10_downside: float
    p50_median: float
    p90_upside_risk: float
    actual: Optional[float] = None

class ForecastBundleResponse(BaseModel):
    timestamp_generated: str
    horizon_hours: int
    da_prices: List[QuantilePricePoint]
    rt_prices: List[QuantilePricePoint]
    net_load_mw: List[Dict[str, Any]]
    weather: List[Dict[str, Any]]
    spike_risk_hours: List[str]

class TradingSignalItem(BaseModel):
    timestamp: str
    asset_id: str
    action: str
    volume_mw: float
    expected_spread: float
    da_price_forecast: float
    rt_price_forecast: float
    spike_risk: float
    rationale: str

START_TIME = time.time()

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Service health and model telemetry check."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - START_TIME, 1),
        model_fitted=pipeline.is_fitted,
        train_samples=len(pipeline.df_features) if pipeline.df_features is not None else 0,
        test_samples=int(len(pipeline.df_features) * 0.2) if pipeline.df_features is not None else 0,
        zone_id=config.zone_id,
        market_node=config.market_node
    )

@app.get("/forecast/bundle")
def get_forecast_bundle(horizon_hours: int = Query(default=24, ge=1, le=168)):
    """
    Feeds complete multi-horizon forecasts:
    - Weather (temp, wind, solar irradiance)
    - Grid supply & demand (load, solar, wind, net load)
    - Electricity price quantiles (DA & RT: P10, P50, P90)
    """
    feed = pipeline.generate_trading_feed(horizon_hours=horizon_hours)
    
    da_list = []
    rt_list = []
    grid_list = []
    weather_list = []
    spike_hours = []

    for _, r in feed.iterrows():
        ts = str(r["timestamp"])
        da_list.append({
            "timestamp": ts,
            "p10_downside": float(r["da_price_p10"]),
            "p50_median": float(r["da_price_p50"]),
            "p90_upside_risk": float(r["da_price_p90"]),
            "actual": float(r["da_price_actual"])
        })
        rt_list.append({
            "timestamp": ts,
            "p10_downside": float(r["rt_price_p10"]),
            "p50_median": float(r["rt_price_p50"]),
            "p90_upside_risk": float(r["rt_price_p90"]),
            "actual": float(r["rt_price_actual"])
        })
        grid_list.append({
            "timestamp": ts,
            "system_demand_mw": float(r["system_demand_pred_mw"]),
            "solar_gen_mw": float(r["solar_gen_pred_mw"]),
            "wind_gen_mw": float(r["wind_gen_pred_mw"]),
            "net_load_mw": float(r["net_load_pred_mw"])
        })
        weather_list.append({
            "timestamp": ts,
            "temperature_c": float(r["temp_pred_c"]),
            "wind_speed_ms": float(r["wind_speed_pred_ms"]),
            "solar_irradiance_wm2": float(r["solar_irradiance_pred_wm2"])
        })
        if float(r["spike_probability"]) > 0.40:
            spike_hours.append(ts)

    return {
        "timestamp_generated": datetime.utcnow().isoformat(),
        "horizon_hours": horizon_hours,
        "da_prices": da_list,
        "rt_prices": rt_list,
        "grid_supply_demand": grid_list,
        "weather": weather_list,
        "spike_risk_hours": spike_hours
    }

@app.get("/trading/signals", response_model=List[TradingSignalItem])
def get_trading_signals(horizon_hours: int = Query(default=24, ge=1, le=72)):
    """
    Produces actionable algorithmic trading order tickets (DEC/INC Virtual Bids and BESS Dispatch)
    consumed directly by trading gateways and execution routers.
    """
    feed = pipeline.generate_trading_feed(horizon_hours=horizon_hours)
    signals = trading_engine.generate_virtual_bids(feed)
    return [
        TradingSignalItem(
            timestamp=s.timestamp,
            asset_id=s.asset_id,
            action=s.action,
            volume_mw=s.volume_mw,
            expected_spread=s.expected_spread,
            da_price_forecast=s.da_price_forecast,
            rt_price_forecast=s.rt_price_forecast,
            spike_risk=s.spike_risk,
            rationale=s.rationale
        ) for s in signals
    ]

@app.get("/trading/bess-schedule")
def get_bess_schedule(horizon_hours: int = Query(default=24, ge=1, le=72)):
    """Returns the optimal 4-hour battery charge/discharge schedule and projected PnL."""
    feed = pipeline.generate_trading_feed(horizon_hours=horizon_hours)
    bess_sched = trading_engine.optimize_bess_dispatch(feed)
    cols = ["timestamp", "rt_price_p50", "bess_charge_mw", "bess_discharge_mw", "bess_soc_mwh", "bess_soc_pct", "bess_hourly_pnl_usd"]
    return json.loads(bess_sched[cols].to_json(orient="records"))

@app.get("/trading/metrics")
def get_trading_metrics():
    """Returns backtested financial Sharpe, Sortino, Win Rate, and Drawdown performance."""
    if not pipeline.is_fitted:
        pipeline.train()
    # Backtest on test split
    split_idx = int(len(pipeline.df_features) * 0.8)
    test_split = pipeline.df_features.iloc[split_idx:].copy().reset_index(drop=True)
    perf = trading_engine.backtest_strategies(test_split)
    # Remove large dict for concise API payload
    perf.pop("daily_pnl", None)
    return perf

@app.get("/stream/ticks")
async def stream_price_ticks():
    """
    Server-Sent Events (SSE) streaming real-time simulated order book & price forecast updates
    to live trading desk terminals at low latency.
    """
    async def event_generator():
        feed = pipeline.generate_trading_feed(horizon_hours=12)
        for _, row in feed.iterrows():
            payload = {
                "tick_time": datetime.utcnow().isoformat(),
                "node": config.market_node,
                "da_price_median": float(row["da_price_p50"]),
                "rt_price_median": float(row["rt_price_p50"]),
                "rt_p90_spike_risk": float(row["rt_price_p90"]),
                "spike_probability": float(row["spike_probability"]),
                "status": "LIVE_FEED"
            }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
