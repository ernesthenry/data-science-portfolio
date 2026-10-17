-- Energy Grid & Market Trading Database Schema
-- Production Relational & Time-Series DDL (SQLite & Timescale/PostgreSQL Compatible)

-- 1. Weather Telemetry & Numerical Weather Predictions
CREATE TABLE IF NOT EXISTS weather_observations (
    timestamp TEXT NOT NULL,
    zone_id TEXT NOT NULL DEFAULT 'ZONE_NORTH',
    temperature_c REAL NOT NULL,
    wind_speed_ms REAL NOT NULL,
    solar_irradiance_wm2 REAL NOT NULL,
    relative_humidity REAL NOT NULL,
    cloud_cover_pct REAL NOT NULL,
    PRIMARY KEY (timestamp, zone_id)
);

-- 2. Grid Supply and Demand Telemetry
CREATE TABLE IF NOT EXISTS grid_telemetry (
    timestamp TEXT NOT NULL,
    zone_id TEXT NOT NULL DEFAULT 'ZONE_NORTH',
    system_demand_mw REAL NOT NULL,
    solar_generation_mw REAL NOT NULL,
    wind_generation_mw REAL NOT NULL,
    net_load_mw REAL NOT NULL,
    spinning_reserves_mw REAL NOT NULL,
    PRIMARY KEY (timestamp, zone_id)
);

-- 3. Wholesale Power Market Prices (LMP: Locational Marginal Price)
CREATE TABLE IF NOT EXISTS market_prices (
    timestamp TEXT NOT NULL,
    market_node TEXT NOT NULL DEFAULT 'HUB_NORTH',
    day_ahead_lmp REAL NOT NULL,
    real_time_lmp REAL NOT NULL,
    congestion_price REAL NOT NULL DEFAULT 0.0,
    loss_price REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (timestamp, market_node)
);

-- 4. Forecasting Model Runs Audit Table
CREATE TABLE IF NOT EXISTS forecast_runs (
    forecast_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    target_variable TEXT NOT NULL,
    horizon_hours INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    execution_time_ms REAL NOT NULL,
    metadata_json TEXT
);

-- 5. Time Series Forecast Points with Quantile Intervals (P10, P50, P90)
CREATE TABLE IF NOT EXISTS forecast_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    target_variable TEXT NOT NULL,
    predicted_p10 REAL NOT NULL,
    predicted_p50 REAL NOT NULL,
    predicted_p90 REAL NOT NULL,
    spike_probability REAL NOT NULL DEFAULT 0.0,
    FOREIGN KEY (forecast_id) REFERENCES forecast_runs(forecast_id)
);

-- 6. Algorithmic Trading Signals for Trading Platform
CREATE TABLE IF NOT EXISTS trading_signals (
    signal_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    action TEXT NOT NULL, -- 'VIRTUAL_BID_BUY', 'VIRTUAL_BID_SELL', 'BESS_CHARGE', 'BESS_DISCHARGE', 'HOLD'
    target_volume_mw REAL NOT NULL,
    expected_spread REAL NOT NULL,
    da_price_forecast REAL NOT NULL,
    rt_price_forecast REAL NOT NULL,
    risk_score REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'ACTIVE'
);

-- 7. Trade Executions & Performance Tracking
CREATE TABLE IF NOT EXISTS trade_executions (
    execution_id TEXT PRIMARY KEY,
    signal_id TEXT,
    timestamp TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    action TEXT NOT NULL,
    volume_mw REAL NOT NULL,
    execution_price REAL NOT NULL,
    settlement_price REAL NOT NULL,
    gross_pnl_usd REAL NOT NULL,
    transaction_cost_usd REAL NOT NULL,
    net_pnl_usd REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'SETTLED',
    FOREIGN KEY (signal_id) REFERENCES trading_signals(signal_id)
);

-- Create Indices for High-Throughput Time-Series Queries
CREATE INDEX IF NOT EXISTS idx_weather_ts ON weather_observations(timestamp);
CREATE INDEX IF NOT EXISTS idx_grid_ts ON grid_telemetry(timestamp);
CREATE INDEX IF NOT EXISTS idx_market_ts ON market_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_forecast_ts ON forecast_points(timestamp, target_variable);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON trading_signals(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trade_executions(timestamp);
