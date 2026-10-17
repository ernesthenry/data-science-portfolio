-- Production SQL Analytical Queries for Energy Quant & Trading Operations
-- Demonstrating Advanced SQL: Window Functions, CTEs, Time-Bucketing, and Quantitative Metrics

-- ============================================================================
-- 1. Day-Ahead vs Real-Time Price Arbitrage Spread with Rolling Volatility
-- Identifies statistical arbitrage opportunities between DA and RT markets.
-- ============================================================================
WITH PriceSpreads AS (
    SELECT 
        timestamp,
        market_node,
        day_ahead_lmp,
        real_time_lmp,
        (real_time_lmp - day_ahead_lmp) AS spread_rt_minus_da,
        CASE 
            WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 7 AND 22 THEN 'PEAK'
            ELSE 'OFF_PEAK'
        END AS peak_category,
        strftime('%Y-%m-%d', timestamp) AS trade_date
    FROM market_prices
)
SELECT 
    trade_date,
    peak_category,
    ROUND(AVG(day_ahead_lmp), 2) AS avg_da_price,
    ROUND(AVG(real_time_lmp), 2) AS avg_rt_price,
    ROUND(AVG(spread_rt_minus_da), 2) AS avg_spread,
    ROUND(MAX(spread_rt_minus_da), 2) AS max_spike_spread,
    ROUND(MIN(spread_rt_minus_da), 2) AS min_drop_spread,
    -- Daily rolling standard deviation of price spread
    ROUND(
        SQRT(
            AVG(spread_rt_minus_da * spread_rt_minus_da) - 
            (AVG(spread_rt_minus_da) * AVG(spread_rt_minus_da))
        ), 
        2
    ) AS spread_volatility
FROM PriceSpreads
GROUP BY trade_date, peak_category
ORDER BY trade_date DESC, peak_category ASC
LIMIT 30;


-- ============================================================================
-- 2. Duck Curve Detection: Renewable Penetration & 3-Hour Net Load Ramp Rates
-- Critical for grid reliability and gas peaking plant dispatch.
-- ============================================================================
WITH HourlyNetLoad AS (
    SELECT 
        g.timestamp,
        g.zone_id,
        g.system_demand_mw,
        g.solar_generation_mw,
        g.wind_generation_mw,
        g.net_load_mw,
        ROUND((g.solar_generation_mw + g.wind_generation_mw) * 100.0 / NULLIF(g.system_demand_mw, 0), 2) AS renewable_penetration_pct,
        -- Net load 3 hours prior using window LAG function
        LAG(g.net_load_mw, 3) OVER (
            PARTITION BY g.zone_id 
            ORDER BY g.timestamp
        ) AS net_load_3h_ago,
        m.real_time_lmp
    FROM grid_telemetry g
    JOIN market_prices m ON g.timestamp = m.timestamp
)
SELECT 
    timestamp,
    zone_id,
    system_demand_mw,
    (solar_generation_mw + wind_generation_mw) AS total_renewable_mw,
    renewable_penetration_pct,
    net_load_mw,
    ROUND(net_load_mw - net_load_3h_ago, 2) AS ramp_3h_mw,
    real_time_lmp,
    CASE 
        WHEN (net_load_mw - net_load_3h_ago) > 8000 THEN 'SEVERE_EVENING_RAMP'
        WHEN (net_load_mw - net_load_3h_ago) > 5000 THEN 'MODERATE_RAMP'
        WHEN (net_load_mw - net_load_3h_ago) < -5000 THEN 'SOLAR_MORNING_DROP'
        ELSE 'NORMAL'
    END AS ramp_severity
FROM HourlyNetLoad
WHERE net_load_3h_ago IS NOT NULL
ORDER BY ramp_3h_mw DESC
LIMIT 50;


-- ============================================================================
-- 3. Model Forecast Telemetry: Rolling 7-Day MAE, RMSE, and Quantile Calibration
-- Monitors real-time forecast degradation against settlement prices.
-- ============================================================================
WITH EvaluatedPoints AS (
    SELECT 
        f.timestamp,
        f.target_variable,
        f.predicted_p10,
        f.predicted_p50,
        f.predicted_p90,
        m.real_time_lmp AS actual_value,
        (m.real_time_lmp - f.predicted_p50) AS forecast_error,
        ABS(m.real_time_lmp - f.predicted_p50) AS abs_error,
        (m.real_time_lmp - f.predicted_p50) * (m.real_time_lmp - f.predicted_p50) AS sq_error,
        -- Check if actual fell within predicted 80% confidence interval [P10, P90]
        CASE 
            WHEN m.real_time_lmp BETWEEN f.predicted_p10 AND f.predicted_p90 THEN 1 
            ELSE 0 
        END AS within_interval_80
    FROM forecast_points f
    JOIN market_prices m ON f.timestamp = m.timestamp
    WHERE f.target_variable = 'RT_LMP'
)
SELECT 
    strftime('%Y-%W', timestamp) AS year_week,
    COUNT(*) AS n_forecasts,
    ROUND(AVG(abs_error), 2) AS mae_usd,
    ROUND(SQRT(AVG(sq_error)), 2) AS rmse_usd,
    ROUND(AVG(within_interval_80) * 100.0, 1) AS empirical_coverage_pct, -- Expected: ~80%
    ROUND(AVG(CASE WHEN forecast_error > 0 THEN forecast_error * 0.9 ELSE forecast_error * -0.1 END), 2) AS pinball_loss_p90
FROM EvaluatedPoints
GROUP BY year_week
ORDER BY year_week DESC
LIMIT 12;


-- ============================================================================
-- 4. Battery Energy Storage (BESS) 4-Hour Daily Arbitrage Opportunity Matrix
-- Computes the theoretical highest spread between cheapest 4h and priciest 4h.
-- ============================================================================
WITH RankedHourlyPrices AS (
    SELECT 
        strftime('%Y-%m-%d', timestamp) AS trade_date,
        CAST(strftime('%H', timestamp) AS INTEGER) AS hour_of_day,
        real_time_lmp,
        ROW_NUMBER() OVER (
            PARTITION BY strftime('%Y-%m-%d', timestamp) 
            ORDER BY real_time_lmp ASC
        ) AS rank_cheapest,
        ROW_NUMBER() OVER (
            PARTITION BY strftime('%Y-%m-%d', timestamp) 
            ORDER BY real_time_lmp DESC
        ) AS rank_expensive
    FROM market_prices
),
DailyArbitrage AS (
    SELECT 
        trade_date,
        -- Average of 4 cheapest hours (charge window)
        AVG(CASE WHEN rank_cheapest <= 4 THEN real_time_lmp END) AS avg_charge_cost,
        -- Average of 4 highest priced hours (discharge window)
        AVG(CASE WHEN rank_expensive <= 4 THEN real_time_lmp END) AS avg_discharge_revenue
    FROM RankedHourlyPrices
    GROUP BY trade_date
)
SELECT 
    trade_date,
    ROUND(avg_charge_cost, 2) AS charge_price_mwh,
    ROUND(avg_discharge_revenue, 2) AS discharge_price_mwh,
    ROUND(avg_discharge_revenue - avg_charge_cost, 2) AS gross_spread,
    -- Assuming 85% round-trip efficiency (RTE)
    ROUND((avg_discharge_revenue * 0.85) - avg_charge_cost, 2) AS net_spread_85pct_rte,
    -- 100MW / 400MWh battery daily potential profit
    ROUND(((avg_discharge_revenue * 0.85) - avg_charge_cost) * 400, 2) AS est_daily_bess_pnl_usd
FROM DailyArbitrage
WHERE avg_charge_cost IS NOT NULL AND avg_discharge_revenue IS NOT NULL
ORDER BY net_spread_85pct_rte DESC
LIMIT 30;
