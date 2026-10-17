"""
Energy Market, Grid Supply/Demand, and Weather Time-Series Data Generator.
Vectorized, ultra-fast generation of physically grounded electricity markets (modelled on ISO/RTOs)
featuring renewable Duck Curves, weather-driven thermal response, and non-linear LMP price spikes.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_DIR, "energy_grid.db")
SCHEMA_PATH = os.path.join(DATA_DIR, "schema.sql")
PARQUET_PATH = os.path.join(DATA_DIR, "energy_dataset.parquet")
CSV_PATH = os.path.join(DATA_DIR, "energy_dataset.csv")

def generate_energy_time_series(start_date="2024-01-01", n_days=730, seed=42) -> pd.DataFrame:
    """
    Generates 2 years of hourly synthetic data with physical weather-grid-price coupling.
    Total records: 730 * 24 = 17,520 hours. Vectorized with NumPy for near-instant generation.
    """
    np.random.seed(seed)
    n_hours = n_days * 24
    timestamps = pd.date_range(start=start_date, periods=n_hours, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    # Temporal features
    day_of_year = df["timestamp"].dt.dayofyear.values
    hour = df["timestamp"].dt.hour.values
    dayofweek = df["timestamp"].dt.dayofweek.values
    is_weekend = (dayofweek >= 5).astype(float)

    # 1. Weather Generation (Vectorized)
    annual_temp = 18.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    diurnal_temp = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    # Filtered AR(1) noise via lfilter or cumulative sum decay
    noise_raw = np.random.normal(0, 0.8, n_hours)
    # Fast AR(1) vector simulation
    ar_decay = 0.95
    # Approximate AR(1) quickly
    temp_noise = np.convolve(noise_raw, ar_decay ** np.arange(48), mode="full")[:n_hours] * (1 - ar_decay)
    df["temperature_c"] = np.round(annual_temp + diurnal_temp + temp_noise, 2)

    # Relative humidity (inversely correlated with temperature)
    humidity_base = 65 - (df["temperature_c"].values - 18) * 1.5 + np.random.normal(0, 5, n_hours)
    df["relative_humidity"] = np.clip(humidity_base, 15, 98).round(1)

    # Cloud cover (Markovian random walk smoothed)
    cloud_noise = np.random.normal(0, 10, n_hours)
    clouds = np.clip(35.0 + np.convolve(cloud_noise, 0.9 ** np.arange(24), mode="full")[:n_hours], 0, 100)
    df["cloud_cover_pct"] = np.round(clouds, 1)

    # Solar Irradiance (W/m2): Zero at night, peaks noon, attenuated by clouds
    solar_zenith = np.maximum(0, np.sin(np.pi * (hour - 6) / 12.0))
    solar_zenith = np.where((hour >= 6) & (hour <= 18), solar_zenith, 0.0)
    seasonal_solar = 0.75 + 0.25 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    cloud_attenuation = (1.0 - 0.75 * (df["cloud_cover_pct"].values / 100.0))
    df["solar_irradiance_wm2"] = np.round(solar_zenith * 950.0 * seasonal_solar * cloud_attenuation, 1)

    # Wind Speed (m/s): Weibull-like, peaks overnight & evening
    diurnal_wind = 1.8 * np.sin(2 * np.pi * (hour - 1) / 24.0)
    wind_raw = np.random.exponential(scale=3.5, size=n_hours)
    wind_smooth = 4.5 + np.convolve(wind_raw - 3.5, 0.88 ** np.arange(18), mode="full")[:n_hours] * 0.15
    df["wind_speed_ms"] = np.round(np.clip(wind_smooth + diurnal_wind, 0.8, 25.0), 2)

    # 2. Grid Supply and Demand (Vectorized)
    base_demand = 32000.0 - is_weekend * 4500.0
    cooling_demand = np.maximum(0.0, df["temperature_c"].values - 20.0) * 850.0
    heating_demand = np.maximum(0.0, 14.0 - df["temperature_c"].values) * 550.0
    daily_profile = (
        2500.0 * np.sin(2 * np.pi * (hour - 4) / 24.0) +
        3500.0 * np.sin(2 * np.pi * (hour - 14) / 12.0)
    )
    demand_noise = np.random.normal(0, 500, n_hours)
    df["system_demand_mw"] = np.round(base_demand + cooling_demand + heating_demand + daily_profile + demand_noise, 1)

    # Solar generation (Capacity = 16,000 MW)
    solar_capacity = 16000.0
    df["solar_generation_mw"] = np.round((df["solar_irradiance_wm2"].values / 1000.0) * solar_capacity * 0.95, 1)

    # Wind generation (Capacity = 22,000 MW)
    wind_capacity = 22000.0
    norm_wind = np.clip((df["wind_speed_ms"].values - 3.0) / 9.0, 0.0, 1.0)
    wind_cf = norm_wind ** 2.2
    df["wind_generation_mw"] = np.round(np.clip(wind_cf * wind_capacity + np.random.normal(0, 250, n_hours), 0, wind_capacity), 1)

    # Net Load = System Demand - (Solar + Wind)
    df["net_load_mw"] = np.round(df["system_demand_mw"].values - (df["solar_generation_mw"].values + df["wind_generation_mw"].values), 1)

    # Spinning reserves
    max_thermal = 48000.0
    df["spinning_reserves_mw"] = np.round(
        np.clip(max_thermal - df["net_load_mw"].values + np.random.normal(0, 400, n_hours), 1600, 12000), 1
    )

    # 3. Market Prices ($/MWh)
    net_load_norm = (df["net_load_mw"].values - 12000.0) / 25000.0
    merit_order_price = 28.0 + 35.0 * net_load_norm + 45.0 * np.maximum(0.0, net_load_norm - 0.6) ** 2.5
    gas_cost_factor = 1.0 + 0.15 * np.sin(2 * np.pi * day_of_year / 365.25)
    
    # Day-Ahead LMP
    da_noise = np.random.normal(0, 3.0, n_hours)
    df["day_ahead_lmp"] = np.round(np.clip(merit_order_price * gas_cost_factor + da_noise, 5.0, 320.0), 2)

    # Real-Time LMP
    rt_noise = np.random.normal(0, 7.0, n_hours)
    scarcity_mask = df["spinning_reserves_mw"].values < 2800.0
    spike_mult = np.where(scarcity_mask, np.random.uniform(2.5, 6.0, n_hours), 1.0)
    negative_mask = (df["net_load_mw"].values < 8000.0) & (df["solar_generation_mw"].values > 11000.0)
    
    rt_lmp = df["day_ahead_lmp"].values * spike_mult + rt_noise
    rt_lmp = np.where(negative_mask, np.random.uniform(-30.0, -1.0, n_hours), rt_lmp)
    df["real_time_lmp"] = np.round(np.clip(rt_lmp, -35.0, 1650.0), 2)

    df["congestion_price"] = np.round(np.where(df["real_time_lmp"].values > 120, np.random.uniform(5, 40, n_hours), 0.0), 2)
    df["loss_price"] = np.round(df["real_time_lmp"].values * 0.02, 2)
    df["zone_id"] = "ZONE_NORTH"
    df["market_node"] = "HUB_NORTH"

    # Format timestamp to ISO string
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df

def init_db_and_load(df: pd.DataFrame):
    """Initializes SQLite database with schema and loads generated data."""
    print(f"Initializing database at: {DB_PATH}")
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
        except OSError:
            pass

    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())

    # Fast bulk insert
    weather_cols = ["timestamp", "zone_id", "temperature_c", "wind_speed_ms", 
                    "solar_irradiance_wm2", "relative_humidity", "cloud_cover_pct"]
    df[weather_cols].to_sql("weather_observations", conn, if_exists="append", index=False)

    grid_cols = ["timestamp", "zone_id", "system_demand_mw", "solar_generation_mw", 
                 "wind_generation_mw", "net_load_mw", "spinning_reserves_mw"]
    df[grid_cols].to_sql("grid_telemetry", conn, if_exists="append", index=False)

    market_cols = ["timestamp", "market_node", "day_ahead_lmp", "real_time_lmp", 
                   "congestion_price", "loss_price"]
    df[market_cols].to_sql("market_prices", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()
    print("Database successfully seeded with 17,520 hourly rows.")

    # Save to parquet and csv
    df.to_parquet(PARQUET_PATH, index=False)
    df.head(2000).to_csv(CSV_PATH, index=False)
    print(f"Saved dataset to {PARQUET_PATH} and {CSV_PATH}")

if __name__ == "__main__":
    print("Generating 2-year physical energy time series dataset...")
    data = generate_energy_time_series()
    init_db_and_load(data)
    print("Data generation complete!")
