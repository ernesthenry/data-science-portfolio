"""
Time Series Feature Engineering for Energy Markets, Grid Supply/Demand, and Weather.
Constructs cyclical, lag, rolling window, and thermodynamic features without lookahead bias.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

def build_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds sine/cosine transformations for periodic time features."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    hour = df["timestamp"].dt.hour
    dayofweek = df["timestamp"].dt.dayofweek
    dayofyear = df["timestamp"].dt.dayofyear

    # Hour of day (24h period)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day of week (7d period)
    df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

    # Day of year (annual seasonality)
    df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

    df["is_weekend"] = dayofweek.isin([5, 6]).astype(float)
    df["is_peak_hour"] = hour.between(7, 22).astype(float)
    return df

def build_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds thermodynamic and renewable power transform features."""
    df = df.copy()
    # Cooling Degree (CDD) and Heating Degree (HDD) response
    df["cooling_degree"] = np.maximum(0.0, df["temperature_c"] - 20.0)
    df["heating_degree"] = np.maximum(0.0, 14.0 - df["temperature_c"])
    
    # Wind power approximation (normalized cubic curve between cut-in and rated)
    norm_wind = np.clip((df["wind_speed_ms"] - 3.0) / 9.0, 0.0, 1.0)
    df["wind_power_factor"] = norm_wind ** 2.2

    # Solar clear-sky interaction
    cloud_attenuation = 1.0 - 0.75 * (df["cloud_cover_pct"] / 100.0)
    df["solar_potential"] = (df["solar_irradiance_wm2"] / 1000.0) * cloud_attenuation
    return df

def build_lag_features(df: pd.DataFrame, target_col: str, lags: List[int] = [1, 2, 24, 48, 168]) -> pd.DataFrame:
    """Generates autoregressive lag features for the specified target."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)
    return df

def build_rolling_features(df: pd.DataFrame, target_col: str, windows: List[int] = [6, 24]) -> pd.DataFrame:
    """Generates rolling window statistics (mean, min, max, std) shifted to prevent leakage."""
    df = df.copy()
    for w in windows:
        # Shift by 1 hour to prevent lookahead of current observation
        shifted = df[target_col].shift(1)
        df[f"{target_col}_roll_{w}h_mean"] = shifted.rolling(w).mean()
        df[f"{target_col}_roll_{w}h_std"] = shifted.rolling(w).std()
        df[f"{target_col}_roll_{w}h_min"] = shifted.rolling(w).min()
        df[f"{target_col}_roll_{w}h_max"] = shifted.rolling(w).max()
    return df

def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the complete feature engineering pipeline for multi-horizon forecasting.
    Includes weather, grid supply/demand, market price lags, and rolling statistics.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = build_cyclical_features(df)
    df = build_weather_features(df)

    # Lag features for weather
    df = build_lag_features(df, "temperature_c", lags=[1, 2, 24])
    df = build_lag_features(df, "wind_speed_ms", lags=[1, 2, 24])
    df = build_lag_features(df, "solar_irradiance_wm2", lags=[1, 24])

    # Lag and rolling features for grid supply/demand
    df = build_lag_features(df, "system_demand_mw", lags=[1, 24, 48, 168])
    df = build_rolling_features(df, "system_demand_mw", windows=[6, 24])

    df = build_lag_features(df, "solar_generation_mw", lags=[1, 24])
    df = build_lag_features(df, "wind_generation_mw", lags=[1, 2, 24])
    df = build_lag_features(df, "net_load_mw", lags=[1, 24, 168])
    df = build_rolling_features(df, "net_load_mw", windows=[6, 24])

    # Lag and rolling features for market prices
    df = build_lag_features(df, "day_ahead_lmp", lags=[24, 48, 168])
    df = build_lag_features(df, "real_time_lmp", lags=[1, 2, 24, 48, 168])
    df = build_rolling_features(df, "real_time_lmp", windows=[6, 24])

    # Grid reserve margin stress ratio
    if "spinning_reserves_mw" in df.columns:
        df["reserve_margin_ratio"] = df["spinning_reserves_mw"] / np.maximum(df["system_demand_mw"], 1.0)

    # Drop early warm-up rows containing NaNs due to 168h (1 week) lag
    df = df.dropna().reset_index(drop=True)
    return df
