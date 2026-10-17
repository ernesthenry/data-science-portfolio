"""
Configuration settings for Energy Market Time Series Forecasting & Trading System.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "energy_grid.db"
PARQUET_PATH = DATA_DIR / "energy_dataset.parquet"
CSV_PATH = DATA_DIR / "energy_dataset.csv"
MODELS_DIR = BASE_DIR / "models_saved"

@dataclass
class MarketConfig:
    zone_id: str = "ZONE_NORTH"
    market_node: str = "HUB_NORTH"
    horizon_hours: int = 24
    quantile_levels: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.90])
    
    # Grid physical capacities (MW)
    solar_capacity_mw: float = 16000.0
    wind_capacity_mw: float = 22000.0
    max_thermal_capacity_mw: float = 48000.0
    critical_reserve_mw: float = 2800.0

    # Battery Energy Storage System (BESS) Asset Specs
    bess_power_capacity_mw: float = 100.0   # 100 MW power rating
    bess_energy_capacity_mwh: float = 400.0 # 4-hour duration (400 MWh)
    bess_round_trip_efficiency: float = 0.88 # 88% RTE
    bess_degradation_cost_per_mwh: float = 3.50 # Degradation cycling cost $/MWh

    # Trading Execution Parameters
    max_position_mw: float = 150.0
    min_spread_threshold_usd: float = 6.50
    transaction_fee_per_mwh: float = 0.45

config = MarketConfig()
