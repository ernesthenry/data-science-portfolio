from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_prophet_forecast(df, periods=30):
    """Run a basic Facebook Prophet forecast."""
    print("Initializing Prophet model...")
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(df)
    
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    print(f"Forecast complete for {periods} days.")
    return m, forecast

if __name__ == "__main__":
    # Sample data
    try:
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=365, freq='D'),
            'y': np.sin(np.linspace(0, 10, 365)) * 10 + 100 + np.random.normal(0, 2, 365)
        })
        model, forecast = run_prophet_forecast(df)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    except ImportError:
        print("Prophet not installed. This is a demonstration script.")
