import pandas as pd
import numpy as np

def prepare_time_series(df, date_col, target_col):
    """Format dataframe for Prophet or LSTM models."""
    df[date_col] = pd.to_datetime(df[date_col])
    # Prophet requires columns 'ds' and 'y'
    df_prophet = df[[date_col, target_col]].copy()
    df_prophet.columns = ['ds', 'y']
    return df_prophet.sort_values('ds')

def add_lag_features(df, target_col, lags=[1, 7, 30]):
    """Create lag features for traditional ML models."""
    df_lags = df.copy()
    for lag in lags:
        df_lags[f'lag_{lag}'] = df_lags[target_col].shift(lag)
    return df_lags.fillna(0)

if __name__ == "__main__":
    dates = pd.date_range('2023-01-01', periods=100)
    sales = np.random.randint(50, 200, 100)
    df = pd.DataFrame({'date': dates, 'sales': sales})
    
    prepared = prepare_time_series(df, 'date', 'sales')
    print("Prepared Data (Top 5):")
    print(prepared.head())
