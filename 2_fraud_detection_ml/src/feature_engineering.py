import pandas as pd
import numpy as np

def create_time_features(df, timestamp_col):
    """Extract time-based features from a timestamp."""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['hour'] = df[timestamp_col].dt.hour
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

def aggregate_features(df, group_col, target_col):
    """Create aggregated features like mean spending per user and rolling averages."""
    # Grouped stats
    agg_df = df.groupby(group_col)[target_col].agg(['mean', 'std', 'max', 'count']).reset_index()
    agg_df.columns = [group_col, f'{group_col}_mean', f'{group_col}_std', f'{group_col}_max', f'{group_col}_count']
    df = df.merge(agg_df, on=group_col, how='left')
    
    # Rolling window stats (last 3 transactions)
    df = df.sort_values([group_col, 'timestamp'])
    df[f'{group_col}_rolling_mean_3'] = df.groupby(group_col)[target_col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df['velocity_amount'] = df['amount'] / (df[f'{group_col}_mean'] + 1e-9)
    
    return df

def handle_missing_values(df):
    """Simple imputation strategy."""
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

if __name__ == "__main__":
    # Example usage / Mock data
    data = {
        'transaction_id': range(1, 6),
        'user_id': [1, 1, 2, 1, 3],
        'amount': [100.5, 20.0, 5000.0, 15.0, 200.0],
        'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00', '2024-01-02 09:00:00', '2024-01-02 15:00:00'],
        'device': ['mobile', 'mobile', 'desktop', 'mobile', 'tablet']
    }
    df = pd.DataFrame(data)
    
    df = create_time_features(df, 'timestamp')
    df = aggregate_features(df, 'user_id', 'amount')
    df = handle_missing_values(df)
    
    print("Feature Engineering Complete. Sample Data:")
    print(df.head())
    df.to_csv('data/processed_features.csv', index=False)
