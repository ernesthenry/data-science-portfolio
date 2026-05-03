# Project 5: Time Series Sales Forecasting

## Overview
Predicting future retail sales using historical data. This project compares classical statistical methods with modern machine learning and deep learning approaches.

## Features
- **Decomposition**: Trend, seasonality, and noise analysis.
- **Classical Models**: ARIMA and SARIMA.
- **Prophet**: Meta's forecasting tool for handle holidays and growth.
- **Deep Learning**: LSTM and GRU for sequential forecasting.

## Tech Stack
- Python
- Statsmodels
- Prophet
- TensorFlow/Keras
- Plotly

## Recommended dataset
**Primary:** **Walmart Store Sales** (weekly sales by store and department) — classic retail forecasting benchmark.

- [Kaggle: Walmart Recruiting — Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

**Alternative:** **Store Item Demand Forecasting** — daily demand across stores.

- [Kaggle: Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only)

## How to Run
1. Navigate to `5_time_series_forecasting`
2. Install dependencies: `pip install -r requirements.txt`
3. EDA & setup: `notebooks/01_eda.ipynb`
4. Forecasting models: `notebooks/02_forecasting_models.ipynb`
5. Production prediction: `python src/forecast.py`
