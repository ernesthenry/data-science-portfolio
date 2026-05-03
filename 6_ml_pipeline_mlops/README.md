# Project 6: Data Pipeline & MLOps System

## Overview
This project demonstrates the transition from a local notebook to a production-ready ML pipeline. It focuses on automation, tracking, and reproducibility.

## Features
- **Orchestration**: Scripted pipeline (Extract → Transform → Train → Evaluate) suitable to wrap in Airflow or Cron.
- **Experiment Tracking**: Mock MLflow-style logging of params and metrics (swap in real `mlflow` when needed).
- **Environment Management**: `Dockerfile` for reproducible runs.
- **Monitoring**: Extend with scheduled metrics checks on top of logged `r2` / `rmse`.

## Tech Stack
- Python
- MLflow
- Docker
- SQL (PostgreSQL)

## Recommended dataset
**Primary (fast pipeline demos):** **California Housing** — regression targets, small footprint, easy to version and replay.

- [scikit-learn: `fetch_california_housing`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)

**Alternative (high-volume ETL / batch feel):** **NYC Taxi Fare Prediction** — large tabular extracts for staging, validation, and monitoring patterns.

- [Kaggle: New York City Taxi Fare Prediction](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

## How to Run
1. Navigate to `6_ml_pipeline_mlops`
2. Install dependencies: `pip install -r requirements.txt`
3. View pipeline notebook: `notebooks/01_pipeline_tracking.ipynb`
4. Run the pipeline locally: `python src/pipeline.py`
5. **Docker:** `docker build -t mlops-pipeline .` then `docker run --rm mlops-pipeline`
