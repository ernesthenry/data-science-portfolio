# Project 6: Data Pipeline & MLOps System

## Overview
This project demonstrates the transition from a local notebook to a production-ready ML pipeline. It focuses on automation, tracking, and reproducibility.

## Features
- **Orchestration**: Using a simplified script to simulate an Airflow pipeline (Extraction -> Transformation -> Training).
- **Experiment Tracking**: Integration with MLflow to track parameters, metrics, and models.
- **Environment Management**: Dockerfile for consistent deployment.
- **Monitoring**: Basic model health check scripts.

## Tech Stack
- Python
- MLflow
- Docker
- SQL (PostgreSQL)

## How to Run
1. Navigate to `6_ml_pipeline_mlops`
2. View Pipeline Trace: `notebooks/01_pipeline_tracking.ipynb`
3. Run the automated pipeline: `python src/pipeline.py`
