# Project 2: End-to-End Fraud Detection ML

## Overview
A comprehensive system to detect fraudulent financial transactions using the IEEE-CIS Fraud Detection dataset. It focuses on feature engineering and handling extreme class imbalance.

## Features
- **Feature Engineering**: Time-based features, transaction frequency, and aggregated spending metrics.
- **Handling Imbalance**: Comparison of SMOTE, ADASYN, and class weighting.
- **Modeling**: XGBoost, LightGBM, and Random Forest.
- **Monitoring**: Drift detection simulation.

## Tech Stack
- Python
- Pandas, NumPy
- XGBoost, LightGBM
- Scikit-learn
- Imbalanced-learn

## How to Run
1. Navigate to `2_fraud_detection_ml`
2. Install dependencies: `pip install -r requirements.txt`
3. Run EDA: `notebooks/01_eda.ipynb`
4. Model performance & diagnostics: `notebooks/02_model_performance.ipynb`
5. Production training: `python src/train.py`
