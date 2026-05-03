# Project 8: Customer Churn Prediction

## Overview
Predict which customers are likely to cancel a subscription (churn) so teams can run retention campaigns. The project follows an end-to-end flow: EDA, feature engineering, **LightGBM** training with class imbalance handling, optional **SHAP** explainability, and a small **Streamlit** dashboard for demos.

## Features
- **Telco-style feature engineering**: tenure × monthly charges, contract encoding helpers.
- **Modeling**: LightGBM with balanced class weights; ready for real CSV or synthetic demo data.
- **Explainability**: SHAP-ready (see notebook notes).
- **Dashboard**: Quick training and feature-importance view in Streamlit.

## Tech Stack
- Python, Pandas, scikit-learn, LightGBM, imbalanced-learn, SHAP, Streamlit

## Recommended dataset
**Primary (matches resume / industry standard):** **Telco Customer Churn** (IBM / Kaggle) — tabular telco subscriptions with `Churn` (Yes/No), tenure, charges, and contract type.

- [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download the CSV, place it e.g. in `8_customer_churn_prediction/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`, then:

```bash
python src/train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv --target Churn
```

**Alternative:** [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) — treat `Attrition` as the churn label (HR context).

## How to Run
1. Navigate to `8_customer_churn_prediction`
2. `pip install -r requirements.txt`
3. EDA: open `notebooks/01_eda_churn.ipynb` (set `DATA_PATH` to your CSV or use synthetic data)
4. Training: `python src/train.py` (synthetic) or `python src/train.py --data path/to/file.csv`
5. Dashboard: `streamlit run app/churn_dashboard.py`
