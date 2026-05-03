# Project 3: A/B Testing & Experimentation Platform

## Overview
A toolkit for designing, running, and analyzing A/B tests. This project provides a statistical framework to evaluate product changes rigorously.

## Features
- **Power Analysis**: Determining sample sizes for experiments.
- **Hypothesis Testing**: T-tests, Chi-Squared tests, and Z-tests.
- **Bayesian Analysis**: Using Beta distributions to model conversion rates.
- **Dashboard**: Interactive results visualization.

## Tech Stack
- Python
- SciPy, Statsmodels
- Streamlit
- Plotly

## Recommended dataset
**Primary:** **Marketing A/B testing** — campaign / conversion style experiments for frequentist and Bayesian demos.

- [Kaggle: Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing)

**Alternative:** **E-commerce A/B testing** — user-level conversion logs.

- [Kaggle: A/B Testing Dataset](https://www.kaggle.com/datasets/zhangluyuan/ab-testing)

## How to Run
1. Navigate to `3_ab_testing_engine`
2. Install dependencies: `pip install -r requirements.txt`
3. Statistical Walkthrough: `notebooks/01_statistical_analysis.ipynb`
4. Run the interactive dashboard: `streamlit run app/main.py`
