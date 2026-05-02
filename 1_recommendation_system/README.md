# Project 1: Large-Scale Recommendation System

## Overview
This project implements a multi-stage recommendation engine using the MovieLens dataset. It covers collaborative filtering, matrix factorization, and neural-based recommenders.

## Features
- **Exploratory Data Analysis (EDA)**: Understanding rating distributions and user-item sparsity.
- **Baseline Models**: Global mean, item-based, and user-based collaborative filtering.
- **Matrix Factorization**: SVD and ALS implementation.
- **Neural Recommenders**: Implementing Neural Collaborative Filtering (NCF) using PyTorch.

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch
- Surprise (for matrix factorization)

## How to Run
1. Navigate to this directory: `cd 1_recommendation_system`
2. Install dependencies: `pip install -r requirements.txt`
3. Explore the data: `notebooks/01_eda.ipynb`
4. Train and evaluate models: `notebooks/02_modeling_evaluation.ipynb`
5. Production training: `python src/train.py`
