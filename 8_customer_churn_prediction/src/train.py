"""Train a churn classifier on synthetic telco-like data or a CSV you provide."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

from feature_engineering import prepare_features, tenure_monthly_interaction


def synthetic_telco_churn(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.normal(65, 30, size=n).clip(20, 120)
    total = tenure * monthly + rng.normal(0, 200, size=n)
    partner = rng.choice(["Yes", "No"], size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    logit = (
        -2.0
        + 0.035 * (monthly / 10)
        - 0.04 * tenure
        + (contract == "Month-to-month").astype(float) * 1.1
        + (partner == "No").astype(float) * 0.35
        + rng.normal(0, 0.4, size=n)
    )
    p = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": np.maximum(total, 0),
            "Partner": partner,
            "Contract": contract,
            "Churn": churn,
        }
    )


def train_churn_model(
    data_path: str | None = None,
    target_col: str = "Churn",
    use_smote: bool = False,
) -> LGBMClassifier:
    if data_path:
        df = pd.read_csv(data_path)
        df = tenure_monthly_interaction(df)
    else:
        df = tenure_monthly_interaction(synthetic_telco_churn())

    X, y = prepare_features(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if use_smote:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    print("Churn model — holdout metrics:\n")
    print(classification_report(y_test, preds, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn model (CSV optional).")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV (e.g. Kaggle Telco Churn). If omitted, uses synthetic demo data.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="Churn",
        help="Name of the binary target column (default: Churn).",
    )
    parser.add_argument(
        "--smote",
        action="store_true",
        help="Apply SMOTE oversampling on the training split only.",
    )
    args = parser.parse_args()
    if args.data and not Path(args.data).is_file():
        raise SystemExit(f"File not found: {args.data}")
    train_churn_model(data_path=args.data, target_col=args.target, use_smote=args.smote)


if __name__ == "__main__":
    main()
