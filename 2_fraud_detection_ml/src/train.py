"""Train fraud classifiers: Random Forest, XGBoost, LightGBM with imbalance handling."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_engineering import aggregate_features, create_time_features, handle_missing_values


def _load_or_synthesize_fraud_frame(data_path: Optional[str]) -> pd.DataFrame:
    if data_path:
        df = pd.read_csv(data_path)
        if "is_fraud" not in df.columns and "Class" in df.columns:
            df = df.rename(columns={"Class": "is_fraud"})
        if "is_fraud" not in df.columns:
            raise ValueError("Expected a target column named is_fraud or Class.")
        return df

    print("Generating synthetic fraud data (imbalanced; amount weakly correlated with fraud)...")
    np.random.seed(42)
    n_samples = 10000
    amount = np.random.exponential(100, n_samples)
    high_risk = amount > np.percentile(amount, 93)
    p_fraud = np.where(high_risk, 0.12, 0.015)
    is_fraud = (np.random.random(n_samples) < p_fraud).astype(int)
    data = {
        "transaction_id": range(n_samples),
        "user_id": np.random.randint(1, 500, n_samples),
        "amount": amount,
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="min"),
        "device": np.random.choice(["mobile", "desktop", "tablet"], n_samples),
        "is_fraud": is_fraud,
    }
    return pd.DataFrame(data)


def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = create_time_features(df.copy(), "timestamp")
    df = aggregate_features(df, "user_id", "amount")
    df = handle_missing_values(df)
    df = pd.get_dummies(df, columns=["device"], drop_first=False)
    drop_cols = [c for c in ("is_fraud", "timestamp", "transaction_id") if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["is_fraud"].astype(int)
    return X, y


def _resample(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    if name == "none":
        return X_train, y_train
    if name == "smote":
        Xr, yr = SMOTE(random_state=42).fit_resample(X_train, y_train)
        return pd.DataFrame(Xr, columns=X_train.columns), pd.Series(yr, name=y_train.name)
    if name == "adasyn":
        try:
            Xr, yr = ADASYN(random_state=42).fit_resample(X_train, y_train)
        except ValueError:
            print("ADASYN failed (insufficient minority density); falling back to SMOTE.")
            Xr, yr = SMOTE(random_state=42).fit_resample(X_train, y_train)
        return pd.DataFrame(Xr, columns=X_train.columns), pd.Series(yr, name=y_train.name)
    raise ValueError(f"Unknown resample strategy: {name}")


def _scale_pos_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = max((y == 1).sum(), 1)
    return float(neg / pos)


def _build_models(scale_pos_w: float) -> Dict[str, Callable[[], object]]:
    return {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_w,
            random_state=42,
            eval_metric="logloss",
        ),
        "lightgbm": lambda: LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            num_leaves=48,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
    }


def train_fraud_models(
    data_path: Optional[str] = None,
    resample: str = "none",
) -> List[Tuple[str, object, float]]:
    df = _load_or_synthesize_fraud_frame(data_path)
    X, y = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = _resample(resample, X_train, y_train)

    print(
        f"Train size={len(X_train)} (fraud rate {y_train.mean():.2%}), "
        f"resample={resample}"
    )

    spw = _scale_pos_weight(y_train)
    results: List[Tuple[str, object, float]] = []
    for name, factory in _build_models(spw).items():
        model = factory()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_test, probs)
        results.append((name, model, auc))
        print(f"\n=== {name} (ROC-AUC={auc:.4f}) ===")
        print(classification_report(y_test, preds, digits=4))

    best_name, _, best_auc = max(results, key=lambda t: t[2])
    print(f"\nBest by ROC-AUC: {best_name} ({best_auc:.4f})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud detection training (multi-model).")
    parser.add_argument("--data", type=str, default=None, help="CSV with is_fraud or Class target.")
    parser.add_argument(
        "--resample",
        choices=("none", "smote", "adasyn"),
        default="none",
        help="Training-only oversampling (compare to class weights / scale_pos_weight).",
    )
    args = parser.parse_args()
    if args.data and not Path(args.data).is_file():
        raise SystemExit(f"File not found: {args.data}")
    train_fraud_models(data_path=args.data, resample=args.resample)


if __name__ == "__main__":
    main()
