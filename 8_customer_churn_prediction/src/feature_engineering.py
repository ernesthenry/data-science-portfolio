"""Feature helpers for telco-style churn modeling."""

from typing import Tuple

import pandas as pd


def tenure_monthly_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction between tenure and monthly charges (common churn signal)."""
    out = df.copy()
    if "tenure" in out.columns and "MonthlyCharges" in out.columns:
        out["tenure_x_monthly"] = out["tenure"] * out["MonthlyCharges"]
    elif "tenure" in out.columns and "monthly_charges" in out.columns:
        out["tenure_x_monthly"] = out["tenure"] * out["monthly_charges"]
    return out


def encode_contract_tenure_risk(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot contract type; short contracts often correlate with churn."""
    out = df.copy()
    col = None
    for c in ("Contract", "contract"):
        if c in out.columns:
            col = c
            break
    if col:
        out = pd.get_dummies(out, columns=[col], prefix="contract")
    return out


def prepare_features(df: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    """Drop leakage-prone ids, encode categoricals, return X and y."""
    data = df.copy()
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    raw = data[target_col]
    if pd.api.types.is_numeric_dtype(raw) and set(raw.dropna().unique().astype(int)).issubset({0, 1}):
        y = raw.astype(int)
    else:
        mapped = raw.astype(str).str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}
        )
        y = mapped.fillna(0).astype(int)

    drop_cols = {target_col, "customerID", "customer_id", "id"}
    X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype(str)

    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    return X, y
