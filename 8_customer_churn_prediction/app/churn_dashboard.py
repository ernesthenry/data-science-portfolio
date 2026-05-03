"""Streamlit dashboard: synthetic or CSV churn data, LightGBM, quick metrics."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from feature_engineering import prepare_features, tenure_monthly_interaction  # noqa: E402
from train import synthetic_telco_churn  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Churn Prediction", layout="wide")
    st.title("Customer churn — demo dashboard")
    st.caption("Use synthetic data by default or upload the recommended Kaggle Telco CSV.")

    use_upload = st.checkbox("Upload my own CSV", value=False)
    target_col = st.text_input("Target column name", value="Churn")

    if use_upload:
        f = st.file_uploader("Churn CSV", type=["csv"])
        if f is None:
            st.info("Upload a CSV to train, or uncheck upload to use synthetic data.")
            return
        df = pd.read_csv(f)
    else:
        df = tenure_monthly_interaction(synthetic_telco_churn())

    st.dataframe(df.head(20), use_container_width=True)

    if not st.button("Train LightGBM"):
        return

    df = tenure_monthly_interaction(df)
    X, y = prepare_features(df, target_col=target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    with st.spinner("Training…"):
        model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    st.success(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    st.text(classification_report(y_test, preds, digits=4))

    importance = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    st.bar_chart(importance)


if __name__ == "__main__":
    main()
