"""ETL-style ML pipeline with mock MLflow logging and a real sklearn training step."""

from __future__ import annotations

import time
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class MLflowMock:
    """Mock MLflow for demonstration of experiment tracking."""

    def start_run(self):
        print("Starting MLflow Run...")
        return self

    def log_param(self, key, value):
        print(f"MLflow Param: {key} = {value}")

    def log_metric(self, key, value):
        print(f"MLflow Metric: {key} = {value:.6f}")

    def log_artifact(self, local_path):
        print(f"MLflow Artifact saved: {local_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Ending MLflow Run.")


def run_ml_pipeline() -> None:
    mlflow = MLflowMock()

    with mlflow.start_run():
        # Extract (cache data inside repo for Docker/CI-friendly paths)
        data_home = Path(__file__).resolve().parents[2] / ".sklearn_data"
        data_home.mkdir(parents=True, exist_ok=True)
        print("Extract: loading California housing (sklearn)...")
        X, y = fetch_california_housing(return_X_y=True, data_home=str(data_home))
        mlflow.log_param("data_source", "sklearn.datasets.fetch_california_housing")
        mlflow.log_param("n_samples", X.shape[0])
        mlflow.log_param("n_features", X.shape[1])

        # Transform / load
        time.sleep(0.3)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        mlflow.log_param("test_size", 0.2)

        # Train
        print("Train: Ridge regression...")
        alpha = 1.0
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", alpha)
        time.sleep(0.3)
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        print(f"Holdout R²: {r2:.4f} | RMSE: {rmse:.4f}")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        artifact = models_dir / "ridge_california.txt"
        artifact.write_text(f"r2={r2:.6f}\nrmse={rmse:.6f}\n", encoding="utf-8")
        mlflow.log_artifact(str(artifact))


if __name__ == "__main__":
    run_ml_pipeline()
