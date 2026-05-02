import time
import random

class MLflowMock:
    """Mock MLflow for demonstration of experiment tracking."""
    def start_run(self):
        print("Starting MLflow Run...")
        return self

    def log_param(self, key, value):
        print(f"MLflow Param: {key} = {value}")

    def log_metric(self, key, value):
        print(f"MLflow Metric: {key} = {value:.4f}")

    def log_artifact(self, local_path):
        print(f"MLflow Artifact saved: {local_path}")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): print("Ending MLflow Run.")

def run_ml_pipeline():
    mlflow = MLflowMock()
    
    with mlflow.start_run():
        # Step 1: Preprocessing
        print("Preprocessing data...")
        time.sleep(1)
        mlflow.log_param("data_version", "v1.2")
        
        # Step 2: Training
        print("Training model...")
        params = {"n_estimators": 100, "max_depth": 5, "lr": 0.01}
        for k, v in params.items():
            mlflow.log_param(k, v)
        
        time.sleep(1)
        
        # Step 3: Evaluation
        accuracy = 0.85 + random.uniform(0, 0.1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", accuracy - 0.05)
        
        print(f"Final Model Accuracy: {accuracy:.4f}")
        mlflow.log_artifact("models/model.pkl")

if __name__ == "__main__":
    run_ml_pipeline()
