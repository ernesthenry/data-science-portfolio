import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from feature_engineering import create_time_features, aggregate_features, handle_missing_values

def train_fraud_model():
    print("Generating synthetic fraud data...")
    # Generate mock transaction data
    np.random.seed(42)
    n_samples = 10000
    data = {
        'transaction_id': range(n_samples),
        'user_id': np.random.randint(1, 500, n_samples),
        'amount': np.random.exponential(100, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='T'),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]) # 2% fraud
    }
    df = pd.DataFrame(data)
    
    # Preprocessing
    df = create_time_features(df, 'timestamp')
    df = aggregate_features(df, 'user_id', 'amount')
    df = handle_missing_values(df)
    
    # Encoding
    df = pd.get_dummies(df, columns=['device'])
    
    # Split
    X = df.drop(['is_fraud', 'timestamp', 'transaction_id'], axis=1)
    y = df['is_fraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    print(f"Training on {len(X_train)} samples with class balance: {y_train.mean():.2%}")
    
    # Model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\nModel Performance:")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    return model

if __name__ == "__main__":
    train_fraud_model()
