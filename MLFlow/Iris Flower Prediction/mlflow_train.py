import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def load_and_split_data(test_size=0.2, random_state=42):
    # Load Iris dataset and split into training and testing sets
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    # Train RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall

def log_to_mlflow(model, params, metrics, model_name="random_forest_model"):
    # Log parameters and metrics to MLflow
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

def save_test_data(X_test, feature_names, output_path="data/test_data.csv"):
    # Save the test data for later use
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(X_test, columns=feature_names).to_csv(output_path, index=False)

def run_mlflow_experiment():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()

    # Train RandomForestClassifier
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)

    # Evaluate the model
    accuracy, precision, recall = evaluate_model(rf_model, X_test, y_test)

    # Log to MLflow
    params = {"n_estimators": 100, "random_state": 42}
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
    log_to_mlflow(rf_model, params, metrics)

    # Save the test data
    save_test_data(X_test, load_iris().feature_names)

if __name__ == "__main__":
    run_mlflow_experiment()
