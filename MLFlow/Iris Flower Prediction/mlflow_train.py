import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def load_and_split_data(test_size=0.2, random_state=42):
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall

def log_to_mlflow(model, params, metrics, model_name="random_forest_model"):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)

def save_test_data(X_test, feature_names, output_path="data/test_data.csv"):
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(X_test, columns=feature_names).to_csv(output_path, index=False)

def run_mlflow_experiment():
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)

    params = {"n_estimators": 100, "random_state": 42}
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
    log_to_mlflow(model, params, metrics)
    save_test_data(X_test, load_iris().feature_names)

if __name__ == "__main__":
    run_mlflow_experiment()
