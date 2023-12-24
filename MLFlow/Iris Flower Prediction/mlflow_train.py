import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    # Train RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall

def log_to_mlflow(model, accuracy, precision, recall, params, model_name="random_forest_model"):
    # Log parameters and metrics to MLflow
    with mlflow.start_run():
        # Log model parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log the model
        mlflow.sklearn.log_model(model, model_name)

def save_test_data(X_test, feature_names, output_path="data/test_data.csv"):
    # Save the test data for later use
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(X_test, columns=feature_names).to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load Iris dataset
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    # Train RandomForestClassifier
    rf_model = train_random_forest(X_train, y_train, n_estimators=100, random_state=42)

    # Evaluate the model
    accuracy, precision, recall = evaluate_model(rf_model, X_test, y_test)

    # Log to MLflow
    params = {"n_estimators": 100, "random_state": 42}
    log_to_mlflow(rf_model, accuracy, precision, recall, params)

    # Save the test data
    save_test_data(X_test, data.feature_names)
