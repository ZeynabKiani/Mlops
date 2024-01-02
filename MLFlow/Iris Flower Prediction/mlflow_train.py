import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

class MLFlowExperiment:
    def __init__(self, model_name="random_forest_model"):
        self.model_name = model_name

    def load_and_split_data(self, test_size=0.2, random_state=42):
        data = load_iris()
        return train_test_split(data.data, data.target, test_size=test_size, random_state=random_state)

    def train_random_forest(self, X_train, y_train, n_estimators=100, random_state=42):
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        return accuracy, precision, recall

    def log_to_mlflow(self, model, params, metrics):
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, self.model_name)

    def save_test_data(self, X_test, feature_names, output_path="data/test_data.csv"):
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(X_test, columns=feature_names).to_csv(output_path, index=False)

    def run_experiment(self, n_estimators=100, random_state=42):
        X_train, X_test, y_train, y_test = self.load_and_split_data()
        model = self.train_random_forest(X_train, y_train, n_estimators=n_estimators, random_state=random_state)
        accuracy, precision, recall = self.evaluate_model(model, X_test, y_test)

        params = {"n_estimators": n_estimators, "random_state": random_state}
        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
        self.log_to_mlflow(model, params, metrics)
        self.save_test_data(X_test, load_iris().feature_names)

if __name__ == "__main__":
    experiment = MLFlowExperiment()
    experiment.run_experiment()
