# iris_regression.py
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def log_params(params):
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def main():
    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run():
        model = train_model(X_train, y_train)
        mse = evaluate_model(model, X_test, y_test)

        # Log parameters and metrics
        params = {"model": "Linear Regression", "random_state": 42}
        metrics = {"mse": mse}

        log_params(params)
        log_metrics(metrics)

        # Save the model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()
