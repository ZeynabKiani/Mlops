import mlflow.pyfunc
from flask import Flask, jsonify, request
import pandas as pd

class ModelServer:
    def __init__(self, model_path):
        self.loaded_model = mlflow.pyfunc.load_model(model_path)

    def predict(self, input_data):
        try:
            predictions = self.loaded_model.predict(input_data)
            return predictions.tolist()

        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

def create_and_run_app(model_path="random_forest_model"):
    app = Flask(__name__)
    model_server = ModelServer(model_path)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json(force=True)
            input_data = pd.DataFrame(data.get('data', []), columns=data.get('columns', []))
            predictions = model_server.predict(input_data)
            result = {'predictions': predictions}
            return jsonify(result)

        except ValueError as ve:
            return jsonify({'error': str(ve)})

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)

    return app

# Load the saved model server and run the Flask app
if __name__ == '__main__':
    create_and_run_app()
