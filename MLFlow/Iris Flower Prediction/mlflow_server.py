import mlflow.pyfunc
from flask import Flask, jsonify, request
import pandas as pd

class ModelServer:
    def __init__(self, model_path):
        self.loaded_model = mlflow.pyfunc.load_model(model_path)

    def predict(self, input_data):
        try:
            # Make predictions using the loaded model
            predictions = self.loaded_model.predict(input_data)
            return predictions.tolist()

        except Exception as e:
            raise ValueError(str(e))

app = Flask(__name__)

# Load the saved model
model_server = ModelServer(model_path="random_forest_model")

# Define a predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        input_data = pd.DataFrame(data['data'], columns=data['columns'])

        # Make predictions using the model server
        predictions = model_server.predict(input_data)

        # Convert predictions to a JSON format
        result = {'predictions': predictions}
        return jsonify(result)

    except ValueError as ve:
        return jsonify({'error': str(ve)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
