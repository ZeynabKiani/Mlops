import mlflow.pyfunc
from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load the saved model
model_path = "random_forest_model"
loaded_model = mlflow.pyfunc.load_model(model_path)

# Define a predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        input_data = pd.DataFrame(data['data'], columns=data['columns'])

        # Make predictions using the loaded model
        predictions = loaded_model.predict(input_data)

        # Convert predictions to a JSON format
        result = {'predictions': predictions.tolist()}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)