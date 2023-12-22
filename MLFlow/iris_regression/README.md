# MLOps Project: Iris Flower Prediction

Welcome to the MLOps project focusing on Iris Flower Prediction! This project demonstrates the use of MLflow for managing the machine learning life cycle, from training to deployment.

## Project Structure

The project is organized as follows:

- **`mlflow_project/`**: Directory containing MLflow-related scripts and notebooks.
  - `mlflow_train.py`: Python script for training a machine learning model using MLflow.
  - `mlflow_server.py`: Python script for serving the MLflow model.
  - **`notebooks/`**: Directory containing Jupyter notebooks related to Iris Flower Prediction.
    - `explore_data.ipynb`: Jupyter notebook for exploring the Iris dataset.
    - `use_mlflow_model.ipynb`: Jupyter notebook demonstrating how to use the trained MLflow model.
    - `train_mlflow_model.ipynb`: Jupyter notebook providing a step-by-step guide for training the MLflow model.

## Getting Started

Follow these steps to get started with the Iris Flower Prediction project:

1. **Install Prerequisites:**
   - Ensure you have Python installed on your machine.
   - Install the required dependencies using the following command:
     ```bash
     pip install -r requirements.txt
     ```

2. **Train MLflow Model:**
   - Navigate to the `mlflow_project/` directory:
     ```bash
     cd mlflow_project
     ```
   - Execute the MLflow training script to train a machine learning model and log the results:
     ```bash
     python mlflow_train.py
     ```

3. **Run MLflow Server:**
   - Execute the MLflow server script to serve the trained machine learning model:
     ```bash
     python mlflow_server.py
     ```

4. **Explore Data:**
   - Open the `notebooks/explore_data.ipynb` notebook to explore the Iris dataset used for training.

5. **Use MLflow Model:**
   - Open the `notebooks/use_mlflow_model.ipynb` notebook to see how to load and use the trained MLflow model for predicting flower types.

6. **Train MLflow Model Notebook:**
   - Open the `notebooks/train_mlflow_model.ipynb` notebook for a detailed guide on training the MLflow model.

## Notebooks

1. **Explore Data (`notebooks/explore_data.ipynb`):**
   - Jupyter notebook for exploring the Iris dataset and understanding its characteristics.

2. **Use MLflow Model (`notebooks/use_mlflow_model.ipynb`):**
   - Jupyter notebook demonstrating how to load the trained MLflow model and make predictions on new data.

3. **Train MLflow Model (`notebooks/train_mlflow_model.ipynb`):**
   - Jupyter notebook providing a step-by-step guide for training the MLflow model using the training script.

## Contributing

Feel free to contribute to the project by submitting pull requests. If you encounter any issues or have suggestions, please open an issue.

## License

This project is licensed under the [MIT License](LICENSE).
