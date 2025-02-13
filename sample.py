from flask import Flask, request
import pandas as pd
import numpy as np
from datetime import datetime
import pickle, os, json

# Initialize Flask app
app = Flask(__name__)


# Model class for Random Forest
import joblib


class RandomForestModel:
    def __init__(self):
        self.model = None
        self.model_path = "./random_forest_model.pkl"
        self.log_file_path = "./logs_monitoring/log.json"  # Path for logging requests

    def load_model(self):
        """Load the model from the pickle file."""
        try:
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(
                f"Model file not found at {self.model_path}. Make sure the file exists."
            )
        except Exception as e:
            print(f"An error occurred while loading the model: {str(e)}")

    def log_data(self, data):
        """Log request and response data to a JSON file."""
        if os.path.exists(self.log_file_path):
            # If log file exists, append to it
            with open(self.log_file_path, "r+") as f:
                logs = json.load(f)
                logs.append(data)
                f.seek(0)
                json.dump(logs, f, indent=4)
        else:
            # Create a new log file if it doesn't exist
            with open(self.log_file_path, "w") as f:
                json.dump([data], f, indent=4)


# Initialize the RandomForestModel class
random_forest_model = RandomForestModel()


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint to get predictions."""
    try:
        payload = request.get_json()
        data = payload["data"]
        features = payload["features"]

        # Convert input data to a Pandas DataFrame
        data = np.array([data])
        data_df = pd.DataFrame(data, columns=features)

        # Make prediction using the loaded model
        prediction = int(random_forest_model.model.predict(data_df)[0])
        print(prediction)
        result = "Diabetic" if prediction else "Not Diabetic"

        # Log request and response
        random_forest_model.log_data(
            {
                "request": payload,
                "response": result,
                "time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )

        return {"prediction": result, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "failed"}


if __name__ == "__main__":
    # Load the model before starting the server
    random_forest_model.load_model()
    app.run(host="0.0.0.0", port=5000)
