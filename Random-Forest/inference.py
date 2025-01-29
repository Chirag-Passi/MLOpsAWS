import joblib
import os
import pandas as pd
import json
import sklearn


# Define the expected feature columns (same as during training)
feature_columns = [
    "Pregnancies",
    "PlasmaGlucose",
    "DiastolicBloodPressure",
    "TricepsThickness",
    "SerumInsulin",
    "BMI",
    "DiabetesPedigree",
    "Age",
]

"""
Deserialize fitted model
"""


def model_fn(model_dir):
    print("Log : Model Loaded in the container.")
    model = joblib.load(os.path.join(model_dir, "random_forest_model.pkl"))
    return model


"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""


def input_fn(request_body, request_content_type):
    print("Input Request Captured for the model.")
    if request_content_type == "application/json":
        # Load the input JSON
        request_body = json.loads(request_body)

        # Ensure the input contains the key 'Input'
        if "Input" not in request_body:
            raise ValueError("Missing 'Input' key in input data")

        # Extract the input data
        input_data = request_body["Input"]

        # Check if input data contains all required feature columns
        if not isinstance(input_data, list) or len(input_data) != len(feature_columns):
            raise ValueError(
                f"Input data must be a list of {len(feature_columns)} values"
            )

        return [input_data]  # Wrap in a list to represent a single sample
    else:
        raise ValueError("This model only supports application/json input")


"""
predict_fn
    input_data: returned array from input_fn above
    model (sklearn model) returned model loaded from model_fn above
"""


def predict_fn(input_data, model):
    print("Inside Predict")
    input_df = pd.DataFrame(input_data, columns=feature_columns)
    prediction = model.predict(input_df)

    return prediction


"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: the content type the endpoint expects to be returned. Ex: JSON, string
"""


def output_fn(prediction, content_type):
    res = int(prediction[0])
    if res == 0:
        respJSON = {"Diabetic": False}
        return respJSON
    elif res == 1:
        respJSON = {"Diabetic": True}
        return respJSON
    else:
        raise ValueError("This model only supports application/json output")
