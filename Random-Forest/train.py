import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import boto3
import os
from io import StringIO
#Load data
s3_client = boto3.client('s3')
bucket_name = 'ml-ops-zenon'
file_key = 'Input/diabetes-dev-1.csv'

# Fetch the file from S3
response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

# Read the file content into a Pandas DataFrame
csv_data = response['Body'].read().decode('utf-8')  # Decode the file content
data = pd.read_csv(StringIO(csv_data))  # Read the CSV data from the string

# Check the first few rows of the dataset
print(data.head())
# print(sklearn.__version__)
#Split Model
X = data.drop(columns=["Diabetic", "PatientID"])
y = data["Diabetic"]
print(X)

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(solver='saga', max_iter=500, C=0.5)

model.fit(X_train, y_train)
model_output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
joblib.dump(model, os.path.join(model_output_dir, "model.pkl"))
# joblib.dump(model, 'random_forest_model.pkl')