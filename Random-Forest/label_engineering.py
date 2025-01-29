import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set the input and output directories
input_dir = '/opt/ml/processing/input'
train_labels_output_dir = '/opt/ml/processing/train_labels'
test_labels_output_dir = '/opt/ml/processing/test_labels'

# Load the input data (CSV file)
input_data_path = os.path.join(input_dir, 'diabetes-dev-1.csv')
df = pd.read_csv(input_data_path)

# Basic label engineering
# Assuming that 'Outcome' is the target variable to be encoded
# You can also perform other encoding techniques if required (e.g., one-hot encoding)
label_column = 'Diabetic'

# Initialize a LabelEncoder
label_encoder = LabelEncoder()

# Encode the labels (binary classification: 0 or 1)
df[label_column] = label_encoder.fit_transform(df[label_column])

# Split the data into features (X) and the target label (y)
X = df.drop(columns=[label_column])  # Features (all columns except the label column)
y = df[label_column]  # Target variable (encoded labels)

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(df))  # 80% for training
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Create directories if they don't exist
os.makedirs(train_labels_output_dir, exist_ok=True)
os.makedirs(test_labels_output_dir, exist_ok=True)

# Save the transformed labels to CSV files
y_train.to_csv(os.path.join(train_labels_output_dir, 'train_labels.csv'), index=False)
y_test.to_csv(os.path.join(test_labels_output_dir, 'test_labels.csv'), index=False)

print("Label engineering complete. Transformed labels saved to train_labels and test_labels directories.")
