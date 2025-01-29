import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Set the input and output directories
model_dir = '/opt/ml/processing/model'  # Directory for the model artifact
evaluation_output_dir = '/opt/ml/processing/evaluation'  # Directory to save evaluation results
test_data_dir = '/opt/ml/processing/test'  # Directory for the test data

# Load the model
model_path = os.path.join(model_dir, 'model.joblib')  # Change this path if needed
model = joblib.load(model_path)

# Load the test data (features and labels)
test_data_path = os.path.join(test_data_dir, 'test.csv')
df = pd.read_csv(test_data_path)

# Assuming the target column is 'Outcome' and the rest are features
label_column = 'Diabetic'
X_test = df.drop(columns=[label_column])  # Features
y_test = df[label_column]  # True labels

# Make predictions with the loaded model
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Create an evaluation summary
evaluation_metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc
}

# Print the evaluation metrics (optional)
print("Evaluation Metrics:")
for metric, value in evaluation_metrics.items():
    print(f"{metric}: {value:.4f}")

# Save the evaluation metrics to a CSV file
evaluation_df = pd.DataFrame([evaluation_metrics])
os.makedirs(evaluation_output_dir, exist_ok=True)
evaluation_df.to_csv(os.path.join(evaluation_output_dir, 'evaluation_metrics.csv'), index=False)

print("Evaluation complete. Metrics saved to evaluation directory.")
