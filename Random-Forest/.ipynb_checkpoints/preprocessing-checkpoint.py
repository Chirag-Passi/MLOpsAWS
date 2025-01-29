import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths where input and output data are located
input_path = "/opt/ml/processing/input"
train_output_path = "/opt/ml/processing/train"
test_output_path = "/opt/ml/processing/test"

# Ensure output directories exist
os.makedirs(train_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)

def main():
    # Load the dataset (assuming it's a CSV file in this case)
    input_file = os.path.join(input_path, "diabetes-dev-1.csv")
    print(f"Reading input data from {input_file}")
    
    df = pd.read_csv(input_file)
    
    # Perform any preprocessing steps (e.g., handle missing values, feature selection)
    print("Performing preprocessing steps...")
    df.fillna(0, inplace=True)  # Example of handling missing values
    
    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the train and test data
    train_file = os.path.join(train_output_path, "train_data.csv")
    test_file = os.path.join(test_output_path, "test_data.csv")
    
    print(f"Saving train data to {train_file}")
    train_df.to_csv(train_file, index=False)
    
    print(f"Saving test data to {test_file}")
    test_df.to_csv(test_file, index=False)

if __name__ == "__main__":
    main()