# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import load_iris

# Step 1: Load the Dataset
# Using Iris dataset as an example
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
print("Dataset Preview:")
print(df.head())

# Step 2: Split Data into Features and Target
X = df.drop('Target', axis=1)
y = df['Target']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain-Test Split Completed.")

# Step 3: Train and Evaluate Models
# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(kernel='linear', random_state=42)
}

# Dictionary to store evaluation metrics
evaluation_metrics = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store metrics
    evaluation_metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }
    
    print(f"\nModel: {model_name}")
    print(classification_report(y_test, y_pred))

# Step 4: Compare Models
print("\nModel Evaluation Metrics:")
evaluation_df = pd.DataFrame(evaluation_metrics).T
print(evaluation_df)

# Save the evaluation metrics to a CSV file
evaluation_df.to_csv("model_evaluation_metrics.csv", index=True)
print("\nModel evaluation metrics saved to 'model_evaluation_metrics.csv'.")
# Specify the full path to save the CSV file
output_path = r"D:\.vscode\task\model_evaluation_metrics.csv"
evaluation_df.to_csv(output_path, index=True)
print(f"\nModel evaluation metrics saved to '{output_path}'.")
