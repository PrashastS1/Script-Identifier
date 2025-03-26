
import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define dataset path
dataset_path = "dataset/"  # Update with actual dataset path
hindi_label = "Hindi"  # Adjust based on actual folder names in the dataset

# Initialize lists for data and labels
X = []
y = []

# Process each script category
for script_name in os.listdir(dataset_path):
    script_path = os.path.join(dataset_path, script_name)
    
    # Check if it's a directory
    if not os.path.isdir(script_path):
        continue
    
    # Assign binary labels (1 for Hindi, 0 for others)
    label = 1 if script_name == hindi_label else 0

    # Load images
    for img_name in os.listdir(script_path):
        img_path = os.path.join(script_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (64, 64))  # Resize
        
        # Extract HOG features
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        
        X.append(hog_features)
        y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train One-vs-All Logistic Regression
model = LogisticRegression(solver='liblinear')  # Binary classification (Hindi vs Others)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
