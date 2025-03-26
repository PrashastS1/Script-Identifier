import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define dataset paths
dataset_path = "data/recognition/"
train_csv = os.path.join(dataset_path, "train.csv")
test_csv = os.path.join(dataset_path, "test.csv")
hindi_label = "hindi"  # Label for Hindi

# Function to load dataset correctly from CSV
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["image_path", "annotation", "script"])
    X, y = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(dataset_path, row["image_path"])  # Full image path
        script = row["script"].strip().lower()  # Language from CSV

        # Assign labels: Hindi = 1, Others = 0
        label = 1 if script == hindi_label else 0

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Missing image {img_path}")  # Log missing images
            continue

        img = cv2.resize(img, (64, 64))  # Resize for uniformity

        # Extract HOG features
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

        X.append(hog_features)
        y.append(label)

    return np.array(X), np.array(y)

# Load train and test data
X_train, y_train = load_dataset(train_csv)
X_test, y_test = load_dataset(test_csv)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train One-vs-All Logistic Regression
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
