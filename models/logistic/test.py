import os
import cv2
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define dataset paths
dataset_path = "data/recognition/"
train_csv = os.path.join(dataset_path, "train.csv")
test_csv = os.path.join(dataset_path, "test.csv")
log_file = "script_log.txt"  # Log file name
hindi_label = "hindi"  # Label for Hindi

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", filemode="w")
logging.info("Script started.")

# Function to load dataset correctly from CSV
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])  # Skip header
    X, y = [], []

    logging.info(f"Loading dataset: {csv_path}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_path = os.path.join(dataset_path, row["image_path"].replace("\\", "/"))  # Ensure uniform paths
        script = row["script"].strip().lower()

        if not os.path.exists(img_path):  # Check if file exists
            logging.warning(f"Image file not found: {img_path}")
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Unable to read image: {img_path}")
            continue  # Skip this image

        img = cv2.resize(img, (64, 64))
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

        X.append(hog_features)
        y.append(1 if script == hindi_label else 0)

    return np.array(X), np.array(y)


# Load train and test data with tqdm progress bar
X_train, y_train = load_dataset(train_csv)
X_test, y_test = load_dataset(test_csv)

logging.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train One-vs-All Logistic Regression
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

# Print results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

logging.info("Script finished.")
