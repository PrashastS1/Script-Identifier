import os
import cv2
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define dataset paths
dataset_path = "data/recognition/"
train_csv = os.path.join(dataset_path, "train.csv")
test_csv = os.path.join(dataset_path, "test.csv")
log_file = "script_log1.txt"  # Log file nam
hindi_label = "hindi"  # Label for Hindi

# Configure logging
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
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

# ===== PLOTTING DECISION BOUNDARY =====

def plot_decision_boundary(X, y, model):
    """Plots decision boundary for logistic regression in 2D space."""
    # Reduce dimensions using PCA for visualization
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Create mesh grid
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Predict on grid
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Plot contour
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Logistic Regression Decision Boundary")
    plt.show()

# Plot decision boundary for training data
plot_decision_boundary(X_train, y_train, model)
