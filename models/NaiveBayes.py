# # import numpy as np
# # import torch
# # from torch.utils.data import DataLoader
# # from sklearn.naive_bayes import GaussianNB
# # from tqdm import tqdm
# # from dataset.BH_scene_dataset import BHSceneDataset
# # import torch.multiprocessing as mp

# # if __name__ == '__main__':

# #     mp.set_start_method("spawn", force=True)

# #     device = "cuda" if torch.cuda.is_available() else "cpu"

# #     # Accuracy Function
# #     def accuracy(y_true, y_pred):
# #         return np.mean(y_true == y_pred)

# #     # Set Random State for Reproducibility
# #     clf = GaussianNB()

# #     # Load Dataset Efficiently
# #     def extract_features(dataset, batch_size=512):
# #         dataloader = DataLoader(
# #             dataset, 
# #             batch_size=batch_size, 
# #             shuffle=False, 
# #             num_workers=4, 
# #             pin_memory=False
# #         )

# #         X_list, y_list = [], []
# #         for batch in tqdm(dataloader, desc="Extracting Features"):
# #             X, y = batch  # Unpack tuple
# #             X = X.to(device)  # Move to GPU

# #             X_list.append(X.cpu().numpy())  # Move to CPU for sklearn
# #             y_list.append(y)

# #         return np.vstack(X_list), np.concatenate(y_list)

# #     # Load Training Data
# #     train_dataset = BHSceneDataset(
# #         root_dir="data/recognition",
# #         train_split=True,
# #         transform=None,
# #         linear_transform=True,
# #         backbone='resnet50',
# #         gap_dim=1
# #     )

# #     X_train, y_train = extract_features(train_dataset)
# #     clf.fit(X_train, y_train)

# #     # Load Test Data
# #     test_dataset = BHSceneDataset(
# #         root_dir="data/recognition",
# #         train_split=False,
# #         transform=None,
# #         linear_transform=True,
# #         backbone='resnet50',
# #         gap_dim=1
# #     )

# #     X_test, y_test = extract_features(test_dataset)
# #     y_pred = clf.predict(X_test)

# #     # Compute Accuracy
# #     acc = accuracy(y_test, y_pred)
# #     print(f"Test Accuracy: {acc:.4f}")


# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from dataset.BH_scene_dataset import BHSceneDataset
# import torch.multiprocessing as mp

# class NaiveBayesScratch:
#     def __init__(self):
#         self.classes = None
#         self.priors = None
#         self.means = None
#         self.variances = None

#     def fit(self, X, y):
#         """Train the Gaussian Naive Bayes model."""
#         self.classes = np.unique(y)
#         n_samples, n_features = X.shape
#         n_classes = len(self.classes)

#         # Initialize storage for priors, means, and variances
#         self.priors = np.zeros(n_classes)
#         self.means = np.zeros((n_classes, n_features))
#         self.variances = np.zeros((n_classes, n_features))

#         # Compute class priors, means, and variances
#         for idx, c in enumerate(self.classes):
#             X_c = X[y == c]
#             self.priors[idx] = X_c.shape[0] / n_samples
#             self.means[idx] = np.mean(X_c, axis=0)
#             self.variances[idx] = np.var(X_c, axis=0) + 1e-9  # Add small epsilon to avoid division by zero

#     def gaussian_pdf(self, X, mean, var):
#         """Compute Gaussian probability density function."""
#         exponent = -((X - mean) ** 2) / (2 * var)
#         return (1 / np.sqrt(2 * np.pi * var)) * np.exp(exponent)

#     def predict(self, X):
#         """Predict labels for test data."""
#         predictions = []
#         for x in tqdm(X, desc="Predicting"):
#             # Compute log probabilities for each class
#             posteriors = []
#             for idx, c in enumerate(self.classes):
#                 prior = np.log(self.priors[idx])
#                 likelihood = np.sum(np.log(self.gaussian_pdf(x, self.means[idx], self.variances[idx])))
#                 posterior = prior + likelihood
#                 posteriors.append(posterior)
#             # Predict the class with the highest posterior
#             pred = self.classes[np.argmax(posteriors)]
#             predictions.append(pred)
#         return np.array(predictions)

# if __name__ == '__main__':

#     mp.set_start_method("spawn", force=True)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Accuracy Function
#     def accuracy(y_true, y_pred):
#         return np.mean(y_true == y_pred)

#     # Initialize Naive Bayes from scratch
#     clf = NaiveBayesScratch()

#     # Load Dataset Efficiently
#     def extract_features(dataset, batch_size=512):
#         dataloader = DataLoader(
#             dataset, 
#             batch_size=batch_size, 
#             shuffle=False, 
#             num_workers=4, 
#             pin_memory=False
#         )

#         X_list, y_list = [], []
#         for batch in tqdm(dataloader, desc="Extracting Features"):
#             X, y = batch  # Unpack tuple
#             X = X.to(device)  # Move to GPU

#             X_list.append(X.cpu().numpy())  # Move to CPU for numpy operations
#             y_list.append(y)

#         return np.vstack(X_list), np.concatenate(y_list)

#     # Load Training Data
#     train_dataset = BHSceneDataset(
#         root_dir="data/recognition",
#         train_split=True,
#         transform=None,
#         linear_transform=True,
#         backbone='resnet50',
#         gap_dim=1
#     )

#     X_train, y_train = extract_features(train_dataset)
#     clf.fit(X_train, y_train)

#     # Load Test Data
#     test_dataset = BHSceneDataset(
#         root_dir="data/recognition",
#         train_split=False,
#         transform=None,
#         linear_transform=True,
#         backbone='resnet50',
#         gap_dim=1
#     )

#     X_test, y_test = extract_features(test_dataset)
#     y_pred = clf.predict(X_test)

#     # Compute Accuracy
#     acc = accuracy(y_test, y_pred)
#     print(f"Test Accuracy: {acc:.4f}")

import os
import cv2
import datetime
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Paths
train_csv = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/train.csv"
test_csv = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/test.csv"
data_path = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/"
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = os.path.join(script_dir, "script_log_naive_bayes_hog.txt")

# Language mapping
Lmap = {
    1: "assamese",
    2: "bengali",
    3: "english",
    4: "gujarati",
    5: "hindi",
    6: "kannada",
    7: "malayalam",
    8: "marathi",
    9: "punjabi",
    10: "tamil",
    11: "telugu",
    12: "urdu"
}

def select_lang():
    while True:
        print("\nSelect the language to train and test on:")
        for num, lang in Lmap.items():
            print(f"Enter {num} to train and test on {lang}")
        try:
            choice = input("\nEnter your choice: ").strip()
            if not choice:
                print("Error: Input cannot be empty. Please enter a number.")
                continue
            choice = int(choice)
            if choice not in Lmap:
                print(f"Error: Invalid choice '{choice}'. Please select a number between 1 and {len(Lmap)}.")
                continue
            return Lmap[choice]
        except ValueError:
            print("Error: Please enter a valid number.")

# Logging setup
logging.basicConfig(filename=logger, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
logging.info("Script started.")

def load_dataset(csv_path, selected_lang):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])
    print(f"Number of rows in CSV: {len(df)}", flush=True)
    
    x, y = [], []
    logging.info(f"Loading dataset from {csv_path} for language: {selected_lang}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images", dynamic_ncols=True, mininterval=0.1):
        img_path = os.path.join(data_path, row["image_path"].replace("\\", "/"))
        if not os.path.exists(img_path):
            logging.warning(f"File not found: {img_path}")
            continue

        label = 1 if row["script"].lower() == selected_lang else 0

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Error reading image: {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        x.append(hog_features)
        y.append(label)

    print(f"Total images processed: {len(x)}", flush=True)
    if len(x) == 0:
        print("Error: No images were successfully loaded. Check image paths and file accessibility.", flush=True)
        exit()

    x, y = np.array(x), np.array(y)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logging.error(f"Only one class found: {unique_classes}. Check dataset.")
        print(f"Error: Only one class found ({unique_classes}). Adjust dataset!", flush=True)
        exit()

    return x, y

# Select language
selected_lang = select_lang()
if not selected_lang:
    print("Invalid choice. Exiting program.", flush=True)
    exit()

# Load data
x_train, y_train = load_dataset(train_csv, selected_lang)
x_test, y_test = load_dataset(test_csv, selected_lang)

# Reduce dataset size for faster testing
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:5000], y_test[:5000]

logging.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")
print(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}", flush=True)

# Feature scaling
print("Scaling features...", flush=True)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Feature scaling completed.", flush=True)

# Reduce dimensionality with PCA
print("Applying PCA to reduce dimensionality...", flush=True)
pca = PCA(n_components=100)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print("PCA completed.", flush=True)

# Handle unbalanced dataset with SMOTE
logging.info("Applying SMOTE to handle unbalanced dataset...")
print("Applying SMOTE...", flush=True)
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
logging.info(f"After SMOTE, train set size: {len(x_train)}")
print(f"After SMOTE, train set size: {len(x_train)}", flush=True)

# Train Naive Bayes model
print("Training Naive Bayes model...", flush=True)
model = GaussianNB()
model.fit(x_train, y_train)
print("Naive Bayes model training completed.", flush=True)

# Predict with progress bar
print("Predicting on test set...", flush=True)
y_pred = []
for i in tqdm(range(len(x_test)), desc="Predicting", dynamic_ncols=True, mininterval=0.1):
    pred = model.predict(x_test[i:i+1])
    y_pred.append(pred[0])
y_pred = np.array(y_pred)
print("Prediction completed.", flush=True)

# Compute accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.2f}%", flush=True)
print("Classification Report:\n", classification_report(y_test, y_pred), flush=True)

logging.info("Script finished.")
print("Script finished.", flush=True)

# Visualization
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def save_plot(x, y, model, language):
    print("Generating visualization...", flush=True)
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)

    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = []
    for i in tqdm(range(len(xx.ravel())), desc="Predicting for plot", dynamic_ncols=True, mininterval=0.1):
        point = pca.inverse_transform([xx.ravel()[i], yy.ravel()[i]])
        pred = model.predict([point])[0]
        Z.append(pred)
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label="Data Points")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + language, language])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundary for {language} (Naive Bayes + HOG)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_naive_bayes_hog_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    
    logging.info(f"Saved decision boundary plot for {language} at {plot_path}")
    logging.info("=========================================")
    print(f"Plot saved at: {plot_path}", flush=True)
    plt.show()
    print("Visualization completed.", flush=True)

save_plot(x_train, y_train, model, selected_lang)