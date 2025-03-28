# # import numpy as np
# # import torch
# # from torch.utils.data import DataLoader
# # from sklearn.neighbors import KNeighborsClassifier
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
# #     clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

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

# class KNNScratch:
#     def __init__(self, k=5):
#         self.k = k
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         """Store the training data."""
#         self.X_train = X
#         self.y_train = y

#     def predict(self, X):
#         """Predict labels for test data."""
#         predictions = []
#         for x in tqdm(X, desc="Predicting"):
#             # Compute Euclidean distances to all training points
#             distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
#             # Get indices of k nearest neighbors
#             k_indices = np.argsort(distances)[:self.k]
#             # Get labels of k nearest neighbors
#             k_nearest_labels = self.y_train[k_indices]
#             # Majority vote
#             unique, counts = np.unique(k_nearest_labels, return_counts=True)
#             pred = unique[np.argmax(counts)]
#             predictions.append(pred)
#         return np.array(predictions)

# if __name__ == '__main__':

#     mp.set_start_method("spawn", force=True)

#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "mps" if torch.backends.mps.is_available() else "cpu"

#     # Accuracy Function
#     def accuracy(y_true, y_pred):
#         return np.mean(y_true == y_pred)

#     # Initialize KNN from scratch
#     clf = KNNScratch(k=5)

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # For handling unbalanced datasets

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../data/recognition/")
train_csv = os.path.join(data_path, "train.csv")
test_csv = os.path.join(data_path, "test.csv")
logger = os.path.join(script_dir, "script_log_knn_hog.txt")

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
    print("\nSelect the language to train and test on:")
    for num, lang in Lmap.items():
        print(f"Enter {num} to train and test on {lang}")
    choice = int(input("\nEnter your choice: "))
    return Lmap.get(choice, None)

# Logging setup
logging.basicConfig(filename=logger, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
logging.info("Script started.")

def load_dataset(csv_path, selected_lang):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])
    x, y = [], []
    logging.info(f"Loading dataset from {csv_path} for language: {selected_lang}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_path = os.path.join(data_path, row["image_path"].replace("\\", "/"))  
        if not os.path.exists(img_path):
            logging.warning(f"File not found: {img_path}")
            continue

        folder_name = os.path.basename(os.path.dirname(img_path)).lower()
        label = 1 if folder_name == selected_lang else 0

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Error reading image: {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        x.append(hog_features)
        y.append(label)

    x, y = np.array(x), np.array(y)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logging.error(f"Only one class found: {unique_classes}. Check dataset.")
        print(f"Error: Only one class found ({unique_classes}). Adjust dataset!")
        exit()

    return x, y

# Select language
selected_lang = select_lang()
if not selected_lang:
    print("Invalid choice. Exiting program.")
    exit()

# Load data
x_train, y_train = load_dataset(train_csv, selected_lang)
x_test, y_test = load_dataset(test_csv, selected_lang)

logging.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Handle unbalanced dataset with SMOTE
logging.info("Applying SMOTE to handle unbalanced dataset...")
smote = SMOTE(random_state=42)
x_train, y_train = smote.fit_resample(x_train, y_train)
logging.info(f"After SMOTE, train set size: {len(x_train)}")

# Train KNN model
model = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
model.fit(x_train, y_train)

# Evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

logging.info("Script finished.")

# Visualization
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def save_plot(x, y, model, language):
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)

    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label="Data Points")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + language, language])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundary for {language} (KNN + HOG)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_knn_hog_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    
    logging.info(f"Saved decision boundary plot for {language} at {plot_path}")
    logging.info("=========================================")   
    print(f"Plot saved at: {plot_path}")
    plt.show()

save_plot(x_train, y_train, model, selected_lang)