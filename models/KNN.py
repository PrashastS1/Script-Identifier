# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from sklearn.neighbors import KNeighborsClassifier
# from tqdm import tqdm
# from dataset.BH_scene_dataset import BHSceneDataset
# import torch.multiprocessing as mp

# if __name__ == '__main__':

#     mp.set_start_method("spawn", force=True)

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Accuracy Function
#     def accuracy(y_true, y_pred):
#         return np.mean(y_true == y_pred)

#     # Set Random State for Reproducibility
#     clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

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

#             X_list.append(X.cpu().numpy())  # Move to CPU for sklearn
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
import torch.multiprocessing as mp

class KNNScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict labels for test data."""
        predictions = []
        for x in tqdm(X, desc="Predicting"):
            # Compute Euclidean distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            # Majority vote
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            predictions.append(pred)
        return np.array(predictions)

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accuracy Function
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    # Initialize KNN from scratch
    clf = KNNScratch(k=5)

    # Load Dataset Efficiently
    def extract_features(dataset, batch_size=512):
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=False
        )

        X_list, y_list = [], []
        for batch in tqdm(dataloader, desc="Extracting Features"):
            X, y = batch  # Unpack tuple
            X = X.to(device)  # Move to GPU

            X_list.append(X.cpu().numpy())  # Move to CPU for numpy operations
            y_list.append(y)

        return np.vstack(X_list), np.concatenate(y_list)

    # Load Training Data
    train_dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=True,
        transform=None,
        linear_transform=True,
        backbone='resnet50',
        gap_dim=1
    )

    X_train, y_train = extract_features(train_dataset)
    clf.fit(X_train, y_train)

    # Load Test Data
    test_dataset = BHSceneDataset(
        root_dir="data/recognition",
        train_split=False,
        transform=None,
        linear_transform=True,
        backbone='resnet50',
        gap_dim=1
    )

    X_test, y_test = extract_features(test_dataset)
    y_pred = clf.predict(X_test)

    # Compute Accuracy
    acc = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")