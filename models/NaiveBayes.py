# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from sklearn.naive_bayes import GaussianNB
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
#     clf = GaussianNB()

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

class NaiveBayesScratch:
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.variances = None

    def fit(self, X, y):
        """Train the Gaussian Naive Bayes model."""
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        # Initialize storage for priors, means, and variances
        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.variances = np.zeros((n_classes, n_features))

        # Compute class priors, means, and variances
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = X_c.shape[0] / n_samples
            self.means[idx] = np.mean(X_c, axis=0)
            self.variances[idx] = np.var(X_c, axis=0) + 1e-9  # Add small epsilon to avoid division by zero

    def gaussian_pdf(self, X, mean, var):
        """Compute Gaussian probability density function."""
        exponent = -((X - mean) ** 2) / (2 * var)
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(exponent)

    def predict(self, X):
        """Predict labels for test data."""
        predictions = []
        for x in tqdm(X, desc="Predicting"):
            # Compute log probabilities for each class
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.means[idx], self.variances[idx])))
                posterior = prior + likelihood
                posteriors.append(posterior)
            # Predict the class with the highest posterior
            pred = self.classes[np.argmax(posteriors)]
            predictions.append(pred)
        return np.array(predictions)

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accuracy Function
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    # Initialize Naive Bayes from scratch
    clf = NaiveBayesScratch()

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