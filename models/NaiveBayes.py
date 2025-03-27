import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
import torch.multiprocessing as mp

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accuracy Function
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    # Set Random State for Reproducibility
    clf = GaussianNB()

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

            X_list.append(X.cpu().numpy())  # Move to CPU for sklearn
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