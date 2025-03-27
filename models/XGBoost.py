import numpy as np
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
import torch.multiprocessing as mp

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accuracy Function
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)

    # Function to Extract Features Using GPU
    def extract_features(dataset, batch_size=4096):
        """Extract features from dataset using GPU acceleration."""
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=False
        )

        X_list, y_list = [], []
        for batch in tqdm(dataloader, desc="Extracting Features"):
            X, y = batch  # Unpack tuple
            X = X.to(device)  # Move to GPU

            X_list.append(X.cpu().numpy())
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

    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Configure XGBoost Parameters
    params = {
        "objective": "multi:softmax",  # Multi-class classification
        "num_class": 14,  # Number of classes
        "max_depth": 10,  # Limit tree depth to prevent overfitting
        "learning_rate": 0.1,  # Default learning rate
        "n_estimators": 100,  # Number of trees
        "tree_method": "hist",  # Use GPU acceleration
        "random_state": 42,
        "device" : device
    }

    # Train XGBoost on GPU
    clf = xgb.train(params, dtrain, num_boost_round=100)

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

    # Convert Test Data to DMatrix
    dtest = xgb.DMatrix(X_test)

    # Predict
    y_pred = clf.predict(dtest)

    # Compute Accuracy
    acc = accuracy(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
