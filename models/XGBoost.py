import numpy as np
import torch
import os
import xgboost as xgb
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
from sklearn.metrics import accuracy_score
from loguru import logger

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


# Accuracy Function
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == '__main__':

    mp.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Load Training Data
    # train_dataset = BHSceneDataset(
    #     root_dir="data/recognition",
    #     train_split=True,
    #     transform=None,
    #     linear_transform=True,
    #     backbone='resnet50',
    #     gap_dim=1
    # )

    # X_train, y_train = extract_features(train_dataset)

    # # Convert to DMatrix for XGBoost
    # dtrain = xgb.DMatrix(X_train, label=y_train)

    # # Configure XGBoost Parameters
    # params = {
    #     "objective": "multi:softmax",  # Multi-class classification
    #     "num_class": 14,  # Number of classes
    #     "max_depth": 10,  # Limit tree depth to prevent overfitting
    #     "learning_rate": 0.1,  # Default learning rate
    #     "n_estimators": 100,  # Number of trees
    #     "tree_method": "hist",  # Use GPU acceleration
    #     "random_state": 42,
    #     "device" : device
    # }

    # # Train XGBoost on GPU
    # clf = xgb.train(params, dtrain, num_boost_round=100)

    # # Load Test Data
    # test_dataset = BHSceneDataset(
    #     root_dir="data/recognition",
    #     train_split=False,
    #     transform=None,
    #     linear_transform=True,
    #     backbone='resnet50',
    #     gap_dim=1
    # )

    # X_test, y_test = extract_features(test_dataset)

    # # Convert Test Data to DMatrix
    # dtest = xgb.DMatrix(X_test)

    # # Predict
    # y_pred = clf.predict(dtest)

    # # Compute Accuracy
    # acc = accuracy(y_test, y_pred)
    # print(f"Test Accuracy: {acc:.4f}")

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.1, 0.3],  # Add learning rate
        'backbone': ['resnet50', 'vgg16'],
        'gap_dim_resnet': [1, 3, 5, 7],
        'gap_dim_vgg': [1, 4, 7, 10, 13]
    }

    best_acc = 0
    best_params = {}
    results = []

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Manual Grid Search
    for backbone in param_grid['backbone']:
        gap_dims = param_grid['gap_dim_resnet'] if backbone == 'resnet50' else param_grid['gap_dim_vgg']
        
        for gap_dim in gap_dims:
            # Load data
            torch.cuda.empty_cache()
            
            train_dataset = BHSceneDataset(root_dir="data/recognition", train_split=True, transform=None,
                                        linear_transform=True, backbone=backbone, gap_dim=gap_dim)
            X_train, y_train = extract_features(train_dataset)
            dtrain = xgb.DMatrix(X_train, label=y_train)

            test_dataset = BHSceneDataset(root_dir="data/recognition", train_split=False, transform=None,
                                        linear_transform=True, backbone=backbone, gap_dim=gap_dim)
            X_test, y_test = extract_features(test_dataset)
            dtest = xgb.DMatrix(X_test)

            # Iterate over all parameter combinations
            for n_estimators, max_depth, learning_rate in product(param_grid['n_estimators'], 
                                                                param_grid['max_depth'], 
                                                                param_grid['learning_rate']):
                clf = xgb.XGBClassifier(
                    objective="multi:softmax",  # or "multi:softprob" if needed
                    num_class=14,
                    tree_method="hist",  # Use GPU Hist tree method
                    random_state=42,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate
                )
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                logger.info(f"Backbone: {backbone}, GAP: {gap_dim}, n_estimators: {n_estimators}, "
                    f"Max Depth: {max_depth}, Learning Rate: {learning_rate}, Acc: {acc:.4f}")

                results.append({'backbone': backbone, 'gap_dim': gap_dim, 'n_estimators': n_estimators,
                                'max_depth': max_depth, 'learning_rate': learning_rate, 'accuracy': acc})

                # Track the best model
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'backbone': backbone, 'gap_dim': gap_dim, 
                                'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate}

    logger.success(f"\nBest Accuracy: {best_acc:.4f}, Best Params: {best_params}")

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='n_estimators', y='accuracy', hue='backbone', data=results_df)  # Use lineplot for trends
    plt.title('XG Accuracy vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(plots_dir, "xg_accuracy_vs_n_estimators.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='backbone', y='accuracy', hue='gap_dim', data=results_df)
    plt.title('XG Accuracy by Backbone and GAP Dimension')
    plt.savefig(os.path.join(plots_dir, "xg_accuracy_by_backbone_gapdim.png"))
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.catplot(x='max_depth', y='accuracy', col='backbone', hue='gap_dim', kind='bar', data=results_df, height=5,
                aspect=0.8)
    plt.suptitle('XG Accuracy by Max Depth, Backbone, and GAP Dimension', y=1.02)
    plt.savefig(os.path.join(plots_dir, "xg_accuracy_by_maxdepth_backbone_gapdim.png"))
    plt.show()


    heatmap_data = results_df.pivot_table(index='gap_dim', columns=['backbone', 'max_depth'], values='accuracy')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title('XG Accuracy Heatmap')
    plt.savefig(os.path.join(plots_dir, "xg_accuracy_heatmap.png"))
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='criterion', y='accuracy', data=results_df)
    plt.title('XG Accuracy by Impurity Criterion')
    plt.savefig(os.path.join(plots_dir, "xg_accuracy_by_criterion.png"))
    plt.show()
