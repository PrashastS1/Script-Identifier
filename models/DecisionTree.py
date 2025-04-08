import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
from sklearn.metrics import accuracy_score
from loguru import logger

def extract_features(dataset, batch_size=4096):
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

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
   

    # # Set Random State for Reproducibility
    # clf = DecisionTreeClassifier(max_depth=10, random_state=42)

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
    # clf.fit(X_train, y_train)

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
    # y_pred = clf.predict(X_test)

    # # Compute Accuracy
    # acc = accuracy(y_test, y_pred)
    # print(f"Test Accuracy: {acc:.4f}")

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15],
        'backbone': ['vgg'],
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
            torch.cuda.empty_cache()

            train_dataset = BHSceneDataset(root_dir="data/recognition", train_split=True,
                                        linear_transform=True, backbone=backbone, gap_dim=gap_dim) # 
            X_train, y_train = extract_features(train_dataset)

            test_dataset = BHSceneDataset(root_dir="data/recognition", train_split=False,
                                        linear_transform=True, backbone=backbone, gap_dim=gap_dim) # 
            X_test, y_test = extract_features(test_dataset)

            # Iterate over all parameter combinations
            for criterion, max_depth in product(param_grid['criterion'], param_grid['max_depth']):
                clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                logger.info(f"Backbone: {backbone}, GAP: {gap_dim}, Criterion: {criterion}, "
                            f"Max Depth: {max_depth}, Acc: {acc:.4f}")

                results.append({'backbone': backbone, 'gap_dim': gap_dim, 'criterion': criterion, 
                                'max_depth': max_depth, 'accuracy': acc})

                # Track the best model
                if acc > best_acc:
                    best_acc = acc
                    best_params = {'backbone': backbone, 'gap_dim': gap_dim, 
                                'criterion': criterion, 'max_depth': max_depth}

    logger.success(f"\nBest Accuracy: {best_acc:.4f}, Best Params: {best_params}")


    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='backbone', y='accuracy', hue='gap_dim', data=results_df)
    plt.title('DT Accuracy by Backbone and GAP Dimension')
    plt.savefig(os.path.join(plots_dir, "vgg_dt_accuracy_by_backbone_gapdim.png"))
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.catplot(x='max_depth', y='accuracy', col='backbone', hue='gap_dim', kind='bar', data=results_df, height=5,
                aspect=0.8)
    plt.suptitle('DT Accuracy by Max Depth, Backbone, and GAP Dimension', y=1.02)
    plt.savefig(os.path.join(plots_dir, "vgg_dt_accuracy_by_maxdepth_backbone_gapdim.png"))
    plt.show()


    heatmap_data = results_df.pivot_table(index='gap_dim', columns=['backbone', 'max_depth'], values='accuracy')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title('DT Accuracy Heatmap')
    plt.savefig(os.path.join(plots_dir, "vgg_dt_accuracy_heatmap.png"))
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='criterion', y='accuracy', data=results_df)
    plt.title('DT Accuracy by Impurity Criterion')
    plt.savefig(os.path.join(plots_dir, "vgg_dt_accuracy_by_criterion.png"))
    plt.show()

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='backbone', y='accuracy', hue='criterion', data=results_df)
    # plt.title('DT Accuracy by Backbone and Criterion')
    # plt.savefig(os.path.join(plots_dir, "vit_dt_accuracy_by_backbone_criterion.png"))
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # sns.catplot(x='max_depth', y='accuracy', col='backbone', hue='criterion', 
    #             kind='bar', data=results_df, height=5, aspect=0.8)
    # plt.suptitle('DT Accuracy by Max Depth, Backbone, and Criterion', y=1.02)
    # plt.savefig(os.path.join(plots_dir, "vit_dt_accuracy_by_maxdepth_backbone_criterion.png"))
    # plt.show()

    # heatmap_data = results_df.pivot_table(index='max_depth', 
    #                                     columns=['backbone', 'criterion'], 
    #                                     values='accuracy')
    # plt.figure(figsize=(12, 8))
    # sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
    # plt.title('DT Accuracy Heatmap (by max_depth, backbone, and criterion)')
    # plt.savefig(os.path.join(plots_dir, "vit_dt_accuracy_heatmap.png"))
    # plt.show()

    # plt.figure(figsize=(8, 6))
    # sns.boxplot(x='criterion', y='accuracy', data=results_df)
    # plt.title('DT Accuracy by Impurity Criterion')
    # plt.savefig(os.path.join(plots_dir, "vit_dt_accuracy_by_criterion.png"))
    # plt.show()
