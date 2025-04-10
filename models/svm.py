import numpy as np
from collections import Counter
import torch
import cupy as cp  # For GPU array operations
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from loguru import logger
from dataset.BH_scene_dataset import BHSceneDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.multiprocessing as mp
from tqdm import tqdm
# Import cuML SVC instead of sklearn's SVM
from cuml.svm import SVC
from cuml.preprocessing import StandardScaler as cuStandardScaler
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from google.colab import files
import pickle

def train(X_train, y_train, config):
    """Run the experiment using the given configuration with cuML SVC."""
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train
    )
    
    # Convert to float32 for GPU efficiency
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    
    # Use cuML's StandardScaler for GPU acceleration
    scaler = cuStandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    logger.info(f"C: {config['C']}")
    logger.info(f"Kernel: {config['kernel']}")
    logger.info(f"Gamma: {config['gamma']}")

    # Create the SVM model with cuML's SVC
    model = SVC(
        C=config["C"], 
        kernel=config["kernel"],
        gamma=config["gamma"],
        probability=False,  # Disable for faster training
        cache_size=2048,    # Larger cache for GPU
        max_iter=10000,     # Prevent infinite training
        tol=0.01,           # Relaxed tolerance
        class_weight='balanced',
        output_type='numpy'  # For compatibility with sklearn metrics
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    logger.info(f"Model trained")
    
    # Evaluate the model
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    logger.info(f"Train Accuracy: {train_acc:.2f}")
    logger.info(f"Validation Accuracy: {val_acc:.2f}")

    return train_acc, val_acc, model, scaler

def extract_features(dataset, device, batch_size=512):
    """Extract features with optimized memory handling."""
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2,
        pin_memory=False,    # Enable pinned memory for faster GPU transfer
        persistent_workers=True
    )

    X_list, y_list = [], []
    for batch in tqdm(dataloader, desc="Extracting Features"):
        X, y = batch
        X = X.to(device, non_blocking=True)  # Async transfer
        logger.info(f"the devive being used before feature extraction is: {device}")
      
        # Convert to float32 immediately to save memory
        X_list.append(X.cpu().float().numpy())
        y_list.append(y.numpy())

    # Stack arrays and return
    return np.vstack(X_list), np.concatenate(y_list)

# def plot_results(results):
#     """Plot the results of the experiments."""
#     # Convert results to DataFrame
#     import pandas as pd
#     df = pd.DataFrame(results)

#     # Plot train accuracy vs C
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, x="C", y="train_acc", hue="kernel", style="gamma")
#     plt.title("Train Accuracy vs C")
#     plt.xlabel("C")
#     plt.ylabel("Train Accuracy")
#     plt.legend()
#     plt.show()

#     # Plot validation accuracy vs C
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=df, x="C", y="val_acc", hue="kernel", style="gamma")
#     plt.title("Validation Accuracy vs C")
#     plt.xlabel("C")
#     plt.ylabel("Validation Accuracy")
#     plt.legend()
#     plt.show()

def plot_results(results):
    """Enhanced function to visualize experiment results across hyperparameters."""
    # Convert results list to a DataFrame
    df = pd.DataFrame(results)

    # --- Line Plots: Metrics vs C for each kernel/gamma combination ---
    metrics = {
        "train_acc": "Train Accuracy",
        "val_acc": "Validation Accuracy",
        "test_acc": "Test Accuracy",
        "f1score": "F1 Score"
    }
    
    for metric, label in metrics.items():
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="C", y=metric, hue="kernel", style="gamma",
                     markers=True, dashes=False)
        plt.title(f"{label} vs C")
        plt.xlabel("C")
        plt.ylabel(label)
        plt.legend(title="Kernel / Gamma")
        plt.show()

    # --- Box Plots: Performance by Kernel ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.boxplot(data=df, x="kernel", y="train_acc", ax=axes[0, 0])
    axes[0, 0].set_title("Train Accuracy by Kernel")
    
    sns.boxplot(data=df, x="kernel", y="val_acc", ax=axes[0, 1])
    axes[0, 1].set_title("Validation Accuracy by Kernel")
    
    sns.boxplot(data=df, x="kernel", y="test_acc", ax=axes[1, 0])
    axes[1, 0].set_title("Test Accuracy by Kernel")
    
    sns.boxplot(data=df, x="kernel", y="f1score", ax=axes[1, 1])
    axes[1, 1].set_title("F1 Score by Kernel")
    
    plt.tight_layout()
    plt.show()

    # --- Box Plots: Performance by Gamma ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.boxplot(data=df, x="gamma", y="train_acc", ax=axes[0, 0])
    axes[0, 0].set_title("Train Accuracy by Gamma")
    
    sns.boxplot(data=df, x="gamma", y="val_acc", ax=axes[0, 1])
    axes[0, 1].set_title("Validation Accuracy by Gamma")
    
    sns.boxplot(data=df, x="gamma", y="test_acc", ax=axes[1, 0])
    axes[1, 0].set_title("Test Accuracy by Gamma")
    
    sns.boxplot(data=df, x="gamma", y="f1score", ax=axes[1, 1])
    axes[1, 1].set_title("F1 Score by Gamma")
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp.set_sharing_strategy('file_system')
    
    # List of languages
    languages = [
        'assamese', 'bengali', 'english', 'gujarati', 'hindi', 
        'kannada', 'malayalam', 'marathi', 'meitei', 'odia', 
        'punjabi', 'tamil', 'telugu', 'urdu'
    ]

    # Create and fit LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(languages)

    # Load Configuration
    with open("./conifg/svm.yaml") as f:
        config = yaml.safe_load(f)

    # Load training dataset
    train_dataset = BHSceneDataset(**config["train_dataset"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"the devive being used before feature extraction is: {device}")
    # Feature extraction with caching
    cache_path = "train_features.pkl"
    if config.get("use_cached", False) and os.path.exists(cache_path):
        logger.info("Loading cached features...")
        X_train, y_train = joblib.load(cache_path)
        # Convert to float32 for cuML
        X_train = X_train.astype('float32')
    else:
        logger.info("Extracting features...")
        X_train, y_train = extract_features(train_dataset, device)
        # Save as float32
        X_train = X_train.astype('float32')
        joblib.dump((X_train, y_train), cache_path)

    # Encode labels
    label_encoder.fit(y_train)  # Critical change
    y_train = label_encoder.transform(y_train)
    
    logger.info(f"Features Extracted")
    logger.info(f"Size of X_train: {X_train.shape}")
    logger.info(f"Size of y_train: {y_train.shape}")

    # Training with default parameters
    train_acc, val_acc, model, scaler = train(X_train, y_train, config["training_params"]["default_params"])

    # Testing with default parameters
    test_dataset = BHSceneDataset(**config["test_dataset"])
    X_test, y_test = extract_features(test_dataset, device)
    X_test = X_test.astype('float32')  # Convert to float32
    y_test = label_encoder.transform(y_test)

    # Scale test features using training scaler
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    test_acc = model.score(X_test_scaled, y_test)
    
    f1score = f1_score(y_test, y_pred, average='weighted')
    print("y_test->")
    print(y_test)
    print("y_pred->")
    print(y_pred)
    
    
    logger.info(f"F1 Score: {f1score:.2f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}")
    # Save the model
    with open(f"svm_model_C{C}_kernel{kernel}_gamma{gamma}.pkl", "wb") as f:
        pickle.dump(model, f)
    results = []
    hyperparameters = config["training_params"]["hyperparameter_range"]

    # Hyperparameter search with fixed parameter passing
    for C in hyperparameters["C"]:
        for kernel in hyperparameters["kernel"]:
            for gamma in hyperparameters["gamma"]:
                current_config = {
                    "C": C,
                    "kernel": kernel,
                    "gamma": gamma
                }
                try:
                    train_acc, val_acc, model, scaler = train(X_train, y_train, current_config)
                    
                    # Scale test features using this model's scaler
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Test evaluation
                    test_acc = model.score(X_test_scaled, y_test)
                    y_pred = model.predict(X_test_scaled)
                    f1score = f1_score(y_test, y_pred, average='weighted')
                    logger.info(f"F1 Score: {f1score:.2f}")
                    logger.info(f"Test Accuracy: {test_acc:.2f}")
                    results.append({
                        "C": C,
                        "kernel": kernel,
                        "gamma": gamma,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "f1score": f1score
                    })
                    with open(f"svm_model_C{C}_kernel{kernel}_gamma{gamma}.pkl", "wb") as f:
                        pickle.dump(model, f)
                    
                except Exception as e:
                    logger.error(f"Failed for {current_config}: {str(e)}")

    print(results)
    # Plotting and results
    plot_results(results)
    
    # Save best model
    if results:
        best_model = max(results, key=lambda x: x["test_acc"])
        best_model_params = {
            "C": best_model["C"],
            "kernel": best_model["kernel"],
            "gamma": best_model["gamma"]
        }
        logger.info(f"Best Model: {best_model_params}")
        logger.info(f"Best Test Accuracy: {best_model['test_acc']:.2f}")
        logger.info(f"F1_Score: {best_model['f1score']:.2f}")
        files.download(f"svm_model_C{ best_model["C"]}_kernel{best_model["kernel"]}_gamma{best_model["gamma"]}.pkl")
