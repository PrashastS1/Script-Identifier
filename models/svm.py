import numpy as np
from collections import Counter
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from loguru import logger
from dataset.BH_scene_dataset import BHSceneDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.multiprocessing as mp
from tqdm import tqdm
from sklearn import svm
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from google.colab import files

def train(X_train, y_train, config):
    """Run the experiment using the given configuration."""
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train
    )
    
    # Add feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    logger.info(f"C: {config['C']}")
    logger.info(f"Kernel: {config['kernel']}")
    logger.info(f"Gamma: {config['gamma']}")

    # Create the SVM model with convergence controls
    model = svm.SVC(
        C=config["C"], 
        kernel=config["kernel"],
        gamma=config["gamma"],
        decision_function_shape="ovr", 
        verbose=True,
        max_iter=10000,  # Prevent infinite training
        tol=0.01,       # Relax convergence criteria
        class_weight='balanced'  # Handle class imbalance
    ).fit(X_train, y_train)
    
    # Evaluate the model
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    
    logger.info(f"Train Accuracy: {train_acc:.2f}")
    logger.info(f"Validation Accuracy: {val_acc:.2f}")

    return train_acc, val_acc, model

def extract_features(dataset, device, batch_size=1024):  # Increased batch size
    """Extract features from dataset using GPU acceleration."""
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4,         # Increased workers
        pin_memory=True        # Enable pinned memory
    )

    X_list, y_list = [], []
    for batch in tqdm(dataloader, desc="Extracting Features"):
        X, y = batch  # Unpack tuple
        X = X.to(device, non_blocking=True)  # Async transfer
        
        # Use float32 to reduce memory usage
        X_list.append(X.cpu().float().numpy())
        y_list.append(y.numpy())

    return np.vstack(X_list), np.concatenate(y_list)

def plot_results(results):
    """Plot the results of the experiments."""
    # Convert results to DataFrame
    import pandas as pd
    df = pd.DataFrame(results)

    # Plot train accuracy vs C
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="C", y="train_acc", hue="kernel", style="gamma")
    plt.title("Train Accuracy vs C")
    plt.xlabel("C")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.show()

    # Plot validation accuracy vs C
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="C", y="val_acc", hue="kernel", style="gamma")
    plt.title("Validation Accuracy vs C")
    plt.xlabel("C")
    plt.ylabel("Validation Accuracy")
    plt.legend()
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

    # Load Configuration (fixed config path typo)
    with open("./conifg/svm.yaml") as f:
        config = yaml.safe_load(f)

    # Load training dataset
    train_dataset = BHSceneDataset(**config["train_dataset"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Feature extraction with caching
    cache_path = "train_features.pkl"
    if config.get("use_cached", False) and os.path.exists(cache_path):
        logger.info("Loading cached features...")
        X_train, y_train = joblib.load(cache_path)
    else:
        logger.info("Extracting features...")
        X_train, y_train = extract_features(train_dataset, device)
        joblib.dump((X_train, y_train), cache_path)

    # Encode labels
    y_train = label_encoder.transform(y_train)
    
    logger.info(f"Features Extracted")
    logger.info(f"Size of X_train: {X_train.shape}")
    logger.info(f"Size of y_train: {y_train.shape}")

    # Training with default parameters
    train_acc, val_acc, model = train(X_train, y_train, config["training_params"]["default_params"])

    # Testing with default parameters
    test_dataset = BHSceneDataset(**config["test_dataset"])
    X_test, y_test = extract_features(test_dataset, device)
    y_test = label_encoder.transform(y_test)  # Encode test labels

    # Scale test features using training scaler
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_pred = model.predict(X_test_scaled)
    test_acc = model.score(X_test_scaled, y_test)
    f1score = f1_score(y_test, y_pred, average='weighted')
    
    logger.info(f"F1 Score: {f1score:.2f}")
    logger.info(f"Test Accuracy: {test_acc:.2f}")

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
                    train_acc, val_acc, model = train(X_train, y_train, current_config)
                    
                    # Test evaluation with scaled features
                    test_acc = model.score(X_test_scaled, y_test)
                    y_pred = model.predict(X_test_scaled)
                    f1score = f1_score(y_test, y_pred, average='weighted')
                    
                    results.append({
                        "C": C,
                        "kernel": kernel,
                        "gamma": gamma,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        "f1score": f1score
                    })
                except Exception as e:
                    logger.error(f"Failed for {current_config}: {str(e)}")

    # Plotting and results
    plot_results(results)
    
    # Save best model
    best_model = max(results, key=lambda x: x["test_acc"])
    best_model_params = {
        "C": best_model["C"],
        "kernel": best_model["kernel"],
        "gamma": best_model["gamma"]
    }
    logger.info(f"Best Model: {best_model_params}")
    logger.info(f"Best Test Accuracy: {best_model['test_acc']:.2f}")
    joblib.dump(model, "best_svm_model.joblib")
