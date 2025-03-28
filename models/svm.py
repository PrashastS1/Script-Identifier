

import numpy as np
from collections import Counter
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from loguru import logger
from dataset.BH_scene_dataset import BHSceneDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn import svm
import yaml
import os

# List of languages from the image
languages = [
    'assamese', 'bengali', 'english', 'gujarati', 'hindi', 
    'kannada', 'malayalam', 'marathi', 'meitei', 'odia', 
    'punjabi', 'tamil', 'telugu', 'urdu'
]

# Create a LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(languages)

#Load Configuration
with open("./conifg/svm.yaml") as f:
    config = yaml.safe_load(f)

def train(X_train, y_train, config):
    """Run the experiment using the given configuration."""
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train
    )
    #log info about the experiment
    logger.info(f"C: {config['C']}")
    logger.info(f"Kernel: {config['kernel']}")
    logger.info(f"Gamma: {config['gamma']}")

    # Create the SVM model for 14 classes
    model = svm.SVC(
        C=config["C"], kernel=config["kernel"], gamma=config["gamma"], decision_function_shape="ovr"
    )

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_val, y_val)
    # Log the results
    logger.info(f"Train Accuracy: {train_acc:.2f}")
    logger.info(f"Validation Accuracy: {val_acc:.2f}")

    return train_acc, val_acc, model

# Load the dataset
training_dataset=config["train_dataset"]
train_dataset = BHSceneDataset(
    root_dir=training_dataset["root_dir"],
    train_split=training_dataset["train_split"],
    transform= training_dataset["transform"],
    linear_transform= training_dataset["linear_transform"], 
    backbone=training_dataset["backbone"],
    gap_dim= training_dataset["gap_dim"]
)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Extract features from the dataset
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


X_train, y_train = extract_features(train_dataset)


#training with default parameters
train_acc, val_acc,model= train(X_train, y_train, config["training_params"]["default_params"])

#testing with default parameters
testing_dataset=config["test_dataset"]

test_dataset = BHSceneDataset(
    root_dir=testing_dataset["root_dir"],
    train_split= testing_dataset["train_split"],
    transform=testing_dataset["transform"],
    linear_transform= testing_dataset["linear_transform"], 
    backbone= testing_dataset["backbone"],
    gap_dim=testing_dataset["gap_dim"]
)

X_test, y_test = extract_features(test_dataset)

test_acc = model.score(X_test, y_test)
logger.info(f"Test Accuracy: {test_acc:.2f}")