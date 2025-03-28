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

# Load the dataset
train_dataset = BHSceneDataset(
    root_dir="data/recognition",
    train_split=True,
    transform=None,
    linear_transform=True,
    backbone='resnet50',
    gap_dim=1
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


with open("./conifg/svm.yaml") as f:
    config = yaml.safe_load(f)