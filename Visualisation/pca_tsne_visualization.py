import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import numpy as np
import plotly.express as px
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from models.backbones.resnet50 import RESNET_backbone
from models.backbones.vgg import VGG_backbone
from models.backbones.vit import VIT_backbone

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_images(base_dir, num_samples=50):
    """Load images and their corresponding labels."""
    scripts = sorted(os.listdir(base_dir))
    image_paths, labels = [], []
    
    for script in scripts:
        script_path = os.path.join(base_dir, script)
        if os.path.isdir(script_path):
            images = os.listdir(script_path)
            selected_images = images[:num_samples]
            for img in selected_images:
                image_paths.append(os.path.join(script_path, img))
                labels.append(script)
    
    return image_paths, labels

def extract_features(model, image_paths):
    """Extract features using a given model."""
    model.to(device).eval()
    features = []
    
    with torch.no_grad():
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = transform(img).unsqueeze(0).to(device)
            feat = model(img).cpu().numpy().flatten()
            features.append(feat)
    
    return np.array(features)

def reduce_dimensions(features, method='pca', n_components=2):
    """Reduce feature dimensions using PCA or t-SNE."""
    if method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=42)
    
    return reducer.fit_transform(features)

def plot_clusters(embeddings, labels, title):
    """Plot the 2D feature embeddings as a scatter plot."""
    fig = px.scatter(
        x=embeddings[:, 0], y=embeddings[:, 1],
        color=labels,
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        title=title,
    )
    fig.show()

# Load images
data_dir = r"C:\Users\Jenish\Desktop\PRML\Project\Script-Identifier\data\recognition\train"  # Adjust the path accordingly
image_paths, labels = load_images(data_dir)

# Load models
resnet_model = RESNET_backbone(pretrained=True)
vgg_model = VGG_backbone(pretrained=True)
vit_model = VIT_backbone(pretrained=True)

# Extract features
resnet_features = extract_features(resnet_model, image_paths)
vgg_features = extract_features(vgg_model, image_paths)
vit_features = extract_features(vit_model, image_paths)

# Dimensionality reduction using TSNE
resnet_embedded_tsne = reduce_dimensions(resnet_features, method='tsne')
vgg_embedded_tsne = reduce_dimensions(vgg_features, method='tsne')
vit_embedded_tsne = reduce_dimensions(vit_features, method='tsne')

# Dimensionality reduction using PCA
resnet_embedded_pca = reduce_dimensions(resnet_features, method='pca')
vgg_embedded_pca = reduce_dimensions(vgg_features, method='pca')
vit_embedded_pca = reduce_dimensions(vit_features, method='pca')

# Plot results
plot_clusters(resnet_embedded_tsne, labels, "TSNE Visualization of ResNet50 Features")
plot_clusters(vgg_embedded_tsne, labels, "TSNE Visualization of VGG16 Features")
plot_clusters(vit_embedded_tsne, labels, "TSNE Visualization of ViT Features")

# Plot results
plot_clusters(resnet_embedded_pca, labels, "PCA Visualization of ResNet50 Features")
plot_clusters(vgg_embedded_pca, labels, "PCA Visualization of VGG16 Features")
plot_clusters(vit_embedded_pca, labels, "PCA Visualization of ViT Features")