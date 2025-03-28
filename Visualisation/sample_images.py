import os
import random
import plotly.subplots as sp
import plotly.graph_objects as go
import cv2
import numpy as np

def load_sample_images(base_dir):
    """Load exactly 5 sample images from each script category."""
    scripts = sorted(os.listdir(base_dir))  # Get all script folder names
    sample_images = {}
    
    for script in scripts:
        script_path = os.path.join(base_dir, script)
        if os.path.isdir(script_path):
            images = os.listdir(script_path)
            if len(images) >= 5:
                selected_images = random.sample(images, 5)  # Ensure exactly 5 images per script
            else:
                selected_images = images  # If fewer than 5 images exist, take all
            sample_images[script] = [os.path.join(script_path, img) for img in selected_images]
    
    return sample_images

def plot_sample_images(sample_images):
    """Plot a grid of sample images using Plotly without axes and minimal empty space."""
    num_scripts = len(sample_images)
    num_samples = 5
    
    fig = sp.make_subplots(rows=num_scripts, cols=num_samples, vertical_spacing=0.05, horizontal_spacing=0.05, subplot_titles=[script for script in sample_images.keys() for _ in range(num_samples)])
    
    row = 1
    for script, images in sample_images.items():
        for col, img_path in enumerate(images, start=1):
            img = cv2.imread(img_path)
            fig.add_trace(go.Image(z=img),row=row, col=col)
        row += 1
    
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(height=num_scripts * 250, width=num_samples * 250, title_text="Sample Images from Different Scripts", title_x=0.5, showlegend=False)
    
    fig.show()

# Set the path to the dataset
data_dir = r"C:\Users\jenis_td7jjpo\Desktop\PRML\Project\Script-Identifier\data\recognition\train"  # Adjust the path accordingly
sample_images = load_sample_images(data_dir)
plot_sample_images(sample_images)