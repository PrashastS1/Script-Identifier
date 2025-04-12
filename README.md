# Script-Identifier

**Scene Text Script Classification using Traditional Machine Learning Techniques**

## Overview

**Script-Identifier** is a Pattern Recognition and Machine Learning (PRML) course project developed as part of CSL2050. The objective of the project is to identify the script/language of scene text images using traditional machine learning techniques. The system is trained and evaluated on the **Bharat Scene Text Dataset**, and the focus is on comparing classical ML algorithms with various handcrafted and deep feature extractors.

This project adheres to the CSL2050 guidelines, emphasizing rigorous evaluation, failure case analysis, and comprehensive deliverables including code, report, demo, and presentation.

## Features

- Classification of scene text into scripts such as Hindi, Tamil, Bengali, etc.
- Traditional ML models: SVM, KNN, ANN, Decision Trees, Logistic Regression, etc.
- Feature extraction techniques: HOG, SIFT, ResNet, VGG, Vision Transformer (ViT)
- Dimensionality reduction: PCA, LDA
- Visualizations: t-SNE, PCA, class distributions, and decision boundaries
- FastAPI and Gradio-based interfaces for inference and demo
- Modular, well-documented codebase with YAML-based configuration

---

## Repository Structure

.├── config/ # Configuration files for experiments ├── dataset/ # Data loaders and transformations ├── models/ # ML models and backbones ├── tools/ # Training and inference scripts ├── utils/ # Plotting and utility functions ├── Visualisation/ # Visualization scripts ├── data/ # Placeholder for processed data and latents ├── frontend/ # Next.js frontend interface ├── main.py # Entry point for inference ├── ui.py # Gradio-based UI ├── fastapi_server.py# REST API with FastAPI ├── requirements.txt # Dependencies └── README.md # Project documentation
