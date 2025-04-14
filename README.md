# Script-Identifier

**Scene Text Script Classification using Traditional Machine Learning Techniques**

## Overview

**Script-Identifier** is a Pattern Recognition and Machine Learning (PRML) course project developed as part of CSL2050. The objective of the project is to identify the script/language of scene text images using traditional machine learning techniques. The system is trained and evaluated on the **Bharat Scene Text Dataset**, and the focus is on comparing classical ML algorithms with various handcrafted and deep feature extractors.

This project adheres to the CSL2050 guidelines, emphasizing rigorous evaluation, failure case analysis, and comprehensive deliverables including code, report, demo, and presentation.

---

## Project Team

Developed by Team **Pending**  
Under **CSL2050 - Pattern Recognition and Machine Learning**

---


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

```
.
├── config/          # Configuration files for experiments
├── dataset/         # Data loaders and transformations
├── models/          # ML models and backbones
├── tools/           # Training and inference scripts
├── utils/           # Plotting and utility functions
├── Visualisation/   # Visualization scripts
├── data/            # Placeholder for processed data and latents
├── frontend/        # Next.js frontend interface
├── main.py          # Entry point for inference
├── ui.py            # Gradio-based UI
├── fastapi_server.py# REST API with FastAPI
├── requirements.txt # Dependencies
└── README.md        # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (for faster training)
- pip / conda for package management

### Setup

```bash
git clone https://github.com/AurindumBanerjee/Script-Identifier.git
cd Script-Identifier
pip install -r requirements.txt
```

---

## How to Run

To be added model-wise.

---

## Dataset

We use the **[Bharat Scene Text Dataset](https://github.com/Bhashini-IITJ/BharatSceneTextDataset)** containing scene text in 13 Indian scripts. 

In the data, we specifically use the Cropped Word Recognition Set. Data splits are provided for training and testing.


| Language | #Train | #Test|
| :---: | :---: | :---: |
| Assamese  | 2,623 | 1,343 |
| Bengali | 4,968 | 1,161 |
| English | 28,778 | 8,113 |
| Gujarati | 1,956 | 693 |
| Hindi | 14,855 | 4,034 |
| Kannada | 2,241 | 693 |
| Malayalam | 2,408 | 567 |
| Marathi | 3,932 | 1,045 |
| Odia | 3,176 | 1,022 |
| Punjabi | 8,544 | 2,560 |
| Tamil | 2,041 | 507 |
| Telugu | 2,227 | 482 |
|Total| 77,749 | 22,220 |


---

## Deliverables

As per CSL2050 project requirements:

| Component            | Status       |
|----------------------|--------------|
| Mid-Project Report   | ✅ Submitted  |
| Final Report         | ✅ Submitted   |
| GitHub Repository    | ✅ Updated    |
| Project Page         | ✅ [`Web Page`](https://aurindumbanerjee.github.io/Script-Identifier/) |
| Web Demo (Gradio)    | ✅ Created   |
| Spotlight Video      | ✅ Submitted |
| Minutes of Meetings  | ✅ Maintained |


---

## License

This project is licensed under the [MIT License](./LICENSE).
