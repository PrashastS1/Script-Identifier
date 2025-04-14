# Logistic Regression - Script Identifier

This folder contains all modules, scripts, and batch pipelines for training and evaluating Logistic Regression models using various feature extraction backbones. The goal is to classify scene-text images based on the script/language.

## Folder Structure

```
models/Logistic/
├── LogBackBone.py             # Baseline using HOG + Logistic Regression
├── LogRegLDA.py               # Pipeline supporting PCA + LDA + Logistic Regression
├── LogRegTest.py              # Batch evaluation for all languages (1-vs-rest LDA)
├── LogRegLDAMulticlass.py     # One-pass multiclass training using PCA + LDA
├── progress/                  # Stores completed language logs to avoid recomputation [created after running your first experiment]
└── LogRegHOG/                 # Old experimentation code

Script-Identifier [created after running your first experiment]
├── plots/                     # Decision boundary plots
└── logs/                      # Per-experiment logs with metrics and reports
```

## Techniques Used

### Feature Extraction
- HOG (Histogram of Oriented Gradients)
- SIFT (Scale-Invariant Feature Transform)
- CNN Backbones:
  - ResNet50
  - VGG
  - ViT (Vision Transformer)

### Dimensionality Reduction
- PCA (Principal Component Analysis)
- LDA (Linear Discriminant Analysis)
  - Binary (for 1-vs-rest setup)
  - Multiclass (support for 13 languages)

### Classification
- Logistic Regression
  - Solver: liblinear, sag
  - Balanced class weighting
  - Configurable regularization (L2)

## Configurable Parameters

All hyperparameters are controlled via `conifg/logreg.yaml`:
- Backbone type
- PCA usage and component count
- LDA mode (binary or multiclass)
- Experiment name and plot saving
- Target language (used in binary LDA setup)

## Utilities

- `plot_decision_boundary`: Visualizes classification boundary using 2D PCA
- YAML-driven pipeline for reproducibility
- Batch logger to resume training after failure

## `logreg.yaml` Configuration Reference

This file controls how Logistic Regression is executed on the script recognition dataset. It includes settings for the dataset, backbone model, pre-processing pipeline, and training target.

---

### `dataset`

Controls how data is loaded and processed.

| Key             | Description                                                       | Type     | Values / Examples                                        |
|------------------|-------------------------------------------------------------------|----------|-----------------------------------------------------------|
| `root_dir`       | Path to the dataset folder                                        | `str`    | `"data/recognition"`                                      |
| `train_split`    | Whether to use training (`True`) or test (`False`) split         | `bool`   | `true`, `false`                                           |
| `transformation` | Whether to apply data augmentation (Albumentations)              | `bool`   | `true`, `false`                                           |
| `backbone`       | Feature extractor model used before classification               | `str`    | `hog`, `sift`, `resnet50`, `vit`, `vit_huge`, `null`      |
| `gap_dim`        | Global average pooling dim used in CNN backbones (if applicable) | `int`    | `1`, `2`, etc. (ignored for HOG/SIFT)                     |

---

### `logreg_params`

Controls the preprocessing and learning pipeline.

| Key                | Description                                                         | Type     | Values / Examples                                  |
|--------------------|---------------------------------------------------------------------|----------|----------------------------------------------------|
| `exp_name`         | Name of the experiment (used in log/plot directories)               | `str`    | `"logreg_vit_pca_lda"`, `"hog_test"`              |
| `save_plots`       | Whether to save decision boundary plots                             | `bool`   | `true`, `false`                                    |
| `use_pca`          | Whether to apply PCA for dimensionality reduction                   | `bool`   | `true`, `false`                                    |
| `pca_components`   | Number of PCA components to keep                                     | `int`    | `2`, `50`, `100`                                   |
| `use_lda`          | Whether to apply Linear Discriminant Analysis (LDA)                 | `bool`   | `true`, `false`                                    |
| `lda_mode`         | Whether to use binary (1-vs-rest) or multiclass LDA                 | `str`    | `binary`, `multiclass`                             |

---

### `target`

Target class configuration for binary classification.

| Key         | Description                                            | Type   | Values / Examples                       |
|-------------|--------------------------------------------------------|--------|------------------------------------------|
| `language`  | The positive class for 1-vs-rest classification        | `str`  | One of: `assamese`, `bengali`, `english`, `gujarati`, `hindi`, `kannada`, `malayalam`, `marathi`, 'meitei', `odia`, `punjabi`, `tamil`, `telugu`, `urdu` |

---

### Sample YAML
```yaml
dataset:
  root_dir: "data/recognition"
  train_split: true
  transformation: true
  backbone: resnet50
  gap_dim: 1

logreg_params:
  save_plots: true
  exp_name: "logreg_resnet50"
  use_pca: true
  pca_components: 100
  use_lda: true
  lda_mode: binary

target:
  language: "bengali"

```

---

## Usage

Adjust yaml parameters to define running conditions.


To train for a single language:

```bash
python -m models.Logistic.LogRegLDA
```

To run multiclass once for all:

```bash
python -m models.Logistic.LogRegLDAMulticlass
```

To run LDA-binary for each language sequentially or just call LDA-Multiclass once. Will take yaml file and do testing correspondingly:

```bash
python -m models.Logistic.LogRegTest
```

---

## References

[How to save models using pickle](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)

[How to save models using joblib](https://www.analyticsvidhya.com/blog/2021/08/quick-hacks-to-save-machine-learning-model-using-pickle-and-joblib/)

[What is Multi-class LDA ?](https://multivariatestatsjl.readthedocs.io/en/latest/mclda.html)

[How to configure Logistic Regression to do Multi-class classification](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

