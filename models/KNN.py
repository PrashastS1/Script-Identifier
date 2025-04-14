import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import sys
import torch.multiprocessing as mp
from loguru import logger
import time
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/kaggle/input/prml-dataset10/script-id')
from dataset.BH_scene_dataset import BHSceneDataset

# --- Configuration ---
PLOTS_DIR = "/kaggle/working/plots"
BACKBONE = "sift"
DATA_ROOT = '/kaggle/input/prml-dataset10/script-id/data/recognition'
FEATURES_DIR = "/kaggle/working/features"
FIXED_GAP_DIM = None

# --- Hyperparameter Grid for GridSearchCV ---
PARAM_GRID = {
    'n_components': [50, 100, 200],
    'whiten': [False, True],
    'n_neighbors': [3, 5, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'cosine']
}

# --- Helper Functions ---
def extract_features(dataset, batch_size=128):
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    logger.info(f"Number of GPUs available: {num_gpus}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory = False
    )
    X_list, y_list = [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Feature extraction using device: {device}")

    # If multiple GPUs, wrap the dataset's backbone (assumed to be ViT) with DataParallel
    if num_gpus > 1 and hasattr(dataset, 'backbone') and dataset.backbone is not None:
        dataset.backbone = torch.nn.DataParallel(dataset.backbone)
        logger.info("Wrapped backbone with DataParallel for multi-GPU processing")
    
    for batch in tqdm(dataloader, desc="Extracting Features"):
        X, y = batch
        # X and y are processed by the backbone (on GPU if DataParallel is used)
        X_list.append(X.cpu().numpy())  # Move back to CPU for NumPy
        y_list.append(y.cpu().numpy())
    return np.vstack(X_list), np.concatenate(y_list)

def save_features(features_dir, backbone, X_train, y_train, X_test, y_test):
    os.makedirs(features_dir, exist_ok=True)
    np.save(os.path.join(features_dir, f"{backbone}_X_train.npy"), X_train)
    np.save(os.path.join(features_dir, f"{backbone}_y_train.npy"), y_train)
    np.save(os.path.join(features_dir, f"{backbone}_X_test.npy"), X_test)
    np.save(os.path.join(features_dir, f"{backbone}_y_test.npy"), y_test)
    logger.info(f"Saved features for backbone '{backbone}' to {features_dir}")

def load_features(features_dir, backbone):
    try:
        X_train = np.load(os.path.join(features_dir, f"{backbone}_X_train.npy"))
        y_train = np.load(os.path.join(features_dir, f"{backbone}_y_train.npy"))
        X_test = np.load(os.path.join(features_dir, f"{backbone}_X_test.npy"))
        y_test = np.load(os.path.join(features_dir, f"{backbone}_y_test.npy"))
        logger.info(f"Loaded pre-extracted features for backbone '{backbone}' from {features_dir}")
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        logger.warning(f"Feature files for backbone '{backbone}' not found in {features_dir}.")
        return None

# --- Plotting Functions ---
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

def plot_accuracy_vs_hyperparameters(grid_results, param_name, title="Accuracy vs Hyperparameter", filename="accuracy_vs_hyperparameter.png"):
    param_values = grid_results['param_' + param_name]
    scores = grid_results['mean_test_score']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=param_values, y=scores)
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Accuracy')
    plt.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close()

# --- Main Execution ---
if __name__ == '__main__':
    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Attempting to set multiprocessing start method to 'spawn'")
    except RuntimeError as e:
        logger.warning(f"Could not set start method to 'spawn': {e}. Using default.")

    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            logger.info(f"Multiple GPUs detected: {torch.cuda.device_count()} devices")
    else:
        device = "cpu"
        logger.warning("CUDA GPU not available. Running on CPU (feature extraction will be slow).")

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    logger.info(f"Using features directory: {FEATURES_DIR}")
    logger.info(f"Using plots directory: {PLOTS_DIR}")

    logger.info(f"--- Processing Backbone: {BACKBONE} ---")
    start_time = time.time()

    loaded_data = load_features(FEATURES_DIR, BACKBONE)

    if loaded_data:
        X_train, y_train, X_test, y_test = loaded_data
    else:
        logger.info(f"Extracting features for backbone '{BACKBONE}'...")
        try:
            if not os.path.isdir(DATA_ROOT):
                raise FileNotFoundError(f"Data root directory not found: {DATA_ROOT}")
            if device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache.")

            logger.info("Loading training dataset...")
            dataset_kwargs = {
                'root_dir': DATA_ROOT,
                'train_split': True,
                'transformation': None,
                'backbone': BACKBONE
            }
            if FIXED_GAP_DIM is not None:
                dataset_kwargs['gap_dim'] = FIXED_GAP_DIM
            train_dataset = BHSceneDataset(**dataset_kwargs)

            X_train, y_train = extract_features(train_dataset)
            logger.info(f"Training features extracted. Shape: {X_train.shape}, Labels shape: {y_train.shape}")

            logger.info("Loading testing dataset...")
            dataset_kwargs['train_split'] = False
            test_dataset = BHSceneDataset(**dataset_kwargs)
            X_test, y_test = extract_features(test_dataset)
            logger.info(f"Testing features extracted. Shape: {X_test.shape}, Labels shape: {y_test.shape}")

            # save_features(FEATURES_DIR, BACKBONE, X_train, y_train, X_test, y_test)

        except FileNotFoundError as fnf_error:
            logger.error(str(fnf_error))
            logger.error("Cannot proceed without data. Exiting.")
            exit()
        except Exception as e:
            logger.error(f"Fatal error during feature extraction for {BACKBONE}: {e}", exc_info=True)
            logger.error("Cannot proceed without features. Exiting.")
            exit()

    feature_proc_time = time.time() - start_time
    logger.info(f"Feature loading/extraction took {feature_proc_time:.2f} seconds.")

    # --- Grid Search for PCA and KNN ---
    logger.info("Starting GridSearchCV for PCA and KNN hyperparameters...")
    grid_start_time = time.time()

    best_acc = 0
    best_params = None
    best_model = None
    all_grid_results = []

    for n_components in PARAM_GRID['n_components']:
        for whiten in PARAM_GRID['whiten']:
            logger.info(f"Applying PCA with n_components={n_components}, whiten={whiten}")
            pca = PCA(n_components=n_components, whiten=whiten)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            logger.info(f"PCA transformed training features to shape: {X_train_pca.shape}")
            logger.info(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

            knn = KNeighborsClassifier(n_jobs=-1)
            grid_search = GridSearchCV(
                knn,
                param_grid={
                    'n_neighbors': PARAM_GRID['n_neighbors'],
                    'weights': PARAM_GRID['weights'],
                    'metric': PARAM_GRID['metric']
                },
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_pca, y_train)
            all_grid_results.append(grid_search.cv_results_)

            logger.info(f"Best KNN parameters for PCA (n_components={n_components}, whiten={whiten}): {grid_search.best_params_}")
            logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

            y_pred = grid_search.predict(X_test_pca)
            test_acc = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            logger.info(f"Test accuracy: {test_acc:.4f}, Test F1 score: {test_f1:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_params = {
                    'n_components': n_components,
                    'whiten': whiten,
                    **grid_search.best_params_
                }
                best_model = grid_search.best_estimator_

    grid_time = time.time() - grid_start_time
    logger.info(f"GridSearchCV completed in {grid_time:.2f} seconds.")

    # --- Final Evaluation with Best Model ---
    logger.success(f"\n--- Best Model Results ---")
    logger.success(f"Backbone: {BACKBONE}")
    logger.success(f"Best Parameters: {best_params}")
    logger.success(f"Best Test Accuracy: {best_acc:.4f}")

    pca = PCA(n_components=best_params['n_components'], whiten=best_params['whiten'])
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    best_model.fit(X_train_pca, y_train)
    y_pred = best_model.predict(X_test_pca)
    final_acc = accuracy_score(y_test, y_pred)
    final_f1 = f1_score(y_test, y_pred, average='weighted')

    logger.success(f"Final Accuracy (recomputed): {final_acc:.4f}")
    logger.success(f"Final F1 Score (weighted): {final_f1:.4f}")

    # --- Plotting ---
    plot_confusion_matrix(
        y_test, y_pred,
        title=f"Confusion Matrix for Best Model (Backbone: {BACKBONE})",
        filename=f"confusion_matrix_{BACKBONE}.png"
    )

    plot_accuracy_vs_hyperparameters(
        all_grid_results[0], 'n_neighbors',
        title=f"Accuracy vs. n_neighbors (Backbone: {BACKBONE})",
        filename=f"accuracy_vs_n_neighbors_{BACKBONE}.png"
    )

    logger.info("--- Script Finished ---")
