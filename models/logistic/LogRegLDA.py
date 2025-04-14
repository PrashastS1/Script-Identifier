import os
import yaml
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
from utils.logreg_plot_utils import plot_decision_boundary

# Only use for Binary Classification (1 vs 0)

# Command to run : 
# python -m models.Logistic.LogRegLDA


def load_features(dataset, target_language_id):
    """
    Function to load features and labels from the dataset.

    Inputs :

        dataset: Dataset object containing the data.
        target_language_id: The language ID for the target language.

    Outputs :

        features: Numpy array of features.
        labels: Numpy array of binary labels (1 for target language, 0 otherwise).

    """


    features = []
    labels = []


    for i in tqdm(range(len(dataset)), desc="Extracting Features"):
        x, y = dataset[i]

        # Convert to numpy array if it's a tensor
        features.append(x.cpu().numpy() if hasattr(x, 'cpu') else x)

        # Assign 1-0 labeling for binary classification
        
        if y == target_language_id:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(features), np.array(labels)



def setup_logger(log_dir: str, exp_name: str):
    """
    Used to create log files for the experiment. Starts the logging process.

    Input :
    
            log_dir: Directory where all logs will be saved.
            exp_name: Name of the experiment conducted. Generally mentions language, processing and backbone done.

    Output :

            log_path: Path to the log file created.    
        
    """

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{exp_name}.txt")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )
    
    # Logging starts
    logging.info("\n \n")
    logging.info("========================New Run Started========================")
    return log_path


def main():
    """
    Main function to run the Logistic Regression model.
    Deals with all the finer stuff like loading the dataset, extracting features, training the model and evaluating it.
    Also handles PCA and LDA if specified in the config file.

    """

    # Load configuration file
    with open("conifg/logreg.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Extract configuration parameters from yaml file
    lang = config["target"]["language"]
    dataset_args = config["dataset"]
    logreg_cfg = config["logreg_params"]

    pca_flag = logreg_cfg.get("use_pca", False)
    pca_dim = logreg_cfg.get("pca_components", 100)
    lda_flag = logreg_cfg.get("use_lda", False)
    lda_mode = logreg_cfg.get("lda_mode", "binary")
    save_plots = logreg_cfg.get("save_plots", False)
    exp_name = logreg_cfg.get("exp_name", "logreg_exp")
    backbone_name = dataset_args.get("backbone", "unknown")

    # Setup logger
    log_path = setup_logger("logs", exp_name)
    print(f"[INFO] Logging to: {log_path}")

    # Load datasets
    train_dataset = BHSceneDataset(**dataset_args)
    test_dataset = BHSceneDataset(**{**dataset_args, "train_split": False})

    # Load language-to-ID mapping
    with open('./dataset/language_encode.json') as f:
        lang_map = yaml.safe_load(f)

    lang_id = lang_map[lang]

    # Load features and binary labels
    x_train, y_train = load_features(train_dataset, lang_id)
    x_test, y_test = load_features(test_dataset, lang_id)

    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Optional PCA, specify in yaml
    if pca_flag:
        print("[INFO] Applying PCA...")
        logging.info(f"PCA enabled: True, Components: {pca_dim}")
        pca = PCA(n_components=pca_dim)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    # Optional LDA, specify in yaml
    if lda_flag:
        print(f"[INFO] Applying LDA ({lda_mode})...")
        logging.info(f"LDA enabled: True, Mode: {lda_mode}")

        n_classes = len(np.unique(y_train))
        logging.info(f"Number of classes in dataset: {n_classes}")
        logging.info(f"Feature dimensions before LDA: {x_train.shape[1]}")

        # Only works if binary, would redirect to multiclass LDA otherwise in LogRegTest code.
        if lda_mode == "binary":
            if x_train.shape[1] < 1:
                print("[WARN] Skipping LDA — feature dimension < 1")
                logging.warning("[SKIP LDA] Feature dimension too low for binary LDA")
            else:
                lda = LDA(n_components=1)
                x_train = lda.fit_transform(x_train, y_train)
                x_test = lda.transform(x_test)
                logging.info("[LDA] Applied binary LDA with 1 component")


    # Train Logistic Regression
    model = LogisticRegression(solver='liblinear', class_weight='balanced', penalty='l2')
    model.fit(x_train, y_train)

    # Evaluate model
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Metrics
    print(f"\n[INFO] Accuracy: {acc * 100:.2f}%")
    print("[INFO] Classification Report:")
    print(report)

    # Logging details and metrics
    logging.info(f"Backbone: {backbone_name}")
    logging.info(f"Target Language: {lang}")
    logging.info(f"Accuracy: {acc * 100:.2f}%")
    logging.info("Classification Report:\n" + report)

    # Save decision boundary plot
    if save_plots:
        plot_dir = os.path.join("plots", "logreg", exp_name)
        plot_decision_boundary(x_train, y_train, model, lang, plot_dir, backbone = backbone_name)

        logging.info(f"Decision boundary plot saved to: {plot_dir}")

    # Logging ends [for current experiment]
    logging.info("====================Run Completed====================")

if __name__ == "__main__":
    main()



