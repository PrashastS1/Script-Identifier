import os
import yaml
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from dataset.BH_scene_dataset import BHSceneDataset
from utils.logreg_plot_utils import plot_decision_boundary


def load_features(dataset, target_language_id):
    features = []
    labels = []

    for i in tqdm(range(len(dataset)), desc="Extracting Features"):
        x, y = dataset[i]
        features.append(x.cpu().numpy() if hasattr(x, 'cpu') else x)
        labels.append(1 if y == target_language_id else 0)

    return np.array(features), np.array(labels)


def main():
    # Load config
    with open("conifg/logreg.yaml", "r") as f:
        config = yaml.safe_load(f)

    lang = config["target"]["language"]
    dataset_args = config["dataset"]
    pca_flag = config["logreg_params"]["use_pca"]
    pca_dim = config["logreg_params"]["pca_components"]
    save_plots = config["logreg_params"]["save_plots"]
    exp_name = config["logreg_params"]["exp_name"]

    train_dataset = BHSceneDataset(**dataset_args)
    test_dataset = BHSceneDataset(**{**dataset_args, "train_split": False})


    # Load language to ID mapping
    with open('./dataset/language_encode.json') as f:
        lang_map = yaml.safe_load(f)  

    lang_id = lang_map[lang]

    # Extract features and binary labels
    x_train, y_train = load_features(train_dataset, lang_id)
    x_test, y_test = load_features(test_dataset, lang_id)

    # Scale features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Optional PCA
    if pca_flag:
        pca = PCA(n_components=pca_dim)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    # Train Logistic Regression
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n[INFO] Accuracy: {acc * 100:.2f}%")
    print("[INFO] Classification Report:")
    print(report)

    # Save plot
    if save_plots:
        plot_dir = os.path.join("plots", "logreg", exp_name)
        plot_decision_boundary(x_train, y_train, model, lang, plot_dir)


if __name__ == "__main__":
    main()
