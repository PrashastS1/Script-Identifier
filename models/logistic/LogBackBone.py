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

# python -m models.Logistic.LogBackBone

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

        # Assigning 1-0 labeling for binary classification
        
        if y == target_language_id:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(features), np.array(labels)


def main():

    # Load config from yaml file
    with open("conifg/logreg.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Assigning configuration parameters
    lang = config["target"]["language"]
    dataset_args = config["dataset"]
    pca_flag = config["logreg_params"]["use_pca"]
    pca_dim = config["logreg_params"]["pca_components"]
    save_plots = config["logreg_params"]["save_plots"]
    exp_name = config["logreg_params"]["exp_name"]

    # Using BHSceneDataset class to load the dataset
    train_dataset = BHSceneDataset(**dataset_args)
    test_dataset = BHSceneDataset(**{**dataset_args, "train_split": False})


    # Loading language map from our json file ddefining mapping languages to ids
    with open('./dataset/language_encode.json') as f:
        lang_map = yaml.safe_load(f)  

    lang_id = lang_map[lang]

    # Extracting features and binary labels
    x_train, y_train = load_features(train_dataset, lang_id)
    x_test, y_test = load_features(test_dataset, lang_id)

    # Scaling features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Optional PCA, specify in yaml 
    if pca_flag:
        pca = PCA(n_components=pca_dim)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

    # Train Logistic Regression
    model = LogisticRegression(solver='liblinear', class_weight='balanced')
    model.fit(x_train, y_train)

    # Evaluate model
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Metrics
    print(f"\n[INFO] Accuracy: {acc * 100:.2f}%")
    print("[INFO] Classification Report:")
    print(report)

    # Save decision boundary plots
    if save_plots:
        plot_dir = os.path.join("plots", "logreg", exp_name)
        plot_decision_boundary(x_train, y_train, model, lang, plot_dir)

if __name__ == "__main__":
    main()
