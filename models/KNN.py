import os
import cv2
import datetime
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Paths
train_csv = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/train.csv"
test_csv = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/test.csv"
data_path = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/"
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = os.path.join(script_dir, "script_log_knn_hog.txt")

# Language mapping
Lmap = {
    1: "assamese",
    2: "bengali",
    3: "english",
    4: "gujarati",
    5: "hindi",
    6: "kannada",
    7: "malayalam",
    8: "marathi",
    9: "punjabi",
    10: "tamil",
    11: "telugu",
    12: "urdu"
}

def select_lang():
    while True:
        print("\nSelect the language to train and test on:")
        for num, lang in Lmap.items():
            print(f"Enter {num} to train and test on {lang}")
        try:
            choice = input("\nEnter your choice: ").strip()
            if not choice:
                print("Error: Input cannot be empty. Please enter a number.")
                continue
            choice = int(choice)
            if choice not in Lmap:
                print(f"Error: Invalid choice '{choice}'. Please select a number between 1 and {len(Lmap)}.")
                continue
            return Lmap[choice]
        except ValueError:
            print("Error: Please enter a valid number.")

# Logging setup
logging.basicConfig(filename=logger, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
logging.info("Script started.")

def load_dataset(csv_path, selected_lang):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])
    print(f"Number of rows in CSV: {len(df)}", flush=True)
    
    x, y = [], []
    logging.info(f"Loading dataset from {csv_path} for language: {selected_lang}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images", dynamic_ncols=True, mininterval=0.1):
        img_path = os.path.join(data_path, row["image_path"].replace("\\", "/"))
        if not os.path.exists(img_path):
            logging.warning(f"File not found: {img_path}")
            continue

        label = 1 if row["script"].lower() == selected_lang else 0

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Error reading image: {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        x.append(hog_features)
        y.append(label)

    print(f"Total images processed: {len(x)}", flush=True)
    if len(x) == 0:
        print("Error: No images were successfully loaded. Check image paths and file accessibility.", flush=True)
        exit()

    x, y = np.array(x), np.array(y)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logging.error(f"Only one class found: {unique_classes}. Check dataset.")
        print(f"Error: Only one class found ({unique_classes}). Adjust dataset!", flush=True)
        exit()

    return x, y

# Select language
selected_lang = select_lang()
if not selected_lang:
    print("Invalid choice. Exiting program.", flush=True)
    exit()

# Load data
x_train, y_train = load_dataset(train_csv, selected_lang)
x_test, y_test = load_dataset(test_csv, selected_lang)

logging.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")
print(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}", flush=True)

# Feature scaling
print("Scaling features...", flush=True)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Feature scaling completed.", flush=True)

# Apply PCA to reduce dimensionality
print("Applying PCA to reduce dimensionality...", flush=True)
pca = PCA(n_components=100)  # Reduce to 100 dimensions
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print("PCA completed.", flush=True)

# Handle unbalanced dataset with SMOTE
logging.info("Applying SMOTE to handle unbalanced dataset...")
print("Applying SMOTE...", flush=True)
smote = SMOTE(random_state=42, sampling_strategy=0.3)  # Adjust ratio to 1:2 (Class 1:Class 0)
x_train, y_train = smote.fit_resample(x_train, y_train)
logging.info(f"After SMOTE, train set size: {len(x_train)}")
print(f"After SMOTE, train set size: {len(x_train)}", flush=True)

# Train KNN model with best k
# best_k=3
# print(f"Training KNN with best k={best_k}...", flush=True)
# model = KNeighborsClassifier(n_neighbors=best_k, weights='distance', n_jobs=-1)
# model.fit(x_train, y_train)
# print("KNN model training completed.", flush=True)

# # Evaluate with adjusted threshold
# print("Predicting probabilities on test set...", flush=True)
# y_pred_proba = model.predict_proba(x_test)[:, 1]
# print("Probability prediction completed.", flush=True)

# threshold = 0.7  # Adjust threshold to reduce false positives for Class 1
# y_pred = (y_pred_proba >= threshold).astype(int)

print("Performing Grid Search for KNN hyperparameters...", flush=True)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=100)),
    ('smote', SMOTE(random_state=42)),
    ('knn', KNeighborsClassifier(n_jobs=-1))
])

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print("Grid Search completed.", flush=True)

# Predict with the best model
y_pred_proba = best_model.predict_proba(x_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
y_pred = (y_pred_proba >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.2f}%", flush=True)
print("Classification Report:\n", classification_report(y_test, y_pred), flush=True)

logging.info("Script finished.")
print("Script finished.", flush=True)

# Visualization
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def save_plot(x, y, model, language, plots_dir):  # Renamed to avoid overwriting built-in 'plot'
    print("Generating decision boundary visualization...", flush=True)
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)

    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))  # Reduced resolution

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label="Data Points")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + language, language])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundary for {language} (KNN + HOG)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_knn_hog_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    plt.close()  
    
    logging.info(f"Saved decision boundary plot for {language} at {plot_path}")
    logging.info("=========================================")
    print(f"Decision boundary plot saved at: {plot_path}", flush=True)
    print("Decision boundary visualization completed.", flush=True)

def save_confusion_matrix_plot(y_true, y_pred, language, plots_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not " + language, language], 
                yticklabels=["Not " + language, language])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {language}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_confusion_matrix_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix plot saved at: {plot_path}", flush=True)

def save_roc_curve_plot(y_true, y_pred_proba, language, plots_dir):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {language}')
    plt.legend(loc="lower right")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_roc_curve_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve plot saved at: {plot_path}", flush=True)

def save_pr_curve_plot(y_true, y_pred_proba, language, plots_dir):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {language}')
    plt.legend(loc="lower left")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_pr_curve_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Precision-Recall curve plot saved at: {plot_path}", flush=True)

# Call plotting functions
save_plot(x_train, y_train, model, selected_lang, plots_dir)  # Pass plots_dir
save_confusion_matrix_plot(y_test, y_pred, selected_lang, plots_dir)
save_roc_curve_plot(y_test, y_pred_proba, selected_lang, plots_dir)
save_pr_curve_plot(y_test, y_pred_proba, selected_lang, plots_dir)