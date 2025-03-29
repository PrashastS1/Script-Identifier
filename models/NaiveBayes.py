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
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE

# Paths
train_csv = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/train.csv"
test_csv = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/test.csv"
data_path = "/Users/prashastasrivastava/Desktop/projects/script-iden/Script-Identifier/data/recognition/"
script_dir = os.path.dirname(os.path.abspath(__file__))
logger = os.path.join(script_dir, "script_log_naive_bayes_hog.txt")

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

# Load data (use full dataset)
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

# Reduce dimensionality with PCA
print("Applying PCA to reduce dimensionality...", flush=True)
pca = PCA(n_components=100)  # Increase to 200 for better feature retention
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print("PCA completed.", flush=True)

# Handle unbalanced dataset with SMOTE
logging.info("Applying SMOTE to handle unbalanced dataset...")
print("Applying SMOTE...", flush=True)
smote = SMOTE(random_state=42, sampling_strategy=0.4)  # Adjust ratio to 1:2.5 (Class 1:Class 0)
x_train, y_train = smote.fit_resample(x_train, y_train)
logging.info(f"After SMOTE, train set size: {len(x_train)}")
print(f"After SMOTE, train set size: {len(x_train)}", flush=True)

# Train Naive Bayes model
print("Training Naive Bayes model...", flush=True)
model = GaussianNB()
model.fit(x_train, y_train)
print("Naive Bayes model training completed.", flush=True)

# # Cross-validation
# print("Performing cross-validation...", flush=True)
# scores = cross_val_score(model, x_train, y_train, cv=5, scoring='f1_macro')
# print(f"Cross-validation F1-macro scores: {scores}", flush=True)
# print(f"Average F1-macro: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})", flush=True)

# Predict with progress bar and threshold optimization
print("Predicting probabilities on test set...", flush=True)
y_pred_proba = []
for i in tqdm(range(len(x_test)), desc="Predicting Probabilities", dynamic_ncols=True, mininterval=0.1):
    proba = model.predict_proba(x_test[i:i+1])[0, 1]
    y_pred_proba.append(proba)
y_pred_proba = np.array(y_pred_proba)
print("Probability prediction completed.", flush=True)

# Precision-Recall Curve to find optimal threshold
print("Generating precision-recall curve...", flush=True)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Avoid division by zero
best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
print(f"Best threshold for maximizing F1-score: {best_threshold:.2f}", flush=True)

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(thresholds, precision[:-1], label="Precision", color="blue")
plt.plot(thresholds, recall[:-1], label="Recall", color="green")
plt.plot(thresholds, f1_scores[:-1], label="F1-score", color="red")
plt.axvline(x=best_threshold, color="black", linestyle="--", label=f"Best Threshold ({best_threshold:.2f})")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title(f"Precision-Recall vs. Threshold for {selected_lang}")
plt.legend()
plt.grid(True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
pr_curve_path = os.path.join(script_dir, "plots", f"{selected_lang}_pr_curve_{timestamp}.png")
os.makedirs(os.path.dirname(pr_curve_path), exist_ok=True)
plt.savefig(pr_curve_path)
# plt.show()
plt.close()  # Close figure to free memory
print(f"Precision-Recall curve saved at: {pr_curve_path}", flush=True)

# Use the best threshold
y_pred = (y_pred_proba >= best_threshold).astype(int)

# Compute accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.2f}%", flush=True)
print("Classification Report:\n", classification_report(y_test, y_pred), flush=True)

# Confusion Matrix
print("Generating confusion matrix...", flush=True)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not " + selected_lang, selected_lang])
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix for {selected_lang} (Naive Bayes + HOG)")
cm_path = os.path.join(script_dir, "plots", f"{selected_lang}_confusion_matrix_{timestamp}.png")
plt.savefig(cm_path)
# plt.show()
plt.close()  # Close figure to free memory
print(f"Confusion matrix saved at: {cm_path}", flush=True)

# Visualization
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def save_decision_boundary_plot(x, y, model, language, best_threshold, plots_dir):
    print("Generating decision boundary plot...", flush=True)
    # Subsample both x and y consistently
    sample_size = min(5000, len(x))  # Use min to avoid index errors if dataset is smaller
    x_sampled = x[:sample_size]
    y_sampled = y[:sample_size]
    
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x_sampled)

    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))  # 50x50 grid

    # Vectorize prediction for efficiency
    grid_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])
    Z_proba = model.predict_proba(grid_points)[:, 1]
    Z = (Z_proba >= best_threshold).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_sampled, cmap=plt.cm.Paired, edgecolors='k', label="Data Points")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + language, language])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundary for {language} (Naive Bayes + HOG)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_naive_bayes_hog_decision_boundary_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    plt.close()  # Close figure to free memory
    print(f"Decision boundary plot saved at: {plot_path}", flush=True)

def save_scatter_plot(x, y, language, plots_dir):
    print("Generating scatter plot...", flush=True)
    # Subsample both x and y consistently
    sample_size = min(5000, len(x))  # Use min to avoid index errors if dataset is smaller
    x_sampled = x[:sample_size]
    y_sampled = y[:sample_size]
    
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x_sampled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y_sampled, cmap=plt.cm.Paired, edgecolors='k', alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + language, language])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Scatter Plot for {language} (PCA Reduced)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_scatter_plot_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    plt.close()  # Close figure to free memory
    print(f"Scatter plot saved at: {plot_path}", flush=True)

# Generate visualizations
save_decision_boundary_plot(x_train, y_train, model, selected_lang, best_threshold, plots_dir)
save_scatter_plot(x_test, y_test, selected_lang, plots_dir)

logging.info("Script finished.")
print("Script finished.", flush=True)