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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../data/recognition/")
train_csv = os.path.join(data_path, "train.csv")
test_csv = os.path.join(data_path, "test.csv")
logger = os.path.join(script_dir, "results.txt")

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
    print("\nSelect the language to train and test on:")
    for num, lang in Lmap.items():
        print(f"Enter {num} to train and test on {lang}")
    choice = int(input("\nEnter your choice: "))
    return Lmap.get(choice, None)

logging.basicConfig(filename=logger, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
logging.info("Script started.")

def load_dataset(csv_path, selected_lang):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])
    x, y = [], []
    logging.info(f"Loading dataset from {csv_path} for language: {selected_lang}")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Images"):
        img_path = os.path.join(data_path, row["image_path"].replace("\\", "/"))  
        if not os.path.exists(img_path):
            logging.warning(f"File not found: {img_path}")
            continue

        folder_name = os.path.basename(os.path.dirname(img_path)).lower()
        label = 1 if folder_name == selected_lang else 0

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Error reading image: {img_path}")
            continue

        img = cv2.resize(img, (64, 64))
        hog_features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        x.append(hog_features)
        y.append(label)

    x, y = np.array(x), np.array(y)
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logging.error(f"Only one class found: {unique_classes}. Check dataset.")
        print(f"Error: Only one class found ({unique_classes}). Adjust dataset!")
        exit()

    return x, y

selected_lang = select_lang()
if not selected_lang:
    print("Invalid choice. Exiting program.")
    exit()

x_train, y_train = load_dataset(train_csv, selected_lang)
x_test, y_test = load_dataset(test_csv, selected_lang)

logging.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

logging.info("Script finished.")

plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

def save_plot(x, y, model, language):
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)

    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label="Data Points")
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + language, language])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Decision Boundary for {language}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{language}_{timestamp}.png"
    plot_path = os.path.join(plots_dir, filename)
    plt.savefig(plot_path)
    
    logging.info(f"Saved decision boundary plot for {language} at {plot_path}")
    logging.info("=========================================")   
    print(f"Plot saved at: {plot_path}")
    plt.show()

save_plot(x_train, y_train, model, selected_lang)
