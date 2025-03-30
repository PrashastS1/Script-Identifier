import os
import datetime
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# ========================== STEP 1: SETUP FILE PATHS ==========================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../../data/recognition/")
train_csv = os.path.join(data_path, "train.csv")
test_csv = os.path.join(data_path, "test.csv")
logger = os.path.join(script_dir, "results.txt")

# ========================== STEP 2: LANGUAGE SELECTION ==========================
Lmap = {
    1: "assamese", 2: "bengali", 3: "english", 4: "gujarati", 5: "hindi",
    6: "kannada", 7: "malayalam", 8: "marathi", 9: "punjabi", 
    10: "tamil", 11: "telugu", 12: "urdu"
}

def select_lang():
    print("\nSelect the language to train and test on:")
    for num, lang in Lmap.items():
        print(f"Enter {num} to train and test on {lang}")
    choice = int(input("\nEnter your choice: "))
    return Lmap.get(choice, None)

logging.basicConfig(filename=logger, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
logging.info("Script started.")

# ========================== STEP 3: LOAD PRETRAINED CNN MODEL ==========================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_cnn_features(image_path):
    """Extract CNN features from an image using VGG16."""
    if not os.path.exists(image_path):
        logging.warning(f"Missing image: {image_path}, skipping...")
        return np.zeros((7 * 7 * 512,))  # Return dummy features to maintain shape

    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

# ========================== STEP 4: LOAD DATASET WITH SINGLE PROGRESS BAR ==========================
def load_dataset(csv_path, selected_lang):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])
    
    # Convert paths
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(data_path, x.replace("\\", "/")))
    
    # Assign labels
    df["label"] = df["image_path"].apply(lambda x: 1 if os.path.basename(os.path.dirname(x)).lower() == selected_lang else 0)

    # Use ThreadPoolExecutor to speed up feature extraction
    features, labels = [], []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for _, row in df.iterrows():
            futures.append(executor.submit(extract_cnn_features, row["image_path"]))
        
        for future in tqdm(futures, desc="Extracting Features", total=len(futures)):
            features.append(future.result())

    features = np.array(features)
    labels = df["label"].values

    if len(np.unique(labels)) < 2:
        logging.error(f"Only one class found: {np.unique(labels)}. Check dataset.")
        print(f"Error: Only one class found ({np.unique(labels)}). Adjust dataset!")
        exit()

    return features, labels

# ========================== STEP 5: SELECT LANGUAGE AND LOAD DATA ==========================
selected_lang = select_lang()
if not selected_lang:
    print("Invalid choice. Exiting program.")
    exit()

x_train, y_train = load_dataset(train_csv, selected_lang)
x_test, y_test = load_dataset(test_csv, selected_lang)

logging.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")

# ========================== STEP 6: PREPROCESS DATA ==========================
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ========================== STEP 7: TRAIN LOGISTIC REGRESSION ==========================
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.4f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.4f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

logging.info("Script finished.")

# ========================== STEP 8: SAVE DECISION BOUNDARY PLOT ==========================
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
    print(f"Plot saved at: {plot_path}")
    plt.show()

save_plot(x_train, y_train, model, selected_lang)
logging.info("=========================================")
