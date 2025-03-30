import os
import datetime
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import zipfile

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

# ========================== STEP 3: EXTRACT DATASET IF NOT EXIST ==========================
zip_path = os.path.join(script_dir, "../../recognition.zip")
if not os.path.exists(data_path):  # Check if already extracted
    print("Extracting recognition.zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(script_dir, "../../"))
    print("Extraction completed!")

# ========================== STEP 4: LOAD PRETRAINED CNN MODEL ==========================
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# ========================== STEP 5: BATCH PROCESSING FUNCTION ==========================
def extract_cnn_features_batch(csv_path, selected_lang, batch_size=32):
    df = pd.read_csv(csv_path, skiprows=1, header=None, names=["image_path", "annotation", "script"])
    
    # Fix image paths
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(data_path, x.replace("\\", "/")))
    
    # Filter data for the selected language (Binary Classification)
    df["label"] = df["image_path"].apply(lambda x: 1 if os.path.basename(os.path.dirname(x)).lower() == selected_lang else 0)

    # Data Generator for Batch Processing
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="image_path",
        y_col="label",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="raw",  # Since it's binary classification
        shuffle=False
    )

    # Extract CNN features in batches
    features = model.predict(generator, verbose=1)
    features = features.reshape(features.shape[0], -1)  # Flatten

    return features, np.array(df["label"])

# ========================== STEP 6: SELECT LANGUAGE AND LOAD DATA ==========================
selected_lang = select_lang()
if not selected_lang:
    print("Invalid choice. Exiting program.")
    exit()

# Extract features in batches
x_train, y_train = extract_cnn_features_batch(train_csv, selected_lang)
x_test, y_test = extract_cnn_features_batch(test_csv, selected_lang)

logging.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")

# ========================== STEP 7: PREPROCESS DATA ==========================
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# ========================== STEP 8: TRAIN LOGISTIC REGRESSION ==========================
model = LogisticRegression(solver='liblinear', class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info("Classification Report:\n" + classification_report(y_test, y_pred))

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))

logging.info("Script finished.")

# ========================== STEP 9: SAVE DECISION BOUNDARY PLOT ==========================
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
