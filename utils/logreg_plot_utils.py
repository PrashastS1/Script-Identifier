import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import logging


def plot_decision_boundary(x, y, model, lang, save_dir, backbone="NotMentioned"):
    # If the feature dimensionality is already less than 2 (e.g., from LDA), skip plotting
    if x.shape[1] < 2:
        logging.warning(f"[SKIP PLOT] Cannot plot decision boundary: features have only {x.shape[1]}D (need ≥ 2)")
        print(f"[WARN] Skipping plot for {lang} ({backbone}) — only {x.shape[1]}D features")
        return

    # Reduce features to 2D for visualization using PCA
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)

    # Create meshgrid over reduced feature space
    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # Predict over the meshgrid in original space
    z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    z = z.reshape(xx.shape)

    # Start plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    # Generate dynamic legend
    classes = np.unique(y)
    if len(classes) == 2:
        labels = [f"Not {lang}", lang] if 1 in classes else ["Class 0", "Class 1"]
    else:
        labels = [str(c) for c in classes]

    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.title(f"Logistic Regression ({backbone}) Decision Boundary ({lang})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{lang.lower()}_{backbone.lower()}_decision_boundary_{timestamp}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] Plot saved at: {plot_path}")
    logging.info(f"[PLOT] Decision boundary saved at: {plot_path}")
