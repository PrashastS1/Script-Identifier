import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import logging


def plot_decision_boundary(x, y, model, lang, save_dir, backbone="NotMentioned"):
    os.makedirs(save_dir, exist_ok=True)

    # Special case: 1D LDA — Plot as histogram or KDE
    if x.shape[1] == 1:
        logging.info(f"[PLOT] Detected 1D LDA, plotting histogram instead.")
        print(f"[INFO] Plotting 1D histogram for {lang} ({backbone})")

        x = x.flatten()
        classes = np.unique(y)
        colors = plt.cm.Paired(np.linspace(0, 1, len(classes)))

        plt.figure(figsize=(8, 4))
        for idx, cls in enumerate(classes):
            plt.hist(x[y == cls], bins=30, alpha=0.6, label=f"Class {cls}", color=colors[idx])

        # Plot decision boundary threshold if available
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            threshold = -model.intercept_[0] / model.coef_[0][0]
            plt.axvline(threshold, color="red", linestyle="--", label="Decision Boundary")
            logging.info(f"[THRESHOLD] Plotted threshold at: {threshold:.2f}")

        plt.title(f"1D LDA Projection ({lang}, {backbone})")
        plt.xlabel("LDA Component 1")
        plt.ylabel("Frequency")
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{lang.lower()}_{backbone.lower()}_lda_1d_hist_{timestamp}.png"
        plot_path = os.path.join(save_dir, filename)

        plt.savefig(plot_path)
        plt.close()

        print(f"[INFO] 1D LDA plot saved at: {plot_path}")
        logging.info(f"[PLOT] 1D LDA saved at: {plot_path}")
        return

    # Standard case: 2D PCA visualization
    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(x)

    x_min, x_max = x_reduced[:, 0].min() - 1, x_reduced[:, 0].max() + 1
    y_min, y_max = x_reduced[:, 1].min() - 1, x_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    z = z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    classes = np.unique(y)
    if len(classes) == 2:
        labels = [f"Not {lang}", lang] if 1 in classes else ["Class 0", "Class 1"]
    else:
        labels = [str(c) for c in classes]

    plt.legend(handles=scatter.legend_elements()[0], labels=labels)
    plt.title(f"Logistic Regression ({backbone}) Decision Boundary ({lang})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{lang.lower()}_{backbone.lower()}_decision_boundary_{timestamp}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    plt.savefig(plot_path)
    plt.close()

    print(f"[INFO] 2D Plot saved at: {plot_path}")
    logging.info(f"[PLOT] 2D decision boundary saved at: {plot_path}")
