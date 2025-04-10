import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_decision_boundary(X, y, model, lang, save_dir, backbone="unknown"):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.legend(handles=scatter.legend_elements()[0], labels=["Not " + lang, lang])
    plt.title(f"LogReg ({backbone}) Decision Boundary ({lang})")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"{lang}_{backbone}_decision_boundary_{timestamp}.png"
    plot_path = os.path.join(save_dir, plot_filename)

    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Plot saved at: {plot_path}")
