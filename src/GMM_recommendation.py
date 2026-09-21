import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from common import (
    NUMERIC_FEATURES,
    RANDOM_STATE,
    load_and_prepare as _common_load_and_prepare,
)

warnings.filterwarnings("ignore", module=r"^sklearn")
warnings.filterwarnings("ignore", module=r"^scipy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "data_entrainement.csv")


def load_and_prepare(csv_path=DEFAULT_DATA_PATH, verbose=True):
    return _common_load_and_prepare(csv_path, verbose=verbose)


def select_n_components(X, n_range=range(2, 9)):
    bics, aics = [], []
    for n in n_range:
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=RANDOM_STATE)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))
    return list(n_range), bics, aics


def fit_gmm(X, n_components):
    model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    probas = model.predict_proba(X)
    return model, labels, probas


def plot_bic_aic(n_values, bics, aics, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, bics, "o-", label="BIC", color="steelblue")
    ax.plot(n_values, aics, "o-", label="AIC", color="darkorange")
    best_n = n_values[int(np.argmin(bics))]
    ax.axvline(best_n, color="red", linestyle="--", alpha=0.6, label=f"n optimal (BIC) = {best_n}")
    ax.set_title("Sélection du nombre de composantes GMM (BIC / AIC)")
    ax.set_xlabel("Nombre de composantes")
    ax.set_ylabel("Score (plus bas = meilleur)")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_n


def _draw_ellipse(ax, mean, cov, color, n_std=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 0))
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
    ax.add_patch(ellipse)


def plot_pca_gmm(X, labels, model, out_path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    components = pca.components_
    means_2d = pca.transform(model.means_)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, model.n_components))

    for k in range(model.n_components):
        cov_full = model.covariances_[k]
        cov_2d = components @ cov_full @ components.T
        _draw_ellipse(ax, means_2d[k], cov_2d, colors[k])

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=45, alpha=0.85)
    ax.scatter(means_2d[:, 0], means_2d[:, 1], c="black", marker="X", s=200,
               edgecolors="white", linewidths=1.5, label="Moyennes des composantes")
    var_exp = pca.explained_variance_ratio_.sum() * 100
    ax.set_title(f"Composantes GMM (projection PCA - {var_exp:.1f}% variance expliquée)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def interpret_clusters(df_retailers, labels, probas):
    df = df_retailers.copy()
    df["cluster"] = labels
    df["confiance_affectation"] = probas.max(axis=1)

    profile = df.groupby("cluster")[NUMERIC_FEATURES].mean().round(2)
    profile["n_retailers"] = df.groupby("cluster").size()
    profile["confiance_moyenne"] = df.groupby("cluster")["confiance_affectation"].mean().round(3)

    return profile


if __name__ == "__main__":
    df_retailers, pivot, X, preprocessor = load_and_prepare()

    n_values, bics, aics = select_n_components(X)
    best_n = plot_bic_aic(n_values, bics, aics, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "bic_aic_gmm.png"))

    model, labels, probas = fit_gmm(X, best_n)
    sil = silhouette_score(X, labels)

    plot_pca_gmm(X, labels, model, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "clusters_pca_gmm.png"))

    interpret_clusters(df_retailers, labels, probas)
