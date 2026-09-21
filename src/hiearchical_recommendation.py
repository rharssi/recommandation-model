import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from common import (
    FORCED_K,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    load_and_prepare as _common_load_and_prepare,
)
from decisions import FORCED_K_DECISION

warnings.filterwarnings("ignore", module=r"^sklearn")
warnings.filterwarnings("ignore", module=r"^scipy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "data_entrainement.csv")


def load_and_prepare(csv_path=DEFAULT_DATA_PATH, verbose=True):
    return _common_load_and_prepare(csv_path, verbose=verbose)


def compute_linkage(X, method="ward"):
    return linkage(X, method=method)


def select_k_from_linkage(Z, X, k_range=range(2, 9)):
    silhouettes = []
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        silhouettes.append(silhouette_score(X, labels))
    best_k = list(k_range)[int(np.argmax(silhouettes))]
    return best_k, list(k_range), silhouettes


def fit_hierarchical(X, k, method="ward"):
    model = AgglomerativeClustering(n_clusters=k, linkage=method)
    labels = model.fit_predict(X)
    return model, labels


def plot_dendrogram(Z, out_path, color_threshold=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, color_threshold=color_threshold, no_labels=True)
    ax.set_title("Dendrogramme - Clustering Hiérarchique (Ward)")
    ax.set_xlabel("Points de vente")
    ax.set_ylabel("Distance (Ward)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_silhouette_by_k(k_values, silhouettes, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_values, silhouettes, "o-", color="darkorange")
    best_k = k_values[int(np.argmax(silhouettes))]
    ax.axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"k optimal = {best_k}")
    ax.set_title("Score de silhouette selon k (coupes du dendrogramme)")
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Score de silhouette")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_clusters(X, labels, out_path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=45, alpha=0.85)
    var_exp = pca.explained_variance_ratio_.sum() * 100
    ax.set_title(f"Clusters hiérarchiques (projection PCA - {var_exp:.1f}% variance expliquée)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def interpret_clusters(df_retailers, labels):
    df = df_retailers.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster")[NUMERIC_FEATURES].mean().round(2)
    profile["n_retailers"] = df.groupby("cluster").size()
    return profile


if __name__ == "__main__":
    FORCED_K_DECISION.warn_if_unvalidated()

    df_retailers, pivot, X, preprocessor = load_and_prepare()

    Z = compute_linkage(X)
    plot_dendrogram(Z, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "dendrogram.png"))

    auto_best_k, k_values, silhouettes = select_k_from_linkage(Z, X)
    plot_silhouette_by_k(k_values, silhouettes, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "silhouette_hierarchical.png"))
    best_k = FORCED_K if FORCED_K is not None else auto_best_k

    model, labels = fit_hierarchical(X, best_k)
    sil = silhouette_score(X, labels)

    plot_pca_clusters(X, labels, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "clusters_pca_hierarchical.png"))

    interpret_clusters(df_retailers, labels)
