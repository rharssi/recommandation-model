import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from common import (
    CATEGORICAL_FEATURES,
    FORCED_K,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_retailer_features,
    build_retailer_game_matrix,
    load_and_prepare as _common_load_and_prepare,
    load_raw_data,
)
from decisions import FORCED_K_DECISION

warnings.filterwarnings("ignore", module=r"^sklearn")
warnings.filterwarnings("ignore", module=r"^scipy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "data_entrainement.csv")


def load_and_prepare(csv_path=DEFAULT_DATA_PATH, verbose=True):
    return _common_load_and_prepare(csv_path, verbose=verbose)


def compute_elbow_and_silhouette(X, k_range=range(2, 9)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
    return list(k_range), inertias, silhouettes


def fit_kmeans(X, k):
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X)
    return model, labels


def plot_elbow_silhouette(k_values, inertias, silhouettes, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k_values, inertias, "o-", color="steelblue")
    axes[0].set_title("Méthode du coude (KMeans)")
    axes[0].set_xlabel("Nombre de clusters (k)")
    axes[0].set_ylabel("Inertie (SSE)")

    axes[1].plot(k_values, silhouettes, "o-", color="darkorange")
    axes[1].set_title("Score de silhouette selon k")
    axes[1].set_xlabel("Nombre de clusters (k)")
    axes[1].set_ylabel("Score de silhouette")
    best_k = k_values[int(np.argmax(silhouettes))]
    axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.6, label=f"k optimal = {best_k}")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return best_k


def plot_pca_clusters(X, labels, model, out_path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    centroids_2d = pca.transform(model.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=45, alpha=0.8)
    ax.scatter(
        centroids_2d[:, 0], centroids_2d[:, 1],
        c="black", marker="X", s=220, edgecolors="white", linewidths=1.5,
        label="Centroïdes",
    )
    var_exp = pca.explained_variance_ratio_.sum() * 100
    ax.set_title(f"Clusters KMeans (projection PCA - {var_exp:.1f}% variance expliquée)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend1)
    ax.legend(loc="lower right")
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

    k_values, inertias, silhouettes = compute_elbow_and_silhouette(X)
    auto_best_k = plot_elbow_silhouette(
        k_values, inertias, silhouettes,
        os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "elbow_silhouette_kmeans.png"),
    )
    best_k = FORCED_K if FORCED_K is not None else auto_best_k

    model, labels = fit_kmeans(X, best_k)
    sil = silhouette_score(X, labels)

    plot_pca_clusters(X, labels, model, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "clusters_pca_kmeans.png"))

    interpret_clusters(df_retailers, labels)
