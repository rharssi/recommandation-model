import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

import Kmeans_recommendation as kmeans_mod
import GMM_recommendation as gmm_mod
import hiearchical_recommendation as hier_mod
from common import FORCED_K, validate_forced_k_consistency
from decisions import FORCED_GMM_ALIGNMENT_DECISION, FORCED_K_DECISION

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

validate_forced_k_consistency(kmeans_mod.FORCED_K, hier_mod.FORCED_K, FORCED_K)
COMPARISON_K = FORCED_K if FORCED_K is not None else 3


def evaluate_kmeans(X, k=COMPARISON_K):
    t0 = time.perf_counter()
    model, labels = kmeans_mod.fit_kmeans(X, k)
    elapsed = time.perf_counter() - t0
    return labels, elapsed, {"k": k}


def evaluate_gmm(X, n_components=COMPARISON_K):
    t0 = time.perf_counter()
    model, labels, probas = gmm_mod.fit_gmm(X, n_components)
    elapsed = time.perf_counter() - t0
    return labels, elapsed, {"n_components": n_components}


def evaluate_hierarchical(X, k=COMPARISON_K):
    t0 = time.perf_counter()
    model, labels = hier_mod.fit_hierarchical(X, k)
    elapsed = time.perf_counter() - t0
    return labels, elapsed, {"k": k}


def compute_metrics(X, labels):
    if len(set(labels)) < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }


def compute_reference_free_k(X):
    k_values, _, silhouettes_km = kmeans_mod.compute_elbow_and_silhouette(X)
    kmeans_free_k = k_values[int(np.argmax(silhouettes_km))]

    n_values, bics, _ = gmm_mod.select_n_components(X)
    gmm_free_n = n_values[int(np.argmin(bics))]

    Z = hier_mod.compute_linkage(X)
    hier_free_k, _, _ = hier_mod.select_k_from_linkage(Z, X)

    return {
        "KMeans": {"free_optimum": kmeans_free_k, "criterion": "silhouette_max"},
        "GMM": {"free_optimum": gmm_free_n, "criterion": "bic_min"},
        "Hierarchical": {"free_optimum": hier_free_k, "criterion": "silhouette_max"},
    }


def plot_comparison(results_df, out_path):
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "execution_time_sec"]
    titles = [
        "Score de Silhouette (↑ meilleur)",
        "Indice Davies-Bouldin (↓ meilleur)",
        "Indice Calinski-Harabasz (↑ meilleur)",
        "Temps d'exécution en secondes (↓ meilleur)",
    ]
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(results_df["model"], results_df[metric], color=colors)
        ax.set_title(title)
        ax.set_ylabel(metric)
        for i, v in enumerate(results_df[metric]):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"Comparaison à k identique (k={COMPARISON_K} pour les 3 modèles)", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def rank_models(results_df):
    df = results_df.copy()
    df["rank_silhouette"] = df["silhouette"].rank(ascending=False)
    df["rank_davies_bouldin"] = df["davies_bouldin"].rank(ascending=True)
    df["rank_calinski_harabasz"] = df["calinski_harabasz"].rank(ascending=False)
    df["rank_moyen"] = df[["rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz"]].mean(axis=1)
    df = df.sort_values("rank_moyen")
    return df


if __name__ == "__main__":
    FORCED_K_DECISION.warn_if_unvalidated()
    FORCED_GMM_ALIGNMENT_DECISION.warn_if_unvalidated()

    df_retailers, pivot, X, preprocessor = kmeans_mod.load_and_prepare()

    evaluators = {
        "KMeans": evaluate_kmeans,
        "GMM": evaluate_gmm,
        "Hierarchical": evaluate_hierarchical,
    }

    rows = []
    all_labels = {}
    for name, evaluator in evaluators.items():
        labels, elapsed, params = evaluator(X)
        metrics = compute_metrics(X, labels)
        rows.append({
            "model": name,
            "params": json.dumps(params),
            "execution_time_sec": round(elapsed, 4),
            **metrics,
        })
        all_labels[name] = labels

    results_df = pd.DataFrame(rows)

    ranked = rank_models(results_df)
    best_model = ranked.iloc[0]["model"]

    reference_free_k = compute_reference_free_k(X)

    plot_comparison(results_df, os.path.join(SCRIPT_DIR, "..", "outputs", "figures", "comparison_metrics.png"))

    best_params = json.loads(results_df.loc[results_df["model"] == best_model, "params"].iloc[0])
    results_json_path = os.path.join(SCRIPT_DIR, "..", "outputs", "comparison_results.json")
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "comparison_k": COMPARISON_K,
            "best_model": best_model,
            "best_params": best_params,
            "full_results": results_df.to_dict(orient="records"),
            "reference_free_k_not_used_for_selection": reference_free_k,
            "business_decisions": [FORCED_K_DECISION.id, FORCED_GMM_ALIGNMENT_DECISION.id],
        }, f, indent=2, ensure_ascii=False)
