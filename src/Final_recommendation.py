import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

import Kmeans_recommendation as kmeans_mod
import GMM_recommendation as gmm_mod
import hiearchical_recommendation as hier_mod
from common import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ClusterPredictor,
    TRANSDUCTIVE_MODELS,
    recommend_games_global,
)
from decisions import PROD_INDUCTIVE_DECISION, RECO_METHOD_DECISION

warnings.filterwarnings("ignore", module=r"^sklearn")
warnings.filterwarnings("ignore", module=r"^scipy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "comparison_results.json")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "delivery", "final_recommender_model.joblib")

FALLBACK_MODEL = "KMeans"

get_recommendations = recommend_games_global


def load_best_model_choice():
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["best_model"], data["best_params"]
    warnings.warn(
        f"{COMPARISON_PATH} introuvable. Lancez d'abord Comparaison_models.py pour une "
        f"sélection basée sur les métriques. Utilisation du fallback : {FALLBACK_MODEL}."
    )
    return FALLBACK_MODEL, {}


def train_final_model(model_name, params, X):
    if model_name == "KMeans":
        k = params.get("k")
        if k is None:
            k_values, _, silhouettes = kmeans_mod.compute_elbow_and_silhouette(X)
            k = k_values[int(np.argmax(silhouettes))]
        native_model, labels = kmeans_mod.fit_kmeans(X, k)
    elif model_name == "GMM":
        n_components = params.get("n_components")
        if n_components is None:
            n_values, bics, _ = gmm_mod.select_n_components(X)
            n_components = n_values[int(np.argmin(bics))]
        native_model, labels, _ = gmm_mod.fit_gmm(X, n_components)
    elif model_name == "Hierarchical":
        k = params.get("k")
        if k is None:
            Z = hier_mod.compute_linkage(X)
            k, _, _ = hier_mod.select_k_from_linkage(Z, X)
        native_model, labels = hier_mod.fit_hierarchical(X, k)
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    if model_name in TRANSDUCTIVE_MODELS:
        PROD_INDUCTIVE_DECISION.warn_if_unvalidated()

    predictor = ClusterPredictor(model_name, native_model, X_train=X, labels_train=labels)
    return predictor, native_model, labels


def save_recommender(model_name, predictor, native_model, preprocessor, df_retailers, labels, pivot, path=MODEL_PATH):
    bundle = {
        "model_name": model_name,
        "predictor": predictor,
        "native_model": native_model,
        "preprocessor": preprocessor,
        "df_retailers": df_retailers,
        "labels": labels,
        "pivot": pivot,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)


def load_recommender(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable. Lancez d'abord `python Final_recommendation.py`.")
    return joblib.load(path)


def export_all_recommendations(df_retailers, pivot, top_n=5,
                               out_path=os.path.join(SCRIPT_DIR, "..", "delivery", "recommandations_finales.csv")):
    rows = []
    for retailer_code in df_retailers["RETAILER_CODE"]:
        recs = get_recommendations(retailer_code, df_retailers, pivot, top_n=top_n)
        for rank, (game_code, score) in enumerate(recs.items(), start=1):
            rows.append({
                "RETAILER_CODE": retailer_code,
                "RANG": rank,
                "GAME_CODE_RECOMMANDE": game_code,
                "SCORE_POPULARITE_GLOBALE": score,
            })

    recommendations_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    recommendations_df.to_csv(out_path, index=False)
    return recommendations_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne et sert le modèle final de segmentation + recommandation MDJS.")
    parser.add_argument("--retailer", type=int, default=None,
                        help="RETAILER_CODE pour lequel générer une recommandation de démonstration.")
    parser.add_argument("--top_n", type=int, default=5, help="Nombre de jeux à recommander.")
    args = parser.parse_args()

    RECO_METHOD_DECISION.warn_if_unvalidated()

    df_retailers, pivot, X, preprocessor = kmeans_mod.load_and_prepare()

    model_name, params = load_best_model_choice()

    predictor, native_model, labels = train_final_model(model_name, params, X)

    save_recommender(model_name, predictor, native_model, preprocessor, df_retailers, labels, pivot)

    if args.retailer is not None:
        recs = get_recommendations(args.retailer, df_retailers, pivot, top_n=args.top_n)
        print(recs)

    export_all_recommendations(df_retailers, pivot, top_n=args.top_n)
