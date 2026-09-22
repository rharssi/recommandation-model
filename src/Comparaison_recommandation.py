"""Compare les 3 methodes de recommandation (popularite globale, popularite
intra-cluster, score de lift) par validation leave-one-out.

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import RECO_METHODS, RECO_METHOD_REQUIRES_CLUSTER, build_cluster_totals
from Kmeans_recommendation import load_and_prepare
from Final_recommendation import load_best_model_choice, train_final_model

warnings.filterwarnings("ignore", module=r"^sklearn")
warnings.filterwarnings("ignore", module=r"^scipy")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "..", "outputs", "comparison_recommandation_results.json")

N_BOOTSTRAP = 2000
RANDOM_STATE_BOOTSTRAP = 42


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE_BOOTSTRAP):
    values = np.asarray(values)
    rng = np.random.RandomState(seed)
    n = len(values)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    df_retailers, pivot, X, _ = load_and_prepare()
    n_total_retailers = len(pivot.index)

    model_name, params = load_best_model_choice()
    predictor, native_model, labels = train_final_model(model_name, params, X)
    retailer_to_cluster = dict(zip(df_retailers["RETAILER_CODE"], labels))
    cluster_totals = build_cluster_totals(pivot, retailer_to_cluster)

    n_games = pivot.shape[1]

    eligible_retailers = [r for r in pivot.index if (pivot.loc[r] > 0).sum() >= 2]

    reciprocal_ranks = {method: [] for method in RECO_METHODS}

    for retailer_code in eligible_retailers:
        row = pivot.loc[retailer_code]
        positive_games = row[row > 0]
        held_out_game = positive_games.idxmax()

        override_row = row.copy()
        override_row[held_out_game] = 0

        for method_name, method_fn in RECO_METHODS.items():
            kwargs = dict(
                retailer_code=retailer_code,
                df_retailers=df_retailers,
                pivot=pivot,
                top_n=n_games,
                override_row=override_row,
            )
            if RECO_METHOD_REQUIRES_CLUSTER[method_name]:
                kwargs["retailer_to_cluster"] = retailer_to_cluster
                kwargs["cluster_totals"] = cluster_totals

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ranked = method_fn(**kwargs)

            ranked_games = list(ranked.index)
            if held_out_game in ranked_games:
                rank_position = ranked_games.index(held_out_game) + 1
                rr = 1.0 / rank_position
            else:
                rr = 0.0
            reciprocal_ranks[method_name].append(rr)

    results = {}
    for method_name, rr_list in reciprocal_ranks.items():
        arr = np.asarray(rr_list)
        lo, hi = bootstrap_ci(arr)
        results[method_name] = {
            "mrr": float(arr.mean()),
            "ci95": [lo, hi],
            "n": len(arr),
        }

    wilcoxon_tests = {}
    method_names = list(RECO_METHODS.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            a, b = method_names[i], method_names[j]
            arr_a = np.asarray(reciprocal_ranks[a])
            arr_b = np.asarray(reciprocal_ranks[b])
            diff = arr_a - arr_b
            if np.allclose(diff, 0):
                wilcoxon_tests[f"{a}_vs_{b}"] = {"statistic": None, "p_value": None, "note": "scores identiques (diff nulle partout)"}
            else:
                stat, p = wilcoxon(arr_a, arr_b)
                wilcoxon_tests[f"{a}_vs_{b}"] = {"statistic": float(stat), "p_value": float(p)}

    global_popularity = pivot.sum(axis=0).sort_values(ascending=False)
    top_game_share_pct = float(global_popularity.iloc[0] / global_popularity.sum() * 100)

    summary = {
        "data_path": "data/data_entrainement.csv",
        "model_used_for_clusters": model_name,
        "n_total_retailers": n_total_retailers,
        "n_eligible_retailers": len(eligible_retailers),
        "n_games": int(n_games),
        "top_game_share_pct_of_net_sales": top_game_share_pct,
        "mrr_by_method": results,
        "wilcoxon_paired_tests": wilcoxon_tests,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Retailers eligibles (>=2 jeux a ventes positives) : "
          f"{len(eligible_retailers)}/{n_total_retailers}")
    print(f"Concentration : le jeu le plus vendu represente "
          f"{top_game_share_pct:.2f}% du volume net total ({n_games} jeux au total)")
    print(f"Modele utilise pour l'affectation en cluster : {model_name}")
    print()
    for method_name, r in results.items():
        print(f"{method_name:15s} MRR = {r['mrr']:.3f}  IC95%=[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  (n={r['n']})")
    print()
    for pair, test in wilcoxon_tests.items():
        print(f"Wilcoxon {pair} : {test}")

    print(f"\nResultats ecrits dans {OUT_PATH}")


if __name__ == "__main__":
    main()
