import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class BusinessDecision:

    id: str
    decision: str
    statistical_optimum: str
    rationale: str
    evidence: str
    validated_by: Optional[str] = None
    validation_date: Optional[str] = None
    validation_ticket: Optional[str] = None

    def is_validated(self) -> bool:
        return self.validated_by is not None

    def warn_if_unvalidated(self):
        if not self.is_validated():
            message = (
                f"[DECISION NON VALIDEE METIER] {self.id} : {self.decision}\n"
                f"    Optimum statistique écarté : {self.statistical_optimum}\n"
                f"    Justification proposée     : {self.rationale}\n"
                f"    -> A faire valider par un référent métier MDJS "
                f"(renseigner validated_by / validation_date / "
                f"validation_ticket dans src/decisions.py)."
            )
            warnings.warn(message, UserWarning, stacklevel=2)


FORCED_K_DECISION = BusinessDecision(
    id="CLUSTERING-K-001",
    decision=(
        "Fixer k=3 pour K-Means et le Clustering Hiérarchique, au lieu de "
        "l'optimum statistique k=2 (score de silhouette maximal)."
    ),
    statistical_optimum="k=2 (silhouette maximale, cf. elbow_silhouette_kmeans.png)",
    rationale=(
        "A k=2, le micro-segment de retailers à très fort risque commercial "
        "(taux de forclusion et d'annulation très élevés) n'est pas isolé : "
        "il est dilué dans un cluster plus large et invisible pour une "
        "action commerciale ciblée. A k=3, ce segment (~15 retailers) "
        "émerge distinctement (cf. interpret_clusters())."
    ),
    evidence=(
        "outputs/figures/elbow_silhouette_kmeans.png ; profil détaillé des "
        "clusters affiché par interpret_clusters() dans Kmeans_recommendation.py."
    ),
    validated_by=None,
    validation_date=None,
    validation_ticket=None,
)

RECO_METHOD_DECISION = BusinessDecision(
    id="RECO-METHOD-001",
    decision=(
        "Utiliser la popularité GLOBALE (tous retailers confondus) comme "
        "SEULE méthode de recommandation, fixée directement dans le code de "
        "production (common.recommend_games_global()). Aucun mécanisme de "
        "sélection à l'exécution."
    ),
    statistical_optimum=(
        "N/A — popularité globale et popularité intra-cluster sont "
        "statistiquement équivalentes en MRR (0,662 vs 0,662, IDENTIQUES "
        "sur les 1364 retailers testés -- Wilcoxon : aucune différence à "
        "tester) ; ce n'est pas un optimum écarté mais un choix de "
        "simplicité à performance mesurée strictement égale."
    ),
    rationale=(
        "Analyse comparative menée hors du pipeline de production : "
        "validation leave-one-out sur 1364/1436 retailers (>= 2 jeux à ventes "
        "positives), MRR calculé pour 3 méthodes, comparé via IC bootstrap "
        "95% + test de Wilcoxon apparié. Résultat : popularite_globale "
        "MRR=0,662 [0,645, 0,679] ; popularite_intra_cluster MRR=0,662 "
        "[0,645, 0,679] (scores identiques retailer par retailer, pas "
        "seulement proches en moyenne) ; score_lift MRR=0,477 [0,464, "
        "0,491], significativement pire (p<0,0001 vs les 2 autres). Le "
        "catalogue de jeux est extrêmement concentré (93,5% du volume "
        "total sur un seul GAME_CODE) : quel que soit le sous-groupe de "
        "retailers sur lequel on agrège les ventes (un cluster ou la "
        "population entière), le classement des jeux par popularité reste "
        "quasi inchangé -- d'où l'égalité stricte entre popularité globale et "
        "intra-cluster. Le score de lift, qui cherche spécifiquement des jeux "
        "sur-représentés DANS un cluster, fait au contraire remonter des jeux "
        "de niche peu vendus globalement -- rarement le jeu réellement caché "
        "lors du test, d'où sa moins bonne performance. À performance égale "
        "entre les 2 meilleures méthodes, la popularité globale est retenue "
        "et fixée dans le code : strictement plus simple (aucune dépendance à "
        "l'affectation de cluster, donc aucun risque de cluster à un seul "
        "membre sans pairs, aucun besoin de ClusterPredictor pour classer un "
        "nouveau retailer avant de pouvoir lui recommander quoi que ce soit)."
    ),
    evidence=(
        "common.py, fonction recommend_games_global() (implémentation "
        "retenue en production) ; Final_recommendation.py (get_recommendations "
        "= recommend_games_global)."
    ),
    validated_by=None,
    validation_date=None,
    validation_ticket=None,
)

PROD_INDUCTIVE_DECISION = BusinessDecision(
    id="PROD-INDUCTIVE-001",
    decision=(
        "Utiliser un classifieur NearestCentroid comme substitut inductif "
        "pour affecter un NOUVEAU retailer à un cluster quand le modèle "
        "retenu par Comparaison_models.py est le Clustering Hiérarchique "
        "(AgglomerativeClustering), au lieu de restreindre la sélection "
        "de modèle de production aux seuls modèles nativement inductifs "
        "(KMeans, GMM)."
    ),
    statistical_optimum=(
        "N/A — il ne s'agit pas d'un optimum statistique écarté mais d'une "
        "limitation structurelle : AgglomerativeClustering n'a pas de "
        "méthode .predict() (algorithme transductif)."
    ),
    rationale=(
        "Sans ce correctif, si Comparaison_models.py désigne 'Hierarchical' "
        "comme meilleur modèle, Final_recommendation.py sauvegardait un "
        "pipeline incapable de classer un nouveau retailer en production "
        "-- un échec silencieux découvert seulement au premier retailer "
        "réellement nouveau. Le NearestCentroid entraîné sur (X, labels) "
        "du clustering hiérarchique final permet de conserver ce modèle "
        "éligible à la sélection sans en restreindre artificiellement le "
        "choix. Limite explicite : pour un nouveau retailer, l'affectation "
        "est une approximation par centroïde le plus proche, pas un "
        "recalcul de la hiérarchie Ward incluant ce point (impossible "
        "sans ré-entraînement complet)."
    ),
    evidence=(
        "common.py, classe ClusterPredictor. Utilisée par "
        "Final_recommendation.py::train_final_model() pour tout modèle "
        "'Hierarchical' sélectionné."
    ),
    validated_by=None,
    validation_date=None,
    validation_ticket=None,
)

FORCED_GMM_ALIGNMENT_DECISION = BusinessDecision(
    id="CLUSTERING-K-002",
    decision=(
        "Dans la comparaison inter-modèles (Comparaison_models.py), aligner "
        "GMM sur n_components=3 au lieu de son optimum BIC propre (n=8), "
        "pour comparer les 3 algorithmes à k identique."
    ),
    statistical_optimum="n_components=8 (BIC minimal, cf. bic_aic_gmm.png)",
    rationale=(
        "Comparer un k=3 imposé (K-Means, Hiérarchique) à un n=8 laissé "
        "libre (GMM) n'est pas une comparaison à conditions égales : le "
        "modèle 'gagnant' pourrait simplement être celui qu'on a le plus "
        "contraint. La sélection du meilleur algorithme se fait donc à k "
        "identique pour les trois. L'optimum statistique propre à GMM "
        "(n=8) reste calculé et publié séparément, à titre de référence "
        "uniquement — jamais utilisé pour la décision finale."
    ),
    evidence=(
        "outputs/figures/bic_aic_gmm.png ; outputs/comparison_results.json "
        "section 'reference_free_k_not_used_for_selection'."
    ),
    validated_by=None,
    validation_date=None,
    validation_ticket=None,
)
