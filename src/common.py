import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
FORCED_K = 3


def validate_forced_k_consistency(*k_values):
    unique_values = set(k_values)
    if len(unique_values) > 1:
        raise ValueError(
            f"Incohérence de FORCED_K détectée entre modules : valeurs "
            f"trouvées {unique_values}. FORCED_K doit être importé depuis "
            f"common.py partout, jamais redéfini localement dans un script."
        )


REQUIRED_RAW_COLUMNS = [
    "RETAILER_CODE",
    "RETAILER_NAME",
    "REGION",
    "GAME_CODE",
    "Total Ventes Brutes",
    "Total Ventes Nettes",
    "Total Tickets Vendus",
    "Total Tickets Annulés",
    "Taux_Forclusion_Pct",
    "TREND_SCORE",
    "AVG_GROWTH_RATE",
    "ACTIVITY_RATE_12M",
    "ACTIVITY_REGULARITY",
]


class SchemaError(ValueError):
    pass


# Correspondance entre le schema historique (colonnes ci-dessus, grain
# retailer x jeu) et le schema du nouvel export PostgreSQL data_final.csv
# (colonnes en minuscules, grain agence x jeu — cf. data_final.sql). Permet
# de lire indifferemment l'un ou l'autre fichier sans dupliquer de donnees :
# le renommage se contente de remettre les noms de colonnes attendus, la
# ligne reste au grain d'origine (le groupby par RETAILER_CODE dans
# build_retailer_features() resomme ensuite naturellement les agences vers
# le retailer, exactement comme le fait la vue SQL vue_analytique_ventes).
RAW_COLUMN_ALIASES = {
    "RETAILER_CODE": "retailer_code",
    "RETAILER_NAME": "nom_retailer_canonique",
    "REGION": "region_canonique",
    "GAME_CODE": "game_code",
    "Total Ventes Brutes": "total_ventes_brutes",
    "Total Ventes Nettes": "total_ventes_nettes",
    "Total Tickets Vendus": "total_tickets_vendus",
    "Total Tickets Annulés": "total_tickets_annules",
    "Taux_Forclusion_Pct": "taux_forclusion_pct",
    "TREND_SCORE": "trend_score",
    "AVG_GROWTH_RATE": "avg_growth_rate",
    "ACTIVITY_RATE_12M": "activity_rate_12m",
    "ACTIVITY_REGULARITY": "activity_regularity",
}


def normalize_raw_columns(df):
    """Renomme les colonnes de l'export data_final.csv (minuscules, grain
    agence x jeu) vers les noms historiques attendus par le reste du
    pipeline, si le schema historique n'est pas deja present. N'a aucun
    effet sur un CSV deja au format historique (data_mdjs.csv,
    data_entrainement.csv / data_test.csv d'origine)."""
    rename_map = {
        alias: expected
        for expected, alias in RAW_COLUMN_ALIASES.items()
        if expected not in df.columns and alias in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_raw_schema(df, csv_path=""):
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Colonnes manquantes dans {csv_path or 'le CSV fourni'} : {missing}. "
            f"Colonnes attendues : {REQUIRED_RAW_COLUMNS}. "
            f"Colonnes trouvées : {list(df.columns)}."
        )
    return df


NUMERIC_FEATURES = [
    "total_ventes_nettes",
    "total_ventes_brutes",
    "total_tickets_vendus",
    "total_tickets_annules",
    "nb_jeux",
    "taux_annulation_pct",
    "taux_forclusion_pct",
    "trend_score",
    "avg_growth_rate",
    "activity_rate_12m",
    "activity_regularity",
]

LOG_FEATURES = [
    "total_ventes_nettes",
    "total_ventes_brutes",
    "total_tickets_vendus",
    "total_tickets_annules",
]
SIGNED_LOG_FEATURES = [
    "trend_score",
]
LINEAR_NUMERIC_FEATURES = [
    f for f in NUMERIC_FEATURES if f not in LOG_FEATURES + SIGNED_LOG_FEATURES
]
CATEGORICAL_FEATURES = ["REGION"]

CANCELLATION_RATE_ANOMALY_THRESHOLD_PCT = 100.0


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def _pct_to_float(series):
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .astype(float)
    )


def load_raw_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = normalize_raw_columns(df)
    validate_raw_schema(df, csv_path)
    return df


def build_retailer_features(df_raw, verbose=True):
    df_raw = df_raw.copy()
    df_raw["_forclusion_x_ventes"] = df_raw["Taux_Forclusion_Pct"] * df_raw["Total Ventes Brutes"]
    df_raw["_trend_x_ventes"] = df_raw["TREND_SCORE"].fillna(0) * df_raw["Total Ventes Brutes"]
    df_raw["_growth_x_ventes"] = df_raw["AVG_GROWTH_RATE"].fillna(0) * df_raw["Total Ventes Brutes"]

    agg = df_raw.groupby(["RETAILER_CODE", "REGION"]).agg(
        total_ventes_nettes=("Total Ventes Nettes", "sum"),
        total_ventes_brutes=("Total Ventes Brutes", "sum"),
        total_tickets_vendus=("Total Tickets Vendus", "sum"),
        total_tickets_annules=("Total Tickets Annulés", "sum"),
        nb_jeux=("GAME_CODE", "nunique"),
        _forclusion_weighted_sum=("_forclusion_x_ventes", "sum"),
        _trend_weighted_sum=("_trend_x_ventes", "sum"),
        _growth_weighted_sum=("_growth_x_ventes", "sum"),
        activity_rate_12m=("ACTIVITY_RATE_12M", "first"),
        activity_regularity=("ACTIVITY_REGULARITY", "first"),
        nb_channels=("RETAILER_NAME", "nunique"),
        RETAILER_NAME=("RETAILER_NAME", "first"),
    ).reset_index()

    agg["taux_annulation_pct"] = np.where(
        agg["total_tickets_vendus"] > 0,
        agg["total_tickets_annules"] / agg["total_tickets_vendus"] * 100,
        0.0,
    )
    agg["taux_forclusion_pct"] = np.where(
        agg["total_ventes_brutes"] > 0,
        agg["_forclusion_weighted_sum"] / agg["total_ventes_brutes"],
        0.0,
    )
    agg["trend_score"] = np.where(
        agg["total_ventes_brutes"] > 0,
        agg["_trend_weighted_sum"] / agg["total_ventes_brutes"],
        0.0,
    )
    agg["avg_growth_rate"] = np.where(
        agg["total_ventes_brutes"] > 0,
        agg["_growth_weighted_sum"] / agg["total_ventes_brutes"],
        0.0,
    )
    agg = agg.drop(columns=["_forclusion_weighted_sum", "_trend_weighted_sum", "_growth_weighted_sum"])

    agg["ANOMALIE_ANNULATION_SUP_100PCT"] = agg["taux_annulation_pct"] > CANCELLATION_RATE_ANOMALY_THRESHOLD_PCT
    n_anomalies = int(agg["ANOMALIE_ANNULATION_SUP_100PCT"].sum())
    if n_anomalies > 0 and verbose:
        anomaly_codes = agg.loc[agg["ANOMALIE_ANNULATION_SUP_100PCT"], "RETAILER_CODE"].tolist()
        message = (
            f"{n_anomalies} retailer(s) avec taux_annulation_pct > "
            f"{CANCELLATION_RATE_ANOMALY_THRESHOLD_PCT:.0f}% (tickets annulés > tickets "
            f"vendus sur la période) : RETAILER_CODE {anomaly_codes}. "
            f"Probable annulation de tickets vendus hors période d'extraction, ou "
            f"erreur source -> à vérifier côté ERP MDJS. Valeur conservée telle "
            f"quelle (non clippée) pour ne pas masquer le signal ; voir la colonne "
            f"ANOMALIE_ANNULATION_SUP_100PCT dans df_retailers."
        )
        warnings.warn(message, UserWarning, stacklevel=2)

    return agg


def build_retailer_game_matrix(df_raw):
    return df_raw.pivot_table(
        index="RETAILER_CODE",
        columns="GAME_CODE",
        values="Total Ventes Nettes",
        aggfunc="sum",
        fill_value=0,
    )


def build_preprocessor():
    log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", StandardScaler()),
    ])
    signed_log_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("signed_log1p", FunctionTransformer(_signed_log1p, validate=True)),
        ("scaler", StandardScaler()),
    ])
    linear_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("log_num", log_pipeline, LOG_FEATURES),
        ("signed_log_num", signed_log_pipeline, SIGNED_LOG_FEATURES),
        ("linear_num", linear_pipeline, LINEAR_NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def load_and_prepare(csv_path, verbose=True):
    df_raw = load_raw_data(csv_path)
    df_retailers = build_retailer_features(df_raw, verbose=verbose)
    pivot = build_retailer_game_matrix(df_raw)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df_retailers[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return df_retailers, pivot, X, preprocessor


def _underexploited_mask(retailer_code, pivot, override_row=None):
    """Renvoie un masque booleen (index = GAME_CODE) : True pour les jeux
    "sous-exploites" par ce retailer (ventes <= sa propre mediane de ventes
    positives), les seuls jeux eligibles a une recommandation.

    `override_row` permet de substituer la ligne reellement observee du
    retailer dans `pivot` par une ligne hypothetique (utilise par la
    validation leave-one-out de Comparaison_recommandation.py pour "cacher"
    un jeu deja exploite et verifier si la methode le retrouve).

    Renvoie None si le retailer est totalement inconnu (pas de ligne dans
    pivot et pas d'override) : dans ce cas l'appelant ne filtre pas les
    candidats.
    """
    if override_row is not None:
        row = override_row
    elif retailer_code in pivot.index:
        row = pivot.loc[retailer_code]
    else:
        return None

    positive_sales = row[row > 0]
    median_sales = positive_sales.median() if not positive_sales.empty else 0
    return row <= median_sales


def recommend_games_global(retailer_code, df_retailers, pivot, top_n=5, override_row=None):
    known = retailer_code in df_retailers["RETAILER_CODE"].values or retailer_code in pivot.index
    if not known:
        warnings.warn(
            f"RETAILER_CODE {retailer_code} absent de df_retailers et de pivot "
            f"(aucun historique connu) : recommandation non personnalisée, "
            f"top-{top_n} de popularité globale renvoyé tel quel.",
            UserWarning, stacklevel=2,
        )

    global_popularity = pivot.loc[pivot.index != retailer_code].sum(axis=0).sort_values(ascending=False)

    mask = _underexploited_mask(retailer_code, pivot, override_row)
    candidates = global_popularity[mask] if mask is not None else global_popularity

    return candidates.sort_values(ascending=False).head(top_n)


def build_cluster_totals(pivot, retailer_to_cluster):
    """DataFrame (index=cluster_id, colonnes=GAME_CODE) : somme de
    "Total Ventes Nettes" par jeu, agregee par cluster. Calcule une seule
    fois puis reutilise par recommend_games_intra_cluster() / score_lift_games()
    pour eviter de refaire un groupby a chaque appel (essentiel pour la
    validation leave-one-out qui appelle ces fonctions des milliers de fois).
    """
    cluster_series = pd.Series(retailer_to_cluster).reindex(pivot.index)
    return pivot.groupby(cluster_series).sum()


def recommend_games_intra_cluster(retailer_code, df_retailers, pivot, retailer_to_cluster,
                                   cluster_totals=None, top_n=5, override_row=None):
    """Popularite calculee UNIQUEMENT parmi les retailers du meme cluster
    que `retailer_code` (le retailer lui-meme est exclu de l'agregation),
    plutot que sur l'ensemble de la population (cf. recommend_games_global).
    """
    if cluster_totals is None:
        cluster_totals = build_cluster_totals(pivot, retailer_to_cluster)

    cluster_id = retailer_to_cluster.get(retailer_code)
    if cluster_id is None or cluster_id not in cluster_totals.index:
        warnings.warn(
            f"RETAILER_CODE {retailer_code} : cluster inconnu, repli sur la "
            f"popularité globale (recommend_games_global).",
            UserWarning, stacklevel=2,
        )
        return recommend_games_global(retailer_code, df_retailers, pivot, top_n=top_n, override_row=override_row)

    cluster_popularity = cluster_totals.loc[cluster_id].copy()
    if retailer_code in pivot.index:
        cluster_popularity = cluster_popularity - pivot.loc[retailer_code]
    cluster_popularity = cluster_popularity.sort_values(ascending=False)

    mask = _underexploited_mask(retailer_code, pivot, override_row)
    candidates = cluster_popularity[mask] if mask is not None else cluster_popularity

    return candidates.sort_values(ascending=False).head(top_n)


def score_lift_games(retailer_code, df_retailers, pivot, retailer_to_cluster,
                      cluster_totals=None, top_n=5, override_row=None):
    """Classe les jeux sous-exploites par `retailer_code` selon leur score de
    LIFT : (part du jeu dans les ventes du cluster) / (part du jeu dans les
    ventes globales), le retailer lui-meme etant exclu des deux agregats.
    Un lift > 1 signifie que le jeu est sur-represente dans ce cluster par
    rapport a la population generale ; c'est ce sur-classement, et non un
    volume de ventes brut, qui sert de score de recommandation ici.
    """
    if cluster_totals is None:
        cluster_totals = build_cluster_totals(pivot, retailer_to_cluster)

    cluster_id = retailer_to_cluster.get(retailer_code)
    if cluster_id is None or cluster_id not in cluster_totals.index:
        warnings.warn(
            f"RETAILER_CODE {retailer_code} : cluster inconnu, repli sur la "
            f"popularité globale (recommend_games_global).",
            UserWarning, stacklevel=2,
        )
        return recommend_games_global(retailer_code, df_retailers, pivot, top_n=top_n, override_row=override_row)

    cluster_pop = cluster_totals.loc[cluster_id].copy()
    global_pop = pivot.sum(axis=0).copy()
    if retailer_code in pivot.index:
        own_row = pivot.loc[retailer_code]
        cluster_pop = cluster_pop - own_row
        global_pop = global_pop - own_row

    cluster_total = cluster_pop.sum()
    global_total = global_pop.sum()
    if cluster_total <= 0 or global_total <= 0:
        warnings.warn(
            f"RETAILER_CODE {retailer_code} : cluster ou population globale "
            f"sans ventes une fois le retailer exclu, repli sur la "
            f"popularité globale (recommend_games_global).",
            UserWarning, stacklevel=2,
        )
        return recommend_games_global(retailer_code, df_retailers, pivot, top_n=top_n, override_row=override_row)

    cluster_share = cluster_pop / cluster_total
    global_share = global_pop / global_total
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = (cluster_share / global_share).replace([np.inf, -np.inf], np.nan)

    mask = _underexploited_mask(retailer_code, pivot, override_row)
    candidates = lift[mask] if mask is not None else lift
    candidates = candidates.dropna()

    return candidates.sort_values(ascending=False).head(top_n)


RECO_METHODS = {
    "global": recommend_games_global,
    "intra_cluster": recommend_games_intra_cluster,
    "lift": score_lift_games,
}

# Les methodes "intra_cluster" et "lift" ont besoin d'une affectation de
# cluster par retailer (retailer_to_cluster / cluster_totals) en plus des
# 3 arguments communs (retailer_code, df_retailers, pivot) ; "global" n'en a
# pas besoin. Utilise par Comparaison_recommandation.py et
# Final_recommendation.py pour appeler chaque methode de maniere uniforme.
RECO_METHOD_REQUIRES_CLUSTER = {
    "global": False,
    "intra_cluster": True,
    "lift": True,
}


RISK_MAD_K = 3.0


def robust_risk_threshold(series, k=RISK_MAD_K):
    median = series.median()
    mad = (series - median).abs().median()
    mad_scaled = mad * 1.4826
    if mad_scaled == 0:
        mad_scaled = series.std()
    return median + k * mad_scaled


def identify_risk_cluster(df_retailers, labels, feature="taux_forclusion_pct"):
    df = df_retailers.copy()
    df["cluster"] = labels
    means = df.groupby("cluster")[feature].mean()
    return int(means.idxmax())


INDUCTIVE_MODELS = ("KMeans", "GMM")
TRANSDUCTIVE_MODELS = ("Hierarchical",)


class ClusterPredictor:

    def __init__(self, model_name, native_model, X_train=None, labels_train=None):
        if model_name not in INDUCTIVE_MODELS + TRANSDUCTIVE_MODELS:
            raise ValueError(f"model_name inconnu : {model_name!r}")

        self.model_name = model_name
        self.native_model = native_model
        self.is_surrogate = model_name in TRANSDUCTIVE_MODELS

        if self.is_surrogate:
            if X_train is None or labels_train is None:
                raise ValueError(
                    f"{model_name} est transductif : X_train et labels_train "
                    f"sont requis pour entraîner le classifieur surrogate "
                    f"NearestCentroid utilisé par .predict()."
                )
            self.surrogate = NearestCentroid()
            self.surrogate.fit(X_train, labels_train)
        else:
            self.surrogate = None

    def predict(self, X):
        if self.is_surrogate:
            return self.surrogate.predict(X)
        return self.native_model.predict(X)
