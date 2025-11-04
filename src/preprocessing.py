# src/preprocessing.py
import numpy as np
from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder


# ---------------------------
# Helper feature functions
# ---------------------------

def log_fare_only(df):
    """
    Apply log1p to Fare (clip at 0 to avoid negatives).
    Expects a pandas DataFrame with 'Fare'.
    """
    out = df.copy()
    out["Fare"] = np.log1p(out["Fare"].clip(lower=0))
    return out


def add_ratio(df):
    """
    Add FamilySize and FarePerPerson.
    Expects columns: 'SibSp', 'Parch', 'Fare'.
    """
    out = df.copy()
    # FamilySize is always >= 1 due to the +1 (representing the passenger themselves),
    # so division by zero is not possible when calculating FarePerPerson
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["FarePerPerson"] = out["Fare"].fillna(0) / out["FamilySize"]
    return out


# ---------------------------
# Custom transformer
# ---------------------------

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Transform X into k features: RBF similarity to KMeans centers.
    Minimal version (no internal scaling).
    """
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto"
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # shape: [n_samples, n_clusters]
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# ---------------------------
# Pipelines
# ---------------------------

def make_cat_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def make_num_pipeline_plain():
    """
    Regular numeric features (no clustering):
      impute -> add ratios -> log(Fare) -> scale
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("ratio", FunctionTransformer(
            add_ratio,
            validate=False,
            feature_names_out=lambda tr, feats: list(feats) + ["FamilySize", "FarePerPerson"]
        )),
        ("log_fare", FunctionTransformer(
            log_fare_only,
            validate=False,
            feature_names_out=lambda tr, feats: list(feats)
        )),
        ("scaler", StandardScaler()),
    ])


def make_cluster_pipeline(n_clusters=5, gamma=0.1, random_state=42):
    """
    Cluster similarity branch:
      impute -> add ratios -> log(Fare) -> scale -> ClusterSimilarity
    Returns k new features: "Cluster 0 similarity", ..., "Cluster k-1 similarity"
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("ratio", FunctionTransformer(
            add_ratio,
            validate=False,
            feature_names_out=lambda tr, feats: list(feats) + ["FamilySize", "FarePerPerson"]
        )),
        ("log_fare", FunctionTransformer(
            log_fare_only,
            validate=False,
            feature_names_out=lambda tr, feats: list(feats)
        )),
        ("scaler", StandardScaler()),
        ("clustering", ClusterSimilarity(
            n_clusters=n_clusters, gamma=gamma, random_state=random_state
        )),
    ])


# ---------------------------
# Combined preprocessing
# ---------------------------

def build_preprocessing(
    num_cols,                  # e.g. ["Age","SibSp","Parch","Fare","Pclass"]
    cat_cols,                  # e.g. ["Sex","Embarked"]
    remainder="drop",
    kcols=("Age","Fare","SibSp","Parch","Pclass"),  # raw columns for cluster branch
    n_clusters=5, gamma=0.1, random_state=42
):
    """
    ColumnTransformer with two parallel numeric branches:
      - "num": regular numeric features
      - "cluster": k RBF similarities to KMeans centers (built from kcols)
      - "cat": categorical OHE
    """
    num_pipe_plain = make_num_pipeline_plain()
    cat_pipe = make_cat_pipeline()
    cluster_pipe = make_cluster_pipeline(
        n_clusters=n_clusters, gamma=gamma, random_state=random_state
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num",     num_pipe_plain, list(num_cols)),
            ("cat",     cat_pipe,       list(cat_cols)),
            ("cluster", cluster_pipe,   list(kcols)),
        ],
        remainder=remainder,
        verbose_feature_names_out=True
    )
    return preproc
