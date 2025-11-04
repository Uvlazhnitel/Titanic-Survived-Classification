# src/preprocessing.py
import numpy as np
from sklearn import set_config
from sklearn.cluster import KMeans
set_config(transform_output="pandas")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Function to log-transform only the 'Fare' column
def log_fare_only(df):
    out = df.copy()
    out["Fare"] = np.log1p(out["Fare"].clip(lower=0))
    return out

# Function to add new ratio features
def add_ratio(df):
    out = df.copy()
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    return out

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    """
    Transforms input X into k features: RBF similarity to k-means centers.
    No internal scaling or additional logic — everything is as simple as possible.
    """
    
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma          # "width" parameter for rbf_kernel
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto"
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # as per sklearn protocol

    def transform(self, X):
        # RBF similarity to centers, shape -> [n_samples, n_clusters]
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

# Numerical pipeline
def make_num_pipeline():
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
        ("clustering", ClusterSimilarity(n_clusters=5, gamma=0.1, random_state=42)),  # Add clustering
    ])


# Categorical pipeline
def make_cat_pipeline():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

# Combined preprocessing pipeline
def build_preprocessing(num_cols, cat_cols, remainder="drop"):
    num_pipe = make_num_pipeline()
    cat_pipe = make_cat_pipeline()
    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder=remainder,
        verbose_feature_names_out=True
    )
    return preproc
