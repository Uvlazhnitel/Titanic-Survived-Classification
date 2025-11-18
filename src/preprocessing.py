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
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler, OneHotEncoder


# ---------------------------
# Constants
# ---------------------------

# Feature names added by add_ratio function
RATIO_FEATURES = ["FamilySize", "FarePerPerson"]


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

def add_family_features(X):
    """
    Add family-related features for Titanic-style data:
    - is_child: 1 if Age < 18, else 0.
    - family_size: SibSp + Parch + 1.
    - is_alone: 1 if family_size == 1, else 0.

    Expects columns: 'Age', 'SibSp', 'Parch'.
    """
    X = X.copy()

    # Fill NaNs in SibSp/Parch to avoid NaN family_size
    sibsp = X["SibSp"].fillna(0)
    parch = X["Parch"].fillna(0)
    X["family_size"] = sibsp + parch + 1

    # is_alone: 1 if family_size == 1, else 0
    X["is_alone"] = (X["family_size"] == 1).astype(int)

    # is_child: 1 if Age < 18, Age missing -> 0
    age = X["Age"]
    X["is_child"] = ((age < 18) & age.notna()).astype(int)

    return X

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
            feature_names_out=lambda tr, feats: list(feats) + RATIO_FEATURES
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
            feature_names_out=lambda tr, feats: list(feats) + RATIO_FEATURES
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

# ---------------------------
# HGB-native (ordinal categories, no num imputation/scaling)
# ---------------------------

def make_cat_pipeline_ordinal():
    """
    Categorical pipeline for HGB-native:
    - OrdinalEncoder -> each category mapped to an integer.
    - Robust to unseen categories via unknown_value=-1.
    NOTE: No imputation here.
    - If encoded_missing_value=-1 is supported (newer sklearn), missing values (NaN) are encoded as -1.
    - If not supported (older sklearn), missing values (NaN) may be passed through, which could cause issues.
    """
    # If your sklearn supports encoded_missing_value, you can add encoded_missing_value=-1
    try:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,                # unseen categories -> -1
            encoded_missing_value=-1         # treat missing as -1 (if supported by your sklearn)
        )
    except TypeError:
        # Fallback for older sklearn without encoded_missing_value
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    return Pipeline(steps=[
        ("ordinal", enc)
    ])


def build_preprocessing_hgb_native(
    num_cols,
    cat_cols,
    cat_first=True,
):
    """
    HGB-native preprocessing WITHOUT extra family features:
    - Categorical: OrdinalEncoder (via make_cat_pipeline_ordinal).
    - Numeric: passthrough.
    - Returns:
        preproc: ColumnTransformer
        cat_indices: np.ndarray with indices of categorical features
                     in the transformed matrix (for categorical_features=...).
    """
    cat_pipe = make_cat_pipeline_ordinal()

    transformers = []
    if cat_first:
        # [cat | num] in the final matrix
        transformers.append(("cat", cat_pipe, list(cat_cols)))
        transformers.append(("num", "passthrough", list(num_cols)))
        # Categorical features are at positions [0 .. len(cat_cols)-1]
        cat_indices = np.arange(len(cat_cols))
    else:
        # [num | cat] in the final matrix
        transformers.append(("num", "passthrough", list(num_cols)))
        transformers.append(("cat", cat_pipe, list(cat_cols)))
        # Categorical features are at positions [len(num_cols) .. len(num_cols)+len(cat_cols)-1]
        cat_indices = np.arange(len(num_cols), len(num_cols) + len(cat_cols))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preproc, cat_indices