import numpy as np
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

set_config(transform_output="pandas")

# ============================================================
# Family-related feature engineering (used by leader pipeline)
# ============================================================

def add_family_features(X):
    """
    Add family-related features for Titanic-style data:
    - is_child: 1 if Age < 18, else 0 (missing Age -> 0).
    - family_size: SibSp + Parch + 1.
    - is_alone: 1 if family_size == 1, else 0.

    Expects columns: 'Age', 'SibSp', 'Parch'.
    """
    X = X.copy()

    sibsp = X["SibSp"].fillna(0)
    parch = X["Parch"].fillna(0)
    X["family_size"] = sibsp + parch + 1
    X["is_alone"] = (X["family_size"] == 1).astype(int)

    age = X["Age"]
    X["is_child"] = ((age < 18) & age.notna()).astype(int)

    return X


def make_cat_pipeline_ordinal():
    """
    Categorical pipeline for HGB-native:
    - OrdinalEncoder -> each category mapped to an integer.
    - unknown categories and missing values mapped to -1.
    """
    try:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
    except TypeError:
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

    return Pipeline(
        steps=[
            ("ordinal", enc),
        ]
    )


def build_preprocessing_hgb_native_with_family(
    num_cols,
    cat_cols,
    cat_first: bool = True,
):
    """
    HGB-native preprocessing WITH extra family features:
    - Step 1: add_family_features (family_size, is_alone, is_child) to the raw DataFrame.
    - Step 2: ColumnTransformer:
        * categorical: OrdinalEncoder
        * numeric: passthrough for num_cols + new family features

    Returns:
        preproc: Pipeline([("family", ...), ("ct", ColumnTransformer(...))])
        cat_indices: indices of categorical features in the final matrix.
    """
    family_transformer = FunctionTransformer(
        add_family_features,
        validate=False,
    )

    cat_pipe = make_cat_pipeline_ordinal()

    extended_num_cols = list(num_cols) + ["family_size", "is_alone", "is_child"]

    transformers = []
    if cat_first:
        transformers.append(("cat", cat_pipe, list(cat_cols)))
        transformers.append(("num", "passthrough", extended_num_cols))
        cat_indices = np.arange(len(cat_cols))
    else:
        transformers.append(("num", "passthrough", extended_num_cols))
        transformers.append(("cat", cat_pipe, list(cat_cols)))
        cat_indices = np.arange(
            len(extended_num_cols),
            len(extended_num_cols) + len(cat_cols),
        )

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    preproc = Pipeline(
        steps=[
            ("family", family_transformer),
            ("ct", ct),
        ]
    )

    return preproc, cat_indices


# Default columns for your Titanic leader
DEFAULT_NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
DEFAULT_CAT_COLS = ["Sex", "Pclass", "Embarked"]


def build_leader_preprocessing(
    num_cols=None,
    cat_cols=None,
    cat_first: bool = True,
):
    """
    Convenience wrapper for the final leader preprocessing.
    """
    if num_cols is None:
        num_cols = DEFAULT_NUM_COLS
    if cat_cols is None:
        cat_cols = DEFAULT_CAT_COLS

    return build_preprocessing_hgb_native_with_family(
        num_cols=num_cols,
        cat_cols=cat_cols,
        cat_first=cat_first,
    )

# ---------------------------
# HGB-native (no family)
# ---------------------------

def build_preprocessing_hgb_native(
    num_cols,
    cat_cols,
    cat_first=True,
):
    """
    HGB-native preprocessing WITHOUT family-related features.
    Numeric: passthrough.
    Categorical: OrdinalEncoder.
    Returns:
        preproc: ColumnTransformer
        cat_indices: np.ndarray with indices of categorical features.
    """
    cat_pipe = make_cat_pipeline_ordinal()

    transformers = []
    if cat_first:
        transformers.append(("cat", cat_pipe, list(cat_cols)))
        transformers.append(("num", "passthrough", list(num_cols)))
        cat_indices = np.arange(len(cat_cols))
    else:
        transformers.append(("num", "passthrough", list(num_cols)))
        transformers.append(("cat", cat_pipe, list(cat_cols)))
        cat_indices = np.arange(len(num_cols), len(num_cols) + len(cat_cols))

    preproc = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preproc, cat_indices


# ============================================================
# (Optional) Baseline preprocessing for other models
# ============================================================

def build_baseline_preprocessing(num_cols, cat_cols, remainder="drop"):
    """
    Simple baseline preprocessing:
    - numeric: median imputation + StandardScaler.
    - categorical: most-frequent imputation + OneHotEncoder.
    Used for LogisticRegression / RandomForest / HistGB with OHE.
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
        ],
        remainder=remainder,
        verbose_feature_names_out=True,
    )
    return preproc

# ============================================================
# Legacy / experimental preprocessing (ratio + clusters)
# Used in some baseline notebooks (LogReg / RF / HGB with OHE).
# ============================================================

# Feature names added by add_ratio function
RATIO_FEATURES = ["FamilySize", "FarePerPerson"]


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
            n_init="auto",
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # shape: [n_samples, n_clusters]
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def make_cat_pipeline():
    """
    Categorical pipeline with OneHotEncoder for experimental models.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def make_num_pipeline_plain():
    """
    Regular numeric features (no clustering) for experimental models:
      impute -> add ratios -> log(Fare) -> scale
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ratio", FunctionTransformer(
                add_ratio,
                validate=False,
                feature_names_out=lambda tr, feats: list(feats) + RATIO_FEATURES,
            )),
            ("log_fare", FunctionTransformer(
                log_fare_only,
                validate=False,
                feature_names_out=lambda tr, feats: list(feats),
            )),
            ("scaler", StandardScaler()),
        ]
    )


def make_cluster_pipeline(n_clusters=5, gamma=0.1, random_state=42):
    """
    Cluster similarity branch:
      impute -> add ratios -> log(Fare) -> scale -> ClusterSimilarity
    Returns k new features: "Cluster 0 similarity", ..., "Cluster k-1 similarity"
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("ratio", FunctionTransformer(
                add_ratio,
                validate=False,
                feature_names_out=lambda tr, feats: list(feats) + RATIO_FEATURES,
            )),
            ("log_fare", FunctionTransformer(
                log_fare_only,
                validate=False,
                feature_names_out=lambda tr, feats: list(feats),
            )),
            ("scaler", StandardScaler()),
            ("clustering", ClusterSimilarity(
                n_clusters=n_clusters,
                gamma=gamma,
                random_state=random_state,
            )),
        ]
    )


def build_preprocessing(
    num_cols,
    cat_cols,
    remainder="drop",
    kcols=("Age", "Fare", "SibSp", "Parch", "Pclass"),
    n_clusters=5,
    gamma=0.1,
    random_state=42,
):
    """
    ColumnTransformer with three branches for experimental models:
      - "num": regular numeric features (impute + ratio + log + scale)
      - "cluster": k RBF similarities to KMeans centers (built from kcols)
      - "cat": categorical OneHotEncoder
    """
    num_pipe_plain = make_num_pipeline_plain()
    cat_pipe = make_cat_pipeline()
    cluster_pipe = make_cluster_pipeline(
        n_clusters=n_clusters,
        gamma=gamma,
        random_state=random_state,
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe_plain, list(num_cols)),
            ("cat", cat_pipe, list(cat_cols)),
            ("cluster", cluster_pipe, list(kcols)),
        ],
        remainder=remainder,
        verbose_feature_names_out=True,
    )
    return preproc