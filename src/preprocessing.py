import numpy as np
from sklearn import set_config
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
# Legacy / experimental code (optional to keep)
# ============================================================

# Here you can leave log_fare_only, add_ratio, ClusterSimilarity, etc.,
# or move them to another module if they are only used in scratch notebooks.
