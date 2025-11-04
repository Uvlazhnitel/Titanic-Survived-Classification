# src/preprocessing.py
import numpy as np
from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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
