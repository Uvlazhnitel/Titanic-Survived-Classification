from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from .preprocessing import build_leader_preprocessing

RANDOM_STATE = 42

NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Pclass", "Embarked"]

def build_pipeline() -> Pipeline:

    preprocessing = build_leader_preprocessing(
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        remainder="drop",
    )

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=30,
        min_samples_leaf=21,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", hgb),
        ]
    )

