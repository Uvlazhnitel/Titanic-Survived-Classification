from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

from .preprocessing import build_leader_preprocessing

RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """
    Build the final leader pipeline:
    - preprocessing: HGB-native with family features
    - model: tuned HistGradientBoostingClassifier with categorical_features
    """

    # build_leader_preprocessing returns (preproc, cat_indices)
    preproc, cat_indices = build_leader_preprocessing()

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=30,
        min_samples_leaf=21,
        random_state=RANDOM_STATE,
        categorical_features=cat_indices,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("model", hgb),
        ]
    )
    return pipeline
