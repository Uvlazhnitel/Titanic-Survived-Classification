from .preprocessing import build_preprocessing
RANDOM_STATE = 42

NUM_COLS = ["Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Pclass", "Embarked"]

def build_pipeline() -> Pipeline:

    preprocessing = build_preprocessing(
        num_cols=NUM_COLS,
        cat_cols=CAT_COLS,
        random_state=RANDOM_STATE
    )
