from src.features.build_features import build_features
import pandas as pd


def test_build_features_adds_columns():
    df = pd.DataFrame({
        "Name": ["Smith, Mr. John", "Doe, Mrs. Jane"],
        "SibSp": [0, 1],
        "Parch": [0, 1],
        "Cabin": [None, "C123"],
    })
    out = build_features(df)
    assert "Title" in out.columns
    assert "FamilySize" in out.columns
    assert "IsAlone" in out.columns
    assert "CabinDeck" in out.columns
