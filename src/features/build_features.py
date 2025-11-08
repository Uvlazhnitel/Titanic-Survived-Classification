import pandas as pd


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Title"] = (
        df["Name"]
        .str.extract(r",\s*([^\.]+)\.", expand=False)
        .replace(
            {
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
                "Lady": "Rare",
                "Countess": "Rare",
                "Dr": "Rare",
                "Col": "Rare",
                "Major": "Rare",
                "Rev": "Rare",
                "Jonkheer": "Rare",
                "Capt": "Rare",
                "Sir": "Rare",
                "Don": "Rare",
            }
        )
    )
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_cabin_deck(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CabinDeck"] = df["Cabin"].str[0]
    df["CabinDeck"] = df["CabinDeck"].fillna("Unknown")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_title(df)
    df = add_family_features(df)
    df = add_cabin_deck(df)
    return df
