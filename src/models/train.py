from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def build_pipeline():
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Pclass", "Sex", "Embarked"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    data_path = Path("data/train.csv")  # adjust if different
    if not data_path.exists():
        raise FileNotFoundError("Expected Titanic training data at data/train.csv. See README for download instructions.")

    df = load_data(data_path)

    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]

    pipe = build_pipeline()

    # Cross-validation for robust estimates
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    roc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

    print(f"CV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"CV ROC AUC: {roc_scores.mean():.4f} ± {roc_scores.std():.4f}")

    # Final train/holdout split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_train, y_train)

    # Persist model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipe, "models/logreg_pipeline.joblib")
    print("Saved model to models/logreg_pipeline.joblib")


if __name__ == "__main__":
    main()
