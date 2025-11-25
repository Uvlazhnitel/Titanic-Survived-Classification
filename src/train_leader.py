import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models import build_pipeline as build_leader_pipeline, RANDOM_STATE

def load_train_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw Titanic data and return X_train, y_train using the same split protocol."""
    df = pd.read_csv(csv_path)

    # Adjust column names if needed
    target_col = "Survived"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, y_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final leader pipeline and save it.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../data/raw/Titanic-Dataset.csv"),
        help="Path to the raw Titanic CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save the trained pipeline.",
    )
    args = parser.parse_args()

    X_train, y_train = load_train_data(args.data_path)

    pipeline = build_leader_pipeline()
    pipeline.fit(X_train, y_train)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    out_path = args.output_dir / "leader_pipeline.joblib"
    joblib.dump(pipeline, out_path)

    print(f"Saved trained leader pipeline to {out_path}")


if __name__ == "__main__":
    main()
