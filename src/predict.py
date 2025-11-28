import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


TARGET_COL = "Survived"  # target column name in training data


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for prediction script."""
    parser = argparse.ArgumentParser(
        description="Apply trained Titanic leader pipeline to new data."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input CSV file with raw features.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output CSV file with predictions.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("../models/leader_pipeline.joblib"),
        help="Path to saved leader pipeline (joblib file).",
    )
    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=None,
        help="Path to numpy file with decision threshold (optional).",
    )

    return parser.parse_args()


def load_input_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data for prediction.

    Returns:
        X: features dataframe (target column removed if present)
        meta: dataframe with columns to keep in output (e.g. PassengerId, optional Survived)
    """
    df = pd.read_csv(csv_path)

    if df.shape[0] == 0:
        raise ValueError("Input CSV has no rows.")

    # Keep meta columns you want to see in the output (if present)
    meta_cols = []
    for col in ["PassengerId", "id"]:
        if col in df.columns:
            meta_cols.append(col)

    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Remove target column if present (we do not use it as feature at prediction time)
    if TARGET_COL in df.columns:
        meta[TARGET_COL] = df[TARGET_COL]
        X = df.drop(columns=[TARGET_COL])
    else:
        X = df

    return X, meta


def main() -> None:
    args = parse_args()

    # 1. Load input data
    # -------------------
    X, meta = load_input_data(args.input)
    print(f"Loaded input data from {args.input} with shape {X.shape}")

    # 2. Load trained pipeline
    # ------------------------
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    pipeline = joblib.load(args.model_path)
    print(f"Loaded model pipeline from {args.model_path}")

    # 3. Predict probabilities for positive class
    # -------------------------------------------
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[:, 1]
    else:
        # For models without predict_proba you may use decision_function
        raise AttributeError("Model does not support predict_proba.")

    # 4. Apply decision threshold (if provided)
    # -----------------------------------------
    if args.threshold_path is not None and args.threshold_path.exists():
        threshold = float(np.load(args.threshold_path))
        print(f"Using decision threshold from file: {threshold:.4f}")
    else:
        threshold = 0.5
        print(f"No threshold file provided. Using default threshold: {threshold:.4f}")

    y_pred = (proba >= threshold).astype(int)

    # 5. Build output dataframe
    # -------------------------
    output_df = meta.copy()

    # Ensure index alignment
    output_df = output_df.reindex(index=X.index)

    output_df["prediction"] = y_pred
    output_df["proba_positive_class"] = proba

    # 6. Save predictions
    # -------------------
    args.output_dir = args.output.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output} with shape {output_df.shape}")


if __name__ == "__main__":
    main()
