import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)


def load_model(path="models/logreg_pipeline.joblib"):
    return joblib.load(path)


def main():
    model = load_model()

    data_path = Path("data/train.csv")
    if not data_path.exists():
        raise FileNotFoundError("Expected Titanic training data at data/train.csv. See README for download instructions.")

    df = pd.read_csv(data_path)
    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]

    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, proba),
    }

    print(classification_report(y, preds))
    print("Metrics:", metrics)

    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], cm.flatten()):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    RocCurveDisplay.from_predictions(y, proba)
    plt.savefig(figures_dir / "roc_curve.png", dpi=120)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y, proba)
    plt.savefig(figures_dir / "pr_curve.png", dpi=120)
    plt.close()

    # Persist metrics
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
