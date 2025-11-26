from typing import Dict
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

def evaluate_metrics(oof_proba: np.ndarray, chosen_thr: float, y_true: np.ndarray) -> Dict[str, float]:
    """
    Evaluate classification metrics based on out-of-fold (OOF) predictions and a chosen threshold.

    Parameters:
    - oof_proba (np.ndarray): Predicted probabilities for the positive class (shape: [n_samples]).
    - chosen_thr (float): Threshold for converting probabilities into class predictions.
    - y_true (np.ndarray): True labels (shape: [n_samples]).

    Returns:
    - metrics (Dict[str, float]): A dictionary containing the confusion matrix, precision, recall, F1 score,
      PR-AUC (average precision), and ROC-AUC.
    """
    # Generate OOF predictions using the chosen threshold
    oof_pred = (oof_proba >= chosen_thr).astype(int)

    # Compute evaluation metrics
    cm = confusion_matrix(y_true, oof_pred)
    prec_at = precision_score(y_true, oof_pred, zero_division=0)
    rec_at = recall_score(y_true, oof_pred, zero_division=0)
    f1_at = f1_score(y_true, oof_pred, zero_division=0)

    # Compute AUC metrics
    ap_oof = average_precision_score(y_true, oof_proba)  # PR-AUC (AP)
    roc_oof = roc_auc_score(y_true, oof_proba)          # ROC-AUC

    # Print metrics for clarity
    print("Confusion matrix @thr:\n", cm)
    print(f"OOF @thr -> Precision={prec_at:.3f} | Recall={rec_at:.3f} | F1={f1_at:.3f}")
    print(f"OOF AUCs -> PR-AUC(AP)={ap_oof:.3f} | ROC-AUC={roc_oof:.3f}")

    # Return metrics as a dictionary
    return {
        "confusion_matrix": cm.tolist(),  # Convert to list for JSON compatibility
        "precision": prec_at,
        "recall": rec_at,
        "f1_score": f1_at,
        "pr_auc": ap_oof,
        "roc_auc": roc_oof,
    }