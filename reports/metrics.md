Metrics

# Model Evaluation Metrics

## Positive Class Definition
- **Positive class**: `survived = 1`
- **Class prevalence (train)**: 38.3% (273 / 712)
- Interpret PR-AUC relative to this baseline precision (≈ 0.383).

---

## Model Performance Summary

| Model                | Features / Preprocessing                                                                 | CV Scheme                                   | ROC-AUC (CV)     | PR-AUC (CV)     | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Notes                                                                                     |
|----------------------|------------------------------------------------------------------------------------------|--------------------------------------------|------------------|-----------------|-------|---------------|------------|--------|-------------------------------------------------------------------------------------------|
| LogisticRegression   | Impute(median) → Ratios(FamilySize, FarePerPerson) → Log(Fare) → Scale + OHE(handle_unknown=ignore) + ClusterSimilarity(k=5, γ=0.1) | 5× StratifiedKFold (shuffle=True, random_state=42) | 0.856 ± 0.031   | 0.834 ± 0.029  | 0.635 | 0.846         | 0.623      | 0.717  | Threshold picked on OOF via precision ≥ 0.85 → max recall. OOF AUCs: PR-AUC(AP)=0.831, ROC-AUC=0.856. |

---

## Confusion Matrix @ Thr=0.635 (OOF)

|               | Predicted Negative | Predicted Positive |
|---------------|---------------------|---------------------|
| **Actual Negative** | 408                 | 31                  |
| **Actual Positive** | 103                 | 170                 |

- **TN**: 408, **FP**: 31, **FN**: 103, **TP**: 170

---

## Per-Fold Cross-Validation Metrics

| Fold | ROC-AUC  | PR-AUC   | Accuracy |
|------|----------|----------|----------|
| 1    | 0.8670   | 0.8465   | 0.7902   |
| 2    | 0.8063   | 0.8072   | 0.7832   |
| 3    | 0.8860   | 0.8771   | 0.7958   |
| 4    | 0.8448   | 0.8313   | 0.7465   |
| 5    | 0.8731   | 0.8087   | 0.8239   |
| **Mean** | 0.8555   | 0.8342   | 0.7879   |
| **Std**  | 0.0313   | 0.0291   | 0.0279   |

---

## Validation Protocol for Thresholding

- **OOF predictions**: Generated from 5× StratifiedKFold (shuffle=True, random_state=42).
- **Metrics computation**: ROC/PR curves and all @threshold metrics are computed on OOF only.
- **Test set**: Remains untouched until final evaluation.

---

## Artifacts

- ROC-AUC Curve: `reports/figures/roc-auc_baseline.png`
- PR Curve: `reports/figures/pr_baseline.png`
- Confusion Matrix: `reports/figures/confusion_matrix.png`
- Threshold: `reports/threshold.npy`

---

## Thresholding Decision Log

- **Goal**: Fix a single operating point to convert probabilities → labels.
- **Strategy**: Precision ≥ 0.85, maximize Recall (on OOF).
- **Points meeting Precision ≥ 0.85**: 197.
- **Chosen index / threshold**: 485 → 0.635.
- **Chosen PR point**: Precision = 0.85, Recall = 0.623.
- **Saved to**: `reports/threshold.npy`.

---

## Baseline Comparison @ Thr=0.50

| Metric      | Value   |
|-------------|---------|
| **Confusion Matrix** | [[372, 67], [84, 189]] |
| **Precision**        | 0.7383  |
| **Recall**           | 0.6923  |
| **F1**               | 0.7146  |

---

## Versions / Seed

- **Python**: 3.11.14
- **scikit-learn**: 1.7.2
- **RANDOM_STATE**: 42
