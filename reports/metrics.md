Model Evaluation Metrics
# Model Metrics Report

## Positive Class Definition

- **Positive class**: `survived = 1`
- **Class prevalence (train)**: 38.3% (273 / 712)
- **Baseline precision**: ≈ 0.383 (used to interpret PR-AUC).

---

## Model Performance Summary

| Model                          | Features / Preprocessing                                                                 | CV Scheme                                      | ROC-AUC (CV)       | PR-AUC (CV)       | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Notes                                                                                                                                                                                                 |
|--------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------|--------------------|-------------------|-------|---------------|------------|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| LogisticRegression (baseline)  | Impute(median) → Ratios(FamilySize, FarePerPerson) → Log(Fare) → Scale + OHE(handle_unknown=ignore) + ClusterSimilarity(k=5, γ=0.1) | 5× StratifiedKFold (shuffle=True, random_state=42) | 0.856 ± 0.031      | 0.834 ± 0.029     | 0.635 | 0.846         | 0.623      | 0.717  | Threshold picked on OOF via “precision ≥ 0.85 → max recall”. OOF AUCs: PR-AUC(AP)=0.831, ROC-AUC=0.856.                                                                                              |
| LogisticRegression (+balanced) | Same as baseline, but `class_weight='balanced'`                                          | 5× StratifiedKFold (same split/seed)           | —                  | —                 | 0.635*| 0.849         | 0.619      | 0.716  | OOF AUCs: PR-AUC(AP)=0.829, ROC-AUC=0.856. Numbers shown at the same operating threshold (0.635) for apples-to-apples comparison. If re-optimized, update Thr/metrics accordingly.                   |

**Verdict**: On this dataset (≈62/38 split), class weights did not improve ranking (PR-AUC/AP slightly lower; ROC-AUC identical) and bring no meaningful gain at the operating point. We keep the baseline configuration.

---

## Confusion Matrices @ Thr=0.635 (OOF)

### Baseline

|                | Predicted Negative | Predicted Positive |
|----------------|---------------------|---------------------|
| **Actual Negative** | 408                 | 31                  |
| **Actual Positive** | 103                 | 170                 |

- **TN**: 408, **FP**: 31, **FN**: 103, **TP**: 170
- **Precision**: 0.846, **Recall**: 0.623, **F1**: 0.717

### Balanced (`class_weight='balanced'`)

|                | Predicted Negative | Predicted Positive |
|----------------|---------------------|---------------------|
| **Actual Negative** | 409                 | 30                  |
| **Actual Positive** | 104                 | 169                 |

- **TN**: 409, **FP**: 30, **FN**: 104, **TP**: 169
- **Precision**: 0.849, **Recall**: 0.619, **F1**: 0.716

**Delta (Balanced − Baseline) @ Thr=0.635**:  
- FP −1, FN +1, TP −1, TN +1  
- Precision +0.003, Recall −0.004, F1 −0.001  

---

## Per-Fold Cross-Validation Metrics (Baseline)

| Fold | ROC-AUC | PR-AUC | Accuracy |
|------|---------|--------|----------|
| 1    | 0.8670  | 0.8465 | 0.7902   |
| 2    | 0.8063  | 0.8072 | 0.7832   |
| 3    | 0.8860  | 0.8771 | 0.7958   |
| 4    | 0.8448  | 0.8313 | 0.7465   |
| 5    | 0.8731  | 0.8087 | 0.8239   |
| **Mean** | **0.8555** | **0.8342** | **0.7879** |
| **Std**  | **0.0313** | **0.0291** | **0.0279** |

*If desired, mirror this table for the weighted model; based on OOF AUCs, we do not expect a material change.*

---

## Validation Protocol for Thresholding

- **OOF predictions**: Generated from 5× StratifiedKFold (shuffle=True, random_state=42).
- **Metrics computation**: ROC/PR curves and all @threshold metrics are computed on OOF only.
- **Test set**: Remains untouched until final evaluation.

---

## Artifacts

- **ROC-AUC Curve**: `reports/figures/roc-auc_baseline.png`
- **PR Curve**: `reports/figures/pr_baseline.png`
- **Confusion Matrix**: `reports/figures/confusion_matrix.png`
- **Threshold (baseline)**: `reports/threshold.npy`
---

## Thresholding Decision Log (Baseline)

- **Goal**: Fix a single operating point to convert probabilities → labels.
- **Strategy**: Precision ≥ 0.85, maximize Recall (on OOF).
- **Points meeting Precision ≥ 0.85**: 197.
- **Chosen index / threshold**: 485 → 0.635.
- **Chosen PR point**: Precision = 0.85, Recall = 0.623.
- **Saved to**: `reports/threshold.npy`.

---

## Baseline Comparison @ Thr=0.50 (OOF)

| Metric          | Value          |
|------------------|----------------|
| **Confusion Matrix** | `[[372, 67], [84, 189]]` |
| **Precision**    | 0.7383         |
| **Recall**       | 0.6923         |
| **F1**           | 0.7146         |

---

## Versions / Seed

- **Python**: 3.11.14
- **scikit-learn**: 1.7.2
- **RANDOM_STATE**: 42