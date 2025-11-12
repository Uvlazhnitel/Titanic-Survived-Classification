Session 7 — Addendum (Models Comparison incl. Class Weights)
# Metrics Report

## Positive Class

- **Positive class**: survived = 1  
- **Train prevalence**: 38.3% (273 / 712)  
- **Baseline precision**: ≈ 0.383 (context for PR-AUC)

---

## A) Ranking Metrics (CV mean ± std)

Metrics below are computed on the validation folds of 5× StratifiedKFold (`shuffle=True, random_state=42`).  
For models where per-fold CV was not yet re-run (only OOF AUCs available), the CV cells are marked with “—”; OOF AUCs are given in Notes.

| Model                          | ROC-AUC (CV)       | PR-AUC / AP (CV) | Notes                                                                 |
|--------------------------------|--------------------|------------------|-----------------------------------------------------------------------|
| LogisticRegression (baseline)  | 0.856 ± 0.031     | 0.834 ± 0.029    | OOF AUCs: ROC=0.856, AP=0.831                                         |
| RandomForestClassifier         | 0.870 ± 0.016     | 0.821 ± 0.031    | OOF AUCs: ROC=0.871, AP=0.812                                         |
| LogisticRegression (+balanced) | 0.856 ± 0.029     | 0.832 ± 0.028    | OOF AUCs: ROC=0.856, AP=0.832                                         |

**Action item (optional)**: Re-run `cross_validate(..., scoring={"roc_auc","average_precision"})` for the balanced model to fill CV mean ± std.

---

## B) Operating Point (@ Threshold on OOF)

Thresholds are selected on OOF predictions using the same rule as Session 5  
(“precision ≥ target → maximize recall”), unless marked with an asterisk.

| Model                          | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Protocol                                                                 |
|--------------------------------|-------|---------------|------------|--------|--------------------------------------------------------------------------|
| LogisticRegression (baseline)  | 0.636 | 0.850         | 0.623      | 0.719  | OOF, 5-fold; Strategy: precision ≥ 0.85 → maximize recall; Chosen index: 486 |
| RandomForestClassifier         | 0.640 | 0.852         | 0.652      | 0.739  | OOF, 5-fold                                                             |
| LogisticRegression (+balanced) | 0.743 | 0.854         | 0.619      | 0.718  | OOF, 5-fold; Strategy: precision ≥ 0.85 → maximize recall; Chosen index: 488 |

---

## C) Confusion Matrices (OOF) @ Listed Thresholds

### RandomForest @ 0.640
- **TN**=408, **FP**=31, **FN**=95, **TP**=178  
- **Precision**=0.852, **Recall**=0.652, **F1**=0.739  

### LogisticRegression (baseline) @ 0.636
- **TN**=409, **FP**=30, **FN**=103, **TP**=170  
- **Precision**=0.850, **Recall**=0.623, **F1**=0.719  

### LogisticRegression (+balanced) @ 0.743
- **TN**=410, **FP**=29, **FN**=104, **TP**=169  
- **Precision**=0.854, **Recall**=0.619, **F1**=0.718  

---

## D) Verdict

- **Ranking**: Logistic baseline leads on AP (0.834 vs 0.821), RF leads on ROC-AUC (0.870 vs 0.856).  
- **Operating point (high-precision regime)**: RF achieves higher recall at essentially the same precision, giving a higher F1.  
- **Class weights**: With the same threshold (0.636), baseline logistic shows slightly ↑Precision and unchanged Recall. Use the optimized balanced threshold (saved) if you plan to keep weights; otherwise, the baseline logistic remains a solid reference.

---

## E) Validation Protocol

- **Train/valid**: 5× StratifiedKFold (`shuffle=True, random_state=42`); OOF predictions for thresholding.  
- **Test set**: untouched until the final evaluation.  
- **Preprocessing**: identical `ColumnTransformer` for all models.

---

## F) Artifacts

- **Baseline threshold**: `reports/threshold.npy`  
- **RF threshold**: `reports/threshold_rf.npy`  
- **Balanced-logistic threshold**: `reports/threshold_bl.npy`  
- **ROC / PR figures**: under `reports/figures/` (per model)

---

## G) Environment

- **Python**: 3.11.14  
- **scikit-learn**: 1.7.2  
- **RANDOM_STATE**: 42
