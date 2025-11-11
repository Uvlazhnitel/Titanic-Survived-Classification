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
| LogisticRegression (+balanced) | —                 | —                | OOF AUCs (currently): ROC=0.856, AP=0.829. Run the same CV to fill mean±std. |

**Action item (optional)**: Re-run `cross_validate(..., scoring={"roc_auc","average_precision"})` for the balanced model to fill CV mean ± std.

---

## B) Operating Point (@ Threshold on OOF)

Thresholds are selected on OOF predictions using the same rule as Session 5  
(“precision ≥ target → maximize recall”), unless marked with an asterisk.

| Model                          | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Protocol                                                                 |
|--------------------------------|-------|---------------|------------|--------|--------------------------------------------------------------------------|
| LogisticRegression (baseline)  | 0.635 | 0.846         | 0.623      | 0.717  | OOF, 5-fold                                                             |
| RandomForestClassifier         | 0.640 | 0.844         | 0.656      | 0.738  | OOF, 5-fold                                                             |
| LogisticRegression (+balanced) | 0.635*| 0.849         | 0.619      | 0.716  | OOF, 5-fold; apples-to-apples at the same 0.635 threshold. Own optimized threshold saved (see Artifacts). |

*Note*: `0.635*` for the balanced model is for a direct, apples-to-apples comparison.  
The optimized threshold for the balanced model based on the same rule is saved — see artifacts below.

---

## C) Confusion Matrices (OOF) @ Listed Thresholds

### RandomForest @ 0.640
- **TN**=406, **FP**=33, **FN**=94, **TP**=179  
- **Precision**=0.844, **Recall**=0.656, **F1**=0.738  

### LogisticRegression (baseline) @ 0.635
- **TN**=408, **FP**=31, **FN**=103, **TP**=170  
- **Precision**=0.846, **Recall**=0.623, **F1**=0.717  

### LogisticRegression (+balanced) @ 0.635*
- **TN**=409, **FP**=30, **FN**=104, **TP**=169  
- **Precision**=0.849, **Recall**=0.619, **F1**=0.716  

---

## D) Verdict (Session 7)

- **Ranking**: Logistic baseline leads on AP (0.834 vs 0.821), RF leads on ROC-AUC (0.870 vs 0.856).  
- **Operating point (high-precision regime)**: RF achieves higher recall at essentially the same precision, giving a higher F1.  
- **Class weights**: With the same threshold (0.635), balanced logistic shows negligible change (slightly ↑Precision, slightly ↓Recall). Use the optimized balanced threshold (saved) if you plan to keep weights; otherwise, the baseline logistic remains a solid reference.

---

## E) Validation Protocol

- **Train/valid**: 5× StratifiedKFold (`shuffle=True, random_state=42`); OOF predictions for thresholding.  
- **Test set**: untouched until the final evaluation.  
- **Preprocessing**: identical `ColumnTransformer` for all models.

---

## F) Artifacts

- **Baseline threshold**: `reports/threshold.npy`  
- **RF threshold**: `reports/thresholds_rf.npy`  
- **Balanced-logistic threshold**: `reports/thresholds_bl.npy`  
- **ROC / PR figures**: under `reports/figures/` (per model)

---

## G) Environment

- **Python**: 3.11.14  
- **scikit-learn**: 1.7.2  
- **RANDOM_STATE**: 42