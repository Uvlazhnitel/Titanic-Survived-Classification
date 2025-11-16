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
| HGB Model                      | 0.868 ± 0.030     | 0.835 ± 0.036    | OOF AUCs: ROC=0.868, AP=0.826                                         |
| HGB Model (native)             | 0.873 ± 0.019     | 0.854 ± 0.021    | OOF AUCs: ROC=0.872, AP=0.847                                         |

---

## B) Operating Point (@ Threshold on OOF)

Thresholds are selected on OOF predictions using the same rule as Session 5  
(“precision ≥ target → maximize recall”), unless marked with an asterisk.

| Model                          | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Protocol                                                                 |
|--------------------------------|-------|---------------|------------|--------|--------------------------------------------------------------------------|
| LogisticRegression (baseline)  | 0.636 | 0.850         | 0.623      | 0.719  | OOF, 5-fold; Chosen index: 486                                           |
| RandomForestClassifier         | 0.640 | 0.852         | 0.652      | 0.739  | OOF, 5-fold; Chosen index: 221                                           |
| LogisticRegression (+balanced) | 0.743 | 0.854         | 0.619      | 0.718  | OOF, 5-fold; Chosen index: 488                                           |
| HGB Model                      | 0.798 | 0.848         | 0.612      | 0.711  | OOF, 5-fold; Chosen index: 485                                           |
| HGB Model (native)             | 0.679 | 0.850         | 0.667      | 0.747  | OOF, 5-fold; Chosen index: 463                                           |
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

### HGB Model @ 0.798
- **TN**=409, **FP**=30, **FN**=106, **TP**=167  
- **Precision**=0.852, **Recall**=0.612, **F1**=0.711  

### HGB Model (native) @ 0.679
- **TN**=407, **FP**=32, **FN**=91, **TP**=182  
- **Precision**=0.850, **Recall**=0.667, **F1**=0.747  

---

# Conclusion

### LogisticRegression (Baseline)
A stable reference model with strong ranking metrics (PR-AUC ≈ 0.834 CV). At the working point (precision ≈ 0.85), it achieves recall ≈ 0.62 and F1 ≈ 0.72. While it serves as a good baseline, it lags behind tree-based models in recall and F1.

### LogisticRegression (+Balanced)
Adding class weights provides no significant benefit. PR-AUC remains at ≈ 0.832 CV, and recall slightly decreases to ≈ 0.619 at the same precision target. F1 is ≈ 0.718. Decision: class weights are not recommended for this model.

### RandomForestClassifier
A strong contender with ranking metrics of ROC-AUC ≈ 0.870 CV and PR-AUC ≈ 0.821. At precision ≈ 0.85, it achieves recall ≈ 0.65 and F1 ≈ 0.739. A solid no-tune option with better operating metrics than LogisticRegression.

### HistGradientBoosting (OHE Pipeline)
This pipeline achieves PR-AUC ≈ 0.835 CV. At precision ≈ 0.85, recall is ≈ 0.61 and F1 ≈ 0.711. However, it introduces extra complexity without clear performance gains over LogisticRegression.

### HistGradientBoosting (Native: OrdinalEncoder + Categorical Features)
The best overall model with ROC-AUC ≈ 0.873 CV and PR-AUC ≈ 0.854 CV. At precision ≈ 0.85, it achieves recall ≈ 0.667 and F1 ≈ 0.747. Current leader for this task.