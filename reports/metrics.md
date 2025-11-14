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
| HBG Model                      | 0.868 ± 0.030     | 0.835 ± 0.036    | OOF AUCs: ROC=0.868, AP=0.826                                         |
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
| HGB Model                      | 0.798 | 0.852         | 0.612      | 0.711  | OOF, 5-fold; Chosen index: 485; Recomputed on OOF: precision=0.847716, recall=0.611722, f1=0.710638 |
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

### HBC Model @ 0.700
- **TN**=409, **FP**=30, **FN**=106, **TP**=167  
- **Precision**=0.852, **Recall**=0.612, **F1**=0.711  

### HGB Model @ 0.679
- **TN**=407, **FP**=32, **FN**=91, **TP**=182  
- **Precision**=0.850, **Recall**=0.667, **F1**=0.747  

---

## Conclusion

- The **RandomForestClassifier** and **HGB Model** are the top-performing models in terms of F1 score and recall.  
  - RandomForest achieves an F1 score of 0.739 and recall of 0.652 at the selected threshold.  
  - HGB achieves a slightly higher F1 score of 0.747 and recall of 0.667, making it the best choice for recall-sensitive tasks.  
- The **HBC Model** performs slightly worse than both RandomForest and HGB in terms of recall (0.612) and F1 score (0.711), but it is comparable in terms of precision (0.848).  
- The **LogisticRegression (+balanced)** model has the highest precision (0.854) but lower recall (0.619) and F1 score (0.718), making it less suitable for recall-sensitive tasks.  
- Overall, the **HGB Model** is recommended for tasks prioritizing recall, while RandomForest remains a strong alternative.