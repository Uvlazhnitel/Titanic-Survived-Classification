- **Positive class**: survived = 1  
- **Train prevalence**: 38.3% (273 / 712)  
- **Baseline precision**: ≈ 0.383 (context for PR-AUC)

---

## A) Ranking Metrics (CV mean ± std)

Metrics below are computed on the validation folds of 5× StratifiedKFold (`shuffle=True, random_state=42`).  
For models where per-fold CV was not yet re-run (only OOF AUCs available), the CV cells are marked with “—”; OOF AUCs are given in Notes.

| Model                          | ROC-AUC (CV)       | PR-AUC / AP (CV) | Notes                                                                 |
|--------------------------------|--------------------|------------------|-----------------------------------------------------------------------|
| LogisticRegression (baseline)  | 0.856 ± 0.031      | 0.834 ± 0.029    | OOF AUCs: ROC=0.856, AP=0.831                                         |
| RandomForestClassifier         | 0.870 ± 0.016      | 0.821 ± 0.031    | OOF AUCs: ROC=0.871, AP=0.812                                         |
| LogisticRegression (+balanced) | 0.856 ± 0.029      | 0.832 ± 0.028    | OOF AUCs: ROC=0.856, AP=0.832                                         |
| HGB Model                      | 0.868 ± 0.030      | 0.835 ± 0.036    | OOF AUCs: ROC=0.868, AP=0.826                                         |
| HGB Model (native)             | 0.873 ± 0.019      | 0.854 ± 0.021    | OOF AUCs: ROC=0.872, AP=0.847; tuned via RandomizedSearchCV + GridSearchCV |

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

## D) Hyperparameter tuning (HistGradientBoostingClassifier)

### 1) RandomizedSearchCV (global search)

- Estimator: `Pipeline(preprocess → HistGradientBoostingClassifier(categorical_features=cat_idx))`
- Metric: Average Precision (`scoring="average_precision"`)
- CV: 5-fold StratifiedKFold (`shuffle=True, random_state=42`)
- Number of random configurations: 40  
- Search space:
  - `learning_rate` ~ `uniform(0.01, 0.19)`  → [0.01, 0.20)
  - `max_leaf_nodes` ~ `randint(15, 50)`
  - `min_samples_leaf` ~ `randint(5, 30)`
  - `max_iter` ~ `randint(100, 600)`

**Best configuration (RandomizedSearch, rank 1):**

- `learning_rate ≈ 0.0678`
- `max_iter = 121`
- `max_leaf_nodes = 39`
- `min_samples_leaf = 21`
- mean CV AP ≈ **0.8549**  
- std across folds ≈ **0.024**  
- mean fit time ≈ **0.09 s** per fold

**Observations (RandomizedSearch):**

- Top configurations have:
  - **small learning_rate** (~0.01–0.08),
  - `max_leaf_nodes` typically between **25 and 40**,
  - relatively large `min_samples_leaf` (≈ 14–30),
  - `max_iter` in a broad range (~120–500).
- Configurations with very large `max_iter` are noticeably slower (fit time 3–4× higher)  
  without providing clear AP gains over the best fast configuration.

RandomizedSearchCV was used as a **global search** to identify a good region in the hyperparameter space.

---

### 2) Local GridSearchCV around the RandomizedSearch optimum

After RandomizedSearchCV, a **local grid search** was performed around the best configuration to refine the hyperparameters.

- Base point (from RandomizedSearch best):  
  - `learning_rate ≈ 0.0678`  
  - `max_iter = 121`  
  - `max_leaf_nodes = 39`  
  - `min_samples_leaf = 21`

**Local grid definition:**

- Tuned hyperparameters:
  - `learning_rate`: **[0.05, 0.07, 0.09]**
  - `max_leaf_nodes`: **[30, 39, 45]**
  - `min_samples_leaf`: **[15, 21, 27]**
- Fixed:
  - `max_iter = 150`
  - all other HistGB parameters kept at their default / previously chosen values.

- Estimator: same pipeline (`preprocess → HGB (native)`).
- Metric: Average Precision (`scoring="average_precision"`).
- CV: same 5-fold StratifiedKFold (`shuffle=True, random_state=42`).
- Total combinations in the grid: 3 × 3 × 3 = **27**.

**Best configuration (local GridSearchCV):**

- `learning_rate = 0.05`
- `max_iter = 150`
- `max_leaf_nodes = 30`
- `min_samples_leaf = 21`
- mean CV AP ≈ **0.8555**

This slightly improves AP compared to the RandomizedSearch best (0.8555 vs 0.8549)  
with similar model complexity and moderate training time.

**Decision:**

- Use **RandomizedSearchCV** for **global exploration** of the hyperparameter space.
- Use **local GridSearchCV** as a **refinement step** around the best random configuration.
- Adopt the **GridSearchCV best configuration** as the **final leader** for the HGB (native) model:

> Final HGB (native) hyperparameters:  
> - `learning_rate = 0.05`  
> - `max_iter = 150`  
> - `max_leaf_nodes = 30`  
> - `min_samples_leaf = 21`  

These values are used in the final pipeline for subsequent OOF/test evaluation.

---

# Conclusion

### LogisticRegression (Baseline)
A stable reference model with strong ranking metrics (PR-AUC ≈ 0.834 CV).  
At the working point (precision ≈ 0.85), it achieves recall ≈ 0.623 and F1 ≈ 0.719.  
While it serves as a good baseline, it lags behind tree-based models in recall and F1.

### LogisticRegression (+Balanced)
Adding class weights provides no significant benefit.  
PR-AUC remains at ≈ 0.832 CV, and recall slightly decreases to ≈ 0.619 at the same precision target.  
F1 stays around ≈ 0.718.  
**Decision:** class weights are not recommended for this model in the current setup.

### RandomForestClassifier
A strong contender with ranking metrics of ROC-AUC ≈ 0.870 CV and PR-AUC ≈ 0.821.  
At precision ≈ 0.85, it achieves recall ≈ 0.652 and F1 ≈ 0.739.  
A solid **no-tune** option with better operating metrics than LogisticRegression,  
but still slightly behind the tuned HistGradientBoosting (native) model.

### HistGradientBoosting (OHE Pipeline)
This pipeline achieves PR-AUC ≈ 0.835 CV.  
At precision ≈ 0.85, recall is ≈ 0.612 and F1 ≈ 0.711.  
However, it introduces extra complexity in preprocessing (OHE) without clear performance gains over LogisticRegression or HGB-native.  
**Decision:** not selected as the final production candidate.

### HistGradientBoosting (Native: OrdinalEncoder + Categorical Features, tuned)
The best overall model:

- ROC-AUC (CV) ≈ **0.873**
- PR-AUC (CV) ≈ **0.854–0.856**
- At the chosen operating point (precision ≈ 0.85),  
  recall ≈ **0.667** and F1 ≈ **0.747** on OOF predictions.

After **RandomizedSearchCV** (global search) and **local GridSearchCV** (refinement),  
the final hyperparameters are:

- `learning_rate = 0.05`
- `max_iter = 150`
- `max_leaf_nodes = 30`
- `min_samples_leaf = 21`

This configuration provides the **best trade-off** between ranking quality (PR-AUC),  
operating metrics (recall/F1 at the target precision), and training time.  
It is selected as the **current leader** and will be used for final test evaluation and error analysis.
