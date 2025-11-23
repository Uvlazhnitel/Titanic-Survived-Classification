# Model Metrics (Train / OOF)

- **Positive class**: `Survived = 1`  
- **Train prevalence**: 38.3% (273 / 712)  
- **Baseline precision** (always predicting 1): ≈ 0.383  
  → context for interpreting PR-AUC / Average Precision.

All metrics below are computed on the **training set** using  
**5-fold StratifiedKFold** (`shuffle=True, random_state=42`) and **OOF predictions**.  
The test set is reserved for final evaluation only.

---

## A) Ranking Metrics (CV mean ± std)

Main goal: compare models by ranking quality (how well they separate positives from negatives),  
using **ROC-AUC** and **PR-AUC (Average Precision)**.

| Model                              | ROC-AUC (CV)       | PR-AUC / AP (CV) | Notes                                                        |
|------------------------------------|--------------------|------------------|--------------------------------------------------------------|
| LogisticRegression (baseline)      | 0.856 ± 0.031      | 0.834 ± 0.029    | OOF AUCs: ROC = 0.856, AP = 0.831                            |
| LogisticRegression (+balanced)     | 0.856 ± 0.029      | 0.832 ± 0.028    | OOF AUCs: ROC = 0.856, AP = 0.832                            |
| RandomForestClassifier             | 0.870 ± 0.016      | 0.821 ± 0.031    | OOF AUCs: ROC = 0.871, AP = 0.812                            |
| HistGB (OHE pipeline)              | 0.868 ± 0.030      | 0.835 ± 0.036    | OOF AUCs: ROC = 0.868, AP = 0.826                            |
| HistGB (native categorical, tuned) | 0.873 ± 0.019      | 0.854 ± 0.021    | OOF AUCs: ROC = 0.872, AP = 0.847; tuned via Random+Grid CV  |

---

## B) Operating Point (OOF, target precision ≈ 0.85)

For deployment we need a **single operating point** per model.  
Thresholds are selected on OOF predictions using the same rule as in Session 5:

> **Rule:** find all points on the PR-curve where  
> `precision ≥ target_precision` (≈ 0.85), then choose the one with **maximum recall**.

Metrics below are computed at these thresholds.

| Model                              | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Notes                                                           |
|------------------------------------|-------|---------------|------------|--------|-----------------------------------------------------------------|
| LogisticRegression (baseline)      | 0.636 | 0.850         | 0.623      | 0.719  | OOF, 5-fold; threshold from PR-curve (precision ≥ 0.85 → max R) |
| RandomForestClassifier             | 0.640 | 0.852         | 0.652      | 0.739  | OOF, 5-fold; same rule                                          |
| LogisticRegression (+balanced)     | 0.743 | 0.854         | 0.619      | 0.718  | OOF, 5-fold; same rule                                          |
| HistGB (OHE pipeline)              | 0.798 | 0.848         | 0.612      | 0.711  | OOF, 5-fold; same rule                                          |
| HistGB (native categorical, tuned) | 0.679 | 0.850         | 0.667      | 0.747  | OOF, 5-fold; same rule                                          |

---

## C) Confusion Matrices (OOF) @ Listed Thresholds

Raw counts of true/false positives/negatives for the thresholds above.  
Positive class: `Survived = 1`.

### LogisticRegression (baseline) @ Thr = 0.636
- **TN** = 409  
- **FP** = 30  
- **FN** = 103  
- **TP** = 170  

---

### RandomForestClassifier @ Thr = 0.640
- **TN** = 408  
- **FP** = 31  
- **FN** = 95  
- **TP** = 178  

---

### LogisticRegression (+balanced) @ Thr = 0.743
- **TN** = 410  
- **FP** = 29  
- **FN** = 104  
- **TP** = 169  

---

### HistGB (OHE pipeline) @ Thr = 0.798
- **TN** = 409  
- **FP** = 30  
- **FN** = 106  
- **TP** = 167  

---

### HistGB (native categorical, tuned) @ Thr = 0.679
- **TN** = 407  
- **FP** = 32  
- **FN** = 91  
- **TP** = 182  

---

## D) Hyperparameter Tuning — HistGradientBoosting (native categorical)

Tuning is done only for the **HistGB (native categorical)** model,  
using the pipeline: `preprocess → HistGradientBoostingClassifier(categorical_features=cat_idx)`.

- **Target metric:** Average Precision (AP / PR-AUC)  
- **CV protocol:** 5-fold StratifiedKFold (`shuffle=True, random_state=42`)  
- **Search strategy:**  
  1. **RandomizedSearchCV** — global exploration of the space  
  2. **GridSearchCV** — local refinement around the best random configuration  

### 1) RandomizedSearchCV (global search)

- Random configurations: **40**  
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
- mean CV AP ≈ **0.8549** (std ≈ **0.024**)  
- mean fit time ≈ **0.09 s** per fold  

**Observations:**

- Good configs share:
  - small `learning_rate` (~0.01–0.08),  
  - `max_leaf_nodes` mostly in **[25, 40]**,  
  - relatively large `min_samples_leaf` (~14–30).  
- Very large `max_iter` values are slower (3–4×) without clear AP gains.

RandomizedSearchCV is used as a **global search** to find a promising region.

---

### 2) Local GridSearchCV (refinement around RandomizedSearch optimum)

- Base point (RandomizedSearch best):  
  - `learning_rate ≈ 0.0678`  
  - `max_iter = 121`  
  - `max_leaf_nodes = 39`  
  - `min_samples_leaf = 21`

**Local grid:**

- Tuned:
  - `learning_rate`: **[0.05, 0.07, 0.09]**  
  - `max_leaf_nodes`: **[30, 39, 45]**  
  - `min_samples_leaf`: **[15, 21, 27]**  
- Fixed:
  - `max_iter = 150`  
  - other params = defaults / values from RandomizedSearch.

- CV: same 5-fold StratifiedKFold  
- Total combinations: 3 × 3 × 3 = **27**

**Best configuration (GridSearchCV):**

- `learning_rate = 0.05`  
- `max_iter = 150`  
- `max_leaf_nodes = 30`  
- `min_samples_leaf = 21`  
- mean CV AP ≈ **0.8555**

This slightly improves AP over the RandomizedSearch best (0.8555 vs 0.8549)  
with similar model complexity and moderate training time.

**Final decision (HistGB native):**

Use the **GridSearchCV best configuration** as the final leader:

> **Final HistGB (native categorical) hyperparameters**  
> - `learning_rate = 0.05`  
> - `max_iter = 150`  
> - `max_leaf_nodes = 30`  
> - `min_samples_leaf = 21`

These hyperparameters are used in the final pipeline for OOF and test evaluation.

---

## E) Model-Level Conclusions

### LogisticRegression (baseline)

- Strong and simple baseline: PR-AUC ≈ **0.834** (CV).  
- At the working point (precision ≈ 0.85): recall ≈ **0.623**, F1 ≈ **0.719**.  
- Serves as a useful reference, but tree-based models provide higher recall/F1.

---

### LogisticRegression (+balanced)

- Adding `class_weight="balanced"` does **not** bring clear benefits.  
- PR-AUC stays ≈ **0.832** (CV); recall slightly decreases to ≈ **0.619**  
  at the same precision target; F1 ≈ **0.718**.  

**Decision:** class weights are **not recommended** for LogisticRegression in this setup.

---

### RandomForestClassifier

- Strong no-tuning baseline:
  - ROC-AUC ≈ **0.870**, PR-AUC ≈ **0.821** (CV).  
- At precision ≈ 0.85:
  - recall ≈ **0.652**, F1 ≈ **0.739**.  

A solid contender with better operating metrics than LogisticRegression,  
but still behind the tuned HistGB (native) model.

---

### HistGB (OHE pipeline)

- Achieves PR-AUC ≈ **0.835** (CV).  
- At precision ≈ 0.85: recall ≈ **0.612**, F1 ≈ **0.711**.  
- Requires extra preprocessing complexity (OHE for categoricals)  
  without clear advantages over LogisticRegression or HistGB-native.

**Decision:** **not selected** as a production candidate.

---

### HistGB (native categorical, tuned) — Final Leader

The best overall model:

- ROC-AUC (CV) ≈ **0.873**  
- PR-AUC (CV) ≈ **0.854–0.856**  
- At the chosen operating point (precision ≈ 0.85, on OOF):
  - recall ≈ **0.667**  
  - F1 ≈ **0.747**

After **RandomizedSearchCV** (global search) and **local GridSearchCV** (refinement),  
the final hyperparameters are:

- `learning_rate = 0.05`  
- `max_iter = 150`  
- `max_leaf_nodes = 30`  
- `min_samples_leaf = 21`

This configuration offers the best trade-off between:

- ranking quality (PR-AUC / ROC-AUC),  
- operating metrics (recall/F1 at target precision),  
- and training time.

It is selected as the **current leader** and will be used for final test evaluation and error analysis.

---

## F) Family-Related Features (HistGB native)

Compared HistGradientBoostingClassifier (native categorical) **with vs without**  
simple family-related features:  
`family_size = SibSp + Parch + 1`, `is_alone`, `is_child`.

5-fold OOF CV (StratifiedKFold, `shuffle=True, random_state=42`):

- **HistGB native — OLD (no family features):**
  - ROC-AUC = 0.873 ± 0.021  
  - PR-AUC  = 0.850 ± 0.023  

- **HistGB native — NEW (with family features):**
  - ROC-AUC = 0.878 ± 0.020  
  - PR-AUC  = 0.853 ± 0.025  

**Conclusion (family features):**

- Family-related features give a **small positive shift** in mean ROC-AUC / PR-AUC,  
  but the improvement is within **1 std** of the CV scores.  
- They are simple and interpretable and help in some family-related FN cases,  
  so we **keep them** in the final pipeline,  
  but do **not** claim a strong, statistically stable gain.

---

## G) Probability calibration (train, OOF)

Calibration metrics are computed on OOF predictions of the final leader
(HistGB, native categorical). Test set is still untouched.

| Variant                         | Brier Score | PR-AUC (AP) | ROC-AUC | Notes                              |
|---------------------------------|------------:|------------:|--------:|------------------------------------|
| HistGB native (baseline)        |      0.1312 |      0.8481 |  0.8763 | Uncalibrated, OOF, CV=5            |
| HistGB native + Platt (sigmoid) |      0.1300 |      0.8492 |  0.8777 | CalibratedClassifierCV, OOF, CV=5  |
| HistGB native + Isotonic        |      0.1286 |      0.8419 |  0.8753 | CalibratedClassifierCV, OOF, CV=5  |

### Calibration conclusion

- The baseline HistGB (native categorical) model already provides reasonably well-calibrated probabilities:
  - Brier ≈ 0.1312, PR-AUC ≈ 0.848, ROC-AUC ≈ 0.876 (OOF, CV=5).
- Platt (sigmoid) calibration slightly improves Brier (0.1312 → 0.1300) and marginally increases PR-AUC / ROC-AUC.
  These changes are very small and within typical CV noise.
- Isotonic calibration further reduces Brier (down to 0.1286), but degrades PR-AUC (0.848 → 0.842) and ROC-AUC.
- Since the main project focus is ranking quality (PR-AUC and metrics at the chosen precision-oriented threshold),
  and the baseline model is already well calibrated, **probability calibration is not included in the final pipeline**.
- If a future use case requires strictly calibrated probabilities (e.g. risk scores used as probabilities),
  Platt scaling would be a reasonable candidate, but it is **not necessary for the current Titanic setup**.

# Leader Metrics

The final evaluation of the leader model (OOF, pre-threshold) is as follows:
- **PR-AUC**: 0.8474
- **ROC-AUC**: 0.8723

