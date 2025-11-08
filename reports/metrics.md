# Metrics

**Positive class:** `survived = 1`
**Class prevalence (train):** **TBD%**
*PR-AUC is interpreted relative to this prevalence baseline.*

|              Model | Features/Preprocessing                                                                                                                          | CV Scheme                                          |      ROC-AUC (CV) |       PR-AUC (CV) | Thr. | Precision@Thr | Recall@Thr | F1@Thr | Notes                                                                                                         |
| -----------------: | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------: | ----------------: | :--: | ------------: | ---------: | -----: | ------------------------------------------------------------------------------------------------------------- |
| LogisticRegression | preproc v1: Impute(median) → Ratios(FamilySize, FarePerPerson) → Log(Fare) → Scale + OHE(handle_unknown=ignore) + ClusterSimilarity(k=5, γ=0.1) | 5× StratifiedKFold (shuffle=True, random_state=42) | **0.855 ± 0.031** | **0.834 ± 0.029** |  TBD |           TBD |        TBD |    TBD | Baseline @ threshold=0.50 → Precision=0.7383, Recall=0.6923, F1=0.7146; Confusion Matrix=[[372,67],[84,189]]. |

**Notes:**

* PR-AUC is interpreted relative to the baseline proportion of the positive class.
* Accuracy is used as an auxiliary metric in the case of class imbalance; the primary metric is PR-AUC, and the secondary metric is ROC-AUC.

---

## Per-fold (CV)

|     fold |    roc_auc |     pr_auc |   accuracy |
| -------: | ---------: | ---------: | ---------: |
|        1 |     0.8670 |     0.8465 |     0.7902 |
|        2 |     0.8063 |     0.8072 |     0.7832 |
|        3 |     0.8860 |     0.8771 |     0.7958 |
|        4 |     0.8448 |     0.8313 |     0.7465 |
|        5 |     0.8731 |     0.8087 |     0.8239 |
| **mean** | **0.8555** | **0.8342** | **0.7879** |
|  **std** | **0.0313** | **0.0291** | **0.0279** |

---

## Validation protocol for thresholding

* Use **OOF (out-of-fold) predictions** from `5× StratifiedKFold (shuffle=True, random_state=42)`.
* Curves (ROC/PR) and all @threshold metrics are computed **on OOF only**.
* **Test set is untouched** until final evaluation.
* Artifacts: `reports/figures/roc-auc_baseline.png`, `reports/figures/pr_baseline.png`, `reports/figures/confusion_matrix.png`

---

## Thresholding Plan

* **Goal:** choose a single operating point to convert probabilities → labels.
* **Strategy:** (pick one)
  * **Precision ≥ T, maximize Recall** on OOF predictions (business: false positives are costly).
* **Target :** `T = 0.85`
* The chosen threshold will be saved to `reports/threshold.npy` and documented in the table above.

---

## Baseline @ 0.50 (from current run)

* Confusion Matrix: `[[372, 67], [84, 189]]`
* Precision = `0.7383`, Recall = `0.6923`, F1 = `0.7146`

## Versions / Seed

* Python: `3.11.14`
* scikit-learn: `1.7.2`
* `RANDOM_STATE = 42`
