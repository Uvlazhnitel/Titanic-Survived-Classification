# Titanic Survival Classification

Predict passenger survival on the RMS Titanic using a modern, reproducible, experiment‑driven machine learning workflow.  
This repository emphasizes: transparent preprocessing, systematic cross‑validation, threshold optimization around precision/recall trade‑offs, and clear reporting of operating points for decision support.

> If you are viewing this on GitHub, ensure you have locally obtained the Kaggle Titanic dataset before running any code.

---

## 1. Overview

- Task: Binary classification (`Survived` ∈ {0,1})
- Primary Objectives:
  1. Develop robust, generalizable models.
  2. Rank model families using ROC‑AUC and PR‑AUC (Average Precision).
  3. Select high‑precision thresholds that maximize recall subject to a precision constraint (≈0.85 in current runs).
- Current Best Performing Variant (ranking metrics): Histogram‑Based Gradient Boosting (native categorical handling).

---

## 2. Key Features

- Reusable preprocessing pipeline via scikit‑learn `ColumnTransformer`.
- Multiple estimator families benchmarked (linear, ensemble, boosting).
- Cross‑validated ranking (5× StratifiedKFold) with out‑of‑fold (OOF) predictions retained for threshold tuning.
- Operating point selection rule: “Among thresholds achieving target precision, maximize recall.”
- Human‑readable metric reports (Markdown + CSV).
- Modular structure prepared for automation / future experiment tracking (MLflow / DVC).

---

## 3. Data

Source: [Kaggle Titanic](https://www.kaggle.com/c/titanic) (download manually; not bundled).

Expected raw file path:
```
data/raw/Titanic-Dataset.csv
```

Primary features currently in scope:
- Numerical: `Age`, `SibSp`, `Parch`, `Fare`
- Categorical: `Sex`, `Pclass`, `Embarked`
- Target: `Survived`

Train prevalence (positive class ratio): **38.3%**  
(Use this as a baseline when interpreting precision & PR‑AUC.)

Future extensions may include engineered features (e.g. family size, fare per person, title extraction); these will be documented as added.

## 4. Environment & Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# or: .venv\Scripts\activate      # Windows PowerShell

pip install -r requirements.txt  # once added
```

If `requirements.txt` is not present yet, install the minimal stack:
```bash
pip install pandas scikit-learn matplotlib
```

(You may also add: `numpy`, `seaborn`, `joblib`, `pyyaml` as needed.)

---

## 5. Preprocessing & Modeling

Defined in `src/preprocessing.py`:
```python
def build_preprocessing(num_cols, cat_cols, remainder="drop"):
    # Returns a ColumnTransformer doing:
    # - Numerical: median imputation + optional scaling
    # - Categorical: most-frequent imputation + OneHotEncoder (or passthrough for native HGB)
```

Model families currently evaluated:
1. Logistic Regression (baseline)
2. Logistic Regression (class-weight balanced)
3. RandomForestClassifier
4. Histogram-Based Gradient Boosting (standard)
5. Histogram-Based Gradient Boosting (native categorical handling)

Each model is wrapped into a `Pipeline(preprocessing, estimator)` except the native HGB variant which may adapt categorical preprocessing (e.g., passthrough encoded vs. native categories).

---

## 7. Evaluation Methodology

| Aspect                | Current Setting |
|-----------------------|-----------------|
| Split Strategy        | Stratified train/test (20% test hold-out; primary analysis on CV OOF) |
| Cross-Validation      | 5-fold StratifiedKFold (`shuffle=True`, `random_state=42`) |
| Ranking Metrics       | Mean ± std ROC‑AUC, PR‑AUC (Average Precision) across folds |
| Threshold Selection   | On aggregated OOF predictions (concatenated folds) |
| Threshold Criterion   | Precision ≥ target (≈0.85) then choose threshold maximizing recall |
| Report Artifacts      | `reports/metrics.md`, `reports/metrics_cv.csv` |

Out‑of‑fold predictions enable:
- Fair threshold tuning (no leakage from test hold‑out).
- Later calibration analysis (planned).

---

## 8. Current Cross‑Validation Results

Extracted from latest `reports/metrics.md`.

### A) Ranking Metrics (CV mean ± std)

| Model                          | ROC-AUC (CV)       | PR-AUC / AP (CV) | OOF ROC | OOF AP | Notes |
|--------------------------------|--------------------|------------------|---------|--------|-------|
| LogisticRegression (baseline)  | 0.856 ± 0.031      | 0.834 ± 0.029    | 0.856   | 0.831  | Baseline reference |
| RandomForestClassifier         | 0.870 ± 0.016      | 0.821 ± 0.031    | 0.871   | 0.812  | Strong ROC; slightly lower AP |
| LogisticRegression (+balanced) | 0.856 ± 0.029      | 0.832 ± 0.028    | 0.856   | 0.832  | Class weighting modest effect |
| HGB Model                      | 0.868 ± 0.030      | 0.835 ± 0.036    | 0.868   | 0.826  | Competitive boosting baseline |
| HGB Model (native)             | 0.873 ± 0.019      | 0.854 ± 0.021    | 0.872   | 0.847  | Current best overall ranking |

### B) Operating Points (OOF Threshold Selection)

| Model                          | Threshold | Precision | Recall | F1    | Selection Context |
|--------------------------------|----------:|----------:|-------:|------:|-------------------|
| LogisticRegression (baseline)  | 0.636     | 0.850     | 0.623  | 0.719 | Index=486 |
| RandomForestClassifier         | 0.640     | 0.852     | 0.652  | 0.739 | Index=221 |
| LogisticRegression (+balanced) | 0.743     | 0.854     | 0.619  | 0.718 | Index=488 |
| HGB Model                      | 0.798     | 0.848     | 0.612  | 0.711 | Index=485 |
| HGB Model (native)             | 0.679     | 0.850     | 0.667  | 0.747 | Index=463 |

### C) Example Confusion Matrix (RandomForest @ 0.640 Threshold)

|        | Pred=0 | Pred=1 |
|--------|--------|--------|
| Actual=0 | TN=408 | FP=31 |
| Actual=1 | FN=95  | TP=178 |

Interpretation:
- Precision baseline satisfied (~0.85).
- Native HGB maintains similar precision while improving recall vs. logistic regression baseline.

---

## 9. Threshold Selection Logic (Pseudo-Code)

```python
def choose_threshold(probs, y_true, precision_target=0.85):
    candidates = np.linspace(0, 1, 1001)
    best = None
    for t in candidates:
        preds = (probs >= t).astype(int)
        prec = precision_score(y_true, preds)
        rec  = recall_score(y_true, preds)
        if prec >= precision_target:
            if best is None or rec > best["recall"]:
                best = {"threshold": t, "precision": prec, "recall": rec}
    return best
```

(Planned enhancement: direct sweep using sorted unique probabilities for efficiency.)

---

## 10. Probability Calibration

- I evaluated probability calibration for the final leader (HistGradientBoostingClassifier with native categoricals)
  using `CalibratedClassifierCV` with both **Platt (sigmoid)** and **isotonic** methods on the training set (OOF).
- Platt scaling slightly improved Brier score but only marginally changed PR-AUC / ROC-AUC.
- Isotonic regression provided the lowest Brier score but degraded PR-AUC, which is the main ranking metric in this project.
- **Decision:** for this project, I do **not** include a separate calibration step in the final pipeline.
  The baseline HistGB probabilities are good enough, and the small gains do not justify extra complexity.


## 11. Reproducing Results

1. Acquire Kaggle dataset; place CSV under `data/raw/`.
2. Launch Jupyter and open `notebooks/03_metrics_thresholds.ipynb`.
3. Run cells to:
   - Load & split data.
   - Build preprocessing pipeline.
   - Iterate over model list, collecting OOF predictions.
4. Generate per-fold metrics -> export `reports/metrics_cv.csv`.
5. Summarize rankings & threshold analyses -> update `reports/metrics.md`.
6. (Future) Run script `src/train.py` (once added) for fully automated replication.

---

## 11. Roadmap / Planned Enhancements

| Category          | Item |
|-------------------|------|
| Experiment Mgmt    | Introduce MLflow or DVC for artifact lineage |
| Automation         | Add `src/train.py` and `src/evaluate.py` |
| Feature Eng        | Title extraction, family size, cabin deck parsing |
| Threshold Config   | Externalize precision target to a config file |
| Calibration        | Reliability curves, Brier score, isotonic/Platt scaling |
| Interpretability   | SHAP value analysis, permutation importance |
| Deployment         | `predict.py` CLI or FastAPI microservice + Docker image |
| Reproducibility    | `pyproject.toml` / `requirements.txt` lock-down |
| Data Handling      | Advanced imputation (age via regression / iterative) |
| CI                 | GitHub Actions for lint + unit tests + metrics drift |
| Security           | Add license, consider model card |

---

## 12. Contributing

1. Create a feature branch from `main`.
2. Use conventional commits (e.g., `feat:`, `fix:`, `docs:`).
3. Update `reports/metrics.md` & `reports/metrics_cv.csv` if model behavior changes.
4. Include:
   - Description of modifications.
   - Impact on ROC‑AUC / PR‑AUC / precision-recall operating points.
5. Open a Pull Request; link to any relevant issues.

---

## 13. Suggested Directory Additions (Future)

- `src/config/` for YAML/JSON experiment configs.
- `tests/` for unit tests (e.g., preprocessing integrity).
- `scripts/` for batch run helpers.
- `models/` for stored serialized artifacts (if tracked via DVC/MLflow).

---

## 14. License & Usage

(Choose an appropriate license: e.g., MIT.)

Data from Kaggle Titanic competition is subject to Kaggle’s terms of use. Do not redistribute raw data outside permitted scope.

---

## 15. Acknowledgments

- Dataset: Kaggle Titanic competition.
- Libraries: scikit-learn, pandas, matplotlib (plus potential extensions).
- Classic benchmark inspiration for end‑to‑end ML workflow demonstration.

---

## 16. Maintainer

@Uvlazhnitel

---

### Quick Reference Badges (Add once CI in place)

| Badge | Purpose |
|-------|---------|
| Build Status | GitHub Actions workflow pass/fail |
| Code Style   | Linting (e.g., flake8, black) |
| Coverage     | Test coverage % |
| License      | Declared project license |

(Insert badges once available.)

---

Feel free to open issues for clarification, enhancements, or additional benchmarking requests.

---

### Model Snapshot Summary

The native HGB model currently offers the best balance between ranking quality and recall at the high precision target. Continued feature engineering and calibration may further improve reliability at deployment thresholds.

#### Key Visualizations

- **ROC Curve (Out-of-Fold):** ![ROC OOF](reports/figures/roc_oof_leader_train.png)
- **ROC Curve (Test Set):** ![ROC Test](reports/figures/roc_test.png)
- **Precision-Recall Curve (Out-of-Fold):** ![PR OOF](reports/figures/pr_oof_leader_train.png)
- **Precision-Recall Curve (Test Set):** ![PR Test](reports/figures/pr_test.png)


## Final Results & Summary

**Task.** Binary classification of Titanic passenger survival using the Kaggle Titanic dataset.

**Final model.** HistGradientBoostingClassifier (native categorical handling) with simple family-related features and tuned hyperparameters:

- `learning_rate = 0.05`
- `max_iter = 150`
- `max_leaf_nodes = 30`
- `min_samples_leaf = 21`

**Validation protocol.**

- Stratified train/test split (20% held-out test).
- 5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the train set.
- Out-of-fold (OOF) probabilities used for model selection, threshold tuning and calibration analysis.
- Decision threshold chosen on OOF via the rule  
  **“among thresholds with precision ≥ 0.85, pick the one with maximum recall”**,  
  which yields the final threshold `t_final = 0.596`.

---

### OOF performance (train, CV=5)

Using OOF probabilities of the final pipeline:

- **ROC-AUC:** 0.8723  
- **PR-AUC / Average Precision:** 0.8474  

At the final threshold `t = 0.596`:

- **Precision:** 0.8510  
- **Recall:** 0.6920  
- **F1-score:** 0.7630  

These values are computed strictly on OOF predictions (no test leakage).

---

### Held-out test performance

Evaluated exactly once on the held-out test set, using the frozen pipeline and the fixed threshold `t = 0.596`:

- **ROC-AUC (test):** 0.8252  
- **PR-AUC / Average Precision (test):** 0.7957  
- **Brier score (test):** 0.1560  

At `t = 0.596` on the test set:

- **Precision:** 0.7895  
- **Recall:** 0.6522  
- **F1-score:** 0.7143  
- **Accuracy:** 0.7989  

Confusion matrix (test, positive class `Survived = 1`):

- **TN = 98**, **FP = 12**  
- **FN = 24**, **TP = 45**

---

### Generalization & interpretation

- All test metrics are lower than the OOF metrics, but the gaps are moderate and consistent with the dataset size and the chosen protocol.
- The model maintains reasonably high precision (~0.79) at the chosen operating point while recovering ~65% of positives.
- There are no obvious signs of severe overfitting; performance on the held-out test set is a realistic estimate of how this pipeline would behave on future data drawn from the same distribution.

---

### Takeaways

- Tree-based gradient boosting with native categorical support and light feature engineering outperforms logistic regression baselines in both ranking metrics (ROC-AUC, PR-AUC) and F1 at the high-precision operating point.
  
**P1 status:** completed (end-to-end workflow from raw CSV to reproducible CV and single-shot test evaluation).
