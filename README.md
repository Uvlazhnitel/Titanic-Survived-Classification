# Titanic Survival Classification

Predict passenger survival on the RMS Titanic using a modern, reproducible, experiment-driven machine learning workflow.

This repository focuses on:

- transparent, code-driven preprocessing (no manual edits in spreadsheets);
- systematic cross-validation with out-of-fold (OOF) predictions;
- threshold selection around a business-like precision/recall constraint;
- clear separation between **training**, **model selection**, and **final test** evaluation;
- a serialized inference artifact (`leader_pipeline.joblib`) ready for reuse.

---

## 1. Problem Overview

- **Task:** Binary classification  
  Target: `Survived ∈ {0, 1}`  
- **Positive class:** `Survived = 1`

**Objective:**

1. Build several reasonable baselines (Logistic Regression, Random Forest, HistGradientBoosting).
2. Select a **leader model** based on ranking metrics (ROC-AUC, PR-AUC / Average Precision).
3. Choose an operating threshold that:
   - satisfies a **precision constraint** (≈ 0.85 on OOF),
   - maximizes **recall** under this constraint.
4. Evaluate the final leader once on the held-out test set and fix both the model and threshold as deployment artifacts.

The project is structured as a small, realistic case study of how to approach a tabular ML problem in an applied data science role.

---

## 2. Data

- **Source:** classic Titanic passenger dataset (Kaggle competition).  
- **Expected raw path** (not tracked in git):

        data/raw/Titanic-Dataset.csv

- **Core columns** currently used:
  - Numeric: `Age`, `SibSp`, `Parch`, `Fare`
  - Categorical: `Sex`, `Pclass`, `Embarked`
  - Target: `Survived`

- **Train prevalence (positive class ratio):** 38.3% (273 / 712).  
  Used as a baseline reference for interpreting precision and PR-AUC.

---

## 3. Repository Structure

High-level layout:

```text
.
├── README.md                # This file
├── environment.yml          # Conda environment (primary)
├── requirements.txt         # Minimal Python dependencies
├── pinned-requirements.txt  # Fully pinned dependency versions
├── project_tree.txt         # Convenience tree dump (optional)
├── notebooks/               # Step-by-step exploration & experiments
├── src/                     # Reusable code (pipelines, models, training, metrics)
├── reports/                 # Metrics, artifacts, figures
└── models/                  # Serialized models
```

**Key subdirectories**

### `notebooks/` – chronological development

- `01_eda_split.ipynb` – data inspection, train/test split.
- `02_preprocessing.ipynb` – preprocessing pipelines, baselines.
- `03_metrics_thresholds.ipynb` – metrics, threshold exploration.
- `04_class_weights.ipynb` – class imbalance and `class_weight="balanced"`.
- `05_random_forest.ipynb` – RandomForestClassifier baseline.
- `06_hist_gradient.ipynb` – HistGradientBoosting with OHE.
- `07_hist_gradient_native.ipynb` – HistGB with native categoricals.
- `08_tuning.ipynb` – hyperparameter tuning for HistGB (native categoricals).
- `09_error_analysis.ipynb` – hardest OOF cases, family features vs no family features.
- `10_calibration.ipynb` – probability calibration experiments.
- `11_final_threshold.ipynb` – final OOF threshold selection.
- `12_test_set_evaluation.ipynb` – final test evaluation.

### `src/`

- `preprocessing.py` – all preprocessing logic:

  - `build_leader_preprocessing()` – final HGB-native pipeline with family features.  
  - `build_preprocessing_hgb_native()` – HGB-native pipeline without family features (used for OLD vs NEW comparison).  
  - `build_baseline_preprocessing()` – generic numeric/categorical preprocessing for LogisticRegression / RandomForest / HistGB with OHE.  
  - legacy experimental functions for ratio + cluster features (`build_preprocessing`, `ClusterSimilarity`, etc.) – used only in early notebooks.

- `models.py`

  - `build_pipeline()` – constructs the final **leader** pipeline:

    ```python
    Pipeline([
        ("preprocess", <HGB-native-with-family ColumnTransformer pipeline>),
        ("model", HistGradientBoostingClassifier(..., categorical_features=cat_indices)),
    ])
    ```

- `choose_threshold.py`

  - Implements threshold search on OOF predictions under the constraint  
    `precision ≥ precision_target → maximize recall`.

- `evaluate_metrics.py`

  - Helpers to compute CV metrics, OOF metrics, confusion matrices, and export CSV/Markdown reports.

- `train_leader.py`

  - CLI script that:
    - loads the raw CSV,
    - reproduces the fixed train/test split,
    - trains the leader pipeline on the train portion,
    - saves `models/leader_pipeline.joblib`.

### `reports/`

- `metrics.md` – main **train/OOF metrics** report.  
- `test_metrics.md` – final evaluation on the held-out test set.  
- `metrics_cv.csv` – per-model CV metrics (ROC-AUC, PR-AUC).  
- `train_oof_leader.csv` – OOF probabilities for the final leader.  
- `threshold_metrics_oof.csv` – metrics vs threshold for the final leader.  
- `tuning/cv_results_hgb_random.csv` – RandomizedSearchCV results.  
- `tuning/cv_results_hgb_grid.csv` – GridSearchCV results.  
- `worst_cases_top10.csv` – top-10 hardest OOF cases by error.  
- `threshold.npy` – final deployment threshold (scalar `t_final`).  
- `figures/` – key diagnostic plots:
  - ROC & PR curves for OOF and test,
  - precision/recall/F1 vs threshold curves,
  - OOF/test confusion matrices.

### `models/`

- `leader_pipeline.joblib` – trained end-to-end pipeline for the final leader.

---

## 4. Environment & Installation

### 4.1. Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate titanic-ml    # or whatever name is defined in environment.yml
```

### 4.2. Using plain virtualenv

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
# or: pip install -r pinned-requirements.txt  # exact versions
```

Place the raw CSV at:

```text
data/raw/Titanic-Dataset.csv
```

---

## 5. Modeling & Preprocessing

### 5.1. Baseline preprocessing (LogisticRegression / RandomForest / HistGB-OHE)

Implemented in `build_baseline_preprocessing`:

- **Numeric features:**

  - `SimpleImputer(strategy="median")`  
  - `StandardScaler()`

- **Categorical features:**

  - `SimpleImputer(strategy="most_frequent")`  
  - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`

Used for:

- `LogisticRegression` (baseline)  
- `LogisticRegression(class_weight="balanced")`  
- `RandomForestClassifier`  
- HistGradientBoosting with OHE

### 5.2. Final leader preprocessing (HGB-native + family features)

Implemented in `build_leader_preprocessing`  
(wrapper over `build_preprocessing_hgb_native_with_family`):

1. **Family-related feature engineering** (`add_family_features`):

   - `family_size = SibSp + Parch + 1`  
   - `is_alone = 1` if `family_size == 1` else `0`  
   - `is_child = 1` if `Age < 18` (and not missing) else `0`

2. **Categoricals (HGB-native):**

   - `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)`  
   - categorical feature indices are passed to `HistGradientBoostingClassifier(categorical_features=...)`.

3. **Numerics:**

   - numeric columns are passed through (no scaling), including the newly created family features.

This preprocessing is paired with a tuned `HistGradientBoostingClassifier` and forms the **final leader pipeline**, serialized to `models/leader_pipeline.joblib`.

---

## 6. Evaluation Methodology

All training-set metrics are computed using:

- **Train / test split**
  - Single split with 20% held out as test.
  - Stratified split with a fixed `random_state` for reproducibility.

- **Cross-validation on the train portion**
  - `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  - Each candidate model is evaluated via CV; ranking is done by mean ROC-AUC and PR-AUC.

- **Out-of-Fold (OOF) predictions**
  - For each model, OOF probabilities are generated using  
    `cross_val_predict(..., method="predict_proba")`.
  - OOF predictions are used to:
    - select thresholds,
    - compute confusion matrices,
    - analyze calibration,
    - perform error analysis (via `train_oof_leader.csv` and `worst_cases_top10.csv`).

- **Test set**
  - Used **once**, after model and threshold have been chosen using only train/OOF data.
  - Final test metrics are reported in `reports/test_metrics.md` and are not used for tuning.

---

## 7. Results

### 7.1. Cross-Validation Ranking (train / OOF)

Average CV metrics (mean ± std):

| Model                              | ROC-AUC (CV)       | PR-AUC / AP (CV) |
|------------------------------------|--------------------|------------------|
| LogisticRegression (baseline)      | 0.856 ± 0.031      | 0.834 ± 0.029    |
| LogisticRegression (+balanced)     | 0.856 ± 0.029      | 0.832 ± 0.028    |
| RandomForestClassifier             | 0.870 ± 0.016      | 0.821 ± 0.031    |
| HistGB (OHE pipeline)              | 0.868 ± 0.030      | 0.835 ± 0.036    |
| HistGB (native categorical, tuned) | 0.873 ± 0.019      | 0.854 ± 0.021    |

The tuned **HistGradientBoosting (native categorical)** model is the best by both ROC-AUC and PR-AUC and is selected as the **leader**.

Leader’s OOF ranking metrics (pre-threshold):

- PR-AUC (AP): 0.8474  
- ROC-AUC: 0.8723  

(Details in `reports/metrics.md`.)

### 7.2. Operating Point (OOF) — Final Threshold

Deployment requires a **single threshold** on predicted probabilities.

Decision rule:

> Among all thresholds where `precision ≥ 0.85` on OOF,  
> choose the one with **maximum recall**.

For the final leader:

- **Threshold:** `t_final = 0.596`
- **OOF metrics at `t_final`:**
  - Precision: 0.851
  - Recall: 0.692
  - F1: 0.763

- **OOF confusion matrix at `t_final`:**

  |        | Pred 0 | Pred 1 |
  |--------|--------|--------|
  | True 0 |    406 |     33 |
  | True 1 |     83 |    190 |

The full sweep of thresholds and metrics is stored in `reports/threshold_metrics_oof.csv`.  
The selected `t_final` is exported to `reports/threshold.npy`.

### 7.3. Test Set Evaluation (final leader @ `t_final`)

On the held-out test set (never used in model or threshold selection):

**Ranking metrics (probabilities)**

- ROC-AUC (test): 0.8374  
- PR-AUC / Average Precision (test): 0.8062  
- Brier score (test): 0.1470  

**Classification metrics at `t_final = 0.596`**

- Precision: 0.8070  
- Recall: 0.6667  
- F1-score: 0.7302  
- Accuracy: 0.8101  

**Confusion matrix (test)**

|        | Pred 0 | Pred 1 |
|--------|--------|--------|
| True 0 |     99 |     11 |
| True 1 |     23 |     46 |

**Comparison: OOF vs Test (Final Leader)**

- ROC-AUC:  OOF = 0.8723, test = 0.8374  
- PR-AUC:   OOF = 0.8474, test = 0.8062  
- F1@t:     OOF = 0.7630, test = 0.7302  
- Precision@t: OOF = 0.8510, test = 0.8070  
- Recall@t:    OOF = 0.6920, test = 0.6667  

**Interpretation**

- Test metrics are consistently lower than OOF metrics, but the gap is moderate and expected for this sample size and protocol.
- There are no strong signs of overfitting: the model generalizes reasonably well, with realistic precision around ~0.8 at the chosen threshold.

### 7.4. Calibration (OOF, train)

Calibration experiments (Platt scaling, isotonic regression) were performed on OOF predictions of the leader.

- Baseline HistGB-native already yields **reasonably well-calibrated** probabilities (Brier ≈ 0.131).
- Platt scaling slightly improves Brier but only marginally changes PR-AUC / ROC-AUC.
- Isotonic regression further improves Brier but **worsens PR-AUC**.

Given that the main focus is **ranking quality** and thresholded metrics, the final pipeline **does not** include an extra calibration layer. See section G of `reports/metrics.md` for details.

---

## 8. Reproducing the Experiments

### 8.1. End-to-End via Notebooks

1. Ensure the environment is set up and the raw CSV is in `data/raw/Titanic-Dataset.csv`.
2. Launch Jupyter:

   ```bash
   jupyter lab
   # or:
   jupyter notebook
   ```

3. Recommended reading / execution order:

   - `01_eda_split.ipynb` – data inspection, train/test split.  
   - `02_preprocessing.ipynb` – baseline pipelines and first models.  
   - `05_random_forest.ipynb` – tree-based baseline.  
   - `06_hist_gradient.ipynb` – HistGB with OHE.  
   - `07_hist_gradient_native.ipynb` – HistGB with native categoricals.  
   - `08_tuning.ipynb` – tuning of HistGB-native.  
   - `09_error_analysis.ipynb` – hardest OOF cases, family features.  
   - `10_calibration.ipynb` – optional calibration analysis.  
   - `11_final_threshold.ipynb` – threshold sweep and selection.  
   - `12_test_set_evaluation.ipynb` – final test evaluation and plots.

Throughout the notebooks, metrics and plots are exported into `reports/` for traceability.

### 8.2. Training the Final Leader Pipeline (Script)

After installing dependencies and placing the data:

```bash
python -m src.train_leader --data-path data/raw/Titanic-Dataset.csv
```

This will:

- load the raw dataset;
- reproduce the fixed train/test split;
- train the final leader pipeline on the train portion;
- save the artifact:

```text
models/leader_pipeline.joblib
```

This artifact is compatible with the scalar threshold stored in `reports/threshold.npy`.

---

## 9. Using the Trained Model

Example inference workflow in a separate script or notebook:

```python
import numpy as np
import pandas as pd
import joblib

# Load serialized pipeline and final threshold
pipeline = joblib.load("models/leader_pipeline.joblib")
t_final = float(np.load("reports/threshold.npy"))

# Load new passenger data (must have the same columns as training X)
df_new = pd.read_csv("data/new_passengers.csv")

# Predict survival probabilities and labels
proba = pipeline.predict_proba(df_new)[:, 1]
pred = (proba >= t_final).astype(int)

df_new["survival_proba"] = proba
df_new["survival_pred"] = pred
```

In a production setting, this logic can be wrapped into a CLI tool or a web API.

---

## 10. Limitations & Future Work

* **Feature coverage**

  * Only a subset of potentially useful features is used (e.g., simple family features).
  * Future work could include:
    * title extraction from names,
    * cabin deck grouping,
    * better handling of missing ages (model-based imputation).

* **Uncertainty & drift**

  * The model is evaluated on a single train/test split.
  * Additional resampling, bootstrapping, or evaluation on external data would better quantify uncertainty.

* **Probability calibration**

  * Calibration is reasonably good for this use case.
  * If strict probabilistic interpretation is required, Platt scaling can be added to the final pipeline.

* **Deployment**

  * The repository currently provides:
    * a serialized pipeline (`leader_pipeline.joblib`),
    * a scalar threshold (`threshold.npy`).
  * Next steps:
    * a small `predict.py` CLI or FastAPI service,
    * containerization (e.g., Docker) and basic monitoring for production scenarios.

---

## 11. Maintainer

* **Author:** [@Uvlazhnitel](https://github.com/Uvlazhnitel)  
* Open to feedback and suggestions for additional benchmarks or extensions.
