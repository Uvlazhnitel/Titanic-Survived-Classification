# Titanic Classification Project

## Goal

Build a reproducible end-to-end pipeline for Titanic survival prediction:
- Proper data split → `Pipeline`/`ColumnTransformer` → Cross-validation (CV) evaluation → Threshold selection → Final evaluation on the test set.

---

## Data

- **Dataset**: `data/raw.csv` (Titanic dataset).
- **Target**: `Survived` (binary: 0/1).
- **Features**:
  - **Numeric**: `Age`, `Fare`, `SibSp`, `Parch`, `Pclass`, etc.
  - **Categorical**: `Sex`, `Embarked` (optionally treat `Pclass` as categorical).
  - **Engineered inside the pipeline**:
    - `FamilySize`, `FarePerPerson`, `log1p(Fare)`, optional cluster-similarity features (e.g., `KMeans` + `RBF`).

---

## Metrics & Targets (Validated via CV/OOF)

### Primary Ranking Metrics (OOF, mean ± std over folds):
- **PR-AUC (Average Precision)**: ≥ 0.60
- **ROC-AUC**: ≥ 0.88

### Operating Point Metrics @ Chosen Threshold:
- **Precision**: ≥ 0.85
- **Recall**: ≥ 0.50
- **F1**: ≥ 0.70
- **Accuracy**: Informational only.

### Thresholding Rule:
- Use one fixed strategy across models:
  - Either maximize F1 or set `Precision ≥ T` (e.g., 0.85) and then maximize Recall.
- Save the chosen threshold to `reports/threshold_<model>.npy`.
- Do not tune the threshold on the test set.

---

## Validation Protocol

1. **Data Splitting**:
   - Split once into train/test and freeze the test indices (`data/splits/*`).

2. **Cross-Validation**:
   - Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

3. **OOF Predictions**:
   - Use `cross_val_predict(..., method="predict_proba")`.

4. **Reporting**:
   - Report **ROC-AUC** and **PR-AUC** as OOF mean ± std.
   - Save plots to:
     - `reports/figures/roc_oof.png`
     - `reports/figures/pr_oof.png`.

5. **Threshold Selection**:
   - Perform on OOF predictions only.
   - Save to `reports/threshold_<model>.npy`.

6. **Final Test Evaluation**:
   - Run once after the pipeline is locked.

---

## Models & Comparison Rules

1. **Baseline**:
   - `LogisticRegression` (with/without `class_weight='balanced'` as decided in Session 6).

2. **Tree Ensemble**:
   - `RandomForestClassifier` (Session 7).

3. **Advanced**:
   - `HistGradientBoostingClassifier` (Session 10).

4. **Preprocessing**:
   - Use the same preprocessing pipeline (`build_preprocessing(...)`) for all models.

5. **Thresholding**:
   - Apply the same thresholding rule for fair comparison across models.

6. **Metrics Reporting**:
   - Record per-model metrics in `reports/metrics.md`:
     - OOF **ROC-AUC**, **PR-AUC** (mean ± std).
     - Threshold, **Precision@thr**, **Recall@thr**, **F1@thr** (OOF).
     - Confusion matrix @ threshold (OOF).

---

## Quality Rules

1. **Test Set**:
   - Create once and do not touch until the end.

2. **Preprocessing**:
   - All preprocessing must reside inside the `Pipeline`/`ColumnTransformer` after the split (to avoid data leakage).

3. **Cross-Validation**:
   - Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for CV and OOF predictions.

4. **Thresholds & Tuning**:
   - Choose thresholds, calibration, feature selection, and hyper-parameter tuning on OOF/CV only (never on the test set).

5. **Reproducibility**:
   - Fix seeds (`random_state=42`) for all stochastic components (CV, models, `KMeans`, etc.).

---

## Leakage & Validation Checklist

### Splits & CV:
- Test/holdout untouched until final evaluation.
- Single CV scheme across experiments (5-fold Stratified, shuffled, fixed seed).
- All reported metrics and curves computed from OOF predictions.

### Pipelines:
- Imputation, scaling, OHE, engineered features, and clustering are inside the `Pipeline`/`ColumnTransformer`.
- No global fit/transform on the full train outside CV used for evaluation.

### Features:
- No target/label/service columns among features (`Survived`, `is_train`, `fold_id`, etc.).
- No future/temporal info (N/A for Titanic); note potential groups (`Ticket`/family).
- High-cardinality/ID-like columns (`Ticket`, `Cabin`, `Name`) marked as monitor.

### Thresholding & Tuning:
- One fixed thresholding rule across models, chosen on OOF; saved to `reports/threshold_<model>.npy`.
- Hyper-parameter search (when used) logged via `cv_results_`; test evaluated once.


---

## Reproducibility

- Python 3.10+, pinned dependencies, fixed seeds.
- Use `set_config(transform_output="pandas")` in preprocessing for clean feature names.
- All experiments are re-runnable end-to-end from notebooks/scripts with the same CV protocol.