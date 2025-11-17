# Titanic Survival Classification

Predicting passenger survival on the RMS Titanic using a modern, reproducible machine learning workflow.  
This project iteratively experiments with several supervised classifiers, evaluates them with robust cross‑validation, and selects operating thresholds based on precision/recall trade‑offs.

> NOTE: Some repository details were gathered via code search and those search results may be incomplete. For full context browse the repository directly: [Code search: "metrics"](https://github.com/Uvlazhnitel/Titanic-Survived-Classification/search?q=metrics)

---

## 1. Problem Statement

Given passenger attributes (e.g. class, sex, age, family relations, fare, embarkation port), predict whether a passenger survived (`Survived = 1`).  
We treat this as a binary classification problem and focus on ranking quality (ROC‑AUC, PR‑AUC) and then deriving a high‑precision operating point to maximize recall subject to a precision constraint.

- Positive class: `Survived = 1`
- Train prevalence (from current processed training split): **38.3%**  
  This prevalence provides a baseline for interpreting precision and PR‑AUC.

---

## 2. Data

Source: Kaggle Titanic dataset (stored locally under `data/raw/Titanic-Dataset.csv`).  
(Ensure you have permission / have downloaded the dataset before running code.)

Primary features currently referenced in notebooks:
- Numerical: `Age`, `SibSp`, `Parch`, `Fare`
- Categorical: `Sex`, `Pclass`, `Embarked`

Target column: `Survived`

(If additional engineered features exist in `src/` they will be documented in future revisions.)

---

## 3. Project Structure (key folders)

```
data/
  raw/              # Original CSV(s)
notebooks/
  03_metrics_thresholds.ipynb  # Evaluation & threshold selection workflow
reports/
  metrics.md        # Human-readable summary of CV & threshold metrics
  metrics_cv.csv    # Raw per-fold and aggregate metrics
src/
  preprocessing.py  # build_preprocessing(...) pipeline
venv or .venv/      # (optional) Local Python virtual environment
```

---

## 4. Environment & Installation

```bash
# Create and activate virtual environment (example)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate (Windows PowerShell)

# Install dependencies (adjust if requirements file is added later)
pip install -r requirements.txt
```

If a `requirements.txt` is not yet present, install the commonly used packages:
```bash
pip install pandas scikit-learn matplotlib
```

---

## 5. Preprocessing & Modeling

The preprocessing pipeline is constructed via `build_preprocessing(num_cols, cat_cols, remainder="drop")` (see `src/preprocessing.py`), then composed with estimators in a `sklearn.pipeline.Pipeline`.

Current models evaluated (as seen in `reports/metrics.md`):
1. LogisticRegression (baseline)
2. LogisticRegression (class-balanced variant)
3. RandomForestClassifier
4. Histogram-Based Gradient Boosting (HGB Model)
5. HGB Model (native categorical handling variant)

---

## 6. Evaluation Methodology

- Split: Stratified train/test split (20% test hold‑out) for exploratory work.
- Cross‑Validation: 5× StratifiedKFold (`shuffle=True`, `random_state=42`) for model ranking.
- Ranking Metrics: Mean ± std of ROC‑AUC and PR‑AUC (Average Precision) across folds.
- Operating Threshold: Selected on out‑of‑fold (OOF) predictions using rule:
  “Find threshold achieving precision ≥ target, then maximize recall.”
  (Target precision is implied by session logic; adjust/configure explicitly in future code.)

Metrics CSV (`reports/metrics_cv.csv`) provides raw per-fold values for one model configuration.  
Human-readable summary (`reports/metrics.md`) consolidates multiple models.

### A) Ranking Metrics (CV mean ± std)

(Extracted verbatim from `reports/metrics.md`)

| Model                          | ROC-AUC (CV)       | PR-AUC / AP (CV) | Notes                                                                 |
|--------------------------------|--------------------|------------------|-----------------------------------------------------------------------|
| LogisticRegression (baseline)  | 0.856 ± 0.031      | 0.834 ± 0.029    | OOF AUCs: ROC=0.856, AP=0.831                                         |
| RandomForestClassifier         | 0.870 ± 0.016      | 0.821 ± 0.031    | OOF AUCs: ROC=0.871, AP=0.812                                         |
| LogisticRegression (+balanced) | 0.856 ± 0.029      | 0.832 ± 0.028    | OOF AUCs: ROC=0.856, AP=0.832                                         |
| HGB Model                      | 0.868 ± 0.030      | 0.835 ± 0.036    | OOF AUCs: ROC=0.868, AP=0.826                                         |
| HGB Model (native)             | 0.873 ± 0.019      | 0.854 ± 0.021    | OOF AUCs: ROC=0.872, AP=0.847                                         |

### B) Operating Point (@ Threshold on OOF)

| Model                          | Thr.  | Precision@Thr | Recall@Thr | F1@Thr | Protocol                                        |
|--------------------------------|-------|---------------|------------|--------|-------------------------------------------------|
| LogisticRegression (baseline)  | 0.636 | 0.850         | 0.623      | 0.719  | OOF, 5-fold; Chosen index: 486                  |
| RandomForestClassifier         | 0.640 | 0.852         | 0.652      | 0.739  | OOF, 5-fold; Chosen index: 221                  |
| LogisticRegression (+balanced) | 0.743 | 0.854         | 0.619      | 0.718  | OOF, 5-fold; Chosen index: 488                  |
| HGB Model                      | 0.798 | 0.848         | 0.612      | 0.711  | OOF, 5-fold; Chosen index: 485                  |
| HGB Model (native)             | 0.679 | 0.850         | 0.667      | 0.747  | OOF, 5-fold; Chosen index: 463                  |

### C) Example Confusion Matrix (OOF) @ Listed Threshold

RandomForest @ 0.640:
- TN = 408
- FP = 31
- FN = 95
- TP = 178

---

## 7. Reproducing Results

1. Ensure data file exists: `data/raw/Titanic-Dataset.csv`.
2. Open `notebooks/03_metrics_thresholds.ipynb`.
3. Run cells sequentially to:
   - Load data
   - Build preprocessing pipeline
   - Fit baseline model
   - (Extend notebook to loop over additional models & record OOF predictions)
4. Consolidate metrics:
   - Update / regenerate `reports/metrics_cv.csv` for raw folds.
   - Render summary & threshold analyses into `reports/metrics.md`.

Future improvements:
- Automate model suite evaluation into a single script (`src/train.py` candidate).
- Add explicit configuration for precision target.
- Persist OOF predictions for auditability.
- Introduce MLflow or DVC for experiment tracking and data versioning.

---

## 8. Model Selection Notes

The best ranking performance (highest mean ROC‑AUC & PR‑AUC) currently belongs to the HGB Model (native categorical variant), edging other models with both strong ROC‑AUC (0.873 ± 0.019) and PR‑AUC (0.854 ± 0.021), while maintaining a competitive operating F1 after thresholding.

Threshold selection prioritizes retaining high precision (~0.85) while pushing recall upward. The RandomForest and HGB native model offer improved recall vs. baseline logistic regression under similar precision constraints.

---

## 9. Next Steps / Roadmap

- Add proper `requirements.txt` / `pyproject.toml`.
- Integrate automated cross‑validation and threshold tuning script.
- Add calibration analysis (e.g. reliability curves) for probability outputs.
- Explore feature importance & SHAP explanations.
- Consider handling of missing data / advanced imputation if not already present.
- Containerize (Docker) for reproducible deployment.
- Provide a simple inference interface (`predict.py` or FastAPI microservice).

---

## 10. Contributing

1. Fork / branch from `main`.
2. Follow conventional commit messages.
3. Add or update metrics in `reports/` when modifying model code.
4. Open a PR describing:
   - Data changes (if any)
   - Model adjustments
   - Impact on ROC/PR‑AUC & operating point metrics

---

## 11. Acknowledgments

- Dataset: Kaggle Titanic competition.
- Libraries: scikit-learn, pandas, matplotlib.
- Inspiration: Classic binary classification benchmark for illustrating end‑to‑end ML workflow.

---

Maintainer: @Uvlazhnitel
