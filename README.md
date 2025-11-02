# Titanic — Survived Classification

A hands-on machine learning project for predicting passenger survival on the Titanic. This repository contains notebooks, data scaffolding, and code to explore the dataset, engineer features, train classification models, and generate a Kaggle-ready submission.

- Problem type: Binary classification (Survived vs. Not Survived)
- Dataset: [Kaggle Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Goals:
  - Exploratory Data Analysis (EDA)
  - Feature engineering and preprocessing
  - Model training, evaluation, and comparison
  - Submission CSV generation for Kaggle

---

## Repository Structure

```
.
├─ data/                  # Put raw & processed data here (not tracked by Git LFS)
│  ├─ train.csv           # Kaggle training data (add locally)
│  ├─ test.csv            # Kaggle test data (add locally)
│  └─ external/           # Optional: any external or intermediate assets
├─ notebooks/             # Jupyter notebooks for EDA, features, modeling
├─ reports/               # Generated reports, figures, and submissions
│  ├─ figures/            # Plots/images saved by notebooks
│  └─ submissions/        # Kaggle-ready CSVs
├─ src/                   # Python source (functions, pipelines, utils)
├─ requirements.txt       # Minimal dependencies for running notebooks/code
├─ pinned-requirements.txt# Fully pinned environment for reproducibility
├─ environment.yml        # (Optional) Conda environment definition
├─ .gitignore
└─ README.md
```

Note: A local `.venv/` directory may exist in this repository — you do not need it to use the project. Create your own environment as shown below.

---

## Quickstart

### 1) Prerequisites
- Python 3.11+
- pip (or uv/poetry/conda, if you prefer — examples below use pip)
- JupyterLab or Jupyter Notebook

### 2) Set up a virtual environment
```bash
# create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3) Install dependencies
Choose one:
```bash
# Less strict (fastest):
pip install -r requirements.txt

# Fully reproducible (every version pinned):
pip install -r pinned-requirements.txt
```

### 4) Get the data
Download `train.csv` and `test.csv` from the Kaggle competition and place them into `data/`.

Optionally, with the Kaggle CLI:
```bash
# Requires `pip install kaggle` and your kaggle.json API token configured
kaggle competitions download -c titanic -p data/
unzip -o data/titanic.zip -d data/
```

---

## How to Use

### Explore the dataset
- Open the notebooks in [notebooks/](notebooks/) and run them top-to-bottom.
- Typical workflow:
  1) EDA: Inspect missingness, distributions, correlations.
  2) Feature Engineering: Impute ages, encode categoricals (e.g., Sex, Embarked), create family size, titles, cabins, ticket groups, etc.
  3) Modeling: Train baseline models (e.g., Logistic Regression, Random Forest, Gradient Boosting), tune hyperparameters, cross-validate.
  4) Inference: Generate predictions on test set and export a `submission.csv`.

If the notebooks already save outputs, you should find figures under [reports/figures/](reports/figures/) and prediction files under [reports/submissions/](reports/submissions/).

### Reproduce a Kaggle submission
- Run the modeling notebook end-to-end (or your preferred one) to generate predictions.
- Make sure a CSV with columns `PassengerId` and `Survived` is saved to:
  - `reports/submissions/submission.csv`
- Upload the CSV to Kaggle on the [competition page](https://www.kaggle.com/c/titanic/submit).

---

## Modeling Notes

- Baselines:
  - Logistic Regression as a strong transparent baseline
  - Tree-based models (RandomForest, GradientBoosting/XGBoost/LightGBM) for non-linear interactions
- Typical features:
  - Encoded `Sex`, `Embarked`
  - `Pclass` as ordinal
  - Imputed `Age` (simple/modeled)
  - Family size features (`SibSp + Parch + 1`)
  - Name-derived titles (e.g., Mr, Mrs, Miss, Master)
  - Fare binning or scaling
  - Cabin/Ticket-derived features (optional, often sparse)
- Evaluation:
  - Local: Stratified K-Fold cross-validation (Accuracy/F1)
  - Kaggle: Accuracy on hidden test labels

Tip: Ensure consistent preprocessing for train/test splits; use pipelines or shared utilities in [src/](src/).

---

## Development

- Code lives in [src/](src/). Consider structuring it like:
  - `src/data.py` for loading/splitting
  - `src/features.py` for transformations/encoders
  - `src/models.py` for model definitions and training
  - `src/eval.py` for metrics and cross-validation
  - `src/utils.py` for shared helpers

- Suggested practices:
  - Use scikit-learn Pipelines to avoid data leakage.
  - Fix a random seed for reproducibility.
  - Log model parameters and CV scores.

---

## Results

- Local CV (example): Accuracy ~0.78–0.83 depending on model and features.
- Kaggle Public LB: TBD — please update with your best score and the corresponding configuration.

If you have a specific best submission in `reports/submissions/`, reference it here.

---

## Troubleshooting

- Missing data files
  - Ensure `data/train.csv` and `data/test.csv` exist before running notebooks.
- Dependency conflicts
  - Prefer `pinned-requirements.txt` for exact versions.
- Notebook kernel issues
  - Select the kernel linked to your virtual environment (Python 3.11).

---

## Roadmap / Ideas

- Add a fully scripted training/evaluation pipeline in `src/` (CLI entry point)
- Hyperparameter search (Optuna/RandomizedSearchCV/Bayesian)
- Model ensembling/blending
- Add unit tests for feature transforms
- Add CI to run a smoke test (data-free unit tests)
- Publish environment via `environment.yml` (Conda) or `uv.lock`

---

## Acknowledgements

- Kaggle: [Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- scikit-learn, pandas, numpy, matplotlib, seaborn, and the Python community.

---

## License

No license has been explicitly provided. If you intend others to use or modify this project, consider adding an open-source license (e.g., MIT, Apache-2.0) as a `LICENSE` file.

---

## Citation

If you use this repository in academic work, please cite the Kaggle competition and your chosen ML libraries (e.g., scikit-learn).
