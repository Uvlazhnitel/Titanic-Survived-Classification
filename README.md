# Titanic Survived Classification

A clean, reproducible baseline for the Kaggle Titanic binary classification task using scikit-learn Pipelines to avoid data leakage.

## Environment & Installation

Choose ONE of the two approaches:

1. Lightweight (pip):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Reproducible (Conda + requirements):
```bash
conda env create -f environment.yml
conda activate titanic
```

To reproduce an exact setup:
```bash
pip install -r pinned-requirements.txt
```

Install pre-commit hooks (recommended):
```bash
pip install pre-commit
pre-commit install
```

## Data

Download the Kaggle Titanic dataset and place the files in `data/`:
- `data/train.csv`
- `data/test.csv` (optional)

See [data/README.md](data/README.md) for details.

## Project Structure

```
.
├── data/                # Raw & sample data (large files not committed)
├── notebooks/           # Exploratory & experiment notebooks
├── src/                 # Reusable Python modules
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Training, evaluation, prediction
│   └── utils/           # Shared helpers (paths, etc.)
├── reports/             # Generated metrics & figures
├── models/              # Saved model artifacts (ignored except docs)
├── requirements.txt     # Top-level dependencies
├── pinned-requirements.txt # Fully pinned environment
├── environment.yml      # Conda environment (optional)
└── .pre-commit-config.yaml
```

## Run Baseline

Train model with cross-validation and save the pipeline:
```bash
python -m src.models.train
```

Evaluate on the training set to produce metrics and plots:
```bash
python -m src.models.evaluate
```

Artifacts:
- `models/logreg_pipeline.joblib` — trained pipeline
- `reports/metrics.json` — metrics (accuracy, precision, recall, F1, ROC AUC)
- `reports/figures/` — confusion matrix, ROC, PR curves

## Next Steps
- Add more feature engineering (Title, FamilySize, IsAlone, CabinDeck, Ticket groups)
- Try alternative models (RandomForest, XGBoost, LightGBM)
- Add permutation importance or SHAP for explainability
- Add CI to run linting and tests
