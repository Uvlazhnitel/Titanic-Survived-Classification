P1 — Titanic (Survival Classification)
# Titanic Classification Project

## Goal

Build a reproducible end-to-end pipeline for Titanic survival prediction:
- Proper data split → `Pipeline`/`ColumnTransformer` → Cross-validation (CV) evaluation → Threshold selection → Final evaluation on the test set.

## Data

- **Dataset**: `data/raw.csv` (Titanic dataset).
- **Target**: `Survived` (binary: 0/1).
- **Features**:
  - **Numeric**: `Age`, `Fare`, etc.
  - **Categorical**: `Sex`, `Pclass`, `Embarked`, etc.

## Metrics & Targets (Validated via CV)

- **PR-AUC**: ≥ 0.60
- **ROC-AUC**: ≥ 0.88
- At the chosen operating threshold:
  - **Precision**: ≥ 0.85
  - **Recall**: ≥ 0.50
  - **F1**: ≥ 0.70
- **Accuracy**: Informational only.

**Note**: Pick the threshold on validation predictions, then fix it and reuse it on the test set.

## Quality Rules

1. Create the test set once and do not touch it until the end.
2. All preprocessing must reside inside the `Pipeline`/`ColumnTransformer` after the split (to avoid data leakage).
3. Use `StratifiedKFold(n_splits=5, random_state=...)` for cross-validation.
