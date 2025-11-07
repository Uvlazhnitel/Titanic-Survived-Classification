# Metrics  

| Model              | Features/Preprocessing                                                                 | CV Scheme          | ROC-AUC (CV) | PR-AUC (CV) | Notes                                                                                 |
|-------------------:|----------------------------------------------------------------------------------------|--------------------|-------------:|------------:|---------------------------------------------------------------------------------------|
| LogisticRegression | preproc v1: Impute(median) → Ratios(FamilySize,FarePerPerson) → Log(Fare) → Scale + OHE(handle_unknown=ignore) + ClusterSimilarity(k=5, γ=0.1) | 5× StratifiedKFold | —            | —           | Baseline @ threshold=0.50. Precision=0.7383, Recall=0.6923, F1=0.7146; CM=[[372,67],[84,189]]. |


**Versions/Seed**:  
- Python: `3.11.14`  
- scikit-learn: `1.7.2`  
- RANDOM_STATE: `42`  
