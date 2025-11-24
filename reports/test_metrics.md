# Test set evaluation

## Ranking metrics (probabilities)

- ROC-AUC (test): 0.8252
- PR-AUC / Average Precision (test): 0.7957
- Brier score (test): 0.1560

## Classification metrics at threshold = 0.596

- Precision: 0.7895
- Recall: 0.6522
- F1-score: 0.7143
- Accuracy: 0.7989

### Confusion matrix (test)

|        | Pred 0 | Pred 1 |
|--------|--------|--------|
| True 0 |     98 |     12 |
| True 1 |     24 |     45 |

## Comparison: OOF vs Test (Final Leader)

- ROC-AUC:  OOF = 0.8723, test = 0.8252  
- PR-AUC:   OOF = 0.8474, test = 0.7957  
- F1@t:     OOF = 0.7630, test = 0.7143  
- Precision@t: OOF = 0.8510, test = 0.7895  
- Recall@t:    OOF = 0.6920, test = 0.6522  

**Notes:**

- All test metrics are lower than the OOF metrics, but the gap is moderate and expected for this dataset size and protocol.  
- The model’s generalization looks reasonable: no signs of severe overfitting, but there is a noticeable drop in precision and PR-AUC when moving from OOF to the held-out test set.  
- In a production setting, these numbers should be interpreted as a realistic performance range rather than exact guarantees (precision on new data is likely to fluctuate around ~0.8 at the chosen threshold).

