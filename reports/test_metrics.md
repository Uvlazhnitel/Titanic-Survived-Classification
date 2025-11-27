# Test set evaluation

## Ranking metrics (probabilities)

- ROC-AUC (test): 0.8374
- PR-AUC / Average Precision (test): 0.8062
- Brier score (test): 0.1470

## Classification metrics at threshold = 0.596

- Precision: 0.8070
- Recall: 0.6667
- F1-score: 0.7302
- Accuracy: 0.8101

### Confusion matrix (test)

|        | Pred 0 | Pred 1 |
|--------|--------|--------|
| True 0 |     99 |     11 |
| True 1 |     23 |     46 |

## Comparison: OOF vs Test (Final Leader)

- ROC-AUC:  OOF = 0.8723, test = 0.8374  
- PR-AUC:   OOF = 0.8474, test = 0.8062
- F1@t:     OOF = 0.7630, test = 0.7302 
- Precision@t: OOF = 0.8510, test = 0.8070
- Recall@t:    OOF = 0.6920, test = 0.6667

**Notes:**

- All test metrics are lower than the OOF metrics, but the gap is moderate and expected for this dataset size and protocol.  
- The model’s generalization looks reasonable: no signs of severe overfitting, but there is a noticeable drop in precision and PR-AUC when moving from OOF to the held-out test set.  
- In a production setting, these numbers should be interpreted as a realistic performance range rather than exact guarantees (precision on new data is likely to fluctuate around ~0.8 at the chosen threshold).

