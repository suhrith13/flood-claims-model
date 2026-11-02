| Setup | Features | Split | MAE (ratio) | MAE ($ per claim) | R2 |
|---|---|---|---|---|---|
| Baseline: always predict the median ratio | none | time (future years) | 0.157 | 6,511 | -0.151 |
| Baseline: policy math (damage - deductible, capped at coverage) | 3 columns | time (future years) | 0.052 | 2,336 | 0.649 |
| Naive: leaky features, random split | all + payout-derived | random rows | 0.010 | 634 | 0.993 |
| Honest features, random split | pre-decision only | random rows | 0.055 | 2,563 | 0.808 |
| Honest features, time split (the real number) | pre-decision only | time (future years) | 0.079 | 4,105 | 0.646 |

'Paid at all' classifier (honest features, time split): AUC 0.955; 94.7% of test claims were paid something.