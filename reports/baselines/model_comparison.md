# Baseline Model Comparison (Milestone 3)

Evaluated on the VAL split (never test — see docs/JUDGE_QNA.md Q8).

| Model | Accuracy | Precision (M/W) | Recall (M/W) | F1 (M/W) | Train (s) | Infer (ms/sample) |
|---|---|---|---|---|---|---|
| xgboost | 0.9986 | 0.8648 / 0.9988 | 0.8856 / 0.9986 | 0.8709 / 0.9987 | 717.7 | 0.0198 |
| random_forest | 0.9953 | 0.8359 / 0.9981 | 0.8603 / 0.9953 | 0.8200 / 0.9965 | 648.8 | 0.0232 |
| logistic_regression | 0.9582 | 0.5537 / 0.9921 | 0.9142 / 0.9582 | 0.6012 / 0.9739 | 473.1 | 0.0071 |

**Ranked by Macro F1** (this project's primary metric, per docs/JUDGE_QNA.md Q12 — macro treats every class equally regardless of size, which matters given BENIGN is ~83% of the data): **xgboost** leads at 0.8709.