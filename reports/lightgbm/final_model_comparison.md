# Final Model Comparison (Milestone 4): Baselines vs LightGBM

Evaluated on the VAL split (never test — see docs/JUDGE_QNA.md Q8).

| Model | Accuracy | Precision (M/W) | Recall (M/W) | F1 (M/W) | Train (s) | Infer (ms/sample) |
|---|---|---|---|---|---|---|
| lightgbm | 0.9987 | 0.9232 / 0.9988 | 0.9252 / 0.9987 | 0.9201 / 0.9988 | 342.0 | 0.0597 |
| xgboost | 0.9986 | 0.8648 / 0.9988 | 0.8856 / 0.9986 | 0.8709 / 0.9987 | 717.7 | 0.0198 |
| random_forest | 0.9953 | 0.8359 / 0.9981 | 0.8603 / 0.9953 | 0.8200 / 0.9965 | 648.8 | 0.0232 |
| logistic_regression | 0.9582 | 0.5537 / 0.9921 | 0.9142 / 0.9582 | 0.6012 / 0.9739 | 473.1 | 0.0071 |

**Ranked by Macro F1** (this project's primary metric, per docs/JUDGE_QNA.md Q12 — macro treats every class equally regardless of size, which matters given BENIGN is ~83% of the data): **lightgbm** leads at 0.9201.

## LightGBM vs XGBoost

**lightgbm surpasses xgboost on f1_macro** (0.9201 vs 0.8709, +0.0492).

| Metric | xgboost | lightgbm | Delta |
|---|---|---|---|
| Accuracy | 0.9986 | 0.9987 | +0.0001 |
| Macro Precision | 0.8648 | 0.9232 | +0.0583 |
| Macro Recall | 0.8856 | 0.9252 | +0.0396 |
| Macro F1 | 0.8709 | 0.9201 | +0.0492 |
| Weighted F1 | 0.9987 | 0.9988 | +0.0001 |
| Training Time (s) | 717.7060 | 341.9734 | -375.7326 |
| Inference (ms/sample) | 0.0198 | 0.0597 | +0.0399 |
