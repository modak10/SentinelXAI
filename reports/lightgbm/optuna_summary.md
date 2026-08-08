# LightGBM Optuna Search Summary (Milestone 4, Phase 2)

- Trials total: 25
- Trials completed: 14
- Trials pruned: 11
- Best trial: #2
- Best Macro F1 (val): 0.9201
- Best Weighted F1 (val): 0.9988
- Best iteration (boosting rounds): 175

## Best Parameters

```json
{
  "learning_rate": 0.08012737503998542,
  "num_leaves": 22,
  "max_depth": 7,
  "min_child_samples": 40,
  "feature_fraction": 0.728034992108518,
  "bagging_fraction": 0.9140703845572055,
  "bagging_freq": 2,
  "lambda_l1": 0.00042472707398058225,
  "lambda_l2": 0.0021465011216654484,
  "min_split_gain": 0.046450412719997725
}
```

## Top 5 Trials by Macro F1

| Trial | Macro F1 | Weighted F1 | Duration |
|---|---|---|---|
| 2 | 0.9201 | 0.9988 | 0 days 00:07:32.361409 |
| 3 | 0.8980 | 0.9985 | 0 days 00:07:08.157393 |
| 23 | 0.8956 | 0.9987 | 0 days 00:23:15.681023 |
| 11 | 0.8941 | 0.9986 | 0 days 05:41:03.455690 |
| 24 | 0.8936 | 0.9986 | 0 days 00:04:25.560842 |