# Two-Stage LightGBM Validation

Baseline MAPE: `51.178136`

## Config

```json
{
  "classifier": "LightGBM binary sale_flag",
  "regressor": "LightGBM regression_l1 on y>0 rows",
  "final_pred": "classifier_prob * regressor_pred",
  "weight_mode": "inv_y",
  "zero_weight": "0.3"
}
```

## Summary

|   mae_quantity |   mape_quantity |   mean_prediction |   min_prediction |   pct_pred_lt_0_1 |   pct_pred_lt_0_5 |   pct_pred_lt_1_0 |   delta_vs_baseline | cache_dir                        | submission   |
|---------------:|----------------:|------------------:|-----------------:|------------------:|------------------:|------------------:|--------------------:|:---------------------------------|:-------------|
|       2.111804 |       65.973218 |          1.095720 |         0.000000 |         23.252839 |         56.751205 |         79.405569 |           14.795082 | outputs/srcoptimized_cache_neg_B |              |

## MAPE By y_true Group

| y_true_group   |    rows |   mae_quantity |   mape_quantity |   mean_prediction |   min_prediction |   pct_pred_lt_0_1 |   pct_pred_lt_0_5 |   pct_pred_lt_1_0 |
|:---------------|--------:|---------------:|----------------:|------------------:|-----------------:|------------------:|------------------:|------------------:|
| y=0            | 1088635 |       0.322066 |      nan        |          0.322066 |         0.002323 |         28.950842 |         80.400502 |         96.507645 |
| 0<y<=1         |  406624 |       0.680404 |       68.040386 |          0.503555 |         0.000000 |         30.220056 |         58.284558 |         89.935665 |
| 1<y<=5         |  439598 |       1.928478 |       65.659555 |          1.222008 |         0.000000 |         12.799876 |         23.645694 |         61.712064 |
| y>5            |  226135 |      13.658020 |       62.865893 |          5.639460 |         0.000000 |          3.614213 |          4.499967 |         12.535432 |
