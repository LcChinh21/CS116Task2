# Optimized Forecast Validation

| model                  |   mae_quantity |   mape_quantity |   mae_revenue |   mape_revenue |
|:-----------------------|---------------:|----------------:|--------------:|---------------:|
| ensemble + postprocess |        2.27582 |         51.1713 |        294799 |        55.8405 |
| LightGBM raw           |        2.27586 |         51.1809 |        294823 |        55.8537 |
| ensemble               |        2.27586 |         51.1809 |        294823 |        55.8537 |
| raw_only_postprocess   |        2.27586 |         51.1809 |        294823 |        55.8537 |
| LightGBM log           |        2.36017 |         55.1977 |        314894 |        61.2684 |
| baseline rolling3      |        1.74376 |         74.6315 |        244263 |        79.743  |
| baseline lag1          |        2.06424 |         88.2174 |        284662 |        97.2728 |

## Selected Ensemble Params

```json
{
  "alpha": 1.0,
  "scale": 1.0,
  "mae_quantity": 2.275864601135254,
  "mape_quantity": 51.18094253540039,
  "mae_revenue": 294823.15625,
  "mape_revenue": 55.85374069213867,
  "blend_mode": "raw_only"
}
```

## Selected Postprocess Params

```json
{
  "clip_kind": "pair",
  "clip_mult": 1.0,
  "floor": 0.0,
  "mae_quantity": 2.2758185863494873,
  "mape_quantity": 51.17127227783203,
  "mae_revenue": 294799.0625,
  "mape_revenue": 55.840538024902344
}
```
