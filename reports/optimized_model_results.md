# Optimized Forecast Validation

| model                  |   mae_quantity |   mape_quantity |   mae_revenue |   mape_revenue |
|:-----------------------|---------------:|----------------:|--------------:|---------------:|
| ensemble + postprocess |        2.00451 |         60.8682 |        254178 |        62.3431 |
| ensemble               |        2.00752 |         60.9431 |        254781 |        62.4461 |
| LightGBM log           |        2.0443  |         62.0137 |        248304 |        61.3213 |
| raw_only_postprocess   |        2.1251  |         63.7328 |        248698 |        64.0251 |
| LightGBM raw           |        2.12506 |         63.7331 |        248695 |        64.0254 |
| baseline rolling3      |        1.74376 |         74.6315 |        244263 |        79.743  |
| baseline lag1          |        2.06424 |         88.2174 |        284662 |        97.2728 |

## Selected Ensemble Params

```json
{
  "alpha": 0.75,
  "scale": 1.05,
  "mae_quantity": 2.007519245147705,
  "mape_quantity": 60.943092346191406,
  "mae_revenue": 254780.703125,
  "mape_revenue": 62.44610595703125,
  "blend_mode": "raw_only"
}
```

## Selected Postprocess Params

```json
{
  "clip_kind": "pair",
  "clip_mult": 1.0,
  "floor": 0.0,
  "mae_quantity": 2.004513740539551,
  "mape_quantity": 60.86819839477539,
  "mae_revenue": 254178.28125,
  "mape_revenue": 62.34312057495117
}
```
