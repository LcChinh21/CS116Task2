# Optimized Forecast Validation

| model                  |   mae_quantity |   mape_quantity |   mae_revenue |   mape_revenue |
|:-----------------------|---------------:|----------------:|--------------:|---------------:|
| ensemble + postprocess |        2.26834 |         51.2068 |        294542 |        55.876  |
| LightGBM raw           |        2.26847 |         51.2138 |        294588 |        55.885  |
| ensemble               |        2.26847 |         51.2138 |        294588 |        55.885  |
| raw_only_postprocess   |        2.26847 |         51.2138 |        294588 |        55.885  |
| LightGBM log           |        2.25299 |         54.9458 |        305766 |        61.1343 |
| baseline rolling3      |        1.74376 |         74.6315 |        244263 |        79.743  |
| baseline lag1          |        2.06424 |         88.2174 |        284662 |        97.2728 |

## Selected Ensemble Params

```json
{
  "alpha": 1.0,
  "scale": 1.0,
  "mae_quantity": 2.2684667110443115,
  "mape_quantity": 51.21379470825195,
  "mae_revenue": 294587.75,
  "mape_revenue": 55.885005950927734,
  "blend_mode": "raw_only"
}
```

## Selected Postprocess Params

```json
{
  "clip_kind": "pair",
  "clip_mult": 1.0,
  "floor": 0.0,
  "mae_quantity": 2.268341302871704,
  "mape_quantity": 51.20681381225586,
  "mae_revenue": 294542.125,
  "mape_revenue": 55.876041412353516
}
```
