# Optimized Forecast Validation

| model                  |   mae_quantity |   mape_quantity |   mae_revenue |   mape_revenue |
|:-----------------------|---------------:|----------------:|--------------:|---------------:|
| LightGBM raw           |        2.29041 |         51.5487 |        297629 |        56.2626 |
| ensemble + postprocess |        2.32445 |         51.9996 |        303631 |        56.0344 |
| ensemble               |        2.32581 |         52.0756 |        303996 |        56.1274 |
| LightGBM log           |        2.358   |         55.2873 |        317909 |        61.3816 |
| baseline rolling3      |        1.74376 |         74.6315 |        244263 |        79.743  |
| baseline lag1          |        2.06424 |         88.2174 |        284662 |        97.2728 |

## Selected Ensemble Params

```json
{
  "alpha": 1.0,
  "scale": 0.9,
  "mae_quantity": 2.3258068561553955,
  "mape_quantity": 52.07560729980469,
  "mae_revenue": 303996.03125,
  "mape_revenue": 56.12736892700195
}
```

## Selected Postprocess Params

```json
{
  "clip_kind": "pair",
  "clip_mult": 1.0,
  "floor": 0.0,
  "mae_quantity": 2.3244502544403076,
  "mape_quantity": 51.9996223449707,
  "mae_revenue": 303631.0625,
  "mape_revenue": 56.034446716308594
}
```
