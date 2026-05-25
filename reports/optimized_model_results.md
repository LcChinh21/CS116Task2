# Optimized Forecast Validation

| model                  |   mae_quantity |   mape_quantity |   mae_revenue |   mape_revenue |
|:-----------------------|---------------:|----------------:|--------------:|---------------:|
| ensemble + postprocess |        2.26838 |         51.2064 |        294543 |        55.8756 |
| LightGBM raw           |        2.2685  |         51.2134 |        294589 |        55.8845 |
| ensemble               |        2.2685  |         51.2134 |        294589 |        55.8845 |
| raw_only_postprocess   |        2.2685  |         51.2134 |        294589 |        55.8845 |
| LightGBM log           |        2.25367 |         54.9442 |        305777 |        61.1358 |
| baseline rolling3      |        1.74376 |         74.6315 |        244263 |        79.743  |
| baseline lag1          |        2.06424 |         88.2174 |        284662 |        97.2728 |

## Selected Ensemble Params

```json
{
  "alpha": 1.0,
  "scale": 1.0,
  "mae_quantity": 2.268500566482544,
  "mape_quantity": 51.21339416503906,
  "mae_revenue": 294589.0,
  "mape_revenue": 55.884544372558594,
  "blend_mode": "raw_only"
}
```

## Selected Postprocess Params

```json
{
  "clip_kind": "pair",
  "clip_mult": 1.0,
  "floor": 0.0,
  "mae_quantity": 2.268375873565674,
  "mape_quantity": 51.2064208984375,
  "mae_revenue": 294543.46875,
  "mape_revenue": 55.8756103515625
}
```
