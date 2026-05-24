# Sale Forecasting Task 2 — Final Plan & Results

## 1. Mô tả bài toán

- **Mục tiêu**: Dự báo tổng `quantity` sự kiện "Purchase" cho tháng **01/2026** theo từng `location × item_id`.
- **Dữ liệu**:
  | File | Nội dung |
  |------|----------|
  | `transaction_full_2025.parquet` | 41.5M rows purchase records, 1038 locations, 20393 items |
  | `event_full_2025.parquet` | 30.3M rows view_item / add_to_cart (không có location) |
  | `items.parquet` | metadata: category, brand, sale_status |
- **Metrics** (thứ tự ưu tiên): **MAPE Sales** > MAE Sales > MAPE Revenue > MAE Revenue
- **Giới hạn đánh giá**: chỉ trên location có giao dịch, loại sale_status=0

## 2. Validation Strategy

Không split random. Validation theo thời gian:

| Fold | Train | Validate |
|------|-------|----------|
| Val_Nov2025 | Jan-Oct 2025 | Nov 2025 |
| Val_Dec2025 | Jan-Nov 2025 | Dec 2025 |
| Final | Jan-Dec 2025 | **Jan 2026** |

MAPE chỉ tính trên `y_true > 0`, chỉ location active.

## 3. Feature Engineering

~70 features được tạo tại mỗi `cutoff_date`:

| Nhóm | Features |
|------|---------|
| **loc×item / purchased** | sum/mean 7/14/28/56/90d, std 28/56d, nonzero_days, days_since_last_sale, trend ratios |
| **item-level** | sum 7/28/90d, num_locations 28/90d, global trend |
| **location-level** | sum 7/28/90d, active_items 28/90d, sales trend |
| **category** | category_sales_sum_28d, item_share_in_category |
| **event view/ATC** | view_count 1/3/7/14/28d, atc_count 1/3/7/14/28d, conversion rates |
| **revenue/price** | revenue_sum 7/28/90d, avg_price 28/90d, last_price, price_change |
| **calendar** | target_month, days_in_month, is_january, quarter |

## 4. Models

| Model | Objective | Transform | Note |
|-------|-----------|-----------|------|
| A `lgbm_poisson` | Poisson | none | phù hợp count data |
| B `lgbm_tweedie` | Tweedie(1.5) | none | compound Poisson-Gamma |
| C `lgbm_l1` | regression_l1 | log1p → expm1 | robust to outliers, tối ưu MAE |

Early stopping trên Dec 2025 validation.

## 5. Kết quả Validation

_(Cập nhật sau khi chạy `python src/train_lgbm.py`)_

| Model | MAE Sales | MAPE Sales | MAE Revenue | MAPE Revenue |
|-------|-----------|-----------|-------------|--------------|
| Baseline | - | - | - | - |
| lgbm_poisson | - | - | - | - |
| lgbm_tweedie | - | - | - | - |
| lgbm_l1 | - | - | - | - |

## 6. Blend Weights

_(Cập nhật sau khi chạy `python src/blend.py`)_

Tối ưu MAPE Sales trên Dec 2025. Grid search 5-step.

```json
{
  "pred_lgbm_poisson": w1,
  "pred_lgbm_tweedie": w2,
  "pred_lgbm_l1":      w3,
  "pred_baseline":     w4
}
```

## 7. Post-processing Rules

1. **Clip**: `pred ≤ max_daily_90d × 31 × 2.0` (monthly cap)
2. **Floor**: `pred < 0.5 → 0` (giảm MAPE trên item thưa)
3. **Engagement fallback**: nếu pred=0 nhưng có ATC gần đây → `atc_7d × (31/7) × conversion_rate`
4. **sale_status=0**: `prediction = 0`

## 8. Cách chạy lại toàn bộ pipeline

```bash
# Cài dependencies
pip install -r requirements.txt

# Bước 1: Kiểm tra dữ liệu
python src/data_audit.py

# Bước 2: Baseline submission
python src/baseline.py

# Bước 3: Validation baseline
python src/validation.py

# Bước 4: Feature engineering (chậm, ~15-30 phút)
python src/features.py

# Bước 5: Train LightGBM
python src/train_lgbm.py

# Bước 6: Blend + Post-processing
python src/blend.py
python src/postprocess.py

# Bước 7: Kiểm tra submission
python src/check_submission.py
```

**Debug mode** (chạy nhanh trên 10% dữ liệu):
Sửa `config.yaml` → `DEBUG_SAMPLE: true`

## 9. Cấu trúc repo

```
├── config.yaml                  # configs chung
├── requirements.txt
├── data/data/
│   ├── transaction_full_2025.parquet
│   ├── event_full_2025.parquet
│   └── items.parquet
├── src/
│   ├── data_audit.py
│   ├── metrics.py
│   ├── baseline.py
│   ├── validation.py
│   ├── features.py
│   ├── train_lgbm.py
│   ├── blend.py
│   ├── postprocess.py
│   └── check_submission.py
├── outputs/
│   ├── submission_baseline.csv
│   ├── features_train.parquet
│   ├── features_predict_jan2026.parquet
│   ├── predictions_lgbm.parquet
│   ├── predictions_blended.parquet
│   ├── feature_importance.csv
│   ├── blend_weights.json
│   └── submission_final.csv
├── models/
│   ├── lgbm_poisson.txt
│   ├── lgbm_tweedie.txt
│   └── lgbm_l1.txt
└── reports/
    ├── data_audit.md
    ├── model_results.md
    └── final_plan_and_result.md
```
