# Data Audit Report

## Mapping từ tên cột thực tế sang ý nghĩa bài toán

| File | Cột thực tế | Ý nghĩa |
|------|-------------|---------|
| transaction_full_2025 | location | location (địa điểm bán) |
| transaction_full_2025 | item_id | item_id (mã sản phẩm) |
| transaction_full_2025 | updated_date | ngày giao dịch |
| transaction_full_2025 | quantity | số lượng mua |
| transaction_full_2025 | price | đơn giá |
| transaction_full_2025 | event_type | loại sự kiện (chỉ có 'Purchase') |
| event_full_2025 | item_id | item_id |
| event_full_2025 | event_date | ngày sự kiện |
| event_full_2025 | event_type | view_item / add_to_cart |
| event_full_2025 | customer_id | khách hàng (không có location) |
| items | item_id | item_id |
| items | sale_status | trạng thái bán (0=không bán) |
| items | category_lv1 | ngành hàng cấp 1 |
| items | price | giá chuẩn |

---

## 1. transaction_full_2025.parquet
- **Rows**: 41,470,317
- **Columns**: ['customer_id', 'item_id', 'price', 'location', 'discount', 'bill_id', 'quantity', 'event_type', 'updated_date']
- **Dtypes**:
```
  customer_id: int32
  item_id: object
  price: object
  location: int32
  discount: object
  bill_id: int32
  quantity: int32
  event_type: object
  updated_date: datetime64[us]
```
- **Null counts**: None
- **location nunique**: 1038
- **item_id nunique**: 20393
- **event_type values**: ['Purchase']
- **Date range**: 2025-01-01 06:51:05.690000 → 2025-12-31 22:30:30.267000
- **quantity stats**: min=1, max=1400, mean=1.59
- **revenue stats**: min=0, max=48760000, mean=220451

## 2. event_full_2025.parquet
- **Rows**: 30,322,216
- **Columns**: ['customer_id', 'item_id', 'price', 'quantity', 'event_type', 'event_date', 'created_date', 'updated_date']
- **Dtypes**:
```
  customer_id: int32
  item_id: object
  price: object
  quantity: int32
  event_type: object
  event_date: datetime64[us]
  created_date: datetime64[us]
  updated_date: datetime64[us]
```
- **Null counts**: None
- **item_id nunique**: 19833
- **event_type values**: ['view_item', 'add_to_cart']
- **Date range**: 2024-05-31 00:00:00 → 2025-12-04 00:00:00
- **NOTE**: Không có cột `location`. Event chỉ có customer_id.

## 3. items.parquet
- **Rows**: 29,823
- **Columns**: ['item_id', 'price', 'category_l1', 'category_l2', 'category_l3', 'category', 'brand', 'manufacturer', 'description', 'sale_status', 'size']
- **Dtypes**:
```
  item_id: object
  price: object
  category_l1: object
  category_l2: object
  category_l3: object
  category: object
  brand: object
  manufacturer: object
  description: object
  sale_status: int32
  size: object
```
- **Null counts**: None
- **sale_status distribution**: {0: 22973, 1: 6850}

## 4. Monthly Purchase Summary (transaction_full_2025)
| month   |   transactions |   total_qty |   total_revenue |   active_locations |   active_items |
|:--------|---------------:|------------:|----------------:|-------------------:|---------------:|
| 2025-01 |        3346259 |     5111048 |     7.4038e+11  |                729 |          12457 |
| 2025-02 |        2958962 |     4496932 |     6.54591e+11 |                729 |          11819 |
| 2025-03 |        3015239 |     4822234 |     6.89954e+11 |                745 |          11952 |
| 2025-04 |        3132138 |     4902122 |     7.05638e+11 |                773 |          11963 |
| 2025-05 |        3556234 |     5451526 |     7.61528e+11 |                806 |          12795 |
| 2025-06 |        3463573 |     5366211 |     7.55404e+11 |                834 |          12408 |
| 2025-07 |        3432648 |     5268305 |     7.45323e+11 |                859 |          12344 |
| 2025-08 |        3767332 |     5875812 |     8.30816e+11 |                877 |          12348 |
| 2025-09 |        3395016 |     5665194 |     7.54742e+11 |                905 |          12362 |
| 2025-10 |        3573167 |     6050888 |     8.23857e+11 |                924 |          12546 |
| 2025-11 |        3933266 |     6639604 |     8.51793e+11 |                959 |          13784 |
| 2025-12 |        3896483 |     6335813 |     8.28134e+11 |               1011 |          13615 |

---
## Kết luận quan trọng
- **Target**: `transaction_full_2025`, event_type == 'Purchase', cột `quantity`.
- **Revenue**: `price` × `quantity` (price cần cast sang float).
- **event_full_2025**: Không có `location`, chỉ dùng để feature view_item/ATC theo item_id.
- **sale_status = 0**: Loại khỏi prediction (set = 0).
- **Submission target**: Dự báo tổng `quantity` Purchase tháng 01/2026 theo `location × item_id`.