#!/bin/bash
# =========================================================================
# KAGGLE SALES FORECASTING TASK 2 - FULL PIPELINE RUNNER
# =========================================================================
# Yêu cầu: 
# - Cấu hình khuyên dùng: GPU RTX A5000 24GB (hoặc tương đương), RAM > 20GB.
# - File config.yaml đã set DEBUG_SAMPLE: false để chạy full dữ liệu.
# =========================================================================

set -e  # Dừng script ngay lập tức nếu có bất kì bước nào văng lỗi (như OOM)

echo "🚀 [1/8] Đang khởi tạo môi trường và kiểm tra thư viện..."
pip install -r requirements.txt -q

echo "📊 [2/8] Chạy Data Audit (Kiểm tra dữ liệu chuẩn bị)..."
python src/data_audit.py

echo "📈 [3/8] Sinh file Baseline Prediction..."
python src/baseline.py

echo "⏱️  [4/8] Chạy Validation đánh giá cho Baseline..."
python src/validation.py

echo "🧠 [5/8] Feature Engineering (Bước nặng nhất - đang đẩy vào cuDF/Pandas)..."
python src/features.py

echo "🤖 [6/8] Train các mô hình LightGBM (Chạy GPU nếu có config)..."
python src/train_lgbm.py

echo "🧪 [7/8] Tìm hệ số Blend tốt nhất và trộn kết quả mô hình..."
python src/blend.py

echo "🧹 [8/8] Sàng lọc Post-processing và kiểm tra luật lệ Submisison..."
python src/postprocess.py
python src/check_submission.py

echo "======================================================================="
echo "✅ HOÀN TẤT! File dự phóng cuối cùng để đi nộp nằm ở:"
echo "👉 outputs/submission_final.csv"
echo "======================================================================="
