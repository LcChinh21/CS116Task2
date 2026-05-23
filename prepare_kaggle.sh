#!/bin/bash
# Script to package code for Kaggle

echo "Packaging code to deploy on Kaggle..."

ZIP_FILE="kaggle_code.zip"

if [ -f "$ZIP_FILE" ]; then
    rm "$ZIP_FILE"
fi

# Need zip installed
if ! command -v zip &> /dev/null
then
    echo "zip command not found. Installing..."
    apt-get update && apt-get install -y zip
fi

# Zip the required source code and files
zip -r "$ZIP_FILE" src/ config.yaml requirements.txt notebooks/Kaggle_Run.ipynb

echo ""
echo "=== KAGGLE SETUP COMPLETE ==="
echo "File created: $ZIP_FILE"
echo ""
echo "HƯỚNG DẪN SETUP KAGGLE:"
echo "1. Bấm 'New Dataset' trên Kaggle và upload file '$ZIP_FILE'."
echo "2. Đặt tên dataset ví dụ: 'sale-forecasting-code'."
echo "3. Bấm 'New Notebook' trên Kaggle."
echo "4. Add 2 Dataset vào notebook này:"
echo "   - Dataset code vừa tạo ('sale-forecasting-code')"
echo "   - Dataset chứa 3 file parquet của BTC"
echo "5. Trong notebook Kaggle, upload file 'notebooks/Kaggle_Run.ipynb' bằng nút (File -> Import Notebook)."
echo "6. Đổi tham số CODE_DATASET_NAME và DATA_DATASET_NAME ở cell thứ 2 cho giống với folder Kaggle."
echo "7. Run All (pipeline đã được tối ưu để tự train và tự blend)."
echo "==============================="
