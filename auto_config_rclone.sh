#!/bin/bash

echo "Đang cấu hình rclone tự động..."

# Tạo thư mục chứa file config nếu chưa có
mkdir -p ~/.config/rclone

# Tạo nội dung cấu hình
cat > ~/.config/rclone/rclone.conf << 'EOF'
[drive]
type = drive
scope = drive
token = {"access_token":"ya29.a0AQvPyIOGs2Ckatr2m6sTJEDhqMFTfd44hi9J11RBzYf06aUqJ3waSDjfNilclEvCgKYc1LhAGUHcPOEEFoAv9-OD39Tx1r9WARpqaAuq68U7BYCR4nORJbB9p__OKOljR5i3l-VQtYWPrgV2Zh6rLqz2kEx40RpV7A3ccUUZEb2VDMVvG99w1yJgC76Vm7m5YwVfwiUaCgYKAc8SARYSFQHGX2MiNLPESmcm9WzSsuwAEALvpA0206","token_type":"Bearer","refresh_token":"1//0eck9B595oDg7CgYIARAAGA4SNwF-L9IrgGeqHeCwLq9OGOPB2-bcNJBvezJFMVzl1fvL0KwNE_k2aB-q953ELRTj3gfb9h07-NE","expiry":"2026-05-25T06:32:50.055743612Z"}
EOF

echo "Cấu hình rclone thành công!"
