#!/bin/bash

# ==============================================================================
# Script Name: setup_and_upload_gdrive.sh
# Description: Installs rclone (if needed) and uploads the outputs directory
#              to Google Drive.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
REMOTE_NAME="drive"
REMOTE_DIR="output"
LOCAL_DIR="outputs/"
LOG_FILE="rclone_upload.log"

echo "============================================================"
echo " Starting Google Drive Upload Pipeline using rclone"
echo "============================================================"

# ------------------------------------------------------------------------------
# 1. Install unzip and rclone (if not installed)
# ------------------------------------------------------------------------------
if ! command -v rclone &> /dev/null; then
    echo "[1/3] rclone is not installed. Installing rclone and dependencies..."
    
    # Check if unzip is installed, install if missing (required for rclone install script)
    if ! command -v unzip &> /dev/null; then
        echo "      Installing unzip..."
        sudo apt-get update && sudo apt-get install -y unzip
    fi
    
    echo "      Installing rclone..."
    curl -s https://rclone.org/install.sh | sudo bash
    
    echo "      rclone installed successfully!"
else
    echo "[1/3] rclone is already installed. Skipping installation."
fi

# ------------------------------------------------------------------------------
# 2. Check if the rclone remote is configured
# ------------------------------------------------------------------------------
echo "[2/3] Checking for rclone remote configuration..."

# Find if the remote exists in the config
if ! rclone listremotes | grep -q "${REMOTE_NAME}:"; then
    echo "      Remote '${REMOTE_NAME}:' not found. Configuring automatically..."
    mkdir -p ~/.config/rclone
    cat >> ~/.config/rclone/rclone.conf << EOF

[${REMOTE_NAME}]
type = drive
scope = drive
token = {"access_token":"ya29.a0AQvPyIOGs2Ckatr2m6sTJEDhqMFTfd44hi9J11RBzYf06aUqJ3waSDjfNilclEvCgKYc1LhAGUHcPOEEFoAv9-OD39Tx1r9WARpqaAuq68U7BYCR4nORJbB9p__OKOljR5i3l-VQtYWPrgV2Zh6rLqz2kEx40RpV7A3ccUUZEb2VDMVvG99w1yJgC76Vm7m5YwVfwiUaCgYKAc8SARYSFQHGX2MiNLPESmcm9WzSsuwAEALvpA0206","token_type":"Bearer","refresh_token":"1//0eck9B595oDg7CgYIARAAGA4SNwF-L9IrgGeqHeCwLq9OGOPB2-bcNJBvezJFMVzl1fvL0KwNE_k2aB-q953ELRTj3gfb9h07-NE","expiry":"2026-05-25T06:32:50.055743612Z"}
EOF
    echo "      Configuration created successfully!"
else
    echo "      Remote '${REMOTE_NAME}:' found."
fi

# ------------------------------------------------------------------------------
# 3. Upload outputs folder to Google Drive
# ------------------------------------------------------------------------------
echo "[3/3] Uploading '${LOCAL_DIR}' to '${REMOTE_NAME}:${REMOTE_DIR}'..."
echo "      (Progress will be printed below)"

# Run rclone copy with progress bar. 
rclone copy "$LOCAL_DIR" "${REMOTE_NAME}:${REMOTE_DIR}" -P

echo "============================================================"
echo " Upload Pipeline Completed Successfully!"
echo "============================================================"
