#!/bin/bash

# ==============================================================================
# Script Name: setup_and_upload_gdrive.sh
# Description: Installs rclone (if needed) and uploads the outputs directory
#              to Google Drive.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
REMOTE_NAME="gdrive"
REMOTE_DIR="CS116_Output"
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
    echo "------------------------------------------------------------"
    echo " ERROR: rclone remote '${REMOTE_NAME}' is not configured!"
    echo "------------------------------------------------------------"
    echo " Please configure rclone manually first:"
    echo " 1. Run: rclone config"
    echo " 2. Type 'n' for new remote and name it '${REMOTE_NAME}'"
    echo " 3. Select '19' (or similar) for Google Drive (drive)"
    echo " 4. Leave Client ID/Secret blank"
    echo " 5. Set scope to '1' (Full access)"
    echo " 6. Enter 'y' for auto config to authenticate via browser"
    echo " 7. Try running this script again after setup."
    exit 1
else
    echo "      Remote '${REMOTE_NAME}:' found."
fi

# ------------------------------------------------------------------------------
# 3. Upload outputs folder to Google Drive
# ------------------------------------------------------------------------------
echo "[3/3] Uploading '${LOCAL_DIR}' to '${REMOTE_NAME}:${REMOTE_DIR}'..."
echo "      (Progress will be printed below)"

# Run رclone copy with progress bar. 
rclone copy "$LOCAL_DIR" "${REMOTE_NAME}:${REMOTE_DIR}" -P --drive-chunk-size 64M

echo "============================================================"
echo " Upload Pipeline Completed Successfully!"
echo "============================================================"
