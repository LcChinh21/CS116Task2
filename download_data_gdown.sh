#!/usr/bin/env bash
set -euo pipefail

# Download CS116 Task 2 data and cached outputs with gdown.
# Run this file from anywhere; it will switch to the repository root.

cd "$(dirname "$0")"

DATA_FOLDER_ID="1gT_Iy4S7ZmpiF4PWhPYWOsHK-f9Ysk-y"
OUTPUTS_FOLDER_ID="1Q6hgEHTZS73n2NzmUJ031wGU1EqssN55"

DATA_DIR="data/data"
OUTPUT_DIR="outputs"

echo "[1/4] Checking Python..."
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python is required but was not found on PATH." >&2
  exit 1
fi
"${PYTHON_BIN}" --version >/dev/null
if ! "${PYTHON_BIN}" -m pip --version >/dev/null 2>&1; then
  echo "pip is required for ${PYTHON_BIN} but was not found." >&2
  exit 1
fi

echo "[2/4] Installing/updating gdown..."
"${PYTHON_BIN}" -m pip install -q --upgrade gdown

echo "[3/4] Downloading data files to ${DATA_DIR}..."
mkdir -p "${DATA_DIR}"
gdown --folder "${DATA_FOLDER_ID}" -O "${DATA_DIR}"

echo "[4/4] Downloading cached outputs to ${OUTPUT_DIR}..."
mkdir -p "${OUTPUT_DIR}"
gdown --folder "${OUTPUTS_FOLDER_ID}" -O "${OUTPUT_DIR}"

echo
echo "Download complete."
echo "Expected data path: $(pwd)/${DATA_DIR}"
echo "Expected outputs path: $(pwd)/${OUTPUT_DIR}"
