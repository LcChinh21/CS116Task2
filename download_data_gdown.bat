@echo off
setlocal EnableExtensions

REM Download CS116 Task 2 data and cached outputs with gdown.
REM Run this file from anywhere; it will switch to the repository root.

cd /d "%~dp0"

set "DATA_FOLDER_ID=1gT_Iy4S7ZmpiF4PWhPYWOsHK-f9Ysk-y"
set "OUTPUTS_FOLDER_ID=1Q6hgEHTZS73n2NzmUJ031wGU1EqssN55"

set "DATA_DIR=data\data"
set "OUTPUT_DIR=outputs"

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not available in PATH.
    exit /b 1
)

echo [2/4] Installing/updating gdown...
python -m pip install -q --upgrade gdown
if errorlevel 1 (
    echo ERROR: Failed to install gdown.
    exit /b 1
)

echo [3/4] Downloading data files to %DATA_DIR%...
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
gdown --folder "%DATA_FOLDER_ID%" -O "%DATA_DIR%" --remaining-ok
if errorlevel 1 (
    echo ERROR: Failed to download data folder.
    exit /b 1
)

echo [4/4] Downloading cached outputs to %OUTPUT_DIR%...
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
gdown --folder "%OUTPUTS_FOLDER_ID%" -O "%OUTPUT_DIR%" --remaining-ok
if errorlevel 1 (
    echo ERROR: Failed to download outputs folder.
    exit /b 1
)

echo.
echo Download complete.
echo Expected data path: %CD%\%DATA_DIR%
echo Expected outputs path: %CD%\%OUTPUT_DIR%
echo.
pause
