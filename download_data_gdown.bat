@echo off
setlocal

REM Di chuyen vao thu muc data, neu chua co thi tao moi
if not exist "data\" (
    mkdir data
)

cd data

REM Cai dat gdown
python -m pip install --upgrade pip
python -m pip install gdown

REM Tai Google Drive folder
gdown --folder 1gT_Iy4S7ZmpiF4PWhPYWOsHK-f9Ysk-y

pause
