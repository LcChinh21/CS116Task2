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

REM Tai Google Drive folder data
gdown --folder 1gT_Iy4S7ZmpiF4PWhPYWOsHK-f9Ysk-y

cd ..

REM Di chuyen vao thu muc outputs, neu chua co thi tao moi
if not exist "outputs\" (
    mkdir outputs
)

cd outputs

REM Tai Google Drive folder outputs
gdown --folder 1Q6hgEHTZS73n2NzmUJ031wGU1EqssN55

pause
