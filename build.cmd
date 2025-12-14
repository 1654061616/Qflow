@echo off
title Qflow Build Script
echo Qflow Build Script
echo ================
echo.

REM Check Python installation
echo [1] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found!
    pause
    exit /b
)

echo.

REM Create virtual environment if not exist
if exist venv (
    echo [2] Virtual environment already exists
) else (
    echo [2] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment!
        pause
        exit /b
    )
)

echo.

REM Activate virtual environment
echo [3] Activating virtual environment...
call venv\Scripts\activate.bat

echo.

REM Install dependencies
echo [4] Installing dependencies...
pip install --upgrade pip
pip install pyinstaller
pip install -r requirements.txt

echo.

REM Clean old files
echo [5] Cleaning old files...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.spec del /q *.spec

echo.

REM Build application
echo [6] Building application...
pyinstaller --windowed --onefile --clean --name Qflow ^
    --hidden-import pynput.keyboard._win32 ^
    --hidden-import pynput.mouse._win32 ^
    --hidden-import comtypes ^
    --hidden-import cv2 ^
    --exclude-module matplotlib ^
    --exclude-module pandas ^
    --exclude-module scipy ^
    --exclude-module PyQt5 ^
    --exclude-module wx ^
    --exclude-module email ^
    --exclude-module http ^
    --exclude-module xml ^
    --exclude-module unittest ^
    main.py

echo.

if %errorlevel% equ 0 (
    echo Build completed successfully!
    echo Executable: dist\Qflow.exe
    echo.
    echo Press any key to exit...
) else (
    echo Build failed!
    echo.
    echo Press any key to exit...
)
pause