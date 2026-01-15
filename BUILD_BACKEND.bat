@echo off
echo ============================================
echo Building Whisper4Windows Backend Executable
echo ============================================
echo.

REM Detect and Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo Activating root .venv...
    call .venv\Scripts\activate.bat
) else if exist "backend\venv\Scripts\activate.bat" (
    echo Activating backend\venv...
    cd backend
    call venv\Scripts\activate.bat
    cd ..
) else (
    echo ❌ Error: Virtual environment not found! 
    echo Please create one with 'python -m venv .venv' in the root.
    exit /b 1
)

REM Build the backend executable
echo.
echo Building backend executable...
cd backend

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
pip install pyinstaller

python build_backend.py

REM Check if build was successful
if exist "dist\whisper-backend.exe" (
    echo.
    echo ============================================
    echo ✅ Build successful!
    echo Backend executable: backend\dist\whisper-backend.exe
    echo ============================================

    REM Copy to Tauri binaries folder
    if not exist "..\frontend\src-tauri\binaries" mkdir "..\frontend\src-tauri\binaries"
    copy /Y "dist\whisper-backend.exe" "..\frontend\src-tauri\binaries\whisper-backend-x86_64-pc-windows-msvc.exe"
    echo Copied to Tauri binaries folder
) else (
    echo.
    echo ============================================
    echo ❌ Build failed! Check the output above for errors.
    echo ============================================
    exit /b 1
)

cd ..
echo.
