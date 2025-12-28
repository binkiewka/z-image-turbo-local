@echo off
setlocal EnableDelayedExpansion
title Z-Image-Turbo Installer

echo ===============================================================================
echo  Z-IMAGE-TURBO WINDOWS INSTALLER
echo ===============================================================================
echo.
echo This script will:
echo 1. Create a Python Virtual Environment
echo 2. Clone/Update ComfyUI
echo 3. Install all dependencies (AI, Torch, Backend, Frontend)
echo 4. Create Directory Junctions for models (linking project models to ComfyUI)
echo 5. Download required AI models (Optional)
echo.

:: 1. PREREQUISITES CHECK & AUTO-INSTALL
echo [1/5] Checking Prerequisites...

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo    - Python not found. Installing Python 3.10...
    echo      Downloading Python installer...
    curl -L -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
    echo      Installing Python (Detailed logs in python_install.log)...
    start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python_installer.exe
    
    :: Refresh Environment Variables without restarting script (limited success in batch, but we try)
    set "PATH=%PATH%;C:\Program Files\Python310;C:\Program Files\Python310\Scripts"
    
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Python installation failed or PATH not updated.
        echo         Please restart this script or install Python manually.
        pause
        exit /b
    )
    echo    - Python installed successfully.
) else (
    echo    - Python found.
)

:: Check Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo    - Git not found. Installing Git...
    echo      Downloading Git installer...
    curl -L -o git_installer.exe https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe
    echo      Installing Git...
    start /wait git_installer.exe /VERYSILENT /NORESTART
    del git_installer.exe
    
    :: Try to add standard git path
    set "PATH=%PATH%;C:\Program Files\Git\cmd"
    
    git --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Git installation failed or PATH not updated.
        echo         Please restart this script or install Git manually.
        pause
        exit /b
    )
    echo    - Git installed successfully.
) else (
    echo    - Git found.
)

:: 2. VENV SETUP
echo.
echo [2/5] Setting up Virtual Environment...
if not exist "venv" (
    echo    - Creating venv...
    python -m venv venv
) else (
    echo    - venv already exists.
)

:: Activate venv
call venv\Scripts\activate
python -m pip install --upgrade pip

:: 3. COMFYUI SETUP
echo.
echo [3/5] Installing ComfyUI (Backend)...
if not exist "ComfyUI" (
    echo    - Cloning ComfyUI...
    git clone https://github.com/comfyanonymous/ComfyUI
) else (
    echo    - ComfyUI already exists. Updating...
    pushd ComfyUI
    git pull
    popd
)

echo    - Installing PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo    - Installing ComfyUI Requirements...
pip install -r ComfyUI\requirements.txt

echo    - Installing Project Requirements...
pip install -r frontend\requirements.txt

echo    - Installing llama-cpp-python (Prebuilt Wheel for CUDA 12.1)...
:: Using prebuilt wheel to avoid Visual Studio compilation errors
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

:: 4. FOLDER LINKING
echo.
echo [4/5] Linking Model folders...
:: Create structure if missing
if not exist "models" mkdir models
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\loras" mkdir models\loras
if not exist "models\vae" mkdir models\vae
if not exist "models\upscale_models" mkdir models\upscale_models

:: Helper to create junction if target empty or non-existent
:: We link ComfyUI internal folders BACK to our project 'models' folder
:: This keeps models in ONE place (our project root)
if exist "ComfyUI\models\checkpoints" (
    rmdir "ComfyUI\models\checkpoints" 2>nul
    mklink /J "ComfyUI\models\checkpoints" "models\checkpoints"
)
if exist "ComfyUI\models\loras" (
    rmdir "ComfyUI\models\loras" 2>nul
    mklink /J "ComfyUI\models\loras" "models\loras"
)
if exist "ComfyUI\models\vae" (
    rmdir "ComfyUI\models\vae" 2>nul
    mklink /J "ComfyUI\models\vae" "models\vae"
)

echo    - Junctions created.

:: 5. MODEL DOWNLOAD
echo.
echo [5/5] Model Downloader
echo.
echo    [1] Download ALL Models (Generic + WAN 2.2 + VAE) - ~15GB
echo    [2] Download Essential Only (WAN 2.2 Low Res) - ~5GB
echo    [3] Skip Downloads
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="3" goto create_launcher
if "%choice%"=="1" goto download_all
if "%choice%"=="2" goto download_essential

:download_all
echo    - Downloading WAN 2.1 Models (High + Low + VAE)...
echo    [1/4] Downloading High-Res Model (14B)...
curl -L -o models\checkpoints\Wan2.1-T2V-14B.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-T2V-14B.pth
echo    [2/4] Downloading Low-Res Model (1.3B)...
curl -L -o models\checkpoints\Wan2.1-T2V-1.3B.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1-T2V-1.3B.pth
echo    [3/4] Downloading VAE...
curl -L -o models\vae\Wan2.1-VAE.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-VAE.pth
echo    [4/4] Downloading LoRAs...
curl -L -o models\loras\Wan2.1-T2V-14B-LoRA.safetensors https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-T2V-14B-LoRA.safetensors
goto create_launcher

:download_essential
echo    - Downloading Essential Models (Low Res Only)...
echo    [1/2] Downloading Low-Res Model (1.3B)...
curl -L -o models\checkpoints\Wan2.1-T2V-1.3B.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1-T2V-1.3B.pth
echo    [2/2] Downloading VAE...
curl -L -o models\vae\Wan2.1-VAE.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-VAE.pth
goto create_launcher

:create_launcher
echo.
echo [Finalizing] Creating Launcher 'run.bat'...
(
echo @echo off
echo call venv\Scripts\activate
echo start "ComfyUI Backend" cmd /k "python ComfyUI\main.py --listen"
echo timeout /t 5
echo start "Z-Image Frontend" cmd /k "set COMFYUI_HOST=127.0.0.1 && set COMFYUI_PORT=8188 && python frontend\api.py"
echo echo System is running!
echo echo Backend: http://127.0.0.1:8188
echo echo Frontend: http://127.0.0.1:7860
) > run.bat

echo.
echo ===============================================================================
echo  INSTALLATION COMPLETE
echo ===============================================================================
echo  Double-click 'run.bat' to start the system.
echo.
pause
