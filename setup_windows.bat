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

:: -----------------------------------------------------------------------------
:: 1. CHECK PYTHON
:: -----------------------------------------------------------------------------
echo [1/5] Checking Prerequisites...

python --version >nul 2>&1
if %errorlevel% neq 0 goto install_python
echo    - Python found.
goto check_git

:install_python
echo    - Python not found. Installing Python 3.10...
echo      Downloading Python installer...
curl -L -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download Python installer.
    pause
    exit /b
)

echo      Installing Python (Detailed logs in python_install.log)...
start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
del python_installer.exe

:: Attempt to add to PATH for this session (Standard Paths)
:: We use NO parenthesis here to avoid syntax errors
set "PATH=%PATH%;C:\Program Files\Python310;C:\Program Files\Python310\Scripts"

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python installation failed or PATH not updated.
    echo         Please restart this script or install Python manually.
    pause
    exit /b
)
echo    - Python installed successfully.

:: -----------------------------------------------------------------------------
:: 2. CHECK GIT
:: -----------------------------------------------------------------------------
:check_git
git --version >nul 2>&1
if %errorlevel% neq 0 goto install_git
echo    - Git found.
goto venv_setup

:install_git
echo    - Git not found. Installing Git...
echo      Downloading Git installer...
curl -L -o git_installer.exe https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe
if %errorlevel% neq 0 (
    echo [ERROR] Failed to download Git installer.
    pause
    exit /b
)

echo      Installing Git...
start /wait git_installer.exe /VERYSILENT /NORESTART
del git_installer.exe

:: Attempt to add to PATH for this session
set "PATH=%PATH%;C:\Program Files\Git\cmd"

git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git installation failed or PATH not updated.
    echo         Please restart this script or install Git manually.
    pause
    exit /b
)
echo    - Git installed successfully.

:: -----------------------------------------------------------------------------
:: 3. VENV SETUP
:: -----------------------------------------------------------------------------
:venv_setup
echo.
echo [2/5] Setting up Virtual Environment...
if exist "venv" goto venv_exists
echo    - Creating venv...
python -m venv venv
goto venv_ready

:venv_exists
echo    - venv already exists.

:venv_ready
call venv\Scripts\activate
python -m pip install --upgrade pip

:: -----------------------------------------------------------------------------
:: 4. COMFYUI SETUP
:: -----------------------------------------------------------------------------
echo.
echo [3/5] Installing ComfyUI (Backend)...
if exist "ComfyUI" goto comfy_update

echo    - Cloning ComfyUI...
git clone https://github.com/comfyanonymous/ComfyUI
goto install_deps

:comfy_update
echo    - ComfyUI already exists. Updating...
pushd ComfyUI
git pull
popd

:install_deps
echo    - Installing PyTorch (CUDA 12.1)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo    - Installing ComfyUI Requirements...
pip install -r ComfyUI\requirements.txt

echo    - Installing Project Requirements...
pip install -r frontend\requirements.txt

echo    - Installing llama-cpp-python (Prebuilt Wheel for CUDA 12.1)...
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

:: -----------------------------------------------------------------------------
:: 5. FOLDER LINKING
:: -----------------------------------------------------------------------------
echo.
echo [4/5] Linking Model folders...
if not exist "models" mkdir models
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\loras" mkdir models\loras
if not exist "models\vae" mkdir models\vae
if not exist "models\upscale_models" mkdir models\upscale_models

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

:: -----------------------------------------------------------------------------
:: 6. MODEL DOWNLOAD
:: -----------------------------------------------------------------------------
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
goto create_launcher

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

:: -----------------------------------------------------------------------------
:: 7. CREATE LAUNCHER
:: -----------------------------------------------------------------------------
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
