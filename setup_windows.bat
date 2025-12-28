@echo off
setlocal
title Z-Image-Turbo Installer

echo ===============================================================================
echo  Z-IMAGE-TURBO WINDOWS INSTALLER
echo ===============================================================================
echo.
echo This script will:
echo 1. Create a Python Virtual Environment
echo 2. Clone and Update ComfyUI
echo 3. Install all dependencies
echo 4. Link project models to ComfyUI models
echo 5. Download required AI models
echo.

:: -----------------------------------------------------------------------------
:: 1. CHECK PYTHON
:: -----------------------------------------------------------------------------
echo Step 1 of 5: Checking Prerequisites...

:: Default to 'python', but we will test it.
set "PYTHON_EXE=python"

:: Check if 'python' command actually runs a real Python (not the Store shim)
%PYTHON_EXE% --version >nul 2>&1
if errorlevel 1 goto install_python

echo    - Python found.
goto check_git

:install_python
echo    - Python not found (or is just the Store shortcut). Installing Python 3.10...
echo      Downloading Python installer...
curl -L -o python_installer.exe https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
if errorlevel 1 (
    echo [ERROR] Failed to download Python installer.
    pause
    exit /b
)

echo      Installing Python...
start /wait python_installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
del python_installer.exe

:: Force use of the specific path we just installed to
set "PYTHON_EXE=C:\Program Files\Python310\python.exe"

:: Add to PATH for future sessions
set "PATH=%PATH%;C:\Program Files\Python310;C:\Program Files\Python310\Scripts"

:: Verify installation with the full path
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python installation failed.
    echo         Please restart this script or install Python manually.
    pause
    exit /b
)
echo    - Python installed successfully.

:: -----------------------------------------------------------------------------
:: 2. CHECK GIT
:: -----------------------------------------------------------------------------
:check_git
where git >nul 2>&1
if errorlevel 1 goto install_git
echo    - Git found.
goto venv_setup

:install_git
echo    - Git not found. Installing Git...
echo      Downloading Git installer...
curl -L -o git_installer.exe https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe
if errorlevel 1 (
    echo [ERROR] Failed to download Git installer.
    pause
    exit /b
)

echo      Installing Git...
start /wait git_installer.exe /VERYSILENT /NORESTART
del git_installer.exe

:: Add to PATH
set "PATH=%PATH%;C:\Program Files\Git\cmd"

where git >nul 2>&1
if errorlevel 1 (
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
echo Step 2 of 5: Setting up Virtual Environment...
if exist "venv" goto venv_exists
echo    - Creating venv...
:: Use the verified PYTHON_EXE path to create the venv
"%PYTHON_EXE%" -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment. 
    echo         Please try running this script as Administrator.
    pause
    exit /b
)
goto venv_ready

:venv_exists
echo    - venv already exists.

:venv_ready
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b
)
python -m pip install --upgrade pip

:: -----------------------------------------------------------------------------
:: 4. COMFYUI SETUP
:: -----------------------------------------------------------------------------
echo.
echo Step 3 of 5: Installing ComfyUI Backend...
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

echo    - Installing llama-cpp-python (Prebuilt Wheel for CUDA 12.1)...
:: Install BEFORE requirements.txt prevents compilation attempt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

echo    - Installing ComfyUI Requirements...
pip install -r ComfyUI\requirements.txt

echo    - Installing Project Requirements...
pip install -r frontend\requirements.txt

:: -----------------------------------------------------------------------------
:: 5. FOLDER LINKING
:: -----------------------------------------------------------------------------
echo.
echo Step 4 of 5: Linking Model folders...
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
echo Step 5 of 5: Model Downloader
echo.
echo    [1] Download ALL Models (Generic + WAN 2.2 + VAE) - 15GB
echo    [2] Download Essential Only (WAN 2.2 Low Res) - 5GB
echo    [3] Skip Downloads
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="3" goto create_launcher
if "%choice%"=="1" goto download_all
if "%choice%"=="2" goto download_essential
goto create_launcher

:download_all
echo    - Downloading WAN 2.1 Models (High + Low + VAE)...
echo    Downloading High-Res Model...
curl -L -o models\checkpoints\Wan2.1-T2V-14B.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-T2V-14B.pth
echo    Downloading Low-Res Model...
curl -L -o models\checkpoints\Wan2.1-T2V-1.3B.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1-T2V-1.3B.pth
echo    Downloading VAE...
curl -L -o models\vae\Wan2.1-VAE.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-VAE.pth
echo    Downloading LoRAs...
curl -L -o models\loras\Wan2.1-T2V-14B-LoRA.safetensors https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1-T2V-14B-LoRA.safetensors
goto create_launcher

:download_essential
echo    - Downloading Essential Models...
echo    Downloading Low-Res Model...
curl -L -o models\checkpoints\Wan2.1-T2V-1.3B.pth https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1-T2V-1.3B.pth
echo    Downloading VAE...
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
