@echo off
setlocal
title Z-Image-Turbo WSL2 Installer

echo ===============================================================================
echo  Z-IMAGE-TURBO WINDOWS (WSL2) BOOTSTRAPPER
echo ===============================================================================
echo.
echo This script will help you set up Z-Image-Turbo inside WSL2.
echo It automates the setup guide found in docs/setup-guides/Windows-WSL2.md.
echo.

:: -----------------------------------------------------------------------------
:: 1. CHECK ADMIN PRIVILEGES
:: -----------------------------------------------------------------------------
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges.
) else (
    echo [ERROR] This script requires Administrator privileges.
    echo         Right-click 'setup_windows.bat' and select 'Run as Administrator'.
    pause
    exit /b
)

:: -----------------------------------------------------------------------------
:: 2. CHECK WSL STATUS
:: -----------------------------------------------------------------------------
echo.
echo Checking WSL2 status...
wsl --status >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] WSL is installed.
) else (
    echo [WARNING] WSL is NOT installed or not working.
    echo.
    echo We can install WSL2 with Ubuntu 22.04 for you now.
    echo This will require a restart of your computer afterwards.
    echo.
    set /p install_wsl="Install WSL2 now? (y/N) "
    if /i "%install_wsl%"=="y" (
        echo Installing WSL...
        wsl --install Ubuntu-22.04
        echo.
        echo [IMPORTANT] Installation started. Please RESTART your computer when asked.
        echo After restart, run this script again to continue setup.
        pause
        exit /b
    ) else (
        echo WSL is required. Exiting.
        pause
        exit /b
    )
)

:: -----------------------------------------------------------------------------
:: 3. PREPARE LINUX SCRIPT
:: -----------------------------------------------------------------------------
echo.
echo Preparing Linux installer...

if not exist "setup.sh" (
    echo [ERROR] setup.sh not found! Please make sure you extracted all files.
    pause
    exit /b
)

:: Fix line endings (CRLF -> LF) just in case
:: We use 'tr' inside WSL to create a temporary clean copy
echo Converting script format...
wsl tr -d '\r' ^< setup.sh ^> setup_wsl.sh
wsl chmod +x setup_wsl.sh

:: -----------------------------------------------------------------------------
:: 4. RUN LINUX SCRIPT
:: -----------------------------------------------------------------------------
echo.
echo Handing off to WSL2...
echo You will be prompted for your sudo password inside the Linux environment.
echo.
echo -------------------------------------------------------------------------------
wsl ./setup_wsl.sh

:: Cleanup
wsl rm setup_wsl.sh

echo.
echo ===============================================================================
echo  WINDOWS SETUP COMPLETE
echo ===============================================================================
echo  If the Linux setup reported success, you are ready to go!
echo.
pause
