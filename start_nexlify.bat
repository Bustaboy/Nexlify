@echo off
REM Nexlify - Arasaka Neural-Net Trading Matrix
REM Cyberpunk-themed cryptocurrency arbitrage system

cd /d "%~dp0"

echo.
echo  ███╗   ██╗███████╗██╗  ██╗██╗     ██╗███████╗██╗   ██╗
echo  ████╗  ██║██╔════╝╚██╗██╔╝██║     ██║██╔════╝╚██╗ ██╔╝
echo  ██╔██╗ ██║█████╗   ╚███╔╝ ██║     ██║█████╗   ╚████╔╝ 
echo  ██║╚██╗██║██╔══╝   ██╔██╗ ██║     ██║██╔══╝    ╚██╔╝  
echo  ██║ ╚████║███████╗██╔╝ ██╗███████╗██║██║        ██║   
echo  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝        ╚═╝   
echo.
echo            ARASAKA NEURAL-NET TRADING MATRIX
echo                    [ Version 2.0.7.7 ]
echo.
echo ========================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher
    pause
    exit /b 1
)

REM Check for emergency stop
if exist EMERGENCY_STOP_ACTIVE (
    echo WARNING: Emergency stop detected!
    echo Delete EMERGENCY_STOP_ACTIVE to continue
    pause
    exit /b 1
)

echo [1/3] Initializing Nexlify Trading Matrix...
timeout /t 2 /nobreak >nul

echo.
echo [2/3] Starting Arasaka Neural-Net Engine...
start "Nexlify Neural-Net" /min cmd /k python arasaka_neural_net.py

timeout /t 5 /nobreak >nul

echo.
echo [3/3] Launching Cyberpunk Interface...
start "Nexlify GUI" cmd /k python cyber_gui.py

echo.
echo ========================================================
echo.
echo  [+] Nexlify Trading Matrix ONLINE
echo  [+] Neural-Net: ACTIVE
echo  [+] GUI Interface: READY
echo.
echo  Default PIN: 2077
echo.
echo  To stop: Close all windows or use KILL SWITCH in GUI
echo.
echo ========================================================
echo.
echo Welcome to the future of trading. Welcome to Nexlify.
echo.
pause