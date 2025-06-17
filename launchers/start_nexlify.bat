@echo off
REM Enhanced Nexlify Startup Script for Windows
REM Addresses all V3 improvements

setlocal enabledelayedexpansion

REM Set window title
title Nexlify Trading Platform v2.0.8

REM Colors for output
REM Note: Using echo with escape sequences for better compatibility

echo ============================================
echo         NEXLIFY TRADING PLATFORM
echo            Night City Trader
echo               v2.0.8
echo ============================================
echo.

REM Check Python installation and version
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher from https://python.org
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Check minimum version (3.11)
if %PYTHON_MAJOR% LSS 3 (
    echo ERROR: Python 3.11 or higher required, found Python %PYTHON_VERSION%
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 11 (
    echo ERROR: Python 3.11 or higher required, found Python %PYTHON_VERSION%
    pause
    exit /b 1
)

echo Python %PYTHON_VERSION% detected - OK
echo.

REM Create logs directory if it doesn't exist
if not exist "logs" mkdir logs
if not exist "logs\startup" mkdir logs\startup

REM Set log files with timestamp
set TIMESTAMP=%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LAUNCHER_LOG=logs\startup\launcher_%TIMESTAMP%.log
set NEURAL_LOG=logs\startup\neural_net_%TIMESTAMP%.log
set GUI_LOG=logs\startup\gui_%TIMESTAMP%.log

echo Starting Nexlify components...
echo Logs will be saved to logs\startup\
echo.

REM Create PID file for process tracking
set PID_FILE=%TEMP%\nexlify_pids.txt
echo. > "%PID_FILE%"

REM Function to get process PID (using WMIC)
REM We'll start processes and track them for cleanup

REM Check if smart_launcher.py exists
if exist "smart_launcher.py" (
    echo Using smart_launcher.py for integrated startup...
    
    REM Start smart launcher with output redirection
    start "Nexlify Launcher" /B cmd /c "python smart_launcher.py > "%LAUNCHER_LOG%" 2>&1 & echo !errorlevel! > "%TEMP%\nexlify_exit_code.txt""
    
    REM Give it time to start
    timeout /t 3 /nobreak > nul
    
    REM Monitor the launcher
    :monitor_launcher
    tasklist /FI "WINDOWTITLE eq Nexlify Launcher" 2>NUL | find /I "cmd.exe" >NUL
    if "%errorlevel%"=="0" (
        timeout /t 5 /nobreak > nul
        goto monitor_launcher
    )
    
) else (
    REM Fallback to direct component startup
    echo smart_launcher.py not found, starting components directly...
    
    REM Start Neural Net with dynamic timeout
    echo Starting Neural Net Engine...
    start "Nexlify Neural Net" /B cmd /c "python -u src\nexlify_neural_net.py > "%NEURAL_LOG%" 2>&1"
    
    REM Wait for neural net to initialize (dynamic wait)
    set WAIT_COUNT=0
    :wait_neural
    if %WAIT_COUNT% GEQ 30 (
        echo WARNING: Neural net took too long to start, continuing anyway...
        goto start_gui
    )
    
    REM Check if port 8000 is listening (neural net API)
    netstat -an | findstr ":8000.*LISTENING" > nul 2>&1
    if errorlevel 1 (
        timeout /t 1 /nobreak > nul
        set /a WAIT_COUNT+=1
        echo Waiting for Neural Net API... %WAIT_COUNT%s
        goto wait_neural
    )
    
    echo Neural Net started successfully
    echo.
    
    :start_gui
    REM Start GUI
    echo Starting GUI...
    start "Nexlify GUI" /B cmd /c "python -u cyber_gui.py > "%GUI_LOG%" 2>&1"
    
    echo GUI started successfully
)

echo.
echo ============================================
echo Nexlify is running!
echo.
echo Press Ctrl+C to initiate graceful shutdown
echo or type 'stop' and press Enter
echo ============================================
echo.

REM Monitor for shutdown command or Ctrl+C
:monitor
set /p USER_INPUT=
if /i "%USER_INPUT%"=="stop" goto shutdown
if /i "%USER_INPUT%"=="exit" goto shutdown
if /i "%USER_INPUT%"=="quit" goto shutdown
timeout /t 1 /nobreak > nul
goto monitor

:shutdown
echo.
echo Initiating graceful shutdown...

REM Create emergency stop file (if components check for it)
echo STOP > EMERGENCY_STOP_ACTIVE

REM Send termination signal to Python processes
echo Stopping Neural Net...
taskkill /FI "WINDOWTITLE eq Nexlify Neural Net*" /T >nul 2>&1

echo Stopping GUI...
taskkill /FI "WINDOWTITLE eq Nexlify GUI*" /T >nul 2>&1

echo Stopping Launcher...
taskkill /FI "WINDOWTITLE eq Nexlify Launcher*" /T >nul 2>&1

REM Wait for processes to terminate gracefully
timeout /t 5 /nobreak > nul

REM Force kill if still running
taskkill /F /FI "WINDOWTITLE eq Nexlify*" /T >nul 2>&1

REM Clean up emergency stop file
if exist EMERGENCY_STOP_ACTIVE del EMERGENCY_STOP_ACTIVE

echo.
echo ============================================
echo Nexlify shutdown complete
echo.
echo Logs saved to:
echo - %LAUNCHER_LOG%
echo - %NEURAL_LOG%
echo - %GUI_LOG%
echo ============================================
echo.
pause

:end
endlocal
exit /b 0
