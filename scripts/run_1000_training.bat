@echo off
REM ###############################################################################
REM Nexlify ML/RL 1000-Round Training Runner (Windows)
REM Quick start script for training the ML/RL agent
REM ###############################################################################

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo   Nexlify ML/RL 1000-Round Training
echo ================================================================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python %PYTHON_VERSION%

REM Check if in correct directory
if not exist "scripts\train_ml_rl_1000_rounds.py" (
    echo Error: Please run this script from the Nexlify root directory
    pause
    exit /b 1
)

REM Check/create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check dependencies
echo Checking dependencies...
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies (this may take a few minutes)...
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM Create necessary directories
echo Creating directories...
if not exist "models\ml_rl_1000" mkdir models\ml_rl_1000
if not exist "logs" mkdir logs
if not exist "data" mkdir data
echo [OK] Directories ready

REM Parse command line arguments
set AGENT_TYPE=adaptive
set BALANCE=10000
set DATA_DAYS=180
set SYMBOL=BTC/USDT
set RESUME=

:parse_args
if "%~1"=="" goto :end_parse
if "%~1"=="--agent-type" (
    set AGENT_TYPE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--balance" (
    set BALANCE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--data-days" (
    set DATA_DAYS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--symbol" (
    set SYMBOL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--resume" (
    set RESUME=--resume %~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    echo Usage: %~nx0 [OPTIONS]
    echo.
    echo Options:
    echo   --agent-type TYPE    Agent type: adaptive, ultra, basic (default: adaptive^)
    echo   --balance AMOUNT     Initial balance (default: 10000^)
    echo   --data-days DAYS     Days of historical data (default: 180^)
    echo   --symbol SYMBOL      Trading symbol (default: BTC/USDT^)
    echo   --resume FILE        Resume from checkpoint file
    echo   --help               Show this help message
    echo.
    echo Examples:
    echo   %~nx0                                    # Train with defaults
    echo   %~nx0 --agent-type ultra --balance 50000
    echo   %~nx0 --resume models\ml_rl_1000\checkpoint_ep500.pth
    pause
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
pause
exit /b 1

:end_parse

REM Display configuration
echo.
echo Training Configuration:
echo   Agent Type:      %AGENT_TYPE%
echo   Initial Balance: $%BALANCE%
echo   Data Days:       %DATA_DAYS%
echo   Symbol:          %SYMBOL%
if not "%RESUME%"=="" (
    echo   Resume From:     %RESUME:~9%
)
echo.

REM Confirm before starting
set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Training cancelled
    pause
    exit /b 0
)

REM Start training
echo.
echo Starting 1000-round training...
echo.
echo Note: This will take several hours. Progress is saved every 50 episodes.
echo You can safely interrupt with Ctrl+C and resume later.
echo.

python scripts\train_ml_rl_1000_rounds.py --agent-type %AGENT_TYPE% --balance %BALANCE% --data-days %DATA_DAYS% --symbol "%SYMBOL%" %RESUME%

REM Check if training completed successfully
if errorlevel 1 (
    echo.
    echo ================================================================================
    echo   Training Failed or Interrupted
    echo ================================================================================
    echo.
    echo Check logs: logs\ml_rl_1000_training.log
    echo.
    if exist "models\ml_rl_1000\checkpoint_ep50.pth" (
        echo Progress was saved. Resume with:
        for /f %%i in ('dir /b /o-d models\ml_rl_1000\checkpoint_ep*.pth 2^>nul') do (
            echo   %~nx0 --resume models\ml_rl_1000\%%i
            goto :found_checkpoint
        )
        :found_checkpoint
    )
    pause
    exit /b 1
)

REM Training completed successfully
echo.
echo ================================================================================
echo   Training Completed Successfully!
echo ================================================================================
echo.
echo Results available in: models\ml_rl_1000\
echo   - Best model: best_model.pth
echo   - Final model: final_model_1000.pth
echo   - Report: training_report_1000.png
echo   - Summary: training_summary_1000.txt
echo.
echo View the report:
echo   start models\ml_rl_1000\training_report_1000.png  (Windows)
echo.
echo Read the summary:
echo   type models\ml_rl_1000\training_summary_1000.txt
echo.

pause
