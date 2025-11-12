@echo off
REM ###############################################################################
REM Nexlify ML/RL 1000-Round Training Runner (Windows)
REM Quick start script for training the ML/RL agent
REM ###############################################################################

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo   [94m^🚀 Nexlify ML/RL 1000-Round Training[0m
echo ================================================================================
echo.

REM Check Python version
echo [93mChecking Python version...[0m
python --version >nul 2>&1
if errorlevel 1 (
    echo [91mError: Python not found. Please install Python 3.9+[0m
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [92m✓ Python %PYTHON_VERSION%[0m

REM Check if in correct directory
if not exist "scripts\train_ml_rl_1000_rounds.py" (
    echo [91mError: Please run this script from the Nexlify root directory[0m
    pause
    exit /b 1
)

REM Check/create virtual environment
if not exist "venv" (
    echo [93mCreating virtual environment...[0m
    python -m venv venv
    echo [92m✓ Virtual environment created[0m
)

REM Activate virtual environment
echo [93mActivating virtual environment...[0m
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [91mError: Failed to activate virtual environment[0m
    pause
    exit /b 1
)

REM Check dependencies
echo [93mChecking dependencies...[0m
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [93mInstalling dependencies (this may take a few minutes)...[0m
    python -m pip install --upgrade pip >nul 2>&1
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [91mError: Failed to install dependencies[0m
        pause
        exit /b 1
    )
    echo [92m✓ Dependencies installed[0m
) else (
    echo [92m✓ Dependencies already installed[0m
)

REM Create necessary directories
echo [93mCreating directories...[0m
if not exist "models\ml_rl_1000" mkdir models\ml_rl_1000
if not exist "logs" mkdir logs
if not exist "data" mkdir data
echo [92m✓ Directories ready[0m

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
echo [91mUnknown option: %~1[0m
echo Use --help for usage information
pause
exit /b 1

:end_parse

REM Display configuration
echo.
echo [94mTraining Configuration:[0m
echo   Agent Type:      [92m%AGENT_TYPE%[0m
echo   Initial Balance: [92m$%BALANCE%[0m
echo   Data Days:       [92m%DATA_DAYS%[0m
echo   Symbol:          [92m%SYMBOL%[0m
if not "%RESUME%"=="" (
    echo   Resume From:     [92m%RESUME:~9%[0m
)
echo.

REM Confirm before starting
set /p CONFIRM="Start training? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo [93mTraining cancelled[0m
    pause
    exit /b 0
)

REM Start training
echo.
echo [92mStarting 1000-round training...[0m
echo.
echo [93mNote: This will take several hours. Progress is saved every 50 episodes.[0m
echo [93mYou can safely interrupt with Ctrl+C and resume later.[0m
echo.

python scripts\train_ml_rl_1000_rounds.py --agent-type %AGENT_TYPE% --balance %BALANCE% --data-days %DATA_DAYS% --symbol "%SYMBOL%" %RESUME%

REM Check if training completed successfully
if errorlevel 1 (
    echo.
    echo [91m================================================================================[0m
    echo [91m  ❌ Training Failed or Interrupted[0m
    echo [91m================================================================================[0m
    echo.
    echo Check logs: [93mlogs\ml_rl_1000_training.log[0m
    echo.
    if exist "models\ml_rl_1000\checkpoint_ep50.pth" (
        echo Progress was saved. Resume with:
        for /f %%i in ('dir /b /o-d models\ml_rl_1000\checkpoint_ep*.pth 2^>nul') do (
            echo   [94m%~nx0 --resume models\ml_rl_1000\%%i[0m
            goto :found_checkpoint
        )
        :found_checkpoint
    )
    pause
    exit /b 1
)

REM Training completed successfully
echo.
echo [92m================================================================================[0m
echo [92m  ✅ Training Completed Successfully![0m
echo [92m================================================================================[0m
echo.
echo Results available in: [92mmodels\ml_rl_1000\[0m
echo   - Best model: [92mbest_model.pth[0m
echo   - Final model: [92mfinal_model_1000.pth[0m
echo   - Report: [92mtraining_report_1000.png[0m
echo   - Summary: [92mtraining_summary_1000.txt[0m
echo.
echo View the report:
echo   [94mstart models\ml_rl_1000\training_report_1000.png[0m  (Windows)
echo.
echo Read the summary:
echo   [94mtype models\ml_rl_1000\training_summary_1000.txt[0m
echo.

pause
