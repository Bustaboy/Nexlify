# Windows Quick Start Guide - 1000-Round Training

## For Windows Users

Since you're on Windows, use the `.bat` file instead of `.sh`:

## Method 1: Using the Batch File (Easiest)

```cmd
cd C:\Nexlify
scripts\run_1000_training.bat
```

The batch file will:
- ✅ Check Python installation
- ✅ Create/activate virtual environment
- ✅ Install dependencies automatically
- ✅ Create necessary directories
- ✅ Start training with interactive prompts

## Method 2: Direct Python Command

```cmd
cd C:\Nexlify

REM Activate virtual environment (if you have one)
venv\Scripts\activate

REM Run training with default settings
python scripts\train_ml_rl_1000_rounds.py

REM Or with custom settings
python scripts\train_ml_rl_1000_rounds.py --agent-type adaptive --balance 10000 --data-days 180
```

## Method 3: PowerShell

```powershell
cd C:\Nexlify

# Activate virtual environment (if you have one)
.\venv\Scripts\Activate.ps1

# Run training
python scripts\train_ml_rl_1000_rounds.py
```

## Command-Line Options

```cmd
python scripts\train_ml_rl_1000_rounds.py --help

Options:
  --agent-type TYPE    Agent type: adaptive, ultra, basic (default: adaptive)
  --balance AMOUNT     Initial balance (default: 10000)
  --data-days DAYS     Days of historical data (default: 180)
  --symbol SYMBOL      Trading symbol (default: BTC/USDT)
  --resume FILE        Resume from checkpoint file
```

## Examples

### Train with defaults (Recommended)
```cmd
python scripts\train_ml_rl_1000_rounds.py
```

### Train with ultra-optimized agent
```cmd
python scripts\train_ml_rl_1000_rounds.py --agent-type ultra --balance 50000
```

### Resume from checkpoint
```cmd
python scripts\train_ml_rl_1000_rounds.py --resume models\ml_rl_1000\checkpoint_ep500.pth
```

### Train on Ethereum
```cmd
python scripts\train_ml_rl_1000_rounds.py --symbol ETH/USDT --checkpoint-dir models\eth_rl_1000
```

## Output Location

All outputs will be saved in: `C:\Nexlify\models\ml_rl_1000\`

- `best_model.pth` - Best performing model
- `final_model_1000.pth` - Final model after 1000 episodes
- `training_report_1000.png` - Visual report with 9 plots
- `training_summary_1000.txt` - Text summary
- `training_results_1000.json` - Complete training data

## View Results

### Open the visual report
```cmd
start models\ml_rl_1000\training_report_1000.png
```

### Read the summary
```cmd
type models\ml_rl_1000\training_summary_1000.txt
```

### View logs
```cmd
type logs\ml_rl_1000_training.log
```

## Troubleshooting

### "Python not found"
Install Python 3.9+ from [python.org](https://www.python.org/downloads/)
Make sure to check "Add Python to PATH" during installation.

### "Module not found" errors
```cmd
cd C:\Nexlify
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Out of Memory
Use the basic agent which requires less memory:
```cmd
python scripts\train_ml_rl_1000_rounds.py --agent-type basic
```

### PowerShell Execution Policy Error
If you get an error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Expected Training Time (Windows)

| Hardware | Approximate Time |
|----------|-----------------|
| CPU only (4 cores) | 12-16 hours |
| CPU + GPU (NVIDIA) | 3-6 hours |
| CPU + High-end GPU | 2-3 hours |

## Stopping and Resuming

To stop training: Press `Ctrl+C`

To resume from where you left off:
```cmd
REM Find the latest checkpoint
dir /b models\ml_rl_1000\checkpoint_*.pth

REM Resume from it
python scripts\train_ml_rl_1000_rounds.py --resume models\ml_rl_1000\checkpoint_ep500.pth
```

## Need Help?

Full documentation: `scripts\TRAINING_1000_ROUNDS.md`

Check logs: `logs\ml_rl_1000_training.log`
