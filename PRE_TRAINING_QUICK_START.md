# Nexlify Pre-Training Quick Start Guide

## Before Every Training Session or App Launch

### Quick Validation (30 seconds)

Run the comprehensive pre-training checklist:

```bash
python pre_training_checklist.py
```

This will check:
- ✓ Python version (3.9+ required)
- ✓ All dependencies installed
- ✓ Directory structure
- ✓ Configuration files
- ✓ Training data availability
- ✓ Disk space
- ✓ GPU availability
- ✓ Internet & Exchange API
- ✓ Database connectivity

### Alternative: Run Existing Preflight Checker

```bash
python nexlify_preflight_checker.py --symbol BTC/USDT --exchange binance --automated
```

## Critical Pre-Training Requirements

### 1. Dependencies Installed
```bash
pip install -r requirements.txt
```

**Key packages:**
- ccxt (exchange connectivity)
- pandas, numpy (data processing)
- tensorflow, torch (deep learning)
- scikit-learn, xgboost (ML algorithms)
- sqlalchemy (database)
- aiohttp (async operations)

### 2. Directory Structure
Ensure these directories exist (auto-created if missing):
```
Nexlify/
├── config/              # Configuration files
├── data/               # Training and market data
│   ├── historical_cache/
│   ├── external_cache/
│   └── sample_datasets/
├── models/             # Trained ML/RL models
├── logs/               # System logs
│   └── crash_reports/
├── backups/            # System backups
└── reports/            # Performance reports
    └── tax/
```

### 3. Configuration Files
Copy example config if needed:
```bash
cp config/neural_config.example.json config/neural_config.json
```

Edit `config/neural_config.json` to customize:
- Exchange API keys (for live trading only)
- Risk management settings
- Trading parameters
- Circuit breaker configuration

**Note:** API keys NOT required for training!

### 4. Training Data Options

**Option A: Use Sample Data** (Fastest, offline)
```
data/sample_datasets/btc_usdt_raw.csv (1,440 rows)
data/sample_datasets/btc_usdt_quick_test.csv (200 rows)
```

**Option B: Fetch Live Data** (Requires internet)
- Training scripts auto-fetch from exchanges
- Data cached in `data/historical_cache/`

**Option C: Use Cached Data**
- Previously downloaded data in `data/historical_cache/`

## Training Scripts Available

### 1. Ultimate Training Pipeline (RECOMMENDED)
```bash
python train_ultimate_full_pipeline.py
```
**Features:**
- Most comprehensive training system
- ALL ML/RL best practices included
- Double DQN, Dueling DQN, PER, PPO, SAC
- Multi-start initialization
- Auto data fetching
- Pre-flight validation

### 2. Auto-Retraining System
```bash
python train_complete_with_auto_retrain.py
```

### 3. Historical Data Training
```bash
python train_with_historical_data.py
```

### 4. Basic RL Training
```bash
python scripts/train_rl_agent.py
```

### 5. Extended Training (1000 rounds)
```bash
python scripts/train_ml_rl_1000_rounds.py
```

## Running the Application

### GUI Launcher
```bash
python scripts/nexlify_launcher.py
```
**Starts:**
- Neural net API
- Cyberpunk GUI
- Full system validation

### Paper Trading (CLI)
```bash
python run_paper_trading.py
```

## Resource Requirements

### Minimum System Requirements:
- **Python:** 3.9+
- **RAM:** 4GB (8GB+ recommended)
- **Disk:** 2GB free (5GB+ recommended)
- **CPU:** 4+ cores recommended
- **Internet:** Optional (required for live data)

### Optional but Recommended:
- **GPU:** NVIDIA CUDA-compatible (10-50x faster training)
- **RAM:** 8GB+ for large datasets
- **Disk:** 10GB+ for extensive caching

## Common Issues & Solutions

### Issue: Missing Dependencies
```bash
# Solution:
pip install -r requirements.txt
```

### Issue: No Training Data
```bash
# Solutions:
# 1. Use sample data (already included)
# 2. Enable internet for auto-fetch
# 3. Use cached data if available
```

### Issue: Config File Missing
```bash
# Solution:
cp config/neural_config.example.json config/neural_config.json
```

### Issue: Permission Errors
```bash
# Solution:
chmod +x *.py
chmod +x scripts/*.py
```

### Issue: Database Errors
```bash
# Solution:
rm data/trading.db  # Will be auto-recreated
python scripts/setup_nexlify.py
```

### Issue: Slow Training (No GPU)
```
# This is normal - CPU training is 10-50x slower
# Consider:
# 1. Using smaller datasets
# 2. Reducing training rounds
# 3. Using cloud GPU (Google Colab, AWS, etc.)
```

## Performance Optimization

### For Faster Training:
1. **Use GPU** - CUDA-compatible NVIDIA GPU
2. **Increase RAM** - 8GB+ recommended
3. **Use Cached Data** - Avoid repeated downloads
4. **Reduce Dataset Size** - For quick experiments
5. **Parallel Processing** - Leverages multi-core CPUs

### GPU Check:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Safety Checklist Before Live Trading

⚠️ **NEVER use real money without thorough testing!**

- [ ] Extensive backtesting completed
- [ ] Paper trading validated (1-2 weeks minimum)
- [ ] Risk management configured
- [ ] Circuit breakers enabled
- [ ] Stop losses configured
- [ ] Position sizing appropriate
- [ ] Emergency stop tested
- [ ] 2FA enabled (if configured)
- [ ] API keys have withdrawal restrictions
- [ ] Small position size for first live trades

## Getting Help

### Check Logs:
```bash
# Recent logs
tail -f logs/*.log

# Crash reports
ls -la logs/crash_reports/
```

### Run Diagnostics:
```bash
python pre_training_checklist.py
python nexlify_preflight_checker.py --automated
```

### Documentation:
- `README.md` - General overview
- `TRAINING_GUIDE.md` - Comprehensive training guide
- `ULTIMATE_PIPELINE_DOCUMENTATION.md` - Ultimate pipeline docs
- `ML_RL_BEST_PRACTICES.md` - ML/RL best practices

## Quick Start Workflow

```bash
# 1. Verify setup
python pre_training_checklist.py

# 2. Train the model (first time)
python train_ultimate_full_pipeline.py

# 3. Test with paper trading
python run_paper_trading.py

# 4. Launch GUI (if satisfied with results)
python scripts/nexlify_launcher.py
```

## Environment Variables (Optional)

Create `.env` file (optional, managed by GUI):
```env
EXCHANGE_API_KEY=your_key_here
EXCHANGE_API_SECRET=your_secret_here
DATABASE_URL=sqlite:///data/trading.db
```

**Note:** Not required for training! Only for live trading.

## Next Steps

1. Run `python pre_training_checklist.py`
2. Review any warnings or errors
3. Start with ultimate training pipeline
4. Monitor logs during training
5. Evaluate results
6. Proceed to paper trading
7. Never skip paper trading phase!

---

**Remember:** Always start with paper trading. Never risk real money without extensive testing!
