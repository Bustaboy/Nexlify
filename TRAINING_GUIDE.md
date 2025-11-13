# Nexlify ML/RL Training Quick Start Guide

## Prerequisites

1. **Python 3.9+** with all dependencies installed
2. **Historical data access** (fetched automatically from exchanges)
3. **GPU recommended** (but CPU works too)

Check your setup:
```bash
python --version
pip install -r requirements.txt
```

---

## Step 0: Run Tests First! (CRITICAL)

**ALWAYS run the test suite before starting training** to catch issues early:

```bash
# Quick test (30 seconds) - TESTS WHICH EXCHANGES WORK!
python test_training_pipeline.py --quick

# Full tests with coverage (2-3 minutes)
python test_training_pipeline.py --coverage

# Or use pytest with HTML coverage report
pip install pytest pytest-cov
pytest test_training_pipeline.py --cov=. --cov-report=html --cov-report=term
```

This validates:
- All dependencies are installed
- **Exchange connectivity works (critical!)** - Tests Kraken, Coinbase, Binance, etc.
- GPU is detected (if available)
- Agent and environment can be created
- Training loop functions properly
- Model save/load works

**If tests fail, DO NOT proceed to training!** Fix the issues first.

### ⚠️ Binance Geo-Blocking

**If you see "restricted location" errors:**

Binance blocks users from USA and other regions. **Solution: Use Kraken instead!**

```bash
# Use Kraken (works globally)
python train_ultimate_full_pipeline.py --exchange kraken --pairs BTC/USD ETH/USD --automated
```

See [GEO_BLOCKING_GUIDE.md](GEO_BLOCKING_GUIDE.md) for complete solutions.

See [TESTING.md](TESTING.md) for detailed testing documentation.

---

## Quick Start (Recommended)

### Option 1: Quick Test (Fast - ~30 minutes)
Perfect for testing the system:
```bash
python train_ultimate_full_pipeline.py --quick-test
```

### Option 2: Full Production Training (24+ hours)
Best for actual trading models:
```bash
python train_ultimate_full_pipeline.py \
    --pairs BTC/USDT ETH/USDT SOL/USDT \
    --initial-episodes 500 \
    --initial-runs 3 \
    --automated
```

### Option 3: Historical Data Training with Auto-Retrain
Trains with extensive historical data and retrains automatically:
```bash
python train_with_historical_data.py --symbol BTC/USDT --years 5
```

---

## Training Scripts Explained

### `train_ultimate_full_pipeline.py` (Most Comprehensive)

**What it includes:**
- All ML/RL best practices (Double DQN, Dueling DQN, PER, etc.)
- Multi-start initialization (trains multiple models, picks best)
- Full Nexlify risk management
- Multi-strategy trading (spot, staking, DeFi, arbitrage)

**Common options:**
```bash
# Basic usage
python train_ultimate_full_pipeline.py

# Custom pairs and episodes
python train_ultimate_full_pipeline.py \
    --pairs BTC/USDT ETH/USDT \
    --initial-episodes 1000 \
    --initial-runs 5

# Skip pre-flight checks (faster but risky)
python train_ultimate_full_pipeline.py --skip-preflight --automated
```

### `train_with_historical_data.py` (Auto-Retraining)

**What it includes:**
- Fetches historical data from exchanges
- Enriches data with external features (sentiment, on-chain)
- Automatic retraining until improvements plateau
- Curriculum learning (easy → hard progression)

**Common options:**
```bash
# Basic usage (5 years of BTC data)
python train_with_historical_data.py --symbol BTC/USDT --years 5

# Quick test (1 year, 3 iterations)
python train_with_historical_data.py --quick-test

# Custom threshold and patience
python train_with_historical_data.py \
    --symbol ETH/USDT \
    --years 3 \
    --threshold 2.0 \
    --patience 5 \
    --max-iterations 15
```

---

## Key Parameters

### Risk Management
```bash
--stop-loss 0.02        # 2% stop loss
--take-profit 0.05      # 5% take profit
--trailing-stop 0.03    # 3% trailing stop
--max-position 0.05     # 5% max position size
--max-trades 3          # Max 3 concurrent trades
--no-kelly              # Disable Kelly Criterion
```

### Training Control
```bash
--initial-runs 3        # Number of independent training runs
--initial-episodes 500  # Episodes per run
--years 5               # Years of historical data
--balance 10000         # Initial balance ($10,000)
```

### Output & Automation
```bash
--output ./my_training  # Custom output directory
--automated             # Skip all prompts
--skip-preflight        # Skip validation checks
--quick-test            # Fast test mode
```

---

## What Happens During Training

1. **Pre-flight Checks** - Validates exchange connectivity, data availability
2. **Data Fetching** - Downloads historical OHLCV data from exchanges
3. **Data Enrichment** - Adds Fear & Greed, on-chain, sentiment data
4. **Multi-Start Training** - Runs multiple training sessions with different seeds
5. **Model Selection** - Evaluates all models and picks the best
6. **Validation** - Tests on held-out validation data
7. **Auto-Retraining** (if using historical data script) - Retrains until improvements plateau

**Progress indicators:**
```
Ep 100/500 | Return: +12.34% | Equity: $11,234 | Trades: 45 | Sharpe: 1.85 | DD: 3.2%
```

---

## Output Files

After training completes:

```
training_output/
├── best_model/
│   ├── best_model_iter3_score85.2.pt     # Best model weights
│   └── best_model_metadata.json           # Model info
├── initial_runs/                          # All training runs
│   ├── run_1/
│   ├── run_2/
│   └── run_3/
├── training_summary.json                  # Complete summary
├── final_training_report.json             # Performance metrics
└── ultimate_training.log                  # Detailed logs
```

**Key files to check:**
- `best_model/*.pt` - Load this in production
- `training_summary.json` - Overall performance
- `ultimate_training.log` - Detailed training logs

---

## Next Steps After Training

1. **Review Results**
   ```bash
   cat training_output/training_summary.json
   ```

2. **Check Best Model**
   - Score should be positive
   - Sharpe ratio > 1.0 is good
   - Drawdown < 15% is acceptable

3. **Paper Trading** (Recommended)
   - Test model with fake money first
   - Run for at least 1 week
   - Monitor performance

4. **Deploy to Production** (Carefully!)
   - Start with small position sizes
   - Enable all risk controls
   - Monitor closely

---

## Troubleshooting

### Training is too slow
- Use `--quick-test` for testing
- Reduce `--initial-runs` or `--initial-episodes`
- Use fewer trading pairs

### Out of memory
- Reduce `--years` (less historical data)
- Lower batch size in config
- Train on single pair first

### Pre-flight checks fail
- Check internet connection
- Verify exchange API access
- Use `--skip-preflight` to bypass (not recommended)

### No improvement after iterations
- This is normal - training stops automatically
- Best model is still saved
- Try different hyperparameters

---

## Advanced Usage

### GPU Acceleration
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Training automatically uses GPU if available
python train_ultimate_full_pipeline.py  # Uses GPU automatically
```

### Custom Configuration
Edit training parameters in the script or create your own config:
```python
# In train_ultimate_full_pipeline.py
# Modify create_agent_config() for custom hyperparameters
```

### Multi-Exchange Training
```bash
# Train on different exchanges
python train_with_historical_data.py --exchange binance --symbol BTC/USDT
python train_with_historical_data.py --exchange kraken --symbol BTC/USD
```

---

## Example Commands Cheat Sheet

```bash
# Fastest test
python train_ultimate_full_pipeline.py --quick-test

# Best for production (BTC only, Binance)
python train_ultimate_full_pipeline.py --pairs BTC/USDT --initial-runs 5 --automated

# Best for production (Kraken - if Binance is geo-blocked)
python train_ultimate_full_pipeline.py --exchange kraken --pairs BTC/USD --initial-runs 5 --automated

# Multi-pair production (Binance)
python train_ultimate_full_pipeline.py --pairs BTC/USDT ETH/USDT SOL/USDT --initial-episodes 1000 --automated

# Multi-pair production (Kraken - works globally)
python train_ultimate_full_pipeline.py --exchange kraken --pairs BTC/USD ETH/USD SOL/USD --initial-episodes 1000 --automated

# Historical data with auto-retrain
python train_with_historical_data.py --symbol BTC/USDT --years 5 --automated

# Conservative risk settings
python train_ultimate_full_pipeline.py --exchange kraken --pairs BTC/USD --stop-loss 0.01 --take-profit 0.03 --max-position 0.03 --automated
```

---

## Support

- Check logs: `tail -f ultimate_training.log`
- Review docs: `README.md`
- Examine training output: `training_output/`

**Remember:** Always test models in paper trading before live deployment!
