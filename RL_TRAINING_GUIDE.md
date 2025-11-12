# RL Model Training Guide - Run on Your Laptop

## Overview
This guide will help you train the Nexlify RL trading agent on your local machine. Training takes approximately **1 hour** for 1000 episodes on a modern CPU.

---

## Prerequisites

### Required Software
- Python 3.11+
- pip (Python package manager)
- 2GB+ free RAM
- 1GB free disk space

---

## Step 1: Install Dependencies

Open a terminal and run:

```bash
cd /path/to/Nexlify

# Install required packages
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib ccxt psutil
```

**Note:** We use CPU-only PyTorch (184MB) instead of CUDA version (2GB+) for faster installation.

---

## Step 2: Verify Files Exist

Make sure these files are present:
- `train_rl_agent.py` - Training script
- `nexlify_rl_agent.py` - DQN agent and trading environment
- `error_handler.py` - Error handling utilities

---

## Step 3: Create Required Directories

```bash
mkdir -p logs
mkdir -p models
mkdir -p data
```

---

## Step 4: Run Training

### Option A: Full Training (1000 episodes, ~1 hour)

```bash
python train_rl_agent.py
```

### Option B: Quick Test (100 episodes, ~6 minutes)

Edit `train_rl_agent.py` and change line 65:
```python
'n_episodes': 100,  # Changed from 1000
```

Then run:
```bash
python train_rl_agent.py
```

---

## Step 5: Monitor Progress

The training will display progress every 10 episodes:

```
Episode 10/1000
  Avg Reward: +0.0681
  Avg Profit: +34.46%
  Avg Win Rate: 54.7%
  Trades: 168
  Epsilon: 0.951
--------------------------------------------------
```

**Key Metrics:**
- **Avg Profit**: Target 30%+ (good performance)
- **Win Rate**: Target 52%+ (better than random)
- **Epsilon**: Starts at 1.0, decreases to 0.01 (less exploration over time)

---

## Step 6: Checkpoints

Models are automatically saved every 50 episodes:
- `models/rl_agent_checkpoint_ep50.pth`
- `models/rl_agent_checkpoint_ep100.pth`
- `models/rl_agent_checkpoint_ep150.pth`
- etc.

**If training is interrupted**, these checkpoints can be used!

---

## Step 7: Training Completion

When complete, you'll see:

```
✅ TRAINING COMPLETE
```

**Generated Files:**
1. `models/rl_agent_trained.pth` - Final trained model
2. `models/training_report.png` - Performance graphs
3. `models/training_summary.json` - Statistics
4. `logs/rl_training.log` - Full training log

---

## Step 8: Verify the Model

```bash
python -c "
import torch
model = torch.load('models/rl_agent_trained.pth')
print('✅ Model loaded successfully!')
print(f'Epsilon: {model['epsilon']:.4f}')
print(f'Training episodes: {len(model['training_history'])}')
"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"
**Solution:** Install the missing package:
```bash
pip install X
```

### Issue: Training is very slow
**Solution:**
- Close other applications
- Use fewer episodes for testing (100 instead of 1000)
- Expected speed: ~2-3 seconds per episode

### Issue: "Error fetching data from Binance"
**Solution:** This is expected! The script will automatically use synthetic data for training. This is actually better for consistent training.

### Issue: Out of memory
**Solution:**
- Close other applications
- Reduce batch_size in `train_rl_agent.py` line 75:
  ```python
  'batch_size': 32,  # Changed from 64
  ```

---

## Understanding the Training

### What is being trained?
A **Deep Q-Network (DQN)** that learns to:
- **BUY** when it predicts price will go up
- **SELL** when it has a position and wants to take profit
- **HOLD** when uncertain

### What data is used?
- **Synthetic BTC price data** (4,320 hourly price points = 180 days)
- Includes realistic price movements, trends, and volatility
- Generated with random walk + trend

### What makes a good model?
- **Avg Profit > 30%** per episode
- **Win Rate > 52%** (better than coin flip)
- **Decreasing Epsilon** (less random, more strategic)
- **Consistent performance** in last 100 episodes

---

## Advanced: Resume Training

If training stops, you can resume from the last checkpoint:

1. Edit `train_rl_agent.py`, add at line 163 (after creating agent):

```python
# Resume from checkpoint
checkpoint_path = "models/rl_agent_checkpoint_ep150.pth"
if os.path.exists(checkpoint_path):
    self.agent.load(checkpoint_path)
    logger.info(f"📥 Resumed from {checkpoint_path}")
```

2. Run training again

---

## Using the Trained Model

Once training is complete, enable it in your trading bot:

1. Edit `neural_config.json`:
```json
{
  "use_rl_agent": true,
  "rl_model_path": "models/rl_agent_trained.pth"
}
```

2. The model will now make trading decisions in your bot!

---

## Expected Training Timeline

| Episodes | Time | Progress |
|----------|------|----------|
| 10 | ~30 sec | 1% |
| 50 | ~2.5 min | 5% - First checkpoint |
| 100 | ~5 min | 10% - Second checkpoint |
| 500 | ~25 min | 50% - Halfway |
| 1000 | ~50 min | 100% - Complete |

---

## Questions?

- **Training logs**: Check `logs/rl_training.log`
- **Model files**: Check `models/` directory
- **Performance graphs**: Open `models/training_report.png` after completion

---

## Summary Checklist

- [ ] Install dependencies (torch, numpy, pandas, matplotlib, ccxt, psutil)
- [ ] Create directories (logs, models, data)
- [ ] Run `python train_rl_agent.py`
- [ ] Wait ~1 hour for 1000 episodes
- [ ] Verify `models/rl_agent_trained.pth` exists
- [ ] View results in `models/training_report.png`
- [ ] Enable in `neural_config.json`

**Good luck with your training!** 🚀
