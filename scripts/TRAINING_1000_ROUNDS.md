# ML/RL 1000-Round Training Guide

## Overview

This guide explains how to use the `train_ml_rl_1000_rounds.py` script to train the Nexlify ML/RL agent for exactly 1000 episodes.

## Features

✨ **Comprehensive Training:**
- Exactly 1000 training episodes
- Automatic hardware detection and optimization
- Multiple agent types (Adaptive, Ultra-Optimized, Basic)
- Real-time progress tracking with ETA

💾 **Checkpointing:**
- Automatic checkpoints every 50 episodes
- Best model tracking (highest profit)
- Resume from checkpoint support
- All checkpoints saved with episode numbers

📊 **Detailed Reporting:**
- Visual training report with 9 plots
- Text summary with statistics
- JSON results for analysis
- Hardware profile saved

🎯 **Performance Tracking:**
- Profit tracking ($ and %)
- Reward accumulation
- Win rate monitoring
- Trade statistics
- Loss tracking
- Epsilon decay

## Quick Start

### Basic Usage

```bash
# Train with default settings (adaptive agent, 180 days data, $10,000 balance)
python scripts/train_ml_rl_1000_rounds.py
```

### Advanced Usage

```bash
# Use ultra-optimized agent with custom balance
python scripts/train_ml_rl_1000_rounds.py \
  --agent-type ultra \
  --balance 50000 \
  --data-days 365

# Resume from checkpoint
python scripts/train_ml_rl_1000_rounds.py \
  --resume models/ml_rl_1000/checkpoint_ep500.pth

# Train on different symbol
python scripts/train_ml_rl_1000_rounds.py \
  --symbol ETH/USDT \
  --checkpoint-dir models/eth_rl_1000
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--agent-type` | `adaptive` | Agent type: `adaptive`, `ultra`, or `basic` |
| `--data-days` | `180` | Days of historical data to fetch |
| `--balance` | `10000` | Initial trading balance |
| `--checkpoint-dir` | `models/ml_rl_1000` | Directory for checkpoints |
| `--resume` | `None` | Path to checkpoint file to resume from |
| `--symbol` | `BTC/USDT` | Trading symbol |

## Agent Types

### Adaptive Agent (Recommended)
- Automatically detects hardware capabilities
- Optimizes model size, batch size, and buffer size
- Supports mixed precision on compatible GPUs
- Best for most users

```bash
python scripts/train_ml_rl_1000_rounds.py --agent-type adaptive
```

### Ultra-Optimized Agent
- GPU-specific optimizations (NVIDIA Tensor Cores, AMD)
- Multi-GPU support
- Thermal monitoring
- Smart caching with compression
- Best for high-end systems

```bash
python scripts/train_ml_rl_1000_rounds.py --agent-type ultra
```

### Basic Agent
- Standard DQN implementation
- No hardware optimization
- Good for testing or low-resource systems

```bash
python scripts/train_ml_rl_1000_rounds.py --agent-type basic
```

## Output Files

After training completes, you'll find these files in the checkpoint directory:

```
models/ml_rl_1000/
├── best_model.pth                    # Best performing model
├── final_model_1000.pth              # Final model after 1000 episodes
├── checkpoint_ep50.pth               # Checkpoint at episode 50
├── checkpoint_ep100.pth              # Checkpoint at episode 100
├── ...                               # More checkpoints every 50 episodes
├── training_results_1000.json        # Complete training data
├── training_report_1000.png          # Visual report (9 plots)
├── training_summary_1000.txt         # Text summary
├── hardware_profile.json             # Hardware configuration used
└── training_data.npy                 # Price data used for training
```

## Understanding the Results

### Visual Report (`training_report_1000.png`)

The report contains 9 plots:

1. **Profit % per Episode**: Individual episode profitability
2. **Cumulative Reward**: Total reward accumulated over time
3. **Exploration Rate**: Epsilon decay (exploration vs exploitation)
4. **Moving Average Profit**: Smoothed profit trend (50-episode window)
5. **Training Loss**: Neural network training loss
6. **Trade Win Rate**: Percentage of profitable trades
7. **Trades per Episode**: Trading frequency
8. **Profit Distribution**: Histogram of episode profits
9. **Reward Distribution**: Histogram of episode rewards

### Text Summary (`training_summary_1000.txt`)

Contains:
- Hardware configuration used
- Overall statistics (best, average, median, std dev)
- Profitability metrics
- Win rate and trade statistics
- Last 100 episodes performance

### JSON Results (`training_results_1000.json`)

Machine-readable data including:
- All episode numbers
- All rewards, profits, and losses
- Win rates and trade counts
- Timestamps for each episode
- Best performance metrics

## Training Progress

During training, you'll see detailed progress every 10 episodes:

```
================================================================================
Episode 100/1000 (10.0% complete)
--------------------------------------------------------------------------------
Current Episode:
  Profit: $  +234.56 ( +2.35%)
  Reward:    1234.56
  Loss:   0.0234
  Trades:  12 (Win Rate: 58.3%)
  Epsilon: 0.6050
Recent Performance (last 10 episodes):
  Avg Profit: $  +189.23 ( +1.89%)
  Avg Reward:    1156.78
  Avg Win Rate: 55.6%
Best Performance:
  Best Profit: $  +456.78 ( +4.57%)
Progress:
  Elapsed: 0:45:23
  ETA: 6.8 hours
  Ep Time: 2.72s
System Performance:
  Batch Time: 12.3ms
  Memory Usage: 1.45 GB
  Buffer Size: 50,000
================================================================================
```

## Resuming Training

If training is interrupted, you can resume from any checkpoint:

```bash
# Resume from the last checkpoint
python scripts/train_ml_rl_1000_rounds.py \
  --resume models/ml_rl_1000/checkpoint_ep500.pth
```

The script will:
- Load the model state
- Continue from the checkpoint episode number
- Preserve all training progress

## Best Practices

### 1. Data Quality
- Use at least 180 days of data for robust training
- More data (365+ days) can improve performance
- The script auto-fetches from Binance or generates synthetic data if offline

### 2. Hardware Optimization
- Use `adaptive` agent type for automatic optimization
- Ensure GPU drivers are up to date for CUDA/ROCm support
- Close other applications to free up memory

### 3. Monitoring
- Check logs in `logs/ml_rl_1000_training.log`
- Monitor the visual report after every 50 episodes
- Watch for:
  - Increasing profit trend
  - Decreasing loss
  - Stable or improving win rate
  - Epsilon approaching minimum (0.01)

### 4. Checkpointing
- Don't delete checkpoints during training
- Keep best_model.pth for production use
- Compare different checkpoints if final model underperforms

### 5. Post-Training
- Review the complete training report
- Check last 100 episodes performance
- Test the best model in paper trading before live use

## Troubleshooting

### Out of Memory
```bash
# Use smaller batch size with basic agent
python scripts/train_ml_rl_1000_rounds.py --agent-type basic
```

### Slow Training
- Ensure GPU is being utilized (check hardware profile)
- Reduce data days if memory-limited
- Use SSD instead of HDD for better I/O

### Poor Performance
- Try longer training (data will continue from 1000)
- Increase data days for more diverse scenarios
- Adjust initial balance to match your trading size

### Data Fetch Errors
- Check internet connection
- Script will auto-generate synthetic data as fallback
- Pre-download data and save as .npy file

## Next Steps

After training completes:

1. **Evaluate the Model**
   ```bash
   # Compare best vs final model
   ls -lh models/ml_rl_1000/*.pth
   ```

2. **Use in Production**
   - Copy `best_model.pth` to your production models directory
   - Update your config to use the trained model
   - Start with paper trading to validate

3. **Further Training**
   - Resume from final checkpoint with different data
   - Try different symbols (ETH, BNB, etc.)
   - Experiment with different initial balances

4. **Analysis**
   - Load `training_results_1000.json` for custom analysis
   - Compare different training runs
   - Optimize hyperparameters based on results

## Example Training Session

Complete example:

```bash
# 1. Create checkpoint directory
mkdir -p models/ml_rl_1000

# 2. Start training with adaptive agent
python scripts/train_ml_rl_1000_rounds.py \
  --agent-type adaptive \
  --balance 10000 \
  --data-days 180 \
  --symbol BTC/USDT \
  --checkpoint-dir models/ml_rl_1000

# 3. Training runs for ~4-8 hours (depends on hardware)

# 4. Review results
ls -lh models/ml_rl_1000/
cat models/ml_rl_1000/training_summary_1000.txt

# 5. View visual report
open models/ml_rl_1000/training_report_1000.png  # macOS
xdg-open models/ml_rl_1000/training_report_1000.png  # Linux

# 6. Use the best model
cp models/ml_rl_1000/best_model.pth models/production/rl_agent.pth
```

## Support

For issues or questions:
1. Check the logs: `logs/ml_rl_1000_training.log`
2. Review the hardware profile to ensure proper detection
3. Try with `--agent-type basic` to isolate optimization issues
4. Ensure all dependencies are installed: `pip install -r requirements.txt`

## Performance Expectations

Typical training times (1000 episodes):

| Hardware | Approximate Time |
|----------|-----------------|
| CPU only (4 cores) | 12-16 hours |
| CPU + Low-end GPU (2GB) | 6-8 hours |
| CPU + Mid-range GPU (6GB) | 3-5 hours |
| CPU + High-end GPU (8GB+) | 2-3 hours |
| Multi-GPU setup | 1-2 hours |

Expected results (varies by market conditions):
- Profitable episodes: 50-70%
- Average win rate: 50-60%
- Best episode profit: 5-15%
- Final epsilon: 0.01-0.05

Happy Training! 🚀
