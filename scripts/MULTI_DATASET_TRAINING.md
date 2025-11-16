# Multi-Dataset Training Guide

## What is Multi-Dataset Training?

Instead of training on just one cryptocurrency, you can train the RL agent on **multiple datasets sequentially**, preserving all learned knowledge between datasets. This creates a more robust agent that understands different market conditions.

## Why Train on Multiple Datasets?

✅ **Learns diverse market behaviors** - BTC, ETH, and BNB behave differently
✅ **Better generalization** - Works well on coins it hasn't seen
✅ **More robust** - Doesn't overfit to one asset
✅ **Accumulated knowledge** - Each dataset adds to the agent's experience
✅ **No knowledge loss** - Preserves learning from previous datasets

## Quick Start

### Use a Preset Configuration

```bash
# Train on top 5 cryptocurrencies (5000 total episodes)
python scripts/train_multi_dataset.py --preset top5

# Train on major 3 (3000 total episodes)
python scripts/train_multi_dataset.py --preset major

# Train on top 10 (10000 total episodes)
python scripts/train_multi_dataset.py --preset top10
```

### Custom Dataset List

```bash
# Train on specific symbols
python scripts/train_multi_dataset.py --datasets BTC/USDT ETH/USDT SOL/USDT

# With more data per dataset
python scripts/train_multi_dataset.py --datasets BTC/USDT ETH/USDT --data-days 365
```

## Available Presets

### `--preset major` (Default)
- BTC/USDT
- ETH/USDT
- BNB/USDT

**Total episodes**: 3,000 (3 × 1000)
**Time**: ~6-12 hours (with GPU)
**Best for**: Quick, solid training on major coins

### `--preset top5`
- BTC/USDT
- ETH/USDT
- BNB/USDT
- XRP/USDT
- ADA/USDT

**Total episodes**: 5,000 (5 × 1000)
**Time**: ~10-20 hours (with GPU)
**Best for**: Good balance of diversity and training time

### `--preset top10`
- Top 10 cryptocurrencies by market cap

**Total episodes**: 10,000 (10 × 1000)
**Time**: ~20-40 hours (with GPU)
**Best for**: Maximum diversity and robustness

## Command-Line Options

```bash
python scripts/train_multi_dataset.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--datasets` | Space-separated list of symbols | None |
| `--data-days` | Days of data per dataset | 180 |
| `--balance` | Initial trading balance | 10000 |
| `--agent-type` | Agent type (adaptive, ultra, basic) | adaptive |
| `--base-dir` | Base directory for checkpoints | models/multi_dataset |
| `--preset` | Use preset (top5, top10, major) | None |

## Examples

### Example 1: Train on Major Cryptocurrencies
```bash
python scripts/train_multi_dataset.py --preset major
```

**Training sequence:**
1. BTC/USDT (1000 episodes) → saves best_model.pth
2. ETH/USDT (1000 episodes) → resumes from BTC model
3. BNB/USDT (1000 episodes) → resumes from ETH model

**Final model**: Trained on all 3 datasets (3000 total episodes)

### Example 2: Custom Coins with More Data
```bash
python scripts/train_multi_dataset.py \
  --datasets BTC/USDT ETH/USDT SOL/USDT MATIC/USDT \
  --data-days 365 \
  --balance 50000
```

### Example 3: Maximum Diversity
```bash
python scripts/train_multi_dataset.py \
  --preset top10 \
  --data-days 365 \
  --agent-type ultra
```

This trains on 10 different cryptocurrencies with 1 year of data each!

## How It Works

```
Dataset 1 (BTC)              Dataset 2 (ETH)              Dataset 3 (BNB)
    1000 episodes                1000 episodes                1000 episodes
         |                            |                            |
         v                            v                            v
  [Fresh Agent] → [Trained Agent] → [Resume] → [More Trained] → [Resume] → [Final Agent]
   Random init     Knows BTC         Keeps BTC   Knows BTC+ETH    Keeps all  Knows all 3
                                    knowledge                     knowledge
```

The agent **accumulates knowledge** across all datasets without forgetting previous learning.

## Output Structure

```
models/multi_dataset/
├── step01_BTC_USDT/
│   ├── best_model.pth
│   ├── final_model_1000.pth
│   ├── training_report_1000.png
│   └── training_summary_1000.txt
├── step02_ETH_USDT/
│   ├── best_model.pth (← Resumed from BTC)
│   ├── final_model_1000.pth
│   ├── training_report_1000.png
│   └── training_summary_1000.txt
├── step03_BNB_USDT/
│   ├── best_model.pth (← Resumed from ETH)
│   ├── final_model_1000.pth
│   ├── training_report_1000.png
│   └── training_summary_1000.txt
└── training_log.json (← Complete training history)
```

## Training Log

After training completes, `training_log.json` contains:
```json
{
  "start_time": "2025-01-15T10:00:00",
  "end_time": "2025-01-15T16:30:00",
  "datasets": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
  "training_sessions": [
    {
      "step": 1,
      "symbol": "BTC/USDT",
      "best_model": "models/multi_dataset/step01_BTC_USDT/best_model.pth",
      "resumed_from": null
    },
    {
      "step": 2,
      "symbol": "ETH/USDT",
      "best_model": "models/multi_dataset/step02_ETH_USDT/best_model.pth",
      "resumed_from": "models/multi_dataset/step01_BTC_USDT/best_model.pth"
    }
  ],
  "final_model": "models/multi_dataset/step03_BNB_USDT/best_model.pth"
}
```

## Best Practices

### 1. Start with Major Coins
Begin with BTC and ETH since they have:
- ✅ High liquidity
- ✅ More predictable patterns
- ✅ Better data quality

### 2. Order Matters (Sort of)
Training order can affect results:
- **Recommended**: Start with largest market caps (BTC → ETH → others)
- **Alternative**: Start with most volatile (for aggressive strategies)
- **Experimental**: Mix low and high volatility

### 3. Monitor Progress
After each dataset, check:
- Does profit trend continue upward?
- Is loss still decreasing?
- Are trading strategies improving?

### 4. Save Intermediate Models
The script automatically saves each step's best model. You can:
- Test each intermediate model
- Compare performance across datasets
- Roll back if a dataset hurts performance

### 5. Different Time Periods
You can also train on the same coin with different time periods:
```bash
# This is technically manual but shows the concept
python scripts/train_ml_rl_1000_rounds.py --symbol BTC/USDT --data-days 180
python scripts/train_ml_rl_1000_rounds.py --symbol BTC/USDT --data-days 365 --resume models/prev/best_model.pth
```

## Advanced: Training Strategy Variations

### Conservative Portfolio (Low Risk)
```bash
python scripts/train_multi_dataset.py \
  --datasets BTC/USDT ETH/USDT \
  --balance 100000 \
  --data-days 365
```

### Aggressive Portfolio (High Risk/Reward)
```bash
python scripts/train_multi_dataset.py \
  --datasets DOGE/USDT SHIB/USDT PEPE/USDT \
  --balance 10000 \
  --data-days 90
```

### Balanced Portfolio
```bash
python scripts/train_multi_dataset.py \
  --datasets BTC/USDT ETH/USDT SOL/USDT DOGE/USDT \
  --balance 50000 \
  --data-days 180
```

## Troubleshooting

### "Training failed for X/USDT"
- Check if the symbol exists on Binance
- Verify internet connection for data fetching
- Check logs: `logs/ml_rl_1000_training.log`

### "Out of memory"
- Reduce `--data-days` (less data per training)
- Use `--agent-type basic` (smaller model)
- Train fewer datasets at once

### Poor Performance on Later Datasets
This is normal! Later datasets might show lower profits because:
1. The agent is trying to balance strategies across all datasets
2. Different market conditions require different approaches
3. Some coins are simply harder to trade

**Solution**: Look at the **final model's** performance across all datasets, not individual training runs.

## Comparing to Single-Dataset Training

| Aspect | Single Dataset | Multi-Dataset |
|--------|---------------|---------------|
| Training time | Shorter | Longer |
| Generalization | Lower | Higher |
| Robustness | Lower | Higher |
| Overfitting risk | Higher | Lower |
| Best use case | Trade one coin only | Trade multiple coins |

## Testing the Final Model

After multi-dataset training, test your model:

```bash
# Copy final model to production
cp models/multi_dataset/step03_BNB_USDT/best_model.pth models/production/

# Test on new data (not used in training)
python scripts/backtest_model.py --model models/production/best_model.pth --symbol BTC/USDT
python scripts/backtest_model.py --model models/production/best_model.pth --symbol ETH/USDT
python scripts/backtest_model.py --model models/production/best_model.pth --symbol SOL/USDT
```

The model should perform reasonably well on **all** tested symbols, not just the training ones.

## Expected Results

With multi-dataset training, expect:
- ✅ **Better generalization** - Works on unseen coins
- ✅ **More consistent** - Less extreme wins/losses
- ✅ **Slower convergence** - Takes longer to reach optimal
- ✅ **Lower peak profit** - On individual coins (but better average)

**Example**:
- Single BTC training: 15% profit on BTC, 2% on ETH
- Multi-dataset: 10% profit on BTC, 9% on ETH (more balanced!)

## Next Steps

1. **Start small**: Try `--preset major` first
2. **Review results**: Check each dataset's training report
3. **Scale up**: If results are good, try `--preset top10`
4. **Production**: Use the final model for live/paper trading
5. **Iterate**: Retrain with more data or different datasets

Happy multi-dataset training! 🚀
