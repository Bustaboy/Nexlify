# Nexlify Advanced Training with Historical Data

## 🚀 Overview

This comprehensive training system pulls extensive historical data from the internet and uses advanced techniques to train your ML/RL models for optimal trading performance.

## ✨ Key Features

### 📊 Multi-Source Data Fetching
- **Multiple Exchanges**: Binance, Coinbase, Kraken, Bitfinex, Bitstamp, Huobi
- **Extensive History**: Up to 5+ years of historical data
- **Smart Caching**: Automatic caching for faster retraining
- **Data Quality Validation**: Comprehensive quality checks and cleaning
- **Robust Error Handling**: Exponential backoff retry logic

### 🎯 External Feature Enrichment
- **Fear & Greed Index**: Real crypto market sentiment data
- **On-Chain Metrics**: Network activity, hashrate, transaction data
- **Social Sentiment**: Reddit and Twitter activity proxies
- **Market Regime Detection**: Bull/bear markets, volatility regimes
- **Temporal Features**: Session indicators, seasonality
- **Macroeconomic Indicators**: Interest rate and liquidity proxies

### 📚 Curriculum Learning
Progressive training from easy to hard:
1. **Phase 1 - Warm-up**: Recent stable periods (6 months)
2. **Phase 2 - Intermediate**: Mixed conditions (1 year)
3. **Phase 3 - Advanced**: Multi-year with volatility (2 years)
4. **Phase 4 - Expert**: Complete historical data (5 years)

### 🔄 Automatic Retraining
- **Marginal Improvement Tracking**: Continues training until gains plateau
- **Smart Early Stopping**: Stops when improvements fall below threshold
- **Patience System**: Allows temporary plateaus before stopping
- **Best Model Selection**: Automatically saves and tracks best models

### 📈 Comprehensive Evaluation
- **Walk-Forward Analysis**: Time-series cross-validation
- **Monte Carlo Simulation**: Robustness testing (1000+ simulations)
- **Risk-Adjusted Metrics**: Sharpe, Sortino, Calmar ratios
- **Model Comparison**: Side-by-side performance analysis

## 🛠️ Installation

### Prerequisites
```bash
# Ensure you have Python 3.8+
python --version

# Install required packages (if not already installed)
pip install -r requirements.txt
```

### Quick Setup
```bash
# Navigate to Nexlify directory
cd /path/to/Nexlify

# Make training script executable
chmod +x train_with_historical_data.py

# Run pre-flight check to verify everything works
python nexlify_preflight_checker.py --symbol BTC/USDT --automated

# If all checks pass, you're ready to train!
python train_with_historical_data.py --help
```

## 📖 Usage

### Pre-Flight Check (Recommended)

Before training, run the pre-flight checker to validate all systems:
```bash
# Interactive check (asks for confirmation if issues found)
python nexlify_preflight_checker.py --symbol BTC/USDT --exchange binance

# Automated check (no prompts)
python nexlify_preflight_checker.py --symbol BTC/USDT --automated

# Save report to file
python nexlify_preflight_checker.py --symbol BTC/USDT --save-report preflight.json
```

The pre-flight checker validates:
- ✅ Internet connectivity
- ✅ Exchange API availability and symbol accessibility
- ✅ Fear & Greed Index API
- ✅ Hardware (GPU, CPU, RAM)
- ✅ Python dependencies
- ✅ Disk space

If any critical issues are found, you'll get:
- Clear explanation of the problem
- Impact assessment (how much it will degrade training)
- Step-by-step troubleshooting guide
- Choice to continue or abort

### Basic Training

Train on 5 years of BTC data with curriculum learning:
```bash
python train_with_historical_data.py --symbol BTC/USDT --years 5
```

Note: Pre-flight checks run automatically before training starts!

### Quick Test

Fast test with 1 year of data and 3 iterations:
```bash
python train_with_historical_data.py --quick-test
```

### Custom Configuration

Train on ETH with custom parameters:
```bash
python train_with_historical_data.py \
    --symbol ETH/USDT \
    --exchange binance \
    --years 3 \
    --threshold 2.0 \
    --patience 5 \
    --max-iterations 10
```

### Without Curriculum Learning

Train on all data at once (not recommended):
```bash
python train_with_historical_data.py \
    --symbol BTC/USDT \
    --years 5 \
    --no-curriculum
```

## 📋 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--symbol` | str | BTC/USDT | Trading pair to train on |
| `--exchange` | str | binance | Exchange to fetch data from |
| `--years` | int | 5 | Years of historical data |
| `--threshold` | float | 1.0 | Minimum improvement % to continue |
| `--patience` | int | 3 | Iterations without improvement before stopping |
| `--max-iterations` | int | 10 | Maximum retraining iterations |
| `--output` | str | ./training_output | Output directory |
| `--no-curriculum` | flag | False | Disable curriculum learning |
| `--quick-test` | flag | False | Quick test mode (1 year, 3 iterations) |
| `--automated` | flag | False | **Fully automated mode** (no prompts, uses fallbacks) |
| `--skip-preflight` | flag | False | Skip pre-flight checks (not recommended) |

### Automated Mode

Use `--automated` for fully unattended training:
```bash
# Perfect for cron jobs, background training, or remote servers
python train_with_historical_data.py --symbol BTC/USDT --automated

# The script will:
# - Skip all user prompts
# - Use fallback values if external APIs fail
# - Continue training even if some features are unavailable
# - Generate complete logs for review
```

**When to use `--automated`:**
- Running in cron jobs or scheduled tasks
- Background training on remote servers
- CI/CD pipelines
- When you can't monitor the training actively

**Note:** Pre-flight checks still run in automated mode but won't prompt for user input.

## 📂 Output Structure

After training, you'll find:

```
training_output/
├── preflight_report.json                      # Pre-flight check results
├── best_model/
│   ├── best_model_iter3_score85.2.pt          # Best model checkpoint
│   └── best_model_metadata.json               # Model metadata
├── iteration_1/
│   ├── models/                                # All checkpoints from iteration 1
│   ├── metrics/                               # Training metrics
│   └── training_data/                         # Prepared datasets
├── iteration_2/
│   └── ...
├── evaluation/
│   ├── results/                               # Evaluation results
│   └── comparisons/                           # Model comparisons
└── final_training_report.json                 # Complete training summary
```

The `preflight_report.json` contains:
- All validation checks performed
- Status of each component (pass/warning/fail)
- Impact assessment for any issues
- Troubleshooting guidance

## 📊 Understanding Results

### Training Report (`final_training_report.json`)

```json
{
  "training_info": {
    "symbol": "BTC/USDT",
    "total_iterations": 5,
    "total_time_hours": 12.5
  },
  "final_results": {
    "best_score": 85.2,
    "best_model_path": "./training_output/best_model/..."
  },
  "improvement_curve": [
    {"iteration": 1, "score": 72.5, "improvement_pct": 100.0},
    {"iteration": 2, "score": 78.3, "improvement_pct": 8.0},
    {"iteration": 3, "score": 85.2, "improvement_pct": 8.8},
    {"iteration": 4, "score": 85.5, "improvement_pct": 0.4}  // Below threshold, stops
  ]
}
```

### Key Metrics

- **Overall Score**: Weighted combination of all performance metrics (0-100+)
- **Return %**: Total return percentage on test data
- **Sharpe Ratio**: Risk-adjusted return (>1.0 is good, >2.0 is excellent)
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Maximum peak-to-trough decline
- **Profit Factor**: Total gains / Total losses (>1.5 is good)

## 🎯 Training Workflow

```
1. Data Fetching
   ↓
   - Fetch 5 years of hourly candles from exchange
   - Validate data quality (>90 score recommended)
   - Cache for faster future runs

2. Feature Enrichment
   ↓
   - Add 100+ technical indicators
   - Fetch Fear & Greed Index
   - Add on-chain metrics
   - Include social sentiment
   - Detect market regimes

3. Curriculum Training (Phase 1-4)
   ↓
   - Phase 1: 200 episodes on easy data
   - Phase 2: 300 episodes on medium data
   - Phase 3: 400 episodes on hard data
   - Phase 4: 500 episodes on expert data

4. Evaluation
   ↓
   - Test on validation data (last 90 days)
   - Calculate comprehensive metrics
   - Compare to previous best model

5. Auto-Retraining Decision
   ↓
   - If improvement >= threshold: Continue training
   - If improvement < threshold for N iterations: Stop
   - Save best model
```

## 🔧 Advanced Usage

### Custom Data Sources

Edit `nexlify_data/nexlify_historical_data_fetcher.py` to add exchanges:
```python
SUPPORTED_EXCHANGES = ['binance', 'coinbase', 'kraken', 'your_exchange']
```

### Custom Features

Add your own features in `nexlify_data/nexlify_external_features.py`:
```python
def _add_custom_feature(self, df: pd.DataFrame) -> pd.DataFrame:
    # Your custom feature logic
    df['my_feature'] = ...
    return df
```

### Custom Curriculum

Modify `create_curriculum_phases()` in `nexlify_training/nexlify_advanced_training_orchestrator.py`:
```python
phases.append(TrainingPhase(
    name="My Custom Phase",
    difficulty="custom",
    data_period=(start_date, end_date),
    episodes=100,
    initial_balance=10000,
    fee_rate=0.001,
    description="Custom training phase"
))
```

## 📈 Performance Optimization

### For Faster Training
```bash
# Use quick test mode
python train_with_historical_data.py --quick-test

# Reduce years of data
python train_with_historical_data.py --years 2

# Reduce max iterations
python train_with_historical_data.py --max-iterations 5
```

### For Better Models
```bash
# More data
python train_with_historical_data.py --years 7

# More iterations
python train_with_historical_data.py --max-iterations 20

# Stricter improvement threshold
python train_with_historical_data.py --threshold 2.0
```

### GPU Acceleration

The system automatically detects and uses GPU if available:
```python
Device: CUDA  # Will use GPU
Device: CPU   # Will use CPU (slower)
```

To force CPU (for testing):
```bash
CUDA_VISIBLE_DEVICES="" python train_with_historical_data.py ...
```

## 🐛 Troubleshooting

### Issue: Pre-flight checks fail
**Solution**: The pre-flight checker provides specific guidance for each failure. Common issues:

```bash
# Run pre-flight check to see detailed diagnostics
python nexlify_preflight_checker.py --symbol BTC/USDT

# If you want to proceed anyway (not recommended):
python train_with_historical_data.py --skip-preflight ...

# For automated training (uses fallbacks for failures):
python train_with_historical_data.py --automated ...
```

**Common pre-flight failures:**

1. **Internet connectivity**: Check your network connection
2. **Exchange API unavailable**: Try a different exchange with `--exchange kraken`
3. **Symbol not found**: Verify symbol format (e.g., BTC/USDT not BTC-USDT)
4. **Insufficient RAM**: Close other applications or reduce data with `--years 1`
5. **Missing dependencies**: Run `pip install -r requirements.txt`

### Issue: "No data fetched"
**Solution**: Check internet connection and exchange availability
```bash
# Test exchange connection
python -c "import ccxt; print(ccxt.binance().fetch_ticker('BTC/USDT'))"
```

### Issue: "CUDA out of memory"
**Solution**: The system auto-adjusts, but you can force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python train_with_historical_data.py ...
```

### Issue: "Rate limit exceeded"
**Solution**: Data fetcher has built-in retry logic. If persistent, try:
- Wait a few minutes and retry
- Use cached data (automatic if available)
- Try a different exchange

### Issue: Training is too slow
**Solution**:
- Use `--quick-test` for faster results
- Reduce `--years` parameter
- Enable GPU if available
- Close other applications

## 📚 Module Reference

### Data Fetching
- `nexlify_data/nexlify_historical_data_fetcher.py`: Multi-exchange data fetching
- `nexlify_data/nexlify_external_features.py`: External feature enrichment

### Training
- `nexlify_training/nexlify_advanced_training_orchestrator.py`: Main training orchestrator
- `nexlify_training/nexlify_model_evaluator.py`: Model evaluation and comparison
- `train_with_historical_data.py`: Main entry point with auto-retraining

### Models
- `nexlify_rl_models/nexlify_ultra_optimized_rl_agent.py`: Ultra-optimized DQN agent
- `nexlify_rl_models/nexlify_adaptive_rl_agent.py`: Adaptive DQN agent
- `nexlify_rl_models/nexlify_rl_agent.py`: Base DQN agent

## 🎓 Best Practices

1. **Start with Quick Test**: Always run `--quick-test` first to verify setup
2. **Use Curriculum Learning**: Don't disable unless you have a good reason
3. **Monitor Improvement**: Check logs to ensure models are improving
4. **Validate Results**: Always test on recent unseen data before live trading
5. **Paper Trade First**: Use paper trading before deploying to live markets
6. **Regular Retraining**: Retrain monthly with latest data for best results

## 🚨 Important Warnings

⚠️ **Risk Disclaimer**: This training system is for educational and research purposes. Trading involves risk of financial loss. Always:
- Test thoroughly in paper trading
- Start with small amounts in live trading
- Never trade more than you can afford to lose
- Monitor performance regularly
- Be prepared to stop trading if performance degrades

⚠️ **Data Quality**: Always check data quality scores. Scores below 85/100 may indicate issues with the data source.

⚠️ **Overfitting**: Curriculum learning and validation help prevent overfitting, but always test on recent unseen data.

## 📞 Support

For issues or questions:
1. Check this README
2. Review the logs in `training_output/`
3. Check the main Nexlify documentation
4. Create an issue on GitHub

## 🎉 Quick Start Example

```bash
# 1. Quick test to verify everything works
python train_with_historical_data.py --quick-test

# 2. If successful, run full training
python train_with_historical_data.py --symbol BTC/USDT --years 5

# 3. Check results
cat training_output/final_training_report.json

# 4. Test best model in paper trading
python nexlify_backtesting/nexlify_paper_trading_runner.py evaluate \
    --model training_output/best_model/best_model_*.pt

# 5. If paper trading is profitable, deploy carefully to live trading
```

## 📊 Expected Results

With 5 years of BTC data and curriculum learning, you can expect:
- **Training time**: 8-16 hours (depending on hardware)
- **Overall score**: 70-90+ (higher is better)
- **Sharpe ratio**: 1.5-3.0 (depends on market conditions)
- **Win rate**: 55-70%
- **Improvement iterations**: 3-7 before plateau

Results vary based on:
- Market conditions during training period
- Quality of historical data
- Hardware capabilities
- Curriculum phases completed
- Hyperparameter settings

---

**Happy Trading! 🚀📈**

Remember: Past performance does not guarantee future results. Always practice proper risk management.
