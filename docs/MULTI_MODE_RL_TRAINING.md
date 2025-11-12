# Multi-Mode RL Training Guide

## Overview

The Nexlify Multi-Mode RL system provides comprehensive reinforcement learning training across **all** trading modes available in the application:

### Supported Trading Modes

✅ **Spot Trading** - Traditional buy/sell
✅ **Futures Long** - Leveraged long positions
✅ **Futures Short** - Short selling with leverage
✅ **Margin Trading** - Leveraged spot trading
✅ **DeFi Operations**:
  - DEX Swaps (Uniswap-style)
  - Liquidity Pool Provision
  - Yield Farming
  - Staking

### Key Features

- **30 Actions** (vs 3 in basic agent)
- **31 State Features** (vs 8 in basic agent)
- **Partial Position Sizing** (25%, 50%, 75%, 100%)
- **Multiple Concurrent Positions**
- **Comprehensive Fee Tracking** (trading fees, gas fees, funding fees)
- **Liquidation Risk Management**
- **GPU-Accelerated Training**

---

## Action Space (30 Actions)

### 0: Hold
Do nothing this step

### Spot Trading (1-8)
- **1-4**: Buy Spot (25%, 50%, 75%, 100% of balance)
- **5-8**: Sell Spot (25%, 50%, 75%, 100% of position)

### Futures Long (9-13)
- **9-12**: Long Futures (25%, 50%, 75%, 100% with 5x leverage)
- **13**: Close All Long Futures

### Futures Short (14-18)
- **14-17**: Short Futures (25%, 50%, 75%, 100% with 5x leverage)
- **18**: Close All Short Futures

### DeFi Operations (19-27)
- **19-22**: Add Liquidity to Pool (25%, 50%, 75%, 100%)
  - Earns 15% APY + trading fees
  - Subject to impermanent loss
  - Requires gas fees
- **23**: Remove All Liquidity
- **24-26**: Stake (25%, 50%, 100%)
  - Earns 8% APY
  - Locked position (must unstake to access)
  - Requires gas fees
- **27**: Unstake All

### Risk Management (28-29)
- **28**: Close All Positions (emergency exit)
- **29**: Reduce All Positions by 50% (de-risk)

---

## State Space (31 Features)

### Account Status (5 features)
1. `balance_ratio` - Current balance / initial balance
2. `margin_available_ratio` - Available margin / initial balance
3. `total_value_ratio` - Total portfolio value / initial balance
4. `equity_ratio` - Current equity / initial balance
5. `margin_ratio` - Margin used / equity

### Market Data (8 features)
6. `price_normalized` - Current price / initial balance
7. `price_change` - Price change from previous step
8. `rsi` - RSI indicator (14-period)
9. `macd` - MACD indicator
10. `volume` - Volume proxy (volatility 10-period)
11. `volatility` - Price volatility (20-period)
12. `trend` - Long-term trend (50-period)
13. `momentum` - Price momentum (20-period)

### Position Status (12 features)

**Spot Positions:**
14. `spot_position` - Amount of spot holdings
15. `spot_entry_ratio` - Spot entry price / current price
16. `spot_pnl_ratio` - Spot P&L / initial balance

**Futures Long:**
17. `long_position` - Size of long futures positions
18. `long_entry_ratio` - Long entry price / current price
19. `long_pnl_ratio` - Long P&L / initial balance
20. `long_liquidation_distance` - Distance to liquidation (0-1)

**Futures Short:**
21. `short_position` - Size of short futures positions
22. `short_entry_ratio` - Short entry price / current price
23. `short_pnl_ratio` - Short P&L / initial balance
24. `short_liquidation_distance` - Distance to liquidation (0-1)

**Total:**
25. `total_exposure_ratio` - Total exposure / initial balance

### DeFi Status (5 features)
26. `lp_value_ratio` - Liquidity pool value / initial balance
27. `lp_apy` - Current LP APY
28. `lp_impermanent_loss_ratio` - Impermanent loss / initial balance
29. `staked_amount` - Amount currently staked
30. `staking_apy` - Current staking APY

### Fee Tracking (1 feature)
31. `cumulative_fees_ratio` - Total fees paid / initial balance

---

## Fee Structure

### Trading Fees
- **Spot/Futures**: 0.1% (0.001) per trade
- **DEX Swaps**: 0.3% (0.003) per swap

### Gas Fees (DeFi Only)
- **Add Liquidity**: $5 USD equivalent
- **Remove Liquidity**: $5 USD equivalent
- **Stake**: $5 USD equivalent
- **Unstake**: $5 USD equivalent

### Funding Fees (Futures Only)
- **Rate**: 0.01% (0.0001) per 8 hours
- **Applied to**: All open futures positions

### Margin Interest
- **Rate**: 0.02% (0.0002) per day
- **Applied to**: Leveraged positions

**The agent learns to minimize fees while maximizing returns!**

---

## Reward Function

The reward function is comprehensive and considers:

```python
total_reward = (
    + immediate_profit_from_trade
    - trading_fees
    - gas_fees
    + unrealized_pnl * 0.01  # Small reward for good positions
    - liquidation_penalty * 1.0  # Heavy penalty (-1.0) for liquidation
    - funding_fees
    - hold_penalty * 0.0001  # Encourage action
)
```

### Reward Components

1. **Trade Profits**: Realized profits from closing positions
2. **Unrealized P&L**: Small ongoing reward for profitable open positions
3. **Fee Penalties**: All fees are subtracted from rewards
4. **Liquidation Penalty**: Massive -1.0 penalty for getting liquidated
5. **Hold Penalty**: Tiny penalty to encourage taking actions

---

## Quick Start

### 1. Basic Training (100 Episodes)

```bash
python scripts/train_multi_mode_rl.py --episodes 100
```

### 2. Extended Training (1000 Episodes)

```bash
python scripts/train_multi_mode_rl.py \
    --episodes 1000 \
    --days 365 \
    --balance 10000 \
    --leverage 10.0 \
    --profile balanced \
    --checkpoint-dir models/multi_mode_1000
```

### 3. GPU-Optimized Training

```bash
# Use MAXIMUM_PERFORMANCE profile for max speed
python scripts/train_multi_mode_rl.py \
    --episodes 1000 \
    --profile maximum
```

### 4. Conservative Training (Lower Leverage)

```bash
# Use lower leverage for safer training
python scripts/train_multi_mode_rl.py \
    --episodes 500 \
    --leverage 3.0
```

---

## Training Parameters

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--episodes` | 100 | Number of training episodes |
| `--days` | 180 | Days of historical price data |
| `--balance` | 10000 | Initial balance (USD) |
| `--leverage` | 10.0 | Maximum leverage allowed |
| `--profile` | balanced | GPU optimization profile |
| `--checkpoint-dir` | models/multi_mode | Where to save models |
| `--save-interval` | 10 | Save checkpoint every N episodes |

### Optimization Profiles

| Profile | Description | Best For |
|---------|-------------|----------|
| `auto` | Automatically benchmarks optimizations | Production training |
| `balanced` | Good performance with < 5% overhead | Development |
| `ultra_low_overhead` | Minimal overhead, < 1% | Quick experiments |
| `maximum` | All optimizations enabled | Maximum speed |

---

## Understanding Results

### Key Metrics

**Return %**: Total portfolio return percentage
**Final Value**: Ending portfolio value
**Trades**: Number of trades executed
**Liquidations**: How many times positions were liquidated (should be 0!)
**Avg Loss**: Training loss (should decrease over time)
**Epsilon**: Exploration rate (decreases over time)

### Good Training Signs

✅ Returns trending upward over episodes
✅ Zero or very few liquidations
✅ Training loss decreasing
✅ Epsilon decay working (1.0 → 0.01)
✅ Average trades per episode stabilizes

### Bad Training Signs

❌ Many liquidations (agent too risky)
❌ Returns always negative (agent losing money)
❌ Training loss not decreasing (learning not happening)
❌ Very few trades (agent too passive)

---

## Model Files

After training, you'll have:

```
models/multi_mode/
├── best_model.pth                    # Best performing model
├── final_model_100.pth               # Final model after all episodes
├── checkpoint_ep10.pth               # Checkpoint at episode 10
├── checkpoint_ep20.pth               # Checkpoint at episode 20
├── ...
├── training_results_100.json         # Full results data
├── training_report.png               # Visual report
└── training_data.npy                 # Price data used
```

---

## Using Trained Models

### Load and Use Model

```python
from nexlify.strategies.nexlify_enhanced_rl_agent import EnhancedTradingEnvironment
from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
from nexlify.ml.nexlify_optimization_manager import OptimizationProfile
import numpy as np

# Load price data
price_data = np.load("models/multi_mode/training_data.npy")

# Create environment
env = EnhancedTradingEnvironment(
    price_data=price_data,
    initial_balance=10000,
    max_leverage=10.0
)

# Create agent
agent = create_ultra_optimized_agent(
    state_size=31,  # Multi-mode uses 31 features
    action_size=30,  # Multi-mode uses 30 actions
    profile=OptimizationProfile.INFERENCE_ONLY
)

# Load trained model
agent.load("models/multi_mode/best_model.pth")

# Use agent (inference mode, no training)
state = env.reset()
action = agent.act(state, training=False)  # Greedy action selection

print(f"Agent action: {env._get_action_name(action)}")
```

### Interpreting Actions

The agent returns an action index (0-29). Use `env._get_action_name(action)` to see what it means:

```python
0 → "hold"
1-4 → "buy_spot_25%", "buy_spot_50%", "buy_spot_75%", "buy_spot_100%"
9-12 → "long_futures_25%", "long_futures_50%", "long_futures_75%", "long_futures_100%"
19-22 → "add_liquidity_25%", "add_liquidity_50%", "add_liquidity_75%", "add_liquidity_100%"
...
```

---

## Comparison: Basic vs Multi-Mode

| Feature | Basic RL | Multi-Mode RL |
|---------|----------|---------------|
| **Trading Modes** | Spot only | Spot, Futures, DeFi |
| **Actions** | 3 (Hold, Buy, Sell) | 30 (comprehensive) |
| **State Features** | 8 | 31 |
| **Position Sizing** | 100% only | 25%, 50%, 75%, 100% |
| **Leverage** | ❌ No | ✅ Yes (up to 10x) |
| **Short Selling** | ❌ No | ✅ Yes |
| **DeFi** | ❌ No | ✅ Yes (LP, Staking) |
| **Fee Tracking** | Basic | Comprehensive |
| **Risk Management** | Basic | Advanced |
| **Multiple Positions** | ❌ One at a time | ✅ Multiple concurrent |

---

## Advanced Topics

### Custom Rewards

Edit `EnhancedTradingEnvironment` to customize the reward function:

```python
# In nexlify/strategies/nexlify_enhanced_rl_agent.py
# Modify the step() method reward calculation

# Example: Penalize high fee spending
if self.total_fees_paid > self.initial_balance * 0.05:
    reward -= 0.1  # Penalty if fees > 5% of capital
```

### Custom Actions

Add new actions by:
1. Increasing `action_space_n`
2. Adding action name to `_get_action_name()`
3. Adding handler in `_execute_action()`

### Custom State Features

Add new features by:
1. Increasing `state_space_n`
2. Computing feature in `_get_state()`
3. Adding to state vector array

---

## Tips for Best Results

### 1. Start Conservative
```bash
python scripts/train_multi_mode_rl.py --leverage 3.0 --episodes 100
```

### 2. Monitor Liquidations
If you see many liquidations, reduce leverage or adjust reward function.

### 3. Use More Data
More historical data = better learning:
```bash
python scripts/train_multi_mode_rl.py --days 365  # 1 year of data
```

### 4. Long Training
1000+ episodes recommended for production:
```bash
python scripts/train_multi_mode_rl.py --episodes 1000
```

### 5. Validate on Different Data
Train on one period, test on another to avoid overfitting.

---

## Troubleshooting

### Agent Always Gets Liquidated

**Problem**: Too much leverage, risky positions
**Solution**:
- Reduce max leverage: `--leverage 3.0`
- Increase liquidation penalty in reward function
- Train longer (agent needs to learn risk management)

### Agent Too Passive (Few Trades)

**Problem**: Hold penalty too small
**Solution**:
- Increase hold penalty in `_execute_action()`
- Reduce epsilon decay rate
- Ensure epsilon starts at 1.0 (full exploration)

### Training Loss Not Decreasing

**Problem**: Learning not happening
**Solution**:
- Check batch size is appropriate
- Verify GPU is being used
- Increase training episodes
- Try different optimization profile

### Poor Performance Despite Low Loss

**Problem**: Overfitting to training data
**Solution**:
- Use more diverse training data
- Add noise/randomness to environment
- Validate on separate test data
- Use regularization

---

## Next Steps

1. **Train Your First Model**:
   ```bash
   python scripts/train_multi_mode_rl.py --episodes 100
   ```

2. **Monitor Training**:
   Watch the console output for returns, liquidations, etc.

3. **Evaluate Results**:
   Check `models/multi_mode/training_report.png`

4. **Iterate**:
   Adjust parameters based on results

5. **Deploy**:
   Load best model and use in production trading

---

## Support

For issues or questions about multi-mode RL training:

1. Check training logs: `logs/multi_mode_training.log`
2. Review results JSON: `models/multi_mode/training_results_*.json`
3. Examine visual report: `models/multi_mode/training_report.png`
4. See example usage: `examples/gpu_training_example.py`

---

**Last Updated**: 2025-11-12
**Nexlify Version**: 1.0+
**Requires**: PyTorch 2.1.0+, CUDA 11.8+ (for GPU)
