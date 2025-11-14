# Dynamic Fee System Documentation

## Overview

The Nexlify Dynamic Fee System provides **real-time fee calculation** for different trading venues (CEX and DEX) to ensure accurate RL training and safe live trading.

## Critical Insights

### Why Dynamic Fees Matter

**Round-trip fees are incurred TWICE** (entry + exit):
- USD → ETH: Entry fee (buy)
- ETH → USD: Exit fee (sell)

For a $1000 trade:
- **Binance (CEX)**: $2.00 total (0.2%)
- **Ethereum L1 (DEX)**: $24.00 total (2.4%) - **12x more expensive!**
- **Polygon (L2)**: $6.00 total (0.6%)

**Using hardcoded 0.1% fees for ETH training = catastrophic learning errors**

## Safety Features

### Strict Fee Mode (CRITICAL for Live Trading)

```python
from nexlify.config import CryptoTradingConfig

# LIVE TRADING: Strict mode enforced automatically
live_config = CryptoTradingConfig(
    trading_mode="live",           # Auto-enables strict_fee_mode
    trading_network="binance",     # Your actual exchange
    use_dynamic_fees=True          # REQUIRED for live
)

# Trading is BLOCKED if real fees cannot be retrieved
```

**Live trading requirements:**
1. ✅ `trading_mode="live"` - Automatically enforces strict fees
2. ✅ `use_dynamic_fees=True` - Must query real fees
3. ✅ Valid fee provider configured
4. ❌ Will raise `RuntimeError` if fees unavailable

### Trading Modes

| Mode | Strict Fees | Fallback Allowed | Use Case |
|------|-------------|------------------|----------|
| `"live"` | ✅ Always | ❌ Never | Production trading |
| `"paper"` | Optional | ⚠️  With warning | Paper trading |
| `"backtest"` | Optional | ✅ Yes | Historical simulation |

## Quick Start

### 1. Basic Setup (Backtesting)

```python
from nexlify.config import CryptoTradingConfig
from nexlify.strategies.nexlify_rl_agent import TradingEnvironment
import numpy as np

# Create config for Binance
config = CryptoTradingConfig(
    trading_network="binance",  # CEX: low fees
    trading_mode="backtest"
)

# Create environment with dynamic fees
price_data = np.random.randn(1000) + 100
env = TradingEnvironment(
    price_data=price_data,
    initial_balance=10000,
    fee_provider=config.fee_provider,
    config=config.to_dict()
)

# Fees are automatically calculated per trade
state = env.reset()
next_state, reward, done, info = env.step(1)  # Buy action
print(f"Entry fees: ${info['total_entry_fees']:.2f}")
print(f"Network: {info['fee_network']}")
```

### 2. Ethereum L1 Setup (High Gas Fees)

```python
# ETH L1: Variable gas fees
eth_config = CryptoTradingConfig(
    trading_network="ethereum",
    trading_mode="backtest"
)

# Check fee estimate before trading
estimate = eth_config.get_fee_estimate(1000.0)
print(f"Entry: ${estimate.calculate_entry_cost(1000.0)[0] + estimate.calculate_entry_cost(1000.0)[1]:.2f}")
print(f"Exit: ${estimate.calculate_exit_cost(1000.0)[0] + estimate.calculate_exit_cost(1000.0)[1]:.2f}")
print(f"Round trip: ${estimate.calculate_round_trip_cost(1000.0):.2f}")
```

### 3. Live Trading Setup (Strict Mode)

```python
# LIVE TRADING: Maximum safety
live_config = CryptoTradingConfig(
    trading_mode="live",        # Auto-enables strict mode
    trading_network="binance",  # Your actual exchange
    use_dynamic_fees=True
)

# If fee provider fails, trading is BLOCKED
try:
    estimate = live_config.get_fee_estimate(1000.0)
    # Safe to trade
except RuntimeError as e:
    print(f"BLOCKED: {e}")
    # DO NOT TRADE - fees unavailable
```

## Fee Providers

### Centralized Exchanges (CEX)

#### BinanceFeeProvider
```python
from nexlify.config import get_fee_provider

binance = get_fee_provider("binance")
estimate = binance.get_fee_estimate(trade_size_usd=1000.0)

# Typical fees:
# - Maker: 0.1%
# - Taker: 0.1%
# - Round trip: 0.2% ($2 on $1000)
```

#### CoinbaseFeeProvider
```python
coinbase = get_fee_provider("coinbase")
estimate = coinbase.get_fee_estimate(trade_size_usd=1000.0)

# Typical fees:
# - Standard: 0.5%
# - Round trip: 1.0% ($10 on $1000)
```

### Decentralized Networks (DEX)

#### EthereumFeeProvider
```python
from nexlify.config import EthereumFeeProvider

# Custom gas price
eth = EthereumFeeProvider(gas_price_gwei=50.0)
estimate = eth.get_fee_estimate(trade_size_usd=1000.0)

# Fees include:
# - DEX swap fee: 0.3% (Uniswap)
# - Gas cost: ~$10-30 per swap (variable!)
# - Round trip: 0.6% + $20-60 gas
```

**Gas Fee Impact by Trade Size:**
- $100 trade: Gas = 10-30% of trade value ❌
- $1000 trade: Gas = 1-3% of trade value ⚠️
- $10000 trade: Gas = 0.1-0.3% of trade value ✅

#### PolygonFeeProvider
```python
polygon = get_fee_provider("polygon")
estimate = polygon.get_fee_estimate(trade_size_usd=1000.0)

# Fees:
# - DEX swap: 0.3%
# - Gas: ~$0.05 per swap
# - Round trip: 0.6% + ~$0.10
```

#### BSCFeeProvider
```python
bsc = get_fee_provider("bsc")
estimate = bsc.get_fee_estimate(trade_size_usd=1000.0)

# Fees:
# - DEX swap: 0.25% (PancakeSwap)
# - Gas: ~$0.30 per swap
# - Round trip: 0.5% + ~$0.60
```

## Fee Comparison

```python
from nexlify.config import compare_fee_providers

# Compare all networks for $1000 trade
compare_fee_providers(1000.0)
```

**Output:**
```
Fee Comparison for $1,000.00 Round-Trip Trade (Buy + Sell)
================================================================================
Network         Entry Fee       Exit Fee        Round Trip      % Cost
--------------------------------------------------------------------------------
binance         $1.00           $1.00           $2.00           0.200%
coinbase        $5.00           $4.98           $9.98           0.998%
ethereum        $12.00          $11.96          $23.96          2.396%
polygon         $3.05           $3.04           $6.09           0.609%
bsc             $2.80           $2.79           $5.59           0.559%
================================================================================
```

## Advanced Usage

### Custom Fee Provider

```python
from nexlify.config.fee_providers import FeeProvider, FeeEstimate

class MyExchangeFeeProvider(FeeProvider):
    def __init__(self, api_key):
        self.api_key = api_key

    def get_fee_estimate(self, asset="ETH", trade_size_usd=1000.0):
        # Query your exchange's API
        maker_fee = self._query_maker_fee(asset)
        taker_fee = self._query_taker_fee(asset)

        return FeeEstimate(
            entry_fee_rate=taker_fee,
            exit_fee_rate=taker_fee,
            network="MyExchange",
            fee_type="percentage"
        )

    def get_network_name(self):
        return "MyExchange"
```

### Integration with Training

```python
from nexlify.config import CryptoTradingConfig
from nexlify.environments.nexlify_rl_training_env import TradingEnvironment

# Train for specific network
config = CryptoTradingConfig(
    trading_network="polygon",  # L2 for reasonable fees
    trading_mode="backtest",
    # ... other RL hyperparameters
)

# Environment automatically uses dynamic fees
env = TradingEnvironment(
    initial_balance=10000,
    fee_provider=config.fee_provider,
    # ... other params
)

# Agent learns with ACCURATE fee accounting
# This prevents overfitting to unrealistic zero-fee scenarios
```

## Impact on RL Training

### Without Dynamic Fees (BAD)
```python
# Agent trains with hardcoded 0.1% fees
# Reality: Trading on Ethereum with 2.4% fees
# Result: Agent makes 100+ trades/day thinking fees are negligible
# Real-world: Loses money on every trade due to gas fees
```

### With Dynamic Fees (GOOD)
```python
# Agent trains with real 2.4% Ethereum fees
# Learns: Minimize trades, only trade on strong signals
# Result: Adapts strategy to high-fee environment
# Real-world: Profitable because strategy accounts for actual costs
```

### Fee-Aware Position Sizing

The environment automatically accounts for fees in position sizing:

```python
# Buy action
amount_to_invest = balance * 0.95  # Leave room for fees
fee_estimate = get_fee_estimate(amount_to_invest)
percentage_fee, fixed_fee = fee_estimate.calculate_entry_cost(amount_to_invest)

# Actual position after fees
position = (amount_to_invest - percentage_fee - fixed_fee) / price
```

## Configuration Reference

### CryptoTradingConfig

```python
@dataclass
class CryptoTradingConfig:
    # Fee configuration
    trading_network: str = "static"        # Network/exchange
    fee_provider: Optional[FeeProvider] = None
    use_dynamic_fees: bool = True
    fee_rate: float = 0.001                # Deprecated fallback

    # Safety
    strict_fee_mode: bool = False          # Block trades if fees unavailable
    trading_mode: str = "backtest"         # "backtest", "paper", "live"

    # RL hyperparameters
    gamma: float = 0.89                    # Discount factor
    learning_rate: float = 0.0015
    epsilon_end: float = 0.22              # High exploration for crypto
    # ... etc
```

### Configuration Presets

```python
from nexlify.config import (
    CRYPTO_24_7_CONFIG,      # Default: 24/7 crypto optimized
    CONSERVATIVE_CONFIG,      # Conservative: lower risk
    AGGRESSIVE_CONFIG,        # Aggressive: fast learning
)

# Or create custom
custom_config = CryptoTradingConfig(
    trading_network="ethereum",
    trading_mode="backtest",
    gamma=0.85,              # Very short-term
    epsilon_end=0.30,        # Very high exploration
)
```

## Best Practices

### ✅ DO

1. **Always use dynamic fees for training**
   ```python
   config = CryptoTradingConfig(
       trading_network="binance",  # Match your target network
       use_dynamic_fees=True
   )
   ```

2. **Match training network to deployment network**
   ```python
   # Train on Binance fees
   train_config = CryptoTradingConfig(trading_network="binance")

   # Deploy on Binance
   # Agent expects 0.2% round-trip fees ✓
   ```

3. **Use strict mode for live trading**
   ```python
   live_config = CryptoTradingConfig(
       trading_mode="live",  # Auto-enables strict mode
       trading_network="binance"
   )
   ```

4. **Consider fee impact on trade size**
   ```python
   # ETH L1: $20 gas per trade
   # Minimum trade size: ~$2000 (1% gas cost)
   # Recommended: $10,000+ (0.2% gas cost)
   ```

### ❌ DON'T

1. **Don't train on CEX fees, deploy on DEX**
   ```python
   # Training
   train_config = CryptoTradingConfig(trading_network="binance")  # 0.2% fees

   # Deployment on Ethereum
   # Reality: 2.4% fees (12x higher!)
   # Result: Agent overtrades and loses money ❌
   ```

2. **Don't use static fees for live trading**
   ```python
   # DANGEROUS - DO NOT DO THIS
   config = CryptoTradingConfig(
       trading_mode="live",
       use_dynamic_fees=False  # ❌ Will raise error
   )
   ```

3. **Don't ignore gas fees on small trades**
   ```python
   # $100 trade on Ethereum
   # Gas: $20
   # You lose 20% before price even moves! ❌
   ```

## Migration Guide

### From Old Hardcoded Fees

**Before:**
```python
env = TradingEnvironment(
    price_data=data,
    initial_balance=10000
    # Hardcoded 0.1% fees inside environment
)
```

**After:**
```python
from nexlify.config import CryptoTradingConfig

config = CryptoTradingConfig(
    trading_network="binance",  # Specify your network
    use_dynamic_fees=True
)

env = TradingEnvironment(
    price_data=data,
    initial_balance=10000,
    fee_provider=config.fee_provider,
    config=config.to_dict()
)
```

### Backward Compatibility

The system maintains backward compatibility:
- Old code without `fee_provider` still works
- Falls back to 0.1% static fees (with warning)
- Tests using 8-feature state space still pass

## Troubleshooting

### "TRADING BLOCKED: Cannot retrieve real-time fees"

**Cause:** Live/strict mode enabled but no fee provider configured

**Solution:**
```python
config = CryptoTradingConfig(
    trading_mode="live",
    trading_network="binance",  # Set your actual network
    use_dynamic_fees=True       # Enable dynamic fees
)
```

### "Using static fallback fees - only safe for backtesting!"

**Cause:** No fee provider configured, using fallback

**Solution:**
```python
# For backtesting: This is OK (just a warning)
# For live trading: Configure fee provider
config.trading_network = "binance"
config.use_dynamic_fees = True
```

### High Fees on Ethereum

**Issue:** Gas fees making small trades unprofitable

**Solutions:**
1. Use L2 (Polygon, BSC) for lower fees
2. Increase minimum trade size
3. Reduce trade frequency
4. Train agent on actual network fees

```python
# Option 1: Switch to L2
config = CryptoTradingConfig(trading_network="polygon")

# Option 2: Train on high-fee environment
# Agent learns to trade less frequently
config = CryptoTradingConfig(trading_network="ethereum")
```

## Examples

See:
- `/examples/fee_comparison_demo.py` - Compare fees across networks
- `/examples/train_with_dynamic_fees.py` - RL training with dynamic fees
- `/examples/live_trading_setup.py` - Safe live trading configuration

## API Reference

Full API documentation: [fee_providers.py](../nexlify/config/fee_providers.py)

## Contributing

When adding new fee providers:
1. Inherit from `FeeProvider` base class
2. Implement `get_fee_estimate()`
3. Implement `get_network_name()`
4. Handle both percentage and fixed fees
5. Add tests for edge cases

## License

Part of the Nexlify project.
