# Model Manifest System Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-15

## Overview

The Model Manifest System ensures AI trading models only make trades they were trained for. It automatically documents training configuration, validates trading capabilities, and supports managing multiple specialized models.

## Key Features

### ✅ Automatic Manifest Generation
- Created automatically during training
- Documents all training parameters
- Records validation performance
- Saves model capabilities

### ✅ Trading Safety
- Models can only trade symbols they were trained on
- Enforces timeframe compatibility
- Validates exchange compatibility
- Requires manual approval for live trading

### ✅ Multi-Model Support
- Multiple specialized models for different assets
- Automatic model selection based on trade
- Performance-based model ranking

## Manifest Structure

### What's Included

**Trading Capabilities:**
- Trading pairs (BTC/USDT, ETH/USDT, etc.)
- Timeframes (1h, 4h, 1d, etc.)
- Exchanges (binance, kraken, etc.)
- Strategies (DQN, etc.)
- Risk parameters
- **DeFi Protocols** (Uniswap, Aave, PancakeSwap, etc.)
- **DeFi Networks** (Ethereum, Polygon, BSC, etc.)
- **DeFi Strategies** (yield farming, liquidity provision, lending, etc.)

**Training Metadata:**
- Walk-forward configuration
- RL agent parameters
- Training duration
- Number of folds
- Validation metrics

**Performance Summary:**
- Mean return ± std
- Mean Sharpe ratio ± std
- Win rate
- Max drawdown
- Best fold performance

## How It Works

### 1. Training Generates Manifest

When training completes:
```python
# Automatic manifest generation
manifest = trainer._generate_manifest(results, best_fold_id)
manifest.save('models/walk_forward/fold_0_manifest.json')
```

### 2. Model Manager Validates Trades

Before trading:
```python
from nexlify.models import ModelManager

manager = ModelManager()
manager.scan_models_directory()  # Find all models
manager.set_active_model('wf_model_20251115_143022')

# Validate trade
is_valid, reason = manager.validate_trade(
    symbol='BTC/USDT',
    timeframe='1h',
    exchange='binance'
)

if not is_valid:
    print(f"Trade rejected: {reason}")
```

### 3. Select Best Model for Trade

```python
# Find best model for specific trade
best_model = manager.get_best_model_for_trade(
    symbol='BTC/USDT',
    timeframe='1h',
    exchange='binance'
)

if best_model:
    print(f"Using model: {best_model.model_name}")
    print(f"Sharpe: {best_model.validation_metrics['sharpe_ratio']:.2f}")
```

## Example Manifest

```json
{
  "model_id": "wf_model_20251115_143022",
  "model_name": "Walk-Forward Model (Fold 2)",
  "version": "1.0.0",
  "created_at": "2025-11-15T14:30:22",

  "capabilities": {
    "symbols": ["BTC/USDT", "ETH/USDT"],
    "timeframes": ["1h"],
    "base_currencies": ["BTC", "ETH"],
    "quote_currencies": ["USDT"],
    "exchanges": ["binance"],
    "strategies": ["DQN", "walk_forward_validated"],
    "max_position_size": 0.1,
    "min_confidence": 0.7,
    "defi_enabled": true,
    "defi_protocols": ["uniswap_v3", "aave"],
    "defi_networks": ["ethereum", "polygon"],
    "defi_strategies": ["yield_farming", "liquidity_provision", "lending", "auto_compound"]
  },

  "training": {
    "method": "walk_forward",
    "total_episodes": 2000,
    "train_size": 1000,
    "test_size": 200,
    "mode": "rolling",
    "num_folds": 6,
    "learning_rate": 0.001,
    "architecture": "medium",
    "validation_metrics": {
      "total_return": 0.125,
      "sharpe_ratio": 1.82,
      "win_rate": 0.65
    }
  },

  "performance_summary": {
    "mean_sharpe": 1.82,
    "mean_win_rate": 0.65,
    "num_folds": 6
  },

  "approved_for_live": false
}
```

## Safety Features

### 1. Symbol Validation
```python
manifest.capabilities.can_trade_symbol('BTC/USDT')  # True
manifest.capabilities.can_trade_symbol('DOGE/USDT')  # False - not trained
```

### 2. Timeframe Validation
```python
manifest.capabilities.can_trade_timeframe('1h')  # True
manifest.capabilities.can_trade_timeframe('15m')  # False - not trained
```

### 3. DeFi Protocol Validation
```python
manifest.capabilities.can_use_defi_protocol('uniswap_v3')  # True
manifest.capabilities.can_use_defi_protocol('sushiswap')  # False - not trained
```

### 4. DeFi Network Validation
```python
manifest.capabilities.can_use_defi_network('ethereum')  # True
manifest.capabilities.can_use_defi_network('arbitrum')  # False - not trained
```

### 5. DeFi Strategy Validation
```python
manifest.capabilities.can_execute_defi_strategy('yield_farming')  # True
manifest.capabilities.can_execute_defi_strategy('flash_loans')  # False - not trained
```

### 6. Performance Thresholds
```python
# Model must meet minimum requirements
manifest.min_sharpe_ratio = 1.0
manifest.min_win_rate = 0.55
manifest.max_drawdown = 0.15

manifest.meets_performance_thresholds()  # True/False
```

### 4. Manual Approval Required
```python
# Models start as NOT approved for live trading
manifest.approved_for_live = False  # Default

# Must be manually approved after review
manifest.approved_for_live = True
manifest.save(path)
```

## Usage Workflow

### Train Model
```bash
python launch_training_ui.py
# Configure and train
# Manifest automatically generated
```

### Review Manifest
```python
from nexlify.models import ModelManifest

manifest = ModelManifest.load('models/walk_forward/fold_0_manifest.json')

print(f"Model: {manifest.model_name}")
print(f"Symbols: {manifest.capabilities.symbols}")
print(f"Sharpe: {manifest.validation_metrics['sharpe_ratio']:.2f}")
print(f"Win Rate: {manifest.validation_metrics['win_rate']:.2%}")
```

### Approve for Live Trading
```python
# After manual review
if manifest.meets_performance_thresholds():
    manifest.approved_for_live = True
    manifest.save('models/walk_forward/fold_0_manifest.json')
```

### Use in Trading
```python
from nexlify.models import ModelManager

manager = ModelManager()
manager.scan_models_directory()

# Set active model
manager.set_active_model('wf_model_20251115_143022')

# Validate trades automatically
is_valid, reason = manager.validate_trade('BTC/USDT', '1h')
```

## Multiple Specialized Models

### Train Different Models
```python
# Model 1: BTC specialist
# Train on BTC/USDT only

# Model 2: ETH specialist
# Train on ETH/USDT only

# Model 3: Multi-asset
# Train on BTC/USDT, ETH/USDT, etc.
```

### Auto-Select Best Model
```python
# System automatically picks best model for each trade
btc_model = manager.get_best_model_for_trade('BTC/USDT', '1h')
eth_model = manager.get_best_model_for_trade('ETH/USDT', '1h')

# Different models can be used for different assets!
```

## Model Organization

### Directory Structure
```
models/
├── walk_forward/
│   ├── fold_0_model.pt
│   ├── fold_0_manifest.json      ← Manifest
│   ├── fold_1_model.pt
│   ├── fold_1_manifest.json
│   └── ...
├── btc_specialist/
│   ├── model.pt
│   └── manifest.json
└── eth_specialist/
    ├── model.pt
    └── manifest.json
```

### Scanning Models
```python
manager = ModelManager(models_dir=Path('models'))
count = manager.scan_models_directory()
print(f"Found {count} models")

for manifest in manager.list_models():
    print(f"- {manifest.model_name}: {manifest.capabilities.symbols}")
```

## Integration with Main App

### In Configuration
```python
# config/neural_config.json
{
  "active_model_id": "wf_model_20251115_143022",
  "require_model_validation": true,
  "min_model_sharpe": 1.0
}
```

### Before Trading
```python
# In trading logic
def execute_trade(symbol, timeframe, ...):
    # Validate model can trade this
    is_valid, reason = model_manager.validate_trade(symbol, timeframe)

    if not is_valid:
        logger.error(f"Trade blocked: {reason}")
        return False

    # Proceed with trade...
```

## Benefits

1. **Safety**: AI only trades what it knows
2. **Transparency**: Full training history documented
3. **Flexibility**: Multiple specialized models
4. **Accountability**: Clear performance metrics
5. **Compliance**: Audit trail for all models

## Best Practices

1. **Always Review**: Check manifest before approving
2. **Validate Performance**: Ensure thresholds met
3. **Test First**: Paper trade before live
4. **Document Changes**: Update manifest if retrained
5. **Version Control**: Keep manifest with model

## Future Enhancements

- UI for manifest management (coming soon)
- Automatic model expiration (retrain schedule)
- Performance monitoring (drift detection)
- A/B testing framework
- Ensemble model support

---

**Last Updated:** 2025-11-15
**Maintainer:** Nexlify Development Team
