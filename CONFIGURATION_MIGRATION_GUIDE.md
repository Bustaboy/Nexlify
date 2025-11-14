# Configuration Migration Guide

**Version:** 2.0.8
**Date:** 2025-11-14
**Branch:** claude/find-hardcoded-values-01XhCbcNfxFFuCDavzrw4f16

## Overview

This guide helps you migrate from hardcoded values to the new centralized configuration system. All previously hardcoded parameters can now be customized through `config/neural_config.json`.

## What Changed?

We've migrated **28 HIGH priority hardcoded values** to centralized configuration:

1. **RL Agent Hyperparameters** (6 values)
2. **Epsilon Decay Parameters** (2 values)
3. **Auto Trader Settings** (3 values)
4. **File Paths & Databases** (2 values)
5. **Risk Management** (Already configurable)
6. **Flash Crash Protection** (Already configurable)

## Benefits

✅ **Flexibility:** Customize all trading parameters without modifying code
✅ **Profiles:** Create multiple configuration profiles (conservative, balanced, aggressive)
✅ **Deployment:** Easier Docker/cloud deployments with environment-specific paths
✅ **Experimentation:** Test different hyperparameters without code changes
✅ **Backward Compatible:** Old configurations still work

---

## Migration Steps

### Step 1: Backup Your Current Config

```bash
cp config/neural_config.json config/neural_config.backup.json
```

### Step 2: Add New Configuration Sections

Add these new sections to your `config/neural_config.json`:

#### A. RL Agent Hyperparameters

```json
{
  "rl_agent": {
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "replay_buffer_size": 100000,
    "batch_size": 64,
    "target_update_frequency": 10,
    "save_frequency": 100,
    "architectures": {
      "tiny": [64, 32],
      "small": [128, 64],
      "medium": [128, 128, 64],
      "large": [256, 256, 128, 64],
      "xlarge": [512, 512, 256, 128, 64]
    },
    "default_architecture": "medium"
  }
}
```

**What this configures:**
- `discount_factor` (gamma): How much to value future rewards (0.89-0.99)
- `epsilon_start/min/decay`: Exploration vs exploitation schedule
- `learning_rate`: How fast the agent learns (0.0001-0.01)
- `replay_buffer_size`: Memory for experience replay (10k-1M)
- `architectures`: Pre-defined network sizes for different hardware

#### B. Epsilon Decay Strategy

```json
{
  "epsilon_decay": {
    "type": "scheduled",
    "start": 1.0,
    "end": 0.22,
    "linear_decay_steps": 2000,
    "scheduled_milestones": {
      "0": 1.0,
      "200": 0.65,
      "800": 0.35,
      "2000": 0.22
    }
  }
}
```

**What this configures:**
- `type`: Decay strategy (linear, scheduled, exponential)
- `end`: Minimum epsilon for 24/7 crypto markets (0.22 = 22% exploration)
- `scheduled_milestones`: Step-by-step epsilon reduction schedule

**Why 0.22?** Crypto markets are volatile and 24/7. Higher ongoing exploration (22%) helps adapt to regime changes, unlike traditional markets which might use 5-10%.

#### C. Auto Trader Exit Strategy

```json
{
  "auto_trader": {
    "enabled": false,
    "check_interval_seconds": 60,
    "exit_strategy": {
      "take_profit_percent": 0.05,
      "stop_loss_percent": 0.02,
      "trailing_stop_percent": 0.03
    }
  }
}
```

**What this configures:**
- `check_interval_seconds`: How often to check for trading opportunities (30-300s)
- `take_profit_percent`: Exit when profit reaches 5%
- `stop_loss_percent`: Exit when loss reaches 2%
- `trailing_stop_percent`: Lock in profits as they increase

#### D. File Paths

```json
{
  "paths": {
    "data_dir": "data",
    "logs_dir": "logs",
    "cache_dir": "cache",
    "models_dir": "models",
    "reports_dir": "reports",
    "trading_database": "data/trading.db",
    "audit_logs_dir": "logs/audit",
    "crash_reports_dir": "logs/crash_reports",
    "defi_positions_file": "data/defi_positions.json",
    "flash_crash_events_log": "data/flash_crash_events.jsonl",
    "risk_state_file": "data/risk_state.json"
  }
}
```

**What this configures:**
- All file and directory paths used by Nexlify
- Supports environment variables (see Advanced Usage below)

#### E. GUI and Monitoring (Optional)

```json
{
  "gui": {
    "cpu_sample_interval_seconds": 1,
    "http_timeout_seconds": 5,
    "default_scan_interval_seconds": 300,
    "chart_update_interval_ms": 5000,
    "intense_mode": {
      "scan_interval_seconds": 60,
      "chart_update_interval_ms": 1000
    },
    "balanced_mode": {
      "scan_interval_seconds": 120,
      "chart_update_interval_ms": 3000
    },
    "conservative_mode": {
      "scan_interval_seconds": 300,
      "chart_update_interval_ms": 5000
    }
  },

  "monitoring": {
    "sample_interval_seconds": 0.1,
    "resize_check_interval_seconds": 60,
    "thread_join_timeout_seconds": 1.0,
    "neural_net_update_interval_seconds": 60
  },

  "cache": {
    "enabled": true,
    "ttl_seconds": 3600,
    "max_size_mb": 500,
    "compression_enabled": true
  }
}
```

### Step 3: Verify Configuration

Check your configuration is valid JSON:

```bash
python3 -c "import json; json.load(open('config/neural_config.json')); print('✓ Config is valid JSON')"
```

### Step 4: Test Changes

Run the quick test pipeline:

```bash
python test_training_pipeline.py --quick
```

---

## Backward Compatibility

**All old configurations continue to work!** The code includes fallback logic:

```python
# New way (preferred):
config['rl_agent']['learning_rate']

# Old way (still works):
# Hardcoded default of 0.001 used if not in config
```

If you don't add the new sections, Nexlify will use the same defaults that were previously hardcoded.

---

## Common Configurations

### Conservative Profile (Low Risk)

```json
{
  "rl_agent": {
    "discount_factor": 0.95,
    "epsilon_end": 0.15,
    "learning_rate": 0.0005
  },
  "auto_trader": {
    "exit_strategy": {
      "take_profit_percent": 0.03,
      "stop_loss_percent": 0.01
    }
  },
  "risk_management": {
    "max_position_size": 0.03,
    "max_daily_loss": 0.02
  }
}
```

### Aggressive Profile (High Risk/Reward)

```json
{
  "rl_agent": {
    "discount_factor": 0.99,
    "epsilon_end": 0.30,
    "learning_rate": 0.002
  },
  "auto_trader": {
    "exit_strategy": {
      "take_profit_percent": 0.10,
      "stop_loss_percent": 0.05
    }
  },
  "risk_management": {
    "max_position_size": 0.10,
    "max_daily_loss": 0.10
  }
}
```

### Fast Trading (Short-term)

```json
{
  "auto_trader": {
    "check_interval_seconds": 30,
    "exit_strategy": {
      "take_profit_percent": 0.02,
      "stop_loss_percent": 0.01
    }
  },
  "gui": {
    "default_scan_interval_seconds": 60
  }
}
```

---

## Advanced Usage

### Environment Variables

You can use environment variables for deployment-specific paths:

```json
{
  "paths": {
    "data_dir": "${DATA_DIR:data}",
    "logs_dir": "${LOGS_DIR:logs}",
    "trading_database": "${DB_PATH:data/trading.db}"
  }
}
```

Then set in your environment:

```bash
export DATA_DIR=/mnt/data
export LOGS_DIR=/var/log/nexlify
export DB_PATH=/mnt/data/trading.db
```

### Architecture Selection

Choose or define your own network architecture:

```json
{
  "rl_agent": {
    "architecture": "large",
    // OR define custom:
    "architecture": [256, 256, 128, 64]
  }
}
```

### Multiple Epsilon Schedules

For different market conditions:

```json
{
  "epsilon_decay": {
    "type": "scheduled",
    "scheduled_milestones": {
      "0": 1.0,
      "100": 0.8,    // Learn basics fast
      "500": 0.5,    // Rapid improvement phase
      "1500": 0.3,   // Refinement
      "3000": 0.25   // Long-term adaptation
    }
  }
}
```

---

## Configuration Reference

### All New Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `rl_agent.discount_factor` | 0.99 | 0.89-0.99 | Future reward weight (gamma) |
| `rl_agent.epsilon_start` | 1.0 | 0.8-1.0 | Initial exploration rate |
| `rl_agent.epsilon_min` | 0.01 | 0.01-0.1 | Minimum exploration |
| `rl_agent.epsilon_decay` | 0.995 | 0.99-0.999 | Epsilon decay rate |
| `rl_agent.learning_rate` | 0.001 | 0.0001-0.01 | Adam optimizer LR |
| `rl_agent.replay_buffer_size` | 100000 | 10k-1M | Experience replay memory |
| `epsilon_decay.end` | 0.22 | 0.05-0.30 | Min epsilon for crypto |
| `auto_trader.check_interval_seconds` | 60 | 10-600 | Trading check frequency |
| `auto_trader.exit_strategy.take_profit_percent` | 0.05 | 0.01-0.20 | Profit target (5%) |
| `auto_trader.exit_strategy.stop_loss_percent` | 0.02 | 0.005-0.10 | Loss limit (2%) |

---

## Troubleshooting

### Issue: "KeyError: 'rl_agent'"

**Solution:** The code has fallbacks. This only happens if you're directly accessing config without `.get()`. Update your code:

```python
# Wrong:
learning_rate = config['rl_agent']['learning_rate']

# Correct:
learning_rate = config.get('rl_agent', {}).get('learning_rate', 0.001)
```

### Issue: Training performance changed

**Solution:** New defaults match the old hardcoded values. If performance changed, check if you modified:
- `discount_factor` (should be ~0.99)
- `epsilon_decay.end` (should be 0.22 for crypto)
- `learning_rate` (should be ~0.001)

### Issue: Different architecture than before

**Solution:** The auto-detection logic remains the same. To force a specific architecture:

```json
{
  "rl_agent": {
    "architecture": "medium"
  }
}
```

---

## Files Modified

1. `config/neural_config.example.json` - Added all new config sections
2. `nexlify/strategies/epsilon_decay.py` - Read from config
3. `nexlify/strategies/nexlify_ultra_optimized_rl_agent.py` - Read RL hyperparameters from config
4. `nexlify/core/nexlify_auto_trader.py` - Read auto trader settings from config
5. `nexlify/utils/error_handler.py` - Read paths from config
6. `nexlify/financial/nexlify_profit_manager.py` - Read database path from config

All changes are **backward compatible** - old configs still work!

---

## Next Steps

1. ✅ Review the audit report: `HARDCODED_VALUES_AUDIT.md`
2. ✅ Update your `config/neural_config.json` with desired parameters
3. ✅ Test with: `python test_training_pipeline.py --quick`
4. ✅ Create configuration profiles for different strategies
5. ✅ Monitor performance and adjust parameters as needed

---

## Questions?

- **Full audit details:** See `HARDCODED_VALUES_AUDIT.md`
- **Implementation roadmap:** See `HARDCODED_VALUES_INDEX.md`
- **Executive summary:** See `HARDCODED_VALUES_SUMMARY.txt`
- **Example config:** `config/neural_config.example.json`

---

**Version History:**
- 2.0.8 (2025-11-14): Initial configuration migration - 28 HIGH priority parameters
