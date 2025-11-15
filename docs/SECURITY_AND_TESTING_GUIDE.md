# Security and Testing Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-15

## Overview

This guide covers three critical features for secure operation and safe testing of Nexlify:

1. **Mandatory PIN Setup** - Forces users to set secure PIN on first boot
2. **Encrypted API Storage** - Secure, centralized API key management
3. **Paper Trading Integration** - Risk-free model testing before live deployment

---

## 1. Mandatory PIN Setup

### Why It's Important

**Problem:** Using default passwords or weak PINs compromises security and puts trading capital at risk.

**Solution:** Mandatory PIN setup dialog that blocks all UI until a secure PIN is configured.

### How It Works

#### First Boot Experience

1. **Application Launches** → User sees main window
2. **PIN Check Triggered** → System detects no PIN or default PIN
3. **Modal Dialog Appears** → Blocks all other UI interactions
4. **PIN Setup Required** → User must set secure PIN
5. **Validation Enforced** → Weak PINs rejected
6. **Cannot Skip** → App exits if user tries to cancel
7. **Application Unlocked** → Normal operation after PIN set

#### PIN Requirements

```
✓ Minimum 6 characters
✓ Maximum 20 characters
✓ Strength score ≥ 40%
✗ No common patterns (123456, password, qwerty, etc.)
✓ Mix of numbers and letters recommended
```

#### Strength Scoring

The system calculates PIN strength based on:

- **Length (40 points max):** Longer is better
- **Character Variety (30 points max):**
  - Lowercase letters (+8)
  - Uppercase letters (+8)
  - Numbers (+7)
  - Special characters (+7)
- **Pattern Avoidance (30 points max):** Penalizes common patterns

**Strength Levels:**
- **0-30%:** ❌ Red - Too weak (rejected)
- **30-60%:** ⚠️  Orange - Acceptable with warning
- **60-100%:** ✓ Green - Strong

### Files Created

**Location:** `nexlify/gui/pin_setup_dialog.py` (450 lines)

**Key Classes:**
- `PINSetupDialog` - Modal dialog for PIN setup
- `check_if_setup_required()` - Detects if setup needed
- `get_setup_reason()` - Determines why setup required

**Storage:**
- PIN hash saved to: `config/.pin_hash`
- Permissions: `600` (owner read/write only)
- Format: SHA-256 hash of PIN
- **Never stores PIN in plaintext**

### Integration Points

#### Training UI (`nexlify/gui/training_ui.py`)

```python
# Shows PIN dialog 100ms after UI loads
QTimer.singleShot(100, self.check_pin_setup)

def check_pin_setup(self):
    """Check if PIN setup required and show dialog"""
    if PINSetupDialog.check_if_setup_required():
        new_pin = show_pin_setup_dialog(self)
        if not new_pin:
            sys.exit(1)  # Exit if PIN not set
```

#### Main Application (`nexlify/gui/cyber_gui.py`)

```python
# PIN setup before login dialog
QTimer.singleShot(100, self._check_pin_setup)

def _check_pin_setup(self):
    """Show PIN setup if needed, then login"""
    if PINSetupDialog.check_if_setup_required():
        new_pin = show_pin_setup_dialog(self)
        # Update API key manager with new PIN
    self._show_login_dialog()
```

### Security Benefits

✅ **Prevents Default PIN Usage:** Forces change on first boot
✅ **Enforces Strong Passwords:** Rejects weak PINs
✅ **Cannot Be Skipped:** App exits if not completed
✅ **Encrypts API Keys:** PIN used for encryption password
✅ **Secure Storage:** PIN hash only, restrictive permissions
✅ **User-Friendly:** Visual feedback on strength

### User Experience

**First Launch:**
```
1. Launch app → "Welcome to Nexlify!"
2. See PIN setup dialog (cannot close)
3. Enter new PIN → Strength bar updates
4. Confirm PIN → Must match
5. Click "Set PIN" → Saved securely
6. Continue to login → Normal operation
```

**Detected Default PIN:**
```
⚠️  Default PIN Detected!

You are using the default PIN which is not secure.
Please set a custom PIN to protect your account.
```

**Weak PIN Attempt:**
```
PIN Requirements:
• 6-20 characters
• Mix of numbers and letters recommended
• Avoid common patterns

[Enter PIN: 123456]
Strength: 10% ❌ Too weak

[Enter PIN: mySecurePin2024]
Strength: 85% ✓ Strong and valid
```

---

## 2. Updated .gitignore Patterns

### The Problem

Previous `.gitignore` used overly broad pattern:
```gitignore
*api_key*  # ❌ Too broad - caught api_key_manager.py source file
```

This prevented tracking of important source code files.

### The Solution

Specific patterns for sensitive files only:

```gitignore
# Encrypted API key storage
config/api_keys.encrypted
config/api_keys.encrypted.backup
config/.api_keys.*
backups/api_keys*.encrypted

# PIN hash file
config/.pin_hash
config/.pin_hash.backup

# Other sensitive files (specific extensions)
*secret.json
*secret.txt
*wallet.json
*wallet.dat
*private.key
*private.pem
!*example*  # Allow example files
```

### What's Protected

✅ **API Key Storage:** `config/api_keys.encrypted`
✅ **PIN Hash:** `config/.pin_hash`
✅ **Backups:** `*.backup` files
✅ **Secrets:** JSON/TXT files with "secret" in name
✅ **Wallets:** Wallet data files
✅ **Private Keys:** `.key` and `.pem` files

### What's Tracked

✓ **Source Code:** `nexlify/security/api_key_manager.py`
✓ **Examples:** `neural_config.example.json`
✓ **Documentation:** All `.md` files
✓ **Tests:** Test files and modules

---

## 3. Paper Trading Integration

### Overview

Paper trading allows you to test trained walk-forward models in a simulated environment before risking real capital.

### Key Features

✅ **Risk-Free Testing:** Simulated trades with real market data
✅ **Model Validation:** Tests against manifest capabilities
✅ **Performance Metrics:** Sharpe ratio, win rate, drawdown
✅ **Batch Testing:** Test multiple models at once
✅ **Results Export:** Save results to JSON

### Architecture

**Location:** `nexlify/training/paper_trading_integration.py` (275 lines)

**Key Classes:**

#### `ModelPaperTester`

Tests individual models using paper trading engine.

```python
from nexlify.training import ModelPaperTester
from nexlify.models import ModelManifest

# Load trained model manifest
manifest = ModelManifest.load('models/walk_forward/fold_0_manifest.json')

# Create paper tester
tester = ModelPaperTester(manifest, initial_balance=10000.0)

# Run 7-day paper test
results = await tester.run_paper_test(duration_days=7)

# Check results
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Total Trades: {results['total_trades']}")
print(f"Final Balance: ${results['end_balance']:,.2f}")
```

### Usage Examples

#### Test Single Model

```python
from nexlify.training import create_paper_test_from_manifest

# Create tester from manifest file
tester = create_paper_test_from_manifest(
    manifest_path=Path('models/walk_forward/fold_0_manifest.json'),
    initial_balance=10000.0
)

# Run test for specific symbol
results = await tester.run_paper_test(
    duration_days=7,
    symbol='BTC/USDT',
    timeframe='1h'
)

# Save results
tester.save_results(results, Path('paper_tests/btc_test.json'))
```

#### Test All Models

```python
from nexlify.training import test_all_models

# Test all models in directory
results = await test_all_models(
    models_dir=Path('models/walk_forward'),
    duration_days=7,
    initial_balance=10000.0
)

# Results is dict: {model_id: results}
for model_id, result in results.items():
    print(f"\n{model_id}:")
    print(f"  Win Rate: {result['win_rate']:.2%}")
    print(f"  Total Trades: {result['total_trades']}")
```

#### Quick Test Helper

```python
from nexlify.training import run_quick_paper_test

# Quick 1-day test
results = await run_quick_paper_test(
    model_path=Path('models/walk_forward/fold_0_model.pt'),
    manifest_path=Path('models/walk_forward/fold_0_manifest.json'),
    symbol='BTC/USDT',
    duration_days=1
)
```

### Validation Against Manifest

Paper trading automatically validates trades against model capabilities:

```python
# Model trained on BTC/USDT, 1h timeframe
manifest = ModelManifest.load('models/fold_0_manifest.json')

# ✓ Valid - matches training
await tester.run_paper_test(symbol='BTC/USDT', timeframe='1h')

# ✗ Invalid - model not trained on this
await tester.run_paper_test(symbol='DOGE/USDT', timeframe='15m')
# Raises: ValueError("Model cannot trade DOGE/USDT on 15m")
```

### Results Format

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "1h",
  "duration_days": 7,
  "start_balance": 10000.0,
  "end_balance": 10543.21,
  "total_trades": 23,
  "winning_trades": 15,
  "losing_trades": 8,
  "win_rate": 0.652,
  "total_fees": 23.45,
  "sharpe_ratio": 1.82,
  "max_drawdown": -0.045,
  "message": "Paper trading completed successfully"
}
```

### Integration with Training UI

The paper trading system is designed to integrate with the training UI for easy model testing:

```python
# After training completes
manifest = trainer.get_best_model_manifest()
tester = ModelPaperTester(manifest)

# Quick test
results = await tester.run_paper_test(duration_days=1)

# Show results in UI
self.display_paper_test_results(results)
```

### Best Practices

#### 1. Test Before Live Trading

```python
# 1. Train model
trainer = WalkForwardTrainer(config)
results = await trainer.train()

# 2. Paper test best model
manifest = ModelManifest.load('models/walk_forward/fold_0_manifest.json')
tester = ModelPaperTester(manifest)
paper_results = await tester.run_paper_test(duration_days=7)

# 3. Validate performance
if paper_results['sharpe_ratio'] > 1.5:
    # 4. Approve for live trading
    manifest.approved_for_live = True
    manifest.save()
```

#### 2. Progressive Testing

```python
# Start with short duration
results_1day = await tester.run_paper_test(duration_days=1)
if results_1day['win_rate'] > 0.55:
    # Extend to full week
    results_7day = await tester.run_paper_test(duration_days=7)
```

#### 3. Multi-Asset Testing

```python
# Test on all trained symbols
for symbol in manifest.capabilities.symbols:
    results = await tester.run_paper_test(
        duration_days=7,
        symbol=symbol
    )
    print(f"{symbol}: Win Rate {results['win_rate']:.2%}")
```

### Current Limitations

⚠️ **Note:** The paper trading integration is a framework. Full implementation requires:

1. **Market Data Integration:** Connect to real-time or historical data feed
2. **Model Loading:** Load trained PyTorch/TensorFlow models
3. **Signal Generation:** Run model inference to generate trading signals
4. **Order Execution:** Execute simulated trades through paper engine
5. **Performance Tracking:** Calculate metrics from actual trades

**Current State:** Structure and API are complete. Core simulation logic is placeholder.

**Roadmap:**
- Phase 1: ✅ API design and integration points
- Phase 2: 🔄 Market data connection
- Phase 3: 🔄 Model inference pipeline
- Phase 4: 🔄 Full simulation engine

---

## Security Checklist

Before deploying to production:

### PIN Security
- [ ] PIN setup dialog tested on first boot
- [ ] Weak PINs properly rejected
- [ ] PIN hash file has 600 permissions
- [ ] PIN hash file is gitignored
- [ ] Cannot skip PIN setup (app exits)
- [ ] PIN used for API key encryption

### API Key Security
- [ ] API keys stored encrypted
- [ ] Encrypted file is gitignored
- [ ] API key manager uses PIN hash as password
- [ ] Test connection works for all exchanges
- [ ] Keys persist across sessions

### Paper Trading Security
- [ ] Models validated against manifest before testing
- [ ] Paper engine properly isolated from live trading
- [ ] Cannot accidentally enable live trading during paper test
- [ ] Results properly saved and tracked

### File Protection
- [ ] `.gitignore` updated with specific patterns
- [ ] No sensitive files committed to git
- [ ] Source code files properly tracked
- [ ] Backup files gitignored

---

## Troubleshooting

### PIN Setup Issues

**Problem:** PIN dialog doesn't appear

**Solution:**
1. Check if `config/.pin_hash` already exists
2. Delete file to force PIN setup
3. Restart application

**Problem:** "Setup Error" on PIN save

**Solution:**
1. Check `config/` directory exists and is writable
2. Check file permissions
3. Review logs for specific error

### API Key Issues

**Problem:** API keys not loading after PIN change

**Solution:**
1. Old API keys were encrypted with old PIN hash
2. Re-enter API keys after PIN change
3. Keys will be encrypted with new PIN hash

### Paper Trading Issues

**Problem:** "Model cannot trade" error

**Solution:**
1. Check model manifest capabilities
2. Verify symbol/timeframe match training
3. Use `manifest.capabilities` to see what's allowed

**Problem:** No results after paper test

**Solution:**
1. Check logs for errors
2. Verify market data connection
3. Ensure model files exist and are readable

---

## FAQ

### Q: What happens if I forget my PIN?

**A:** You'll need to delete `config/.pin_hash` and set a new PIN. You'll also need to re-enter all API keys as they're encrypted with the old PIN hash.

### Q: Can I change my PIN after initial setup?

**A:** Currently, you'd need to delete `config/.pin_hash` and restart. A "Change PIN" feature could be added to settings.

### Q: Is paper trading free?

**A:** Yes! Paper trading uses no real money. It's purely simulated.

### Q: Can I test multiple models simultaneously?

**A:** Yes, use `test_all_models()` to batch test all models in a directory.

### Q: How accurate is paper trading?

**A:** Paper trading simulates slippage and fees but can't perfectly replicate live market conditions (liquidity, order execution, etc.). Use it for initial validation, then paper trade live before going fully live.

---

## Summary

This update adds three critical features:

1. **Mandatory PIN Setup**
   - Forces secure PIN on first boot
   - Cannot be skipped
   - Strength validation
   - Encrypts API keys

2. **Fixed .gitignore**
   - Specific patterns only
   - Protects sensitive files
   - Allows source code tracking

3. **Paper Trading Integration**
   - Risk-free model testing
   - Manifest validation
   - Batch testing support
   - Results export

All three features work together to create a more secure and testable trading platform.

---

**Version:** 1.0.0
**Last Updated:** 2025-11-15
**Maintainer:** Nexlify Development Team
