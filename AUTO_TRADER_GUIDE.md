# 🤖 Nexlify Auto-Execution Engine Guide

## Overview

The Auto-Execution Engine transforms Nexlify from **semi-autonomous** to **fully autonomous** trading. It monitors opportunities 24/7 and executes trades automatically based on neural net signals and risk management rules.

---

## 🚀 How It Works

### 1. **Opportunity Monitor** (30-second cycle)
```
Neural Net → Finds profitable pairs
     ↓
Risk Manager → Validates opportunity
     ↓
Auto-Trader → Executes trade automatically
```

### 2. **Position Manager** (10-second cycle)
```
Active Positions → Monitor prices
     ↓
Check Exit Rules → TP/SL/Trailing/Time
     ↓
Auto-Close → Execute sell order
```

### 3. **Performance Reporter** (hourly)
```
Statistics → Win rate, profit, trades
     ↓
Logging → Performance reports
```

---

## ⚙️ Configuration

### Enable Auto-Trading
**config/neural_config.json:**
```json
{
  "auto_trade": true,
  "trading": {
    "min_profit_percent": 0.5,
    "max_position_size": 100,
    "max_concurrent_trades": 5,
    "max_daily_loss": 100,
    "take_profit": 5.0,
    "stop_loss": 2.0,
    "trailing_stop": 3.0,
    "max_hold_time_hours": 24,
    "min_confidence": 0.7
  }
}
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auto_trade` | false | Master switch for auto-trading |
| `min_profit_percent` | 0.5% | Minimum profit to execute trade |
| `max_position_size` | $100 | Maximum size per trade (USD) |
| `max_concurrent_trades` | 5 | Max open positions at once |
| `max_daily_loss` | $100 | Auto-stop if daily loss exceeds |
| `take_profit` | 5.0% | Profit target for exits |
| `stop_loss` | 2.0% | Maximum loss per trade |
| `trailing_stop` | 3.0% | Trailing stop from peak |
| `max_hold_time_hours` | 24 | Force close after time |
| `min_confidence` | 0.7 | Minimum neural confidence (0-1) |

---

## 🎯 Risk Management

### Pre-Trade Checks
Before executing ANY trade, the system validates:

✅ **Daily Loss Limit** - Has daily loss exceeded limit?
✅ **Concurrent Trades** - Are we at max positions?
✅ **Profit Threshold** - Is profit > minimum?
✅ **Confidence Score** - Is neural confidence high enough?
✅ **Available Balance** - Do we have sufficient funds?

**If ANY check fails → Trade is skipped**

### Position Sizing
Uses simplified Kelly Criterion:
```python
position_size = min(
    balance * (risk_percent / 100),
    max_position_size
)
```

Default: 2% risk per trade, capped at $100

---

## 🔄 Exit Strategies

### 1. Take Profit (Default: +5%)
```
Entry: $100
Target: $105 (+5%)
Action: Sell when price hits $105
```

### 2. Stop Loss (Default: -2%)
```
Entry: $100
Stop: $98 (-2%)
Action: Sell when price hits $98
```

### 3. Trailing Stop (Default: 3%)
```
Entry: $100
Price rises to $110 (+10%)
Trailing stop: $106.70 (3% from peak)
Action: Sell if price drops below $106.70
```

### 4. Time-Based Exit (Default: 24 hours)
```
Entry: Monday 10:00 AM
Max hold: 24 hours
Action: Force close Tuesday 10:00 AM regardless of PnL
```

### Exit Priority
1. Stop Loss (immediate)
2. Take Profit (immediate)
3. Trailing Stop (for profitable positions)
4. Max Hold Time (forced)

---

## 📊 Trading Flow

### Execution Sequence

```
1. SCAN (Neural Net)
   ├─ Discover pairs
   ├─ Rank by profit
   └─ Score by confidence

2. VALIDATE (Risk Manager)
   ├─ Check daily loss
   ├─ Check concurrent trades
   ├─ Check thresholds
   └─ Approve/Reject

3. EXECUTE (Auto-Trader)
   ├─ Get balance
   ├─ Calculate size
   ├─ Place market buy
   ├─ Set exit levels
   └─ Log to audit trail

4. MONITOR (Position Manager)
   ├─ Check price every 10s
   ├─ Evaluate exit rules
   ├─ Execute close if triggered
   └─ Update statistics

5. REPORT (Performance Reporter)
   └─ Hourly performance logs
```

---

## 🛑 Emergency Controls

### Disable Auto-Trading
**Option 1: Config File**
```json
{
  "auto_trade": false
}
```

**Option 2: Code**
```python
neural_net.toggle_auto_trading(False)
```

**Option 3: Kill Switch**
- GUI "KILL SWITCH" button stops everything
- Closes all positions
- Disables auto-trading

### Emergency Position Close
```python
# Close all positions immediately
await auto_trader.close_all_positions()
```

---

## 📈 Performance Monitoring

### View Statistics
```python
stats = neural_net.get_auto_trader_stats()

# Returns:
{
    'total_trades': 42,
    'winning_trades': 28,
    'losing_trades': 14,
    'win_rate': 66.67,
    'total_profit': 145.50,
    'active_positions': 3,
    'daily_profit': 23.40,
    'auto_trade_enabled': True
}
```

### Hourly Reports
Auto-trader logs performance every hour:
```
==================================================
📊 AUTO-TRADER PERFORMANCE REPORT
==================================================
Total Trades: 42
Win Rate: 66.67%
Total Profit: $145.50
Avg Profit/Trade: $3.46
Active Positions: 3
==================================================
```

---

## 🔍 Logging

### Log Levels
All auto-trader actions are logged:

**INFO** - Normal operations
```
🎯 Trade opportunity: BTC/USDT (All checks passed)
📈 Executing BUY: 0.002100 BTC/USDT @ $45000.00
✅ Trade executed: BTC/USDT - TP: $47250.00, SL: $44100.00
```

**WARNING** - Skipped opportunities
```
⏭️ Skipping ETH/USDT: Confidence too low: 0.65 < 0.70
⚠️ Daily loss limit reached: $105.00
```

**ERROR** - Failed operations
```
❌ Trade failed: Order rejected by exchange
❌ Close failed: Insufficient balance
```

---

## 🧪 Testing

### Testnet Mode (REQUIRED FIRST)
1. Configure exchange with testnet credentials
2. Enable auto-trading
3. Monitor for 24-48 hours
4. Verify:
   - Trades execute correctly
   - Exits trigger properly
   - Risk limits work
   - Statistics accurate

### Production Checklist
- [ ] Tested on testnet for 48+ hours
- [ ] Win rate > 55%
- [ ] Max drawdown < 10%
- [ ] All risk limits working
- [ ] Emergency stop tested
- [ ] Balance tracking accurate
- [ ] Audit logs complete

---

## ⚠️ Important Notes

### Position Entry
- Uses **market orders** (immediate execution)
- Slippage may occur on low-liquidity pairs
- Entry price is last trade price

### Position Exit
- Uses **market orders** (immediate exit)
- May not hit exact TP/SL levels
- Priority: Safety > Profit

### Balance Management
- Checks USDT balance before each trade
- Reserves funds for fees
- Won't overleverage

### Network Issues
- If exchange unreachable, skips cycle
- Retries on next check (30s for entries, 10s for exits)
- Logs all errors

---

## 🚨 Risk Warnings

1. **Auto-trading involves risk** - Can lose money quickly
2. **Start small** - Max position $10-50 initially
3. **Test thoroughly** - Weeks on testnet before production
4. **Monitor daily** - Check logs and positions regularly
5. **Set conservative limits** - Max daily loss, position sizes
6. **Keep kill switch ready** - Emergency stop always available
7. **API failures happen** - Exchange outages can strand positions
8. **Market volatility** - Sudden moves can exceed stop losses
9. **Gas fees add up** - Especially on high-frequency trading
10. **Not financial advice** - Use at your own risk

---

## 📚 Code Examples

### Basic Usage
```python
from arasaka_neural_net import ArasakaNeuralNet

# Load config with auto_trade enabled
config = {
    'auto_trade': True,
    'trading': {
        'min_profit_percent': 0.5,
        'max_position_size': 50,  # Start small!
        'max_concurrent_trades': 2,
        'stop_loss': 2.0
    }
}

# Initialize neural net (auto-trader starts automatically)
neural_net = ArasakaNeuralNet(config)
await neural_net.initialize()

# Auto-trader now running!
```

### Manual Control
```python
# Disable auto-trading
neural_net.toggle_auto_trading(False)

# Re-enable
neural_net.toggle_auto_trading(True)

# Get statistics
stats = neural_net.get_auto_trader_stats()
print(f"Win Rate: {stats['win_rate']:.2f}%")
print(f"Total Profit: ${stats['total_profit']:.2f}")

# Shutdown cleanly
await neural_net.shutdown()  # Closes all positions first
```

---

## 🎓 Best Practices

### Conservative Settings (Recommended for Start)
```json
{
  "auto_trade": true,
  "trading": {
    "min_profit_percent": 1.0,
    "max_position_size": 20,
    "max_concurrent_trades": 2,
    "max_daily_loss": 50,
    "take_profit": 3.0,
    "stop_loss": 1.5,
    "min_confidence": 0.75
  }
}
```

### Aggressive Settings (Advanced Users Only)
```json
{
  "auto_trade": true,
  "trading": {
    "min_profit_percent": 0.3,
    "max_position_size": 500,
    "max_concurrent_trades": 10,
    "max_daily_loss": 1000,
    "take_profit": 10.0,
    "stop_loss": 3.0,
    "min_confidence": 0.60
  }
}
```

---

## 🆘 Troubleshooting

### "No trades executing"
- Check `auto_trade: true` in config
- Verify neural net found opportunities
- Lower `min_profit_percent` or `min_confidence`
- Check available balance

### "Too many trades"
- Lower `max_concurrent_trades`
- Raise `min_profit_percent`
- Raise `min_confidence`
- Reduce `max_position_size`

### "Losses too high"
- Lower `max_position_size`
- Lower `max_concurrent_trades`
- Reduce `max_daily_loss`
- Tighten `stop_loss` percentage

### "Positions not closing"
- Check exchange connectivity
- Verify balance sufficient for sell
- Check logs for errors
- Manual close via GUI if needed

---

## 📞 Support

For issues:
1. Check `logs/neural_net.log`
2. Review auto-trader section in logs
3. Verify configuration
4. Test on testnet first
5. Report issues with log excerpts

---

**Built with 💚 for autonomous trading**
**Use responsibly. Trade at your own risk.**
