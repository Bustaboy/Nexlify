# 🌃 Nexlify Quick Reference Guide v2.0.8

## 🚀 Quick Start Commands

### Initial Setup
```bash
# First time setup
python setup_nexlify.py

# Start the platform
python smart_launcher.py        # Recommended - auto resource management
python start_nexlify.py         # Alternative - cross-platform
start_nexlify.bat              # Windows only
./start_nexlify.sh             # Linux/Mac only
```

### Stop Commands
```bash
# Graceful shutdown (choose one)
1. Press Ctrl+C in launcher window
2. Type 'stop' in interactive console
3. Click Kill Switch in GUI
4. Create EMERGENCY_STOP_ACTIVE file
```

## 📋 Common Tasks

### GUI Operations

#### First Time Setup
1. Start launcher: `python smart_launcher.py`
2. Wait for "All systems operational" message
3. GUI opens automatically
4. Default PIN: `2077` (change immediately!)
5. Follow setup wizard

#### Configure Exchange
1. Go to **Settings → Exchanges**
2. Select exchange from dropdown
3. Enter API Key and Secret
4. Test connection
5. Click Save

#### Start Trading
1. **Dashboard** → Check system status
2. **Trading** → Select pairs
3. **Settings** → Configure risk level
4. Click **Start Trading**

#### Monitor Performance
- **Dashboard**: Real-time overview
- **Analytics**: Detailed metrics
- **Audit Trail**: All transactions
- **Profit Matrix**: Visual profit map

### Configuration

#### Change Trading Settings
1. **Settings → Trading**
2. Adjust parameters:
   - Risk Level: Low/Medium/High
   - Max Position Size
   - Stop Loss %
   - Take Profit %
3. Click Save

#### Setup Notifications
1. **Settings → Notifications**
2. Configure:
   - Telegram Bot Token
   - Telegram Chat ID
   - Email settings
3. Test notifications
4. Save

#### Enable Security Features
1. **Settings → Security**
2. Options:
   - Change PIN
   - Enable 2FA (optional)
   - Set master password (optional)
   - IP whitelist
3. Apply changes

## 🛠️ Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| GUI won't start | Ensure launcher shows "All systems operational" |
| No trading pairs | Wait 2-3 minutes for initial market scan |
| API errors | Verify API keys in Settings → Exchanges |
| High fees shown | Normal - all fees are calculated transparently |
| Can't login | Default PIN: 2077, check caps lock |
| No notifications | Configure in Settings → Notifications |
| Database locked | Close duplicate instances |
| Performance slow | Reduce active pairs, check system resources |

### Quick Fixes

#### Reset Configuration
```bash
# Backup current config
cp enhanced_config.json enhanced_config.backup

# Reset to defaults
python setup_nexlify.py --reset
```

#### Clear Logs
```bash
# Windows
forfiles /p logs /s /m *.log /d -30 /c "cmd /c del @path"

# Linux/Mac
find logs -name "*.log" -mtime +30 -delete
```

#### Check System Health
```bash
# Run health check
python scripts/health_check.py

# View logs
tail -f logs/errors/error.log

# Check API status
curl http://127.0.0.1:8000/health
```

### Emergency Procedures

#### Emergency Stop
1. Create file named `EMERGENCY_STOP_ACTIVE` in root directory
2. Or press Ctrl+C in startup console
3. Or click Kill Switch in GUI

#### Data Recovery
1. Backups are in `backups/` directory
2. Run: `python scripts/backup.py` for manual backup
3. Database files in `data/` directory

#### Reset to Defaults
1. Stop Nexlify
2. Rename `enhanced_config.json` to `enhanced_config.backup`
3. Run setup again: `python setup_nexlify.py`

## 📊 Performance Optimization

### System Requirements
- **Minimum**: 8GB RAM, 4 CPU cores, 20GB disk
- **Recommended**: 16GB RAM, 8 CPU cores, 50GB disk
- **GPU**: NVIDIA GTX 1060+ for ML features

### Optimization Tips
1. **Reduce Active Pairs**: Limit to 10-20 pairs
2. **Adjust Scan Frequency**: Increase interval in quiet markets
3. **Disable Unused Features**: Turn off AI/Mobile if not needed
4. **Use SSD**: Database on SSD improves performance
5. **Close Unused Tabs**: Reduces GUI memory usage

## 🔐 Security Best Practices

1. **Change default PIN immediately**
2. **Enable 2FA for admin access**
3. **Use unique API keys per exchange**
4. **Rotate API keys monthly**
5. **Enable IP whitelist for production**
6. **Keep audit logs for compliance**
7. **Regular backups to external storage**
8. **Monitor logs/errors/ for anomalies**

## 📱 Mobile App

### Setup
1. Go to Mobile tab in GUI
2. Click "Generate QR Code"
3. Scan with Nexlify mobile app
4. Enter pairing code
5. Confirm on both devices

### Features
- Real-time portfolio monitoring
- Trade execution
- Emergency stop
- Push notifications
- Secure WebSocket connection

## 🤖 AI Companion

### Commands
- "Show market analysis" - Current market overview
- "Analyze BTC/USDT" - Specific pair analysis
- "What's my performance?" - Trading statistics
- "Explain arbitrage" - Educational content
- "Execute trade..." - Trade with confirmation

### Tips
- Be specific with commands
- AI considers your risk settings
- Always verify trade suggestions
- Use for education and insights

## 📈 Backtesting

### Quick Backtest
1. Go to Analytics → Backtesting
2. Select strategy and timeframe
3. Configure parameters
4. Click "Run Backtest"
5. Review results and metrics

### Important Metrics
- **Sharpe Ratio**: Risk-adjusted returns (>1 good)
- **Max Drawdown**: Largest loss from peak
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss

## 🆘 Support Resources

### Logs Location
- **System**: `logs/errors/`
- **Trading**: `logs/trading/`
- **Audit**: `logs/audit/`
- **Startup**: `logs/startup/`
- **Crash Reports**: `logs/crash_reports/`

### Configuration Files
- **Main Config**: `enhanced_config.json` (GUI managed)
- **System Secrets**: `.env` (never edit)
- **Legacy**: `neural_config.json` (compatibility)

### Getting Help
1. Check this quick reference
2. Review `docs/QUICKSTART.md`
3. Check logs for specific errors
4. Run system health check: `python scripts/health_check.py`

## 🎯 Pro Tips

1. **Start Small**: Test with small amounts first
2. **Use Testnet**: Configure exchanges for testnet
3. **Monitor Actively**: Check dashboard regularly
4. **Set Alerts**: Configure notifications for important events
5. **Review Audit**: Check audit trail weekly
6. **Update Regularly**: Keep exchange APIs current
7. **Backup Often**: Automate backups with cron/scheduler

---
*Last Updated: v2.0.8 - Night City Trading Platform*
