# Nexlify Quick Reference Guide v2.0.8

## 🚀 Quick Start

1. **First Run**: `python start_nexlify.py` (or `start_nexlify.bat` on Windows)
2. **Default PIN**: 2077 (change on first login)
3. **Main API**: http://localhost:8000
4. **Mobile API**: http://localhost:8001

## 🎮 GUI Navigation

### Main Tabs
- **Dashboard**: System overview, active pairs, market status
- **Trading Matrix**: Active positions, smart routing controls
- **Profit Chart**: Performance visualization, metrics
- **Settings**: Configuration for all features
- **Security**: 2FA, API keys, IP whitelist
- **Exchanges**: Exchange credentials and settings
- **Analytics**: Backtesting, performance analysis
- **Audit Trail**: Compliance logs, blockchain verification
- **AI Companion**: Trading assistant chat
- **Mobile**: QR code pairing, device management
- **Logs**: System logs with filtering
- **Error Report**: Crash reports and diagnostics

### Keyboard Shortcuts
- `F1`: Show help
- `F5`: Refresh data
- `Ctrl+S`: Save settings
- `Ctrl+Q`: Quit application
- `ESC`: Cancel current operation

## ⚙️ Configuration

All settings are managed through the GUI. **Never edit .env or config files directly!**

### Exchange Setup
1. Go to Settings → Exchanges
2. Select exchange from dropdown
3. Enter API credentials
4. Click "Test Connection"
5. Save settings

### Notifications Setup
1. Go to Settings → Notifications
2. Enable desired services:
   - **Telegram**: Enter bot token and chat ID
   - **Email**: Configure SMTP settings
   - **Webhook**: Add webhook URL
3. Test notifications
4. Save settings

### DeFi/DEX Setup (Full Installation)
1. Go to Settings → DeFi
2. Enter Ethereum RPC URL (e.g., Alchemy)
3. Add private key for transactions
4. Configure gas settings
5. Save settings

### Security Configuration
1. Go to Security tab
2. **Change PIN** from default (2077)
3. **Enable 2FA** (recommended):
   - Click "Enable 2FA"
   - Scan QR code with authenticator
   - Enter verification code
4. **API Key Rotation**: Set rotation schedule
5. **IP Whitelist**: Add allowed IPs (optional)

## 🔧 Troubleshooting

### Common Issues

#### "Database locked" error
- **Cause**: Multiple instances running
- **Fix**: 
  1. Close all Nexlify instances
  2. Check Task Manager/Activity Monitor
  3. Delete `EMERGENCY_STOP_ACTIVE` if exists
  4. Restart

#### GUI not opening
- **Windows**: Install Visual C++ Redistributable
- **Linux**: Install tkinter: `sudo apt-get install python3-tk`
- **All**: Check `logs/startup/` for errors

#### API connection failed
- **Check**: Port conflicts on 8000
- **Fix**: Change port in Settings → Advanced
- **Verify**: Firewall allows connections

#### Exchange connection errors
- **Verify**: API credentials are correct
- **Check**: Exchange requires IP whitelist
- **Test**: Use exchange testnet first

#### High memory usage
- **Cause**: Log accumulation
- **Fix**: 
  1. Clear old logs in `logs/` directory
  2. Reduce log retention in Settings
  3. Restart application

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
