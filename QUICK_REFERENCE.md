# 🌃 Nexlify - Quick Reference

## 🚀 Quick Start Commands

```bash
# 1. Setup (first time only)
cd nexlify
python setup_nexlify.py

# 2. Launch trader
python nexlify_launcher.py
# OR
python launch.py
# OR
start_nexlify.bat

# 3. First Launch:
# - GUI shows onboarding screen
# - Enter your API keys
# - Set BTC wallet
# - Choose risk level
# - Click "JACK INTO THE MATRIX"

# 4. Default PIN: 2077
```

## 📁 Essential Files

| File | Purpose | Edit? |
|------|---------|-------|
| `arasaka_neural_net.py` | Main trading engine | No |
| `cyber_gui.py` | GUI interface | No |
| `config/neural_config.json` | Auto-generated settings | Via GUI |
| `.env` | Environment variables | Via GUI |
| `logs/neural_net.log` | System logs | View only |

## ⚙️ Key Configuration

### First Time Setup (via GUI)
1. **Exchange APIs**: Enter directly in onboarding screen
2. **BTC Wallet**: Set in onboarding or control panel
3. **Risk Level**: Choose during setup
4. **Testnet Mode**: Toggle per exchange

### Accessing API Config Later
- Go to "🔐 API CONFIG" tab in main GUI
- Update credentials for any exchange
- Click "SAVE [EXCHANGE]" to update

### Critical Settings
- **PIN**: Change from default `2077` in Settings tab
- **Testnet**: Keep enabled until ready for real trading
- **Risk Level**: Start with `"low"`
- **Min Withdrawal**: Default `$100`

## 🎮 GUI Controls

| Feature | Location | Action |
|---------|----------|--------|
| BTC Wallet | Control Panel | Enter address → SAVE |
| Auto-Trade | Control Panel | Toggle checkbox |
| Kill Switch | Control Panel | Red button (emergency) |
| Active Pairs | First tab | Auto-updates every 5 min |
| Profit Chart | Second tab | Shows 24h history |
| Settings | Third tab | Risk level, withdrawal |
| Environment | Fourth tab | Debug, logs, notifications |
| API Config | Fifth tab | Exchange credentials |
| Logs | Sixth tab | Real-time system logs |
| Error Report | Last tab | Error history & stats |

## 📊 Understanding the Display

### Active Pairs Panel Shows:
- **Symbol**: Trading pair (e.g., BTC/USDT)
- **Profit %**: Net profit after all fees
- **Volume**: 24h trading volume
- **Volatility**: Price movement indicator
- **AI Confidence**: Neural-net's confidence score
- **Status**: 🟢 Active or 🟡 Monitoring

### Color Coding:
- 🟢 **Green**: Profitable/Active
- 🟡 **Yellow**: Monitoring/Warning
- 🔴 **Red**: Loss/Danger
- 🔵 **Blue**: Neutral/Info

## 🛠️ Common Tasks

### Change BTC Wallet
1. Enter address in Control Panel
2. Click SAVE
3. Check logs for confirmation

### Adjust Risk Level
1. Go to Settings tab
2. Select risk level
3. Click SAVE CONFIGURATION

### View Profits
1. Check Profit Matrix tab
2. Green area = profits
3. Updates hourly

### Emergency Stop
1. Click KILL SWITCH
2. Confirm action
3. Delete `EMERGENCY_STOP_ACTIVE` to restart

### Enable Notifications
1. Go to Environment tab
2. Enter Telegram bot token
3. Enter chat ID
4. Click SAVE

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| GUI won't start | Make sure API is running first |
| No pairs showing | Wait 5 minutes for initial scan |
| API errors | Check API keys in API CONFIG tab |
| High fees | Normal - all fees are calculated |
| Can't login | Default PIN is 2077 |
| Need more logs | Enable Debug Mode in Environment |
| Want notifications | Set up Telegram in Environment |
| Database locked | Close duplicate instances |

## 📈 Performance Tips

1. **Start Small**: Use minimum trade amounts
2. **Monitor Fees**: Check profit calculations include all fees
3. **Watch Confidence**: Higher AI confidence = better trades
4. **Diversify**: Let AI manage multiple pairs
5. **Be Patient**: Best profits come from many small trades
6. **Use Limits**: Set max position size to manage risk

## 🔧 Advanced Commands

```bash
# View logs in real-time
tail -f logs/neural_net.log

# Check API health
curl http://127.0.0.1:8000/health

# Backup configuration
copy config\neural_config.json config\backup.json

# Check Python version
python --version

# Update packages
pip install -r requirements.txt --upgrade

# Run in debug mode
# Set Debug Mode = True in Environment tab

# Database operations
sqlite3 data/trading.db ".tables"
sqlite3 data/trading.db ".schema trades"

# Clean old logs (Windows)
forfiles /p logs /s /m *.log /d -30 /c "cmd /c del @path"

# Clean old logs (Linux/Mac)
find logs -name "*.log" -mtime +30 -delete
```

## 📤 GitHub Commands

```bash
# First time setup
git init
git add .
git commit -m "Initial commit - Nexlify v2.0.7.7"
git remote add origin YOUR_GITHUB_URL
git push -u origin main

# Updates
git add .
git commit -m "Update: description"
git push

# Create release
git tag -a v2.0.7.7 -m "Production release"
git push origin v2.0.7.7
```

## 🔍 Monitoring Commands

```bash
# Check running processes (Windows)
tasklist | findstr python

# Check running processes (Linux/Mac)
ps aux | grep python

# Monitor resource usage
# GUI: Check Task Manager / Activity Monitor
# CLI: Use htop (Linux/Mac) or Process Explorer (Windows)

# Check port usage
netstat -an | findstr 8000
```

## ⚠️ Safety Checklist

- [ ] Changed default PIN
- [ ] Set BTC wallet address
- [ ] Tested on testnet first
- [ ] Understand fee structure
- [ ] Set appropriate risk level
- [ ] Know how to use kill switch
- [ ] Have backup of config
- [ ] Reviewed error logs

## 🆘 Emergency Procedures

### Immediate Stop
1. **GUI**: Click red KILL SWITCH button
2. **File**: Create `EMERGENCY_STOP_ACTIVE` file
3. **Terminal**: Press Ctrl+C in launcher window
4. **System**: Close all Python processes

### Recovery Steps
1. Check logs for last error
2. Verify no trades are pending
3. Delete `EMERGENCY_STOP_ACTIVE`
4. Restart with launcher
5. Monitor closely for issues

### Data Recovery
```bash
# Restore config backup
copy config\backup.json config\neural_config.json

# Export trade history
sqlite3 data/trading.db "SELECT * FROM trades" > trades_backup.csv

# Reset database (last resort)
move data\trading.db data\trading_old.db
python nexlify_launcher.py
```

## 📞 Quick Debug Info

When reporting issues, provide:
1. **Error message** from GUI or console
2. **Last 50 lines** of neural_net.log
3. **Python version**: `python --version`
4. **OS**: Windows/Mac/Linux version
5. **Time** when error occurred
6. **Actions** taken before error

---

**Remember**: In Night City, quick reflexes save lives. In trading, quick reference saves profits! 🌃🤖💰
