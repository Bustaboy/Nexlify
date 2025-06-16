# 🌃 Night City Trader - Quick Reference

## 🚀 Quick Start Commands

```bash
# 1. Setup (first time only)
cd C:\Night-City-Trader
python setup_night_city.py

# 2. Launch trader
python launch.py
# OR
start_night_city.bat

# 3. First Launch:
# - GUI will show onboarding screen
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
| `config/config.json` | Auto-generated settings | Via GUI |
| `.env` | Environment variables | Yes |
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
| Logs | Last tab | Real-time system logs |

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

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| GUI won't start | Make sure neural_net.py is running first |
| No pairs showing | Wait 5 minutes for initial scan |
| API errors | Check API keys in API CONFIG tab |
| High fees | Normal - all fees are calculated |
| Can't login | Default PIN is 2077 |
| Need more logs | Enable Debug Mode in Environment tab |
| Want notifications | Set up Telegram in Environment tab |

## 📈 Performance Tips

1. **Start Small**: Use minimum trade amounts
2. **Monitor Fees**: Check profit calculations include all fees
3. **Watch Confidence**: Higher AI confidence = better trades
4. **Diversify**: Let AI manage multiple pairs
5. **Be Patient**: Best profits come from many small trades

## 🔧 Advanced Commands

```bash
# View logs in real-time
tail -f logs/neural_net.log

# Backup configuration
copy config\config.json config\config.backup.json

# Check Python version
python --version

# Update packages
pip install -r requirements.txt --upgrade

# Run in verbose mode
set DEBUG=True && python arasaka_neural_net.py
```

## 📤 GitHub Commands

```bash
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin YOUR_GITHUB_URL
git push -u origin main

# Updates
git add .
git commit -m "Update description"
git push
```

## ⚠️ Safety Checklist

- [ ] Changed default PIN
- [ ] Set BTC wallet address
- [ ] Tested on testnet first
- [ ] Understand fee structure
- [ ] Set appropriate risk level
- [ ] Know how to use kill switch
- [ ] Have backup of config

## 🆘 Emergency Contacts

- **Logs**: `C:\Night-City-Trader\logs\neural_net.log`
- **Config**: `C:\Night-City-Trader\config\config.json`
- **Stop File**: Delete `C:\Night-City-Trader\EMERGENCY_STOP_ACTIVE`

---

**Remember**: In Night City, patience and caution keep you alive! 🌃🤖
