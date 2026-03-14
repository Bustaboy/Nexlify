# 🌃 Nexlify - Implementation Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [First Launch Configuration](#first-launch-configuration)
4. [Exchange Setup](#exchange-setup)
5. [Advanced Configuration](#advanced-configuration)
6. [Troubleshooting](#troubleshooting)
7. [Architecture Overview](#architecture-overview)

## 🔧 Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.12 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB free space
- **Internet**: Stable connection required

### Python Dependencies
All dependencies are in `requirements.txt` and include:
- ccxt (exchange connectivity)
- pandas/numpy (data processing)
- tensorflow/scikit-learn (AI models)
- aiohttp/websockets (async operations)

## 🚀 Installation Steps

### 1. Clone or Download Repository
```bash
git clone https://github.com/yourusername/nexlify.git
cd nexlify
```

### 2. Run Setup Script
```bash
python setup_nexlify.py
```
This will:
- Check Python version
- Create directory structure
- Install all dependencies
- Create initial config files

### 3. Copy Core Files
**IMPORTANT**: Copy these files from Night-City-Trader:
- `nexlify/core/arasaka_neural_net.py` → Main trading engine
- `nexlify/gui/cyber_gui.py` → GUI interface
- `nexlify/utils/error_handler.py` → Error management
- `nexlify/utils/utils_module.py` → Utility functions

### 4. Launch the System
```bash
# Launcher (recommended)
python nexlify_launcher.py

# OR Windows batch file
start_nexlify.bat
```

## 🎮 First Launch Configuration

### Step 1: Onboarding Screen
On first launch, the GUI displays an onboarding wizard:

1. **Welcome Screen**
   - Overview of features
   - Risk disclaimer

2. **API Configuration**
   ```
   Exchange: Binance (or your choice)
   API Key: [Your API Key]
   Secret Key: [Your Secret Key]
   ☑ Enable Testnet (recommended initially)
   ```

3. **Wallet Setup**
   ```
   BTC Wallet Address: [Your BTC Address]
   ☑ Enable Auto-Withdrawal
   Min Amount: $100
   ```

4. **Risk Configuration**
   - **Low**: Conservative, 0.5% profit targets
   - **Medium**: Balanced, 1% profit targets
   - **High**: Aggressive, 2%+ profit targets

5. **Complete Setup**
   - Click "JACK INTO THE MATRIX"
   - System initializes and starts scanning

### Step 2: PIN Security
- Set a secure PIN during first launch
- Change PIN anytime in Settings tab
- 5 failed attempts = 5-minute lockout

## 🔐 Exchange Setup

### Binance Setup
1. Log into Binance
2. Go to API Management
3. Create new API key
4. Set permissions:
   - ☑ Enable Reading
   - ☑ Enable Spot Trading
   - ☑ Enable Withdrawal (optional)
   - ☐ Disable Margin/Futures
5. Whitelist IP (recommended)
6. Copy API Key and Secret

### Other Exchanges
Similar process for:
- **Coinbase Pro**: API → New API Key
- **Kraken**: Settings → API → Generate Key
- **KuCoin**: API Management → Create API

### API Security Best Practices
- Use IP whitelisting when possible
- Enable only required permissions
- Use separate API keys for testing
- Rotate keys periodically
- Never share or commit keys

## ⚙️ Advanced Configuration

### Neural Network Settings
Located in Settings tab:
```json
{
  "confidence_threshold": 0.7,    // Min AI confidence
  "retrain_interval_hours": 168,  // Weekly retraining
  "max_concurrent_trades": 5      // Parallel trades
}
```

### Trading Parameters
```json
{
  "min_profit_percent": 0.5,     // After all fees
  "max_position_size": 0.1,      // 10% of balance
  "stop_loss": 0.02,             // 2% stop loss
  "take_profit": 0.05            // 5% take profit
}
```

### Environment Variables
Configure in Settings tab (Environment section):
- **Debug Mode**: Verbose logging
- **Log Level**: INFO/DEBUG/ERROR
- **API Port**: Default 8000
- **Database**: SQLite default
- **Notifications**: Telegram/Email

### Custom Strategies
To add custom strategies:
1. Edit `nexlify/core/arasaka_neural_net.py`
2. Add strategy to `STRATEGIES` dict
3. Implement in `calculate_signals()`
4. Restart system

## 🐛 Troubleshooting

### Common Issues

#### "Cannot connect to API"
```bash
# Check if API is running
curl http://127.0.0.1:8000/health

# Check logs
tail -f logs/neural_net.log
```

#### "No trading pairs found"
- Verify API keys are correct
- Check exchange connectivity
- Ensure balance > minimum trade
- Wait 5 minutes for initial scan

#### "High fees eating profits"
- This is normal - all fees calculated
- Increase min_profit_percent
- Use limit orders when possible
- Check exchange fee tier

#### Database Errors
```bash
# Reset database
rm data/trading.db
python nexlify_launcher.py
```

#### GUI Won't Start
1. Ensure API is running first
2. Check Python version (3.12+)
3. Verify all dependencies installed
4. Check error logs

### Log File Locations
- **Main Log**: `logs/neural_net.log`
- **Error Log**: `logs/error.log`
- **Crash Reports**: `logs/crash_reports/`
- **Trade History**: `data/trades.csv`

## 🏗️ Architecture Overview

### System Components

```
Nexlify Trading System
├── API Layer (nexlify/core/arasaka_neural_net.py)
│   ├── FastAPI Server
│   ├── Exchange Connectors
│   ├── Trading Engine
│   └── ML Models
├── GUI Layer (nexlify/gui/cyber_gui.py)
│   ├── PyQt5 Interface
│   ├── Real-time Updates
│   ├── Configuration Management
│   └── Monitoring Dashboard
├── Data Layer
│   ├── SQLite Database
│   ├── Model Storage
│   └── Log Management
└── Security Layer
    ├── API Key Encryption
    ├── PIN Protection
    └── Error Handling
```

### Data Flow
1. **Market Data** → Exchange APIs → Neural Net
2. **Analysis** → ML Models → Confidence Scores
3. **Decisions** → Trading Engine → Order Execution
4. **Results** → Database → GUI Display

### Neural Network Pipeline
```python
Market Data → Feature Extraction → Neural Network
    ↓               ↓                    ↓
Price/Volume → Technical Indicators → Predictions
    ↓               ↓                    ↓
OrderBook → Pattern Recognition → Confidence Score
```

## 🚀 Performance Optimization

### Tips for Better Performance
1. **Use SSD** for database storage
2. **Limit concurrent trades** initially
3. **Start with major pairs** (BTC, ETH)
4. **Monitor memory usage** with many pairs
5. **Use testnet** for strategy testing

### Scaling Considerations
- Each exchange connection uses ~100MB RAM
- Neural net retraining uses ~2GB RAM
- Database grows ~10MB per day
- Log rotation recommended weekly

## 🔒 Security Checklist

- [ ] Set secure PIN
- [ ] API keys have minimum permissions
- [ ] Testnet verified before mainnet
- [ ] Backup of configuration created
- [ ] Emergency stop tested
- [ ] Logs don't contain sensitive data
- [ ] .gitignore properly configured

## 📞 Getting Help

### Resources
- **Logs**: Primary debugging tool
- **GUI Error Tab**: Real-time error display
- **Documentation**: This guide + README
- **Code Comments**: Inline documentation

### Emergency Procedures
1. **Kill Switch**: Red button in GUI
2. **Manual Stop**: Delete `EMERGENCY_STOP_ACTIVE`
3. **Force Quit**: Close all terminals
4. **API Shutdown**: `Ctrl+C` in API window

---

*Remember: With Nexlify, knowledge is power. In trading, preparation is profit.* 🤖💰
