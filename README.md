# 🤖 Nexlify - AI-Powered Cryptocurrency Trading System

![Python](https://img.shields.io/badge/python-v3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

> Advanced neural network-based cryptocurrency trading platform with autonomous trading, multi-exchange support, and comprehensive risk management.

## ✨ Key Features

### Core Trading
- **🤖 Autonomous Trading**: AI-driven pair selection and execution using neural networks
- **📊 Multi-Exchange Support**: Trade across multiple exchanges simultaneously (Binance, Kraken, Coinbase, and more)
- **⚡ Real-Time Analysis**: Continuous market scanning and opportunity detection
- **💰 Smart Fee Calculation**: All trades automatically account for gas and exchange fees
- **🎯 Arbitrage Detection**: Automatic cross-exchange arbitrage opportunity identification

### Risk Management
- **🛡️ Advanced Risk Controls**: Customizable position sizing, stop-loss, and take-profit
- **⚠️ Circuit Breaker**: Automatic trading halt on consecutive failures
- **📉 Flash Crash Protection**: Multi-threshold detection with automatic response
- **🔴 Emergency Kill Switch**: Instant shutdown with position closing
- **📊 Kelly Criterion**: Optimal position sizing based on win probability

### Security & Monitoring
- **🔐 PIN Authentication**: Multi-layer security with lockout protection
- **🔍 Integrity Monitor**: File integrity verification and process monitoring
- **📝 Comprehensive Logging**: Detailed audit trails and error tracking
- **💾 Performance Tracking**: Trade history, analytics, and Sharpe ratio calculation
- **📱 Telegram Notifications**: Real-time alerts and system status updates

### Advanced Features
- **🏦 DeFi Integration**: Auto-compound idle funds in yield protocols (Aave, Uniswap)
- **💸 Profit Management**: Automated withdrawal and cold wallet integration
- **📈 Tax Reporting**: Built-in trade logging for tax compliance (FIFO, LIFO)
- **🎨 Cyberpunk GUI**: Full-featured interface with real-time charts and controls

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Exchange API keys (for trading)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Nexlify.git
cd Nexlify
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the system**
```bash
# Copy example configuration
cp config/neural_config.example.json config/neural_config.json

# Edit configuration with your settings
# Set your PIN, exchange APIs, wallet addresses, etc.
```

4. **Launch Nexlify**
```bash
# Using the launcher (recommended)
python nexlify_launcher.py

# Or on Windows
start_nexlify.bat
```

5. **First Time Setup**
- GUI will show onboarding screen
- Enter your exchange API keys
- Set your BTC wallet address
- Choose risk level (start with "low")
- Set a secure PIN
- Click "JACK INTO THE MATRIX"

## 📖 Documentation

- **[Quick Reference Guide](docs/QUICK_REFERENCE.md)** - Common commands and tasks
- **[Implementation Guide](docs/IMPLEMENTATION_GUIDE.md)** - Detailed setup and configuration
- **[GUI Features](docs/GUI_FEATURES.md)** - Complete GUI functionality reference
- **[Trading Integration](docs/TRADING_INTEGRATION.md)** - Trading engine integration details
- **[Auto-Trader Guide](docs/AUTO_TRADER_GUIDE.md)** - Automated trading setup
- **[Environment Settings](docs/ENVIRONMENT_SETTINGS.md)** - Configuration options
- **[Advanced Features](docs/ADVANCED_FEATURES_GUIDE.md)** - DeFi, profit management, and more
- **[Enhancements Guide](docs/ENHANCEMENTS_GUIDE.md)** - Feature enhancements and updates

## 🎮 Basic Usage

### Starting the System
```bash
python nexlify_launcher.py
```

### Key GUI Tabs
- **Active Pairs**: Real-time trading pairs and performance
- **Profit Chart**: 24-hour profit visualization
- **Settings**: Risk management and system configuration
- **Environment**: Debug mode, logging, notifications
- **API Config**: Exchange credentials management
- **Logs**: Real-time system logs

### Emergency Stop
1. Click the red **KILL SWITCH** button in the GUI
2. All active trades will be closed
3. System will shut down safely

## ⚙️ Configuration

Main configuration file: `config/neural_config.json`

Key settings:
- **Risk Management**: Position sizing, stop-loss, take-profit
- **Circuit Breaker**: Failure thresholds and timeouts
- **Flash Crash Protection**: Price drop thresholds
- **DeFi Integration**: Yield protocol settings
- **Security**: PIN requirements and authentication
- **Environment**: API port, database, logging

See [neural_config.example.json](config/neural_config.example.json) for all available options.

## 🔧 Architecture

### Core Components
- **arasaka_neural_net.py** - Main trading engine with AI decision-making
- **cyber_gui.py** - Full-featured graphical interface
- **nexlify_launcher.py** - System launcher with health checks
- **error_handler.py** - Centralized error management

### Feature Modules
- **nexlify_risk_manager.py** - Position sizing and risk controls
- **nexlify_advanced_security.py** - Security and authentication
- **nexlify_emergency_kill_switch.py** - Emergency shutdown system
- **nexlify_flash_crash_protection.py** - Market crash detection
- **nexlify_pin_manager.py** - PIN authentication
- **nexlify_integrity_monitor.py** - File and process monitoring
- **nexlify_defi_integration.py** - DeFi yield protocols
- **nexlify_profit_manager.py** - Automated profit handling
- **nexlify_tax_reporting.py** - Trade logging for taxes

## 🛡️ Security Best Practices

- ✅ Set a strong PIN (not sequential or repeated digits)
- ✅ Use API keys with minimum required permissions
- ✅ Test thoroughly on testnet before mainnet
- ✅ Keep `config/neural_config.json` out of version control
- ✅ Enable integrity monitoring for critical files
- ✅ Set up Telegram notifications for alerts
- ✅ Regularly backup your configuration and database
- ✅ Monitor logs for unusual activity

## 📊 Performance Tracking

The system automatically tracks:
- Trade history and P&L
- Win rate and average profit
- Sharpe ratio calculation
- Maximum drawdown
- Fee analysis
- Exchange performance comparison

Data is stored in `data/trading.db` (SQLite) and can be exported to JSON/CSV.

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_risk_manager.py

# Run with coverage
python -m pytest --cov=. tests/
```

### Backtest Mode
Use testnet mode on exchanges to validate strategies without real funds.

## 📈 Monitoring

### View Logs
```bash
# Real-time logs
tail -f logs/neural_net.log

# Search logs for errors
grep ERROR logs/neural_net.log
```

### Check System Health
```bash
# API health check
curl http://127.0.0.1:8000/health

# Database query
sqlite3 data/trading.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10"
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| GUI won't start | Ensure all dependencies installed: `pip install -r requirements.txt` |
| No pairs showing | Wait 5 minutes for initial market scan |
| API errors | Verify API keys in API CONFIG tab |
| Can't login | Reset PIN in `config/neural_config.json` |
| Database locked | Close any duplicate running instances |
| High CPU usage | Reduce scan frequency in settings |

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: Cryptocurrency trading carries significant risk. This software is provided "as-is" without any guarantees. Always:
- Start with small amounts
- Use testnet mode first
- Understand the risks involved
- Never invest more than you can afford to lose
- This is not financial advice

## 🙏 Acknowledgments

- Built with [CCXT](https://github.com/ccxt/ccxt) for exchange connectivity
- Uses [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for modern GUI
- Powered by neural networks for AI decision-making

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Nexlify/issues)
- **Documentation**: See docs folder for detailed guides
- **Community**: Join our discussions

---

**Built with intelligence and precision** 🤖💰

*Version: 2.0.7.7*
