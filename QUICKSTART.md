# 🚀 Nexlify Quick Start Guide

**Get trading in Night City in under 5 minutes**

## 🏃‍♂️ 60-Second Setup (Docker)

```bash
# 1. Clone and enter
git clone https://github.com/nexlify/nexlify.git && cd nexlify

# 2. Quick install
curl -sSL https://raw.githubusercontent.com/nexlify/nexlify/main/quickstart.sh | bash

# 3. Access the platform
# Web UI: http://localhost:3000
# API: http://localhost:8000
```

## 🎯 First Trade in 3 Steps

### Step 1: Create Account
```bash
nexlify auth register
# Enter username, email, password
# SAVE YOUR PIN - Won't be shown again!
```

### Step 2: Create Portfolio
```bash
nexlify portfolio create --name "Test Fund" --paper
# Note the portfolio ID returned
```

### Step 3: Get AI Signal & Trade
```bash
# Get AI recommendation
nexlify trade signal BTC/USDT

# Place order if signal looks good
nexlify trade place \
  --portfolio YOUR_PORTFOLIO_ID \
  --symbol BTC/USDT \
  --side buy \
  --quantity 0.1
```

## 🖥️ GUI Access

1. Open browser to `http://localhost:3000`
2. Login with your credentials
3. Navigate to Dashboard
4. Click "AI Signal" for any symbol
5. Click "Execute Trade" to place order

## 📊 Monitor Your Trades

### Via CLI:
```bash
# Watch positions
nexlify portfolio positions YOUR_PORTFOLIO_ID

# View live market
nexlify market watch BTC/USDT ETH/USDT

# Check system status
nexlify system status
```

### Via Web:
- **Dashboard**: Real-time P&L and positions
- **Trading**: Execute trades with one click
- **AI Lab**: Backtest and train models
- **Monitoring**: System health at `http://localhost:3001`

## 🔥 Pro Tips

### 1. Enable GPU for AI (10x faster)
```bash
# Edit docker-compose.yml
# Uncomment GPU section
docker-compose --profile ml up -d
```

### 2. Connect Real Exchange
```bash
# Add to .env file
BINANCE_API_KEY=your-key
BINANCE_API_SECRET=your-secret

# Switch portfolio to live mode
nexlify portfolio create --name "Real Money" --live
```

### 3. Set Up Alerts
```bash
# Configure webhook in .env
ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook

# Alerts auto-trigger on:
# - Stop loss hit
# - High confidence AI signals
# - System issues
```

### 4. Backtest Before Trading
```bash
nexlify backtest run \
  --symbol BTC/USDT \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --capital 10000
```

## 🛠️ Troubleshooting

### Issue: "Cannot connect to database"
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Restart if needed
docker-compose restart postgres
```

### Issue: "AI model not found"
```bash
# Download pre-trained model
nexlify ml download --model transformer-v1

# Or train your own
nexlify ml train --symbol BTC/USDT --gpu
```

### Issue: "Rate limit exceeded"
```bash
# Check your limits
nexlify system limits

# Upgrade rate limits in config
# Edit config/nexlify.yaml
# rate_limit: 1000  # requests per minute
```

## 📈 Example Trading Session

```bash
# Morning routine of a Nexlify trader

# 1. Check system health
nexlify system status

# 2. Review overnight positions
nexlify portfolio positions abc-123-def

# 3. Get market overview
nexlify market watch BTC/USDT ETH/USDT SOL/USDT

# 4. Check AI signals for opportunities
nexlify trade signal BTC/USDT
nexlify trade signal ETH/USDT

# 5. Execute high-confidence trades
nexlify trade place --portfolio abc-123-def \
  --symbol ETH/USDT --side buy --quantity 2.5

# 6. Set alerts and walk away
nexlify alerts set --price BTC/USDT --above 55000
nexlify alerts set --price ETH/USDT --below 2800
```

## 🎮 Keyboard Shortcuts (Web UI)

- `Space` - Toggle AI signals panel
- `T` - Open trade dialog
- `P` - Show positions
- `M` - Toggle market data
- `Esc` - Close all dialogs
- `Ctrl+K` - Command palette

## 🔗 Quick Links

- **API Docs**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3001 (admin/n3xl1fy_gr4f4n4)
- **Logs**: `docker logs nexlify-api -f`
- **Support**: https://discord.gg/nexlify

## 💡 Next Steps

1. **Join Discord** for trading signals and community
2. **Read the full docs** at `/docs` for advanced features
3. **Train custom models** with your own strategies
4. **Deploy to cloud** for 24/7 trading

---

**Welcome to the future of trading. May the profits be with you.**

*Remember: In Night City, only the smart survive. Trade wisely.*

🌃 🚀 💰
