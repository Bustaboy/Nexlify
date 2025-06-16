# 🌃 Nexlify - Environment Settings Guide

## 📋 Overview

The Environment tab in Nexlify provides advanced configuration options for debugging, monitoring, and notifications. All settings are managed through the GUI - no manual file editing required!

## 🔧 Available Settings

### 1. Debug Mode
**Location**: Environment Tab → Debug Mode checkbox

**Purpose**: Enables verbose logging for troubleshooting

**When Enabled**:
- Detailed API request/response logs
- Order execution step-by-step traces
- Neural network decision explanations
- Performance timing information
- Memory usage statistics

**When to Use**:
- Troubleshooting connection issues
- Understanding why trades aren't executing
- Monitoring system performance
- Development and testing

**Impact**: 
- Log files grow faster (10-50MB/day)
- Slightly higher CPU usage
- More detailed error messages

### 2. Log Level
**Location**: Environment Tab → Log Level dropdown

**Options**:
- **ERROR**: Only critical errors and failures
- **WARNING**: Errors + important warnings
- **INFO**: Standard operation logs (default)
- **DEBUG**: Everything including internal details

**Recommendations**:
- **Production**: Use INFO
- **Testing**: Use DEBUG
- **Minimal Logging**: Use WARNING
- **Troubleshooting**: Use DEBUG

**Log File Locations**:
```
logs/
├── neural_net.log      # Main application log
├── error.log          # Error-specific log
├── trades.log         # Trade execution log
└── crash_reports/     # Fatal error dumps
```

### 3. API Port
**Location**: Environment Tab → API Port field

**Default**: 8000

**Valid Range**: 1024-65535

**When to Change**:
- Port 8000 is already in use
- Running multiple instances
- Firewall restrictions
- Corporate network requirements

**Common Alternatives**:
- 8080 (alternative HTTP)
- 8888 (Jupyter default)
- 3000 (Node.js default)
- 5000 (Flask default)

**How to Check Port**:
```bash
# Windows
netstat -an | findstr 8000

# Linux/Mac
lsof -i :8000
```

### 4. Database URL
**Location**: Environment Tab → Database URL field

**Default**: `sqlite:///data/trading.db`

**Format Options**:
```
# SQLite (default - single file)
sqlite:///data/trading.db
sqlite:///C:/Nexlify/data/trading.db  # Absolute path

# PostgreSQL (advanced users)
postgresql://user:password@localhost/nexlify

# MySQL (advanced users)
mysql://user:password@localhost/nexlify
```

**SQLite Benefits**:
- No server required
- Zero configuration
- Portable database file
- Perfect for single-user

**When to Use PostgreSQL/MySQL**:
- Multiple simultaneous users
- Remote database access
- Better concurrent performance
- Professional deployment

### 5. Emergency Contact Email
**Location**: Environment Tab → Emergency Contact field

**Purpose**: Receive alerts for critical events

**Triggers**:
- System crashes
- Failed emergency stops
- Critical trading errors
- Unusual market conditions
- Security alerts

**Format**: `your.email@example.com`

**Email Notifications Include**:
- Error description
- Timestamp
- System state
- Recommended actions
- Log excerpts

**Note**: Requires SMTP configuration in advanced settings

### 6. Telegram Notifications
**Location**: Environment Tab → Telegram Bot Token & Chat ID

**Setup Process**:

#### Step 1: Create Telegram Bot
1. Open Telegram
2. Search for `@BotFather`
3. Send `/newbot`
4. Choose name: "Nexlify Trader"
5. Choose username: `nexlify_bot`
6. Copy the token

#### Step 2: Get Chat ID
1. Start chat with your bot
2. Send any message
3. Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Find `"chat":{"id":YOUR_CHAT_ID}`
5. Copy the chat ID

#### Step 3: Configure in GUI
```
Telegram Bot Token: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
Telegram Chat ID: 987654321
```

**Notifications Sent**:
- Trade executions
- Profit milestones
- Error alerts
- Daily summaries
- System status changes

**Notification Format**:
```
🤖 Nexlify Alert

✅ Trade Executed
Pair: BTC/USDT
Type: BUY
Price: $45,231.50
Profit: +0.75%
```

## 🔐 Advanced Environment Configuration

### Custom Environment Variables
For advanced users, additional settings can be configured:

**Create `.env.local` file**:
```bash
# Advanced Settings (optional)
NEXLIFY_MAX_MEMORY=4096
NEXLIFY_THREAD_POOL=8
NEXLIFY_CACHE_SIZE=1000
NEXLIFY_WEBHOOK_URL=https://your-webhook.com
NEXLIFY_SENTRY_DSN=your-sentry-dsn
```

### Performance Tuning
```bash
# Increase memory limit (Windows)
set NEXLIFY_MAX_MEMORY=8192

# Increase memory limit (Linux/Mac)
export NEXLIFY_MAX_MEMORY=8192
```

### Network Configuration
```bash
# Use proxy
HTTPS_PROXY=http://proxy:8080
HTTP_PROXY=http://proxy:8080

# Custom DNS
NEXLIFY_DNS_SERVERS=8.8.8.8,1.1.1.1
```

## 📊 Monitoring & Metrics

### Enable Metrics Collection
In Environment tab, you can enable:
- Performance metrics
- Trade analytics
- System health monitoring

### Metrics Dashboard
Access via: `http://localhost:8000/metrics`

Shows:
- CPU/Memory usage
- API response times
- Trade success rates
- Profit trends
- Error rates

## 🐛 Troubleshooting Environment Issues

### Debug Mode Not Working
```python
# Verify in logs/neural_net.log
2024-01-15 10:30:45 - INFO - Debug mode: True
```

### Telegram Not Sending
1. Verify bot token is correct
2. Check chat ID (should be number)
3. Test manually:
```bash
curl https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<CHAT_ID>&text=Test
```

### Email Alerts Not Working
1. Check spam folder
2. Verify email format
3. May need SMTP config:
```python
# In advanced settings
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your.email@gmail.com
SMTP_PASS=app-specific-password
```

### Database Connection Issues
```bash
# Test connection
sqlite3 data/trading.db ".tables"

# Check permissions
ls -la data/trading.db  # Linux/Mac
dir data\trading.db     # Windows
```

## 🔒 Security Best Practices

### Environment Security
1. **Never commit** `.env` files
2. **Use** `.env.local` for overrides
3. **Encrypt** sensitive values
4. **Rotate** tokens periodically
5. **Limit** API permissions

### Secure Configuration
```python
# Good practices
telegram_token = os.getenv('TELEGRAM_TOKEN')  # From environment
database_url = config.get('database_url')      # From encrypted config

# Bad practices
telegram_token = "123456789:ABC..."  # Hardcoded
database_url = "postgresql://admin:password123@localhost"  # Plain text
```

## 📝 Configuration Templates

### Development Environment
```json
{
  "debug": true,
  "log_level": "DEBUG",
  "api_port": 8000,
  "database_url": "sqlite:///data/dev.db",
  "emergency_contact": "dev@example.com",
  "telegram_bot_token": "",
  "telegram_chat_id": ""
}
```

### Production Environment
```json
{
  "debug": false,
  "log_level": "INFO",
  "api_port": 8000,
  "database_url": "postgresql://user:pass@localhost/nexlify",
  "emergency_contact": "alerts@company.com",
  "telegram_bot_token": "ENCRYPTED_TOKEN",
  "telegram_chat_id": "ENCRYPTED_ID"
}
```

### Testing Environment
```json
{
  "debug": true,
  "log_level": "DEBUG",
  "api_port": 8001,
  "database_url": "sqlite:///data/test.db",
  "emergency_contact": "",
  "telegram_bot_token": "",
  "telegram_chat_id": ""
}
```

## 🚀 Quick Setup Checklist

- [ ] Set appropriate log level
- [ ] Configure emergency email
- [ ] Setup Telegram notifications
- [ ] Verify database connection
- [ ] Test port availability
- [ ] Enable debug mode if needed
- [ ] Configure any proxies
- [ ] Set up monitoring

---

*In Night City, your environment shapes your destiny. Configure wisely, trade profitably.* 🌃🤖💰
