# Nexlify Advanced Trading Features Guide

## 🚀 Overview

This guide covers three major advanced features added to Nexlify:

1. **Risk Management System** - Professional-grade position sizing and loss limits
2. **Circuit Breaker Pattern** - Intelligent failure handling for exchange APIs
3. **Performance Metrics Calculator** - Comprehensive trading analytics

---

## 📊 Feature 1: Risk Management System

### What It Does

The Risk Manager protects your capital with:
- **Position size limits** (default: max 5% per trade)
- **Daily loss limits** (default: stops trading at 5% daily loss)
- **Automatic stop-loss** (default: 2% per trade)
- **Automatic take-profit** (default: 5% per trade)
- **Kelly Criterion position sizing** (optional, for optimal bet sizing)

### Configuration

Edit `config/neural_config.json`:

```json
{
  "risk_management": {
    "enabled": true,
    "max_position_size": 0.05,
    "max_daily_loss": 0.05,
    "stop_loss_percent": 0.02,
    "take_profit_percent": 0.05,
    "use_kelly_criterion": true,
    "kelly_fraction": 0.5,
    "min_kelly_confidence": 0.6,
    "max_concurrent_trades": 3
  }
}
```

### Integration Example

```python
from nexlify_risk_manager import RiskManager

# Initialize
config = load_config()  # Your config loader
risk_manager = RiskManager(config)

# Validate a trade before execution
validation = await risk_manager.validate_trade(
    symbol="BTC/USDT",
    side="buy",
    quantity=0.1,
    price=50000,
    balance=10000,
    confidence=0.75  # Neural net confidence
)

if validation.approved:
    # Execute trade with recommended parameters
    execute_trade(
        quantity=validation.adjusted_size or quantity,
        stop_loss=validation.stop_loss,
        take_profit=validation.take_profit
    )
else:
    logger.warning(f"Trade rejected: {validation.reason}")

# After trade closes, record the result
risk_manager.record_trade_result(
    symbol="BTC/USDT",
    side="buy",
    entry_price=50000,
    exit_price=51000,
    quantity=0.1,
    balance=10000
)

# Check current risk status
status = risk_manager.get_risk_status()
print(f"Daily P&L: {status['net_pnl']}")
print(f"Trading halted: {status['trading_halted']}")
```

### Key Features

✅ **Position Size Validation**
- Automatically rejects trades exceeding max position size
- Suggests adjusted size within limits

✅ **Daily Loss Protection**
- Tracks daily losses automatically
- Halts trading when limit is reached
- Resets at midnight UTC

✅ **Kelly Criterion**
- Calculates optimal position size based on win probability
- Uses conservative 50% Kelly by default
- Only applies when confidence ≥ 60%

✅ **State Persistence**
- Saves risk metrics to disk
- Survives restarts
- Automatic daily reset

---

## 🔌 Feature 2: Circuit Breaker Pattern

### What It Does

Protects against cascading failures and API issues:
- **Automatic failure detection** (opens after 3 consecutive failures)
- **Blocks calls when OPEN** (prevents hammering failing APIs)
- **Automatic recovery testing** (after 5-minute timeout)
- **Smart state transitions** (CLOSED → OPEN → HALF_OPEN → CLOSED)

### Configuration

Edit `config/neural_config.json`:

```json
{
  "circuit_breaker": {
    "enabled": true,
    "failure_threshold": 3,
    "timeout_seconds": 300,
    "half_open_max_calls": 1
  }
}
```

### Integration Example

```python
from nexlify_circuit_breaker import CircuitBreakerManager

# Initialize manager
config = load_config()
circuit_manager = CircuitBreakerManager(config)

# Get circuit breaker for each exchange
binance_breaker = circuit_manager.get_or_create("binance")
kraken_breaker = circuit_manager.get_or_create("kraken")

# Wrap all exchange API calls
async def fetch_ticker_safe(exchange_id: str, symbol: str):
    breaker = circuit_manager.get_or_create(exchange_id)

    try:
        # Call is protected by circuit breaker
        result = await breaker.call(
            exchanges[exchange_id].fetch_ticker,
            symbol
        )
        return result
    except Exception as e:
        logger.error(f"Failed to fetch {symbol} from {exchange_id}: {e}")
        return None

# Check circuit breaker status
status = binance_breaker.get_status()
print(f"State: {status['state']}")
print(f"Success rate: {status['success_rate']}")
print(f"Consecutive failures: {status['consecutive_failures']}")

# Get overall health
health = circuit_manager.get_health_summary()
print(f"Healthy exchanges: {health['healthy']}/{health['total_breakers']}")
```

### Circuit States

1. **CLOSED** (🟢 Normal)
   - All calls go through
   - Failures are tracked
   - Opens after threshold failures

2. **OPEN** (🔴 Blocking)
   - All calls blocked immediately
   - No actual API calls made
   - Waits for timeout period
   - Transitions to HALF_OPEN after timeout

3. **HALF_OPEN** (🟡 Testing)
   - Allows limited test calls
   - Success → CLOSED
   - Failure → OPEN

### Key Features

✅ **Intelligent Failure Detection**
- Tracks consecutive failures per exchange
- Distinguishes between transient and persistent issues

✅ **Prevents API Hammering**
- Stops calling failing APIs immediately
- Reduces rate limit exhaustion

✅ **Automatic Recovery**
- Tests recovery after timeout
- Gradually reopens circuit on success

✅ **Detailed Statistics**
- Success/failure rates
- State change history
- Last failure/success timestamps

---

## 📊 Feature 3: Performance Metrics Calculator

### What It Does

Comprehensive trading analytics:
- **Basic metrics** (win rate, total trades, P&L)
- **Profit metrics** (average win/loss, profit factor)
- **Advanced metrics** (Sharpe ratio, max drawdown)
- **Trade recording** (SQLite database storage)
- **Export capabilities** (JSON/CSV)

### Configuration

Edit `config/neural_config.json`:

```json
{
  "performance_tracking": {
    "enabled": true,
    "database_path": "data/trading.db",
    "calculate_sharpe_ratio": true,
    "risk_free_rate": 0.02,
    "track_drawdown": true,
    "export_formats": ["json", "csv"]
  }
}
```

### Integration Example

```python
from nexlify_performance_tracker import PerformanceTracker

# Initialize
config = load_config()
tracker = PerformanceTracker(config)

# Record a trade
trade_id = tracker.record_trade(
    exchange="binance",
    symbol="BTC/USDT",
    side="buy",
    quantity=0.1,
    entry_price=50000,
    exit_price=51000,  # None if still open
    fee=10.0,
    strategy="arbitrage",
    notes="Cross-exchange opportunity"
)

# Update an open trade when it closes
tracker.update_trade(
    trade_id=trade_id,
    exit_price=52000,
    status="closed"
)

# Get performance metrics
metrics = tracker.get_performance_metrics()

print(f"Total Trades: {metrics.total_trades}")
print(f"Win Rate: {metrics.win_rate:.1f}%")
print(f"Total P&L: ${metrics.total_pnl:.2f}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {metrics.max_drawdown_percent:.1f}%")

# Filter by date/exchange/symbol
metrics = tracker.get_performance_metrics(
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 12, 31),
    exchange="binance",
    symbol="BTC/USDT"
)

# Export trades
tracker.export_trades(
    filepath="trades_2025.json",
    format="json",
    start_date=datetime(2025, 1, 1)
)

# Get recent trades summary
recent = tracker.get_trades_summary(limit=10)
for trade in recent:
    print(f"{trade['timestamp']}: {trade['symbol']} P&L: ${trade['pnl']:.2f}")
```

### Metrics Explained

**Basic Metrics:**
- **Total Trades**: Number of closed trades
- **Winning Trades**: Trades with positive P&L
- **Losing Trades**: Trades with negative P&L
- **Win Rate**: (Winning Trades / Total Trades) × 100%

**Profit Metrics:**
- **Total P&L**: Sum of all trade profits/losses
- **Average Win**: Average profit of winning trades
- **Average Loss**: Average loss of losing trades
- **Profit Factor**: Total Wins / Total Losses (>1 is profitable)
- **Best/Worst Trade**: Largest win/loss

**Advanced Metrics:**
- **Sharpe Ratio**: Risk-adjusted returns (higher is better)
  - Formula: `(Mean Return - Risk Free Rate) / Std Dev × √252`
  - Good: > 1.0, Excellent: > 2.0
- **Max Drawdown**: Largest peak-to-trough decline
  - Shows worst-case risk exposure
  - Important for position sizing

### Key Features

✅ **Comprehensive Recording**
- Every trade stored in SQLite database
- Timestamp, prices, fees, P&L tracked
- Optional strategy and notes fields

✅ **Professional Analytics**
- Industry-standard metrics
- Sharpe ratio for risk-adjusted performance
- Max drawdown for risk assessment

✅ **Flexible Filtering**
- Filter by date range
- Filter by exchange
- Filter by symbol

✅ **Export Capabilities**
- JSON format for programmatic access
- CSV format for spreadsheet analysis
- Filtered exports supported

---

## 🔧 Complete Integration Example

Here's how to integrate all three features into `arasaka_neural_net.py`:

```python
# In arasaka_neural_net.py

from nexlify_risk_manager import RiskManager
from nexlify_circuit_breaker import CircuitBreakerManager
from nexlify_performance_tracker import PerformanceTracker

class ArasakaNeuralNet:
    def __init__(self, config: Dict):
        self.config = config

        # Initialize new components
        self.risk_manager = RiskManager(config)
        self.circuit_manager = CircuitBreakerManager(config)
        self.performance_tracker = PerformanceTracker(config)

        # ... existing initialization ...

    async def initialize(self):
        """Initialize the Neural-Net and connect to exchanges"""
        # ... existing code ...

        # Create circuit breakers for each exchange
        for exchange_id in self.exchanges.keys():
            self.circuit_manager.get_or_create(exchange_id)
            logger.info(f"🔌 Circuit breaker ready for {exchange_id}")

    async def fetch_ticker(self, exchange_id: str, symbol: str):
        """Fetch ticker with circuit breaker protection"""
        breaker = self.circuit_manager.get_or_create(exchange_id)

        return await breaker.call(
            self.exchanges[exchange_id].fetch_ticker,
            symbol
        )

    async def execute_trade(
        self,
        exchange_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """Execute trade with risk management and tracking"""

        # 1. Validate with risk manager
        balance = await self.get_balance(exchange_id)
        confidence = self.get_neural_confidence(symbol)

        validation = await self.risk_manager.validate_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=confidence
        )

        if not validation.approved:
            logger.warning(f"❌ Trade rejected: {validation.reason}")
            return None

        # Use adjusted size if recommended
        final_quantity = validation.adjusted_size or quantity

        # 2. Execute trade with circuit breaker protection
        breaker = self.circuit_manager.get_or_create(exchange_id)

        try:
            order = await breaker.call(
                self.exchanges[exchange_id].create_order,
                symbol=symbol,
                type='market',
                side=side,
                amount=final_quantity
            )

            # 3. Set stop-loss and take-profit orders
            if validation.stop_loss:
                await self.set_stop_loss(
                    exchange_id, symbol, validation.stop_loss
                )

            if validation.take_profit:
                await self.set_take_profit(
                    exchange_id, symbol, validation.take_profit
                )

            # 4. Record trade in performance tracker
            trade_id = self.performance_tracker.record_trade(
                exchange=exchange_id,
                symbol=symbol,
                side=side,
                quantity=final_quantity,
                entry_price=order['price'],
                exit_price=None,  # Still open
                fee=order.get('fee', {}).get('cost', 0)
            )

            logger.info(f"✅ Trade executed: {symbol} {side} {final_quantity}")
            return trade_id

        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return None

    async def close_trade(self, trade_id: int, exit_price: float):
        """Close trade and update metrics"""

        # Update performance tracker
        self.performance_tracker.update_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            status="closed"
        )

        # Update risk manager with result
        # (You'll need to fetch trade details from tracker)
        # risk_manager.record_trade_result(...)

    def get_trading_status(self) -> Dict:
        """Get comprehensive trading status"""
        return {
            'risk': self.risk_manager.get_risk_status(),
            'circuit_breakers': self.circuit_manager.get_all_status(),
            'health': self.circuit_manager.get_health_summary(),
            'performance': self.performance_tracker.get_performance_metrics().to_dict()
        }
```

---

## 🧪 Testing

All features include comprehensive unit tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific feature tests
pytest tests/test_risk_manager.py -v
pytest tests/test_circuit_breaker.py -v
pytest tests/test_performance_tracker.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

---

## 📈 Success Criteria Checklist

✅ **Risk Manager**
- [x] Prevents trades exceeding position size limits
- [x] Stops trading after daily loss limit
- [x] Calculates stop-loss and take-profit
- [x] Kelly Criterion position sizing
- [x] State persistence across restarts

✅ **Circuit Breaker**
- [x] Opens after repeated exchange failures
- [x] Recovers automatically after timeout
- [x] Blocks calls when OPEN
- [x] Smart state transitions

✅ **Performance Tracker**
- [x] Records all trades to database
- [x] Calculates win rate and profit metrics
- [x] Computes Sharpe ratio
- [x] Tracks max drawdown
- [x] Exports to JSON/CSV

✅ **Integration**
- [x] All features have passing unit tests
- [x] No breaking changes to existing code
- [x] Configuration is flexible
- [x] Comprehensive documentation

---

## 🎯 Usage Tips

### Risk Management
- Start conservative: 2% position size, 3% daily loss limit
- Monitor `get_risk_status()` regularly
- Review and adjust limits based on strategy performance

### Circuit Breakers
- Check `get_health_summary()` in your dashboard
- Manually force-close circuits if needed: `breaker.force_close()`
- Monitor `success_rate` - below 80% indicates issues

### Performance Tracking
- Export trades monthly for tax records
- Track Sharpe ratio - aim for > 1.0
- Watch max drawdown - should be < 20% for most strategies
- Use filters to analyze specific pairs or exchanges

---

## 🆘 Troubleshooting

**Risk Manager stops all trading:**
```python
# Check status
status = risk_manager.get_risk_status()
print(status)

# Force reset if needed (use with caution!)
risk_manager.force_reset()

# Or resume trading
risk_manager.resume_trading("Manual override")
```

**Circuit Breaker stuck OPEN:**
```python
# Check status
status = breaker.get_status()
print(status)

# Force close if needed
breaker.force_close("Manual recovery")

# Or reset completely
breaker.reset()
```

**Performance Tracker database issues:**
```python
# Check database path
print(tracker.db_path)

# Manually backup database
import shutil
shutil.copy(tracker.db_path, "backup_trading.db")
```

---

## 📚 Additional Resources

- **Risk Management Theory**: Kelly Criterion, Position Sizing
- **Circuit Breaker Pattern**: Martin Fowler's blog
- **Performance Metrics**: Sharpe Ratio, Sortino Ratio, Max Drawdown
- **Nexlify Documentation**: See `README.md` and `IMPLEMENTATION_GUIDE.md`

---

## 🚀 Next Steps

1. Review configuration in `config/neural_config.json`
2. Test each feature individually with small amounts
3. Integrate into your trading strategy
4. Monitor performance metrics daily
5. Adjust parameters based on results

**Stay safe in the Matrix! 🌆**
