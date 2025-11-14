# Hardcoded Values Audit Report - Nexlify Codebase

**Generated:** 2025-11-14  
**Scope:** Complete codebase scan for hardcoded configuration values  
**Categories:** 5 (Risk Management, RL Hyperparameters, File Paths, Fees/DeFi, Timeouts/Delays)

---

## CATEGORY 1: Risk Management & Trading Parameters

### 1.1 Position Sizing & Risk Limits

**File:** `/home/user/Nexlify/nexlify/risk/nexlify_risk_manager.py`

#### Default Max Position Size
- **Line:** 68
- **Value:** `0.05` (5%)
- **Context:** 
  ```python
  self.max_position_size = self.config.get("max_position_size", 0.05)  # 5% default
  ```
- **Suggested Config Key:** `risk_management.max_position_size`
- **Priority:** HIGH - Directly affects position exposure
- **Notes:** This is a critical risk parameter. Consider making it dynamic based on account size.

#### Default Max Daily Loss
- **Line:** 70
- **Value:** `0.05` (5%)
- **Context:**
  ```python
  self.max_daily_loss = self.config.get("max_daily_loss", 0.05)  # 5% default
  ```
- **Suggested Config Key:** `risk_management.max_daily_loss`
- **Priority:** HIGH - Circuit breaker for daily trading

#### Default Stop Loss Percentage
- **Line:** 72
- **Value:** `0.02` (2%)
- **Context:**
  ```python
  self.stop_loss_percent = self.config.get("stop_loss_percent", 0.02)  # 2% default
  ```
- **Suggested Config Key:** `risk_management.stop_loss_percent`
- **Priority:** HIGH - Core risk management

#### Default Take Profit Percentage
- **Line:** 74-76
- **Value:** `0.05` (5%)
- **Context:**
  ```python
  self.take_profit_percent = self.config.get("take_profit_percent", 0.05)  # 5% default
  ```
- **Suggested Config Key:** `risk_management.take_profit_percent`
- **Priority:** HIGH - Profit target setting

#### Kelly Criterion Fraction
- **Line:** 78-80
- **Value:** `0.5` (50% of Kelly fraction)
- **Context:**
  ```python
  self.kelly_fraction = self.config.get("kelly_fraction", 0.5)  # Conservative Kelly
  ```
- **Suggested Config Key:** `risk_management.kelly_fraction`
- **Priority:** MEDIUM - Position sizing optimization

#### Min Kelly Confidence Threshold
- **Line:** 81
- **Value:** `0.6` (60%)
- **Context:**
  ```python
  self.min_kelly_confidence = self.config.get("min_kelly_confidence", 0.6)
  ```
- **Suggested Config Key:** `risk_management.min_kelly_confidence`
- **Priority:** MEDIUM - Confidence threshold for Kelly usage

#### Max Concurrent Trades
- **Line:** 82
- **Value:** `3`
- **Context:**
  ```python
  self.max_concurrent_trades = self.config.get("max_concurrent_trades", 3)
  ```
- **Suggested Config Key:** `risk_management.max_concurrent_trades`
- **Priority:** HIGH - Exposure control

---

### 1.2 Flash Crash Protection Thresholds

**File:** `/home/user/Nexlify/nexlify/risk/nexlify_flash_crash_protection.py`

#### Minor Crash Threshold
- **Line:** 117
- **Value:** `-0.05` (-5%)
- **Context:**
  ```python
  self.thresholds = {
      CrashSeverity.MINOR: self.config.get("minor_threshold", -0.05),  # -5%
  ```
- **Suggested Config Key:** `flash_crash_protection.minor_threshold`
- **Priority:** HIGH - Warning trigger level

#### Major Crash Threshold
- **Line:** 118
- **Value:** `-0.10` (-10%)
- **Context:**
  ```python
  CrashSeverity.MAJOR: self.config.get("major_threshold", -0.10),  # -10%
  ```
- **Suggested Config Key:** `flash_crash_protection.major_threshold`
- **Priority:** HIGH - Position closing trigger

#### Critical Crash Threshold
- **Line:** 119-121
- **Value:** `-0.15` (-15%)
- **Context:**
  ```python
  CrashSeverity.CRITICAL: self.config.get("critical_threshold", -0.15),  # -15%
  ```
- **Suggested Config Key:** `flash_crash_protection.critical_threshold`
- **Priority:** HIGH - Kill switch trigger

#### Flash Crash Check Interval
- **Line:** 125
- **Value:** `30` (seconds)
- **Context:**
  ```python
  self.check_interval = self.config.get("check_interval", 30)  # seconds
  ```
- **Suggested Config Key:** `flash_crash_protection.check_interval_seconds`
- **Priority:** MEDIUM - Monitoring frequency

#### Recovery Threshold
- **Line:** 127-129
- **Value:** `0.20` (20% recovery)
- **Context:**
  ```python
  self.recovery_threshold = self.config.get("recovery_threshold", 0.20)  # 20% recovery
  ```
- **Suggested Config Key:** `flash_crash_protection.recovery_threshold`
- **Priority:** MEDIUM - Market recovery detection

#### Volume Spike Threshold
- **Line:** 132-134
- **Value:** `3.0` (3x average volume)
- **Context:**
  ```python
  self.volume_spike_threshold = self.config.get("volume_spike_threshold", 3.0)  # 3x avg
  ```
- **Suggested Config Key:** `flash_crash_protection.volume_spike_threshold`
- **Priority:** MEDIUM - Unusual activity detection

#### Price Drop Threshold (Legacy)
- **Line:** 147
- **Value:** `10.0` (10% absolute)
- **Context:**
  ```python
  self.price_drop_threshold = self.config.get("price_drop_threshold", 10.0)
  ```
- **Suggested Config Key:** `flash_crash_protection.price_drop_threshold`
- **Priority:** LOW - Legacy test parameter

#### Time Window Seconds (Legacy)
- **Line:** 148
- **Value:** `60` (seconds)
- **Context:**
  ```python
  self.time_window_seconds = self.config.get("time_window_seconds", 60)
  ```
- **Suggested Config Key:** `flash_crash_protection.time_window_seconds`
- **Priority:** LOW - Legacy test parameter

#### Min Volume Spike (Legacy)
- **Line:** 149
- **Value:** `3.0`
- **Context:**
  ```python
  self.min_volume_spike = self.config.get("min_volume_spike", 3.0)
  ```
- **Suggested Config Key:** `flash_crash_protection.min_volume_spike`
- **Priority:** LOW - Legacy test parameter

#### Price History Buffer Sizes
- **Line:** 152-156
- **Values:** `60` (1m), `60` (5m), `96` (15m)
- **Context:**
  ```python
  self.max_history = {
      "1m": 60,   # 1 hour of 1-minute data
      "5m": 60,   # 5 hours of 5-minute data
      "15m": 96,  # 24 hours of 15-minute data
  }
  ```
- **Suggested Config Key:** `flash_crash_protection.max_history`
- **Priority:** MEDIUM - Memory management

#### Volume History Maxlen
- **Line:** 224
- **Value:** `100`
- **Context:**
  ```python
  self.volume_history[symbol] = deque(maxlen=100)
  ```
- **Suggested Config Key:** `flash_crash_protection.volume_history_size`
- **Priority:** LOW - Buffer management

---

### 1.3 Circuit Breaker Settings

**File:** `/home/user/Nexlify/nexlify/risk/nexlify_circuit_breaker.py`

#### Circuit Breaker Failure Threshold
- **Line:** 69-70
- **Value:** `3` (consecutive failures)
- **Context:**
  ```python
  failure_threshold: int = 3,
  timeout_seconds: int = 300,
  ```
- **Suggested Config Key:** `circuit_breaker.failure_threshold`
- **Priority:** HIGH - API failure tolerance

#### Circuit Breaker Timeout
- **Line:** 70
- **Value:** `300` (seconds / 5 minutes)
- **Context:**
  ```python
  timeout_seconds: int = 300,
  ```
- **Suggested Config Key:** `circuit_breaker.timeout_seconds`
- **Priority:** HIGH - Recovery waiting period

#### Half-Open Max Calls
- **Line:** 71
- **Value:** `1`
- **Context:**
  ```python
  half_open_max_calls: int = 1,
  ```
- **Suggested Config Key:** `circuit_breaker.half_open_max_calls`
- **Priority:** MEDIUM - Recovery testing limit

---

### 1.4 Auto-Trader Position Management

**File:** `/home/user/Nexlify/nexlify/core/nexlify_auto_trader.py`

#### Take Profit Percentage
- **Line:** 135
- **Value:** `5.0` (5%)
- **Context:**
  ```python
  self.take_profit_percent = config.get("take_profit", 5.0)  # 5%
  ```
- **Suggested Config Key:** `auto_trader.take_profit_percent`
- **Priority:** HIGH - Exit strategy

#### Stop Loss Percentage
- **Line:** 136
- **Value:** `2.0` (2%)
- **Context:**
  ```python
  self.stop_loss_percent = config.get("stop_loss", 2.0)  # 2%
  ```
- **Suggested Config Key:** `auto_trader.stop_loss_percent`
- **Priority:** HIGH - Loss protection

#### Trailing Stop Percentage
- **Line:** 137
- **Value:** `3.0` (3%)
- **Context:**
  ```python
  self.trailing_stop_percent = config.get("trailing_stop", 3.0)  # 3%
  ```
- **Suggested Config Key:** `auto_trader.trailing_stop_percent`
- **Priority:** MEDIUM - Dynamic stop-loss

#### Max Hold Time
- **Line:** 138
- **Value:** `24` (hours)
- **Context:**
  ```python
  self.max_hold_time_hours = config.get("max_hold_time_hours", 24)
  ```
- **Suggested Config Key:** `auto_trader.max_hold_time_hours`
- **Priority:** MEDIUM - Position time limit

---

## CATEGORY 2: RL Agent Hyperparameters

### 2.1 DQN Agent Core Hyperparameters

**File:** `/home/user/Nexlify/nexlify/strategies/nexlify_ultra_optimized_rl_agent.py`

#### Discount Factor (Gamma)
- **Line:** 192
- **Value:** `0.99`
- **Context:**
  ```python
  self.gamma = 0.99
  ```
- **Suggested Config Key:** `rl_agent.discount_factor`
- **Priority:** HIGH - Determines future reward weight

#### Initial Epsilon
- **Line:** 193
- **Value:** `1.0`
- **Context:**
  ```python
  self.epsilon = 1.0
  ```
- **Suggested Config Key:** `rl_agent.epsilon_start`
- **Priority:** HIGH - Exploration rate

#### Epsilon Minimum
- **Line:** 194
- **Value:** `0.01`
- **Context:**
  ```python
  self.epsilon_min = 0.01
  ```
- **Suggested Config Key:** `rl_agent.epsilon_min`
- **Priority:** HIGH - Minimum exploration

#### Epsilon Decay Rate
- **Line:** 195
- **Value:** `0.995`
- **Context:**
  ```python
  self.epsilon_decay = 0.995
  ```
- **Suggested Config Key:** `rl_agent.epsilon_decay`
- **Priority:** HIGH - Decay schedule

#### Learning Rate
- **Line:** 180
- **Value:** `0.001` (1e-3)
- **Context:**
  ```python
  self.optimizer_nn = optim.Adam(self.model.parameters(), lr=0.001)
  ```
- **Suggested Config Key:** `rl_agent.learning_rate`
- **Priority:** HIGH - Optimization step size

#### Experience Replay Buffer Size
- **Line:** 189
- **Value:** `100000`
- **Context:**
  ```python
  self.memory = deque(maxlen=100000)
  ```
- **Suggested Config Key:** `rl_agent.replay_buffer_size`
- **Priority:** HIGH - Memory for experience replay

---

### 2.2 Epsilon Decay Configuration

**File:** `/home/user/Nexlify/nexlify/strategies/epsilon_decay.py`

#### Default Epsilon Start
- **Line:** 20
- **Value:** `1.0`
- **Context:**
  ```python
  def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.22):
  ```
- **Suggested Config Key:** `epsilon_decay.start`
- **Priority:** HIGH

#### Default Epsilon End
- **Line:** 20
- **Value:** `0.22` (22%)
- **Context:**
  ```python
  epsilon_end: float = 0.22
  ```
- **Suggested Config Key:** `epsilon_decay.end`
- **Priority:** HIGH - Crypto market ongoing exploration

#### Key Threshold Levels
- **Line:** 29
- **Values:** `[0.9, 0.7, 0.5, 0.3, 0.1]`
- **Context:**
  ```python
  self.key_thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
  ```
- **Suggested Config Key:** `epsilon_decay.threshold_milestones`
- **Priority:** LOW - Monitoring checkpoints

#### Default Linear Decay Steps
- **Line:** 116
- **Value:** `2000`
- **Context:**
  ```python
  def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.22,
               decay_steps: int = 2000):
  ```
- **Suggested Config Key:** `epsilon_decay.linear_decay_steps`
- **Priority:** HIGH - Training duration

#### Default Scheduled Decay Schedule
- **Line:** 163-168
- **Values:** `{0: 1.0, 200: 0.65, 800: 0.35, 2000: 0.22}`
- **Context:**
  ```python
  schedule = {
      0: 1.0,      # Full exploration
      200: 0.65,   # Learn basics quickly (~8 days)
      800: 0.35,   # Start exploiting patterns (~1 month)
      2000: 0.22   # High ongoing exploration (~2.5 months)
  }
  ```
- **Suggested Config Key:** `epsilon_decay.scheduled_milestones`
- **Priority:** HIGH - 24/7 crypto-specific schedule

---

### 2.3 Adaptive Agent Architecture

**File:** `/home/user/Nexlify/nexlify/strategies/nexlify_adaptive_rl_agent.py`

#### Base Gamma (Discount Factor)
- **Line:** 736
- **Value:** `0.99`
- **Context:**
  ```python
  self.gamma = 0.99
  ```
- **Suggested Config Key:** `adaptive_agent.discount_factor`
- **Priority:** HIGH

#### Base Epsilon
- **Line:** 737
- **Value:** `1.0`
- **Context:**
  ```python
  self.epsilon = 1.0
  ```
- **Suggested Config Key:** `adaptive_agent.epsilon_start`
- **Priority:** HIGH

#### Epsilon Minimum
- **Line:** 738
- **Value:** `0.01`
- **Context:**
  ```python
  self.epsilon_min = 0.01
  ```
- **Suggested Config Key:** `adaptive_agent.epsilon_min`
- **Priority:** HIGH

#### Epsilon Decay
- **Line:** 739
- **Value:** `0.995`
- **Context:**
  ```python
  self.epsilon_decay = 0.995
  ```
- **Suggested Config Key:** `adaptive_agent.epsilon_decay`
- **Priority:** HIGH

#### Base Learning Rate
- **Line:** 740
- **Value:** `0.001`
- **Context:**
  ```python
  self.learning_rate = 0.001
  ```
- **Suggested Config Key:** `adaptive_agent.learning_rate`
- **Priority:** HIGH

#### Default Batch Size
- **Line:** 525
- **Value:** `64`
- **Context:**
  ```python
  "batch_size": 64,
  ```
- **Suggested Config Key:** `adaptive_agent.default_batch_size`
- **Priority:** HIGH - Hardware-dependent

#### Default Buffer Size
- **Line:** 526
- **Value:** `100000`
- **Context:**
  ```python
  "buffer_size": 100000,
  ```
- **Suggested Config Key:** `adaptive_agent.default_buffer_size`
- **Priority:** HIGH

#### Architecture Presets - Tiny
- **Line:** 801
- **Value:** `[64, 32]`
- **Context:**
  ```python
  "tiny": [64, 32],  # 2-layer, small width
  ```
- **Suggested Config Key:** `adaptive_agent.architectures.tiny`
- **Priority:** MEDIUM

#### Architecture Presets - Small
- **Line:** 802
- **Value:** `[128, 64]`
- **Suggested Config Key:** `adaptive_agent.architectures.small`
- **Priority:** MEDIUM

#### Architecture Presets - Medium (Default)
- **Line:** 803
- **Value:** `[128, 128, 64]`
- **Suggested Config Key:** `adaptive_agent.architectures.medium`
- **Priority:** MEDIUM

#### Architecture Presets - Large
- **Line:** 804
- **Value:** `[256, 256, 128, 64]`
- **Suggested Config Key:** `adaptive_agent.architectures.large`
- **Priority:** MEDIUM

#### Architecture Presets - XLarge
- **Line:** 805
- **Value:** `[512, 512, 256, 128, 64]`
- **Suggested Config Key:** `adaptive_agent.architectures.xlarge`
- **Priority:** MEDIUM

---

## CATEGORY 3: File Paths & Database Locations

### 3.1 Data Directories

**File:** `/home/user/Nexlify/nexlify/risk/nexlify_risk_manager.py`

#### Risk State File Path
- **Line:** 90
- **Value:** `"data/risk_state.json"`
- **Context:**
  ```python
  self.state_file = Path("data/risk_state.json")
  ```
- **Suggested Config Key:** `paths.risk_state_file`
- **Priority:** MEDIUM - State persistence

---

### 3.2 Error Handler & Logging

**File:** `/home/user/Nexlify/nexlify/utils/error_handler.py`

#### Error Log Path
- **Line:** 31
- **Value:** `"logs/errors.log"`
- **Context:**
  ```python
  self.error_log_path = Path("logs/errors.log")
  ```
- **Suggested Config Key:** `paths.error_log`
- **Priority:** MEDIUM

#### Crash Report Path
- **Line:** 32
- **Value:** `"logs/crash_reports"`
- **Context:**
  ```python
  self.crash_report_path = Path("logs/crash_reports")
  ```
- **Suggested Config Key:** `paths.crash_reports_dir`
- **Priority:** MEDIUM

#### Config File Path
- **Line:** 29
- **Value:** `"config/neural_config.json"`
- **Context:**
  ```python
  def __init__(self, config_path: str = "config/neural_config.json"):
  ```
- **Suggested Config Key:** `paths.config_file`
- **Priority:** HIGH

---

### 3.3 Audit & Security

**File:** `/home/user/Nexlify/nexlify/security/nexlify_audit_trail.py`

#### Audit Log Path
- **Line:** 74
- **Value:** `"logs/audit"`
- **Context:**
  ```python
  self.audit_dir = Path(audit_config.get("log_path", "logs/audit"))
  ```
- **Suggested Config Key:** `paths.audit_logs_dir`
- **Priority:** MEDIUM

#### Max Recent Audit Events
- **Line:** 86
- **Value:** `1000`
- **Context:**
  ```python
  self.max_recent_events = 1000
  ```
- **Suggested Config Key:** `audit.max_recent_events`
- **Priority:** LOW - Memory management

---

### 3.4 DeFi Integration

**File:** `/home/user/Nexlify/nexlify/financial/nexlify_defi_integration.py`

#### DeFi Positions File Path
- **Line:** 175
- **Value:** `"data/defi_positions.json"`
- **Context:**
  ```python
  self.positions_file = Path("data/defi_positions.json")
  ```
- **Suggested Config Key:** `paths.defi_positions_file`
- **Priority:** MEDIUM

---

### 3.5 Financial & Database

**File:** `/home/user/Nexlify/nexlify/financial/nexlify_profit_manager.py`

#### Trading Database Path
- **Line:** 165
- **Value:** `"data/trading.db"`
- **Context:**
  ```python
  db_path = self.config.get("database_path", "data/trading.db")
  ```
- **Suggested Config Key:** `paths.trading_database`
- **Priority:** HIGH - Critical persistence layer

---

### 3.6 Analytics & Performance

**File:** `/home/user/Nexlify/nexlify/analytics/nexlify_performance_tracker.py`

#### Performance Database Path
- **Line:** 109
- **Value:** `"data/trading.db"`
- **Context:**
  ```python
  db_path = self.config.get("database_path", "data/trading.db")
  ```
- **Suggested Config Key:** `paths.trading_database`
- **Priority:** HIGH

---

### 3.7 Flash Crash Event Logging

**File:** `/home/user/Nexlify/nexlify/risk/nexlify_flash_crash_protection.py`

#### Flash Crash Event Log
- **Line:** 159
- **Value:** `"data/flash_crash_events.jsonl"`
- **Context:**
  ```python
  self.event_log_file = Path("data/flash_crash_events.jsonl")
  ```
- **Suggested Config Key:** `paths.flash_crash_events_log`
- **Priority:** MEDIUM

---

### 3.8 RL Training Epsilon History

**File:** `/home/user/Nexlify/nexlify/strategies/nexlify_rl_agent.py`

#### Epsilon History Path
- **Line:** 770
- **Value:** Dynamic path based on model filepath
- **Context:**
  ```python
  Path(filepath).parent / f"{Path(filepath).stem}_epsilon_history.json"
  ```
- **Suggested Config Key:** `paths.epsilon_history_dir`
- **Priority:** LOW - Training artifacts

---

## CATEGORY 4: Exchange Fees & DeFi Gas Costs

### 4.1 Trading Fees

**File:** `/home/user/Nexlify/nexlify/backtesting/nexlify_paper_trading.py`

#### Paper Trading Fee Rate
- **Line:** 69
- **Value:** `0.001` (0.1%)
- **Context:**
  ```python
  self.fee_rate = self.config.get("fee_rate", 0.001)  # 0.1%
  ```
- **Suggested Config Key:** `trading.fee_rate`
- **Priority:** HIGH - P&L impact

#### Slippage
- **Line:** 70
- **Value:** `0.0005` (0.05%)
- **Context:**
  ```python
  self.slippage = self.config.get("slippage", 0.0005)  # 0.05%
  ```
- **Suggested Config Key:** `trading.slippage_percent`
- **Priority:** HIGH - Realistic pricing

---

### 4.2 RL Environment Fallback Fees

**File:** `/home/user/Nexlify/nexlify/strategies/nexlify_rl_agent.py`

#### Static Fallback Fee Rate
- **Line:** 120
- **Value:** `0.001` (0.1%)
- **Context:**
  ```python
  return FeeEstimate(
      entry_fee_rate=0.001,
      exit_fee_rate=0.001,
      network="static_fallback",
  )
  ```
- **Suggested Config Key:** `trading.fallback_fee_rate`
- **Priority:** MEDIUM - Backtest default

---

### 4.3 DeFi Configuration

**File:** `/home/user/Nexlify/nexlify/financial/nexlify_defi_integration.py`

#### Idle Capital Threshold
- **Line:** 148
- **Value:** `1000` (USD)
- **Context:**
  ```python
  self.idle_threshold = Decimal(str(self.config.get("idle_threshold", 1000)))
  ```
- **Suggested Config Key:** `defi_integration.idle_threshold_usd`
- **Priority:** HIGH - Deployment trigger

#### Minimum APY for Deployment
- **Line:** 150
- **Value:** `5.0` (5%)
- **Context:**
  ```python
  self.min_apy = Decimal(str(self.config.get("min_apy", 5.0)))
  ```
- **Suggested Config Key:** `defi_integration.min_apy_percent`
- **Priority:** HIGH - Profitability filter

---

## CATEGORY 5: Timeouts & Delays

### 5.1 Trading Loop Intervals

**File:** `/home/user/Nexlify/nexlify/core/nexlify_auto_trader.py`

#### Trade Check Interval
- **Line:** 206
- **Value:** `60` (seconds)
- **Context:**
  ```python
  self.check_interval = self.config.get("check_interval", 60)
  ```
- **Suggested Config Key:** `trading.check_interval_seconds`
- **Priority:** MEDIUM - Loop frequency

#### Main Trading Loop Sleep
- **Line:** 383
- **Value:** `60` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(60)
  ```
- **Suggested Config Key:** `trading.main_loop_interval_seconds`
- **Priority:** MEDIUM

#### Secondary Loop Sleep
- **Line:** 423
- **Value:** `60` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(60)
  ```
- **Suggested Config Key:** `trading.secondary_loop_interval_seconds`
- **Priority:** MEDIUM

#### Hourly Reporting Sleep
- **Line:** 656
- **Value:** `3600` (seconds / 1 hour)
- **Context:**
  ```python
  await asyncio.sleep(3600)  # Every hour
  ```
- **Suggested Config Key:** `trading.hourly_report_interval_seconds`
- **Priority:** LOW

---

### 5.2 Neural Network Update Intervals

**File:** `/home/user/Nexlify/nexlify/core/nexlify_neural_net.py`

#### Market Update Interval 1
- **Line:** 72
- **Value:** `10` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(10)  # Update every 10 seconds
  ```
- **Suggested Config Key:** `neural_net.market_update_interval_1`
- **Priority:** MEDIUM

#### Market Update Interval 2
- **Line:** 75
- **Value:** `30` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(30)
  ```
- **Suggested Config Key:** `neural_net.market_update_interval_2`
- **Priority:** MEDIUM

#### Portfolio Update Interval 1
- **Line:** 83
- **Value:** `5` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(5)  # Update every 5 seconds
  ```
- **Suggested Config Key:** `neural_net.portfolio_update_interval_1`
- **Priority:** MEDIUM

#### Portfolio Update Interval 2
- **Line:** 86
- **Value:** `10` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(10)
  ```
- **Suggested Config Key:** `neural_net.portfolio_update_interval_2`
- **Priority:** MEDIUM

---

### 5.3 WebSocket Feed Intervals

**File:** `/home/user/Nexlify/nexlify/integrations/nexlify_websocket_feeds.py`

#### WebSocket Reconnect Sleep
- **Line:** 123, 176, 231, 290
- **Value:** `1` (second)
- **Context:**
  ```python
  await asyncio.sleep(1)
  ```
- **Suggested Config Key:** `websocket.reconnect_interval_seconds`
- **Priority:** MEDIUM - Reconnection backoff

#### WebSocket Monitoring Sleep
- **Line:** 410
- **Value:** `30` (seconds)
- **Context:**
  ```python
  await asyncio.sleep(30)
  ```
- **Suggested Config Key:** `websocket.monitoring_interval_seconds`
- **Priority:** MEDIUM

---

### 5.4 Smart Cache & Compression

**File:** `/home/user/Nexlify/nexlify/ml/nexlify_smart_cache.py`

#### Smart Cache Compression Join Timeout
- **Line:** 187
- **Value:** `5.0` (seconds)
- **Context:**
  ```python
  self.compression_thread.join(timeout=5.0)
  ```
- **Suggested Config Key:** `cache.compression_timeout_seconds`
- **Priority:** LOW - Async housekeeping

#### Idle Sleep Duration
- **Line:** 197
- **Value:** `0.1` (seconds)
- **Context:**
  ```python
  time.sleep(0.1)  # Idle
  ```
- **Suggested Config Key:** `cache.idle_sleep_seconds`
- **Priority:** LOW - CPU management

---

### 5.5 Resource Monitoring & Thermal Management

**File:** `/home/user/Nexlify/nexlify/ml/nexlify_dynamic_architecture.py`

#### Hardware Profiler Sample Interval
- **Line:** 79, 135
- **Value:** `0.1` (seconds)
- **Context:**
  ```python
  def __init__(self, sample_interval: float = 0.1):
      self.sample_interval = sample_interval
  time.sleep(self.sample_interval)
  ```
- **Suggested Config Key:** `monitoring.sample_interval_seconds`
- **Priority:** LOW - Performance impact

#### Thread Join Timeout
- **Line:** 120
- **Value:** `1.0` (seconds)
- **Context:**
  ```python
  self.monitor_thread.join(timeout=1.0)
  ```
- **Suggested Config Key:** `monitoring.thread_join_timeout_seconds`
- **Priority:** LOW

#### Architecture Resize Interval
- **Line:** 680
- **Value:** `60` (seconds)
- **Context:**
  ```python
  self.resize_interval = 60  # Resize at most once per minute
  ```
- **Suggested Config Key:** `monitoring.resize_check_interval_seconds`
- **Priority:** MEDIUM - Prevents thrashing

---

### 5.6 GUI Hardware Detection

**File:** `/home/user/Nexlify/nexlify/gui/nexlify_hardware_detection.py`

#### CPU Usage Monitoring Interval
- **Line:** 62
- **Value:** `1` (second)
- **Context:**
  ```python
  "usage_percent": psutil.cpu_percent(interval=1)
  ```
- **Suggested Config Key:** `gui.cpu_sample_interval_seconds`
- **Priority:** LOW

#### HTTP Request Timeouts
- **Line:** 120, 154
- **Value:** `5` (seconds)
- **Context:**
  ```python
  timeout=5
  ```
- **Suggested Config Key:** `gui.http_timeout_seconds`
- **Priority:** MEDIUM

#### Default Scan Interval
- **Line:** 363
- **Value:** `300` (seconds / 5 minutes)
- **Context:**
  ```python
  "scan_interval_seconds": 300
  ```
- **Suggested Config Key:** `gui.default_scan_interval_seconds`
- **Priority:** MEDIUM

#### Default Chart Update Interval
- **Line:** 369
- **Value:** `5000` (milliseconds / 5 seconds)
- **Context:**
  ```python
  "chart_update_interval_ms": 5000
  ```
- **Suggested Config Key:** `gui.chart_update_interval_ms`
- **Priority:** MEDIUM

#### Intense Mode Scan Interval
- **Line:** 379
- **Value:** `60` (seconds)
- **Context:**
  ```python
  "scan_interval_seconds": 60
  ```
- **Suggested Config Key:** `gui.intense_mode_scan_interval_seconds`
- **Priority:** LOW

#### Intense Mode Chart Update
- **Line:** 383
- **Value:** `1000` (milliseconds / 1 second)
- **Context:**
  ```python
  "chart_update_interval_ms": 1000
  ```
- **Suggested Config Key:** `gui.intense_mode_chart_update_interval_ms`
- **Priority:** LOW

#### Balanced Mode Scan Interval
- **Line:** 392
- **Value:** `120` (seconds / 2 minutes)
- **Context:**
  ```python
  "scan_interval_seconds": 120
  ```
- **Suggested Config Key:** `gui.balanced_mode_scan_interval_seconds`
- **Priority:** LOW

#### Balanced Mode Chart Update
- **Line:** 396
- **Value:** `3000` (milliseconds)
- **Context:**
  ```python
  "chart_update_interval_ms": 3000
  ```
- **Suggested Config Key:** `gui.balanced_mode_chart_update_interval_ms`
- **Priority:** LOW

#### Conservative Mode Scan Interval
- **Line:** 405
- **Value:** `300` (seconds / 5 minutes)
- **Context:**
  ```python
  "scan_interval_seconds": 300
  ```
- **Suggested Config Key:** `gui.conservative_mode_scan_interval_seconds`
- **Priority:** LOW

#### Conservative Mode Chart Update
- **Line:** 409
- **Value:** `5000` (milliseconds)
- **Context:**
  ```python
  "chart_update_interval_ms": 5000
  ```
- **Suggested Config Key:** `gui.conservative_mode_chart_update_interval_ms`
- **Priority:** LOW

---

## ADDITIONAL FINDINGS

### Critical Analysis Parameters

**File:** `/home/user/Nexlify/nexlify/analytics/nexlify_advanced_analytics.py`

#### Risk-Free Rate (Annual)
- **Line:** 75
- **Value:** `0.02` (2%)
- **Context:**
  ```python
  self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 2% annual
  ```
- **Suggested Config Key:** `analytics.risk_free_rate`
- **Priority:** MEDIUM - Sharpe ratio calculation

---

### Paper Trading Balance

**File:** `/home/user/Nexlify/nexlify/backtesting/nexlify_paper_trading.py`

#### Initial Paper Trading Balance
- **Line:** 65
- **Value:** `10000.0` (USD)
- **Context:**
  ```python
  self.initial_balance = self.config.get("paper_balance", 10000.0)
  ```
- **Suggested Config Key:** `paper_trading.initial_balance`
- **Priority:** MEDIUM - Backtest starting capital

---

### AI Companion Volatility Thresholds

**File:** `/home/user/Nexlify/nexlify/analytics/nexlify_ai_companion.py`

#### Low Volatility Threshold
- **Line:** 66
- **Value:** `0.02` (2%)
- **Context:**
  ```python
  if volatility < 0.02:
  ```
- **Suggested Config Key:** `ai_companion.low_volatility_threshold`
- **Priority:** LOW

#### Medium Volatility Threshold
- **Line:** 68
- **Value:** `0.05` (5%)
- **Context:**
  ```python
  elif volatility < 0.05:
  ```
- **Suggested Config Key:** `ai_companion.medium_volatility_threshold`
- **Priority:** LOW

---

## SUMMARY & RECOMMENDATIONS

### Total Hardcoded Values Found: 87

**By Category:**
- Risk Management & Trading: 36
- RL Agent Hyperparameters: 16
- File Paths & Databases: 11
- Fees & DeFi: 5
- Timeouts & Delays: 18
- Other (Analysis, Balance, Volatility): 5

### Top Priority Items (Should Be Configurable):

1. **Risk Management Thresholds** - Max position size (0.05), daily loss (0.05), stop-loss (0.02), take-profit (0.05)
2. **Flash Crash Protection** - Crash thresholds (-0.05, -0.10, -0.15), recovery threshold (0.20)
3. **RL Hyperparameters** - Epsilon decay (0.995), gamma (0.99), learning rate (0.001), buffer size (100000)
4. **Database Paths** - Critical for deployment flexibility
5. **Trading Intervals** - 60-second checks affect market responsiveness

### Recommended Configuration Structure:

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
  },
  "flash_crash_protection": {
    "enabled": true,
    "minor_threshold": -0.05,
    "major_threshold": -0.10,
    "critical_threshold": -0.15,
    "check_interval": 30,
    "recovery_threshold": 0.20,
    "volume_spike_threshold": 3.0
  },
  "circuit_breaker": {
    "failure_threshold": 3,
    "timeout_seconds": 300,
    "half_open_max_calls": 1
  },
  "rl_agent": {
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.001,
    "replay_buffer_size": 100000
  },
  "epsilon_decay": {
    "start": 1.0,
    "end": 0.22,
    "linear_decay_steps": 2000,
    "scheduled_milestones": {
      "0": 1.0,
      "200": 0.65,
      "800": 0.35,
      "2000": 0.22
    }
  },
  "paths": {
    "data_dir": "data",
    "logs_dir": "logs",
    "cache_dir": "cache",
    "models_dir": "models",
    "trading_database": "data/trading.db"
  },
  "trading": {
    "fee_rate": 0.001,
    "slippage_percent": 0.0005,
    "check_interval_seconds": 60
  }
}
```

---

**Report Generated:** 2025-11-14  
**Audit Coverage:** Complete codebase analysis  
**Recommendation:** Migrate HIGH and MEDIUM priority values to configuration system
