# Training Dashboard Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-14

This guide covers the real-time training monitoring system for Nexlify RL agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Components](#components)
4. [Integration](#integration)
5. [Configuration](#configuration)
6. [Features](#features)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The training dashboard provides real-time monitoring and visualization of RL agent training with:

- **Live Dashboard**: Web-based dashboard with real-time plots
- **Metrics Logging**: High-performance async logging with minimal overhead
- **Alert System**: Email/Slack notifications for critical events
- **Experiment Tracking**: Compare multiple training runs
- **Performance**: < 1% overhead, < 1ms per log operation

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Training Loop                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  Episode → Metrics Logger → Dashboard        │  │
│  │              ↓              ↓                 │  │
│  │         Alert System    Experiment Tracker   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         ↓                ↓              ↓
    Disk Storage    Email/Slack    Leaderboard
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install dash plotly pandas requests
```

### 2. Launch Dashboard (Demo Mode)

```bash
python launch_training_dashboard.py --demo
```

Dashboard will be available at: http://localhost:8050

### 3. Use in Training Script

```python
from nexlify.monitoring.metrics_logger import MetricsLogger
from nexlify.monitoring.training_dashboard import TrainingDashboard

# Create metrics logger
logger = MetricsLogger(experiment_name="my_training")

# Start dashboard
dashboard = TrainingDashboard(logger, port=8050)
dashboard.start(blocking=False)

# Training loop
for episode in range(1000):
    # ... your training code ...

    # Log metrics
    logger.log_episode(
        episode=episode,
        profit=profit,
        sharpe=sharpe,
        win_rate=win_rate,
        epsilon=epsilon,
        learning_rate=lr
    )

    logger.log_model_metrics(
        loss=loss,
        q_values=q_values
    )
```

---

## Components

### 1. MetricsLogger

High-performance metrics logging with async I/O.

**Features:**
- < 1ms logging overhead
- Async I/O for disk writes
- Thread-safe operations
- Automatic aggregation and smoothing
- JSON and CSV export

**Usage:**

```python
from nexlify.monitoring.metrics_logger import MetricsLogger

logger = MetricsLogger(
    experiment_name="dqn_training",
    output_dir="training_logs",
    buffer_size=100,
    auto_save_interval=50,
    enable_async=True
)

# Log episode
logger.log_episode(
    episode=1,
    profit=100.0,
    sharpe=1.5,
    win_rate=0.6,
    drawdown=50.0,
    num_trades=10,
    epsilon=0.5,
    learning_rate=0.001
)

# Log model metrics
logger.log_model_metrics(
    loss=0.5,
    q_values=[1.0, 2.0, 3.0],
    gradients={'mean': 0.01, 'std': 0.001}
)

# Get statistics
stats = logger.get_statistics()
print(f"Best profit: ${stats['best_profit']:.2f}")

# Save
logger.save_metrics(format='json')
logger.save_metrics(format='csv')
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `log_episode()` | Log episode-level metrics |
| `log_model_metrics()` | Log model training metrics |
| `get_latest_episode()` | Get most recent episode |
| `get_best_episode(metric)` | Get best episode by metric |
| `get_smoothed_metric(name, window)` | Get smoothed metric value |
| `get_statistics()` | Get comprehensive statistics |
| `save_metrics(format)` | Save to disk (JSON/CSV) |

### 2. TrainingDashboard

Real-time web dashboard with live plots.

**Features:**
- 8 real-time plots updating every 2 seconds
- Color-coded KPI cards
- Multiple theme options (cyberpunk, dark, light)
- Responsive design
- Low resource usage

**Plots:**
1. Profit/Loss curve (raw + smoothed)
2. Training loss (with smoothing)
3. Epsilon decay
4. Learning rate schedule
5. Win rate trend
6. Sharpe ratio evolution
7. Maximum drawdown
8. Q-value distribution

**Usage:**

```python
from nexlify.monitoring.training_dashboard import TrainingDashboard

dashboard = TrainingDashboard(
    metrics_logger=logger,
    port=8050,
    update_interval=2000,  # ms
    theme='cyberpunk'  # or 'dark', 'light'
)

# Start in background
dashboard.start(blocking=False)

# ... training happens ...

# Stop when done
dashboard.stop()
```

**Launching Standalone:**

```bash
# Basic launch
python launch_training_dashboard.py

# Custom port
python launch_training_dashboard.py --port 8080

# Different theme
python launch_training_dashboard.py --theme dark

# Demo mode
python launch_training_dashboard.py --demo
```

### 3. AlertSystem

Multi-channel alert notifications.

**Features:**
- Email alerts (SMTP)
- Slack webhooks
- Configurable thresholds
- Alert throttling
- Health checks

**Usage:**

```python
from nexlify.monitoring.alert_system import AlertSystem, AlertThresholds

config = {
    'enable_alerts': True,
    'email': {
        'enabled': True,
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'username': 'your@email.com',
        'password': 'your_password',
        'to_addrs': ['recipient@email.com']
    },
    'slack': {
        'enabled': True,
        'webhook_url': 'https://hooks.slack.com/...'
    },
    'thresholds': AlertThresholds.MODERATE
}

alert_system = AlertSystem(config)

# Send alert
alert_system.send_alert(
    level='warning',
    title='Slow Learning',
    message='No improvement in 100 episodes'
)

# Check training health
alert_system.check_training_health(
    latest_episode=latest,
    recent_history=recent_episodes
)

# Training complete notification
alert_system.send_training_complete(
    total_episodes=1000,
    best_profit=5000.0,
    best_sharpe=2.5,
    training_time=3600.0
)
```

**Alert Levels:**

| Level | Color | Use Case |
|-------|-------|----------|
| `info` | Green | Normal events (training complete, new best) |
| `warning` | Yellow | Potential issues (slow learning, high loss) |
| `critical` | Red | Serious problems (NaN loss, extreme values) |

**Predefined Thresholds:**

```python
# Conservative (more alerts)
AlertThresholds.CONSERVATIVE = {
    'no_improvement_episodes': 50,
    'critical_loss_threshold': 500.0,
    'min_profit_warning': -200.0
}

# Moderate (balanced)
AlertThresholds.MODERATE = {
    'no_improvement_episodes': 100,
    'critical_loss_threshold': 1000.0,
    'min_profit_warning': -500.0
}

# Aggressive (fewer alerts)
AlertThresholds.AGGRESSIVE = {
    'no_improvement_episodes': 200,
    'critical_loss_threshold': 2000.0,
    'min_profit_warning': -1000.0
}
```

### 4. ExperimentTracker

Track and compare multiple experiments.

**Features:**
- Hyperparameter tracking
- Experiment comparison
- Leaderboard
- Export to JSON/CSV
- Archive/delete old experiments

**Usage:**

```python
from nexlify.monitoring.experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(experiments_dir="experiments")

# Create experiment
exp_id = tracker.create_experiment(
    name="dqn_baseline",
    hyperparameters={
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_decay': 0.995
    },
    description="Baseline DQN with standard hyperparameters",
    tags=['dqn', 'baseline']
)

# Log results
for episode in range(1000):
    # ... training ...

    tracker.log_result(
        exp_id,
        episode=episode,
        profit=profit,
        sharpe=sharpe,
        win_rate=win_rate
    )

# Complete experiment
tracker.complete_experiment(exp_id, final_metrics={
    'total_episodes': 1000,
    'best_profit': 5000.0
})

# Compare experiments
comparison = tracker.compare_experiments([exp_id1, exp_id2])

# Get leaderboard
leaderboard = tracker.get_leaderboard(metric='profit', top_n=10)

# Export comparison
tracker.export_comparison([exp_id1, exp_id2], format='json')
```

---

## Integration

### Full Integration Example

See `examples/train_with_monitoring.py` for a complete example.

**Key Integration Points:**

```python
# 1. Setup (before training)
metrics_logger = MetricsLogger(experiment_name=exp_name)
dashboard = TrainingDashboard(metrics_logger, port=8050)
alert_system = AlertSystem(config)
tracker = ExperimentTracker()

dashboard.start(blocking=False)
exp_id = tracker.create_experiment(name, hyperparameters)

# 2. During training (each episode)
metrics_logger.log_episode(episode, profit, sharpe, ...)
metrics_logger.log_model_metrics(loss, q_values, ...)
tracker.log_result(exp_id, episode, profit, ...)

# Check health periodically
if episode % 10 == 0:
    alert_system.check_training_health(latest, recent)

# 3. After training
tracker.complete_experiment(exp_id)
alert_system.send_training_complete(...)
metrics_logger.save_metrics()
dashboard.stop()
```

### Minimal Integration (Just Logging)

```python
# Minimal - just logging, no dashboard
logger = MetricsLogger("my_training")

for episode in range(1000):
    # ... training ...
    logger.log_episode(episode, profit, sharpe)

logger.save_metrics()
```

---

## Configuration

### Config File (neural_config.json)

```json
{
  "training_dashboard": {
    "enabled": true,
    "port": 8050,
    "update_interval_ms": 2000,
    "theme": "cyberpunk",
    "auto_start": false,

    "metrics_logging": {
      "enabled": true,
      "output_dir": "training_logs",
      "buffer_size": 100,
      "auto_save_interval": 50,
      "enable_async": true
    },

    "alerts": {
      "enabled": true,
      "email": {
        "enabled": false,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "your@email.com",
        "password": "your_password"
      },
      "slack": {
        "enabled": false,
        "webhook_url": "https://hooks.slack.com/..."
      },
      "thresholds": {
        "no_improvement_episodes": 100,
        "critical_loss_threshold": 1000.0,
        "min_profit_warning": -500.0
      }
    },

    "experiment_tracking": {
      "enabled": true,
      "experiments_dir": "experiments",
      "auto_save": true
    }
  }
}
```

### Loading Config

```python
import json

with open('config/neural_config.json') as f:
    config = json.load(f)

dashboard_config = config['training_dashboard']
alert_config = dashboard_config['alerts']

# Use config
alert_system = AlertSystem(alert_config)
```

---

## Features

### Real-Time Plots

**1. Profit/Loss Chart**
- Raw episode profits (scatter)
- 10-episode smoothed line
- Zero baseline

**2. Training Loss**
- Model training loss
- 20-step smoothing
- Log scale Y-axis

**3. Epsilon Decay**
- Exploration rate over time
- Shows epsilon-greedy decay schedule

**4. Learning Rate Schedule**
- Current learning rate
- Shows LR decay/scheduling

**5. Win Rate Trend**
- Rolling 20-episode win rate
- 50% reference line
- Percentage scale (0-100%)

**6. Sharpe Ratio Evolution**
- 10-episode smoothed Sharpe ratio
- Reference lines at 0 and 1.0
- Shows risk-adjusted performance

**7. Drawdown Curve**
- Maximum drawdown per episode
- Shows risk exposure

**8. Q-Value Distribution**
- Mean Q-values over time
- Standard deviation band
- Shows learning progress

### KPI Cards

Color-coded status indicators:

- **Green**: Good performance
- **Yellow**: Warning
- **Red**: Critical issue

**Metrics:**
- Current episode
- Latest profit
- Best profit (all-time)
- Best episode number
- Recent average (last 50)
- Sharpe ratio

### Alert Conditions

**Automatic Alerts:**
1. No improvement for N episodes
2. Critical loss threshold exceeded
3. Profit below warning threshold
4. NaN values detected
5. Training complete
6. New best model found

### Performance

**Benchmarks:**

| Operation | Time | Notes |
|-----------|------|-------|
| log_episode() | < 1ms | Single episode log |
| log_model_metrics() | < 1ms | Single metrics log |
| save_metrics() | 10-50ms | JSON/CSV write |
| Dashboard update | ~100ms | Every 2 seconds |
| Alert send | 50-200ms | Email/Slack |

**Overhead:**
- Training overhead: < 1%
- Memory overhead: ~50-100MB
- Disk usage: ~1-10MB per 1000 episodes

---

## Best Practices

### 1. Logging Frequency

**Recommended:**
```python
# Log every episode
logger.log_episode(episode, ...)

# Log model metrics every 5-10 episodes
if episode % 5 == 0:
    logger.log_model_metrics(loss, q_values, ...)
```

**Avoid:**
```python
# Don't log every training step (too frequent)
for batch in dataloader:
    logger.log_model_metrics(...)  # NO!
```

### 2. Auto-Save Interval

```python
# Good: Auto-save every 50-100 episodes
MetricsLogger(auto_save_interval=50)

# Too frequent (disk I/O overhead)
MetricsLogger(auto_save_interval=1)

# Too infrequent (risk of data loss)
MetricsLogger(auto_save_interval=1000)
```

### 3. Alert Throttling

```python
# Health checks every 10-20 episodes
if episode % 10 == 0:
    alert_system.check_training_health(latest, recent)

# Not every episode (alert spam)
# alert_system.check_training_health(...)  # NO!
```

### 4. Dashboard Update Interval

```python
# Good: 2-5 seconds
TrainingDashboard(update_interval=2000)

# Too fast (high CPU usage)
TrainingDashboard(update_interval=100)

# Too slow (not "real-time")
TrainingDashboard(update_interval=30000)
```

### 5. Experiment Organization

```python
# Good: Descriptive names and tags
tracker.create_experiment(
    name="dqn_lr0.001_gamma0.99",
    tags=['dqn', 'tuning', 'lr_sweep']
)

# Bad: Generic names
tracker.create_experiment(name="test")
```

### 6. Resource Management

```python
# Use context managers
with MetricsLogger("training") as logger:
    # Training code
    pass
# Auto-saves on exit

# Or explicit cleanup
logger = MetricsLogger("training")
try:
    # Training
    pass
finally:
    logger.close()  # Ensures final save
```

---

## Troubleshooting

### Dashboard Not Loading

**Issue:** Dashboard shows blank page

**Solutions:**
1. Check Dash is installed: `pip install dash plotly`
2. Verify port is not in use: `lsof -i :8050`
3. Try different port: `--port 8080`
4. Check browser console for errors

### High Memory Usage

**Issue:** Python process using too much memory

**Solutions:**
1. Reduce buffer size:
   ```python
   MetricsLogger(buffer_size=50)  # Default: 100
   ```

2. Increase auto-save frequency:
   ```python
   MetricsLogger(auto_save_interval=25)  # Default: 50
   ```

3. Clear old episodes:
   ```python
   # Keep only last N episodes
   logger.episode_metrics = logger.episode_metrics[-1000:]
   ```

### Slow Logging

**Issue:** Logging taking > 10ms per call

**Solutions:**
1. Enable async mode:
   ```python
   MetricsLogger(enable_async=True)
   ```

2. Reduce buffer flushes:
   ```python
   MetricsLogger(auto_save_interval=100)
   ```

3. Disable unnecessary metrics:
   ```python
   # Don't log heavy data every episode
   if episode % 10 == 0:
       logger.log_model_metrics(...)
   ```

### Alerts Not Sending

**Issue:** Email/Slack alerts not arriving

**Solutions:**

**Email:**
1. Check SMTP credentials
2. Enable "Less secure apps" (Gmail)
3. Use app-specific password
4. Check firewall/port 587

**Slack:**
1. Verify webhook URL
2. Check workspace permissions
3. Test webhook manually:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test"}' \
     YOUR_WEBHOOK_URL
   ```

### Dashboard Plots Not Updating

**Issue:** Plots frozen or not updating

**Solutions:**
1. Check browser console for errors
2. Verify training is actually logging:
   ```python
   print(len(logger.episode_metrics))
   ```
3. Restart dashboard
4. Clear browser cache

### Experiment Not Saving

**Issue:** Experiment data lost after restart

**Solutions:**
1. Explicitly save:
   ```python
   tracker.save_experiment(exp_id)
   ```

2. Enable auto-save:
   ```python
   tracker.complete_experiment(exp_id)  # Auto-saves
   ```

3. Check directory permissions:
   ```bash
   ls -la experiments/
   ```

---

## Advanced Usage

### Custom Metrics

```python
# Log custom metrics
logger.log_episode(
    episode=1,
    profit=100.0,
    # Custom metrics
    custom_score=0.95,
    volatility=0.15,
    risk_score=0.3
)
```

### Multi-Experiment Comparison

```python
# Compare multiple experiments
exp_ids = ['exp1_timestamp', 'exp2_timestamp', 'exp3_timestamp']
comparison = tracker.compare_experiments(exp_ids)

# Export for analysis
tracker.export_comparison(exp_ids, format='csv')

# Analyze in pandas
import pandas as pd
df = pd.read_csv('comparison_timestamp.csv')
print(df.describe())
```

### Programmatic Dashboard Control

```python
# Start dashboard
dashboard.start(blocking=False)

# Check if running
if dashboard.is_running():
    print("Dashboard active")

# Stop dashboard
dashboard.stop()
```

### Custom Alert Thresholds

```python
custom_thresholds = {
    'no_improvement_episodes': 75,
    'critical_loss_threshold': 750.0,
    'min_profit_warning': -300.0
}

alert_system = AlertSystem({
    'enable_alerts': True,
    'thresholds': custom_thresholds
})
```

---

## API Reference

### MetricsLogger

```python
class MetricsLogger:
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "training_logs",
        buffer_size: int = 100,
        auto_save_interval: int = 50,
        enable_async: bool = True
    )

    def log_episode(
        self,
        episode: int,
        profit: float,
        sharpe: float = 0.0,
        win_rate: float = 0.0,
        drawdown: float = 0.0,
        num_trades: int = 0,
        epsilon: float = 0.0,
        learning_rate: float = 0.0,
        **kwargs
    ) -> None

    def log_model_metrics(
        self,
        loss: float,
        q_values: Optional[List[float]] = None,
        gradients: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None

    def get_latest_episode(self) -> Dict[str, Any]
    def get_best_episode(self, metric: str = 'profit') -> Optional[Dict[str, Any]]
    def get_statistics(self) -> Dict[str, Any]
    def save_metrics(self, format: str = 'json') -> Path
    def close(self) -> None
```

### TrainingDashboard

```python
class TrainingDashboard:
    def __init__(
        self,
        metrics_logger: MetricsLogger,
        port: int = 8050,
        update_interval: int = 2000,
        theme: str = 'cyberpunk'
    )

    def start(self, blocking: bool = False) -> None
    def stop(self) -> None
    def is_running(self) -> bool
```

### AlertSystem

```python
class AlertSystem:
    def __init__(self, config: Dict[str, Any])

    def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool

    def check_training_health(
        self,
        latest_episode: Dict[str, Any],
        recent_history: List[Dict[str, Any]]
    ) -> None

    def send_training_complete(...) -> None
    def send_new_best_model(...) -> None
```

### ExperimentTracker

```python
class ExperimentTracker:
    def __init__(self, experiments_dir: str = "experiments")

    def create_experiment(
        self,
        name: str,
        hyperparameters: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str

    def log_result(self, exp_id: str, episode: int, **metrics) -> None
    def complete_experiment(
        self,
        exp_id: str,
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> None

    def compare_experiments(
        self,
        exp_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]

    def get_leaderboard(
        self,
        metric: str = 'profit',
        top_n: int = 10
    ) -> List[Dict[str, Any]]
```

---

## Conclusion

The training dashboard provides comprehensive monitoring with minimal overhead. Key benefits:

- **Real-time visibility** into training progress
- **Quick identification** of issues
- **Easy comparison** of experiments
- **Professional presentation** of results
- **< 1% performance** overhead

For more information, see:
- Example: `examples/train_with_monitoring.py`
- Tests: `tests/test_monitoring.py`
- Config: `config/neural_config.example.json`

---

**Questions or issues?** Check the troubleshooting section or open an issue on GitHub.
