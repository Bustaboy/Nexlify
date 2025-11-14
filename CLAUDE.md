# CLAUDE.md - AI Assistant Guide for Nexlify

**Version:** 2.0.7.7
**Last Updated:** 2025-11-14
**Repository:** https://github.com/Bustaboy/Nexlify

This document provides comprehensive guidance for AI assistants (like Claude) working with the Nexlify codebase. It covers architecture, conventions, workflows, and best practices.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Architecture](#codebase-architecture)
3. [Directory Structure](#directory-structure)
4. [Module Reference](#module-reference)
5. [Development Workflows](#development-workflows)
6. [Testing Guidelines](#testing-guidelines)
7. [Coding Conventions](#coding-conventions)
8. [Configuration Management](#configuration-management)
9. [Common Tasks](#common-tasks)
10. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Nexlify** is an advanced AI-powered cryptocurrency trading platform with:
- Neural network-based autonomous trading
- Multi-exchange support (Binance, Kraken, Coinbase, etc.)
- Comprehensive risk management and circuit breakers
- DeFi integration for yield optimization
- Cyberpunk-themed PyQt5 GUI
- Reinforcement learning (RL) agents using DQN variants
- Real-time monitoring and Telegram notifications

### Key Technologies
- **Python:** 3.9+ (tested on 3.10, 3.11)
- **ML/RL:** PyTorch 2.1.0, TensorFlow 2.13.0, scikit-learn
- **Trading:** CCXT 4.1.22 for exchange connectivity
- **GUI:** PyQt5 5.15.10 with custom cyberpunk theme
- **Database:** SQLAlchemy 2.0.21 with SQLite
- **Testing:** pytest 7.4.2 with asyncio support
- **Async:** aiohttp, asyncio, qasync for GUI integration

### Project Maturity
- **90+ Python files** (~39,500 lines of code)
- **70-90% test coverage** target for core modules
- **24 documentation files** covering all major features
- **Production-ready** with comprehensive error handling

---

## Codebase Architecture

### Layered Architecture

```
Layer 0: Foundation (no internal dependencies)
  ├── utils/          - Error handling, utilities
  ├── security/       - Authentication, integrity monitoring
  └── ml/             - ML components, GPU optimization

Layer 1: Core Business Logic
  ├── core/           - Trading engines, neural networks
  ├── risk/           - Risk management, circuit breakers
  └── strategies/     - RL agents, trading strategies

Layer 2: Integration & Services
  ├── financial/      - DeFi, profit management, tax reporting
  ├── analytics/      - Performance tracking, analytics
  └── backtesting/    - Paper trading, backtesting

Layer 3: User Interface
  └── gui/            - Cyberpunk GUI (2,452 lines)
```

### Design Patterns

**1. Dependency Injection**
```python
class RiskManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_handler = get_error_handler()
```

**2. Singleton Pattern (Error Handler)**
```python
_error_handler_instance = None

def get_error_handler():
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = ErrorHandler()
    return _error_handler_instance
```

**3. Strategy Pattern**
Used in `nexlify/strategies/nexlify_multi_strategy.py` for dynamic strategy selection.

**4. Observer Pattern**
Circuit breaker notifies listeners on state changes.

**5. Factory Pattern**
Agent creation based on configuration type.

### Key Architectural Decisions

1. **Graceful Degradation:** Optional imports with try/except for advanced features
2. **Backward Compatibility:** Aliases for renamed classes (e.g., `AutoTrader` → `AutoExecutionEngine`)
3. **Centralized Error Handling:** Single error handler singleton used throughout
4. **Configuration-Driven:** All behavior configurable via `neural_config.json`
5. **Modular Design:** Each component can function independently
6. **Lazy Imports:** Package-level imports are lazy to avoid circular dependencies
7. **Type Safety:** Extensive use of type hints throughout

---

## Directory Structure

```
Nexlify/
├── nexlify/                      # Main package (67 files, ~31,500 lines)
│   ├── __init__.py              # Package exports (lazy loading)
│   ├── core/                    # Core trading engines (4 files)
│   │   ├── arasaka_neural_net.py          # Main AI trading engine
│   │   ├── nexlify_neural_net.py          # Base neural network
│   │   ├── nexlify_auto_trader.py         # Auto-execution engine
│   │   └── nexlify_trading_integration.py # Exchange integration
│   ├── strategies/              # Trading strategies (7 files)
│   │   ├── nexlify_rl_agent.py            # Base DQN agent
│   │   ├── nexlify_adaptive_rl_agent.py   # Self-tuning agent
│   │   ├── nexlify_ultra_optimized_rl_agent.py  # GPU-optimized
│   │   ├── nexlify_multi_strategy.py      # Strategy orchestration
│   │   └── nexlify_multi_timeframe.py     # Multi-timeframe analysis
│   ├── risk/                    # Risk management (4 files)
│   │   ├── nexlify_risk_manager.py        # Position sizing, Kelly criterion
│   │   ├── nexlify_circuit_breaker.py     # Exchange failure protection
│   │   ├── nexlify_flash_crash_protection.py  # Crash detection
│   │   └── nexlify_emergency_kill_switch.py   # Emergency shutdown
│   ├── security/                # Security features (5 files)
│   │   ├── nexlify_security_suite.py      # Security orchestration
│   │   ├── nexlify_advanced_security.py   # Advanced features
│   │   ├── nexlify_pin_manager.py         # PIN authentication
│   │   ├── nexlify_integrity_monitor.py   # File integrity
│   │   └── nexlify_audit_trail.py         # Audit logging
│   ├── ml/                      # Machine learning (14 files)
│   │   ├── nexlify_feature_engineering.py # Feature creation
│   │   ├── nexlify_gpu_optimizations.py   # GPU optimization
│   │   ├── nexlify_smart_cache.py         # LZ4 caching
│   │   ├── nexlify_model_compilation.py   # ONNX/TensorRT
│   │   └── nexlify_sentiment_analysis.py  # Sentiment features
│   ├── financial/               # Financial services (4 files)
│   │   ├── nexlify_defi_integration.py    # DeFi protocols
│   │   ├── nexlify_profit_manager.py      # Profit withdrawal
│   │   ├── nexlify_tax_reporter.py        # Tax reporting
│   │   └── nexlify_portfolio_rebalancer.py # Rebalancing
│   ├── analytics/               # Analytics (3 files)
│   │   ├── nexlify_performance_tracker.py # Trade tracking
│   │   ├── nexlify_advanced_analytics.py  # Advanced analytics
│   │   └── nexlify_ai_companion.py        # AI assistant
│   ├── backtesting/             # Testing systems (5 files)
│   │   ├── nexlify_backtester.py          # Backtesting engine
│   │   ├── nexlify_paper_trading.py       # Paper trading
│   │   └── nexlify_paper_trading_orchestrator.py
│   ├── gui/                     # User interface (9 files)
│   │   ├── cyber_gui.py                   # Main GUI (2,452 lines)
│   │   ├── nexlify_gui_integration.py     # GUI-backend integration
│   │   └── nexlify_cyberpunk_effects.py   # Visual effects
│   ├── integrations/            # External services (2 files)
│   │   ├── nexlify_websocket_server.py    # WebSocket API
│   │   └── nexlify_telegram_notifier.py   # Telegram alerts
│   ├── environments/            # RL environments (3 files)
│   │   └── nexlify_trading_env.py         # Trading environment
│   └── utils/                   # Utilities (2 files)
│       └── error_handler.py               # Centralized error handling
│
├── tests/                       # Test suite (4 files, ~64,000 lines)
│   ├── conftest.py                        # Shared fixtures
│   ├── test_risk_manager.py               # 16,718 lines
│   ├── test_circuit_breaker.py            # 16,204 lines
│   ├── test_performance_tracker.py        # 15,615 lines
│   └── test_paper_trading_system.py       # 15,410 lines
│
├── scripts/                     # Standalone scripts
│   ├── nexlify_launcher.py                # Main launcher
│   ├── setup_nexlify.py                   # Initial setup
│   └── train_rl_agent.py                  # RL training
│
├── config/                      # Configuration files
│   └── neural_config.example.json         # Example config (DO NOT COMMIT REAL CONFIG)
│
├── docs/                        # Documentation (24 markdown files)
│   ├── QUICK_REFERENCE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   └── AUTO_TRADER_GUIDE.md
│
├── data/                        # Runtime data (gitignored)
│   └── trading.db                         # SQLite database
│
├── nexlify_training/            # Training orchestration
├── nexlify_environments/        # Custom RL environments
├── nexlify_data/                # Data fetchers
├── examples/                    # Example integrations
│
├── [Root training scripts]      # 19 training scripts (~8,000 lines)
│   ├── train_ultimate_full_pipeline.py    # Complete pipeline
│   ├── train_complete_with_auto_retrain.py
│   └── train_with_historical_data.py
│
├── requirements.txt             # Dependencies (128 lines)
├── setup.py                     # Package configuration
├── pytest.ini                   # Test configuration
├── .gitignore                   # Git ignore patterns
└── README.md                    # User documentation
```

---

## Module Reference

### Core Trading (nexlify/core/)

**arasaka_neural_net.py** - Main AI trading engine
- Autonomous pair selection using neural networks
- Real-time market scanning and analysis
- Integration with all exchange connectors
- Location: `nexlify/core/arasaka_neural_net.py`

**nexlify_auto_trader.py** (alias: AutoExecutionEngine)
- Automated trade execution
- Fee calculation and accounting
- Order management
- Location: `nexlify/core/nexlify_auto_trader.py`

**nexlify_trading_integration.py**
- Multi-exchange integration manager
- CCXT wrapper with error handling
- Rate limiting and retry logic
- Location: `nexlify/core/nexlify_trading_integration.py`

### Risk Management (nexlify/risk/)

**nexlify_risk_manager.py**
- Position sizing using Kelly criterion
- Stop-loss and take-profit management
- Risk validation for all trades
- Max drawdown tracking
- Location: `nexlify/risk/nexlify_risk_manager.py`

**nexlify_circuit_breaker.py**
- Automatic trading halt on consecutive failures
- Per-exchange failure tracking
- Configurable thresholds and cooldown
- Location: `nexlify/risk/nexlify_circuit_breaker.py`

**nexlify_flash_crash_protection.py**
- Multi-threshold price drop detection
- Automatic position closing on crash
- Recovery monitoring
- Location: `nexlify/risk/nexlify_flash_crash_protection.py`

**nexlify_emergency_kill_switch.py**
- Instant system shutdown
- Position closing and backup
- Telegram emergency notifications
- Location: `nexlify/risk/nexlify_emergency_kill_switch.py`

### Strategies (nexlify/strategies/)

**nexlify_rl_agent.py** - Base DQN agent
- Experience replay buffer
- Target network with soft updates
- Epsilon-greedy exploration
- Location: `nexlify/strategies/nexlify_rl_agent.py`

**nexlify_adaptive_rl_agent.py** - Self-tuning agent
- Dynamic architecture adjustment
- Auto-tuning hyperparameters
- Performance-based adaptation
- Location: `nexlify/strategies/nexlify_adaptive_rl_agent.py`

**nexlify_ultra_optimized_rl_agent.py** - GPU-optimized
- Mixed precision training (FP16)
- Dynamic batch sizing
- Vendor-specific optimizations
- Location: `nexlify/strategies/nexlify_ultra_optimized_rl_agent.py`

### Security (nexlify/security/)

**nexlify_pin_manager.py**
- PIN authentication with lockout
- Session timeout management
- Secure PIN storage (Argon2)
- Location: `nexlify/security/nexlify_pin_manager.py`

**nexlify_integrity_monitor.py**
- File integrity verification (SHA-256)
- Process monitoring
- Tamper detection
- Location: `nexlify/security/nexlify_integrity_monitor.py`

### Machine Learning (nexlify/ml/)

**nexlify_feature_engineering.py**
- Automated feature creation
- Technical indicators
- Market microstructure features
- Location: `nexlify/ml/nexlify_feature_engineering.py`

**nexlify_gpu_optimizations.py**
- Vendor-specific GPU optimization
- Thermal monitoring
- Memory management
- Location: `nexlify/ml/nexlify_gpu_optimizations.py`

**nexlify_smart_cache.py**
- LZ4-compressed caching (2-3 GB/s decompression)
- Automatic invalidation
- Memory-efficient storage
- Location: `nexlify/ml/nexlify_smart_cache.py`

### Financial (nexlify/financial/)

**nexlify_defi_integration.py**
- Aave, Uniswap, PancakeSwap integration
- Auto-compound idle funds
- Gas optimization
- Location: `nexlify/financial/nexlify_defi_integration.py`

**nexlify_profit_manager.py**
- Automated profit withdrawal
- Cold wallet integration
- Threshold-based transfers
- Location: `nexlify/financial/nexlify_profit_manager.py`

**nexlify_tax_reporter.py**
- Trade logging for tax compliance
- FIFO/LIFO calculation
- Export to JSON/CSV
- Location: `nexlify/financial/nexlify_tax_reporter.py`

### Analytics (nexlify/analytics/)

**nexlify_performance_tracker.py**
- Trade history tracking
- Sharpe ratio calculation
- Drawdown analysis
- Win rate statistics
- Location: `nexlify/analytics/nexlify_performance_tracker.py`

### GUI (nexlify/gui/)

**cyber_gui.py** - Main GUI (2,452 lines)
- Cyberpunk-themed interface
- Real-time charts and monitoring
- Settings and configuration
- Emergency controls
- Location: `nexlify/gui/cyber_gui.py`

### Utilities (nexlify/utils/)

**error_handler.py** - Centralized error handling
- Global exception handler
- Error logging and categorization
- Telegram error notifications
- Location: `nexlify/utils/error_handler.py`
- **Usage:** `from nexlify.utils.error_handler import get_error_handler`

---

## Development Workflows

### Initial Setup

```bash
# Clone repository
git clone https://github.com/Bustaboy/Nexlify.git
cd Nexlify

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy example configuration
cp config/neural_config.example.json config/neural_config.json

# Edit configuration (set API keys, PIN, etc.)
# IMPORTANT: Never commit config/neural_config.json
```

### Branch Strategy

- **main** - Stable production releases
- **develop** - Development branch
- **claude/** - AI assistant feature branches (e.g., `claude/feature-name-sessionid`)

**Creating a feature branch:**
```bash
git checkout -b claude/feature-name-$(date +%s)
```

### Testing Workflow

```bash
# Quick validation (30 seconds)
python test_training_pipeline.py --quick

# Full test suite with coverage (2-3 minutes)
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_risk_manager.py -v

# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run with coverage threshold
pytest --cov=. --cov-fail-under=70
```

### Code Quality

```bash
# Format code
black nexlify/
black tests/

# Sort imports
isort nexlify/
isort tests/

# Type checking
mypy nexlify/

# Linting
pylint nexlify/
```

### Commit Conventions

Use conventional commits:
- `feat: Add new feature`
- `fix: Fix bug in risk manager`
- `docs: Update CLAUDE.md`
- `test: Add tests for circuit breaker`
- `refactor: Reorganize strategies module`
- `perf: Optimize GPU memory usage`
- `chore: Update dependencies`

### CI/CD Pipeline (.github/workflows/tests.yml)

**Triggers:** Push to main, develop, claude/**, pull requests
**Matrix:** Python 3.10, 3.11
**Steps:**
1. Install dependencies (with pip cache)
2. Install PyTorch CPU version
3. Run quick tests
4. Run pytest with coverage
5. Upload to Codecov
6. Store HTML coverage reports

**All tests run on push to claude/** branches**

---

## Testing Guidelines

### Test Structure

Tests use **pytest** with **asyncio** support. All tests are in `tests/` directory.

**conftest.py** - Shared fixtures and configuration:
```python
@pytest.fixture
def test_config():
    """Standard test configuration"""
    return {
        'risk_management': {...},
        'circuit_breaker': {...}
    }
```

### Test Categories (Markers)

```python
@pytest.mark.unit           # Fast, isolated tests
@pytest.mark.integration    # Multi-component tests
@pytest.mark.slow           # Long-running tests
@pytest.mark.requires_network  # Needs internet
@pytest.mark.requires_gpu   # Needs GPU
```

**Running specific categories:**
```bash
pytest -m "not slow"              # Skip slow tests
pytest -m integration             # Only integration tests
pytest -m "unit and not slow"     # Fast unit tests only
```

### Test Pattern

```python
@pytest.mark.asyncio
async def test_feature_name(fixture):
    """Test description following pytest conventions"""
    # Arrange
    manager = RiskManager(test_config)
    trade_params = {
        'symbol': 'BTC/USDT',
        'amount': 0.1,
        'price': 50000
    }

    # Act
    result = await manager.validate_trade(trade_params)

    # Assert
    assert result['approved'] is True
    assert 'position_size' in result
    assert result['position_size'] > 0
```

### Coverage Targets

- **Core modules (core/, risk/, strategies/):** 80-90%
- **Financial modules:** 70-80%
- **GUI modules:** 50-60% (harder to test)
- **Utilities:** 90%+

**Check coverage:**
```bash
pytest --cov=nexlify --cov-report=term --cov-report=html
# Open htmlcov/index.html to view detailed report
```

### Standalone Test Suite

`test_training_pipeline.py` - Comprehensive 23-test suite:
```bash
# Quick mode (essential tests only)
python test_training_pipeline.py --quick

# Full suite
python test_training_pipeline.py

# With coverage
python test_training_pipeline.py --coverage
```

Tests include:
- Import validation
- Data fetching
- Training pipeline
- Model save/load
- Configuration validation

---

## Coding Conventions

### Import Organization

Follow this order (use `isort` to automate):

```python
# 1. Standard library
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

# 2. Third-party libraries
import numpy as np
import pandas as pd
import torch

# 3. Local imports (relative)
from nexlify.utils.error_handler import get_error_handler
from nexlify.risk.nexlify_risk_manager import RiskManager
```

### Type Hints

**Always use type hints** for function signatures:

```python
def validate_trade(
    self,
    symbol: str,
    amount: float,
    price: float,
    side: str = 'buy'
) -> Dict[str, Any]:
    """
    Validate trade parameters and calculate position sizing

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        amount: Trade amount in base currency
        price: Current price
        side: Trade side ('buy' or 'sell')

    Returns:
        Dict containing:
            - approved: bool
            - position_size: float
            - stop_loss: float
            - take_profit: float

    Raises:
        ValueError: If parameters are invalid
        TradingError: If validation fails
    """
    pass
```

### Docstring Convention

Use **Google-style docstrings**:

```python
def calculate_kelly_criterion(
    self,
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly criterion for position sizing

    Args:
        win_rate: Historical win rate (0.0-1.0)
        avg_win: Average winning trade size
        avg_loss: Average losing trade size

    Returns:
        Optimal position size as fraction of capital (0.0-1.0)

    Example:
        >>> kelly = risk_manager.calculate_kelly_criterion(0.6, 100, 50)
        >>> print(f"Kelly criterion: {kelly:.2%}")
        Kelly criterion: 20.00%
    """
    pass
```

### Error Handling Pattern

**Use centralized error handler:**

```python
from nexlify.utils.error_handler import get_error_handler

class TradingComponent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_handler = get_error_handler()

    async def risky_operation(self):
        try:
            result = await self.execute_trade()
            return result
        except Exception as e:
            self.error_handler.log_error(
                e,
                context={'operation': 'execute_trade'}
            )
            raise
```

**Or use decorator:**

```python
@handle_errors
async def execute_trade(self, params):
    """Automatically wrapped with error handling"""
    # Implementation
    pass
```

### Configuration Access Pattern

```python
class Component:
    def __init__(self, config: Dict[str, Any]):
        # Use .get() with sensible defaults
        self.enabled = config.get('enabled', True)
        self.threshold = config.get('threshold', 0.05)
        self.max_retries = config.get('max_retries', 3)

        # Validate required fields
        if 'exchange' not in config:
            raise ValueError("Exchange configuration required")
```

### Async Patterns

**Async functions for I/O:**
```python
async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
    """Use async for network/database operations"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

**Async testing:**
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result is not None
```

### Naming Conventions

- **Classes:** PascalCase (`RiskManager`, `CircuitBreaker`)
- **Functions/methods:** snake_case (`validate_trade`, `calculate_position_size`)
- **Constants:** UPPER_CASE (`MAX_RETRIES`, `DEFAULT_TIMEOUT`)
- **Private methods:** Leading underscore (`_internal_method`)
- **Module names:** snake_case (`nexlify_risk_manager.py`)

### Backward Compatibility

**Provide aliases for renamed classes:**
```python
# In nexlify/__init__.py
AutoTrader = AutoExecutionEngine  # Backward compatibility
```

**Graceful feature degradation:**
```python
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False
    # Continue with CPU-only mode
```

---

## Configuration Management

### Main Configuration File

**Location:** `config/neural_config.json` (gitignored)
**Example:** `config/neural_config.example.json` (committed)

**CRITICAL: Never commit actual `neural_config.json` containing API keys!**

### Configuration Structure

```json
{
  "exchanges": {
    "binance": {
      "api_key": "YOUR_API_KEY",
      "secret": "YOUR_SECRET",
      "testnet": false
    }
  },
  "risk_management": {
    "max_position_size": 0.1,
    "use_kelly_criterion": true,
    "kelly_fraction": 0.5,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.05
  },
  "circuit_breaker": {
    "enabled": true,
    "failure_threshold": 5,
    "cooldown_period": 300
  },
  "trading": {
    "min_profit_threshold": 0.01,
    "max_position_size": 0.2,
    "confidence_threshold": 0.7
  },
  "environment": {
    "debug": false,
    "log_level": "INFO",
    "api_port": 8000,
    "database_url": "sqlite:///data/trading.db"
  }
}
```

### Configuration Loading Pattern

```python
import json
from pathlib import Path

def load_config() -> Dict[str, Any]:
    """Load configuration from neural_config.json"""
    config_path = Path('config/neural_config.json')

    if not config_path.exists():
        raise FileNotFoundError(
            "Configuration not found. "
            "Copy config/neural_config.example.json to config/neural_config.json"
        )

    with open(config_path) as f:
        return json.load(f)
```

### Environment Variables

For sensitive data, also support environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv('BINANCE_API_KEY', config.get('api_key'))
```

---

## Common Tasks

### Adding a New Module

1. **Create module in appropriate package:**
```bash
# Example: Adding new strategy
touch nexlify/strategies/nexlify_momentum_strategy.py
```

2. **Follow package structure:**
```python
"""
Momentum Trading Strategy
Implements momentum-based trading logic
"""
import logging
from typing import Dict, Any

from nexlify.utils.error_handler import get_error_handler

logger = logging.getLogger(__name__)

class MomentumStrategy:
    """Momentum-based trading strategy"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_handler = get_error_handler()
        logger.info("Momentum strategy initialized")

    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on momentum"""
        pass
```

3. **Add to package __init__.py:**
```python
# nexlify/strategies/__init__.py
from nexlify.strategies.nexlify_momentum_strategy import MomentumStrategy

__all__ = ['MomentumStrategy', ...]
```

4. **Create tests:**
```bash
touch tests/test_momentum_strategy.py
```

5. **Write unit tests:**
```python
import pytest
from nexlify.strategies.nexlify_momentum_strategy import MomentumStrategy

@pytest.fixture
def test_config():
    return {'threshold': 0.05}

@pytest.mark.unit
def test_momentum_strategy_initialization(test_config):
    strategy = MomentumStrategy(test_config)
    assert strategy.config == test_config
```

### Adding Configuration Options

1. **Update example config:**
```bash
# Edit config/neural_config.example.json
{
  "momentum_strategy": {
    "enabled": true,
    "lookback_period": 14,
    "threshold": 0.05
  }
}
```

2. **Document in code:**
```python
def __init__(self, config: Dict[str, Any]):
    """
    Initialize momentum strategy

    Config options:
        momentum_strategy.enabled (bool): Enable strategy
        momentum_strategy.lookback_period (int): Lookback in days
        momentum_strategy.threshold (float): Signal threshold
    """
    self.enabled = config.get('momentum_strategy', {}).get('enabled', True)
```

### Running Training

```bash
# Quick training (100 episodes)
python scripts/train_rl_agent.py --episodes 100

# Full training pipeline
python train_ultimate_full_pipeline.py

# With auto-retraining
python train_complete_with_auto_retrain.py

# Historical data training
python train_with_historical_data.py --symbol BTC/USDT --days 365
```

### Running Paper Trading

```bash
# Start paper trading
python run_paper_trading.py

# With specific config
python run_paper_trading.py --config config/paper_trading.json

# Monitor logs
tail -f logs/paper_trading.log
```

### Launching GUI

```bash
# Main launcher (recommended)
python scripts/nexlify_launcher.py

# Direct GUI launch
python -m nexlify.gui.cyber_gui

# Windows batch file
start_nexlify.bat
```

### Database Operations

```bash
# Query trades
sqlite3 data/trading.db "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 10"

# Export trades to CSV
sqlite3 data/trading.db -csv -header "SELECT * FROM trades" > trades_export.csv

# Check database schema
sqlite3 data/trading.db ".schema"
```

### Checking Available Pairs

```bash
# Check available trading pairs on exchange
python check_available_pairs.py --exchange binance

# Check specific base currency
python check_available_pairs.py --exchange kraken --base BTC
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ImportError: cannot import name 'RiskManager'
```
**Solution:** Check if module is in correct package and `__init__.py` exports it.

**2. Configuration Not Found**
```
FileNotFoundError: Configuration not found
```
**Solution:** Copy example config:
```bash
cp config/neural_config.example.json config/neural_config.json
```

**3. Pytest Argparse Conflict**
```
ValueError: conflicting option string: --verbose
```
**Solution:** Fixed in recent commit (qasync==0.24.0 version pinning)

**4. GPU Out of Memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size in config:
```json
{
  "training": {
    "batch_size": 32  // Reduce from 64
  }
}
```

**5. Database Locked**
```
sqlite3.OperationalError: database is locked
```
**Solution:** Close any duplicate running instances.

**6. API Rate Limiting**
```
ccxt.errors.RateLimitExceeded
```
**Solution:** Increase rate limit delay in config:
```json
{
  "exchanges": {
    "binance": {
      "rateLimit": 2000  // ms between requests
    }
  }
}
```

### Debug Mode

Enable debug mode in configuration:
```json
{
  "environment": {
    "debug": true,
    "log_level": "DEBUG"
  }
}
```

### Log Locations

- **Main logs:** `logs/neural_net.log`
- **Error logs:** `logs/errors.log`
- **Trading logs:** `logs/trading.log`
- **Paper trading:** `logs/paper_trading.log`

**View logs:**
```bash
# Real-time monitoring
tail -f logs/neural_net.log

# Search for errors
grep ERROR logs/neural_net.log

# Last 100 lines
tail -n 100 logs/neural_net.log
```

### Performance Issues

**High CPU usage:**
- Reduce scan frequency in settings
- Decrease number of monitored pairs
- Disable unnecessary features

**High memory usage:**
- Clear cache: Delete `cache/` directory
- Reduce replay buffer size
- Limit historical data lookback

**Slow GUI:**
- Disable cyberpunk effects
- Reduce chart update frequency
- Use performance mode

---

## Best Practices for AI Assistants

### When Making Changes

1. **Always read existing code first** before making modifications
2. **Run tests** after changes: `python test_training_pipeline.py --quick`
3. **Check imports** are correct and follow conventions
4. **Update docstrings** if function signatures change
5. **Maintain backward compatibility** when refactoring
6. **Never commit sensitive data** (API keys, config, logs)

### Code Review Checklist

- [ ] Type hints added for new functions
- [ ] Docstrings follow Google style
- [ ] Error handling implemented
- [ ] Tests written for new features
- [ ] No hardcoded credentials
- [ ] Imports organized correctly
- [ ] Configuration options documented
- [ ] Backward compatibility maintained

### When Analyzing Code

1. **Check layer dependencies:** Lower layers shouldn't import from higher layers
2. **Verify error handling:** Should use centralized error handler
3. **Review async usage:** I/O operations should be async
4. **Check test coverage:** Core modules need >80% coverage
5. **Validate config usage:** Should have sensible defaults

### When Adding Features

1. **Start with tests:** Write test cases first (TDD)
2. **Follow existing patterns:** Match architectural style
3. **Update documentation:** Add to relevant .md files
4. **Consider configuration:** Make feature configurable
5. **Handle errors gracefully:** Don't crash on failures

### Git Workflow for AI Assistants

```bash
# Always work on feature branches
git checkout -b claude/feature-name-$(date +%s)

# Make changes, run tests
python test_training_pipeline.py --coverage

# Commit with conventional commits
git add .
git commit -m "feat: Add momentum strategy module

- Implements momentum-based signal generation
- Adds configuration options
- Includes unit tests with 85% coverage"

# Push to feature branch
git push -u origin claude/feature-name-sessionid
```

---

## Key Files Reference

### Critical Files (Never Delete)

- `nexlify/utils/error_handler.py` - Used throughout codebase
- `nexlify/core/arasaka_neural_net.py` - Main trading engine
- `config/neural_config.example.json` - Configuration template
- `requirements.txt` - Dependency definitions

### Entry Points

- `scripts/nexlify_launcher.py` - Main system launcher
- `nexlify/gui/cyber_gui.py` - GUI entry point
- `run_paper_trading.py` - Paper trading entry
- `train_ultimate_full_pipeline.py` - Training pipeline

### Configuration Files

- `config/neural_config.json` - Main config (gitignored)
- `pytest.ini` - Test configuration
- `setup.py` - Package configuration
- `.gitignore` - Git ignore rules

### Documentation

- `README.md` - User-facing documentation
- `CLAUDE.md` - This file (AI assistant guide)
- `docs/QUICK_REFERENCE.md` - Quick command reference
- `docs/IMPLEMENTATION_GUIDE.md` - Detailed implementation
- `docs/TRAINING_GUIDE.md` - Training instructions

---

## Version Information

- **Current Version:** 2.0.7.7
- **Python Required:** 3.9+ (tested on 3.10, 3.11)
- **Last Updated:** 2025-11-14
- **Maintainer:** Nexlify Development Team

---

## Additional Resources

### Documentation Files

The `docs/` directory contains 24 comprehensive guides:
- Implementation guides
- Training documentation
- Feature guides
- Best practices
- Gap analysis reports

### Example Scripts

The `examples/` directory contains integration examples for:
- Exchange connectivity
- Risk management
- Custom strategies
- API usage

### Training Scripts

19 root-level training scripts covering:
- Complete training pipelines
- Historical data training
- Multi-strategy training
- Auto-retraining systems

---

## Summary

This guide provides comprehensive information for AI assistants working with Nexlify. Key takeaways:

1. **Architecture:** Layered design with clear separation of concerns
2. **Testing:** Comprehensive pytest suite with 70-90% coverage target
3. **Conventions:** Type hints, Google docstrings, conventional commits
4. **Configuration:** JSON-based, never commit sensitive data
5. **Error Handling:** Centralized error handler pattern
6. **Async:** Use async for I/O operations
7. **Modularity:** Each component can function independently

**Always prioritize:**
- Code quality and test coverage
- Security (never commit credentials)
- Backward compatibility
- Clear documentation
- Following established patterns

---

**For questions or issues:** Check existing documentation in `docs/` or review similar modules for patterns.

**End of CLAUDE.md**
