# Import Update Quick Reference Guide

This guide shows the exact import changes needed for files with complex dependencies.

---

## Files Requiring Complex Import Updates

### 1. cyber_gui.py (10 internal imports) ⭐ MOST COMPLEX

**Current imports:**
```python
from error_handler import get_error_handler, handle_errors
from nexlify_advanced_security import AdvancedSecurity
from nexlify_ai_companion import AICompanion
from nexlify_audit_trail import AuditTrail
from nexlify_cyberpunk_effects import CyberpunkEffects
from nexlify_gui_integration import GUIIntegration
from nexlify_multi_strategy import MultiStrategy
from nexlify_neural_net import NexlifyNeuralNet
from nexlify_predictive_features import PredictiveFeatures
from utils_module import some_utility_function
```

**New imports:**
```python
from nexlify.utils.error_handler import get_error_handler, handle_errors
from nexlify.security.nexlify_advanced_security import AdvancedSecurity
from nexlify.analytics.nexlify_ai_companion import AICompanion
from nexlify.security.nexlify_audit_trail import AuditTrail
from nexlify.gui.nexlify_cyberpunk_effects import CyberpunkEffects
from nexlify.gui.nexlify_gui_integration import GUIIntegration
from nexlify.strategies.nexlify_multi_strategy import MultiStrategy
from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
from nexlify.strategies.nexlify_predictive_features import PredictiveFeatures
from nexlify.utils.utils_module import some_utility_function
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.utils import get_error_handler, handle_errors
from nexlify.security import AdvancedSecurity, AuditTrail
from nexlify.analytics import AICompanion
from nexlify.gui import CyberpunkEffects, GUIIntegration
from nexlify.strategies import MultiStrategy, PredictiveFeatures
from nexlify.core import NexlifyNeuralNet
from nexlify.utils.utils_module import some_utility_function
```

---

### 2. nexlify_trading_integration.py (6 internal imports)

**Current imports:**
```python
from nexlify_security_suite import SecuritySuite
from nexlify_tax_reporter import TaxReporter
from nexlify_profit_manager import ProfitManager, WithdrawalDestination
from nexlify_defi_integration import DeFiIntegration
from nexlify_emergency_kill_switch import KillSwitchTrigger
from nexlify_flash_crash_protection import CrashSeverity
```

**New imports:**
```python
from nexlify.security.nexlify_security_suite import SecuritySuite
from nexlify.financial.nexlify_tax_reporter import TaxReporter
from nexlify.financial.nexlify_profit_manager import ProfitManager, WithdrawalDestination
from nexlify.financial.nexlify_defi_integration import DeFiIntegration
from nexlify.risk.nexlify_emergency_kill_switch import KillSwitchTrigger
from nexlify.risk.nexlify_flash_crash_protection import CrashSeverity
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.security import SecuritySuite
from nexlify.financial import TaxReporter, ProfitManager, WithdrawalDestination, DeFiIntegration
from nexlify.risk import KillSwitchTrigger, CrashSeverity
```

---

### 3. nexlify_gui_integration.py (5 internal imports)

**Current imports:**
```python
from nexlify_defi_integration import DeFiIntegration
from nexlify_emergency_kill_switch import KillSwitchTrigger
from nexlify_profit_manager import ProfitManager
from nexlify_security_suite import SecuritySuite
from nexlify_tax_reporter import TaxReporter
```

**New imports:**
```python
from nexlify.financial.nexlify_defi_integration import DeFiIntegration
from nexlify.risk.nexlify_emergency_kill_switch import KillSwitchTrigger
from nexlify.financial.nexlify_profit_manager import ProfitManager
from nexlify.security.nexlify_security_suite import SecuritySuite
from nexlify.financial.nexlify_tax_reporter import TaxReporter
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.financial import DeFiIntegration, ProfitManager, TaxReporter
from nexlify.risk import KillSwitchTrigger
from nexlify.security import SecuritySuite
```

---

### 4. nexlify_security_suite.py (4 internal imports)

**Current imports:**
```python
from error_handler import get_error_handler, handle_errors
from nexlify_emergency_kill_switch import EmergencyKillSwitch
from nexlify_flash_crash_protection import FlashCrashProtection
from nexlify_integrity_monitor import IntegrityMonitor
from nexlify_pin_manager import PINManager
```

**New imports:**
```python
from nexlify.utils.error_handler import get_error_handler, handle_errors
from nexlify.risk.nexlify_emergency_kill_switch import EmergencyKillSwitch
from nexlify.risk.nexlify_flash_crash_protection import FlashCrashProtection
from nexlify.security.nexlify_integrity_monitor import IntegrityMonitor
from nexlify.security.nexlify_pin_manager import PINManager
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.utils import get_error_handler, handle_errors
from nexlify.risk import EmergencyKillSwitch, FlashCrashProtection
from nexlify.security import IntegrityMonitor, PINManager
```

---

### 5. arasaka_neural_net.py (3 internal imports)

**Current imports:**
```python
from nexlify_circuit_breaker import CircuitBreaker
from nexlify_performance_tracker import PerformanceTracker
from nexlify_risk_manager import RiskManager
```

**New imports:**
```python
from nexlify.risk.nexlify_circuit_breaker import CircuitBreaker
from nexlify.analytics.nexlify_performance_tracker import PerformanceTracker
from nexlify.risk.nexlify_risk_manager import RiskManager
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.risk import CircuitBreaker, RiskManager
from nexlify.analytics import PerformanceTracker
```

---

### 6. nexlify_neural_net.py (2 internal imports)

**Current imports:**
```python
from error_handler import get_error_handler
from arasaka_neural_net import ArasakaNeuralNet
```

**New imports:**
```python
from nexlify.utils.error_handler import get_error_handler
from nexlify.core.arasaka_neural_net import ArasakaNeuralNet
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.utils import get_error_handler
from nexlify.core import ArasakaNeuralNet
```

---

### 7. train_rl_agent.py (2 internal imports)

**Current imports:**
```python
from error_handler import get_error_handler
from nexlify_rl_agent import RLAgent
```

**New imports:**
```python
from nexlify.utils.error_handler import get_error_handler
from nexlify.strategies.nexlify_rl_agent import RLAgent
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.utils import get_error_handler
from nexlify.strategies import RLAgent
```

---

### 8. example_integration.py (3 internal imports)

**Current imports:**
```python
from nexlify_circuit_breaker import CircuitBreaker
from nexlify_performance_tracker import PerformanceTracker
from nexlify_risk_manager import RiskManager
```

**New imports:**
```python
from nexlify.risk.nexlify_circuit_breaker import CircuitBreaker
from nexlify.analytics.nexlify_performance_tracker import PerformanceTracker
from nexlify.risk.nexlify_risk_manager import RiskManager
```

**Or use package-level imports (RECOMMENDED):**
```python
from nexlify.risk import CircuitBreaker, RiskManager
from nexlify.analytics import PerformanceTracker
```

---

## Standard Pattern for All Other Files (27 files)

Most files only import error_handler, so the update is simple:

**Current:**
```python
from error_handler import get_error_handler, handle_errors
```

**New:**
```python
from nexlify.utils.error_handler import get_error_handler, handle_errors
```

**Or (RECOMMENDED):**
```python
from nexlify.utils import get_error_handler, handle_errors
```

**Files with this pattern:**
- nexlify_advanced_analytics.py
- nexlify_advanced_security.py
- nexlify_ai_companion.py
- nexlify_audit_trail.py
- nexlify_auto_trader.py
- nexlify_backtester.py
- nexlify_circuit_breaker.py
- nexlify_defi_integration.py
- nexlify_emergency_kill_switch.py
- nexlify_flash_crash_protection.py
- nexlify_hardware_detection.py
- nexlify_integrity_monitor.py
- nexlify_launcher.py
- nexlify_multi_strategy.py
- nexlify_multi_timeframe.py
- nexlify_paper_trading.py
- nexlify_performance_tracker.py
- nexlify_pin_manager.py
- nexlify_portfolio_rebalancer.py
- nexlify_predictive_features.py
- nexlify_profit_manager.py
- nexlify_risk_manager.py
- nexlify_rl_agent.py
- nexlify_tax_reporter.py
- nexlify_telegram_bot.py
- nexlify_websocket_feeds.py
- utils_module.py

---

## Files Requiring NO Import Updates

These files have no internal imports:
- backtest_phase1_phase2_integration.py
- setup_nexlify.py
- nexlify_cyberpunk_effects.py

---

## Testing Imports After Migration

### Test package-level imports:
```python
# Test utils
from nexlify.utils import get_error_handler, handle_errors
print("✓ Utils imports work")

# Test core
from nexlify.core import ArasakaNeuralNet, NexlifyNeuralNet, AutoTrader
print("✓ Core imports work")

# Test risk
from nexlify.risk import RiskManager, CircuitBreaker, FlashCrashProtection
print("✓ Risk imports work")

# Test security
from nexlify.security import SecuritySuite, AdvancedSecurity
print("✓ Security imports work")

# Test financial
from nexlify.financial import ProfitManager, TaxReporter, DeFiIntegration
print("✓ Financial imports work")

# Test strategies
from nexlify.strategies import MultiStrategy, RLAgent
print("✓ Strategy imports work")

# Test analytics
from nexlify.analytics import PerformanceTracker, AdvancedAnalytics
print("✓ Analytics imports work")

# Test GUI
from nexlify.gui import CyberGUI, GUIIntegration
print("✓ GUI imports work")

# Test backtesting
from nexlify.backtesting import Backtester, PaperTrading
print("✓ Backtesting imports work")

# Test integrations
from nexlify.integrations import WebSocketFeeds, TelegramBot
print("✓ Integration imports work")

print("\n✅ ALL IMPORTS SUCCESSFUL!")
```

### Test direct imports:
```python
# Test direct module imports
from nexlify.core.nexlify_neural_net import NexlifyNeuralNet
from nexlify.security.nexlify_security_suite import SecuritySuite
from nexlify.risk.nexlify_risk_manager import RiskManager

print("✓ Direct imports work")
```

---

## sed Commands for Batch Updates

If you prefer to use sed for quick updates, here are the commands:

### Update error_handler imports:
```bash
find nexlify/ scripts/ -name "*.py" -type f -exec sed -i 's/from error_handler import/from nexlify.utils.error_handler import/g' {} +
```

### Update utils_module imports:
```bash
find nexlify/ scripts/ -name "*.py" -type f -exec sed -i 's/from utils_module import/from nexlify.utils.utils_module import/g' {} +
```

### Update all nexlify_* imports (use with caution):
```bash
# This requires careful review - shown as example only
sed -i 's/from nexlify_risk_manager import/from nexlify.risk.nexlify_risk_manager import/g' nexlify/**/*.py
sed -i 's/from nexlify_circuit_breaker import/from nexlify.risk.nexlify_circuit_breaker import/g' nexlify/**/*.py
# ... (repeat for each module)
```

**⚠️ WARNING**: Always review sed changes before committing. The migration script is safer as it handles all updates automatically.

---

## Import Style Recommendations

### ✅ RECOMMENDED: Package-level imports
```python
from nexlify.core import AutoTrader, NexlifyNeuralNet
from nexlify.risk import RiskManager, CircuitBreaker
from nexlify.security import SecuritySuite
```

**Benefits:**
- Cleaner, more readable
- Easier to maintain
- Better IDE autocomplete
- Follows Python best practices

### ⚠️ ACCEPTABLE: Direct module imports
```python
from nexlify.core.nexlify_auto_trader import AutoTrader
from nexlify.risk.nexlify_risk_manager import RiskManager
```

**Use when:**
- Name conflicts exist
- You need to be very explicit
- Importing from a module not exposed in __init__.py

### ❌ AVOID: Star imports
```python
from nexlify.core import *  # Don't do this
```

**Why:**
- Makes it unclear what's being imported
- Can cause name conflicts
- Harder to track dependencies
- Not PEP 8 compliant

---

## Common Pitfalls and Solutions

### Pitfall 1: Forgot to update path in sys.path
**Problem:**
```python
sys.path.insert(0, '/path/to/Nexlify')
from nexlify_auto_trader import AutoTrader  # Old import
```

**Solution:**
```python
sys.path.insert(0, '/path/to/Nexlify')
from nexlify.core import AutoTrader  # New import
```

### Pitfall 2: Circular imports in __init__.py
**Problem:**
```python
# In nexlify/core/__init__.py
from nexlify.core.nexlify_neural_net import NexlifyNeuralNet  # May cause issues
```

**Solution:**
Use lazy imports or restructure to avoid cycles. Our migration has no circular dependencies, so this shouldn't be an issue.

### Pitfall 3: Missing __init__.py files
**Problem:**
```
ImportError: No module named 'nexlify.core'
```

**Solution:**
Ensure all directories have __init__.py files. The migration script creates these automatically.

### Pitfall 4: Relative imports confusion
**Problem:**
```python
# In nexlify/core/arasaka_neural_net.py
from ..risk.nexlify_risk_manager import RiskManager  # Relative import
```

**Solution:**
Use absolute imports (already in migration script):
```python
from nexlify.risk.nexlify_risk_manager import RiskManager
```

Or use package-level:
```python
from nexlify.risk import RiskManager
```

---

## Verification Checklist

After updating imports:

- [ ] All files compile without syntax errors
- [ ] Import test script runs successfully
- [ ] No ImportError exceptions
- [ ] IDE autocomplete works for new imports
- [ ] Tests pass (if any)
- [ ] Scripts in scripts/ directory work
- [ ] Main application (cyber_gui.py) launches
- [ ] No "module not found" warnings in logs

---

## Quick Reference Table

| Old Import | New Import | Package |
|------------|------------|---------|
| `from error_handler import` | `from nexlify.utils.error_handler import` | utils |
| `from utils_module import` | `from nexlify.utils.utils_module import` | utils |
| `from nexlify_auto_trader import` | `from nexlify.core.nexlify_auto_trader import` | core |
| `from arasaka_neural_net import` | `from nexlify.core.arasaka_neural_net import` | core |
| `from nexlify_neural_net import` | `from nexlify.core.nexlify_neural_net import` | core |
| `from nexlify_trading_integration import` | `from nexlify.core.nexlify_trading_integration import` | core |
| `from nexlify_risk_manager import` | `from nexlify.risk.nexlify_risk_manager import` | risk |
| `from nexlify_circuit_breaker import` | `from nexlify.risk.nexlify_circuit_breaker import` | risk |
| `from nexlify_flash_crash_protection import` | `from nexlify.risk.nexlify_flash_crash_protection import` | risk |
| `from nexlify_emergency_kill_switch import` | `from nexlify.risk.nexlify_emergency_kill_switch import` | risk |
| `from nexlify_security_suite import` | `from nexlify.security.nexlify_security_suite import` | security |
| `from nexlify_advanced_security import` | `from nexlify.security.nexlify_advanced_security import` | security |
| `from nexlify_pin_manager import` | `from nexlify.security.nexlify_pin_manager import` | security |
| `from nexlify_integrity_monitor import` | `from nexlify.security.nexlify_integrity_monitor import` | security |
| `from nexlify_audit_trail import` | `from nexlify.security.nexlify_audit_trail import` | security |
| `from nexlify_profit_manager import` | `from nexlify.financial.nexlify_profit_manager import` | financial |
| `from nexlify_tax_reporter import` | `from nexlify.financial.nexlify_tax_reporter import` | financial |
| `from nexlify_portfolio_rebalancer import` | `from nexlify.financial.nexlify_portfolio_rebalancer import` | financial |
| `from nexlify_defi_integration import` | `from nexlify.financial.nexlify_defi_integration import` | financial |
| `from nexlify_performance_tracker import` | `from nexlify.analytics.nexlify_performance_tracker import` | analytics |
| `from nexlify_advanced_analytics import` | `from nexlify.analytics.nexlify_advanced_analytics import` | analytics |
| `from nexlify_ai_companion import` | `from nexlify.analytics.nexlify_ai_companion import` | analytics |
| `from nexlify_multi_strategy import` | `from nexlify.strategies.nexlify_multi_strategy import` | strategies |
| `from nexlify_multi_timeframe import` | `from nexlify.strategies.nexlify_multi_timeframe import` | strategies |
| `from nexlify_predictive_features import` | `from nexlify.strategies.nexlify_predictive_features import` | strategies |
| `from nexlify_rl_agent import` | `from nexlify.strategies.nexlify_rl_agent import` | strategies |
| `from nexlify_backtester import` | `from nexlify.backtesting.nexlify_backtester import` | backtesting |
| `from nexlify_paper_trading import` | `from nexlify.backtesting.nexlify_paper_trading import` | backtesting |
| `from backtest_phase1_phase2_integration import` | `from nexlify.backtesting.backtest_phase1_phase2_integration import` | backtesting |
| `from nexlify_websocket_feeds import` | `from nexlify.integrations.nexlify_websocket_feeds import` | integrations |
| `from nexlify_telegram_bot import` | `from nexlify.integrations.nexlify_telegram_bot import` | integrations |
| `from cyber_gui import` | `from nexlify.gui.cyber_gui import` | gui |
| `from nexlify_gui_integration import` | `from nexlify.gui.nexlify_gui_integration import` | gui |
| `from nexlify_cyberpunk_effects import` | `from nexlify.gui.nexlify_cyberpunk_effects import` | gui |
| `from nexlify_hardware_detection import` | `from nexlify.gui.nexlify_hardware_detection import` | gui |
