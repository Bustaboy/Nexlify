# Nexlify Dependency Graph

## Visual Dependency Structure

This document shows the dependency relationships between modules.

---

## Layer 0: Foundation (No Internal Dependencies)

```
┌─────────────────────────────────────────────────────────────┐
│                    FOUNDATION LAYER                         │
│  (These modules have NO internal dependencies or only      │
│   import error_handler)                                     │
└─────────────────────────────────────────────────────────────┘

error_handler.py          ← NO DEPENDENCIES (Base module)
    ↑
    │ (imported by almost everything)
    │
    ├── utils_module.py
    │
    ├── nexlify_cyberpunk_effects.py  ← NO DEPENDENCIES
    │
    └── [27 other modules only depend on error_handler]
        ├── nexlify_risk_manager.py
        ├── nexlify_circuit_breaker.py
        ├── nexlify_flash_crash_protection.py
        ├── nexlify_emergency_kill_switch.py
        ├── nexlify_performance_tracker.py
        ├── nexlify_advanced_analytics.py
        ├── nexlify_ai_companion.py
        ├── nexlify_multi_strategy.py
        ├── nexlify_multi_timeframe.py
        ├── nexlify_predictive_features.py
        ├── nexlify_rl_agent.py
        ├── nexlify_advanced_security.py
        ├── nexlify_pin_manager.py
        ├── nexlify_integrity_monitor.py
        ├── nexlify_audit_trail.py
        ├── nexlify_profit_manager.py
        ├── nexlify_tax_reporter.py
        ├── nexlify_portfolio_rebalancer.py
        ├── nexlify_defi_integration.py
        ├── nexlify_websocket_feeds.py
        ├── nexlify_telegram_bot.py
        ├── nexlify_backtester.py
        ├── nexlify_paper_trading.py
        ├── nexlify_hardware_detection.py
        ├── nexlify_auto_trader.py
        ├── nexlify_launcher.py
        └── backtest_phase1_phase2_integration.py  ← NO DEPENDENCIES
```

---

## Layer 1: Integration Layer (Depend on Layer 0)

```
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER 1                      │
│  (Modules that import other internal modules)              │
└─────────────────────────────────────────────────────────────┘

arasaka_neural_net.py
    │
    ├──→ nexlify_circuit_breaker.py      (risk/)
    ├──→ nexlify_performance_tracker.py  (analytics/)
    └──→ nexlify_risk_manager.py         (risk/)


nexlify_security_suite.py
    │
    ├──→ nexlify_emergency_kill_switch.py    (risk/)
    ├──→ nexlify_flash_crash_protection.py   (risk/)
    ├──→ nexlify_integrity_monitor.py        (security/)
    └──→ nexlify_pin_manager.py              (security/)


train_rl_agent.py
    │
    └──→ nexlify_rl_agent.py  (strategies/)


example_integration.py
    │
    ├──→ nexlify_circuit_breaker.py      (risk/)
    ├──→ nexlify_performance_tracker.py  (analytics/)
    └──→ nexlify_risk_manager.py         (risk/)
```

---

## Layer 2: Advanced Integration (Depend on Layers 0 & 1)

```
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER 2                      │
│  (Modules that depend on Layer 1 modules)                  │
└─────────────────────────────────────────────────────────────┘

nexlify_neural_net.py
    │
    └──→ arasaka_neural_net.py  (core/ - Layer 1)


nexlify_gui_integration.py
    │
    ├──→ nexlify_defi_integration.py       (financial/ - Layer 0)
    ├──→ nexlify_emergency_kill_switch.py  (risk/ - Layer 0)
    ├──→ nexlify_profit_manager.py         (financial/ - Layer 0)
    ├──→ nexlify_security_suite.py         (security/ - Layer 1) ⭐
    └──→ nexlify_tax_reporter.py           (financial/ - Layer 0)


nexlify_trading_integration.py
    │
    ├──→ nexlify_defi_integration.py         (financial/ - Layer 0)
    ├──→ nexlify_emergency_kill_switch.py    (risk/ - Layer 0)
    ├──→ nexlify_flash_crash_protection.py   (risk/ - Layer 0)
    ├──→ nexlify_profit_manager.py           (financial/ - Layer 0)
    ├──→ nexlify_security_suite.py           (security/ - Layer 1) ⭐
    └──→ nexlify_tax_reporter.py             (financial/ - Layer 0)
```

---

## Layer 3: Top-Level UI (Depend on All Layers)

```
┌─────────────────────────────────────────────────────────────┐
│                    TOP-LEVEL LAYER 3                        │
│  (Main application - depends on all previous layers)       │
└─────────────────────────────────────────────────────────────┘

cyber_gui.py  (Main application GUI)
    │
    ├──→ error_handler                    (utils/ - Layer 0)
    ├──→ utils_module                     (utils/ - Layer 0)
    ├──→ nexlify_advanced_security        (security/ - Layer 0)
    ├──→ nexlify_ai_companion             (analytics/ - Layer 0)
    ├──→ nexlify_audit_trail              (security/ - Layer 0)
    ├──→ nexlify_cyberpunk_effects        (gui/ - Layer 0)
    ├──→ nexlify_multi_strategy           (strategies/ - Layer 0)
    ├──→ nexlify_predictive_features      (strategies/ - Layer 0)
    ├──→ nexlify_neural_net               (core/ - Layer 2) ⭐⭐
    └──→ nexlify_gui_integration          (gui/ - Layer 2) ⭐⭐
```

---

## Dependency Chains

### Chain 1: Neural Network Stack
```
error_handler
    ↓
nexlify_risk_manager, nexlify_circuit_breaker, nexlify_performance_tracker
    ↓
arasaka_neural_net
    ↓
nexlify_neural_net
    ↓
cyber_gui
```

### Chain 2: Security Stack
```
error_handler
    ↓
nexlify_pin_manager, nexlify_integrity_monitor,
nexlify_emergency_kill_switch, nexlify_flash_crash_protection
    ↓
nexlify_security_suite
    ↓
nexlify_trading_integration, nexlify_gui_integration
    ↓
cyber_gui
```

### Chain 3: Financial Stack
```
error_handler
    ↓
nexlify_profit_manager, nexlify_tax_reporter, nexlify_defi_integration
    ↓
nexlify_trading_integration, nexlify_gui_integration
    ↓
cyber_gui
```

---

## Package Dependency Map

```
nexlify/
│
├── utils/               ← FOUNDATION (imported by everything)
│   ├── error_handler.py
│   └── utils_module.py
│
├── risk/                ← LAYER 0
│   ├── nexlify_risk_manager.py
│   ├── nexlify_circuit_breaker.py
│   ├── nexlify_flash_crash_protection.py
│   └── nexlify_emergency_kill_switch.py
│
├── analytics/           ← LAYER 0
│   ├── nexlify_performance_tracker.py
│   ├── nexlify_advanced_analytics.py
│   └── nexlify_ai_companion.py
│
├── strategies/          ← LAYER 0
│   ├── nexlify_multi_strategy.py
│   ├── nexlify_multi_timeframe.py
│   ├── nexlify_predictive_features.py
│   └── nexlify_rl_agent.py
│
├── security/            ← LAYER 0 + LAYER 1
│   ├── nexlify_advanced_security.py        (Layer 0)
│   ├── nexlify_pin_manager.py              (Layer 0)
│   ├── nexlify_integrity_monitor.py        (Layer 0)
│   ├── nexlify_audit_trail.py              (Layer 0)
│   └── nexlify_security_suite.py           (Layer 1) ⭐
│
├── financial/           ← LAYER 0
│   ├── nexlify_profit_manager.py
│   ├── nexlify_tax_reporter.py
│   ├── nexlify_portfolio_rebalancer.py
│   └── nexlify_defi_integration.py
│
├── integrations/        ← LAYER 0
│   ├── nexlify_websocket_feeds.py
│   └── nexlify_telegram_bot.py
│
├── backtesting/         ← LAYER 0
│   ├── nexlify_backtester.py
│   ├── nexlify_paper_trading.py
│   └── backtest_phase1_phase2_integration.py
│
├── gui/                 ← LAYER 0, 2, 3
│   ├── nexlify_cyberpunk_effects.py        (Layer 0)
│   ├── nexlify_hardware_detection.py       (Layer 0)
│   ├── nexlify_gui_integration.py          (Layer 2) ⭐⭐
│   └── cyber_gui.py                         (Layer 3) ⭐⭐⭐
│
└── core/                ← LAYER 0, 1, 2
    ├── nexlify_auto_trader.py              (Layer 0)
    ├── arasaka_neural_net.py               (Layer 1) ⭐
    ├── nexlify_neural_net.py               (Layer 2) ⭐⭐
    └── nexlify_trading_integration.py      (Layer 2) ⭐⭐
```

---

## Cross-Package Dependencies

### From core/ imports:
```
core/arasaka_neural_net.py
    → risk/nexlify_circuit_breaker.py
    → analytics/nexlify_performance_tracker.py
    → risk/nexlify_risk_manager.py

core/nexlify_neural_net.py
    → core/arasaka_neural_net.py

core/nexlify_trading_integration.py
    → financial/nexlify_defi_integration.py
    → risk/nexlify_emergency_kill_switch.py
    → risk/nexlify_flash_crash_protection.py
    → financial/nexlify_profit_manager.py
    → security/nexlify_security_suite.py
    → financial/nexlify_tax_reporter.py
```

### From security/ imports:
```
security/nexlify_security_suite.py
    → risk/nexlify_emergency_kill_switch.py
    → risk/nexlify_flash_crash_protection.py
    → security/nexlify_integrity_monitor.py
    → security/nexlify_pin_manager.py
```

### From gui/ imports:
```
gui/nexlify_gui_integration.py
    → financial/nexlify_defi_integration.py
    → risk/nexlify_emergency_kill_switch.py
    → financial/nexlify_profit_manager.py
    → security/nexlify_security_suite.py
    → financial/nexlify_tax_reporter.py

gui/cyber_gui.py
    → utils/error_handler.py
    → utils/utils_module.py
    → security/nexlify_advanced_security.py
    → analytics/nexlify_ai_companion.py
    → security/nexlify_audit_trail.py
    → gui/nexlify_cyberpunk_effects.py
    → strategies/nexlify_multi_strategy.py
    → strategies/nexlify_predictive_features.py
    → core/nexlify_neural_net.py
    → gui/nexlify_gui_integration.py
```

---

## Critical Dependencies

### Most Imported Module:
**error_handler.py** - Imported by 29+ modules

### Hub Modules (import many other modules):
1. **cyber_gui.py** - Imports 10 internal modules
2. **nexlify_trading_integration.py** - Imports 6 internal modules
3. **nexlify_gui_integration.py** - Imports 5 internal modules
4. **nexlify_security_suite.py** - Imports 4 internal modules
5. **arasaka_neural_net.py** - Imports 3 internal modules

### Most Depended Upon (excluding error_handler):
1. **nexlify_security_suite.py** - Used by 2 modules
2. **nexlify_emergency_kill_switch.py** - Used by 2 modules
3. **arasaka_neural_net.py** - Used by 1 module

---

## Import Update Requirements

### Files requiring NO import updates:
- backtest_phase1_phase2_integration.py
- setup_nexlify.py
- nexlify_cyberpunk_effects.py

### Files requiring SIMPLE updates (only error_handler):
27 files that only import error_handler - straightforward update:
```python
from error_handler import ...
→
from nexlify.utils.error_handler import ...
```

### Files requiring COMPLEX updates (multiple imports):
- arasaka_neural_net.py (3 internal imports)
- nexlify_security_suite.py (4 internal imports)
- nexlify_gui_integration.py (5 internal imports)
- nexlify_trading_integration.py (6 internal imports)
- cyber_gui.py (10 internal imports) ⭐ MOST COMPLEX

---

## Circular Dependency Check

**✅ NO CIRCULAR DEPENDENCIES DETECTED**

The dependency graph forms a **Directed Acyclic Graph (DAG)**, which means:
- Safe to reorganize
- Clear dependency hierarchy
- No risk of import deadlocks
- Can build dependency layers naturally

---

## Migration Safety Notes

1. **error_handler.py MUST be moved first** - Everything depends on it
2. **utils_module.py should be moved second** - Used by GUI components
3. **Layer 0 modules can be moved in any order** - No interdependencies
4. **Layer 1+ must be moved after their dependencies**
5. **cyber_gui.py should be moved last** - Depends on the most modules

This is the exact order implemented in the migration script.
