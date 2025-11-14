# Nexlify Testing Guide

## Overview

Nexlify has a comprehensive test suite covering critical trading, risk management, security, and financial modules.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=nexlify --cov-report=html

# Run specific tests
pytest tests/test_risk_manager.py -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

## Test Coverage (~60-65%)

| Module | Coverage | Test File |
|--------|----------|-----------|
| risk/ | ~85% | test_risk_manager.py, test_circuit_breaker.py, test_emergency_systems.py |
| core/ | ~70% | test_auto_trader.py |
| strategies/ | ~65% | test_rl_agent.py |
| security/ | ~70% | test_security.py |
| financial/ | ~70% | test_financial.py |
| utils/ | ~80% | test_utils.py |

## CI/CD Workflow

### Quick Tests (Every Push) - 5-10 min
- Unit tests only
- Python 3.11
- Fast feedback

### Full Tests (Pull Requests) - 15 min
- All non-slow tests
- Python 3.10, 3.11
- Coverage reports

### Integration Tests (PR to Main) - 30 min
- Integration and slow tests
- Full validation

## Writing Tests

```python
import pytest

@pytest.fixture
def config():
    return {'enabled': True}

class TestYourModule:
    def test_initialization(self, config):
        assert config['enabled'] is True
```

## Best Practices

1. ✅ Write tests before implementation (TDD)
2. ✅ Test edge cases (zero, negative, None)
3. ✅ Mock external dependencies
4. ✅ Use descriptive test names
5. ✅ Aim for 70%+ coverage on new code

## Running Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock

# Run quick tests (what CI runs)
pytest tests/ -m "not slow and not integration" -v

# Run with coverage
pytest tests/ --cov=nexlify --cov-report=html
open htmlcov/index.html
```

---

**Total Tests**: 11 files, 4,508 lines
**Coverage Target**: 60-70%
