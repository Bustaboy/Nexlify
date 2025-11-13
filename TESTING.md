# Nexlify Testing Guide

## Quick Start

### Run All Tests
```bash
python test_training_pipeline.py
```

### Run with Coverage Report
```bash
python test_training_pipeline.py --coverage
```

### Quick Tests Only (Skip slow network tests)
```bash
python test_training_pipeline.py --quick
```

---

## Pytest with Coverage (Advanced)

### Install pytest and coverage tools
```bash
pip install pytest pytest-cov
```

### Run pytest with coverage
```bash
# Run all tests with coverage
pytest test_training_pipeline.py --cov=. --cov-report=html --cov-report=term

# Run only fast tests
pytest test_training_pipeline.py -m "not slow" --cov=. --cov-report=term

# Run with detailed coverage
pytest test_training_pipeline.py --cov=. --cov-report=html --cov-report=term-missing

# View HTML coverage report (opens in browser)
# Coverage report saved to: htmlcov/index.html
```

### Coverage Report Locations
- **Terminal output**: Shown immediately after tests
- **HTML report**: `htmlcov/index.html`
- **Coverage data**: `.coverage` (binary file)

---

## Test Categories

### Unit Tests
Fast, isolated component tests:
```bash
pytest -m unit
```

### Integration Tests
Tests multiple components together:
```bash
pytest -m integration
```

### Network Tests
Require internet connectivity:
```bash
pytest -m requires_network
```

### Skip Slow Tests
```bash
pytest -m "not slow"
```

---

## Expected Coverage

| Module | Target Coverage |
|--------|----------------|
| `nexlify_advanced_dqn_agent.py` | >80% |
| `nexlify_complete_strategy_env.py` | >75% |
| `nexlify_historical_data_fetcher.py` | >70% |
| `train_ultimate_full_pipeline.py` | >60% |

---

## Continuous Integration

Add to `.github/workflows/test.yml`:
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt pytest pytest-cov
      - run: pytest --cov=. --cov-report=xml
      - uses: codecov/codecov-action@v2
```

---

## Test Before Training

**ALWAYS run tests before starting a long training run:**

```bash
# 1. Quick validation (30 seconds)
python test_training_pipeline.py --quick

# 2. Full tests with coverage (2-3 minutes)
python test_training_pipeline.py --coverage

# 3. If all tests pass, start training
python train_ultimate_full_pipeline.py --pairs BTC/USDT ETH/USDT --automated
```

---

## Troubleshooting

### Tests Fail on Import
```bash
# Fix: Install missing dependencies
pip install -r requirements.txt
```

### Network Tests Fail
```bash
# Skip network tests if offline
python test_training_pipeline.py --quick
```

### Coverage Tool Not Found
```bash
# Install pytest-cov
pip install pytest-cov
```

### Low Coverage Warning
Low coverage doesn't necessarily mean bad code - some modules may be difficult to test (e.g., GUI components, live trading). Focus on covering:
- Core training logic
- Data processing
- Risk management
- Model save/load

---

## Writing New Tests

Add tests to `test_training_pipeline.py`:

```python
def test_my_new_feature():
    """Test description"""
    # Arrange
    agent = create_test_agent()

    # Act
    result = agent.do_something()

    # Assert
    assert result is not None
    test_status("My new feature", result is not None)
```

Mark slow tests:
```python
@pytest.mark.slow
def test_fetch_5_years_of_data():
    # ...
```

---

## Coverage Goals

- **Critical path code**: 90%+ coverage
- **Core algorithms**: 80%+ coverage
- **Utility functions**: 70%+ coverage
- **UI/visualization**: 50%+ coverage acceptable

Focus on testing:
1. Training loop logic
2. Risk management calculations
3. Data fetching and validation
4. Model save/load operations
5. Environment step logic
