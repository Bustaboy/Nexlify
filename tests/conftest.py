#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Nexlify tests
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_config():
    """Standard test configuration"""
    return {
        'risk_management': {
            'enabled': True,
            'max_position_size': 0.05,
            'max_daily_loss': 0.05,
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.05,
            'use_kelly_criterion': True,
            'kelly_fraction': 0.5,
            'min_kelly_confidence': 0.6,
            'max_concurrent_trades': 3
        },
        'circuit_breaker': {
            'enabled': True,
            'failure_threshold': 3,
            'timeout_seconds': 2,
            'half_open_max_calls': 1
        },
        'performance_tracking': {
            'enabled': True,
            'database_path': 'data/trading.db',
            'calculate_sharpe_ratio': True,
            'risk_free_rate': 0.02,
            'track_drawdown': True
        }
    }


@pytest.fixture
def mock_exchange_config():
    """Mock exchange configuration for testing"""
    return {
        'binance': {
            'api_key': 'test_api_key',
            'secret': 'test_secret',
            'enabled': True
        }
    }


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Custom pytest configuration"""
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
