#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Nexlify tests
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an instance of the default event loop for the test session.
    Session-scoped to reuse across all tests for better performance.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def sample_config():
    """Standard test configuration"""
    return {
        "risk_management": {
            "enabled": True,
            "max_position_size": 0.05,
            "max_daily_loss": 0.05,
            "stop_loss_percent": 0.02,
            "take_profit_percent": 0.05,
            "use_kelly_criterion": True,
            "kelly_fraction": 0.5,
            "min_kelly_confidence": 0.6,
            "max_concurrent_trades": 3,
        },
        "circuit_breaker": {
            "enabled": True,
            "failure_threshold": 3,
            "timeout_seconds": 2,
            "half_open_max_calls": 1,
        },
        "performance_tracking": {
            "enabled": True,
            "database_path": "data/trading.db",
            "calculate_sharpe_ratio": True,
            "risk_free_rate": 0.02,
            "track_drawdown": True,
        },
    }


@pytest.fixture(scope="session")
def mock_exchange_config():
    """Mock exchange configuration for testing (session-scoped for performance)"""
    return {
        "binance": {"api_key": "test_api_key", "secret": "test_secret", "enabled": True}
    }


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture(scope="session")
def in_memory_db_url():
    """
    In-memory SQLite database URL for testing (much faster than disk-based).
    Session-scoped for reuse across tests.
    """
    return "sqlite:///:memory:"


@pytest.fixture
def test_db_config(in_memory_db_url):
    """
    Test configuration with in-memory database for faster tests.
    Function-scoped so each test gets a fresh database.
    """
    return {
        "performance_tracking": {
            "enabled": True,
            "database_path": ":memory:",  # In-memory database
            "calculate_sharpe_ratio": True,
            "risk_free_rate": 0.02,
            "track_drawdown": True,
        }
    }


# Pytest hooks for custom behavior
def pytest_configure(config):
    """Custom pytest configuration"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "requires_network: requires network connectivity")
    config.addinivalue_line("markers", "requires_gpu: requires GPU hardware")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to optimize test execution order.
    Run fast unit tests first for quicker feedback.
    """
    # Separate tests into categories
    unit_tests = []
    integration_tests = []
    slow_tests = []
    other_tests = []

    for item in items:
        if "slow" in item.keywords:
            slow_tests.append(item)
        elif "integration" in item.keywords:
            integration_tests.append(item)
        elif "unit" in item.keywords:
            unit_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: unit tests first (fast feedback), then others, then slow tests last
    items[:] = unit_tests + other_tests + integration_tests + slow_tests
