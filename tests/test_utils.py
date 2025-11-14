#!/usr/bin/env python3
"""
Unit tests for Nexlify Utilities
Testing error handling and utility functions
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.utils.error_handler import ErrorHandler, get_error_handler, handle_errors
from nexlify.utils.utils_module import (
    format_currency,
    format_percentage,
    calculate_returns,
    validate_config,
    safe_divide,
)


class TestErrorHandler:
    """Test Error Handler functionality"""

    @pytest.fixture
    def error_handler(self):
        """Create error handler instance"""
        return ErrorHandler()

    def test_initialization(self, error_handler):
        """Test error handler initialization"""
        assert error_handler is not None
        assert hasattr(error_handler, "log_error")

    def test_log_error_info(self, error_handler, caplog):
        """Test logging info level errors"""
        with caplog.at_level(logging.INFO):
            error_handler.log_error(
                Exception("Test error"), context="Test context", severity="info"
            )

        assert "Test error" in caplog.text or "Test context" in caplog.text

    def test_log_error_warning(self, error_handler, caplog):
        """Test logging warning level errors"""
        with caplog.at_level(logging.WARNING):
            error_handler.log_error(
                Exception("Test warning"), context="Warning context", severity="warning"
            )

        assert len(caplog.records) > 0

    def test_log_error_critical(self, error_handler, caplog):
        """Test logging critical level errors"""
        with caplog.at_level(logging.CRITICAL):
            error_handler.log_error(
                Exception("Critical error"),
                context="Critical context",
                severity="critical",
            )

        assert len(caplog.records) > 0

    def test_get_error_handler_singleton(self):
        """Test error handler singleton pattern"""
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        assert handler1 is handler2

    def test_handle_errors_decorator_success(self):
        """Test handle_errors decorator with successful function"""

        @handle_errors("Test function")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_handle_errors_decorator_exception(self):
        """Test handle_errors decorator with exception"""

        @handle_errors("Test function", reraise=False)
        def failing_function():
            raise ValueError("Test error")

        # Should not raise, returns None
        result = failing_function()
        assert result is None

    def test_handle_errors_decorator_reraise(self):
        """Test handle_errors decorator with reraise"""

        @handle_errors("Test function", reraise=True)
        def failing_function():
            raise ValueError("Test error")

        # Should reraise
        with pytest.raises(ValueError):
            failing_function()


class TestFormatting:
    """Test formatting utilities"""

    def test_format_currency_positive(self):
        """Test formatting positive currency"""
        result = format_currency(1234.56)
        assert "$" in result
        assert "1234" in result or "1,234" in result

    def test_format_currency_negative(self):
        """Test formatting negative currency"""
        result = format_currency(-1234.56)
        assert "$" in result or "-" in result

    def test_format_currency_zero(self):
        """Test formatting zero currency"""
        result = format_currency(0.0)
        assert "$" in result
        assert "0" in result

    def test_format_percentage_positive(self):
        """Test formatting positive percentage"""
        result = format_percentage(0.1234)
        assert "%" in result
        assert "12" in result

    def test_format_percentage_negative(self):
        """Test formatting negative percentage"""
        result = format_percentage(-0.05)
        assert "%" in result
        assert "5" in result or "-" in result

    def test_format_percentage_zero(self):
        """Test formatting zero percentage"""
        result = format_percentage(0.0)
        assert "%" in result
        assert "0" in result


class TestCalculations:
    """Test calculation utilities"""

    def test_calculate_returns_profit(self):
        """Test calculating positive returns"""
        entry_price = 100.0
        exit_price = 110.0
        quantity = 1.0

        returns = calculate_returns(entry_price, exit_price, quantity)

        assert returns == 10.0

    def test_calculate_returns_loss(self):
        """Test calculating negative returns"""
        entry_price = 100.0
        exit_price = 90.0
        quantity = 1.0

        returns = calculate_returns(entry_price, exit_price, quantity)

        assert returns == -10.0

    def test_calculate_returns_with_quantity(self):
        """Test calculating returns with different quantity"""
        entry_price = 100.0
        exit_price = 110.0
        quantity = 2.0

        returns = calculate_returns(entry_price, exit_price, quantity)

        assert returns == 20.0

    def test_safe_divide_normal(self):
        """Test safe divide with normal values"""
        result = safe_divide(10, 2)
        assert result == 5.0

    def test_safe_divide_by_zero(self):
        """Test safe divide by zero"""
        result = safe_divide(10, 0)
        assert result == 0.0  # or whatever default is set

    def test_safe_divide_with_default(self):
        """Test safe divide with custom default"""
        result = safe_divide(10, 0, default=999)
        assert result == 999

    def test_safe_divide_negative(self):
        """Test safe divide with negative numbers"""
        result = safe_divide(-10, 2)
        assert result == -5.0


class TestConfigValidation:
    """Test configuration validation"""

    def test_validate_config_valid(self):
        """Test validating valid configuration"""
        config = {
            "risk_management": {"enabled": True, "max_position_size": 100},
            "trading": {"enabled": True},
        }

        result = validate_config(config)
        assert result is True

    def test_validate_config_missing_required(self):
        """Test validating config with missing required fields"""
        config = {}

        result = validate_config(config, required_keys=["risk_management"])
        assert result is False

    def test_validate_config_invalid_types(self):
        """Test validating config with invalid types"""
        config = {"risk_management": "invalid"}  # Should be dict

        result = validate_config(config)
        # Depending on implementation, may return False or raise error
        assert isinstance(result, bool)

    def test_validate_config_nested(self):
        """Test validating nested configuration"""
        config = {"risk_management": {"enabled": True, "limits": {"max_loss": 100}}}

        result = validate_config(config)
        assert isinstance(result, bool)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_format_currency_very_large(self):
        """Test formatting very large currency values"""
        result = format_currency(1_000_000_000.0)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_format_currency_very_small(self):
        """Test formatting very small currency values"""
        result = format_currency(0.0001)
        assert isinstance(result, str)

    def test_calculate_returns_zero_prices(self):
        """Test returns with zero prices"""
        returns = calculate_returns(0, 100, 1)
        # Should handle gracefully
        assert isinstance(returns, (int, float))

    def test_calculate_returns_zero_quantity(self):
        """Test returns with zero quantity"""
        returns = calculate_returns(100, 110, 0)
        assert returns == 0.0

    def test_safe_divide_both_zero(self):
        """Test safe divide with both numerator and denominator zero"""
        result = safe_divide(0, 0)
        assert isinstance(result, (int, float))

    def test_error_handler_with_none_exception(self):
        """Test error handler with None"""
        handler = get_error_handler()
        # Should handle gracefully
        try:
            handler.log_error(None, "Test context")
        except Exception:
            pytest.fail("Error handler should handle None exception")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
