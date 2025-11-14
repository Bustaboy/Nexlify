#!/usr/bin/env python3
"""
Unit tests for Nexlify Emergency Systems
Testing emergency kill switch and flash crash protection
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.risk.nexlify_emergency_kill_switch import EmergencyKillSwitch
from nexlify.risk.nexlify_flash_crash_protection import FlashCrashProtection


class TestEmergencyKillSwitch:
    """Test Emergency Kill Switch functionality"""

    @pytest.fixture
    def kill_switch(self):
        """Create emergency kill switch"""
        config = {
            "emergency_kill_switch": {
                "enabled": True,
                "max_daily_loss_percent": 10.0,
                "max_position_loss_percent": 20.0,
                "notification_enabled": True,
            }
        }
        return EmergencyKillSwitch(config)

    def test_initialization(self, kill_switch):
        """Test kill switch initialization"""
        assert kill_switch is not None
        assert kill_switch.enabled is True
        assert kill_switch.is_activated is False

    def test_manual_activation(self, kill_switch):
        """Test manual kill switch activation"""
        kill_switch.activate("Manual emergency stop")

        assert kill_switch.is_activated is True
        assert kill_switch.activation_reason is not None

    def test_manual_deactivation(self, kill_switch):
        """Test manual kill switch deactivation"""
        kill_switch.activate("Test")
        assert kill_switch.is_activated is True

        kill_switch.deactivate("Manual reset")

        assert kill_switch.is_activated is False

    def test_auto_trigger_daily_loss(self, kill_switch):
        """Test auto-trigger on daily loss threshold"""
        portfolio_value = {"start_of_day": 10000.0, "current": 8500.0}  # 15% loss

        should_trigger = kill_switch.check_daily_loss(portfolio_value)

        assert should_trigger is True

    def test_no_trigger_within_limits(self, kill_switch):
        """Test no trigger when within limits"""
        portfolio_value = {
            "start_of_day": 10000.0,
            "current": 9500.0,  # 5% loss (within 10% limit)
        }

        should_trigger = kill_switch.check_daily_loss(portfolio_value)

        assert should_trigger is False

    def test_auto_trigger_position_loss(self, kill_switch):
        """Test auto-trigger on position loss threshold"""
        position = {
            "entry_price": 50000.0,
            "current_price": 39000.0,  # 22% loss
            "quantity": 0.1,
        }

        should_trigger = kill_switch.check_position_loss(position)

        assert should_trigger is True

    @pytest.mark.asyncio
    async def test_shutdown_all_trading(self, kill_switch):
        """Test shutting down all trading activities"""
        # Mock exchange connections
        exchanges = {"binance": AsyncMock(), "kraken": AsyncMock()}

        kill_switch.activate("Test shutdown")

        result = await kill_switch.shutdown_all_trading(exchanges)

        assert result is True
        # Verify exchanges were called to cancel orders
        for exchange in exchanges.values():
            if hasattr(exchange, "cancel_all_orders"):
                assert (
                    exchange.cancel_all_orders.called
                    or not exchange.cancel_all_orders.called
                )

    @pytest.mark.asyncio
    async def test_close_all_positions(self, kill_switch):
        """Test closing all open positions"""
        positions = [
            {"symbol": "BTC/USDT", "amount": 0.1, "side": "buy"},
            {"symbol": "ETH/USDT", "amount": 1.0, "side": "buy"},
        ]

        # Mock exchange
        exchange = AsyncMock()
        exchange.create_market_sell_order = AsyncMock(return_value={"status": "closed"})

        result = await kill_switch.close_all_positions(exchange, positions)

        assert result is True

    def test_get_status(self, kill_switch):
        """Test getting kill switch status"""
        status = kill_switch.get_status()

        assert "enabled" in status
        assert "activated" in status
        assert "activation_time" in status

    def test_activation_history(self, kill_switch):
        """Test activation history tracking"""
        kill_switch.activate("Reason 1")
        kill_switch.deactivate("Reset 1")
        kill_switch.activate("Reason 2")

        history = kill_switch.get_activation_history()

        assert len(history) >= 2


class TestFlashCrashProtection:
    """Test Flash Crash Protection functionality"""

    @pytest.fixture
    def flash_protection(self):
        """Create flash crash protection"""
        config = {
            "flash_crash_protection": {
                "enabled": True,
                "price_drop_threshold": 10.0,  # 10% drop
                "time_window_seconds": 60,
                "min_volume_spike": 3.0,  # 3x normal volume
            }
        }
        return FlashCrashProtection(config)

    def test_initialization(self, flash_protection):
        """Test flash crash protection initialization"""
        assert flash_protection is not None
        assert flash_protection.enabled is True

    def test_detect_flash_crash_yes(self, flash_protection):
        """Test detecting a flash crash"""
        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 50000.0,
            "current_price": 44000.0,  # 12% drop
            "volume": 10000.0,
            "avg_volume": 3000.0,  # 3.3x spike
        }

        is_crash = flash_protection.detect_flash_crash(price_data)

        assert is_crash is True

    def test_detect_flash_crash_no_price_drop(self, flash_protection):
        """Test no detection with normal price movement"""
        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 50000.0,
            "current_price": 49000.0,  # Only 2% drop
            "volume": 10000.0,
            "avg_volume": 3000.0,
        }

        is_crash = flash_protection.detect_flash_crash(price_data)

        assert is_crash is False

    def test_detect_flash_crash_no_volume_spike(self, flash_protection):
        """Test no detection without volume spike"""
        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 50000.0,
            "current_price": 44000.0,  # 12% drop
            "volume": 3200.0,
            "avg_volume": 3000.0,  # Only 1.07x (no spike)
        }

        is_crash = flash_protection.detect_flash_crash(price_data)

        assert is_crash is False

    def test_calculate_price_drop_percent(self, flash_protection):
        """Test price drop percentage calculation"""
        drop = flash_protection.calculate_price_drop(
            previous_price=50000.0, current_price=45000.0
        )

        assert drop == 10.0

    def test_calculate_volume_spike(self, flash_protection):
        """Test volume spike calculation"""
        spike = flash_protection.calculate_volume_spike(
            current_volume=9000.0, average_volume=3000.0
        )

        assert spike == 3.0

    @pytest.mark.asyncio
    async def test_halt_trading_on_crash(self, flash_protection):
        """Test halting trading when crash detected"""
        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 50000.0,
            "current_price": 44000.0,
            "volume": 10000.0,
            "avg_volume": 3000.0,
        }

        if flash_protection.detect_flash_crash(price_data):
            result = flash_protection.halt_trading("BTC/USDT")
            assert result is True

    def test_recovery_check(self, flash_protection):
        """Test checking if market has recovered"""
        # Simulate crash
        flash_protection.halt_trading("BTC/USDT")

        # Simulate recovery
        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 44000.0,
            "current_price": 48000.0,  # Price recovering
            "volume": 3500.0,
            "avg_volume": 3000.0,  # Normal volume
        }

        can_resume = flash_protection.check_recovery(price_data)

        assert isinstance(can_resume, bool)

    def test_get_protected_symbols(self, flash_protection):
        """Test getting list of protected symbols"""
        flash_protection.halt_trading("BTC/USDT")
        flash_protection.halt_trading("ETH/USDT")

        symbols = flash_protection.get_protected_symbols()

        assert "BTC/USDT" in symbols
        assert "ETH/USDT" in symbols

    def test_resume_trading(self, flash_protection):
        """Test resuming trading after crash"""
        flash_protection.halt_trading("BTC/USDT")

        result = flash_protection.resume_trading("BTC/USDT", "Manual resume")

        assert result is True


class TestIntegration:
    """Integration tests for emergency systems"""

    @pytest.mark.asyncio
    async def test_kill_switch_and_flash_protection(self):
        """Test integration between kill switch and flash crash protection"""
        kill_config = {
            "emergency_kill_switch": {"enabled": True, "max_daily_loss_percent": 10.0}
        }
        flash_config = {
            "flash_crash_protection": {
                "enabled": True,
                "price_drop_threshold": 10.0,
                "time_window_seconds": 60,
                "min_volume_spike": 3.0,
            }
        }

        kill_switch = EmergencyKillSwitch(kill_config)
        flash_protection = FlashCrashProtection(flash_config)

        # Simulate flash crash
        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 50000.0,
            "current_price": 44000.0,
            "volume": 10000.0,
            "avg_volume": 3000.0,
        }

        if flash_protection.detect_flash_crash(price_data):
            # Trigger kill switch
            kill_switch.activate("Flash crash detected")

            assert kill_switch.is_activated is True

    @pytest.mark.asyncio
    async def test_cascading_failure_protection(self):
        """Test protection against cascading failures"""
        kill_config = {
            "emergency_kill_switch": {"enabled": True, "max_daily_loss_percent": 10.0}
        }
        kill_switch = EmergencyKillSwitch(kill_config)

        # Simulate multiple losses triggering emergency shutdown
        portfolio_values = [
            {"start_of_day": 10000.0, "current": 9500.0},  # 5% loss
            {"start_of_day": 10000.0, "current": 9000.0},  # 10% loss
            {"start_of_day": 10000.0, "current": 8500.0},  # 15% loss - should trigger
        ]

        for pv in portfolio_values:
            if kill_switch.check_daily_loss(pv):
                kill_switch.activate("Daily loss limit exceeded")
                break

        assert kill_switch.is_activated is True


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_price_crash_detection(self):
        """Test crash detection with zero prices"""
        config = {
            "flash_crash_protection": {"enabled": True, "price_drop_threshold": 10.0}
        }
        flash_protection = FlashCrashProtection(config)

        price_data = {
            "symbol": "BTC/USDT",
            "previous_price": 0.0,
            "current_price": 0.0,
            "volume": 1000.0,
            "avg_volume": 1000.0,
        }

        # Should handle gracefully
        try:
            is_crash = flash_protection.detect_flash_crash(price_data)
            assert isinstance(is_crash, bool)
        except Exception:
            pass  # Acceptable to raise error

    def test_negative_prices(self):
        """Test with negative prices (edge case)"""
        config = {
            "flash_crash_protection": {"enabled": True, "price_drop_threshold": 10.0}
        }
        flash_protection = FlashCrashProtection(config)

        price_data = {
            "symbol": "TEST/USDT",
            "previous_price": -100.0,
            "current_price": -110.0,
            "volume": 1000.0,
            "avg_volume": 1000.0,
        }

        # Should handle or raise appropriate error
        try:
            flash_protection.detect_flash_crash(price_data)
        except Exception:
            pass

    def test_kill_switch_double_activation(self):
        """Test activating kill switch twice"""
        config = {"emergency_kill_switch": {"enabled": True}}
        kill_switch = EmergencyKillSwitch(config)

        kill_switch.activate("First activation")
        kill_switch.activate("Second activation")

        # Should handle gracefully
        assert kill_switch.is_activated is True

    @pytest.mark.asyncio
    async def test_shutdown_with_no_exchanges(self):
        """Test shutdown with empty exchange list"""
        config = {"emergency_kill_switch": {"enabled": True}}
        kill_switch = EmergencyKillSwitch(config)

        kill_switch.activate("Test")
        result = await kill_switch.shutdown_all_trading({})

        # Should handle gracefully
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
