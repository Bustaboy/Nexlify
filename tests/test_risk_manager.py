#!/usr/bin/env python3
"""
Unit tests for Nexlify Risk Manager
Comprehensive testing of risk management functionality
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.risk.nexlify_risk_manager import (RiskManager, RiskMetrics,
                                               TradeValidation)


@pytest.fixture
def test_config():
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
        }
    }


@pytest.fixture
def risk_manager(test_config, tmp_path):
    """Create a risk manager with temporary state file"""
    # Use temporary directory for state file
    state_file = tmp_path / "risk_state.json"
    rm = RiskManager(test_config)
    rm.state_file = state_file
    return rm


class TestRiskManagerInitialization:
    """Test risk manager initialization"""

    def test_initialization(self, risk_manager):
        """Test proper initialization"""
        assert risk_manager.enabled is True
        assert risk_manager.max_position_size == 0.05
        assert risk_manager.max_daily_loss == 0.05
        assert risk_manager.stop_loss_percent == 0.02
        assert risk_manager.take_profit_percent == 0.05
        assert risk_manager.use_kelly is True

    def test_disabled_risk_manager(self):
        """Test disabled risk manager"""
        config = {"risk_management": {"enabled": False}}
        rm = RiskManager(config)
        assert rm.enabled is False


class TestPositionSizeLimits:
    """Test position size validation"""

    @pytest.mark.asyncio
    async def test_valid_position_size(self, risk_manager):
        """Test trade within position size limits"""
        balance = 10000.0
        quantity = 0.1
        price = 5000.0  # $500 position = 5% of balance

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=0.7,
        )

        assert validation.approved is True
        assert "risk checks passed" in validation.reason.lower()

    @pytest.mark.asyncio
    async def test_exceeds_position_size(self, risk_manager):
        """Test trade exceeding position size limit"""
        balance = 10000.0
        quantity = 0.2
        price = 5000.0  # $1000 position = 10% of balance (exceeds 5% limit)

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=0.7,
        )

        assert validation.approved is False
        assert "position size" in validation.reason.lower()
        assert validation.adjusted_size is not None
        # Adjusted size should be within limits
        assert (
            validation.adjusted_size * price <= balance * 0.05 * 1.01
        )  # Small tolerance

    @pytest.mark.asyncio
    async def test_exact_limit_position(self, risk_manager):
        """Test trade at exact position size limit"""
        balance = 10000.0
        price = 50000.0
        quantity = (balance * 0.05) / price  # Exactly 5%

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=0.7,
        )

        assert validation.approved is True


class TestDailyLossLimits:
    """Test daily loss limit enforcement"""

    @pytest.mark.asyncio
    async def test_within_daily_loss(self, risk_manager):
        """Test trading with daily loss under limit"""
        risk_manager.metrics.daily_loss = 0.02  # 2% loss

        validation = await risk_manager.validate_trade(
            symbol="ETH/USDT",
            side="buy",
            quantity=1.0,
            price=3000.0,
            balance=100000.0,
            confidence=0.7,
        )

        assert validation.approved is True

    @pytest.mark.asyncio
    async def test_daily_loss_limit_reached(self, risk_manager):
        """Test trading halted when daily loss limit reached"""
        risk_manager.metrics.daily_loss = 0.06  # 6% loss (exceeds 5% limit)

        validation = await risk_manager.validate_trade(
            symbol="ETH/USDT",
            side="buy",
            quantity=1.0,
            price=3000.0,
            balance=100000.0,
            confidence=0.7,
        )

        assert validation.approved is False
        assert "daily loss limit" in validation.reason.lower()
        assert risk_manager.trading_halted is True

    def test_record_losing_trade(self, risk_manager):
        """Test recording a losing trade"""
        balance = 10000.0
        initial_loss = risk_manager.metrics.daily_loss

        risk_manager.record_trade_result(
            symbol="BTC/USDT",
            side="buy",
            entry_price=50000.0,
            exit_price=49000.0,  # $1000 loss on 1 BTC
            quantity=1.0,
            balance=balance,
        )

        # Loss should increase
        assert risk_manager.metrics.daily_loss > initial_loss
        assert risk_manager.metrics.trades_today == 1

    def test_record_winning_trade(self, risk_manager):
        """Test recording a winning trade"""
        balance = 10000.0
        initial_profit = risk_manager.metrics.daily_profit

        risk_manager.record_trade_result(
            symbol="BTC/USDT",
            side="buy",
            entry_price=50000.0,
            exit_price=51000.0,  # $1000 profit on 1 BTC
            quantity=1.0,
            balance=balance,
        )

        # Profit should increase
        assert risk_manager.metrics.daily_profit > initial_profit
        assert risk_manager.metrics.trades_today == 1

    def test_daily_loss_triggers_halt(self, risk_manager):
        """Test that hitting loss limit halts trading"""
        balance = 10000.0

        # Record a trade that puts us over the limit
        risk_manager.record_trade_result(
            symbol="BTC/USDT",
            side="buy",
            entry_price=50000.0,
            exit_price=45000.0,  # Big loss
            quantity=0.1,  # $500 loss = 5% of balance
            balance=balance,
        )

        # Should trigger halt
        assert risk_manager.trading_halted is True


class TestStopLossTakeProfit:
    """Test stop-loss and take-profit calculation"""

    @pytest.mark.asyncio
    async def test_buy_stop_loss_take_profit(self, risk_manager):
        """Test stop-loss and take-profit for buy order"""
        price = 50000.0

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=price,
            balance=10000.0,
            confidence=0.7,
        )

        assert validation.stop_loss is not None
        assert validation.take_profit is not None

        # Stop loss should be 2% below entry
        expected_stop = price * (1 - 0.02)
        assert abs(validation.stop_loss - expected_stop) < 0.01

        # Take profit should be 5% above entry
        expected_tp = price * (1 + 0.05)
        assert abs(validation.take_profit - expected_tp) < 0.01

    @pytest.mark.asyncio
    async def test_sell_stop_loss_take_profit(self, risk_manager):
        """Test stop-loss and take-profit for sell order"""
        price = 50000.0

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="sell",
            quantity=0.1,
            price=price,
            balance=10000.0,
            confidence=0.7,
        )

        # Stop loss should be 2% above entry (for short)
        expected_stop = price * (1 + 0.02)
        assert abs(validation.stop_loss - expected_stop) < 0.01

        # Take profit should be 5% below entry (for short)
        expected_tp = price * (1 - 0.05)
        assert abs(validation.take_profit - expected_tp) < 0.01


class TestKellyCriterion:
    """Test Kelly Criterion position sizing"""

    def test_kelly_calculation(self, risk_manager):
        """Test Kelly Criterion calculation"""
        balance = 10000.0
        price = 50000.0
        confidence = 0.7

        kelly_size = risk_manager._calculate_kelly_size(balance, price, confidence)

        # Kelly size should be positive and reasonable
        assert kelly_size > 0
        assert kelly_size * price <= balance  # Can't exceed balance

    @pytest.mark.asyncio
    async def test_kelly_reduces_position(self, risk_manager):
        """Test that Kelly Criterion can reduce position size"""
        balance = 10000.0
        quantity = 0.05  # Relatively large position
        price = 4000.0
        confidence = 0.65  # Moderate confidence

        validation = await risk_manager.validate_trade(
            symbol="ETH/USDT",
            side="buy",
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=confidence,
        )

        # With moderate confidence, Kelly might suggest smaller size
        if validation.adjusted_size:
            assert validation.adjusted_size <= quantity

    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self, risk_manager):
        """Test that low confidence trades use standard position sizing"""
        balance = 10000.0
        quantity = 0.1
        price = 1000.0
        confidence = 0.5  # Below min_kelly_confidence (0.6)

        validation = await risk_manager.validate_trade(
            symbol="ETH/USDT",
            side="buy",
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=confidence,
        )

        # Should still validate, but Kelly not applied
        assert validation.approved is True

    @pytest.mark.asyncio
    async def test_kelly_disabled(self):
        """Test with Kelly Criterion disabled"""
        config = {
            "risk_management": {
                "enabled": True,
                "max_position_size": 0.05,
                "use_kelly_criterion": False,
            }
        }
        rm = RiskManager(config)

        validation = await rm.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=5000.0,
            balance=10000.0,
            confidence=0.9,
        )

        # Should not apply Kelly adjustment
        assert validation.adjusted_size is None or validation.adjusted_size == 0.1


class TestConcurrentTrades:
    """Test max concurrent trades limit"""

    @pytest.mark.asyncio
    async def test_within_concurrent_limit(self, risk_manager):
        """Test trading with open positions under limit"""
        risk_manager.metrics.open_positions = 2  # Under limit of 3

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,  # Fixed: $500 = 5% of $10000 balance
            price=50000.0,
            balance=10000.0,
            confidence=0.7,
        )

        assert validation.approved is True

    @pytest.mark.asyncio
    async def test_exceeds_concurrent_limit(self, risk_manager):
        """Test rejection when max concurrent trades reached"""
        risk_manager.metrics.open_positions = 3  # At limit

        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.01,  # Fixed: $500 = 5% of $10000 balance
            price=50000.0,
            balance=10000.0,
            confidence=0.7,
        )

        assert validation.approved is False
        assert "concurrent" in validation.reason.lower()


class TestStatePersistence:
    """Test state saving and loading"""

    def test_save_and_load_state(self, risk_manager):
        """Test state persistence across restarts"""
        # Set some state
        risk_manager.metrics.daily_loss = 0.03
        risk_manager.metrics.daily_profit = 0.02
        risk_manager.metrics.trades_today = 5
        risk_manager._save_state()

        # Create new instance (simulating restart)
        new_rm = RiskManager(risk_manager.config)
        new_rm.state_file = risk_manager.state_file
        new_rm._load_state()

        # State should be restored
        assert new_rm.metrics.daily_loss == 0.03
        assert new_rm.metrics.daily_profit == 0.02
        assert new_rm.metrics.trades_today == 5

    def test_daily_reset_on_new_day(self, risk_manager, tmp_path):
        """Test that metrics reset on new trading day"""
        # Set state from "yesterday"
        yesterday = datetime.now() - timedelta(days=1)
        state_data = {
            "daily_loss": 0.04,
            "daily_profit": 0.02,
            "trades_today": 10,
            "last_reset": yesterday.isoformat(),
            "trading_halted": False,
            "halt_reason": "",
        }

        # Save state
        with open(risk_manager.state_file, "w") as f:
            json.dump(state_data, f)

        # Load state (should trigger reset)
        risk_manager._load_state()

        # Metrics should be reset
        assert risk_manager.metrics.daily_loss == 0.0
        assert risk_manager.metrics.daily_profit == 0.0
        assert risk_manager.metrics.trades_today == 0


class TestRiskStatus:
    """Test risk status reporting"""

    def test_get_risk_status(self, risk_manager):
        """Test risk status dictionary"""
        status = risk_manager.get_risk_status()

        assert "enabled" in status
        assert "trading_halted" in status
        assert "daily_profit" in status
        assert "daily_loss" in status
        assert "net_pnl" in status
        assert "trades_today" in status
        assert "max_position_size" in status
        assert "kelly_enabled" in status

    def test_force_reset(self, risk_manager):
        """Test force reset functionality"""
        # Set some metrics
        risk_manager.metrics.daily_loss = 0.04
        risk_manager.metrics.trades_today = 5
        risk_manager.trading_halted = True

        # Force reset
        risk_manager.force_reset()

        # Should be reset
        assert risk_manager.metrics.daily_loss == 0.0
        assert risk_manager.metrics.trades_today == 0
        assert risk_manager.trading_halted is False

    def test_resume_trading(self, risk_manager):
        """Test resume trading after halt"""
        # Halt trading
        risk_manager.trading_halted = True
        risk_manager.halt_reason = "Test halt"

        # Resume
        risk_manager.resume_trading("Test resume")

        assert risk_manager.trading_halted is False
        assert risk_manager.halt_reason == ""


class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_zero_balance(self, risk_manager):
        """Test with zero balance"""
        validation = await risk_manager.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            balance=0.0,
            confidence=0.7,
        )

        assert validation.approved is False

    @pytest.mark.asyncio
    async def test_disabled_risk_management(self):
        """Test with risk management disabled"""
        config = {"risk_management": {"enabled": False}}
        rm = RiskManager(config)

        validation = await rm.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=100.0,  # Unreasonable amount
            price=50000.0,
            balance=10000.0,
            confidence=0.7,
        )

        # Should approve when disabled
        assert validation.approved is True
        assert "disabled" in validation.reason.lower()

    def test_record_trade_zero_balance(self, risk_manager):
        """Test recording trade with zero balance"""
        # Should not crash
        risk_manager.record_trade_result(
            symbol="BTC/USDT",
            side="buy",
            entry_price=50000.0,
            exit_price=51000.0,
            quantity=0.1,
            balance=0.0,
        )

        # Should handle gracefully
        assert risk_manager.metrics.trades_today == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
