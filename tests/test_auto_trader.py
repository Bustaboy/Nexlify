#!/usr/bin/env python3
"""
Unit tests for Nexlify Auto Trader
Comprehensive testing of autonomous trading functionality
"""

import pytest
import asyncio
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.core.nexlify_auto_trader import (
    AutoTrader,
    RiskManager,
    TradeExecution
)


@pytest.fixture
def risk_config():
    """Standard risk management configuration"""
    return {
        'max_position_size': 100.0,
        'max_concurrent_trades': 5,
        'max_daily_loss': 100.0,
        'min_profit_percent': 0.5,
        'min_confidence': 0.7
    }


@pytest.fixture
def risk_manager(risk_config):
    """Create a risk manager"""
    return RiskManager(risk_config)


@pytest.fixture
def auto_trader_config():
    """Standard auto trader configuration"""
    return {
        'enabled': True,
        'check_interval': 60,
        'risk_management': {
            'max_position_size': 100.0,
            'max_concurrent_trades': 5,
            'max_daily_loss': 100.0,
            'min_profit_percent': 0.5,
            'min_confidence': 0.7
        }
    }


class TestRiskManager:
    """Test Risk Manager functionality"""

    def test_initialization(self, risk_manager):
        """Test risk manager initialization"""
        assert risk_manager.max_position_size == 100.0
        assert risk_manager.max_concurrent_trades == 5
        assert risk_manager.max_daily_loss == 100.0
        assert risk_manager.min_profit_threshold == 0.5
        assert risk_manager.min_confidence == 0.7
        assert risk_manager.daily_profit == 0.0
        assert risk_manager.daily_trades == 0

    def test_daily_loss_limit_check(self, risk_manager):
        """Test daily loss limit checking"""
        # Within limit
        risk_manager.daily_profit = -50.0
        assert risk_manager.check_daily_loss_limit() is True

        # Exceeded limit
        risk_manager.daily_profit = -150.0
        assert risk_manager.check_daily_loss_limit() is False

    def test_daily_stats_reset(self, risk_manager):
        """Test daily statistics reset"""
        # Set some stats
        risk_manager.daily_profit = -50.0
        risk_manager.daily_trades = 10

        # Simulate next day
        risk_manager.last_reset = datetime.now().date() - timedelta(days=1)
        risk_manager.reset_daily_stats()

        # Should be reset
        assert risk_manager.daily_profit == 0.0
        assert risk_manager.daily_trades == 0

    def test_concurrent_trades_check(self, risk_manager):
        """Test concurrent trades limit checking"""
        # Under limit
        assert risk_manager.check_concurrent_trades(3) is True

        # At limit
        assert risk_manager.check_concurrent_trades(5) is False

        # Over limit
        assert risk_manager.check_concurrent_trades(6) is False

    def test_position_size_calculation(self, risk_manager):
        """Test position size calculation"""
        balance = 10000.0

        # 2% risk on $10k = $200, but capped at max_position_size
        position_size = risk_manager.calculate_position_size(balance, risk_percent=2.0)
        assert position_size == 100.0  # Capped at max

        # Lower risk
        position_size = risk_manager.calculate_position_size(balance, risk_percent=0.5)
        assert position_size == 50.0

    def test_should_trade_all_checks_pass(self, risk_manager):
        """Test should_trade when all checks pass"""
        pair_data = {
            'profit_score': 1.5,
            'confidence': 0.8
        }

        should_trade, reason = risk_manager.should_trade(
            pair_data=pair_data,
            balance=10000.0,
            active_trades=2
        )

        assert should_trade is True
        assert "approved" in reason.lower()

    def test_should_trade_daily_loss_exceeded(self, risk_manager):
        """Test should_trade rejects when daily loss exceeded"""
        risk_manager.daily_profit = -150.0

        pair_data = {
            'profit_score': 1.5,
            'confidence': 0.8
        }

        should_trade, reason = risk_manager.should_trade(
            pair_data=pair_data,
            balance=10000.0,
            active_trades=2
        )

        assert should_trade is False
        assert "daily loss" in reason.lower()

    def test_should_trade_max_trades_reached(self, risk_manager):
        """Test should_trade rejects when max concurrent trades reached"""
        pair_data = {
            'profit_score': 1.5,
            'confidence': 0.8
        }

        should_trade, reason = risk_manager.should_trade(
            pair_data=pair_data,
            balance=10000.0,
            active_trades=5  # At limit
        )

        assert should_trade is False
        assert "concurrent" in reason.lower()

    def test_should_trade_low_profit_score(self, risk_manager):
        """Test should_trade rejects when profit score too low"""
        pair_data = {
            'profit_score': 0.3,  # Below 0.5 threshold
            'confidence': 0.8
        }

        should_trade, reason = risk_manager.should_trade(
            pair_data=pair_data,
            balance=10000.0,
            active_trades=2
        )

        assert should_trade is False
        assert "profit" in reason.lower()

    def test_should_trade_low_confidence(self, risk_manager):
        """Test should_trade rejects when confidence too low"""
        pair_data = {
            'profit_score': 1.5,
            'confidence': 0.5  # Below 0.7 threshold
        }

        should_trade, reason = risk_manager.should_trade(
            pair_data=pair_data,
            balance=10000.0,
            active_trades=2
        )

        assert should_trade is False
        assert "confidence" in reason.lower()


class TestTradeExecution:
    """Test TradeExecution dataclass"""

    def test_trade_execution_creation(self):
        """Test creating a trade execution"""
        trade = TradeExecution(
            trade_id="trade_123",
            symbol="BTC/USDT",
            exchange="binance",
            side="buy",
            amount=0.1,
            price=50000.0,
            timestamp=datetime.now(),
            profit_target=51000.0,
            stop_loss=49000.0,
            strategy="neural_net",
            status="open"
        )

        assert trade.trade_id == "trade_123"
        assert trade.symbol == "BTC/USDT"
        assert trade.side == "buy"
        assert trade.amount == 0.1
        assert trade.status == "open"


class TestAutoTrader:
    """Test Auto Trader functionality"""

    @pytest.fixture
    def auto_trader(self, auto_trader_config):
        """Create auto trader instance"""
        trader = AutoTrader(auto_trader_config)
        # Mock exchanges to avoid real connections
        trader.exchanges = {
            'binance': AsyncMock(),
            'kraken': AsyncMock()
        }
        return trader

    def test_initialization(self, auto_trader):
        """Test auto trader initialization"""
        assert auto_trader.enabled is True
        assert auto_trader.check_interval == 60
        assert auto_trader.risk_manager is not None
        assert len(auto_trader.active_trades) == 0

    @pytest.mark.asyncio
    async def test_start_and_stop(self, auto_trader):
        """Test starting and stopping auto trader"""
        # Start
        task = asyncio.create_task(auto_trader.start())
        await asyncio.sleep(0.1)

        assert auto_trader.is_running is True

        # Stop
        await auto_trader.stop()
        await asyncio.sleep(0.1)

        assert auto_trader.is_running is False

    @pytest.mark.asyncio
    async def test_get_account_balance(self, auto_trader):
        """Test getting account balance"""
        # Mock exchange balance
        auto_trader.exchanges['binance'].fetch_balance = AsyncMock(return_value={
            'USDT': {'free': 10000.0, 'used': 0.0, 'total': 10000.0}
        })

        balance = await auto_trader.get_account_balance('binance')
        assert balance == 10000.0

    @pytest.mark.asyncio
    async def test_get_account_balance_error(self, auto_trader):
        """Test getting account balance with error"""
        # Mock exchange error
        auto_trader.exchanges['binance'].fetch_balance = AsyncMock(
            side_effect=Exception("API Error")
        )

        balance = await auto_trader.get_account_balance('binance')
        assert balance == 0.0  # Returns 0 on error

    def test_get_active_trades_count(self, auto_trader):
        """Test counting active trades"""
        # Add some trades
        auto_trader.active_trades = {
            'trade1': TradeExecution(
                trade_id="trade1",
                symbol="BTC/USDT",
                exchange="binance",
                side="buy",
                amount=0.1,
                price=50000.0,
                timestamp=datetime.now(),
                profit_target=51000.0,
                stop_loss=49000.0,
                strategy="test",
                status="open"
            ),
            'trade2': TradeExecution(
                trade_id="trade2",
                symbol="ETH/USDT",
                exchange="binance",
                side="buy",
                amount=1.0,
                price=3000.0,
                timestamp=datetime.now(),
                profit_target=3100.0,
                stop_loss=2900.0,
                strategy="test",
                status="open"
            )
        }

        assert auto_trader.get_active_trades_count() == 2

    def test_record_trade_profit(self, auto_trader):
        """Test recording trade profit"""
        initial_profit = auto_trader.risk_manager.daily_profit

        auto_trader.record_trade_profit(50.0)

        assert auto_trader.risk_manager.daily_profit == initial_profit + 50.0
        assert auto_trader.risk_manager.daily_trades == 1

    def test_record_trade_loss(self, auto_trader):
        """Test recording trade loss"""
        initial_profit = auto_trader.risk_manager.daily_profit

        auto_trader.record_trade_profit(-30.0)

        assert auto_trader.risk_manager.daily_profit == initial_profit - 30.0
        assert auto_trader.risk_manager.daily_trades == 1

    def test_get_status(self, auto_trader):
        """Test getting auto trader status"""
        status = auto_trader.get_status()

        assert 'enabled' in status
        assert 'running' in status
        assert 'active_trades' in status
        assert 'daily_profit' in status
        assert 'daily_trades' in status


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_balance_position_size(self, risk_manager):
        """Test position size calculation with zero balance"""
        position_size = risk_manager.calculate_position_size(0.0, risk_percent=2.0)
        assert position_size == 0.0

    def test_negative_balance_position_size(self, risk_manager):
        """Test position size calculation with negative balance"""
        position_size = risk_manager.calculate_position_size(-1000.0, risk_percent=2.0)
        # Should return 0 or handle gracefully
        assert position_size <= 0.0

    def test_missing_pair_data_fields(self, risk_manager):
        """Test should_trade with missing pair data fields"""
        pair_data = {}  # Empty

        should_trade, reason = risk_manager.should_trade(
            pair_data=pair_data,
            balance=10000.0,
            active_trades=2
        )

        # Should handle missing fields gracefully
        assert isinstance(should_trade, bool)
        assert isinstance(reason, str)

    def test_very_high_risk_percent(self, risk_manager):
        """Test position size with very high risk percent"""
        position_size = risk_manager.calculate_position_size(10000.0, risk_percent=50.0)
        # Should still be capped at max_position_size
        assert position_size == 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
