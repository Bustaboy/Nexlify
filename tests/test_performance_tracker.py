#!/usr/bin/env python3
"""
Unit tests for Nexlify Performance Tracker
Comprehensive testing of performance tracking functionality
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import json
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify_performance_tracker import (
    PerformanceTracker,
    PerformanceMetrics,
    Trade
)


@pytest.fixture
def tracker(tmp_path):
    """Create a performance tracker with temporary database"""
    config = {
        'performance_tracking': {
            'enabled': True,
            'database_path': str(tmp_path / 'test_trading.db'),
            'calculate_sharpe_ratio': True,
            'risk_free_rate': 0.02,
            'track_drawdown': True
        }
    }
    return PerformanceTracker(config)


class TestPerformanceTrackerInitialization:
    """Test performance tracker initialization"""

    def test_initialization(self, tracker):
        """Test proper initialization"""
        assert tracker.enabled is True
        assert tracker.calculate_sharpe is True
        assert tracker.risk_free_rate == 0.02
        assert tracker.track_drawdown is True
        assert tracker.db_path.exists()

    def test_disabled_tracker(self, tmp_path):
        """Test disabled tracker"""
        config = {
            'performance_tracking': {'enabled': False}
        }
        tracker = PerformanceTracker(config)
        assert tracker.enabled is False


class TestTradeRecording:
    """Test trade recording functionality"""

    def test_record_closed_trade(self, tracker):
        """Test recording a closed trade"""
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            entry_price=50000,
            exit_price=51000,
            fee=10.0,
            strategy="test"
        )

        assert trade_id is not None
        assert isinstance(trade_id, int)

    def test_record_open_trade(self, tracker):
        """Test recording an open trade"""
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="ETH/USDT",
            side="buy",
            quantity=1.0,
            entry_price=3000,
            exit_price=None,  # Still open
            fee=0.0
        )

        assert trade_id is not None

    def test_record_winning_trade(self, tracker):
        """Test recording a winning trade"""
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            entry_price=50000,
            exit_price=52000,
            fee=100.0
        )

        assert trade_id is not None

        # Verify P&L calculation
        metrics = tracker.get_performance_metrics()
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.total_pnl > 0

    def test_record_losing_trade(self, tracker):
        """Test recording a losing trade"""
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            entry_price=50000,
            exit_price=48000,
            fee=100.0
        )

        assert trade_id is not None

        # Verify P&L calculation
        metrics = tracker.get_performance_metrics()
        assert metrics.total_trades == 1
        assert metrics.losing_trades == 1
        assert metrics.total_pnl < 0

    def test_sell_trade_pnl(self, tracker):
        """Test P&L calculation for sell orders"""
        # Short trade (sell high, buy low)
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="sell",
            quantity=1.0,
            entry_price=50000,
            exit_price=48000,  # Profit on short
            fee=100.0
        )

        metrics = tracker.get_performance_metrics()
        assert metrics.total_pnl > 0  # Should be profitable


class TestTradeUpdating:
    """Test trade updating functionality"""

    def test_update_open_trade(self, tracker):
        """Test updating an open trade with exit price"""
        # Record open trade
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            entry_price=50000,
            exit_price=None,
            fee=50.0
        )

        # Update with exit price
        tracker.update_trade(trade_id, exit_price=52000)

        # Verify update
        metrics = tracker.get_performance_metrics()
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1

    def test_update_nonexistent_trade(self, tracker):
        """Test updating a trade that doesn't exist"""
        # Should not crash
        tracker.update_trade(999999, exit_price=50000)


class TestPerformanceMetrics:
    """Test performance metrics calculation"""

    def test_basic_metrics(self, tracker):
        """Test basic metrics calculation"""
        # Record multiple trades
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 52000, 100)
        tracker.record_trade("binance", "ETH/USDT", "buy", 1.0, 3000, 2900, 30)
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 51000, 53000, 100)

        metrics = tracker.get_performance_metrics()

        assert metrics.total_trades == 3
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 1
        assert metrics.win_rate == pytest.approx(66.67, rel=0.1)

    def test_win_rate_calculation(self, tracker):
        """Test win rate calculation"""
        # 3 wins, 2 losses = 60% win rate
        for i in range(3):
            tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 50)

        for i in range(2):
            tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 49000, 50)

        metrics = tracker.get_performance_metrics()
        assert metrics.win_rate == 60.0

    def test_profit_metrics(self, tracker):
        """Test profit metrics calculation"""
        # Record trades with known P&L
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)  # $1000 win
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 49000, 0)  # $1000 loss

        metrics = tracker.get_performance_metrics()

        assert metrics.average_win == 1000.0
        assert metrics.average_loss == 1000.0
        assert metrics.profit_factor == 1.0
        assert metrics.best_trade == 1000.0
        assert metrics.worst_trade == -1000.0

    def test_profit_factor_calculation(self, tracker):
        """Test profit factor calculation"""
        # Total wins: $3000, Total losses: $1000 = PF 3.0
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 52000, 0)  # $2000
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)  # $1000
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 49000, 0)  # -$1000

        metrics = tracker.get_performance_metrics()
        assert metrics.profit_factor == 3.0

    def test_empty_metrics(self, tracker):
        """Test metrics with no trades"""
        metrics = tracker.get_performance_metrics()

        assert metrics.total_trades == 0
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 0.0


class TestSharpeRatio:
    """Test Sharpe ratio calculation"""

    def test_sharpe_ratio_calculation(self, tracker):
        """Test Sharpe ratio with multiple trades"""
        # Record trades with varying returns
        returns = [2, -1, 3, 1, -0.5, 2.5]

        for ret in returns:
            entry = 50000
            exit = entry * (1 + ret / 100)
            tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, entry, exit, 0)

        metrics = tracker.get_performance_metrics()

        # Should have a Sharpe ratio calculated
        assert metrics.sharpe_ratio != 0.0

    def test_sharpe_with_consistent_returns(self, tracker):
        """Test Sharpe ratio with consistent positive returns"""
        # All positive returns should give good Sharpe
        for i in range(10):
            tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        metrics = tracker.get_performance_metrics()
        assert metrics.sharpe_ratio > 0

    def test_sharpe_with_single_trade(self, tracker):
        """Test Sharpe ratio with insufficient data"""
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        metrics = tracker.get_performance_metrics()
        # Should return 0 with only one trade
        assert metrics.sharpe_ratio == 0.0


class TestMaxDrawdown:
    """Test maximum drawdown calculation"""

    def test_max_drawdown_calculation(self, tracker):
        """Test max drawdown with known sequence"""
        # Create a drawdown scenario:
        # +1000, +1000, -500, -1000, +500
        # Equity: 0 -> 1000 -> 2000 -> 1500 -> 500 -> 1000
        # Max drawdown: 2000 - 500 = 1500

        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)  # +1000
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)  # +1000
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 49500, 0)  # -500
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 49000, 0)  # -1000
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 50500, 0)  # +500

        metrics = tracker.get_performance_metrics()

        assert metrics.max_drawdown == 1500.0
        assert metrics.max_drawdown_percent == pytest.approx(75.0, rel=0.01)

    def test_no_drawdown(self, tracker):
        """Test max drawdown with only winning trades"""
        # All wins, no drawdown
        for i in range(5):
            tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        metrics = tracker.get_performance_metrics()
        assert metrics.max_drawdown == 0.0


class TestFiltering:
    """Test metrics filtering by date, exchange, symbol"""

    def test_date_filtering(self, tracker):
        """Test filtering by date range"""
        # Record trades at different times
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        # Get metrics for future date range
        future_start = datetime.now() + timedelta(days=1)
        future_end = datetime.now() + timedelta(days=2)

        metrics = tracker.get_performance_metrics(
            start_date=future_start,
            end_date=future_end
        )

        # Should have no trades
        assert metrics.total_trades == 0

    def test_exchange_filtering(self, tracker):
        """Test filtering by exchange"""
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)
        tracker.record_trade("kraken", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        metrics = tracker.get_performance_metrics(exchange="binance")

        assert metrics.total_trades == 1

    def test_symbol_filtering(self, tracker):
        """Test filtering by symbol"""
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)
        tracker.record_trade("binance", "ETH/USDT", "buy", 1.0, 3000, 3100, 0)

        metrics = tracker.get_performance_metrics(symbol="BTC/USDT")

        assert metrics.total_trades == 1


class TestExporting:
    """Test trade export functionality"""

    def test_export_json(self, tracker, tmp_path):
        """Test exporting trades to JSON"""
        # Record some trades
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)
        tracker.record_trade("binance", "ETH/USDT", "buy", 1.0, 3000, 3100, 0)

        # Export
        export_path = tmp_path / "trades.json"
        success = tracker.export_trades(str(export_path), format="json")

        assert success is True
        assert export_path.exists()

        # Verify content
        with open(export_path) as f:
            data = json.load(f)

        assert len(data) == 2

    def test_export_csv(self, tracker, tmp_path):
        """Test exporting trades to CSV"""
        # Record some trades
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        # Export
        export_path = tmp_path / "trades.csv"
        success = tracker.export_trades(str(export_path), format="csv")

        assert success is True
        assert export_path.exists()

        # Verify content
        with open(export_path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) >= 2  # Header + at least 1 trade

    def test_export_with_filters(self, tracker, tmp_path):
        """Test export with date filtering"""
        tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        export_path = tmp_path / "trades.json"
        future_date = datetime.now() + timedelta(days=1)

        success = tracker.export_trades(
            str(export_path),
            format="json",
            start_date=future_date
        )

        assert success is True

        with open(export_path) as f:
            data = json.load(f)

        # Should be empty due to date filter
        assert len(data) == 0


class TestTradesSummary:
    """Test trades summary functionality"""

    def test_get_recent_trades(self, tracker):
        """Test getting recent trades summary"""
        # Record trades
        for i in range(5):
            tracker.record_trade("binance", "BTC/USDT", "buy", 1.0, 50000, 51000, 0)

        summary = tracker.get_trades_summary(limit=3)

        assert len(summary) == 3
        assert 'timestamp' in summary[0]
        assert 'symbol' in summary[0]
        assert 'pnl' in summary[0]

    def test_empty_summary(self, tracker):
        """Test summary with no trades"""
        summary = tracker.get_trades_summary()

        assert len(summary) == 0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_quantity_trade(self, tracker):
        """Test trade with zero quantity"""
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.0,
            entry_price=50000,
            exit_price=51000,
            fee=0.0
        )

        assert trade_id is not None

    def test_negative_price(self, tracker):
        """Test handling negative prices"""
        # Should still record (validation is application layer)
        trade_id = tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            entry_price=-50000,
            exit_price=51000,
            fee=0.0
        )

        assert trade_id is not None

    def test_very_large_pnl(self, tracker):
        """Test handling very large P&L"""
        tracker.record_trade(
            exchange="binance",
            symbol="BTC/USDT",
            side="buy",
            quantity=1000.0,
            entry_price=50000,
            exit_price=100000,
            fee=0.0
        )

        metrics = tracker.get_performance_metrics()
        assert metrics.total_pnl == 50000000.0  # $50M profit


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
