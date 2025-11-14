#!/usr/bin/env python3
"""
Unit tests for Nexlify Financial Modules
Testing profit management, portfolio rebalancing, and tax reporting
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.financial.nexlify_profit_manager import ProfitManager
from nexlify.financial.nexlify_portfolio_rebalancer import PortfolioRebalancer
from nexlify.financial.nexlify_tax_reporter import TaxReporter


class TestProfitManager:
    """Test Profit Manager functionality"""

    @pytest.fixture
    def profit_manager(self):
        """Create profit manager"""
        config = {
            "profit_tracking": {
                "enabled": True,
                "target_profit_percent": 10.0,
                "stop_loss_percent": 5.0,
            }
        }
        return ProfitManager(config)

    def test_initialization(self, profit_manager):
        """Test profit manager initialization"""
        assert profit_manager is not None
        assert hasattr(profit_manager, "calculate_profit")

    def test_calculate_profit_buy_win(self, profit_manager):
        """Test profit calculation for winning buy trade"""
        entry_price = 50000.0
        exit_price = 52000.0
        quantity = 0.1

        profit = profit_manager.calculate_profit(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side="buy",
        )

        # 0.1 BTC * $2000 gain = $200 profit
        assert profit == 200.0

    def test_calculate_profit_buy_loss(self, profit_manager):
        """Test profit calculation for losing buy trade"""
        entry_price = 50000.0
        exit_price = 48000.0
        quantity = 0.1

        profit = profit_manager.calculate_profit(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side="buy",
        )

        # 0.1 BTC * -$2000 = -$200 loss
        assert profit == -200.0

    def test_calculate_profit_with_fees(self, profit_manager):
        """Test profit calculation including fees"""
        entry_price = 50000.0
        exit_price = 52000.0
        quantity = 0.1
        fee = 10.0

        profit = profit_manager.calculate_profit(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side="buy",
            fee=fee,
        )

        # $200 profit - $10 fee = $190
        assert profit == 190.0

    def test_calculate_profit_sell_short(self, profit_manager):
        """Test profit for short position"""
        entry_price = 50000.0
        exit_price = 48000.0  # Price went down
        quantity = 0.1

        profit = profit_manager.calculate_profit(
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            side="sell",
        )

        # Profit from shorting: sold at 50k, bought back at 48k
        assert profit == 200.0

    def test_should_take_profit(self, profit_manager):
        """Test profit target detection"""
        current_price = 55000.0  # 10% gain
        entry_price = 50000.0

        should_take = profit_manager.should_take_profit(
            current_price=current_price, entry_price=entry_price, side="buy"
        )

        assert should_take is True

    def test_should_stop_loss(self, profit_manager):
        """Test stop loss detection"""
        current_price = 47500.0  # 5% loss
        entry_price = 50000.0

        should_stop = profit_manager.should_stop_loss(
            current_price=current_price, entry_price=entry_price, side="buy"
        )

        assert should_stop is True

    def test_calculate_roi(self, profit_manager):
        """Test ROI calculation"""
        profit = 500.0
        investment = 10000.0

        roi = profit_manager.calculate_roi(profit, investment)

        assert roi == 5.0  # 5% ROI

    def test_get_profit_summary(self, profit_manager):
        """Test profit summary generation"""
        # Record some profits
        profit_manager.record_profit(100.0, "BTC/USDT", datetime.now())
        profit_manager.record_profit(50.0, "ETH/USDT", datetime.now())
        profit_manager.record_profit(-30.0, "BTC/USDT", datetime.now())

        summary = profit_manager.get_profit_summary()

        assert "total_profit" in summary
        assert "winning_trades" in summary
        assert "losing_trades" in summary
        assert summary["total_profit"] == 120.0


class TestPortfolioRebalancer:
    """Test Portfolio Rebalancer functionality"""

    @pytest.fixture
    def rebalancer(self):
        """Create portfolio rebalancer"""
        config = {
            "rebalancing": {
                "enabled": True,
                "threshold_percent": 5.0,
                "target_allocations": {"BTC": 50.0, "ETH": 30.0, "USDT": 20.0},
            }
        }
        return PortfolioRebalancer(config)

    def test_initialization(self, rebalancer):
        """Test rebalancer initialization"""
        assert rebalancer is not None
        assert hasattr(rebalancer, "calculate_rebalance")

    def test_calculate_current_allocation(self, rebalancer):
        """Test current allocation calculation"""
        portfolio = {
            "BTC": {"value": 5000.0},
            "ETH": {"value": 3000.0},
            "USDT": {"value": 2000.0},
        }

        allocations = rebalancer.calculate_current_allocation(portfolio)

        assert allocations["BTC"] == 50.0
        assert allocations["ETH"] == 30.0
        assert allocations["USDT"] == 20.0

    def test_needs_rebalancing_yes(self, rebalancer):
        """Test rebalancing needed detection"""
        current_allocation = {"BTC": 60.0, "ETH": 25.0, "USDT": 15.0}  # 10% off target

        needs_rebal = rebalancer.needs_rebalancing(current_allocation)

        assert needs_rebal is True

    def test_needs_rebalancing_no(self, rebalancer):
        """Test when rebalancing not needed"""
        current_allocation = {
            "BTC": 51.0,  # Within 5% threshold
            "ETH": 29.0,
            "USDT": 20.0,
        }

        needs_rebal = rebalancer.needs_rebalancing(current_allocation)

        assert needs_rebal is False

    def test_calculate_rebalance_trades(self, rebalancer):
        """Test rebalance trade calculation"""
        portfolio = {
            "BTC": {"value": 6000.0, "quantity": 0.12},
            "ETH": {"value": 2000.0, "quantity": 1.0},
            "USDT": {"value": 2000.0, "quantity": 2000.0},
        }

        trades = rebalancer.calculate_rebalance_trades(portfolio)

        assert isinstance(trades, list)
        # Should have trades to rebalance
        assert len(trades) > 0

    def test_get_rebalancing_report(self, rebalancer):
        """Test rebalancing report generation"""
        portfolio = {
            "BTC": {"value": 5000.0, "quantity": 0.1},
            "ETH": {"value": 3000.0, "quantity": 1.5},
            "USDT": {"value": 2000.0, "quantity": 2000.0},
        }

        report = rebalancer.get_rebalancing_report(portfolio)

        assert "current_allocation" in report
        assert "target_allocation" in report
        assert "needs_rebalancing" in report


class TestTaxReporter:
    """Test Tax Reporter functionality"""

    @pytest.fixture
    def tax_reporter(self, tmp_path):
        """Create tax reporter"""
        config = {
            "tax_reporting": {"enabled": True, "jurisdiction": "US", "tax_rate": 0.25}
        }
        reporter = TaxReporter(config)
        reporter.data_path = tmp_path
        return reporter

    def test_initialization(self, tax_reporter):
        """Test tax reporter initialization"""
        assert tax_reporter is not None
        assert hasattr(tax_reporter, "calculate_tax_liability")

    def test_record_taxable_event(self, tax_reporter):
        """Test recording taxable event"""
        event = {
            "type": "trade",
            "asset": "BTC",
            "quantity": 0.1,
            "cost_basis": 50000.0,
            "proceeds": 52000.0,
            "date": datetime.now(),
        }

        tax_reporter.record_taxable_event(event)

        assert len(tax_reporter.taxable_events) == 1

    def test_calculate_capital_gain(self, tax_reporter):
        """Test capital gain calculation"""
        cost_basis = 50000.0
        proceeds = 52000.0

        gain = tax_reporter.calculate_capital_gain(cost_basis, proceeds)

        assert gain == 2000.0

    def test_calculate_capital_loss(self, tax_reporter):
        """Test capital loss calculation"""
        cost_basis = 50000.0
        proceeds = 48000.0

        gain = tax_reporter.calculate_capital_gain(cost_basis, proceeds)

        assert gain == -2000.0

    def test_calculate_tax_liability(self, tax_reporter):
        """Test tax liability calculation"""
        # Record some gains
        tax_reporter.record_taxable_event(
            {
                "type": "trade",
                "asset": "BTC",
                "cost_basis": 50000.0,
                "proceeds": 52000.0,
                "date": datetime.now(),
            }
        )

        liability = tax_reporter.calculate_tax_liability()

        # $2000 gain * 25% = $500
        assert liability == 500.0

    def test_generate_tax_report(self, tax_reporter):
        """Test tax report generation"""
        # Record events
        tax_reporter.record_taxable_event(
            {
                "type": "trade",
                "asset": "BTC",
                "cost_basis": 50000.0,
                "proceeds": 52000.0,
                "date": datetime.now(),
            }
        )

        report = tax_reporter.generate_tax_report(year=2024)

        assert "total_gains" in report
        assert "total_losses" in report
        assert "tax_liability" in report
        assert "year" in report

    def test_export_tax_report(self, tax_reporter, tmp_path):
        """Test exporting tax report"""
        # Record event
        tax_reporter.record_taxable_event(
            {
                "type": "trade",
                "asset": "BTC",
                "cost_basis": 50000.0,
                "proceeds": 52000.0,
                "date": datetime.now(),
            }
        )

        export_path = tmp_path / "tax_report_2024.csv"
        success = tax_reporter.export_report(str(export_path), format="csv")

        assert success is True
        assert export_path.exists()

    def test_wash_sale_detection(self, tax_reporter):
        """Test wash sale rule detection"""
        # Sell at a loss
        sell_event = {
            "type": "sell",
            "asset": "BTC",
            "quantity": 0.1,
            "cost_basis": 50000.0,
            "proceeds": 48000.0,
            "date": datetime.now(),
        }

        # Buy back within 30 days
        buy_event = {
            "type": "buy",
            "asset": "BTC",
            "quantity": 0.1,
            "price": 47000.0,
            "date": datetime.now() + timedelta(days=10),
        }

        is_wash_sale = tax_reporter.check_wash_sale(sell_event, buy_event)

        assert is_wash_sale is True


class TestIntegration:
    """Integration tests for financial modules"""

    def test_profit_and_tax_integration(self):
        """Test integration between profit manager and tax reporter"""
        profit_config = {"profit_tracking": {"enabled": True}}
        tax_config = {"tax_reporting": {"enabled": True, "tax_rate": 0.25}}

        profit_mgr = ProfitManager(profit_config)
        tax_reporter = TaxReporter(tax_config)

        # Record profitable trade
        profit = profit_mgr.calculate_profit(
            entry_price=50000.0, exit_price=52000.0, quantity=0.1, side="buy"
        )

        # Record for tax
        tax_reporter.record_taxable_event(
            {
                "type": "trade",
                "asset": "BTC",
                "cost_basis": 50000.0 * 0.1,
                "proceeds": 52000.0 * 0.1,
                "date": datetime.now(),
            }
        )

        # Calculate tax
        tax_liability = tax_reporter.calculate_tax_liability()

        # Tax should be 25% of profit
        assert tax_liability == profit * 0.25


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_quantity_profit(self):
        """Test profit calculation with zero quantity"""
        config = {"profit_tracking": {"enabled": True}}
        pm = ProfitManager(config)

        profit = pm.calculate_profit(
            entry_price=50000.0, exit_price=52000.0, quantity=0.0, side="buy"
        )

        assert profit == 0.0

    def test_negative_prices(self):
        """Test handling negative prices"""
        config = {"profit_tracking": {"enabled": True}}
        pm = ProfitManager(config)

        # Should handle or raise appropriate error
        try:
            profit = pm.calculate_profit(
                entry_price=-50000.0, exit_price=52000.0, quantity=0.1, side="buy"
            )
            assert isinstance(profit, (int, float))
        except Exception:
            pass  # Acceptable to raise error

    def test_empty_portfolio_rebalance(self):
        """Test rebalancing with empty portfolio"""
        config = {
            "rebalancing": {"enabled": True, "target_allocations": {"BTC": 100.0}}
        }
        rebalancer = PortfolioRebalancer(config)

        portfolio = {}
        allocations = rebalancer.calculate_current_allocation(portfolio)

        assert isinstance(allocations, dict)

    def test_tax_report_no_events(self):
        """Test tax report with no taxable events"""
        config = {"tax_reporting": {"enabled": True, "tax_rate": 0.25}}
        reporter = TaxReporter(config)

        liability = reporter.calculate_tax_liability()

        assert liability == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
