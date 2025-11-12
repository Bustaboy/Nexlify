"""Backtesting and paper trading systems."""

from nexlify.backtesting.nexlify_backtester import StrategyBacktester
from nexlify.backtesting.nexlify_paper_trading import PaperTradingEngine
from nexlify.backtesting.backtest_phase1_phase2_integration import (
    MockExchange,
    TradingScenario
)

# Backward compatibility aliases
Backtester = StrategyBacktester
PaperTrading = PaperTradingEngine

__all__ = [
    'StrategyBacktester',
    'Backtester',
    'PaperTradingEngine',
    'PaperTrading',
    'MockExchange',
    'TradingScenario',
]
