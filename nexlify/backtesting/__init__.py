"""Backtesting and paper trading systems."""

from nexlify.backtesting.nexlify_backtester import StrategyBacktester
from nexlify.backtesting.nexlify_paper_trading import PaperTradingEngine
from nexlify.backtesting.backtest_phase1_phase2_integration import (
    MockExchange,
    TradingScenario,
)
from nexlify.backtesting.nexlify_paper_trading_orchestrator import (
    PaperTradingOrchestrator,
    AgentConfig,
    PerformanceSnapshot,
    create_orchestrator,
)
from nexlify.backtesting.nexlify_paper_trading_runner import PaperTradingRunner

# Backward compatibility aliases
Backtester = StrategyBacktester
PaperTrading = PaperTradingEngine

__all__ = [
    "StrategyBacktester",
    "Backtester",
    "PaperTradingEngine",
    "PaperTrading",
    "MockExchange",
    "TradingScenario",
    "PaperTradingOrchestrator",
    "AgentConfig",
    "PerformanceSnapshot",
    "create_orchestrator",
    "PaperTradingRunner",
]
