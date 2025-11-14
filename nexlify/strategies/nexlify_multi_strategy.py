#!/usr/bin/env python3
"""
Nexlify Multi-Strategy Optimizer
Manages and optimizes multiple trading strategies
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class StrategyMetrics:
    """Metrics for a trading strategy"""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_profit_per_trade: float = 0.0


class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str, config: Dict = None):
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.metrics = StrategyMetrics()
        self.trade_history: List[Dict] = []

    async def analyze(self, market_data: Dict) -> Optional[Dict]:
        """
        Analyze market data and return trading signal

        Returns:
            Trading signal dictionary or None
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    def update_metrics(self, trade_result: Dict):
        """Update strategy metrics with trade result"""
        self.metrics.total_trades += 1

        if trade_result.get("profit", 0) > 0:
            self.metrics.winning_trades += 1
            self.metrics.total_profit += trade_result["profit"]
        else:
            self.metrics.losing_trades += 1
            self.metrics.total_loss += abs(trade_result["profit"])

        # Calculate derived metrics
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = (
                self.metrics.winning_trades / self.metrics.total_trades
            )

        if self.metrics.total_loss > 0:
            self.metrics.profit_factor = (
                self.metrics.total_profit / self.metrics.total_loss
            )

        if self.metrics.total_trades > 0:
            self.metrics.avg_profit_per_trade = (
                self.metrics.total_profit - self.metrics.total_loss
            ) / self.metrics.total_trades

        self.trade_history.append(trade_result)


class ArbitrageStrategy(TradingStrategy):
    """Arbitrage trading strategy"""

    def __init__(self, config: Dict = None):
        super().__init__("Arbitrage Scanner", config)
        self.min_profit_threshold = config.get("min_profit_percent", 0.5)

    async def analyze(self, market_data: Dict) -> Optional[Dict]:
        """Analyze for arbitrage opportunities"""
        try:
            if "arbitrage_opportunities" not in market_data:
                return None

            opportunities = market_data["arbitrage_opportunities"]

            for opp in opportunities:
                if opp["profit_percent"] >= self.min_profit_threshold:
                    return {
                        "strategy": self.name,
                        "signal": "arbitrage",
                        "symbol": opp["symbol"],
                        "buy_exchange": opp["buy_exchange"],
                        "sell_exchange": opp["sell_exchange"],
                        "expected_profit": opp["profit_percent"],
                        "confidence": 0.9,
                    }

            return None

        except Exception as e:
            logger.error(f"Arbitrage analysis error: {e}")
            return None


class MomentumStrategy(TradingStrategy):
    """Momentum trading strategy"""

    def __init__(self, config: Dict = None):
        super().__init__("Momentum Trading", config)
        self.momentum_threshold = config.get("momentum_threshold", 2.0)

    async def analyze(self, market_data: Dict) -> Optional[Dict]:
        """Analyze for momentum signals"""
        try:
            if "indicators" not in market_data:
                return None

            indicators = market_data["indicators"]
            rsi = indicators.get("rsi", 50)
            macd_histogram = indicators.get("macd_histogram", 0)

            # Strong upward momentum
            if rsi > 60 and macd_histogram > 0:
                return {
                    "strategy": self.name,
                    "signal": "buy",
                    "symbol": market_data["symbol"],
                    "reason": "Strong upward momentum",
                    "confidence": min((rsi - 50) / 50, 0.9),
                }

            # Strong downward momentum
            elif rsi < 40 and macd_histogram < 0:
                return {
                    "strategy": self.name,
                    "signal": "sell",
                    "symbol": market_data["symbol"],
                    "reason": "Strong downward momentum",
                    "confidence": min((50 - rsi) / 50, 0.9),
                }

            return None

        except Exception as e:
            logger.error(f"Momentum analysis error: {e}")
            return None


class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy"""

    def __init__(self, config: Dict = None):
        super().__init__("Mean Reversion", config)
        self.bollinger_std = config.get("bollinger_std", 2.0)

    async def analyze(self, market_data: Dict) -> Optional[Dict]:
        """Analyze for mean reversion opportunities"""
        try:
            if "indicators" not in market_data:
                return None

            indicators = market_data["indicators"]
            current_price = market_data.get("current_price")

            if not current_price:
                return None

            bb_upper = indicators.get("bb_upper")
            bb_lower = indicators.get("bb_lower")
            bb_middle = indicators.get("bb_middle")

            if not all([bb_upper, bb_lower, bb_middle]):
                return None

            # Price touched lower band - potential buy
            if current_price <= bb_lower:
                return {
                    "strategy": self.name,
                    "signal": "buy",
                    "symbol": market_data["symbol"],
                    "reason": "Price at lower Bollinger Band",
                    "target_price": bb_middle,
                    "confidence": 0.7,
                }

            # Price touched upper band - potential sell
            elif current_price >= bb_upper:
                return {
                    "strategy": self.name,
                    "signal": "sell",
                    "symbol": market_data["symbol"],
                    "reason": "Price at upper Bollinger Band",
                    "target_price": bb_middle,
                    "confidence": 0.7,
                }

            return None

        except Exception as e:
            logger.error(f"Mean reversion analysis error: {e}")
            return None


class MarketMakingStrategy(TradingStrategy):
    """Market making strategy"""

    def __init__(self, config: Dict = None):
        super().__init__("Market Making", config)
        self.spread_target = config.get("spread_target", 0.002)  # 0.2%

    async def analyze(self, market_data: Dict) -> Optional[Dict]:
        """Analyze for market making opportunities"""
        try:
            current_spread = market_data.get("spread", 0)
            volume = market_data.get("volume_24h", 0)

            # Only market make on liquid pairs with reasonable spread
            if current_spread < self.spread_target * 2 and volume > 1000000:
                return {
                    "strategy": self.name,
                    "signal": "market_make",
                    "symbol": market_data["symbol"],
                    "bid_offset": -self.spread_target / 2,
                    "ask_offset": self.spread_target / 2,
                    "confidence": 0.6,
                }

            return None

        except Exception as e:
            logger.error(f"Market making analysis error: {e}")
            return None


class MultiStrategyOptimizer:
    """
    Manages and optimizes multiple trading strategies
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategies: Dict[str, TradingStrategy] = {}
        self.active_signals: List[Dict] = []

        # Initialize default strategies
        self._initialize_strategies()

        logger.info("📊 Multi-Strategy Optimizer initialized")

    def _initialize_strategies(self):
        """Initialize default trading strategies"""
        try:
            # Add arbitrage strategy
            self.strategies["arbitrage"] = ArbitrageStrategy(
                self.config.get("arbitrage", {})
            )

            # Add momentum strategy
            self.strategies["momentum"] = MomentumStrategy(
                self.config.get("momentum", {})
            )

            # Add mean reversion strategy
            self.strategies["mean_reversion"] = MeanReversionStrategy(
                self.config.get("mean_reversion", {})
            )

            # Add market making strategy
            self.strategies["market_making"] = MarketMakingStrategy(
                self.config.get("market_making", {})
            )

            logger.info(f"Initialized {len(self.strategies)} trading strategies")

        except Exception as e:
            error_handler.log_error(
                e, "Strategy initialization failed", severity="error"
            )

    async def analyze_market(self, market_data: Dict) -> List[Dict]:
        """
        Run all enabled strategies on market data

        Returns:
            List of trading signals from all strategies
        """
        signals = []

        for strategy_name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue

            try:
                signal = await strategy.analyze(market_data)
                if signal:
                    signal["strategy_name"] = strategy_name
                    signal["timestamp"] = datetime.now().isoformat()
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy_name}: {e}")

        return signals

    def get_best_signal(self, signals: List[Dict]) -> Optional[Dict]:
        """
        Select the best signal from multiple strategies

        Returns:
            The highest confidence signal
        """
        if not signals:
            return None

        # Sort by confidence
        sorted_signals = sorted(
            signals, key=lambda s: s.get("confidence", 0), reverse=True
        )

        return sorted_signals[0]

    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            logger.info(f"Strategy enabled: {strategy_name}")
            return True
        return False

    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            logger.info(f"Strategy disabled: {strategy_name}")
            return True
        return False

    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """Get metrics for a specific strategy"""
        if strategy_name in self.strategies:
            return self.strategies[strategy_name].metrics
        return None

    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """Get metrics for all strategies"""
        return {name: strategy.metrics for name, strategy in self.strategies.items()}

    def record_trade_result(self, strategy_name: str, trade_result: Dict):
        """Record a trade result for a strategy"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].update_metrics(trade_result)

    def get_performance_report(self) -> Dict:
        """Generate performance report for all strategies"""
        report = {"timestamp": datetime.now().isoformat(), "strategies": {}}

        for name, strategy in self.strategies.items():
            metrics = strategy.metrics
            report["strategies"][name] = {
                "enabled": strategy.enabled,
                "total_trades": metrics.total_trades,
                "win_rate": f"{metrics.win_rate * 100:.2f}%",
                "profit_factor": f"{metrics.profit_factor:.2f}",
                "total_profit": f"${metrics.total_profit:.2f}",
                "avg_profit_per_trade": f"${metrics.avg_profit_per_trade:.2f}",
            }

        return report

    def optimize_strategy_weights(self):
        """
        Optimize strategy weights based on performance
        (Placeholder for future ML-based optimization)
        """
        # This would implement a more sophisticated optimization algorithm
        # For now, just disable strategies with very poor performance
        for name, strategy in self.strategies.items():
            if strategy.metrics.total_trades >= 10:
                if strategy.metrics.win_rate < 0.3:
                    logger.warning(
                        f"Strategy {name} has low win rate, consider disabling"
                    )
                if strategy.metrics.profit_factor < 0.5:
                    logger.warning(
                        f"Strategy {name} has poor profit factor, consider disabling"
                    )
