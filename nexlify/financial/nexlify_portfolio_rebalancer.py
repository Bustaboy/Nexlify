#!/usr/bin/env python3
"""
Nexlify Portfolio Rebalancing
Automated portfolio rebalancing to maintain target allocations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class PortfolioRebalancer:
    """
    Automated portfolio rebalancing engine
    Maintains target asset allocations
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Handle both top-level config and nested "rebalancing" config
        rebalancing_config = self.config.get("rebalancing", self.config)

        # Target allocations (symbol: percentage)
        self.target_allocations = rebalancing_config.get(
            "target_allocations",
            {
                "BTC/USDT": 50.0,  # 50% BTC
                "ETH/USDT": 30.0,  # 30% ETH
                "USDT": 20.0,  # 20% cash
            },
        )

        # Rebalancing parameters (support both decimal and percentage formats)
        threshold = rebalancing_config.get("rebalance_threshold", rebalancing_config.get("threshold_percent", 5.0))
        # Convert to percentage if it's a decimal
        self.rebalance_threshold = threshold if threshold > 1 else threshold * 100

        self.rebalance_interval = rebalancing_config.get("rebalance_interval_hours", 24)
        self.min_trade_size = rebalancing_config.get("min_trade_size", 10)  # $10 minimum

        # State
        self.last_rebalance = None
        self.rebalance_count = 0

        logger.info(
            f"⚖️ Portfolio Rebalancer initialized (threshold: {self.rebalance_threshold:.1f}%)"
        )

    @handle_errors("Portfolio Rebalancing", reraise=False)
    async def check_and_rebalance(
        self,
        neural_net,
        current_holdings: Dict[str, float],
        current_prices: Dict[str, float],
        total_value: float,
    ) -> Dict:
        """
        Check if rebalancing is needed and execute if necessary

        Args:
            neural_net: Trading engine instance
            current_holdings: Dict of symbol: amount
            current_prices: Dict of symbol: current_price
            total_value: Total portfolio value in USD

        Returns:
            Rebalancing result dict
        """
        # Check if enough time has passed
        if self.last_rebalance:
            time_since = datetime.now() - self.last_rebalance
            if time_since < timedelta(hours=self.rebalance_interval):
                return {
                    "rebalanced": False,
                    "reason": f"Too soon (last: {time_since.total_seconds()/3600:.1f}h ago)",
                }

        # Calculate current allocations
        current_allocations = self._calculate_allocations(
            current_holdings, current_prices, total_value
        )

        # Check if rebalancing needed
        needs_rebalance, deviations = self._check_deviation(current_allocations)

        if not needs_rebalance:
            return {
                "rebalanced": False,
                "reason": "Within threshold",
                "current_allocations": current_allocations,
                "deviations": deviations,
            }

        logger.info("⚖️ Portfolio rebalancing required")
        logger.info(f"   Deviations: {deviations}")

        # Calculate trades needed
        trades = self._calculate_rebalance_trades(
            current_allocations, current_prices, total_value
        )

        # Execute trades
        executed_trades = []
        for trade in trades:
            if abs(trade["usd_value"]) < self.min_trade_size:
                logger.debug(f"Skipping small trade: {trade}")
                continue

            try:
                result = await neural_net.execute_manual_trade(
                    exchange_id="binance",
                    symbol=trade["symbol"],
                    side=trade["side"],
                    order_type="market",
                    amount=trade["amount"],
                )

                if result.get("success"):
                    executed_trades.append(trade)
                    logger.info(
                        f"✅ Executed rebalance trade: {trade['symbol']} {trade['side']} {trade['amount']:.4f}"
                    )

            except Exception as e:
                logger.error(f"Failed to execute rebalance trade: {e}")

        # Update state
        self.last_rebalance = datetime.now()
        self.rebalance_count += 1

        return {
            "rebalanced": True,
            "trades_executed": len(executed_trades),
            "trades": executed_trades,
            "previous_allocations": current_allocations,
            "target_allocations": self.target_allocations,
            "deviations": deviations,
            "timestamp": self.last_rebalance.isoformat(),
        }

    def _calculate_allocations(
        self, holdings: Dict[str, float], prices: Dict[str, float], total_value: float
    ) -> Dict[str, float]:
        """Calculate current allocation percentages"""
        allocations = {}

        for symbol in self.target_allocations.keys():
            if symbol == "USDT":
                # Cash allocation
                value = holdings.get("USDT", 0)
            else:
                # Asset allocation
                amount = holdings.get(symbol, 0)
                price = prices.get(symbol, 0)
                value = amount * price

            allocation = value / total_value if total_value > 0 else 0
            allocations[symbol] = allocation

        return allocations

    def _check_deviation(
        self, current_allocations: Dict[str, float]
    ) -> tuple[bool, Dict[str, float]]:
        """Check if allocations deviate beyond threshold"""
        deviations = {}
        needs_rebalance = False

        for symbol, target in self.target_allocations.items():
            current = current_allocations.get(symbol, 0)
            deviation = current - target
            deviations[symbol] = deviation

            if abs(deviation) > self.rebalance_threshold:
                needs_rebalance = True

        return needs_rebalance, deviations

    def _calculate_rebalance_trades(
        self,
        current_allocations: Dict[str, float],
        prices: Dict[str, float],
        total_value: float,
    ) -> List[Dict]:
        """Calculate trades needed to rebalance"""
        trades = []

        for symbol, target_allocation in self.target_allocations.items():
            if symbol == "USDT":
                continue  # Handle cash separately

            current_allocation = current_allocations.get(symbol, 0)
            deviation = target_allocation - current_allocation

            # Calculate USD value to trade
            usd_value = deviation * total_value
            price = prices.get(symbol, 0)

            if price == 0:
                continue

            # Calculate amount
            amount = abs(usd_value) / price

            # Determine side
            side = "buy" if deviation > 0 else "sell"

            trades.append(
                {
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "usd_value": usd_value,
                    "current_allocation": current_allocation,
                    "target_allocation": target_allocation,
                    "deviation": deviation,
                }
            )

        return trades

    def set_target_allocation(self, symbol: str, percentage: float):
        """Update target allocation for an asset"""
        if percentage < 0 or percentage > 1:
            raise ValueError("Percentage must be between 0 and 1")

        self.target_allocations[symbol] = percentage
        logger.info(f"Updated target allocation: {symbol} = {percentage:.1%}")

    def get_status(self) -> Dict:
        """Get rebalancer status"""
        return {
            "target_allocations": self.target_allocations,
            "rebalance_threshold": self.rebalance_threshold,
            "rebalance_interval_hours": self.rebalance_interval,
            "last_rebalance": (
                self.last_rebalance.isoformat() if self.last_rebalance else None
            ),
            "rebalance_count": self.rebalance_count,
        }

    # Backward compatibility methods for tests

    def calculate_current_allocation(self, portfolio: Dict) -> Dict:
        """Calculate current portfolio allocation percentages"""
        total_value = sum(asset.get("value", 0) for asset in portfolio.values())
        if total_value == 0:
            return {}

        allocations = {}
        for symbol, data in portfolio.items():
            value = data.get("value", 0)
            allocations[symbol] = (value / total_value) * 100
        return allocations

    def needs_rebalancing(self, current_allocation: Dict) -> bool:
        """Check if rebalancing is needed"""
        for symbol, current_pct in current_allocation.items():
            target_pct = self.target_allocations.get(symbol, 0)
            diff = abs(current_pct - target_pct)
            if diff > self.rebalance_threshold:
                return True
        return False

    def calculate_rebalance_trades(self, portfolio: Dict) -> List[Dict]:
        """Calculate trades needed to rebalance"""
        current_allocation = self.calculate_current_allocation(portfolio)
        if not self.needs_rebalancing(current_allocation):
            return []

        trades = []
        total_value = sum(asset.get("value", 0) for asset in portfolio.values())

        for symbol, target_pct in self.target_allocations.items():
            current_pct = current_allocation.get(symbol, 0)
            diff_pct = target_pct - current_pct

            if abs(diff_pct) > self.rebalance_threshold:
                diff_value = (diff_pct / 100) * total_value
                trades.append({
                    "symbol": symbol,
                    "action": "buy" if diff_value > 0 else "sell",
                    "value": abs(diff_value)
                })

        return trades

    def get_rebalancing_report(self, portfolio: Dict) -> Dict:
        """Get rebalancing analysis report"""
        current = self.calculate_current_allocation(portfolio)
        needs_rebal = self.needs_rebalancing(current)
        trades = self.calculate_rebalance_trades(portfolio) if needs_rebal else []

        return {
            "current_allocation": current,
            "target_allocation": self.target_allocations,
            "needs_rebalancing": needs_rebal,
            "threshold_percent": self.rebalance_threshold,
            "recommended_trades": trades
        }

    def calculate_rebalance(self, portfolio: Dict) -> Dict:
        """Alias for get_rebalancing_report"""
        return self.get_rebalancing_report(portfolio)


if __name__ == "__main__":
    print("Nexlify Portfolio Rebalancer")
    print("Use via integration with trading engine")
