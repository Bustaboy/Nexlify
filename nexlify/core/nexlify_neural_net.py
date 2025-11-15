#!/usr/bin/env python3
"""
Nexlify Neural Net - Main Trading Engine
Wrapper around arasaka_neural_net.py for GUI compatibility
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from nexlify.core.arasaka_neural_net import ArasakaNeuralNet, CyberPair
from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class NexlifyNeuralNet:
    """
    Main neural net interface for the GUI
    Wraps the ArasakaNeuralNet engine with additional features
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.engine = None
        self.is_initialized = False
        self.btc_price = 0.0
        self.active_pairs = []
        self.exchanges = {}
        self.total_profit = 0.0

    @handle_errors("Neural Net Initialization", reraise=False)
    async def initialize(self):
        """Initialize the neural net trading engine"""
        try:
            logger.info("🧠 Initializing Nexlify Neural Net...")

            # Create the core engine
            self.engine = ArasakaNeuralNet(self.config)
            await self.engine.initialize()

            # Copy references for quick access
            self.exchanges = self.engine.exchanges

            # Start background update tasks
            asyncio.create_task(self._update_btc_price())
            asyncio.create_task(self._update_active_pairs())

            self.is_initialized = True
            logger.info("✅ Neural Net initialized successfully")

        except Exception as e:
            error_handler.log_error(
                e, "Neural Net initialization failed", severity="critical"
            )
            raise

    async def _update_btc_price(self):
        """Background task to update BTC price"""
        while self.is_initialized:
            try:
                if self.exchanges:
                    # Get BTC price from first available exchange
                    exchange_id = list(self.exchanges.keys())[0]
                    exchange = self.exchanges[exchange_id]

                    ticker = await exchange.fetch_ticker("BTC/USDT")
                    self.btc_price = ticker["last"]

                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.debug(f"Error updating BTC price: {e}")
                await asyncio.sleep(30)

    async def _update_active_pairs(self):
        """Background task to update active pairs list"""
        while self.is_initialized:
            try:
                if self.engine and hasattr(self.engine, "active_pairs"):
                    self.active_pairs = list(self.engine.active_pairs.values())
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.debug(f"Error updating active pairs: {e}")
                await asyncio.sleep(10)

    def get_active_pairs_display(self) -> List[Dict]:
        """Get active pairs formatted for GUI display"""
        if not self.engine:
            return []

        try:
            return self.engine.get_active_pairs_display()
        except Exception as e:
            logger.error(f"Error getting active pairs display: {e}")
            return []

    async def execute_manual_trade(
        self,
        exchange_id: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
    ) -> Dict:
        """
        Execute a manual trade

        Args:
            exchange_id: Exchange identifier
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', or 'stop_limit'
            amount: Amount to trade
            price: Price for limit orders

        Returns:
            Order result dictionary
        """
        try:
            if exchange_id not in self.exchanges:
                raise ValueError(f"Exchange {exchange_id} not connected")

            exchange = self.exchanges[exchange_id]

            # Prepare order parameters
            params = {}

            # Execute order based on type
            if order_type.lower() == "market":
                if side.lower() == "buy":
                    order = await exchange.create_market_buy_order(symbol, amount)
                else:
                    order = await exchange.create_market_sell_order(symbol, amount)
            elif order_type.lower() == "limit":
                if not price:
                    raise ValueError("Price required for limit orders")
                if side.lower() == "buy":
                    order = await exchange.create_limit_buy_order(symbol, amount, price)
                else:
                    order = await exchange.create_limit_sell_order(
                        symbol, amount, price
                    )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            logger.info(f"✅ Trade executed: {side} {amount} {symbol} on {exchange_id}")

            # Record trade in performance tracker (integration with AI/trainer)
            if hasattr(self.engine, "performance_tracker"):
                try:
                    trade_id = self.engine.performance_tracker.record_trade(
                        exchange=exchange_id,
                        symbol=symbol,
                        side=side.lower(),
                        quantity=amount,
                        entry_price=order.get("price", price or 0),
                        exit_price=None,  # Still open
                        fee=order.get("fee", {}).get("cost", 0),
                        strategy="manual",
                        notes=f"Manual trade via GUI - Order type: {order_type}",
                    )
                    logger.info(f"📊 Manual trade recorded in performance tracker (ID: {trade_id})")
                except Exception as track_err:
                    logger.warning(f"Failed to record manual trade in tracker: {track_err}")

            return order

        except Exception as e:
            error_handler.log_error(
                e, f"Trade execution failed: {side} {amount} {symbol}", severity="error"
            )
            raise

    async def get_account_balance(self, exchange_id: str = None) -> Dict:
        """Get account balance for exchange(s)"""
        try:
            if exchange_id:
                if exchange_id not in self.exchanges:
                    raise ValueError(f"Exchange {exchange_id} not connected")
                return await self.exchanges[exchange_id].fetch_balance()
            else:
                # Get balances from all exchanges
                balances = {}
                for ex_id, exchange in self.exchanges.items():
                    balances[ex_id] = await exchange.fetch_balance()
                return balances
        except Exception as e:
            error_handler.log_error(e, "Failed to fetch balance", severity="error")
            raise

    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions across exchanges"""
        positions = []

        try:
            for exchange_id, exchange in self.exchanges.items():
                # Get open orders
                try:
                    open_orders = await exchange.fetch_open_orders()
                    for order in open_orders:
                        positions.append(
                            {
                                "exchange": exchange_id,
                                "symbol": order["symbol"],
                                "side": order["side"],
                                "amount": order["amount"],
                                "price": order["price"],
                                "status": order["status"],
                                "timestamp": order["timestamp"],
                            }
                        )
                except Exception as e:
                    logger.debug(f"Error fetching orders from {exchange_id}: {e}")

        except Exception as e:
            logger.error(f"Error getting open positions: {e}")

        return positions

    async def cancel_order(self, exchange_id: str, order_id: str, symbol: str) -> bool:
        """Cancel an open order"""
        try:
            if exchange_id not in self.exchanges:
                raise ValueError(f"Exchange {exchange_id} not connected")

            exchange = self.exchanges[exchange_id]
            result = await exchange.cancel_order(order_id, symbol)

            logger.info(f"✅ Order cancelled: {order_id} on {exchange_id}")
            return True

        except Exception as e:
            error_handler.log_error(
                e, f"Failed to cancel order {order_id}", severity="error"
            )
            return False

    async def withdraw_profits_to_btc(self, amount_usd: float) -> bool:
        """Withdraw profits to BTC wallet"""
        try:
            if not self.engine:
                raise ValueError("Engine not initialized")

            success = await self.engine.withdraw_profits_to_btc(amount_usd)

            if success:
                logger.info(f"✅ Withdrawal successful: ${amount_usd:.2f}")
            else:
                logger.warning(f"⚠️ Withdrawal failed: ${amount_usd:.2f}")

            return success

        except Exception as e:
            error_handler.log_error(
                e, f"Withdrawal failed: ${amount_usd}", severity="error"
            )
            return False

    async def calculate_total_profits(self) -> float:
        """Calculate total unrealized profits"""
        try:
            if self.engine:
                self.total_profit = await self.engine.calculate_total_profits()
            return self.total_profit
        except Exception as e:
            logger.error(f"Error calculating profits: {e}")
            return 0.0

    async def shutdown(self):
        """Gracefully shutdown the neural net"""
        try:
            self.is_initialized = False
            if self.engine:
                await self.engine.shutdown()
            logger.info("👋 Neural Net shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
