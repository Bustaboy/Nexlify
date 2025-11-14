#!/usr/bin/env python3
"""
Nexlify WebSocket Real-Time Feeds
High-performance real-time market data streaming
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional

import ccxt.async_support as ccxt

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class WebSocketFeedManager:
    """
    Manages WebSocket connections for real-time market data
    Supports multiple exchanges and symbols simultaneously
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.active_streams: Dict[str, asyncio.Task] = {}

        # Callbacks for different data types
        self.ticker_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        self.orderbook_callbacks: List[Callable] = []
        self.ohlcv_callbacks: List[Callable] = []

        # Data cache
        self.latest_tickers: Dict[str, Dict] = {}
        self.latest_trades: Dict[str, List] = defaultdict(list)
        self.latest_orderbooks: Dict[str, Dict] = {}

        # Performance tracking
        self.message_count = 0
        self.start_time = datetime.now()

        logger.info("🌐 WebSocket Feed Manager initialized")

    async def initialize(self, exchange_configs: Dict[str, Dict]):
        """Initialize WebSocket connections for exchanges"""
        for exchange_id, config in exchange_configs.items():
            try:
                # Create exchange instance with WebSocket support
                exchange_class = getattr(ccxt, exchange_id)

                exchange = exchange_class(
                    {
                        "apiKey": config.get("apiKey", ""),
                        "secret": config.get("secret", ""),
                        "enableRateLimit": True,
                        "options": {
                            "defaultType": "spot",
                            "warnOnFetchOHLCVLimitArgument": False,
                        },
                    }
                )

                # Check if exchange supports WebSocket
                if hasattr(exchange, "watch_ticker"):
                    self.exchanges[exchange_id] = exchange
                    logger.info(f"✅ WebSocket enabled for {exchange_id}")
                else:
                    logger.warning(f"⚠️ {exchange_id} doesn't support WebSocket")
                    await exchange.close()

            except Exception as e:
                logger.error(f"Failed to initialize WebSocket for {exchange_id}: {e}")

    async def subscribe_ticker(self, exchange_id: str, symbols: List[str]):
        """
        Subscribe to real-time ticker updates

        Args:
            exchange_id: Exchange name (e.g., 'binance')
            symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        """
        if exchange_id not in self.exchanges:
            logger.error(f"Exchange {exchange_id} not initialized")
            return

        stream_key = f"{exchange_id}_ticker_{'_'.join(symbols)}"

        if stream_key not in self.active_streams:
            task = asyncio.create_task(self._ticker_stream(exchange_id, symbols))
            self.active_streams[stream_key] = task
            logger.info(f"📡 Subscribed to tickers: {exchange_id} {symbols}")

    async def _ticker_stream(self, exchange_id: str, symbols: List[str]):
        """Internal ticker stream handler"""
        exchange = self.exchanges[exchange_id]

        try:
            while True:
                for symbol in symbols:
                    try:
                        ticker = await exchange.watch_ticker(symbol)

                        # Update cache
                        cache_key = f"{exchange_id}:{symbol}"
                        self.latest_tickers[cache_key] = ticker
                        self.message_count += 1

                        # Trigger callbacks
                        for callback in self.ticker_callbacks:
                            try:
                                await callback(exchange_id, symbol, ticker)
                            except Exception as e:
                                logger.error(f"Ticker callback error: {e}")

                    except Exception as e:
                        logger.debug(f"Error in ticker stream for {symbol}: {e}")
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Ticker stream cancelled: {exchange_id}")
        except Exception as e:
            logger.error(f"Fatal error in ticker stream: {e}")

    async def subscribe_trades(self, exchange_id: str, symbols: List[str]):
        """
        Subscribe to real-time trade updates

        Args:
            exchange_id: Exchange name
            symbols: List of trading pairs
        """
        if exchange_id not in self.exchanges:
            logger.error(f"Exchange {exchange_id} not initialized")
            return

        stream_key = f"{exchange_id}_trades_{'_'.join(symbols)}"

        if stream_key not in self.active_streams:
            task = asyncio.create_task(self._trades_stream(exchange_id, symbols))
            self.active_streams[stream_key] = task
            logger.info(f"📡 Subscribed to trades: {exchange_id} {symbols}")

    async def _trades_stream(self, exchange_id: str, symbols: List[str]):
        """Internal trades stream handler"""
        exchange = self.exchanges[exchange_id]

        try:
            while True:
                for symbol in symbols:
                    try:
                        trades = await exchange.watch_trades(symbol)

                        # Update cache (keep last 100 trades)
                        cache_key = f"{exchange_id}:{symbol}"
                        self.latest_trades[cache_key].extend(trades)
                        self.latest_trades[cache_key] = self.latest_trades[cache_key][
                            -100:
                        ]
                        self.message_count += len(trades)

                        # Trigger callbacks
                        for callback in self.trade_callbacks:
                            try:
                                await callback(exchange_id, symbol, trades)
                            except Exception as e:
                                logger.error(f"Trade callback error: {e}")

                    except Exception as e:
                        logger.debug(f"Error in trades stream for {symbol}: {e}")
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Trades stream cancelled: {exchange_id}")
        except Exception as e:
            logger.error(f"Fatal error in trades stream: {e}")

    async def subscribe_orderbook(
        self, exchange_id: str, symbols: List[str], limit: int = 20
    ):
        """
        Subscribe to real-time order book updates

        Args:
            exchange_id: Exchange name
            symbols: List of trading pairs
            limit: Order book depth (default 20)
        """
        if exchange_id not in self.exchanges:
            logger.error(f"Exchange {exchange_id} not initialized")
            return

        stream_key = f"{exchange_id}_orderbook_{'_'.join(symbols)}"

        if stream_key not in self.active_streams:
            task = asyncio.create_task(
                self._orderbook_stream(exchange_id, symbols, limit)
            )
            self.active_streams[stream_key] = task
            logger.info(f"📡 Subscribed to order books: {exchange_id} {symbols}")

    async def _orderbook_stream(self, exchange_id: str, symbols: List[str], limit: int):
        """Internal order book stream handler"""
        exchange = self.exchanges[exchange_id]

        try:
            while True:
                for symbol in symbols:
                    try:
                        orderbook = await exchange.watch_order_book(symbol, limit)

                        # Update cache
                        cache_key = f"{exchange_id}:{symbol}"
                        self.latest_orderbooks[cache_key] = orderbook
                        self.message_count += 1

                        # Trigger callbacks
                        for callback in self.orderbook_callbacks:
                            try:
                                await callback(exchange_id, symbol, orderbook)
                            except Exception as e:
                                logger.error(f"Orderbook callback error: {e}")

                    except Exception as e:
                        logger.debug(f"Error in orderbook stream for {symbol}: {e}")
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"Orderbook stream cancelled: {exchange_id}")
        except Exception as e:
            logger.error(f"Fatal error in orderbook stream: {e}")

    async def subscribe_ohlcv(
        self, exchange_id: str, symbols: List[str], timeframe: str = "1m"
    ):
        """
        Subscribe to real-time OHLCV (candlestick) updates

        Args:
            exchange_id: Exchange name
            symbols: List of trading pairs
            timeframe: Timeframe (e.g., '1m', '5m', '1h')
        """
        if exchange_id not in self.exchanges:
            logger.error(f"Exchange {exchange_id} not initialized")
            return

        stream_key = f"{exchange_id}_ohlcv_{timeframe}_{'_'.join(symbols)}"

        if stream_key not in self.active_streams:
            task = asyncio.create_task(
                self._ohlcv_stream(exchange_id, symbols, timeframe)
            )
            self.active_streams[stream_key] = task
            logger.info(
                f"📡 Subscribed to OHLCV ({timeframe}): {exchange_id} {symbols}"
            )

    async def _ohlcv_stream(self, exchange_id: str, symbols: List[str], timeframe: str):
        """Internal OHLCV stream handler"""
        exchange = self.exchanges[exchange_id]

        try:
            # Check if exchange supports watch_ohlcv
            if not hasattr(exchange, "watch_ohlcv"):
                logger.warning(f"{exchange_id} doesn't support OHLCV streaming")
                return

            while True:
                for symbol in symbols:
                    try:
                        ohlcv = await exchange.watch_ohlcv(symbol, timeframe)

                        self.message_count += len(ohlcv)

                        # Trigger callbacks
                        for callback in self.ohlcv_callbacks:
                            try:
                                await callback(exchange_id, symbol, timeframe, ohlcv)
                            except Exception as e:
                                logger.error(f"OHLCV callback error: {e}")

                    except Exception as e:
                        logger.debug(f"Error in OHLCV stream for {symbol}: {e}")
                        await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(f"OHLCV stream cancelled: {exchange_id}")
        except Exception as e:
            logger.error(f"Fatal error in OHLCV stream: {e}")

    def on_ticker(self, callback: Callable):
        """Register callback for ticker updates"""
        self.ticker_callbacks.append(callback)

    def on_trade(self, callback: Callable):
        """Register callback for trade updates"""
        self.trade_callbacks.append(callback)

    def on_orderbook(self, callback: Callable):
        """Register callback for order book updates"""
        self.orderbook_callbacks.append(callback)

    def on_ohlcv(self, callback: Callable):
        """Register callback for OHLCV updates"""
        self.ohlcv_callbacks.append(callback)

    def get_latest_ticker(self, exchange_id: str, symbol: str) -> Optional[Dict]:
        """Get latest cached ticker data"""
        cache_key = f"{exchange_id}:{symbol}"
        return self.latest_tickers.get(cache_key)

    def get_latest_trades(self, exchange_id: str, symbol: str, limit: int = 10) -> List:
        """Get latest cached trades"""
        cache_key = f"{exchange_id}:{symbol}"
        return self.latest_trades.get(cache_key, [])[-limit:]

    def get_latest_orderbook(self, exchange_id: str, symbol: str) -> Optional[Dict]:
        """Get latest cached order book"""
        cache_key = f"{exchange_id}:{symbol}"
        return self.latest_orderbooks.get(cache_key)

    def get_statistics(self) -> Dict:
        """Get WebSocket performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        msg_per_sec = self.message_count / uptime if uptime > 0 else 0

        return {
            "active_streams": len(self.active_streams),
            "total_messages": self.message_count,
            "messages_per_second": msg_per_sec,
            "uptime_seconds": uptime,
            "cached_tickers": len(self.latest_tickers),
            "cached_orderbooks": len(self.latest_orderbooks),
        }

    async def unsubscribe_all(self):
        """Cancel all active streams"""
        logger.info("🛑 Unsubscribing from all WebSocket streams...")

        for stream_key, task in self.active_streams.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self.active_streams.clear()
        logger.info("✅ All streams cancelled")

    async def close(self):
        """Close all WebSocket connections"""
        await self.unsubscribe_all()

        for exchange_id, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info(f"✅ Closed WebSocket connection: {exchange_id}")
            except Exception as e:
                logger.error(f"Error closing {exchange_id}: {e}")

        self.exchanges.clear()


# Example usage and testing
if __name__ == "__main__":

    async def ticker_callback(exchange_id, symbol, ticker):
        """Example ticker callback"""
        print(
            f"[{exchange_id}] {symbol} - Price: {ticker['last']:.2f} "
            f"Vol: {ticker['baseVolume']:.2f}"
        )

    async def trade_callback(exchange_id, symbol, trades):
        """Example trade callback"""
        for trade in trades:
            print(
                f"[{exchange_id}] {symbol} - Trade: "
                f"{trade['side']} {trade['amount']:.4f} @ {trade['price']:.2f}"
            )

    async def main():
        print("=" * 70)
        print("NEXLIFY WEBSOCKET FEED DEMO")
        print("=" * 70)

        # Initialize feed manager
        feed_manager = WebSocketFeedManager()

        # Register callbacks
        feed_manager.on_ticker(ticker_callback)
        feed_manager.on_trade(trade_callback)

        # Initialize exchanges
        await feed_manager.initialize({"binance": {"apiKey": "", "secret": ""}})

        # Subscribe to streams
        symbols = ["BTC/USDT", "ETH/USDT"]
        await feed_manager.subscribe_ticker("binance", symbols)
        await feed_manager.subscribe_trades("binance", symbols)

        # Run for 30 seconds
        print("\n📡 Streaming market data for 30 seconds...\n")
        await asyncio.sleep(30)

        # Show statistics
        stats = feed_manager.get_statistics()
        print("\n" + "=" * 70)
        print("PERFORMANCE STATISTICS")
        print("=" * 70)
        print(f"Active Streams: {stats['active_streams']}")
        print(f"Total Messages: {stats['total_messages']}")
        print(f"Messages/sec: {stats['messages_per_second']:.2f}")
        print(f"Uptime: {stats['uptime_seconds']:.1f}s")
        print("=" * 70)

        # Cleanup
        await feed_manager.close()

    asyncio.run(main())
