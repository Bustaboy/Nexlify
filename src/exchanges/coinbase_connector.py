#!/usr/bin/env python3
"""
src/exchanges/coinbase_connector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY COINBASE ADVANCED TRADE CONNECTOR v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ultra-low latency Coinbase connector using latest ccxt 4.3.24+ with
advanced order types, websocket feeds, and AWS co-location support.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import aiohttp
import websockets
import orjson  # Fastest JSON parser
import ccxt.pro as ccxt_pro
from ccxt.base.errors import (
    NetworkError, ExchangeError, InvalidOrder,
    OrderNotFound, InsufficientFunds, RateLimitExceeded
)
import uvloop
from sortedcontainers import SortedDict
from collections import deque
import zstandard as zstd  # For compression
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hmac
import base64
from prometheus_client import Counter, Histogram, Gauge
import msgpack
from aiocache import cached, Cache
from aiocache.serializers import JsonSerializer
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)

# Set uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger("NEXLIFY.COINBASE")

# Prometheus metrics
ORDERS_PLACED = Counter('nexlify_coinbase_orders_placed_total', 'Total orders placed')
ORDERS_FILLED = Counter('nexlify_coinbase_orders_filled_total', 'Total orders filled')
API_LATENCY = Histogram('nexlify_coinbase_api_latency_seconds', 'API call latency')
WS_MESSAGES = Counter('nexlify_coinbase_ws_messages_total', 'WebSocket messages received')
ORDERBOOK_DEPTH = Gauge('nexlify_coinbase_orderbook_depth', 'Current orderbook depth', ['symbol', 'side'])

class OrderType(Enum):
    """Coinbase Advanced Trade order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    FILL_OR_KILL = "fill_or_kill"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"
    POST_ONLY = "post_only"

class TimeInForce(Enum):
    """Time in force options"""
    GTC = "GTC"  # Good Till Cancelled
    GTT = "GTT"  # Good Till Time
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTD = "GTD"  # Good Till Date

@dataclass
class OrderBookLevel:
    """Single orderbook level"""
    price: Decimal
    amount: Decimal
    timestamp: float = field(default_factory=time.time)
    order_count: int = 1

@dataclass
class OrderBook:
    """High-performance orderbook using SortedDict"""
    symbol: str
    bids: SortedDict = field(default_factory=lambda: SortedDict(lambda x: -x))  # Reverse sort
    asks: SortedDict = field(default_factory=SortedDict)
    last_update: float = field(default_factory=time.time)
    sequence: int = 0
    
    def update(self, side: str, price: float, amount: float):
        """Update orderbook level"""
        book = self.bids if side == 'bid' else self.asks
        price_decimal = Decimal(str(price))
        
        if amount == 0:
            book.pop(price_decimal, None)
        else:
            book[price_decimal] = OrderBookLevel(
                price=price_decimal,
                amount=Decimal(str(amount))
            )
        
        self.last_update = time.time()
    
    def get_best_bid(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best bid price and amount"""
        if self.bids:
            price = next(iter(self.bids))
            return price, self.bids[price].amount
        return None
    
    def get_best_ask(self) -> Optional[Tuple[Decimal, Decimal]]:
        """Get best ask price and amount"""
        if self.asks:
            price = next(iter(self.asks))
            return price, self.asks[price].amount
        return None
    
    def get_spread(self) -> Optional[Decimal]:
        """Get current bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask[0] - best_bid[0]
        return None
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid[0] + best_ask[0]) / 2
        return None

class CoinbaseConnector:
    """
    High-performance Coinbase Advanced Trade connector
    Optimized for AWS co-location and ultra-low latency
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.passphrase = config.get('passphrase', '')
        self.testnet = config.get('testnet', False)
        
        # Initialize exchange
        exchange_config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'password': self.passphrase,
            'enableRateLimit': True,
            'rateLimit': 10,  # requests per second
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 5000,
                'timeDifference': 0,
                'test': self.testnet
            }
        }
        
        # Use pro version for websocket support
        self.exchange = ccxt_pro.coinbase(exchange_config)
        
        # AWS optimizations
        self.use_aws_endpoints = config.get('use_aws_endpoints', False)
        if self.use_aws_endpoints:
            self._configure_aws_endpoints()
        
        # Data structures
        self.orderbooks: Dict[str, OrderBook] = {}
        self.active_orders: Dict[str, Dict] = {}
        self.position_tracker: Dict[str, Decimal] = {}
        self.trade_history = deque(maxlen=10000)
        
        # WebSocket management
        self.ws_connection = None
        self.ws_subscriptions: Dict[str, bool] = {}
        self.ws_sequence_tracker: Dict[str, int] = {}
        
        # Performance tracking
        self.latency_tracker = deque(maxlen=1000)
        self.message_compressor = zstd.ZstdCompressor(level=1)  # Fast compression
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            'orderbook': [],
            'ticker': [],
            'trade': [],
            'order': [],
            'error': []
        }
        
    def _configure_aws_endpoints(self):
        """Configure AWS co-located endpoints for lowest latency"""
        # Coinbase Advanced Trade AWS endpoints
        aws_endpoints = {
            'api': 'https://api.exchange.coinbase.com',
            'ws': 'wss://ws-feed.exchange.coinbase.com',
            'fix': 'fix.exchange.coinbase.com:4198'  # FIX protocol for institutions
        }
        
        if self.testnet:
            aws_endpoints = {
                'api': 'https://api-public.sandbox.exchange.coinbase.com',
                'ws': 'wss://ws-feed-public.sandbox.exchange.coinbase.com'
            }
        
        self.exchange.urls['api'] = aws_endpoints['api']
        self.ws_endpoint = aws_endpoints['ws']
        logger.info(f"Configured AWS endpoints: {aws_endpoints}")
    
    async def initialize(self):
        """Initialize connector and establish connections"""
        logger.info("Initializing Coinbase connector...")
        
        # Load markets
        await self.exchange.load_markets()
        
        # Test connectivity
        await self._test_connectivity()
        
        # Start WebSocket connection
        asyncio.create_task(self._maintain_websocket())
        
        # Start performance monitor
        asyncio.create_task(self._performance_monitor())
        
        logger.info("Coinbase connector initialized successfully")
    
    async def _test_connectivity(self):
        """Test API connectivity and measure latency"""
        start = time.perf_counter()
        
        try:
            await self.exchange.fetch_time()
            latency = (time.perf_counter() - start) * 1000  # ms
            
            logger.info(f"Coinbase API latency: {latency:.2f}ms")
            self.latency_tracker.append(latency)
            
            if latency > 100:
                logger.warning(f"High API latency detected: {latency:.2f}ms")
                
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(NetworkError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Place order with advanced options and retry logic
        """
        with API_LATENCY.time():
            try:
                # Validate inputs
                if symbol not in self.exchange.markets:
                    raise ValueError(f"Invalid symbol: {symbol}")
                
                # Prepare order parameters
                order_params = {
                    'symbol': symbol,
                    'type': order_type.value,
                    'side': side,
                    'amount': amount
                }
                
                if price is not None:
                    order_params['price'] = price
                
                # Advanced parameters
                if params:
                    # Time in force
                    if 'timeInForce' in params:
                        order_params['timeInForce'] = params['timeInForce']
                    
                    # Post-only for maker orders
                    if params.get('postOnly'):
                        order_params['postOnly'] = True
                    
                    # Stop orders
                    if 'stopPrice' in params:
                        order_params['stopPrice'] = params['stopPrice']
                    
                    # Client order ID for tracking
                    if 'clientOrderId' in params:
                        order_params['clientOrderId'] = params['clientOrderId']
                
                # Place order
                start = time.perf_counter()
                
                if order_type == OrderType.MARKET:
                    order = await self.exchange.create_market_order(
                        symbol, side, amount, order_params
                    )
                else:
                    order = await self.exchange.create_limit_order(
                        symbol, side, amount, price, order_params
                    )
                
                # Track latency
                latency = (time.perf_counter() - start) * 1000
                self.latency_tracker.append(latency)
                
                # Update tracking
                order_id = order['id']
                self.active_orders[order_id] = order
                ORDERS_PLACED.inc()
                
                # Notify callbacks
                await self._notify_callbacks('order', order)
                
                logger.info(f"Order placed: {order_id} ({latency:.2f}ms)")
                return order
                
            except InsufficientFunds as e:
                logger.error(f"Insufficient funds for order: {e}")
                raise
            except InvalidOrder as e:
                logger.error(f"Invalid order parameters: {e}")
                raise
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order"""
        try:
            result = await self.exchange.cancel_order(order_id, symbol)
            
            # Remove from tracking
            self.active_orders.pop(order_id, None)
            
            logger.info(f"Order cancelled: {order_id}")
            return result
            
        except OrderNotFound:
            logger.warning(f"Order not found: {order_id}")
            self.active_orders.pop(order_id, None)
            raise
    
    async def get_orderbook(self, symbol: str, limit: int = 50) -> OrderBook:
        """Get current orderbook with caching"""
        # Return cached if recent
        if symbol in self.orderbooks:
            ob = self.orderbooks[symbol]
            if time.time() - ob.last_update < 0.1:  # 100ms cache
                return ob
        
        # Fetch new orderbook
        raw_ob = await self.exchange.fetch_order_book(symbol, limit)
        
        # Convert to our format
        orderbook = OrderBook(symbol=symbol)
        
        for bid in raw_ob['bids']:
            orderbook.update('bid', bid[0], bid[1])
            
        for ask in raw_ob['asks']:
            orderbook.update('ask', ask[0], ask[1])
        
        # Update metrics
        ORDERBOOK_DEPTH.labels(symbol=symbol, side='bid').set(len(orderbook.bids))
        ORDERBOOK_DEPTH.labels(symbol=symbol, side='ask').set(len(orderbook.asks))
        
        self.orderbooks[symbol] = orderbook
        return orderbook
    
    async def _maintain_websocket(self):
        """Maintain WebSocket connection with auto-reconnect"""
        while True:
            try:
                await self._connect_websocket()
                await self._websocket_handler()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Reconnect delay
    
    async def _connect_websocket(self):
        """Establish WebSocket connection"""
        # Use the configured endpoint
        endpoint = getattr(self, 'ws_endpoint', 'wss://ws-feed.exchange.coinbase.com')
        
        # Add authentication headers
        auth_headers = self._generate_ws_auth()
        
        self.ws_connection = await websockets.connect(
            endpoint,
            extra_headers=auth_headers,
            ping_interval=20,
            ping_timeout=10,
            compression='deflate'
        )
        
        logger.info("WebSocket connected to Coinbase")
    
    def _generate_ws_auth(self) -> Dict[str, str]:
        """Generate WebSocket authentication headers"""
        timestamp = str(int(time.time()))
        message = timestamp + 'GET' + '/users/self/verify'
        
        hmac_key = base64.b64decode(self.api_secret)
        signature = hmac.new(
            hmac_key,
            message.encode('utf-8'),
            hashes.SHA256()
        )
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.passphrase
        }
    
    async def _websocket_handler(self):
        """Handle incoming WebSocket messages"""
        async for message in self.ws_connection:
            try:
                # Parse message
                data = orjson.loads(message)
                WS_MESSAGES.inc()
                
                # Route to appropriate handler
                msg_type = data.get('type')
                
                if msg_type == 'l2update':
                    await self._handle_orderbook_update(data)
                elif msg_type == 'ticker':
                    await self._handle_ticker(data)
                elif msg_type == 'match':
                    await self._handle_trade(data)
                elif msg_type == 'received' or msg_type == 'done':
                    await self._handle_order_update(data)
                elif msg_type == 'error':
                    logger.error(f"WebSocket error: {data}")
                    await self._notify_callbacks('error', data)
                    
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
    
    async def _handle_orderbook_update(self, data: Dict):
        """Handle orderbook updates"""
        symbol = data['product_id'].replace('-', '/')
        
        if symbol not in self.orderbooks:
            self.orderbooks[symbol] = OrderBook(symbol=symbol)
        
        orderbook = self.orderbooks[symbol]
        
        # Apply changes
        for change in data.get('changes', []):
            side, price, amount = change
            orderbook.update(
                'bid' if side == 'buy' else 'ask',
                float(price),
                float(amount)
            )
        
        # Update sequence
        orderbook.sequence = data.get('sequence', orderbook.sequence)
        
        # Notify callbacks
        await self._notify_callbacks('orderbook', {
            'symbol': symbol,
            'orderbook': orderbook,
            'sequence': orderbook.sequence
        })
    
    async def _handle_ticker(self, data: Dict):
        """Handle ticker updates"""
        symbol = data['product_id'].replace('-', '/')
        
        ticker = {
            'symbol': symbol,
            'bid': float(data.get('best_bid', 0)),
            'ask': float(data.get('best_ask', 0)),
            'last': float(data.get('price', 0)),
            'volume': float(data.get('volume_24h', 0)),
            'timestamp': data.get('time')
        }
        
        await self._notify_callbacks('ticker', ticker)
    
    async def _handle_trade(self, data: Dict):
        """Handle trade updates"""
        trade = {
            'id': data.get('trade_id'),
            'symbol': data['product_id'].replace('-', '/'),
            'price': float(data.get('price', 0)),
            'amount': float(data.get('size', 0)),
            'side': data.get('side'),
            'timestamp': data.get('time')
        }
        
        # Store in history
        self.trade_history.append(trade)
        
        await self._notify_callbacks('trade', trade)
    
    async def _handle_order_update(self, data: Dict):
        """Handle order updates"""
        order_id = data.get('order_id')
        
        if order_id in self.active_orders:
            # Update order status
            if data['type'] == 'done':
                reason = data.get('reason')
                if reason == 'filled':
                    ORDERS_FILLED.inc()
                
                # Remove from active
                self.active_orders.pop(order_id, None)
            
            await self._notify_callbacks('order', data)
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = 50):
        """Subscribe to orderbook updates"""
        if not self.ws_connection:
            await asyncio.sleep(1)  # Wait for connection
        
        subscribe_msg = {
            'type': 'subscribe',
            'product_ids': [s.replace('/', '-') for s in symbols],
            'channels': ['level2', 'ticker', 'matches']
        }
        
        await self.ws_connection.send(orjson.dumps(subscribe_msg).decode())
        
        for symbol in symbols:
            self.ws_subscriptions[symbol] = True
        
        logger.info(f"Subscribed to orderbook updates for: {symbols}")
    
    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        while True:
            await asyncio.sleep(60)  # Every minute
            
            if self.latency_tracker:
                avg_latency = np.mean(self.latency_tracker)
                p99_latency = np.percentile(list(self.latency_tracker), 99)
                
                logger.info(f"Performance - Avg latency: {avg_latency:.2f}ms, P99: {p99_latency:.2f}ms")
    
    async def _notify_callbacks(self, event_type: str, data: Any):
        """Notify registered callbacks"""
        for callback in self.callbacks.get(event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register event callback"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
        else:
            raise ValueError(f"Invalid event type: {event_type}")
    
    async def close(self):
        """Clean shutdown"""
        logger.info("Closing Coinbase connector...")
        
        if self.ws_connection:
            await self.ws_connection.close()
        
        await self.exchange.close()
        
        logger.info("Coinbase connector closed")
