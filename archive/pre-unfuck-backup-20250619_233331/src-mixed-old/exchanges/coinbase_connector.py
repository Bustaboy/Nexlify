#!/usr/bin/env python3
"""
src/exchanges/coinbase_connector.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY COINBASE ADVANCED TRADE CONNECTOR v3.1 (MERGED EDITION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ultra-low latency Coinbase connector combining:
- ccxt 4.3.24+ proven functionality  
- Rust/PyO3 acceleration capability
- QuestDB 4.3M rows/sec data ingestion
- JWT authentication for Advanced Trade API
- AWS co-location optimization
"""

import os
import time
import hmac
import hashlib
import base64
import asyncio
import jwt
import zstd
import websockets
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from collections import defaultdict, deque
import numpy as np
from sortedcontainers import SortedDict

# Core libraries
import ccxt.pro as ccxt_pro
import orjson
import aiohttp
import structlog
from prometheus_client import Counter, Histogram, Gauge
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

# Future Rust integration via PyO3
try:
    from nexlify_rust import RustWebsocketHandler, RustOrderBook, RustLatencyTracker
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    RustWebsocketHandler = None
    RustOrderBook = None
    RustLatencyTracker = None

# Import our config system
from ..utils.config_loader import get_config_loader, CyberColors

# Initialize logger
logger = structlog.get_logger("NEXLIFY.EXCHANGE.COINBASE")

# Metrics
ORDERBOOK_UPDATES = Counter('nexlify_coinbase_orderbook_updates_total', 'Orderbook updates received')
ORDERS_PLACED = Counter('nexlify_coinbase_orders_placed_total', 'Orders placed')
ORDERS_FILLED = Counter('nexlify_coinbase_orders_filled_total', 'Orders filled')
API_LATENCY = Histogram('nexlify_coinbase_api_latency_seconds', 'API call latency', ['operation'])
WS_MESSAGES = Counter('nexlify_coinbase_ws_messages_total', 'WebSocket messages received')
ORDERBOOK_DEPTH = Gauge('nexlify_coinbase_orderbook_depth', 'Orderbook depth', ['symbol', 'side'])

# Constants
COINBASE_API_URL = "https://api.exchange.coinbase.com"
COINBASE_ADV_API_URL = "https://api.coinbase.com/api/v3/brokerage"
COINBASE_WS_URL = "wss://ws-feed.exchange.coinbase.com"
COINBASE_ADV_WS_URL = "wss://advanced-trade-ws.coinbase.com"
MAX_REQUESTS_PER_SECOND = 30
ORDER_BOOK_DEPTH = 50
TICK_BUFFER_SIZE = 10000
LATENCY_WARNING_MS = 100


@dataclass
class OrderBookLevel:
    """Single orderbook level"""
    price: Decimal
    amount: Decimal
    num_orders: int = 1
    
    def to_rust_format(self) -> Tuple[float, float, int]:
        """Convert to Rust-compatible tuple"""
        return (float(self.price), float(self.amount), self.num_orders)


@dataclass
class OrderBook:
    """High-performance orderbook with Rust acceleration support"""
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
    
    def to_rust_format(self) -> Dict[str, Any]:
        """Convert to Rust-compatible format for processing"""
        return {
            'symbol': self.symbol,
            'bids': [(float(p), float(l.amount), l.num_orders) for p, l in self.bids.items()],
            'asks': [(float(p), float(l.amount), l.num_orders) for p, l in self.asks.items()],
            'sequence': self.sequence,
            'timestamp': int(self.last_update * 1e9)
        }
    
    def to_questdb_format(self) -> List[str]:
        """Format orderbook snapshot for QuestDB ingestion"""
        lines = []
        timestamp = int(self.last_update * 1e9)
        
        # Format top levels for time series storage
        for i, (price, level) in enumerate(list(self.bids.items())[:10]):
            lines.append(
                f"orderbook,symbol={self.symbol},side=bid,level={i} "
                f"price={float(price)},amount={float(level.amount)},orders={level.num_orders} "
                f"{timestamp}"
            )
        
        for i, (price, level) in enumerate(list(self.asks.items())[:10]):
            lines.append(
                f"orderbook,symbol={self.symbol},side=ask,level={i} "
                f"price={float(price)},amount={float(level.amount)},orders={level.num_orders} "
                f"{timestamp}"
            )
        
        return lines


@dataclass
class Trade:
    """Trade data structure optimized for QuestDB ingestion"""
    symbol: str
    trade_id: str
    price: Decimal
    size: Decimal
    side: str  # 'buy' or 'sell'
    timestamp: int
    
    def to_questdb_format(self) -> str:
        """Format for QuestDB line protocol (4.3M rows/sec ingestion)"""
        return (
            f"trades,symbol={self.symbol},side={self.side} "
            f"price={float(self.price)},size={float(self.size)},trade_id=\"{self.trade_id}\" "
            f"{self.timestamp}"
        )


class CoinbaseAdvancedAuth:
    """JWT authentication for Coinbase Advanced Trade API"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        
    def generate_jwt(self, request_method: str, request_path: str) -> str:
        """Generate JWT token for Advanced Trade API"""
        service = "retail_rest_api_proxy"
        uri = f"{request_method} {request_path}"
        
        try:
            # Load private key
            private_key = serialization.load_pem_private_key(
                self.api_secret.encode('utf-8'),
                password=None,
                backend=default_backend()
            )
            
            # Generate JWT
            jwt_payload = {
                'sub': self.api_key,
                'iss': "coinbase-cloud",
                'nbf': int(time.time()),
                'exp': int(time.time()) + 60,
                'aud': [service],
                'uri': uri
            }
            
            return jwt.encode(
                jwt_payload,
                private_key,
                algorithm='ES256',
                headers={'kid': self.api_key, 'typ': 'JWT'}
            )
        except Exception as e:
            logger.error(f"JWT generation failed: {e}")
            raise


class CoinbaseConnector:
    """
    🚀 NEXLIFY Enhanced Coinbase Connector
    
    Merged features:
    - ccxt.pro for proven exchange connectivity
    - Rust acceleration capability via PyO3
    - QuestDB integration for 4.3M rows/sec ingestion
    - JWT auth for Advanced Trade API
    - AWS co-location optimization
    - Sub-millisecond execution targets
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')
        self.passphrase = config.get('passphrase', '')
        self.testnet = config.get('testnet', False)
        self.use_advanced_api = config.get('use_advanced_api', False)
        
        # Initialize exchange (ccxt)
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
        
        # Advanced Trade API auth (if enabled)
        if self.use_advanced_api and self.api_key and self.api_secret:
            self.advanced_auth = CoinbaseAdvancedAuth(self.api_key, self.api_secret)
        else:
            self.advanced_auth = None
        
        # AWS optimizations
        self.use_aws_endpoints = config.get('use_aws_endpoints', False)
        if self.use_aws_endpoints:
            self._configure_aws_endpoints()
        
        # Data structures
        self.orderbooks: Dict[str, OrderBook] = {}
        self.active_orders: Dict[str, Dict] = {}
        self.position_tracker: Dict[str, Decimal] = {}
        self.trade_history = deque(maxlen=10000)
        self.tick_buffer = deque(maxlen=TICK_BUFFER_SIZE)
        
        # WebSocket management
        self.ws_connection = None
        self.ws_subscriptions: Dict[str, bool] = {}
        self.ws_sequence_tracker: Dict[str, int] = {}
        
        # Performance tracking
        self.latency_tracker = deque(maxlen=1000)
        self.message_compressor = zstd.ZstdCompressor(level=1)  # Fast compression
        self.message_count = 0
        self.last_heartbeat = time.time()
        
        # Rust acceleration (if available)
        self.rust_handler = None
        self.rust_orderbook = None
        self.rust_latency_tracker = None
        if RUST_AVAILABLE:
            self._init_rust_acceleration()
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # HTTP session for Advanced API
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _init_rust_acceleration(self):
        """Initialize Rust components for maximum performance"""
        try:
            self.rust_handler = RustWebsocketHandler()
            self.rust_orderbook = RustOrderBook()
            self.rust_latency_tracker = RustLatencyTracker()
            logger.info(f"{CyberColors.NEON_GREEN}✓ Rust acceleration enabled{CyberColors.RESET}")
        except Exception as e:
            logger.warning(f"{CyberColors.NEON_YELLOW}⚠ Rust acceleration failed: {e}{CyberColors.RESET}")
            RUST_AVAILABLE = False
    
    def _configure_aws_endpoints(self):
        """Configure AWS co-located endpoints for lowest latency"""
        # Coinbase Advanced Trade AWS endpoints
        aws_endpoints = {
            'api': 'https://api.exchange.coinbase.com',
            'ws': 'wss://ws-feed.exchange.coinbase.com',
            'fix': 'fix.exchange.coinbase.com:4198',  # FIX protocol for institutions
            'advanced_api': 'https://api.coinbase.com',
            'advanced_ws': 'wss://advanced-trade-ws.coinbase.com'
        }
        
        if self.testnet:
            aws_endpoints = {
                'api': 'https://api-public.sandbox.exchange.coinbase.com',
                'ws': 'wss://ws-feed-public.sandbox.exchange.coinbase.com',
                'advanced_api': 'https://api-public.sandbox.coinbase.com',
                'advanced_ws': 'wss://ws-feed-public.sandbox.advanced-trade.coinbase.com'
            }
        
        self.exchange.urls['api'] = aws_endpoints['api']
        self.ws_endpoint = aws_endpoints['advanced_ws' if self.use_advanced_api else 'ws']
        logger.info(f"{CyberColors.NEON_CYAN}AWS endpoints configured: {aws_endpoints}{CyberColors.RESET}")
    
    async def initialize(self):
        """Initialize connector and establish connections"""
        logger.info(f"{CyberColors.NEON_CYAN}🔌 Initializing Coinbase connector...{CyberColors.RESET}")
        
        # Create HTTP session for Advanced API
        if self.use_advanced_api:
            self.session = aiohttp.ClientSession()
        
        # Load markets
        await self.exchange.load_markets()
        
        # Test connectivity
        await self._test_connectivity()
        
        # Start WebSocket connection
        asyncio.create_task(self._maintain_websocket())
        
        # Start performance monitor
        asyncio.create_task(self._performance_monitor())
        
        # Start latency monitor
        asyncio.create_task(self._latency_monitor())
        
        logger.info(f"{CyberColors.NEON_GREEN}✓ Coinbase connector initialized - Neural link established{CyberColors.RESET}")
    
    async def _test_connectivity(self):
        """Test API connectivity and measure latency"""
        start_time = time.perf_counter()
        
        try:
            if self.use_advanced_api and self.advanced_auth:
                # Test Advanced Trade API
                response = await self._make_advanced_request('GET', '/accounts')
                if response:
                    logger.info(f"{CyberColors.NEON_GREEN}Advanced Trade API connected{CyberColors.RESET}")
            else:
                # Test standard API
                balance = await self.exchange.fetch_balance()
                
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_tracker.append(latency_ms)
            
            API_LATENCY.labels(operation='connectivity_test').observe(latency_ms / 1000)
            logger.info(f"API latency: {latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")
            raise
    
    async def _make_advanced_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Make authenticated request to Advanced Trade API"""
        if not self.session or not self.advanced_auth:
            return None
        
        url = f"{COINBASE_ADV_API_URL}{endpoint}"
        headers = {
            'Authorization': f'Bearer {self.advanced_auth.generate_jwt(method, endpoint)}',
            'Content-Type': 'application/json'
        }
        
        start_time = time.perf_counter()
        
        try:
            async with self.session.request(
                method, 
                url, 
                json=data, 
                params=params, 
                headers=headers
            ) as response:
                # Track latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                self.latency_tracker.append(latency_ms)
                
                if RUST_AVAILABLE and self.rust_latency_tracker:
                    self.rust_latency_tracker.record(latency_ms)
                
                if latency_ms > LATENCY_WARNING_MS:
                    logger.warning(
                        f"{CyberColors.NEON_YELLOW}High latency: {latency_ms:.1f}ms{CyberColors.RESET}"
                    )
                
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    async def place_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        order_type: str,  # 'market', 'limit', etc.
        amount: float,
        price: Optional[float] = None,
        params: Dict[str, Any] = {}
    ) -> Optional[Dict]:
        """
        Place order with sub-millisecond execution target
        
        Supports both standard and Advanced Trade API
        """
        start_time = time.perf_counter()
        
        try:
            if self.use_advanced_api and self.advanced_auth:
                # Use Advanced Trade API
                order_config = {
                    "product_id": symbol.replace('/', '-'),
                    "side": side,
                    "order_configuration": {}
                }
                
                if 'client_order_id' in params:
                    order_config["client_order_id"] = params['client_order_id']
                
                # Configure order type
                if order_type == "market":
                    order_config["order_configuration"]["market_market_ioc"] = {
                        "base_size": str(amount) if side == "sell" else None,
                        "quote_size": str(amount) if side == "buy" else None
                    }
                elif order_type == "limit":
                    order_config["order_configuration"]["limit_limit_gtc"] = {
                        "base_size": str(amount),
                        "limit_price": str(price),
                        "post_only": params.get("post_only", False)
                    }
                
                response = await self._make_advanced_request('POST', '/orders', data=order_config)
                
                if response and response.get('success'):
                    order = response
                else:
                    order = None
            else:
                # Use standard ccxt API
                order = await self.exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    price=price,
                    params=params
                )
            
            # Track execution latency
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            API_LATENCY.labels(operation='place_order').observe(execution_time_ms / 1000)
            
            if order:
                ORDERS_PLACED.inc()
                self.active_orders[order['id']] = order
                
                logger.info(
                    f"{CyberColors.NEON_GREEN}✓ Order placed in {execution_time_ms:.1f}ms: "
                    f"{side} {amount} {symbol} @ {price or 'market'}{CyberColors.RESET}"
                )
                
                # Notify callbacks
                await self._notify_callbacks('order', {
                    'order': order,
                    'latency_ms': execution_time_ms
                })
                
                return order
            else:
                logger.error(f"Order placement failed")
                return None
                
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order with minimal latency"""
        start_time = time.perf_counter()
        
        try:
            if self.use_advanced_api and self.advanced_auth:
                # Use Advanced Trade API
                response = await self._make_advanced_request(
                    'POST', 
                    f'/orders/batch_cancel', 
                    data={"order_ids": [order_id]}
                )
                
                if response and response.get('results'):
                    result = response['results'][0]
                    success = result.get('success', False)
                else:
                    success = False
            else:
                # Use standard ccxt API
                await self.exchange.cancel_order(order_id, symbol)
                success = True
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            API_LATENCY.labels(operation='cancel_order').observe(latency_ms / 1000)
            
            if success:
                self.active_orders.pop(order_id, None)
                logger.info(
                    f"{CyberColors.NEON_YELLOW}Order cancelled in {latency_ms:.1f}ms{CyberColors.RESET}"
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return False
    
    async def fetch_orderbook(self, symbol: str, limit: int = ORDER_BOOK_DEPTH) -> OrderBook:
        """Fetch orderbook with ccxt"""
        start_time = time.perf_counter()
        
        # Fetch from exchange
        raw_orderbook = await self.exchange.fetch_order_book(symbol, limit)
        
        # Convert to our format
        orderbook = OrderBook(symbol=symbol)
        
        for bid in raw_orderbook['bids']:
            orderbook.update('bid', bid[0], bid[1])
        
        for ask in raw_orderbook['asks']:
            orderbook.update('ask', ask[0], ask[1])
        
        # Track latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        API_LATENCY.labels(operation='fetch_orderbook').observe(latency_ms / 1000)
        
        # Update metrics
        ORDERBOOK_DEPTH.labels(symbol=symbol, side='bid').set(len(orderbook.bids))
        ORDERBOOK_DEPTH.labels(symbol=symbol, side='ask').set(len(orderbook.asks))
        
        self.orderbooks[symbol] = orderbook
        return orderbook
    
    async def _maintain_websocket(self):
        """Maintain WebSocket connection with auto-reconnect"""
        while True:
            try:
                if RUST_AVAILABLE and self.rust_handler:
                    # Use Rust WebSocket handler for maximum performance
                    await self._connect_rust_websocket()
                else:
                    # Use Python WebSocket
                    await self._connect_websocket()
                    await self._websocket_handler()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await asyncio.sleep(5)  # Reconnect delay
    
    async def _connect_rust_websocket(self):
        """Connect using Rust WebSocket handler"""
        logger.info(f"{CyberColors.NEURAL_PURPLE}Initializing Rust WebSocket handler...{CyberColors.RESET}")
        
        # Configure Rust handler
        auth_headers = self._generate_ws_auth()
        endpoint = getattr(self, 'ws_endpoint', COINBASE_WS_URL)
        
        # Connect via Rust
        self.rust_handler.set_callback(self._process_rust_message)
        await self.rust_handler.connect(endpoint, auth_headers)
        
        # Subscribe to channels
        for symbol in self.ws_subscriptions:
            await self.rust_handler.subscribe(symbol, ['level2', 'ticker', 'matches'])
        
        logger.info(f"{CyberColors.NEON_GREEN}✓ Rust WebSocket connected{CyberColors.RESET}")
    
    async def _connect_websocket(self):
        """Establish WebSocket connection (Python fallback)"""
        # Use the configured endpoint
        endpoint = getattr(self, 'ws_endpoint', COINBASE_WS_URL)
        
        # Add authentication headers
        auth_headers = self._generate_ws_auth()
        
        self.ws_connection = await websockets.connect(
            endpoint,
            extra_headers=auth_headers,
            ping_interval=20,
            ping_timeout=10,
            compression='deflate'
        )
        
        logger.info(f"{CyberColors.NEON_CYAN}WebSocket connected to Coinbase{CyberColors.RESET}")
    
    def _generate_ws_auth(self) -> Dict[str, str]:
        """Generate WebSocket authentication headers"""
        if self.use_advanced_api and self.advanced_auth:
            # Advanced Trade API uses JWT
            return {
                'Authorization': f'Bearer {self.advanced_auth.generate_jwt("GET", "/ws")}'
            }
        else:
            # Standard API uses HMAC
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
                self.message_count += 1
                
                # Route to appropriate handler
                msg_type = data.get('type')
                
                if msg_type == 'l2update':
                    await self._handle_orderbook_update(data)
                elif msg_type == 'ticker':
                    await self._handle_ticker(data)
                elif msg_type == 'match':
                    await self._handle_match(data)
                elif msg_type == 'heartbeat':
                    self.last_heartbeat = time.time()
                elif msg_type in ['received', 'open', 'done', 'change']:
                    await self._handle_order_update(data)
                
            except Exception as e:
                logger.error(f"Message processing error: {e}")
    
    async def _process_rust_message(self, message: Dict):
        """Process message from Rust WebSocket handler"""
        # Rust handler pre-processes messages for efficiency
        msg_type = message.get('type')
        
        if msg_type == 'orderbook':
            # Rust already parsed and optimized the orderbook
            symbol = message['symbol']
            self.orderbooks[symbol] = message['orderbook']
            ORDERBOOK_UPDATES.inc()
            await self._notify_callbacks('orderbook', message)
            
        elif msg_type == 'trade':
            # Convert to Trade object for QuestDB
            trade = Trade(
                symbol=message['symbol'],
                trade_id=message['trade_id'],
                price=Decimal(message['price']),
                size=Decimal(message['size']),
                side=message['side'],
                timestamp=message['timestamp']
            )
            
            # Queue for QuestDB ingestion
            questdb_line = trade.to_questdb_format()
            await self._notify_callbacks('trade', {
                'trade': trade,
                'questdb': questdb_line
            })
    
    async def _handle_orderbook_update(self, data: Dict):
        """Process orderbook update"""
        symbol = data['product_id'].replace('-', '/')
        
        if symbol not in self.orderbooks:
            self.orderbooks[symbol] = OrderBook(symbol=symbol)
        
        orderbook = self.orderbooks[symbol]
        
        # Check sequence
        if 'sequence' in data:
            if data['sequence'] <= orderbook.sequence:
                return  # Old message
            orderbook.sequence = data['sequence']
        
        # Apply changes
        for change in data.get('changes', []):
            side_str, price_str, size_str = change
            side = 'bid' if side_str == 'buy' else 'ask'
            orderbook.update(side, float(price_str), float(size_str))
        
        ORDERBOOK_UPDATES.inc()
        
        # Use Rust for heavy computations if available
        if RUST_AVAILABLE and self.rust_orderbook:
            rust_metrics = self.rust_orderbook.calculate_metrics(orderbook.to_rust_format())
            await self._notify_callbacks('orderbook_metrics', rust_metrics)
        
        # Queue QuestDB lines
        questdb_lines = orderbook.to_questdb_format()
        await self._notify_callbacks('orderbook', {
            'symbol': symbol,
            'orderbook': orderbook,
            'questdb': questdb_lines
        })
    
    async def _handle_ticker(self, data: Dict):
        """Process ticker update"""
        symbol = data['product_id'].replace('-', '/')
        
        ticker_data = {
            'symbol': symbol,
            'bid': float(data.get('best_bid', 0)),
            'ask': float(data.get('best_ask', 0)),
            'last': float(data.get('price', 0)),
            'volume': float(data.get('volume_24h', 0)),
            'timestamp': int(time.time() * 1e9)
        }
        
        self.tick_buffer.append(ticker_data)
        await self._notify_callbacks('ticker', ticker_data)
    
    async def _handle_match(self, data: Dict):
        """Process trade/match message"""
        trade = Trade(
            symbol=data['product_id'].replace('-', '/'),
            trade_id=data['trade_id'],
            price=Decimal(data['price']),
            size=Decimal(data['size']),
            side=data['side'],
            timestamp=int(datetime.fromisoformat(
                data['time'].replace('Z', '+00:00')
            ).timestamp() * 1e9)
        )
        
        # Store in history
        self.trade_history.append(trade)
        
        # Format for QuestDB
        questdb_line = trade.to_questdb_format()
        await self._notify_callbacks('trade', {
            'trade': trade,
            'questdb': questdb_line
        })
    
    async def _handle_order_update(self, data: Dict):
        """Process order status update"""
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
    
    async def subscribe_orderbook(self, symbols: List[str], depth: int = ORDER_BOOK_DEPTH):
        """Subscribe to orderbook updates"""
        if RUST_AVAILABLE and self.rust_handler:
            # Rust handler manages subscriptions
            for symbol in symbols:
                await self.rust_handler.subscribe(
                    symbol.replace('/', '-'),
                    ['level2', 'ticker', 'matches']
                )
        else:
            # Python WebSocket subscription
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
                p95_latency = np.percentile(list(self.latency_tracker), 95)
                p99_latency = np.percentile(list(self.latency_tracker), 99)
                
                logger.info(
                    f"{CyberColors.NEON_CYAN}Performance - "
                    f"Avg: {avg_latency:.1f}ms, P95: {p95_latency:.1f}ms, P99: {p99_latency:.1f}ms"
                    f"{CyberColors.RESET}"
                )
                
                # Get Rust metrics if available
                if RUST_AVAILABLE and self.rust_latency_tracker:
                    rust_stats = self.rust_latency_tracker.get_stats()
                    logger.info(f"Rust performance: {rust_stats}")
    
    async def _latency_monitor(self):
        """Monitor connection health and latency"""
        while True:
            await asyncio.sleep(30)
            
            # Check heartbeat
            if time.time() - self.last_heartbeat > 60:
                logger.warning(f"{CyberColors.NEON_RED}WebSocket heartbeat timeout{CyberColors.RESET}")
                if self.ws_connection:
                    await self.ws_connection.close()
    
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
        """Register event callback (backward compatibility)"""
        self.subscribe(event_type, callback)
    
    def subscribe(self, event: str, callback: Callable):
        """Subscribe to connector events"""
        self.callbacks[event].append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connector performance statistics"""
        latencies = list(self.latency_tracker) if self.latency_tracker else [0]
        
        stats = {
            'exchange': 'coinbase',
            'message_count': self.message_count,
            'active_orders': len(self.active_orders),
            'orderbooks': len(self.orderbooks),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'tick_buffer_size': len(self.tick_buffer),
            'rust_enabled': RUST_AVAILABLE,
            'advanced_api': self.use_advanced_api,
            'aws_optimized': self.use_aws_endpoints
        }
        
        # Add Rust stats if available
        if RUST_AVAILABLE and self.rust_latency_tracker:
            stats['rust_metrics'] = self.rust_latency_tracker.get_stats()
        
        return stats
    
    async def close(self):
        """Clean shutdown"""
        logger.info(f"{CyberColors.NEURAL_PURPLE}Closing Coinbase connector...{CyberColors.RESET}")
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            symbol = self.active_orders[order_id].get('symbol', '')
            await self.cancel_order(order_id, symbol)
        
        # Close WebSocket
        if self.ws_connection:
            await self.ws_connection.close()
        
        # Close Rust handler
        if RUST_AVAILABLE and self.rust_handler:
            self.rust_handler.shutdown()
        
        # Close HTTP session
        if self.session:
            await self.session.close()
        
        # Close exchange
        await self.exchange.close()
        
        logger.info(f"{CyberColors.NEURAL_PURPLE}Coinbase connector closed{CyberColors.RESET}")
    
    async def shutdown(self):
        """Alias for close (new interface compatibility)"""
        await self.close()


# Rust integration stubs (would be replaced by actual PyO3 module)
if not RUST_AVAILABLE:
    class RustWebsocketHandler:
        """Placeholder for Rust WebSocket handler"""
        def set_callback(self, callback): pass
        async def connect(self, endpoint, headers): pass
        async def subscribe(self, symbol, channels): pass
        def shutdown(self): pass
    
    class RustOrderBook:
        """Placeholder for Rust order book processor"""
        def calculate_metrics(self, data):
            return {
                'spread': 0.0,
                'mid_price': 0.0,
                'imbalance': 0.0,
                'depth': 0,
                'liquidity_score': 0.0
            }
    
    class RustLatencyTracker:
        """Placeholder for Rust latency tracker"""
        def record(self, latency_ms): pass
        def get_stats(self):
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'count': 0
            }


# Example usage
if __name__ == "__main__":
    async def main():
        # Get config
        config_loader = get_config_loader()
        await config_loader.initialize()
        
        # Create connector
        config = {
            'api_key': config_loader.get('exchanges.coinbase.api_key'),
            'api_secret': config_loader.get('exchanges.coinbase.api_secret'),
            'passphrase': config_loader.get('exchanges.coinbase.passphrase'),
            'testnet': config_loader.get('exchanges.coinbase.testnet', False),
            'use_advanced_api': config_loader.get('exchanges.coinbase.use_advanced_api', True),
            'use_aws_endpoints': config_loader.get('exchanges.coinbase.use_aws_endpoints', True)
        }
        
        connector = CoinbaseConnector(config)
        
        # Subscribe to events
        async def on_ticker(data):
            print(f"Ticker: {data['symbol']} @ ${data['last']}")
        
        async def on_trade(data):
            trade = data['trade']
            print(f"Trade: {trade.side} {trade.size} {trade.symbol} @ ${trade.price}")
        
        async def on_orderbook(data):
            orderbook = data['orderbook']
            spread = orderbook.get_spread()
            if spread:
                print(f"Orderbook: {orderbook.symbol} spread=${spread}")
        
        connector.subscribe('ticker', on_ticker)
        connector.subscribe('trade', on_trade)
        connector.subscribe('orderbook', on_orderbook)
        
        # Initialize
        try:
            await connector.initialize()
            
            # Subscribe to symbols
            await connector.subscribe_orderbook(['BTC/USD', 'ETH/USD'])
            
            # Place a test order (if in testnet)
            if config['testnet']:
                order = await connector.place_order(
                    symbol='BTC/USD',
                    side='buy',
                    order_type='limit',
                    amount=0.001,
                    price=50000,
                    params={'post_only': True}
                )
                print(f"Test order placed: {order}")
            
            # Get statistics
            stats = connector.get_statistics()
            print(f"\nConnector statistics: {stats}")
            
            # Run for a while
            await asyncio.sleep(300)
            
        finally:
            await connector.shutdown()
    
    asyncio.run(main())
