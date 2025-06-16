# nexlify/connectors/exchange_connector.py
"""
Exchange Connectors - Neural Interfaces to the Global Markets
Connect to any exchange like a netrunner jacks into cyberspace
Support for Binance, Kraken, and more Night City approved exchanges
"""

import asyncio
import aiohttp
import websockets
import hmac
import hashlib
import time
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timezone
from decimal import Decimal
import logging
from urllib.parse import urlencode
from collections import defaultdict
import ccxt.async_support as ccxt

from database.models import Market, Symbol, Candle, OrderStatus, OrderType, OrderSide
from monitoring.sentinel import get_sentinel, MetricType

logger = logging.getLogger("nexlify.connectors")

class ExchangeEvent:
    """Base class for exchange events - the digital signals"""
    pass

@dataclass
class TradeEvent(ExchangeEvent):
    """Trade execution event - someone pulled the trigger"""
    symbol: str
    price: Decimal
    quantity: Decimal
    side: str  # buy/sell
    timestamp: datetime
    trade_id: str

@dataclass
class OrderBookEvent(ExchangeEvent):
    """Order book update - the market's neural state"""
    symbol: str
    bids: List[Tuple[Decimal, Decimal]]
    asks: List[Tuple[Decimal, Decimal]]
    timestamp: datetime

@dataclass
class TickerEvent(ExchangeEvent):
    """Ticker update - the market's pulse"""
    symbol: str
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    timestamp: datetime

class ExchangeConnector(ABC):
    """
    Abstract base for exchange connectors - the neural interface standard
    Every exchange speaks different languages, but we translate them all
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.name = self.__class__.__name__.replace("Connector", "")
        
        # WebSocket management
        self.ws_connection = None
        self.ws_subscriptions = set()
        self.event_handlers: Dict[type, List[Callable]] = defaultdict(list)
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Monitoring
        self.sentinel = get_sentinel()
        
        # Connection state
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
    
    @abstractmethod
    async def connect(self):
        """Establish connection to exchange - jack in"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from exchange - jack out"""
        pass
    
    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str]):
        """Subscribe to market data streams"""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Place an order on the exchange"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> TickerEvent:
        """Get current ticker data"""
        pass
    
    def add_event_handler(self, event_type: type, handler: Callable):
        """Register event handler - wire up the neural pathways"""
        self.event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: ExchangeEvent):
        """Emit event to all handlers - broadcast the signal"""
        for handler in self.event_handlers[type(event)]:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    async def reconnect(self):
        """Reconnect to exchange - re-establish the neural link"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts reached for {self.name}")
            return False
        
        self.reconnect_attempts += 1
        wait_time = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
        
        logger.info(f"Reconnecting to {self.name} in {wait_time}s (attempt {self.reconnect_attempts})")
        await asyncio.sleep(wait_time)
        
        try:
            await self.connect()
            self.reconnect_attempts = 0
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

class BinanceConnector(ExchangeConnector):
    """
    Binance Connector - Interface to the biggest crypto casino in Night City
    Fast, liquid, and dangerous if you're not careful
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        self.ws_url = "wss://testnet.binance.vision/ws" if testnet else "wss://stream.binance.com:9443/ws"
        
        # Initialize CCXT for easier API interaction
        self.ccxt_exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'testnet': testnet
            }
        })
    
    async def connect(self):
        """Connect to Binance - enter the dragon's lair"""
        try:
            # Test connection with account info
            await self.ccxt_exchange.fetch_balance()
            
            # Establish WebSocket connection
            self.ws_connection = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # Start message handler
            asyncio.create_task(self._handle_ws_messages())
            
            logger.info(f"Connected to Binance {'testnet' if self.testnet else 'mainnet'}")
            
            # Record metric
            self.sentinel.record_metric(
                MetricType.API,
                f"exchange_connection_{self.name}",
                {"status": "connected", "testnet": self.testnet}
            )
            
        except Exception as e:
            logger.error(f"Binance connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Binance - exit cleanly"""
        self.is_connected = False
        
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        await self.ccxt_exchange.close()
        
        logger.info("Disconnected from Binance")
    
    async def subscribe_market_data(self, symbols: List[str]):
        """Subscribe to Binance market streams - tap into the data flow"""
        if not self.ws_connection:
            raise ConnectionError("WebSocket not connected")
        
        # Convert symbols to Binance format (lowercase, no slash)
        streams = []
        for symbol in symbols:
            binance_symbol = symbol.replace("/", "").lower()
            streams.extend([
                f"{binance_symbol}@ticker",      # 24hr ticker
                f"{binance_symbol}@depth20@100ms", # Order book
                f"{binance_symbol}@trade"         # Trades
            ])
        
        # Subscribe message
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time())
        }
        
        await self.ws_connection.send(json.dumps(subscribe_msg))
        self.ws_subscriptions.update(symbols)
        
        logger.info(f"Subscribed to Binance streams for: {symbols}")
    
    async def _handle_ws_messages(self):
        """Handle incoming WebSocket messages - process the data stream"""
        while self.is_connected and self.ws_connection:
            try:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                
                # Skip subscription confirmations
                if "result" in data:
                    continue
                
                # Process stream data
                if "stream" in data:
                    stream_type = data["stream"].split("@")[1].split("@")[0]
                    stream_data = data["data"]
                    
                    if stream_type == "ticker":
                        await self._process_ticker(stream_data)
                    elif stream_type == "depth20":
                        await self._process_orderbook(stream_data)
                    elif stream_type == "trade":
                        await self._process_trade(stream_data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Binance WebSocket connection closed")
                if self.is_connected:
                    await self.reconnect()
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
    
    async def _process_ticker(self, data: Dict):
        """Process ticker data - the market's vital signs"""
        symbol = self._convert_symbol_from_binance(data['s'])
        
        event = TickerEvent(
            symbol=symbol,
            bid=Decimal(data['b']),  # Best bid price
            ask=Decimal(data['a']),  # Best ask price
            last=Decimal(data['c']),  # Last price
            volume_24h=Decimal(data['v']),  # 24h volume
            timestamp=datetime.fromtimestamp(data['E'] / 1000, tz=timezone.utc)
        )
        
        await self.emit_event(event)
    
    async def _process_orderbook(self, data: Dict):
        """Process order book data - see the market depth"""
        symbol = self._convert_symbol_from_binance(data['s'])
        
        bids = [(Decimal(price), Decimal(qty)) for price, qty in data['bids']]
        asks = [(Decimal(price), Decimal(qty)) for price, qty in data['asks']]
        
        event = OrderBookEvent(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(data['E'] / 1000, tz=timezone.utc)
        )
        
        await self.emit_event(event)
    
    async def _process_trade(self, data: Dict):
        """Process trade data - witness the deals"""
        symbol = self._convert_symbol_from_binance(data['s'])
        
        event = TradeEvent(
            symbol=symbol,
            price=Decimal(data['p']),
            quantity=Decimal(data['q']),
            side="buy" if data['m'] else "sell",  # m = true means buyer is maker
            timestamp=datetime.fromtimestamp(data['T'] / 1000, tz=timezone.utc),
            trade_id=str(data['t'])
        )
        
        await self.emit_event(event)
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """
        Place order on Binance - execute the trade
        Returns order info including exchange order ID
        """
        try:
            # Convert to CCXT format
            ccxt_side = 'buy' if side == OrderSide.BUY else 'sell'
            ccxt_type = order_type.value.lower()
            
            # Create order
            if order_type == OrderType.MARKET:
                order = await self.ccxt_exchange.create_order(
                    symbol=symbol,
                    type=ccxt_type,
                    side=ccxt_side,
                    amount=float(quantity)
                )
            else:  # Limit order
                order = await self.ccxt_exchange.create_order(
                    symbol=symbol,
                    type=ccxt_type,
                    side=ccxt_side,
                    amount=float(quantity),
                    price=float(price)
                )
            
            # Convert response
            return {
                'exchange_order_id': order['id'],
                'status': self._convert_order_status(order['status']),
                'filled_quantity': Decimal(str(order['filled'])),
                'average_price': Decimal(str(order['average'])) if order['average'] else None,
                'timestamp': datetime.fromtimestamp(order['timestamp'] / 1000, tz=timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Binance order placement failed: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance - abort the mission"""
        try:
            await self.ccxt_exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status from Binance - check the mission status"""
        try:
            order = await self.ccxt_exchange.fetch_order(order_id, symbol)
            
            return {
                'exchange_order_id': order['id'],
                'status': self._convert_order_status(order['status']),
                'filled_quantity': Decimal(str(order['filled'])),
                'remaining_quantity': Decimal(str(order['remaining'])),
                'average_price': Decimal(str(order['average'])) if order['average'] else None,
                'fee': Decimal(str(order['fee']['cost'])) if order['fee'] else Decimal(0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            raise
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get account balance - count your eddies"""
        try:
            balance = await self.ccxt_exchange.fetch_balance()
            
            return {
                asset: Decimal(str(bal['free'] + bal['used']))
                for asset, bal in balance['total'].items()
                if bal > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> TickerEvent:
        """Get current ticker - quick market check"""
        try:
            ticker = await self.ccxt_exchange.fetch_ticker(symbol)
            
            return TickerEvent(
                symbol=symbol,
                bid=Decimal(str(ticker['bid'])),
                ask=Decimal(str(ticker['ask'])),
                last=Decimal(str(ticker['last'])),
                volume_24h=Decimal(str(ticker['baseVolume'])),
                timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000, tz=timezone.utc)
            )
            
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
            raise
    
    def _convert_symbol_from_binance(self, binance_symbol: str) -> str:
        """Convert Binance symbol format to standard - translation matrix"""
        # Binance uses BTCUSDT, we use BTC/USDT
        # Simple conversion for common pairs
        for quote in ['USDT', 'USDC', 'BUSD', 'BTC', 'ETH', 'BNB']:
            if binance_symbol.endswith(quote):
                base = binance_symbol[:-len(quote)]
                return f"{base}/{quote}"
        
        return binance_symbol
    
    def _convert_order_status(self, ccxt_status: str) -> OrderStatus:
        """Convert CCXT order status to our format"""
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED
        }
        return status_map.get(ccxt_status, OrderStatus.PENDING)

class KrakenConnector(ExchangeConnector):
    """
    Kraken Connector - The European crypto fortress
    Secure, compliant, and professional
    """
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        
        # Kraken doesn't have a testnet, so we'll use the main API in read-only mode for testing
        self.base_url = "https://api.kraken.com"
        self.ws_url = "wss://ws.kraken.com"
        
        # Initialize CCXT
        self.ccxt_exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
    
    async def connect(self):
        """Connect to Kraken - enter the vault"""
        try:
            # Test connection
            await self.ccxt_exchange.fetch_balance()
            
            # Establish WebSocket
            self.ws_connection = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # Start handler
            asyncio.create_task(self._handle_ws_messages())
            
            logger.info("Connected to Kraken")
            
        except Exception as e:
            logger.error(f"Kraken connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Kraken"""
        self.is_connected = False
        
        if self.ws_connection:
            await self.ws_connection.close()
        
        await self.ccxt_exchange.close()
    
    async def subscribe_market_data(self, symbols: List[str]):
        """Subscribe to Kraken market data"""
        if not self.ws_connection:
            raise ConnectionError("WebSocket not connected")
        
        # Convert symbols to Kraken format
        kraken_pairs = []
        for symbol in symbols:
            # Kraken uses XBT for BTC
            kraken_symbol = symbol.replace("BTC/", "XBT/")
            kraken_pairs.append(kraken_symbol)
        
        # Subscribe to ticker and trades
        subscribe_msg = {
            "event": "subscribe",
            "pair": kraken_pairs,
            "subscription": {
                "name": "ticker"
            }
        }
        
        await self.ws_connection.send(json.dumps(subscribe_msg))
        
        # Also subscribe to trades
        trade_sub = {
            "event": "subscribe",
            "pair": kraken_pairs,
            "subscription": {
                "name": "trade"
            }
        }
        
        await self.ws_connection.send(json.dumps(trade_sub))
        
        self.ws_subscriptions.update(symbols)
        logger.info(f"Subscribed to Kraken streams for: {symbols}")
    
    async def _handle_ws_messages(self):
        """Handle Kraken WebSocket messages"""
        while self.is_connected and self.ws_connection:
            try:
                message = await self.ws_connection.recv()
                data = json.loads(message)
                
                # Skip heartbeats and system messages
                if isinstance(data, dict):
                    continue
                
                # Process channel data
                if isinstance(data, list) and len(data) >= 4:
                    channel_id = data[0]
                    payload = data[1]
                    channel_name = data[2]
                    pair = data[3]
                    
                    if channel_name == "ticker":
                        await self._process_kraken_ticker(payload, pair)
                    elif channel_name == "trade":
                        await self._process_kraken_trades(payload, pair)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Kraken WebSocket closed")
                if self.is_connected:
                    await self.reconnect()
                break
            except Exception as e:
                logger.error(f"Kraken message error: {e}")
    
    async def _process_kraken_ticker(self, data: Dict, pair: str):
        """Process Kraken ticker data"""
        # Convert pair back to standard format
        symbol = pair.replace("XBT", "BTC")
        
        event = TickerEvent(
            symbol=symbol,
            bid=Decimal(data['b'][0]),  # Best bid
            ask=Decimal(data['a'][0]),  # Best ask
            last=Decimal(data['c'][0]),  # Last trade
            volume_24h=Decimal(data['v'][1]),  # 24h volume
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.emit_event(event)
    
    async def _process_kraken_trades(self, trades: List, pair: str):
        """Process Kraken trade data"""
        symbol = pair.replace("XBT", "BTC")
        
        for trade in trades:
            event = TradeEvent(
                symbol=symbol,
                price=Decimal(trade[0]),
                quantity=Decimal(trade[1]),
                side="buy" if trade[3] == "b" else "sell",
                timestamp=datetime.fromtimestamp(float(trade[2]), tz=timezone.utc),
                trade_id=str(int(float(trade[2]) * 1000000))  # Create unique ID
            )
            
            await self.emit_event(event)
    
    # Implement remaining abstract methods similar to Binance...
    async def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                         quantity: Decimal, price: Optional[Decimal] = None) -> Dict[str, Any]:
        """Place order on Kraken"""
        # Similar implementation to Binance
        pass
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Kraken"""
        pass
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status from Kraken"""
        pass
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get Kraken account balance"""
        pass
    
    async def get_ticker(self, symbol: str) -> TickerEvent:
        """Get Kraken ticker"""
        pass

class MockExchangeConnector(ExchangeConnector):
    """
    Mock Exchange Connector - For testing and paper trading
    Simulates a real exchange without risking real eddies
    """
    
    def __init__(self, api_key: str = "mock", api_secret: str = "mock", testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)
        
        # Simulated data
        self.balance = {
            "USDT": Decimal("10000"),
            "BTC": Decimal("0.5"),
            "ETH": Decimal("5")
        }
        
        self.orders = {}
        self.next_order_id = 1000
        
        # Base prices for simulation
        self.base_prices = {
            "BTC/USDT": Decimal("50000"),
            "ETH/USDT": Decimal("3000"),
            "SOL/USDT": Decimal("100")
        }
    
    async def connect(self):
        """Connect to mock exchange - instant connection"""
        self.is_connected = True
        logger.info("Connected to Mock Exchange")
        
        # Start price simulator
        asyncio.create_task(self._simulate_market_data())
    
    async def disconnect(self):
        """Disconnect from mock exchange"""
        self.is_connected = False
        logger.info("Disconnected from Mock Exchange")
    
    async def subscribe_market_data(self, symbols: List[str]):
        """Subscribe to mock market data"""
        self.ws_subscriptions.update(symbols)
        logger.info(f"Subscribed to mock streams for: {symbols}")
    
    async def _simulate_market_data(self):
        """Simulate market data - create the matrix"""
        while self.is_connected:
            for symbol in self.ws_subscriptions:
                if symbol in self.base_prices:
                    # Add some randomness
                    base = self.base_prices[symbol]
                    change = Decimal(str(np.random.normal(0, 0.001)))
                    new_price = base * (1 + change)
                    self.base_prices[symbol] = new_price
                    
                    # Emit ticker event
                    spread = new_price * Decimal("0.0001")
                    event = TickerEvent(
                        symbol=symbol,
                        bid=new_price - spread/2,
                        ask=new_price + spread/2,
                        last=new_price,
                        volume_24h=Decimal(str(np.random.uniform(1000, 10000))),
                        timestamp=datetime.now(timezone.utc)
                    )
                    
                    await self.emit_event(event)
            
            await asyncio.sleep(1)  # Update every second
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Place mock order - simulate the trade"""
        order_id = f"MOCK-{self.next_order_id}"
        self.next_order_id += 1
        
        # Get current price
        current_price = self.base_prices.get(symbol, Decimal("100"))
        
        # Simulate immediate fill for market orders
        if order_type == OrderType.MARKET:
            execution_price = current_price * (Decimal("1.0001") if side == OrderSide.BUY else Decimal("0.9999"))
            status = OrderStatus.FILLED
            filled = quantity
        else:
            execution_price = price
            status = OrderStatus.OPEN
            filled = Decimal(0)
        
        # Update balance for filled orders
        if status == OrderStatus.FILLED:
            base, quote = symbol.split("/")
            if side == OrderSide.BUY:
                self.balance[quote] -= quantity * execution_price
                self.balance[base] = self.balance.get(base, Decimal(0)) + quantity
            else:
                self.balance[base] -= quantity
                self.balance[quote] = self.balance.get(quote, Decimal(0)) + quantity * execution_price
        
        order_info = {
            'exchange_order_id': order_id,
            'status': status,
            'filled_quantity': filled,
            'average_price': execution_price if filled > 0 else None,
            'timestamp': datetime.now(timezone.utc)
        }
        
        self.orders[order_id] = order_info
        
        return order_info
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel mock order"""
        if order_id in self.orders:
            self.orders[order_id]['status'] = OrderStatus.CANCELLED
            return True
        return False
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get mock order status"""
        return self.orders.get(order_id, {
            'exchange_order_id': order_id,
            'status': OrderStatus.REJECTED,
            'filled_quantity': Decimal(0),
            'average_price': None
        })
    
    async def get_balance(self) -> Dict[str, Decimal]:
        """Get mock balance"""
        return self.balance.copy()
    
    async def get_ticker(self, symbol: str) -> TickerEvent:
        """Get mock ticker"""
        price = self.base_prices.get(symbol, Decimal("100"))
        spread = price * Decimal("0.0001")
        
        return TickerEvent(
            symbol=symbol,
            bid=price - spread/2,
            ask=price + spread/2,
            last=price,
            volume_24h=Decimal("5000"),
            timestamp=datetime.now(timezone.utc)
        )

class ExchangeFactory:
    """
    Exchange Factory - Creates the right connector for your needs
    Like a fixer who knows all the right people
    """
    
    @staticmethod
    def create_connector(
        exchange_name: str,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ) -> ExchangeConnector:
        """Create exchange connector by name"""
        
        connectors = {
            'binance': BinanceConnector,
            'kraken': KrakenConnector,
            'mock': MockExchangeConnector
        }
        
        connector_class = connectors.get(exchange_name.lower())
        
        if not connector_class:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        
        return connector_class(api_key, api_secret, testnet)

class RateLimiter:
    """
    Rate Limiter - Don't get banned from the casino
    Manages API request rates to stay under limits
    """
    
    def __init__(self):
        self.requests = defaultdict(deque)
        self.limits = {
            'binance': {'requests': 1200, 'window': 60},  # 1200/min
            'kraken': {'requests': 15, 'window': 3},       # 15/3s
            'default': {'requests': 10, 'window': 1}       # Conservative default
        }
    
    async def check_limit(self, exchange: str, endpoint: str = "default") -> bool:
        """Check if we can make a request"""
        now = time.time()
        key = f"{exchange}:{endpoint}"
        
        # Get limits for exchange
        limits = self.limits.get(exchange, self.limits['default'])
        
        # Clean old requests
        cutoff = now - limits['window']
        while self.requests[key] and self.requests[key][0] < cutoff:
            self.requests[key].popleft()
        
        # Check if under limit
        if len(self.requests[key]) < limits['requests']:
            self.requests[key].append(now)
            return True
        
        return False
    
    async def wait_if_needed(self, exchange: str, endpoint: str = "default"):
        """Wait if rate limited"""
        while not await self.check_limit(exchange, endpoint):
            await asyncio.sleep(0.1)
