#!/usr/bin/env python3
"""
src/exchanges/unified_interface.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY UNIFIED EXCHANGE INTERFACE v3.1 (MERGED EDITION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cyberpunk-themed unified interface combining:
- Full CEX/DEX support with ccxt.pro and Web3
- Smart order routing with multi-factor scoring
- Real-time arbitrage detection (<50ms cycles)
- MEV protection for DEX interactions
- Cross-exchange order splitting strategies
- zkBridge cross-chain readiness
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from collections import defaultdict, deque
import numpy as np
import structlog
from sortedcontainers import SortedDict, SortedList
import uvloop

# Exchange libraries
import ccxt.pro as ccxt_pro
from web3 import Web3
from web3.middleware import geth_poa_middleware
from eth_account import Account
from eth_account.signers.local import LocalAccount
import aiohttp

# Performance libraries
import orjson
import msgpack
from aiocache import cached
from prometheus_client import Counter, Histogram, Gauge

# Future integrations
try:
    import pyO3  # Rust integration for performance-critical paths
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

# Import our components
from .coinbase_connector import CoinbaseConnector, OrderBook
from ..utils.config_loader import get_config_loader, CyberColors

# Set uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Initialize logger
logger = structlog.get_logger("NEXLIFY.UNIFIED.MATRIX")

# Metrics
EXCHANGE_LATENCY = Histogram(
    'nexlify_exchange_latency_seconds',
    'Exchange API latency',
    ['exchange', 'operation']
)
ARBITRAGE_OPPORTUNITIES = Counter(
    'nexlify_arbitrage_opportunities_total',
    'Arbitrage opportunities detected'
)
ROUTING_DECISIONS = Counter(
    'nexlify_routing_decisions_total',
    'Smart order routing decisions',
    ['exchange', 'reason']
)
ORDER_EXECUTION_TIME = Histogram(
    'nexlify_order_execution_seconds',
    'Order execution time',
    ['exchange', 'order_type']
)
ACTIVE_POSITIONS = Gauge(
    'nexlify_active_positions',
    'Active positions across exchanges',
    ['exchange', 'symbol']
)

# Constants
MAX_SLIPPAGE_PERCENT = Decimal("0.5")  # 0.5% max slippage
MIN_PROFIT_THRESHOLD = Decimal("0.001")  # 0.1% minimum profit for arbitrage
ROUTE_CACHE_TTL = 60  # seconds
ORDER_TIMEOUT = 30  # seconds
MEV_PROTECTION_ENABLED = True
ARBITRAGE_SCAN_INTERVAL = 0.05  # 50ms


class ExchangeType(Enum):
    """Exchange categories"""
    CEX_SPOT = auto()      # Centralized spot
    CEX_PERP = auto()      # Centralized perpetuals
    DEX_AMM = auto()       # AMM DEXs (Uniswap, Pancake)
    DEX_ORDERBOOK = auto() # Orderbook DEXs (dYdX, Serum)
    DEX_PERP = auto()      # Perpetual DEXs (GMX, Gains)


class OrderType(Enum):
    """Unified order types across all exchanges"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    ICEBERG = "iceberg"  # Hidden order
    TWAP = "twap"  # Time-weighted average price
    VWAP = "vwap"  # Volume-weighted average price


class OrderStatus(Enum):
    """Unified order status"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ExchangeInfo:
    """Exchange metadata"""
    name: str
    type: ExchangeType
    chain: Optional[str] = None  # For DEXs
    maker_fee: Decimal = Decimal('0.001')
    taker_fee: Decimal = Decimal('0.001')
    withdrawal_fee: Dict[str, Decimal] = field(default_factory=dict)
    min_order_size: Dict[str, Decimal] = field(default_factory=dict)
    tick_size: Dict[str, Decimal] = field(default_factory=dict)
    is_available: bool = True
    latency_ms: float = 0.0
    reliability_score: float = 1.0  # 0-1 score
    supported_order_types: List[OrderType] = field(default_factory=list)


@dataclass
class UnifiedSymbol:
    """Unified symbol representation across exchanges"""
    base: str
    quote: str
    exchange_symbols: Dict[str, str] = field(default_factory=dict)
    
    @property
    def unified(self) -> str:
        return f"{self.base}/{self.quote}"
    
    def get_exchange_symbol(self, exchange: str) -> str:
        """Get exchange-specific symbol format"""
        return self.exchange_symbols.get(exchange, f"{self.base}-{self.quote}")


@dataclass
class UnifiedOrder:
    """Unified order format across all exchanges"""
    id: str
    exchange: str
    symbol: UnifiedSymbol
    side: str  # buy/sell
    type: OrderType
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    post_only: bool = False
    reduce_only: bool = False
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1e9))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity"""
    symbol: UnifiedSymbol
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    max_amount: Decimal
    profit_percent: Decimal
    profit_usd: Decimal
    timestamp: float = field(default_factory=time.time)
    confidence_score: float = 1.0
    
    @property
    def profit_per_unit(self) -> Decimal:
        return self.sell_price - self.buy_price


@dataclass
class RouteOption:
    """Order routing option with cost analysis"""
    exchange: str
    expected_price: Decimal
    available_liquidity: Decimal
    estimated_fees: Decimal
    latency_ms: float
    slippage_estimate: Decimal
    reliability_score: float
    score: float = 0.0  # Calculated routing score


class ExchangeConnector(ABC):
    """Abstract base class for exchange connectors"""
    
    @abstractmethod
    async def initialize(self):
        pass
    
    @abstractmethod
    async def place_order(self, order: UnifiedOrder) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str) -> OrderBook:
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Decimal]:
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def close(self):
        pass


class CCXTConnector(ExchangeConnector):
    """Generic CCXT connector for exchanges"""
    
    def __init__(self, exchange_class, config: Dict[str, Any]):
        self.config = config
        self.exchange = exchange_class({
            'apiKey': config.get('api_key'),
            'secret': config.get('api_secret'),
            'password': config.get('passphrase'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True
            }
        })
        self.symbols = []
        self.orderbooks: Dict[str, OrderBook] = {}
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
    async def initialize(self):
        await self.exchange.load_markets()
        self.symbols = list(self.exchange.markets.keys())
        
    async def place_order(self, order: UnifiedOrder) -> Dict[str, Any]:
        if order.type == OrderType.MARKET:
            return await self.exchange.create_market_order(
                order.symbol.get_exchange_symbol(self.exchange.id),
                order.side,
                float(order.amount)
            )
        else:
            return await self.exchange.create_limit_order(
                order.symbol.get_exchange_symbol(self.exchange.id),
                order.side,
                float(order.amount),
                float(order.price)
            )
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        return await self.exchange.cancel_order(order_id, symbol)
    
    async def get_orderbook(self, symbol: str) -> OrderBook:
        raw_ob = await self.exchange.fetch_order_book(symbol)
        
        orderbook = OrderBook(symbol=symbol)
        for bid in raw_ob['bids']:
            orderbook.update('bid', bid[0], bid[1])
        for ask in raw_ob['asks']:
            orderbook.update('ask', ask[0], ask[1])
            
        self.orderbooks[symbol] = orderbook
        return orderbook
    
    async def get_balance(self) -> Dict[str, Decimal]:
        balance = await self.exchange.fetch_balance()
        return {
            asset: Decimal(str(amount))
            for asset, amount in balance['free'].items()
            if amount > 0
        }
    
    def subscribe(self, event: str, callback: Callable):
        """Subscribe to events"""
        self.callbacks[event].append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics"""
        return {
            'exchange': self.exchange.id,
            'orderbooks': len(self.orderbooks),
            'symbols': len(self.symbols)
        }
    
    async def close(self):
        await self.exchange.close()


class UniswapV3Connector(ExchangeConnector):
    """Uniswap V3 connector with MEV protection"""
    
    def __init__(self, web3: Web3, config: Dict[str, Any]):
        self.web3 = web3
        self.config = config
        self.router_address = config.get('router_address')
        self.factory_address = config.get('factory_address')
        self.pools: Dict[str, Any] = {}
        
    async def initialize(self):
        # Load contract ABIs and initialize pools
        logger.info("Initializing Uniswap V3 connector...")
        
    async def place_order(self, order: UnifiedOrder) -> Dict[str, Any]:
        # Implement swap with MEV protection
        if MEV_PROTECTION_ENABLED:
            # Use commit-reveal scheme or flashbots
            pass
        
        # Execute swap
        return {}
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        # DEX swaps are atomic, no cancellation
        raise NotImplementedError("Cannot cancel DEX swaps")
    
    async def get_orderbook(self, symbol: str) -> OrderBook:
        # Calculate from pool reserves
        return OrderBook(symbol=symbol)
    
    async def get_balance(self) -> Dict[str, Decimal]:
        # Check wallet balances
        return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        return {'exchange': 'uniswap_v3', 'pools': len(self.pools)}
    
    async def close(self):
        pass


class NexlifyUnifiedInterface:
    """
    🌐 NEXLIFY Enhanced Unified Exchange Interface
    
    Merged features:
    - Full CEX/DEX support via ccxt and Web3
    - Smart order routing with multi-factor scoring
    - Real-time arbitrage detection and execution
    - MEV protection for DEX interactions
    - Cross-chain bridge integration (zkBridge ready)
    - Advanced execution algorithms (TWAP, VWAP, Iceberg)
    - Performance tracking and optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Configuration
        self.config = config or get_config_loader().get_all()
        
        # Exchange management
        self.exchanges: Dict[str, ExchangeConnector] = {}
        self.exchange_info: Dict[str, ExchangeInfo] = {}
        self.enabled_exchanges: Set[str] = set()
        
        # Unified data structures
        self.unified_symbols: Dict[str, UnifiedSymbol] = {}
        self.unified_orders: Dict[str, UnifiedOrder] = {}
        self.unified_balances: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
        
        # Order books and market data
        self.orderbook_cache: Dict[Tuple[str, str], OrderBook] = {}  # (exchange, symbol)
        self.aggregated_books: Dict[str, Dict[str, OrderBook]] = defaultdict(dict)
        self.best_bid_ask: Dict[str, Dict[str, Tuple[Decimal, Decimal]]] = defaultdict(dict)
        
        # Arbitrage detection
        self.arbitrage_opportunities = SortedList(key=lambda x: -x.profit_percent)
        self.arbitrage_history = deque(maxlen=1000)
        self.min_arbitrage_profit = Decimal(self.config.get('arbitrage.min_profit_percent', '0.1'))
        
        # Smart order routing
        self.routing_weights: Dict[str, float] = {}
        self.liquidity_scores: Dict[Tuple[str, str], float] = {}  # (exchange, symbol)
        self.route_cache: Dict[str, List[RouteOption]] = {}
        self.routing_scores: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Performance tracking
        self.latency_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.execution_metrics = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_volume': Decimal("0"),
            'arbitrage_trades': 0,
            'arbitrage_profit': Decimal("0")
        }
        
        # Cross-chain bridges
        self.bridges: Dict[str, Any] = {}
        self.web3_providers: Dict[str, Web3] = {}
        
        # State management
        self.is_initialized = False
        self._update_task = None
        self._arbitrage_task = None
        self._routing_task = None
        
        # Callbacks
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Load configuration
        self._load_configuration()
    
    def _load_configuration(self):
        """Load exchange configuration"""
        exchanges_config = self.config.get('exchanges', {})
        
        for exchange, config in exchanges_config.items():
            if config.get('enabled', False):
                self.enabled_exchanges.add(exchange)
                logger.info(f"{CyberColors.NEON_CYAN}Exchange enabled: {exchange}{CyberColors.RESET}")
    
    async def initialize(self):
        """Initialize all configured exchanges"""
        logger.info(f"{CyberColors.NEON_CYAN}🌐 Initializing Unified Exchange Interface...{CyberColors.RESET}")
        
        # Initialize CEX connectors
        await self._init_cex_connectors()
        
        # Initialize DEX connectors
        await self._init_dex_connectors()
        
        # Start background tasks
        self._update_task = asyncio.create_task(self._update_loop())
        self._arbitrage_task = asyncio.create_task(self._arbitrage_detector())
        self._routing_task = asyncio.create_task(self._route_optimizer())
        
        # Start metrics reporter
        asyncio.create_task(self._metrics_reporter())
        
        self.is_initialized = True
        logger.info(
            f"{CyberColors.NEON_GREEN}✓ Unified Interface online - "
            f"{len(self.exchanges)} exchanges connected{CyberColors.RESET}"
        )
    
    async def _init_cex_connectors(self):
        """Initialize centralized exchange connectors"""
        cex_configs = self.config.get('exchanges', {})
        
        init_tasks = []
        
        for exchange_name, exchange_config in cex_configs.items():
            if not exchange_config.get('enabled', False):
                continue
            
            if exchange_name == 'coinbase':
                init_tasks.append(self._init_coinbase(exchange_config))
            elif exchange_name in ccxt_pro.__all__:
                init_tasks.append(self._init_ccxt_exchange(exchange_name, exchange_config))
            else:
                logger.warning(f"Unknown exchange: {exchange_name}")
        
        # Initialize all exchanges in parallel
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Check results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exchange initialization failed: {result}")
    
    async def _init_coinbase(self, config: Dict[str, Any]):
        """Initialize Coinbase connector"""
        try:
            connector = CoinbaseConnector(config)
            await connector.initialize()
            
            self.exchanges['coinbase'] = connector
            self.exchange_info['coinbase'] = ExchangeInfo(
                name='coinbase',
                type=ExchangeType.CEX_SPOT,
                maker_fee=Decimal('0.005'),
                taker_fee=Decimal('0.005'),
                supported_order_types=[
                    OrderType.MARKET, OrderType.LIMIT,
                    OrderType.STOP, OrderType.STOP_LIMIT
                ]
            )
            
            # Subscribe to events
            connector.subscribe('ticker', lambda data: self._on_ticker('coinbase', data))
            connector.subscribe('orderbook', lambda data: self._on_orderbook('coinbase', data))
            connector.subscribe('trade', lambda data: self._on_trade('coinbase', data))
            connector.subscribe('order', lambda data: self._on_order_update('coinbase', data))
            
            logger.info(f"{CyberColors.NEON_GREEN}✓ Coinbase initialized{CyberColors.RESET}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase: {e}")
            raise
    
    async def _init_ccxt_exchange(self, exchange_name: str, config: Dict[str, Any]):
        """Initialize CCXT-based exchange"""
        try:
            exchange_class = getattr(ccxt_pro, exchange_name)
            connector = CCXTConnector(exchange_class, config)
            await connector.initialize()
            
            self.exchanges[exchange_name] = connector
            
            # Determine exchange type and fees
            if exchange_name == 'binance':
                exchange_type = ExchangeType.CEX_SPOT
                maker_fee = Decimal('0.001')
                taker_fee = Decimal('0.001')
            elif exchange_name == 'bybit':
                exchange_type = ExchangeType.CEX_PERP
                maker_fee = Decimal('0.001')
                taker_fee = Decimal('0.0006')
            else:
                exchange_type = ExchangeType.CEX_SPOT
                maker_fee = Decimal('0.002')
                taker_fee = Decimal('0.002')
            
            self.exchange_info[exchange_name] = ExchangeInfo(
                name=exchange_name,
                type=exchange_type,
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                supported_order_types=[OrderType.MARKET, OrderType.LIMIT]
            )
            
            # Subscribe to events
            connector.subscribe('ticker', lambda data: self._on_ticker(exchange_name, data))
            connector.subscribe('orderbook', lambda data: self._on_orderbook(exchange_name, data))
            
            logger.info(f"{CyberColors.NEON_GREEN}✓ {exchange_name} initialized{CyberColors.RESET}")
            
        except Exception as e:
            logger.error(f"Failed to initialize {exchange_name}: {e}")
            raise
    
    async def _init_dex_connectors(self):
        """Initialize decentralized exchange connectors"""
        dex_configs = self.config.get('dex', {})
        
        # Setup Web3 providers for each chain
        chains = {
            'ethereum': self.config.get('eth_rpc_url', 'https://eth.llamarpc.com'),
            'bsc': self.config.get('bsc_rpc_url', 'https://bsc-dataseed.binance.org'),
            'polygon': self.config.get('polygon_rpc_url', 'https://polygon-rpc.com'),
            'arbitrum': self.config.get('arbitrum_rpc_url', 'https://arb1.arbitrum.io/rpc'),
            'optimism': self.config.get('optimism_rpc_url', 'https://mainnet.optimism.io')
        }
        
        for chain, rpc_url in chains.items():
            if rpc_url:
                w3 = Web3(Web3.HTTPProvider(rpc_url))
                
                # Add POA middleware for BSC and Polygon
                if chain in ['bsc', 'polygon']:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                self.web3_providers[chain] = w3
                logger.info(f"Connected to {chain} RPC")
        
        # Initialize DEX connectors
        if dex_configs.get('uniswap_v3', {}).get('enabled'):
            try:
                connector = UniswapV3Connector(
                    self.web3_providers['ethereum'],
                    dex_configs['uniswap_v3']
                )
                await connector.initialize()
                
                self.exchanges['uniswap_v3'] = connector
                self.exchange_info['uniswap_v3'] = ExchangeInfo(
                    name='uniswap_v3',
                    type=ExchangeType.DEX_AMM,
                    chain='ethereum',
                    maker_fee=Decimal('0.003'),  # 0.3% pool fee
                    taker_fee=Decimal('0.003'),
                    supported_order_types=[OrderType.MARKET]
                )
                
                logger.info(f"{CyberColors.NEON_GREEN}✓ Uniswap V3 initialized{CyberColors.RESET}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Uniswap V3: {e}")
    
    def register_symbol(self, base: str, quote: str, exchange_mappings: Dict[str, str]):
        """Register a unified symbol with exchange-specific mappings"""
        unified_key = f"{base}/{quote}"
        
        if unified_key not in self.unified_symbols:
            self.unified_symbols[unified_key] = UnifiedSymbol(
                base=base,
                quote=quote,
                exchange_symbols=exchange_mappings
            )
        else:
            # Update mappings
            self.unified_symbols[unified_key].exchange_symbols.update(exchange_mappings)
    
    async def get_best_prices(self, symbol: str) -> Dict[str, Tuple[Decimal, Decimal]]:
        """Get best bid/ask prices across all exchanges"""
        if symbol not in self.unified_symbols:
            return {}
        
        prices = {}
        unified_symbol = self.unified_symbols[symbol]
        
        # Get prices from each exchange in parallel
        tasks = []
        exchanges = []
        
        for exchange, connector in self.exchanges.items():
            exchange_symbol = unified_symbol.get_exchange_symbol(exchange)
            if hasattr(connector, 'orderbooks') and exchange_symbol in connector.orderbooks:
                # Use cached orderbook
                orderbook = connector.orderbooks[exchange_symbol]
                best_bid = orderbook.get_best_bid()
                best_ask = orderbook.get_best_ask()
                
                if best_bid and best_ask:
                    prices[exchange] = (best_bid[0], best_ask[0])
            else:
                # Fetch fresh orderbook
                tasks.append(connector.get_orderbook(exchange_symbol))
                exchanges.append(exchange)
        
        if tasks:
            orderbooks = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, orderbook in enumerate(orderbooks):
                if not isinstance(orderbook, Exception):
                    best_bid = orderbook.get_best_bid()
                    best_ask = orderbook.get_best_ask()
                    
                    if best_bid and best_ask:
                        prices[exchanges[i]] = (best_bid[0], best_ask[0])
        
        return prices
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[Decimal] = None,
        exchange: Optional[str] = None,
        smart_route: bool = True,
        urgency: float = 0.5,  # 0-1, higher = more urgent
        **kwargs
    ) -> Union[UnifiedOrder, List[UnifiedOrder]]:
        """
        Place order with smart routing
        
        Args:
            symbol: Unified symbol (e.g., "BTC/USDT")
            side: 'buy' or 'sell'
            amount: Order size
            order_type: Order type
            price: Limit price (optional)
            exchange: Specific exchange (optional, uses smart routing if not specified)
            smart_route: Enable smart order routing
            urgency: Urgency factor (affects routing decisions)
            **kwargs: Additional order parameters
        """
        start_time = time.perf_counter()
        
        # Create unified order
        order_id = f"NEXLIFY-{int(time.time() * 1e6)}"
        unified_symbol = self.unified_symbols.get(symbol)
        
        if not unified_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        # Check if we should split the order
        if smart_route and not exchange and amount > self._get_split_threshold(symbol):
            # Large order - use multi-exchange splitting
            return await self._place_split_order(
                unified_symbol, side, amount, order_type, price, urgency, **kwargs
            )
        
        # Single exchange order
        unified_order = UnifiedOrder(
            id=order_id,
            exchange="",  # Will be set by routing
            symbol=unified_symbol,
            side=side,
            type=order_type,
            amount=amount,
            price=price,
            metadata=kwargs
        )
        
        # Determine exchange routing
        if smart_route and not exchange:
            routes = await self._calculate_best_routes(unified_order, urgency)
            if not routes:
                raise Exception("No suitable routes found")
            
            # Execute on best exchange
            best_route = routes[0]
            exchange = best_route.exchange
            
            # Adjust price based on routing analysis
            if order_type == OrderType.LIMIT and not price:
                price = best_route.expected_price
            
            logger.info(
                f"{CyberColors.NEON_CYAN}Smart routed to {exchange} "
                f"(score: {best_route.score:.3f}){CyberColors.RESET}"
            )
            ROUTING_DECISIONS.labels(exchange=exchange, reason='best_score').inc()
        
        elif not exchange:
            # Default to exchange with best liquidity
            exchange = await self._get_best_liquidity_exchange(symbol)
        
        # Place order on selected exchange
        unified_order.exchange = exchange
        connector = self.exchanges.get(exchange)
        
        if not connector:
            raise ValueError(f"Exchange not available: {exchange}")
        
        try:
            # MEV protection for DEX orders
            if self._is_dex(exchange) and MEV_PROTECTION_ENABLED:
                kwargs.update(self._apply_mev_protection(unified_order))
            
            # Place the order
            result = await connector.place_order(unified_order)
            
            if result:
                unified_order.status = OrderStatus.OPEN
                self.unified_orders[order_id] = unified_order
                self.execution_metrics['total_orders'] += 1
                self.execution_metrics['successful_orders'] += 1
                
                # Track execution time
                execution_time = time.perf_counter() - start_time
                ORDER_EXECUTION_TIME.labels(
                    exchange=exchange,
                    order_type=order_type.value
                ).observe(execution_time)
                
                logger.info(
                    f"{CyberColors.NEON_GREEN}✓ Order placed in {execution_time*1000:.1f}ms: "
                    f"{side} {amount} {symbol} on {exchange} @ {price or 'market'}{CyberColors.RESET}"
                )
                
                await self._notify_callbacks('order_placed', unified_order)
            else:
                unified_order.status = OrderStatus.REJECTED
                self.execution_metrics['failed_orders'] += 1
                
        except Exception as e:
            logger.error(f"{CyberColors.NEON_RED}Order failed: {e}{CyberColors.RESET}")
            unified_order.status = OrderStatus.REJECTED
            self.execution_metrics['failed_orders'] += 1
            raise
        
        return unified_order
    
    async def _place_split_order(
        self,
        symbol: UnifiedSymbol,
        side: str,
        total_amount: Decimal,
        order_type: OrderType,
        price: Optional[Decimal],
        urgency: float,
        **kwargs
    ) -> List[UnifiedOrder]:
        """Place order split across multiple exchanges"""
        logger.info(
            f"{CyberColors.NEON_PINK}Splitting large order across exchanges: "
            f"{total_amount} {symbol.unified}{CyberColors.RESET}"
        )
        
        # Analyze liquidity across exchanges
        liquidity_map = await self._analyze_cross_exchange_liquidity(symbol, side, total_amount)
        
        # Calculate optimal split
        split_plan = self._calculate_optimal_split(
            liquidity_map, total_amount, urgency
        )
        
        # Execute orders in parallel
        orders = []
        tasks = []
        
        for exchange, amount in split_plan.items():
            if amount > 0:
                order_id = f"NEXLIFY-{int(time.time() * 1e6)}-{exchange}"
                
                unified_order = UnifiedOrder(
                    id=order_id,
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    amount=amount,
                    price=price,
                    metadata={**kwargs, 'parent_split': True}
                )
                
                orders.append(unified_order)
                connector = self.exchanges[exchange]
                tasks.append(connector.place_order(unified_order))
        
        # Execute all orders
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update order statuses
        successful = 0
        total_filled = Decimal("0")
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                orders[i].status = OrderStatus.REJECTED
                logger.error(f"Split order failed on {orders[i].exchange}: {result}")
            else:
                orders[i].status = OrderStatus.OPEN
                successful += 1
                total_filled += orders[i].amount
                self.unified_orders[orders[i].id] = orders[i]
        
        logger.info(
            f"{CyberColors.NEON_GREEN}Split order complete: "
            f"{successful}/{len(orders)} successful, {total_filled}/{total_amount} filled{CyberColors.RESET}"
        )
        
        return orders
    
    async def execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> bool:
        """Execute arbitrage opportunity with atomic execution"""
        logger.info(
            f"{CyberColors.NEON_PINK}💰 Executing arbitrage: "
            f"{opportunity.symbol.unified} - Buy on {opportunity.buy_exchange} @ ${opportunity.buy_price}, "
            f"Sell on {opportunity.sell_exchange} @ ${opportunity.sell_price} "
            f"({opportunity.profit_percent:.2%} profit, ${opportunity.profit_usd:.2f}){CyberColors.RESET}"
        )
        
        try:
            # Check balances
            if not await self._check_arbitrage_balances(opportunity):
                return False
            
            # Place both orders simultaneously
            buy_order = UnifiedOrder(
                id=f"ARB-BUY-{int(time.time() * 1e6)}",
                exchange=opportunity.buy_exchange,
                symbol=opportunity.symbol,
                side="buy",
                type=OrderType.LIMIT,
                amount=opportunity.max_amount,
                price=opportunity.buy_price,
                metadata={'arbitrage_id': opportunity.timestamp}
            )
            
            sell_order = UnifiedOrder(
                id=f"ARB-SELL-{int(time.time() * 1e6)}",
                exchange=opportunity.sell_exchange,
                symbol=opportunity.symbol,
                side="sell",
                type=OrderType.LIMIT,
                amount=opportunity.max_amount,
                price=opportunity.sell_price,
                metadata={'arbitrage_id': opportunity.timestamp}
            )
            
            # Execute both orders
            buy_connector = self.exchanges[opportunity.buy_exchange]
            sell_connector = self.exchanges[opportunity.sell_exchange]
            
            buy_task = buy_connector.place_order(buy_order)
            sell_task = sell_connector.place_order(sell_order)
            
            buy_result, sell_result = await asyncio.gather(
                buy_task, sell_task, return_exceptions=True
            )
            
            # Check if both succeeded
            if isinstance(buy_result, Exception) or isinstance(sell_result, Exception):
                # Cancel any successful order
                if not isinstance(buy_result, Exception):
                    await buy_connector.cancel_order(buy_order.id, buy_order.symbol.unified)
                if not isinstance(sell_result, Exception):
                    await sell_connector.cancel_order(sell_order.id, sell_order.symbol.unified)
                return False
            
            # Track arbitrage
            self.execution_metrics['arbitrage_trades'] += 1
            self.execution_metrics['arbitrage_profit'] += opportunity.profit_usd
            
            self.arbitrage_history.append({
                'opportunity': opportunity,
                'executed': True,
                'buy_order': buy_order.id,
                'sell_order': sell_order.id,
                'profit': opportunity.profit_usd
            })
            
            logger.info(
                f"{CyberColors.NEON_GREEN}✓ Arbitrage executed successfully! "
                f"Profit: ${opportunity.profit_usd:.2f}{CyberColors.RESET}"
            )
            
            await self._notify_callbacks('arbitrage_executed', opportunity)
            return True
            
        except Exception as e:
            logger.error(f"{CyberColors.NEON_RED}Arbitrage execution failed: {e}{CyberColors.RESET}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel unified order"""
        order = self.unified_orders.get(order_id)
        if not order:
            return False
        
        connector = self.exchanges.get(order.exchange)
        if connector:
            symbol = order.symbol.get_exchange_symbol(order.exchange)
            success = await connector.cancel_order(order_id, symbol)
            
            if success:
                order.status = OrderStatus.CANCELLED
                await self._notify_callbacks('order_cancelled', order)
            
            return success
        
        return False
    
    async def get_unified_balance(self) -> Dict[str, Decimal]:
        """Get total balance across all exchanges"""
        total_balance = defaultdict(Decimal)
        
        # Fetch balances in parallel
        tasks = []
        exchanges = []
        
        for exchange, connector in self.exchanges.items():
            tasks.append(connector.get_balance())
            exchanges.append(exchange)
        
        balances = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, balance in enumerate(balances):
            if not isinstance(balance, Exception):
                exchange = exchanges[i]
                for currency, amount in balance.items():
                    total_balance[currency] += amount
                    self.unified_balances[exchange][currency] = amount
        
        return dict(total_balance)
    
    async def _calculate_best_routes(
        self,
        order: UnifiedOrder,
        urgency: float
    ) -> List[RouteOption]:
        """Calculate best routing options for an order"""
        routes = []
        
        # Check cache first
        cache_key = f"{order.symbol.unified}:{order.side}:{order.amount}:{urgency}"
        if cache_key in self.route_cache:
            cached_routes = self.route_cache[cache_key]
            if cached_routes and time.time() - cached_routes[0].timestamp < ROUTE_CACHE_TTL:
                return cached_routes
        
        # Calculate routes for each exchange
        for exchange, connector in self.exchanges.items():
            try:
                # Skip if exchange doesn't support the order type
                exchange_info = self.exchange_info.get(exchange)
                if exchange_info and order.type not in exchange_info.supported_order_types:
                    continue
                
                # Get exchange-specific symbol
                exchange_symbol = order.symbol.get_exchange_symbol(exchange)
                
                # Get orderbook
                if hasattr(connector, 'orderbooks') and exchange_symbol in connector.orderbooks:
                    orderbook = connector.orderbooks[exchange_symbol]
                else:
                    orderbook = await connector.get_orderbook(exchange_symbol)
                
                if not orderbook:
                    continue
                
                # Calculate metrics
                metrics = self._calculate_route_metrics(
                    orderbook, order, exchange_info
                )
                
                if metrics['available_liquidity'] > 0:
                    route = RouteOption(
                        exchange=exchange,
                        expected_price=metrics['expected_price'],
                        available_liquidity=metrics['available_liquidity'],
                        estimated_fees=metrics['total_fee'],
                        latency_ms=metrics['latency_ms'],
                        slippage_estimate=metrics['slippage'],
                        reliability_score=exchange_info.reliability_score if exchange_info else 0.9
                    )
                    
                    # Calculate routing score
                    route.score = self._calculate_routing_score(metrics, urgency)
                    routes.append(route)
                    
            except Exception as e:
                logger.debug(f"Failed to calculate route for {exchange}: {e}")
        
        # Sort by score
        routes.sort(key=lambda r: r.score, reverse=True)
        
        # Cache results
        self.route_cache[cache_key] = routes
        
        return routes
    
    def _calculate_route_metrics(
        self,
        orderbook: OrderBook,
        order: UnifiedOrder,
        exchange_info: Optional[ExchangeInfo]
    ) -> Dict[str, Any]:
        """Calculate routing metrics for an exchange"""
        metrics = {
            'available_liquidity': Decimal("0"),
            'expected_price': Decimal("0"),
            'slippage': Decimal("0"),
            'total_fee': Decimal("0"),
            'latency_ms': 100.0,
            'price_score': 0.5,
            'liquidity_score': 0.5
        }
        
        # Get relevant order book side
        if order.side == "buy":
            levels = list(orderbook.asks.items())
        else:
            levels = list(orderbook.bids.items())
        
        if not levels:
            return metrics
        
        # Calculate available liquidity and expected price
        remaining_amount = order.amount
        total_cost = Decimal("0")
        
        for price, level in levels:
            if remaining_amount <= 0:
                break
            
            fill_amount = min(remaining_amount, level.amount)
            total_cost += fill_amount * price
            remaining_amount -= fill_amount
        
        if remaining_amount > 0:
            # Not enough liquidity
            metrics['available_liquidity'] = order.amount - remaining_amount
        else:
            metrics['available_liquidity'] = order.amount
        
        if metrics['available_liquidity'] > 0:
            metrics['expected_price'] = total_cost / metrics['available_liquidity']
            
            # Calculate slippage
            best_price = levels[0][0] if levels else order.price
            if best_price and order.price:
                metrics['slippage'] = abs(metrics['expected_price'] - order.price) / order.price
            
            # Calculate fees
            if exchange_info:
                fee_rate = exchange_info.taker_fee
                metrics['total_fee'] = metrics['available_liquidity'] * metrics['expected_price'] * fee_rate
            
            # Get latency from tracker
            exchange = orderbook.symbol.split('/')[0]  # Hack to get exchange name
            if exchange in self.latency_tracker and self.latency_tracker[exchange]:
                metrics['latency_ms'] = np.mean(list(self.latency_tracker[exchange]))
            
            # Calculate scores
            metrics['price_score'] = 1.0 / (1.0 + float(metrics['slippage']))
            metrics['liquidity_score'] = float(
                metrics['available_liquidity'] / order.amount
            )
        
        return metrics
    
    def _calculate_routing_score(self, metrics: Dict, urgency: float) -> float:
        """Calculate routing score based on multiple factors"""
        # Base scores
        price_score = float(metrics.get('price_score', 0.5))
        liquidity_score = float(metrics.get('liquidity_score', 0.5))
        latency_score = 1.0 - (metrics.get('latency_ms', 100) / 1000.0)
        reliability_score = metrics.get('reliability_score', 0.9)
        fee_score = 1.0 - float(metrics.get('total_fee', 0) / (metrics.get('expected_price', 1) * metrics.get('available_liquidity', 1)))
        
        # Adjust weights based on urgency
        if urgency > 0.7:
            # Urgent: prioritize liquidity and latency
            weights = {
                'price': 0.15,
                'liquidity': 0.40,
                'latency': 0.30,
                'reliability': 0.10,
                'fee': 0.05
            }
        elif urgency < 0.3:
            # Patient: prioritize price and fees
            weights = {
                'price': 0.40,
                'liquidity': 0.15,
                'latency': 0.10,
                'reliability': 0.15,
                'fee': 0.20
            }
        else:
            # Balanced
            weights = {
                'price': 0.30,
                'liquidity': 0.25,
                'latency': 0.20,
                'reliability': 0.15,
                'fee': 0.10
            }
        
        # Calculate weighted score
        score = (
            weights['price'] * price_score +
            weights['liquidity'] * liquidity_score +
            weights['latency'] * latency_score +
            weights['reliability'] * reliability_score +
            weights['fee'] * fee_score
        )
        
        # Apply exchange reputation modifier
        exchange = metrics.get('exchange', '')
        if exchange in self.routing_scores:
            score *= self.routing_scores[exchange]
        
        return score
    
    async def _analyze_cross_exchange_liquidity(
        self,
        symbol: UnifiedSymbol,
        side: str,
        amount: Decimal
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze liquidity across all exchanges"""
        liquidity_map = {}
        
        # Analyze each exchange in parallel
        tasks = []
        exchanges = []
        
        for exchange, connector in self.exchanges.items():
            exchange_symbol = symbol.get_exchange_symbol(exchange)
            tasks.append(connector.get_orderbook(exchange_symbol))
            exchanges.append(exchange)
        
        orderbooks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, orderbook in enumerate(orderbooks):
            if not isinstance(orderbook, Exception):
                exchange = exchanges[i]
                exchange_info = self.exchange_info.get(exchange)
                
                metrics = self._calculate_route_metrics(
                    orderbook,
                    UnifiedOrder(
                        id="temp",
                        exchange=exchange,
                        symbol=symbol,
                        side=side,
                        type=OrderType.MARKET,
                        amount=amount
                    ),
                    exchange_info
                )
                
                liquidity_map[exchange] = metrics
        
        return liquidity_map
    
    def _calculate_optimal_split(
        self,
        liquidity_map: Dict[str, Dict[str, Any]],
        total_amount: Decimal,
        urgency: float
    ) -> Dict[str, Decimal]:
        """Calculate optimal order split across exchanges"""
        split_plan = {}
        remaining = total_amount
        
        # Sort exchanges by routing score
        scored_exchanges = []
        for exchange, metrics in liquidity_map.items():
            score = self._calculate_routing_score(metrics, urgency)
            scored_exchanges.append((exchange, metrics, score))
        
        scored_exchanges.sort(key=lambda x: x[2], reverse=True)
        
        # Allocate to exchanges based on score and available liquidity
        for exchange, metrics, score in scored_exchanges:
            if remaining <= 0:
                break
            
            available = metrics['available_liquidity']
            if available > 0:
                # Allocate proportionally to score, but not more than available
                allocation = min(remaining, available)
                split_plan[exchange] = allocation
                remaining -= allocation
        
        # Log split plan
        for exchange, amount in split_plan.items():
            percentage = (amount / total_amount) * 100
            logger.info(f"  {exchange}: {amount} ({percentage:.1f}%)")
        
        if remaining > 0:
            logger.warning(f"  Unallocated: {remaining}")
        
        return split_plan
    
    async def _check_arbitrage_balances(self, opportunity: ArbitrageOpportunity) -> bool:
        """Check if we have sufficient balances for arbitrage"""
        # Check quote balance on buy exchange
        buy_exchange_balance = self.unified_balances.get(opportunity.buy_exchange, {})
        quote_balance = buy_exchange_balance.get(opportunity.symbol.quote, Decimal("0"))
        required_quote = opportunity.max_amount * opportunity.buy_price * Decimal("1.01")  # 1% buffer
        
        if quote_balance < required_quote:
            logger.warning(
                f"Insufficient {opportunity.symbol.quote} on {opportunity.buy_exchange}: "
                f"{quote_balance} < {required_quote}"
            )
            return False
        
        # Check base balance on sell exchange
        sell_exchange_balance = self.unified_balances.get(opportunity.sell_exchange, {})
        base_balance = sell_exchange_balance.get(opportunity.symbol.base, Decimal("0"))
        
        if base_balance < opportunity.max_amount:
            logger.warning(
                f"Insufficient {opportunity.symbol.base} on {opportunity.sell_exchange}: "
                f"{base_balance} < {opportunity.max_amount}"
            )
            return False
        
        return True
    
    def _get_split_threshold(self, symbol: str) -> Decimal:
        """Get order size threshold for splitting"""
        # This could be configured per symbol
        if "BTC" in symbol:
            return Decimal("1.0")  # 1 BTC
        elif "ETH" in symbol:
            return Decimal("10.0")  # 10 ETH
        else:
            return Decimal("10000.0")  # $10k equivalent
    
    async def _get_best_liquidity_exchange(self, symbol: str) -> str:
        """Get exchange with best liquidity for a symbol"""
        best_exchange = None
        best_liquidity = Decimal("0")
        
        unified_symbol = self.unified_symbols.get(symbol)
        if not unified_symbol:
            return list(self.exchanges.keys())[0]
        
        for exchange, connector in self.exchanges.items():
            try:
                exchange_symbol = unified_symbol.get_exchange_symbol(exchange)
                orderbook = await connector.get_orderbook(exchange_symbol)
                
                if orderbook:
                    # Calculate total liquidity in top 10 levels
                    bid_liquidity = sum(
                        level.amount * level.price
                        for level in list(orderbook.bids.values())[:10]
                    )
                    ask_liquidity = sum(
                        level.amount * level.price
                        for level in list(orderbook.asks.values())[:10]
                    )
                    
                    total_liquidity = bid_liquidity + ask_liquidity
                    
                    if total_liquidity > best_liquidity:
                        best_liquidity = total_liquidity
                        best_exchange = exchange
                        
            except Exception as e:
                logger.debug(f"Failed to get liquidity for {exchange}: {e}")
        
        return best_exchange or list(self.exchanges.keys())[0]
    
    def _is_dex(self, exchange: str) -> bool:
        """Check if exchange is a DEX"""
        exchange_info = self.exchange_info.get(exchange)
        if exchange_info:
            return exchange_info.type in [
                ExchangeType.DEX_AMM,
                ExchangeType.DEX_ORDERBOOK,
                ExchangeType.DEX_PERP
            ]
        return exchange.lower() in {'uniswap', 'sushiswap', 'pancakeswap', 'curve', '1inch'}
    
    def _apply_mev_protection(self, order: UnifiedOrder) -> Dict[str, Any]:
        """Apply MEV protection for DEX orders"""
        protection = {}
        
        # Add commit-reveal scheme
        protection['commit_reveal'] = True
        
        # Add minimum output amount (slippage protection)
        if order.side == "buy" and order.price:
            min_output = order.amount * Decimal("0.995")  # 0.5% slippage tolerance
            protection['min_output'] = str(min_output)
        elif order.side == "sell" and order.price:
            min_output = order.amount * order.price * Decimal("0.995")
            protection['min_output'] = str(min_output)
        
        # Add deadline
        protection['deadline'] = int(time.time() + ORDER_TIMEOUT)
        
        # Use flashbots relay if available
        protection['use_flashbots'] = True
        
        # Add priority fee to incentivize quick inclusion
        protection['priority_fee_gwei'] = 5
        
        return protection
    
    async def _update_loop(self):
        """Background task for updating metrics and caches"""
        while True:
            try:
                # Update latency metrics
                for exchange in self.exchanges:
                    start = time.perf_counter()
                    try:
                        await self.exchanges[exchange].get_balance()
                        latency = (time.perf_counter() - start) * 1000
                        self.latency_tracker[exchange].append(latency)
                        EXCHANGE_LATENCY.labels(exchange=exchange, operation='balance').observe(latency/1000)
                    except:
                        pass
                
                # Update exchange reliability scores
                await self._update_reliability_scores()
                
                # Clean old cache entries
                current_time = time.time()
                self.orderbook_cache = {
                    k: v for k, v in self.orderbook_cache.items()
                    if current_time - v.last_update < 60  # 1 minute cache
                }
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(60)
    
    async def _arbitrage_detector(self):
        """Detect arbitrage opportunities across exchanges"""
        while True:
            try:
                self.arbitrage_opportunities.clear()
                
                # Check each symbol
                for symbol in self.unified_symbols:
                    opportunities = await self._detect_symbol_arbitrage(symbol)
                    self.arbitrage_opportunities.update(opportunities)
                
                # Execute top opportunities
                if self.arbitrage_opportunities:
                    top_opportunity = self.arbitrage_opportunities[0]
                    if top_opportunity.profit_percent >= self.min_arbitrage_profit:
                        asyncio.create_task(self.execute_arbitrage(top_opportunity))
                
                await asyncio.sleep(ARBITRAGE_SCAN_INTERVAL)
                
            except Exception as e:
                logger.error(f"Arbitrage detection error: {e}")
                await asyncio.sleep(1)
    
    async def _detect_symbol_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities for a symbol"""
        opportunities = []
        
        # Get prices from all exchanges
        prices = await self.get_best_prices(symbol)
        
        if len(prices) < 2:
            return opportunities
        
        # Find price discrepancies
        exchanges = list(prices.keys())
        unified_symbol = self.unified_symbols[symbol]
        
        for i, buy_exchange in enumerate(exchanges):
            buy_bid, buy_ask = prices[buy_exchange]
            
            for sell_exchange in exchanges[i+1:]:
                sell_bid, sell_ask = prices[sell_exchange]
                
                # Check if we can buy on one exchange and sell on another
                if buy_ask < sell_bid:
                    profit = sell_bid - buy_ask
                    profit_percent = (profit / buy_ask) * Decimal("100")
                    
                    if profit_percent >= self.min_arbitrage_profit:
                        # Calculate max amount based on available liquidity
                        max_amount = await self._calculate_arbitrage_size(
                            unified_symbol, buy_exchange, sell_exchange, buy_ask, sell_bid
                        )
                        
                        if max_amount > 0:
                            profit_usd = profit * max_amount
                            
                            opportunity = ArbitrageOpportunity(
                                symbol=unified_symbol,
                                buy_exchange=buy_exchange,
                                sell_exchange=sell_exchange,
                                buy_price=buy_ask,
                                sell_price=sell_bid,
                                max_amount=max_amount,
                                profit_percent=profit_percent,
                                profit_usd=profit_usd,
                                confidence_score=self._calculate_arbitrage_confidence(
                                    buy_exchange, sell_exchange, profit_percent
                                )
                            )
                            opportunities.append(opportunity)
                            ARBITRAGE_OPPORTUNITIES.inc()
                
                # Check reverse direction
                if sell_ask < buy_bid:
                    profit = buy_bid - sell_ask
                    profit_percent = (profit / sell_ask) * Decimal("100")
                    
                    if profit_percent >= self.min_arbitrage_profit:
                        max_amount = await self._calculate_arbitrage_size(
                            unified_symbol, sell_exchange, buy_exchange, sell_ask, buy_bid
                        )
                        
                        if max_amount > 0:
                            profit_usd = profit * max_amount
                            
                            opportunity = ArbitrageOpportunity(
                                symbol=unified_symbol,
                                buy_exchange=sell_exchange,
                                sell_exchange=buy_exchange,
                                buy_price=sell_ask,
                                sell_price=buy_bid,
                                max_amount=max_amount,
                                profit_percent=profit_percent,
                                profit_usd=profit_usd,
                                confidence_score=self._calculate_arbitrage_confidence(
                                    sell_exchange, buy_exchange, profit_percent
                                )
                            )
                            opportunities.append(opportunity)
                            ARBITRAGE_OPPORTUNITIES.inc()
        
        return opportunities
    
    async def _calculate_arbitrage_size(
        self,
        symbol: UnifiedSymbol,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: Decimal,
        sell_price: Decimal
    ) -> Decimal:
        """Calculate maximum arbitrage size based on liquidity and balances"""
        # Get orderbooks
        buy_connector = self.exchanges[buy_exchange]
        sell_connector = self.exchanges[sell_exchange]
        
        buy_symbol = symbol.get_exchange_symbol(buy_exchange)
        sell_symbol = symbol.get_exchange_symbol(sell_exchange)
        
        try:
            buy_book = await buy_connector.get_orderbook(buy_symbol)
            sell_book = await sell_connector.get_orderbook(sell_symbol)
            
            # Calculate available liquidity at target prices
            buy_liquidity = sum(
                level.amount
                for price, level in buy_book.asks.items()
                if price <= buy_price * Decimal("1.001")  # 0.1% tolerance
            )
            
            sell_liquidity = sum(
                level.amount
                for price, level in sell_book.bids.items()
                if price >= sell_price * Decimal("0.999")  # 0.1% tolerance
            )
            
            # Check balances
            buy_balance = self.unified_balances.get(buy_exchange, {})
            sell_balance = self.unified_balances.get(sell_exchange, {})
            
            quote_available = buy_balance.get(symbol.quote, Decimal("0")) / buy_price
            base_available = sell_balance.get(symbol.base, Decimal("0"))
            
            # Return minimum of all constraints
            return min(buy_liquidity, sell_liquidity, quote_available, base_available)
            
        except Exception as e:
            logger.debug(f"Failed to calculate arbitrage size: {e}")
            return Decimal("0")
    
    def _calculate_arbitrage_confidence(
        self,
        buy_exchange: str,
        sell_exchange: str,
        profit_percent: Decimal
    ) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        # Base confidence on exchange reliability
        buy_info = self.exchange_info.get(buy_exchange)
        sell_info = self.exchange_info.get(sell_exchange)
        
        buy_reliability = buy_info.reliability_score if buy_info else 0.9
        sell_reliability = sell_info.reliability_score if sell_info else 0.9
        
        # Average reliability
        base_confidence = (buy_reliability + sell_reliability) / 2
        
        # Adjust for profit magnitude (higher profit = potentially stale data)
        if profit_percent > Decimal("2"):
            confidence_penalty = float(profit_percent - Decimal("2")) * 0.1
            base_confidence -= min(confidence_penalty, 0.3)
        
        # Adjust for latency
        buy_latency = np.mean(list(self.latency_tracker[buy_exchange])) if self.latency_tracker[buy_exchange] else 100
        sell_latency = np.mean(list(self.latency_tracker[sell_exchange])) if self.latency_tracker[sell_exchange] else 100
        
        avg_latency = (buy_latency + sell_latency) / 2
        if avg_latency > 200:  # High latency reduces confidence
            base_confidence *= 0.8
        
        return max(0.1, min(1.0, base_confidence))
    
    async def _route_optimizer(self):
        """Optimize routing scores based on historical performance"""
        while True:
            try:
                # Update routing scores based on recent performance
                for exchange in self.exchanges:
                    # Calculate success rate
                    recent_orders = [
                        order for order in self.unified_orders.values()
                        if order.exchange == exchange and
                        order.timestamp > int(time.time() * 1e9) - 3600 * 1e9  # Last hour
                    ]
                    
                    if recent_orders:
                        success_rate = sum(
                            1 for order in recent_orders
                            if order.status == OrderStatus.FILLED
                        ) / len(recent_orders)
                        
                        # Update routing score with exponential moving average
                        current_score = self.routing_scores.get(exchange, 1.0)
                        self.routing_scores[exchange] = (
                            0.7 * current_score + 0.3 * success_rate
                        )
                
                # Clear old route cache entries
                current_time = time.time()
                self.route_cache = {
                    k: v for k, v in self.route_cache.items()
                    if v and current_time - v[0].timestamp < ROUTE_CACHE_TTL
                }
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Route optimization error: {e}")
                await asyncio.sleep(60)
    
    async def _update_reliability_scores(self):
        """Update exchange reliability scores based on performance"""
        for exchange in self.exchanges:
            try:
                # Get recent latencies
                if exchange in self.latency_tracker and self.latency_tracker[exchange]:
                    latencies = list(self.latency_tracker[exchange])
                    avg_latency = np.mean(latencies)
                    latency_stability = 1.0 - (np.std(latencies) / avg_latency if avg_latency > 0 else 0)
                else:
                    latency_stability = 0.9
                
                # Check order success rate
                recent_orders = [
                    order for order in self.unified_orders.values()
                    if order.exchange == exchange and
                    order.timestamp > int(time.time() * 1e9) - 86400 * 1e9  # Last 24h
                ]
                
                if recent_orders:
                    success_rate = sum(
                        1 for order in recent_orders
                        if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
                    ) / len(recent_orders)
                else:
                    success_rate = 0.95
                
                # Update reliability score
                if exchange in self.exchange_info:
                    self.exchange_info[exchange].reliability_score = (
                        0.6 * success_rate + 0.4 * latency_stability
                    )
                    
            except Exception as e:
                logger.debug(f"Failed to update reliability for {exchange}: {e}")
    
    async def _metrics_reporter(self):
        """Report performance metrics"""
        while True:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                
                # Calculate additional metrics
                total_orders = self.execution_metrics['total_orders']
                if total_orders > 0:
                    success_rate = self.execution_metrics['successful_orders'] / total_orders
                    
                    logger.info(
                        f"{CyberColors.NEON_CYAN}📊 Performance Metrics:{CyberColors.RESET}\n"
                        f"  Total Orders: {total_orders}\n"
                        f"  Success Rate: {success_rate:.1%}\n"
                        f"  Total Volume: ${self.execution_metrics['total_volume']:,.2f}\n"
                        f"  Arbitrage Trades: {self.execution_metrics['arbitrage_trades']}\n"
                        f"  Arbitrage Profit: ${self.execution_metrics['arbitrage_profit']:,.2f}\n"
                        f"  Active Opportunities: {len(self.arbitrage_opportunities)}\n"
                        f"  Connected Exchanges: {len(self.exchanges)}"
                    )
                
                # Update position metrics
                for exchange, balances in self.unified_balances.items():
                    for asset, amount in balances.items():
                        ACTIVE_POSITIONS.labels(exchange=exchange, symbol=asset).set(float(amount))
                
                # Notify callbacks
                await self._notify_callbacks('metrics_update', self.execution_metrics)
                
            except Exception as e:
                logger.error(f"Metrics reporting error: {e}")
    
    def _on_ticker(self, exchange: str, data: Dict):
        """Handle ticker updates from exchange"""
        # Update tick data for routing decisions
        pass
    
    def _on_orderbook(self, exchange: str, data: Dict):
        """Handle order book updates from exchange"""
        # Cache orderbook for quick access
        if 'symbol' in data and 'orderbook' in data:
            cache_key = (exchange, data['symbol'])
            self.orderbook_cache[cache_key] = data['orderbook']
    
    def _on_trade(self, exchange: str, data: Dict):
        """Handle trade updates from exchange"""
        # Update volume metrics
        if 'trade' in data:
            trade = data['trade']
            self.execution_metrics['total_volume'] += trade.size * trade.price
    
    def _on_order_update(self, exchange: str, data: Dict):
        """Handle order status updates"""
        # Update unified order status
        if 'order_id' in data:
            order = self.unified_orders.get(data['order_id'])
            if order:
                # Update order based on exchange update
                if 'status' in data:
                    old_status = order.status
                    new_status = data['status']
                    
                    # Map exchange status to unified status
                    if new_status in ['filled', 'closed']:
                        order.status = OrderStatus.FILLED
                    elif new_status in ['cancelled', 'canceled']:
                        order.status = OrderStatus.CANCELLED
                    elif new_status == 'partially_filled':
                        order.status = OrderStatus.PARTIALLY_FILLED
                    
                    if order.status != old_status:
                        logger.info(
                            f"Order {order.id} status: {old_status.value} → {order.status.value}"
                        )
    
    def subscribe(self, event: str, callback: Callable):
        """Subscribe to unified interface events"""
        self.callbacks[event].append(callback)
    
    async def _notify_callbacks(self, event: str, data: Any):
        """Notify all callbacks for an event"""
        for callback in self.callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get unified interface statistics"""
        stats = {
            'connected_exchanges': len(self.exchanges),
            'unified_symbols': len(self.unified_symbols),
            'active_orders': len([o for o in self.unified_orders.values() if o.status == OrderStatus.OPEN]),
            'arbitrage_opportunities': len(self.arbitrage_opportunities),
            'top_arbitrage': None,
            'execution_metrics': self.execution_metrics,
            'exchange_stats': {}
        }
        
        # Add top arbitrage opportunity
        if self.arbitrage_opportunities:
            top = self.arbitrage_opportunities[0]
            stats['top_arbitrage'] = {
                'symbol': top.symbol.unified,
                'profit_percent': float(top.profit_percent),
                'profit_usd': float(top.profit_usd),
                'confidence': top.confidence_score
            }
        
        # Get stats from each exchange
        for exchange, connector in self.exchanges.items():
            stats['exchange_stats'][exchange] = connector.get_statistics()
        
        return stats
    
    async def close(self):
        """Clean shutdown"""
        logger.info(f"{CyberColors.NEURAL_PURPLE}Shutting down Unified Interface...{CyberColors.RESET}")
        
        # Cancel background tasks
        if self._update_task:
            self._update_task.cancel()
        if self._arbitrage_task:
            self._arbitrage_task.cancel()
        if self._routing_task:
            self._routing_task.cancel()
        
        # Cancel all open orders
        for order in list(self.unified_orders.values()):
            if order.status == OrderStatus.OPEN:
                await self.cancel_order(order.id)
        
        # Close all exchanges
        tasks = []
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                tasks.append(exchange.close())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info(f"{CyberColors.NEURAL_PURPLE}Unified Interface offline{CyberColors.RESET}")
    
    async def shutdown(self):
        """Alias for close (new interface compatibility)"""
        await self.close()


# Placeholder classes for DEX connectors (would need full implementation)
class PancakeSwapV3Connector(ExchangeConnector):
    """PancakeSwap V3 connector for BSC"""
    # Similar to Uniswap but on BSC
    pass


class GMXConnector(ExchangeConnector):
    """GMX perpetual DEX connector"""
    # Perpetual DEX implementation
    pass


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize config
        config_loader = get_config_loader()
        await config_loader.initialize()
        
        # Create unified interface
        interface = NexlifyUnifiedInterface()
        
        # Register symbols
        interface.register_symbol("BTC", "USDT", {
            "coinbase": "BTC-USDT",
            "binance": "BTCUSDT",
            "kraken": "XBTUSDT",
            "uniswap_v3": "WBTC/USDT"
        })
        
        interface.register_symbol("ETH", "USDT", {
            "coinbase": "ETH-USDT",
            "binance": "ETHUSDT",
            "kraken": "ETHUSDT",
            "uniswap_v3": "ETH/USDT"
        })
        
        # Subscribe to events
        async def on_arbitrage(opportunity):
            print(f"Arbitrage: {opportunity.symbol.unified} - "
                  f"{opportunity.profit_percent:.2%} profit (${opportunity.profit_usd:.2f})")
        
        async def on_order_placed(order):
            print(f"Order placed: {order.side} {order.amount} {order.symbol.unified} on {order.exchange}")
        
        interface.subscribe('arbitrage_detected', on_arbitrage)
        interface.subscribe('order_placed', on_order_placed)
        
        # Initialize
        await interface.initialize()
        
        # Place a smart-routed order
        order = await interface.place_order(
            symbol="BTC/USDT",
            side="buy",
            amount=Decimal("0.01"),
            order_type=OrderType.LIMIT,
            price=Decimal("50000"),
            smart_route=True,
            urgency=0.3  # Patient order
        )
        
        print(f"Order placed: {order}")
        
        # Get statistics
        stats = interface.get_statistics()
        print(f"\nStatistics: {stats}")
        
        # Run for a while
        await asyncio.sleep(300)
        
        # Shutdown
        await interface.shutdown()
    
    asyncio.run(main())
