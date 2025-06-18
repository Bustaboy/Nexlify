#!/usr/bin/env python3
"""
src/exchanges/unified_interface.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEXLIFY UNIFIED EXCHANGE INTERFACE v3.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cyberpunk-themed unified interface for all exchanges (CEX + DEX).
Supports parallel execution, smart order routing, and cross-exchange arbitrage.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import time
from decimal import Decimal
import logging
from collections import defaultdict, deque
import numpy as np
from sortedcontainers import SortedList

# Exchange connectors
import ccxt.pro as ccxt_pro
from web3 import Web3
from web3.middleware import geth_poa_middleware
import aiohttp
from eth_account import Account
from eth_account.signers.local import LocalAccount
import uvloop

# Performance libs
import orjson
import msgpack
from aiocache import cached
from prometheus_client import Counter, Histogram, Gauge
import pyO3  # Rust integration for performance-critical paths

# Import our connectors
from .coinbase_connector import CoinbaseConnector, OrderBook
from ..utils.config_loader import get_config_loader

# Set uvloop for maximum performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger("NEXLIFY.UNIFIED")

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

class ExchangeType(Enum):
    """Exchange categories"""
    CEX_SPOT = auto()      # Centralized spot
    CEX_PERP = auto()      # Centralized perpetuals
    DEX_AMM = auto()       # AMM DEXs (Uniswap, Pancake)
    DEX_ORDERBOOK = auto() # Orderbook DEXs (dYdX, Serum)
    DEX_PERP = auto()      # Perpetual DEXs (GMX, Gains)

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

@dataclass
class UnifiedOrder:
    """Unified order format across all exchanges"""
    exchange: str
    symbol: str
    side: str  # buy/sell
    type: str  # market/limit
    amount: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    post_only: bool = False
    reduce_only: bool = False
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArbitrageOpportunity:
    """Cross-exchange arbitrage opportunity"""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: Decimal
    sell_price: Decimal
    max_amount: Decimal
    profit_percent: Decimal
    timestamp: float = field(default_factory=time.time)

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

class UnifiedExchangeInterface:
    """
    Master controller for all exchanges with smart routing and arbitrage detection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config_loader().get_all()
        self.exchanges: Dict[str, ExchangeConnector] = {}
        self.exchange_info: Dict[str, ExchangeInfo] = {}
        
        # Performance tracking
        self.latency_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.orderbook_cache: Dict[Tuple[str, str], OrderBook] = {}  # (exchange, symbol)
        
        # Arbitrage detection
        self.arbitrage_opportunities = SortedList(key=lambda x: -x.profit_percent)
        self.min_arbitrage_profit = Decimal(self.config.get('arbitrage.min_profit_percent', '0.1'))
        
        # Smart order routing
        self.routing_weights: Dict[str, float] = {}
        self.liquidity_scores: Dict[Tuple[str, str], float] = {}  # (exchange, symbol)
        
        # Cross-chain bridges
        self.bridges: Dict[str, Any] = {}
        self.web3_providers: Dict[str, Web3] = {}
        
        # State
        self.is_initialized = False
        self._update_task = None
        
    async def initialize(self):
        """Initialize all configured exchanges"""
        logger.info("Initializing unified exchange interface...")
        
        # Initialize CEX connectors
        await self._init_cex_connectors()
        
        # Initialize DEX connectors
        await self._init_dex_connectors()
        
        # Start background tasks
        self._update_task = asyncio.create_task(self._update_loop())
        
        self.is_initialized = True
        logger.info(f"Initialized {len(self.exchanges)} exchanges")
    
    async def _init_cex_connectors(self):
        """Initialize centralized exchange connectors"""
        cex_configs = self.config.get('exchanges', {})
        
        for exchange_name, exchange_config in cex_configs.items():
            if not exchange_config.get('enabled', False):
                continue
            
            try:
                if exchange_name == 'coinbase':
                    connector = CoinbaseConnector(exchange_config)
                    self.exchange_info['coinbase'] = ExchangeInfo(
                        name='coinbase',
                        type=ExchangeType.CEX_SPOT,
                        maker_fee=Decimal('0.005'),
                        taker_fee=Decimal('0.005')
                    )
                
                elif exchange_name == 'binance':
                    # Use ccxt pro for all other exchanges
                    exchange_class = getattr(ccxt_pro, exchange_name)
                    connector = CCXTConnector(exchange_class, exchange_config)
                    self.exchange_info[exchange_name] = ExchangeInfo(
                        name=exchange_name,
                        type=ExchangeType.CEX_SPOT,
                        maker_fee=Decimal('0.001'),
                        taker_fee=Decimal('0.001')
                    )
                
                elif exchange_name in ['bybit', 'okx', 'gateio', 'kucoin']:
                    exchange_class = getattr(ccxt_pro, exchange_name)
                    connector = CCXTConnector(exchange_class, exchange_config)
                    self.exchange_info[exchange_name] = ExchangeInfo(
                        name=exchange_name,
                        type=ExchangeType.CEX_PERP if exchange_name == 'bybit' else ExchangeType.CEX_SPOT
                    )
                
                else:
                    logger.warning(f"Unknown exchange: {exchange_name}")
                    continue
                
                await connector.initialize()
                self.exchanges[exchange_name] = connector
                logger.info(f"Initialized {exchange_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {exchange_name}: {e}")
    
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
        
        # Initialize DEX connectors
        if dex_configs.get('uniswap_v3', {}).get('enabled'):
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
                maker_fee=Decimal('0.003'),
                taker_fee=Decimal('0.003')
            )
        
        if dex_configs.get('pancakeswap_v3', {}).get('enabled'):
            connector = PancakeSwapV3Connector(
                self.web3_providers['bsc'],
                dex_configs['pancakeswap_v3']
            )
            await connector.initialize()
            self.exchanges['pancakeswap_v3'] = connector
            self.exchange_info['pancakeswap_v3'] = ExchangeInfo(
                name='pancakeswap_v3',
                type=ExchangeType.DEX_AMM,
                chain='bsc',
                maker_fee=Decimal('0.0025'),
                taker_fee=Decimal('0.0025')
            )
        
        if dex_configs.get('gmx', {}).get('enabled'):
            connector = GMXConnector(
                self.web3_providers['arbitrum'],
                dex_configs['gmx']
            )
            await connector.initialize()
            self.exchanges['gmx'] = connector
            self.exchange_info['gmx'] = ExchangeInfo(
                name='gmx',
                type=ExchangeType.DEX_PERP,
                chain='arbitrum',
                maker_fee=Decimal('0.001'),
                taker_fee=Decimal('0.001')
            )
    
    async def get_best_price(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        exchanges: Optional[List[str]] = None
    ) -> Tuple[str, Decimal, Decimal]:
        """
        Get best price across exchanges with liquidity consideration
        Returns: (exchange, price, available_amount)
        """
        exchanges = exchanges or list(self.exchanges.keys())
        best_exchange = None
        best_price = Decimal('Infinity') if side == 'buy' else Decimal('0')
        best_amount = Decimal('0')
        
        # Parallel orderbook fetching
        tasks = []
        for exchange in exchanges:
            if exchange in self.exchanges:
                tasks.append(self._get_exchange_quote(exchange, symbol, side, amount))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to get quote from {exchanges[i]}: {result}")
                continue
            
            exchange, price, available = result
            
            # Compare prices
            if side == 'buy' and price < best_price:
                best_exchange, best_price, best_amount = exchange, price, available
            elif side == 'sell' and price > best_price:
                best_exchange, best_price, best_amount = exchange, price, available
        
        return best_exchange, best_price, best_amount
    
    async def _get_exchange_quote(
        self,
        exchange: str,
        symbol: str,
        side: str,
        amount: Decimal
    ) -> Tuple[str, Decimal, Decimal]:
        """Get quote from specific exchange"""
        try:
            orderbook = await self.exchanges[exchange].get_orderbook(symbol)
            
            # Calculate weighted average price for the amount
            levels = orderbook.asks if side == 'buy' else orderbook.bids
            remaining = amount
            total_cost = Decimal('0')
            total_amount = Decimal('0')
            
            for price, level in levels.items():
                available = min(remaining, level.amount)
                total_cost += price * available
                total_amount += available
                remaining -= available
                
                if remaining == 0:
                    break
            
            if total_amount == 0:
                return exchange, Decimal('Infinity'), Decimal('0')
            
            avg_price = total_cost / total_amount
            return exchange, avg_price, total_amount
            
        except Exception as e:
            logger.error(f"Error getting quote from {exchange}: {e}")
            raise
    
    async def smart_order_routing(
        self,
        symbol: str,
        side: str,
        amount: Decimal,
        order_type: str = 'limit',
        limit_price: Optional[Decimal] = None,
        urgency: float = 0.5  # 0=patient, 1=urgent
    ) -> List[UnifiedOrder]:
        """
        Smart order routing with split execution across exchanges
        """
        logger.info(f"Smart routing for {amount} {symbol} {side}")
        
        # Get liquidity distribution
        liquidity_map = await self._analyze_liquidity(symbol, side, amount)
        
        # Calculate optimal splits based on:
        # 1. Available liquidity
        # 2. Exchange fees
        # 3. Historical latency
        # 4. Reliability scores
        # 5. Urgency parameter
        
        orders = []
        remaining = amount
        
        for exchange, metrics in sorted(
            liquidity_map.items(),
            key=lambda x: self._calculate_routing_score(x[1], urgency),
            reverse=True
        ):
            if remaining <= 0:
                break
            
            # Calculate allocation
            allocation = min(remaining, metrics['available_amount'])
            
            # Skip if too small
            min_size = self.exchange_info[exchange].min_order_size.get(symbol, Decimal('0'))
            if allocation < min_size:
                continue
            
            # Create order
            order = UnifiedOrder(
                exchange=exchange,
                symbol=symbol,
                side=side,
                type=order_type,
                amount=allocation,
                price=limit_price or metrics['price']
            )
            
            orders.append(order)
            remaining -= allocation
            
            # Update routing metrics
            ROUTING_DECISIONS.labels(
                exchange=exchange,
                reason='liquidity' if metrics['available_amount'] > amount * Decimal('0.5') else 'split'
            ).inc()
        
        if remaining > 0:
            logger.warning(f"Could not route full amount. Remaining: {remaining}")
        
        return orders
    
    def _calculate_routing_score(self, metrics: Dict, urgency: float) -> float:
        """Calculate routing score for an exchange"""
        # Weighted scoring based on multiple factors
        price_score = float(metrics.get('price_score', 0.5))
        liquidity_score = float(metrics.get('liquidity_score', 0.5))
        latency_score = 1.0 - (metrics.get('latency_ms', 100) / 1000.0)  # Lower is better
        reliability_score = metrics.get('reliability_score', 0.9)
        fee_score = 1.0 - float(metrics.get('total_fee', 0.001))
        
        # Adjust weights based on urgency
        if urgency > 0.7:
            # Urgent: prioritize liquidity and latency
            weights = {
                'price': 0.2,
                'liquidity': 0.4,
                'latency': 0.3,
                'reliability': 0.05,
                'fee': 0.05
            }
        else:
            # Patient: prioritize price and fees
            weights = {
                'price': 0.4,
                'liquidity': 0.2,
                'latency': 0.1,
                'reliability': 0.1,
                'fee': 0.2
            }
        
        score = (
            weights['price'] * price_score +
            weights['liquidity'] * liquidity_score +
            weights['latency'] * latency_score +
            weights['reliability'] * reliability_score +
            weights['fee'] * fee_score
        )
        
        return score
    
    async def execute_orders(self, orders: List[UnifiedOrder]) -> List[Dict[str, Any]]:
        """Execute multiple orders in parallel"""
        tasks = []
        
        for order in orders:
            if order.exchange in self.exchanges:
                task = self.exchanges[order.exchange].place_order(order)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Order failed on {orders[i].exchange}: {result}")
            else:
                successful.append(result)
        
        return successful
    
    async def detect_arbitrage(self, symbols: Optional[List[str]] = None) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across exchanges"""
        symbols = symbols or self._get_common_symbols()
        opportunities = []
        
        for symbol in symbols:
            try:
                # Get orderbooks from all exchanges
                orderbooks = {}
                for exchange in self.exchanges:
                    try:
                        ob = await self.exchanges[exchange].get_orderbook(symbol)
                        orderbooks[exchange] = ob
                    except:
                        continue
                
                # Compare prices across exchanges
                for ex1, ob1 in orderbooks.items():
                    for ex2, ob2 in orderbooks.items():
                        if ex1 == ex2:
                            continue
                        
                        # Check if we can buy on ex1 and sell on ex2
                        best_bid_ex2 = ob2.get_best_bid()
                        best_ask_ex1 = ob1.get_best_ask()
                        
                        if not best_bid_ex2 or not best_ask_ex1:
                            continue
                        
                        # Calculate profit
                        buy_price = best_ask_ex1[0]
                        sell_price = best_bid_ex2[0]
                        
                        # Account for fees
                        total_fee = (
                            self.exchange_info[ex1].taker_fee +
                            self.exchange_info[ex2].taker_fee
                        )
                        
                        profit_percent = ((sell_price - buy_price) / buy_price - total_fee) * 100
                        
                        if profit_percent > float(self.min_arbitrage_profit):
                            max_amount = min(best_ask_ex1[1], best_bid_ex2[1])
                            
                            opportunity = ArbitrageOpportunity(
                                symbol=symbol,
                                buy_exchange=ex1,
                                sell_exchange=ex2,
                                buy_price=buy_price,
                                sell_price=sell_price,
                                max_amount=max_amount,
                                profit_percent=Decimal(str(profit_percent))
                            )
                            
                            opportunities.append(opportunity)
                            ARBITRAGE_OPPORTUNITIES.inc()
                
            except Exception as e:
                logger.error(f"Error detecting arbitrage for {symbol}: {e}")
        
        # Sort by profit
        opportunities.sort(key=lambda x: x.profit_percent, reverse=True)
        
        # Update internal tracking
        self.arbitrage_opportunities = SortedList(
            opportunities[:100],  # Keep top 100
            key=lambda x: -x.profit_percent
        )
        
        return opportunities
    
    def _get_common_symbols(self) -> List[str]:
        """Get symbols available on multiple exchanges"""
        if not self.exchanges:
            return []
        
        # Get symbols from each exchange
        all_symbols = []
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'symbols'):
                all_symbols.append(set(exchange.symbols))
        
        if not all_symbols:
            return []
        
        # Find intersection
        common = all_symbols[0]
        for symbols in all_symbols[1:]:
            common = common.intersection(symbols)
        
        return list(common)
    
    async def _analyze_liquidity(
        self,
        symbol: str,
        side: str,
        amount: Decimal
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze liquidity across exchanges"""
        liquidity_map = {}
        
        # Get quotes from all exchanges
        tasks = []
        exchanges = list(self.exchanges.keys())
        
        for exchange in exchanges:
            task = self._get_exchange_quote(exchange, symbol, side, amount)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        prices = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            
            exchange, price, available = result
            prices.append(float(price))
            
            liquidity_map[exchange] = {
                'price': price,
                'available_amount': available,
                'liquidity_score': float(available / amount),
                'latency_ms': np.mean(list(self.latency_tracker[exchange])) if self.latency_tracker[exchange] else 100,
                'reliability_score': self.exchange_info[exchange].reliability_score,
                'total_fee': float(self.exchange_info[exchange].taker_fee)
            }
        
        # Calculate price scores (normalized)
        if prices:
            best_price = min(prices) if side == 'buy' else max(prices)
            worst_price = max(prices) if side == 'buy' else min(prices)
            price_range = worst_price - best_price if worst_price != best_price else 1
            
            for exchange, metrics in liquidity_map.items():
                price = float(metrics['price'])
                if side == 'buy':
                    metrics['price_score'] = 1.0 - ((price - best_price) / price_range)
                else:
                    metrics['price_score'] = (price - worst_price) / price_range
        
        return liquidity_map
    
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
    
    async def close(self):
        """Clean shutdown"""
        logger.info("Closing unified exchange interface...")
        
        if self._update_task:
            self._update_task.cancel()
        
        # Close all exchanges
        tasks = []
        for exchange in self.exchanges.values():
            if hasattr(exchange, 'close'):
                tasks.append(exchange.close())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Unified exchange interface closed")


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
        
    async def initialize(self):
        await self.exchange.load_markets()
        self.symbols = list(self.exchange.markets.keys())
        
    async def place_order(self, order: UnifiedOrder) -> Dict[str, Any]:
        if order.type == 'market':
            return await self.exchange.create_market_order(
                order.symbol,
                order.side,
                float(order.amount)
            )
        else:
            return await self.exchange.create_limit_order(
                order.symbol,
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
            
        return orderbook
    
    async def get_balance(self) -> Dict[str, Decimal]:
        balance = await self.exchange.fetch_balance()
        return {
            asset: Decimal(str(amount))
            for asset, amount in balance['free'].items()
            if amount > 0
        }


# Placeholder classes for DEX connectors (would need full implementation)
class UniswapV3Connector(ExchangeConnector):
    def __init__(self, web3: Web3, config: Dict[str, Any]):
        self.web3 = web3
        self.config = config
        # Implementation would include contract ABIs, pool addresses, etc.
    
    async def initialize(self):
        pass
    
    async def place_order(self, order: UnifiedOrder) -> Dict[str, Any]:
        # Would implement swap logic
        pass
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        # DEX swaps are atomic, no cancellation
        raise NotImplementedError("Cannot cancel DEX swaps")
    
    async def get_orderbook(self, symbol: str) -> OrderBook:
        # Would calculate from pool reserves
        pass
    
    async def get_balance(self) -> Dict[str, Decimal]:
        # Would check wallet balances
        pass


class PancakeSwapV3Connector(ExchangeConnector):
    # Similar to Uniswap but on BSC
    pass


class GMXConnector(ExchangeConnector):
    # Perpetual DEX implementation
    pass
