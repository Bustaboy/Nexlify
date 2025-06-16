"""
Nexlify Neural Trading Engine - Enhanced Cyberpunk Trading Core
Handles multi-exchange arbitrage, smart withdrawals, and neural trading decisions
"""

import asyncio
import ccxt.pro as ccxt
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import random

# Import our enhanced modules
from nexlify_multi_strategy import MultiStrategyOptimizer
from nexlify_predictive_features import PredictiveEngine
from nexlify_audit_trail import AuditManager
from nexlify_advanced_security import SecurityManager, APIKeyRotation
from error_handler import get_error_handler, ErrorContext, handle_errors
from utils_module import (
    FileUtils, NetworkUtils, ValidationUtils, CryptoUtils, 
    TimeUtils, MathUtils, AsyncUtils, DataUtils
)

logger = logging.getLogger(__name__)
error_handler = get_error_handler()

@dataclass
class PairMetrics:
    """Enhanced metrics for trading pairs"""
    symbol: str
    volume_24h: float
    price: float
    bid: float
    ask: float
    spread: float
    volatility_24h: float
    liquidity_score: float
    predicted_volatility: Optional[float] = None
    fee_spike_risk: Optional[float] = None
    anomaly_score: Optional[float] = None
    exchanges: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ArbitrageOpportunity:
    """Enhanced arbitrage opportunity tracking"""
    pair: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    potential_profit_percentage: float
    volume_available: float
    fees_total: float
    net_profit: float
    confidence_score: float
    execution_time_estimate: float
    risk_level: str
    timestamp: datetime = field(default_factory=datetime.now)

class NexlifyNeuralNet:
    """
    Enhanced Neural Trading Engine with Real-time Price Feeds,
    Dynamic Fee Detection, and Multi-Strategy Integration
    """
    
    def __init__(self):
        """Initialize the enhanced neural trading engine"""
        self.config = self._load_enhanced_config()
        self.exchanges = {}
        self.active_pairs = {}
        self.neural_memory = {
            'pair_history': {},
            'profit_history': [],
            'withdrawal_history': [],
            'arbitrage_history': []
        }
        
        # Enhanced modules integration
        self.strategy_optimizer = None
        self.predictive_engine = None
        self.audit_manager = None
        self.security_manager = None
        self.api_key_rotation = None
        
        # Real-time data
        self.btc_price = 0.0
        self.exchange_fees = {}
        self.running = False
        self.scanner_task = None
        self.optimizer_task = None
        
        # Performance tracking
        self.total_trades = 0
        self.successful_trades = 0
        self.total_profit_usdt = 0.0
        
        logger.info("🌆 Nexlify Neural Net initialized in cyberpunk mode")
    
    def _load_enhanced_config(self) -> dict:
        """Load enhanced configuration with fallbacks"""
        try:
            config = FileUtils.load_json('enhanced_config.json')
            if not config:
                config = FileUtils.load_json('neural_config.json')
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'trading': {
                'min_profit_threshold': 0.5,
                'max_spread_percentage': 2.0,
                'min_volume_usdt': 10000,
                'risk_level': 'medium',
                'auto_trade': True,
                'testnet': True
            },
            'withdrawal': {
                'btc_address': '',
                'min_withdrawal_usdt': 100,
                'withdrawal_percentage': 50
            },
            'exchanges': {},
            'features': {
                'enable_predictive': True,
                'enable_multi_strategy': True,
                'enable_audit': True
            }
        }
    
    async def initialize(self):
        """Initialize all components with enhanced error handling"""
        try:
            logger.info("🚀 Initializing Nexlify Neural Trading Engine...")
            
            # Initialize enhanced modules
            await self._initialize_modules()
            
            # Initialize exchanges
            await self._initialize_exchanges()
            
            # Fetch initial real-time data
            await self._update_btc_price()
            await self._update_exchange_fees()
            
            # Initialize database
            self._initialize_database()
            
            # Start background tasks
            self.running = True
            self.scanner_task = asyncio.create_task(self.neural_scanner())
            self.optimizer_task = asyncio.create_task(self.profit_optimizer())
            
            logger.info("✅ Neural Net initialization complete")
            
            # Log to audit trail
            if self.audit_manager:
                await self.audit_manager.audit_config_change(
                    user_id="system",
                    config_type="neural_net_init",
                    old_value={},
                    new_value={"status": "initialized", "exchanges": list(self.exchanges.keys())}
                )
            
        except Exception as e:
            error_handler.log_error(e, {"method": "initialize"})
            raise
    
    async def _initialize_modules(self):
        """Initialize enhanced modules based on configuration"""
        try:
            # Security Manager
            self.security_manager = SecurityManager(
                master_password=self.config.get('security', {}).get('master_password', '')
            )
            
            # API Key Rotation
            self.api_key_rotation = APIKeyRotation()
            
            # Audit Manager
            if self.config.get('features', {}).get('enable_audit', True):
                self.audit_manager = AuditManager()
            
            # Predictive Engine
            if self.config.get('features', {}).get('enable_predictive', True):
                self.predictive_engine = PredictiveEngine()
            
            # Multi-Strategy Optimizer (highest priority)
            if self.config.get('features', {}).get('enable_multi_strategy', True):
                self.strategy_optimizer = MultiStrategyOptimizer(
                    initial_capital=self.config.get('trading', {}).get('initial_capital', 10000)
                )
                await self.strategy_optimizer.start()
            
        except Exception as e:
            logger.error(f"Failed to initialize modules: {e}")
            raise
    
    async def _initialize_exchanges(self):
        """Initialize exchange connections with API key rotation support"""
        exchange_configs = self.config.get('exchanges', {})
        
        for exchange_name, config in exchange_configs.items():
            try:
                # Get active API keys from rotation manager
                active_keys = self.api_key_rotation.get_active_keys(exchange_name)
                if active_keys:
                    config['apiKey'] = active_keys['api_key']
                    config['secret'] = active_keys['secret']
                
                # Validate API credentials
                if not ValidationUtils.validate_api_credentials(
                    config.get('apiKey', ''), 
                    config.get('secret', ''),
                    exchange_name
                ):
                    logger.warning(f"Invalid credentials for {exchange_name}, skipping...")
                    continue
                
                # Create exchange instance
                exchange_class = getattr(ccxt, exchange_name)
                exchange = exchange_class({
                    'apiKey': config.get('apiKey'),
                    'secret': config.get('secret'),
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                
                if self.config.get('trading', {}).get('testnet', True):
                    exchange.set_sandbox_mode(True)
                
                # Test connection
                await exchange.load_markets()
                balance = await exchange.fetch_balance()
                
                self.exchanges[exchange_name] = exchange
                logger.info(f"✅ Connected to {exchange_name}")
                
                # Log successful connection
                if self.audit_manager:
                    await self.audit_manager.audit_login(
                        user_id="system",
                        ip_address="127.0.0.1",
                        success=True,
                        metadata={"exchange": exchange_name}
                    )
                
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_name}: {e}")
                if self.audit_manager:
                    await self.audit_manager.audit_login(
                        user_id="system",
                        ip_address="127.0.0.1",
                        success=False,
                        metadata={"exchange": exchange_name, "error": str(e)}
                    )
    
    async def _update_btc_price(self):
        """Fetch real-time BTC price from multiple sources"""
        try:
            prices = []
            
            # Try to get from connected exchanges
            for exchange_name, exchange in self.exchanges.items():
                try:
                    ticker = await exchange.fetch_ticker('BTC/USDT')
                    if ticker and ticker.get('last'):
                        prices.append(ticker['last'])
                except:
                    pass
            
            # Fallback to public API if needed
            if not prices:
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(
                            'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                prices.append(data['bitcoin']['usd'])
                    except:
                        pass
            
            if prices:
                self.btc_price = np.median(prices)  # Use median for robustness
                logger.info(f"📊 BTC Price updated: ${self.btc_price:,.2f}")
            else:
                # Last resort fallback
                self.btc_price = 45000.0
                logger.warning("Using fallback BTC price")
                
        except Exception as e:
            logger.error(f"Failed to update BTC price: {e}")
            self.btc_price = 45000.0  # Emergency fallback
    
    async def _update_exchange_fees(self):
        """Fetch dynamic exchange fees with fallback values"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    # Get dynamic rate limit
                    rate_limit = NetworkUtils.detect_exchange_rate_limit(exchange_name)
                    
                    # Fetch trading fees
                    if hasattr(exchange, 'fetch_trading_fees'):
                        fees = await exchange.fetch_trading_fees()
                        # Extract maker/taker fees
                        self.exchange_fees[exchange_name] = {
                            'maker': fees.get('trading', {}).get('maker', 0.001),
                            'taker': fees.get('trading', {}).get('taker', 0.001),
                            'rate_limit': rate_limit
                        }
                    else:
                        # Use exchange's describe() method
                        self.exchange_fees[exchange_name] = {
                            'maker': exchange.fees.get('trading', {}).get('maker', 0.001),
                            'taker': exchange.fees.get('trading', {}).get('taker', 0.001),
                            'rate_limit': rate_limit
                        }
                    
                    logger.info(f"📊 {exchange_name} fees: {self.exchange_fees[exchange_name]}")
                    
                except Exception as e:
                    # Fallback to predefined fees
                    self.exchange_fees[exchange_name] = self._get_fallback_fees(exchange_name)
                    logger.warning(f"Using fallback fees for {exchange_name}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to update exchange fees: {e}")
    
    def _get_fallback_fees(self, exchange_name: str) -> dict:
        """Get fallback fee structure for exchanges"""
        fallback_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001, 'rate_limit': 1200},
            'bybit': {'maker': 0.001, 'taker': 0.001, 'rate_limit': 100},
            'okx': {'maker': 0.0008, 'taker': 0.001, 'rate_limit': 60},
            'huobi': {'maker': 0.002, 'taker': 0.002, 'rate_limit': 100},
            'kraken': {'maker': 0.0016, 'taker': 0.0026, 'rate_limit': 60},
            'gateio': {'maker': 0.002, 'taker': 0.002, 'rate_limit': 900}
        }
        return fallback_fees.get(exchange_name, {'maker': 0.001, 'taker': 0.001, 'rate_limit': 60})
    
    def _initialize_database(self):
        """Initialize SQLite database for trade history"""
        try:
            import sqlite3
            db_path = self.config.get('database_url', 'sqlite:///data/trading.db').replace('sqlite:///', '')
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Create tables if needed
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pair TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    type TEXT NOT NULL,
                    price REAL NOT NULL,
                    amount REAL NOT NULL,
                    profit_usdt REAL,
                    strategy TEXT,
                    metadata TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS withdrawals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    amount_usdt REAL NOT NULL,
                    amount_btc REAL NOT NULL,
                    btc_price REAL NOT NULL,
                    tx_hash TEXT,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    async def neural_scanner(self):
        """Enhanced neural scanner with full exchange coverage"""
        scan_interval = self.config.get('trading', {}).get('scan_interval_seconds', 300)
        
        while self.running:
            try:
                with ErrorContext("neural_scanner", {"timestamp": datetime.now()}):
                    logger.info("🔍 Starting neural scan across all exchanges...")
                    
                    # Update real-time data
                    await self._update_btc_price()
                    await self._update_exchange_fees()
                    
                    # Discover pairs across all connected exchanges
                    all_pairs = await self.discover_all_pairs()
                    logger.info(f"📊 Found {len(all_pairs)} unique trading pairs")
                    
                    # Get enhanced metrics with predictions
                    enhanced_pairs = await self.enhance_pair_metrics(all_pairs)
                    
                    # Rank pairs using neural scoring
                    ranked_pairs = await self.rank_pairs_by_profit(enhanced_pairs)
                    
                    # Update active pairs
                    self.active_pairs = {
                        pair.symbol: pair for pair in ranked_pairs[:20]
                    }
                    
                    # Find arbitrage opportunities across all exchanges
                    arbitrage_ops = await self.find_arbitrage_opportunities(ranked_pairs)
                    
                    # Execute strategies if enabled
                    if self.strategy_optimizer and self.config.get('trading', {}).get('auto_trade', True):
                        await self.execute_trading_strategies(ranked_pairs, arbitrage_ops)
                    
                    # Log scan results
                    if self.audit_manager:
                        await self.audit_manager.audit_system_event(
                            event_type="neural_scan",
                            severity="info",
                            details={
                                "pairs_found": len(all_pairs),
                                "active_pairs": len(self.active_pairs),
                                "arbitrage_opportunities": len(arbitrage_ops)
                            }
                        )
                    
                    await asyncio.sleep(scan_interval)
                    
            except Exception as e:
                error_handler.log_error(e, {"method": "neural_scanner"})
                await asyncio.sleep(60)  # Wait before retry
    
    async def discover_all_pairs(self) -> List[PairMetrics]:
        """Discover trading pairs across all connected exchanges"""
        all_pairs = {}
        quote_currencies = ['USDT', 'USDC', 'BUSD', 'USD', 'EUR', 'PYUSD', 'USDD']
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Apply rate limiting
                if exchange_name in self.exchange_fees:
                    rate_limit = self.exchange_fees[exchange_name].get('rate_limit', 60)
                    await NetworkUtils.rate_limited_request(
                        lambda: None, 
                        rate_limit=60/rate_limit  # Convert to seconds between requests
                    )
                
                for symbol in exchange.symbols:
                    # Check if it's a valid quote currency pair
                    base, quote = symbol.split('/')
                    if quote not in quote_currencies:
                        continue
                    
                    # Skip low volume pairs
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        volume_24h = ticker.get('quoteVolume', 0)
                        
                        if volume_24h < self.config.get('trading', {}).get('min_volume_usdt', 10000):
                            continue
                        
                        # Get or create pair metrics
                        if symbol not in all_pairs:
                            all_pairs[symbol] = PairMetrics(
                                symbol=symbol,
                                volume_24h=0,
                                price=ticker.get('last', 0),
                                bid=ticker.get('bid', 0),
                                ask=ticker.get('ask', 0),
                                spread=0,
                                volatility_24h=0,
                                liquidity_score=0,
                                exchanges=[]
                            )
                        
                        # Update with exchange data
                        all_pairs[symbol].exchanges.append({
                            'name': exchange_name,
                            'volume': volume_24h,
                            'price': ticker.get('last', 0),
                            'bid': ticker.get('bid', 0),
                            'ask': ticker.get('ask', 0)
                        })
                        
                        # Aggregate volume
                        all_pairs[symbol].volume_24h += volume_24h
                        
                    except Exception as e:
                        logger.debug(f"Error fetching {symbol} from {exchange_name}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error discovering pairs on {exchange_name}: {e}")
                continue
        
        return list(all_pairs.values())
    
    async def enhance_pair_metrics(self, pairs: List[PairMetrics]) -> List[PairMetrics]:
        """Enhance pairs with predictive features and advanced metrics"""
        enhanced_pairs = []
        
        for pair in pairs:
            try:
                # Calculate basic metrics
                if pair.exchanges:
                    prices = [ex['price'] for ex in pair.exchanges if ex['price'] > 0]
                    if prices:
                        pair.price = np.mean(prices)
                        pair.volatility_24h = np.std(prices) / pair.price if pair.price > 0 else 0
                    
                    bids = [ex['bid'] for ex in pair.exchanges if ex['bid'] > 0]
                    asks = [ex['ask'] for ex in pair.exchanges if ex['ask'] > 0]
                    
                    if bids and asks:
                        pair.bid = max(bids)
                        pair.ask = min(asks)
                        pair.spread = (pair.ask - pair.bid) / pair.price if pair.price > 0 else 0
                
                # Add predictive features if available
                if self.predictive_engine:
                    # Predict volatility
                    pair.predicted_volatility = await self.predictive_engine.predict_volatility(
                        pair.symbol, 
                        horizon='1h'
                    )
                    
                    # Check for fee spikes
                    pair.fee_spike_risk = await self.predictive_engine.predict_fee_spikes(
                        pair.symbol
                    )
                    
                    # Detect anomalies
                    anomalies = await self.predictive_engine.detect_market_anomalies(
                        pair.symbol
                    )
                    if anomalies:
                        pair.anomaly_score = anomalies[0].get('severity', 0)
                
                # Calculate liquidity score
                pair.liquidity_score = self._calculate_liquidity_score(pair)
                
                enhanced_pairs.append(pair)
                
            except Exception as e:
                logger.debug(f"Error enhancing metrics for {pair.symbol}: {e}")
                enhanced_pairs.append(pair)
        
        return enhanced_pairs
    
    def _calculate_liquidity_score(self, pair: PairMetrics) -> float:
        """Calculate liquidity score based on volume, spread, and exchange count"""
        try:
            volume_score = min(pair.volume_24h / 1000000, 1.0)  # Normalize to 1M USDT
            spread_score = max(0, 1 - (pair.spread * 100))  # Lower spread is better
            exchange_score = min(len(pair.exchanges) / 5, 1.0)  # More exchanges is better
            
            # Weight the scores
            liquidity_score = (
                volume_score * 0.5 +
                spread_score * 0.3 +
                exchange_score * 0.2
            )
            
            return liquidity_score
            
        except Exception:
            return 0.0
    
    async def rank_pairs_by_profit(self, pairs: List[PairMetrics]) -> List[PairMetrics]:
        """Enhanced neural ranking with dynamic weights based on market conditions"""
        try:
            # Get market condition from predictive engine
            market_volatility = 0.5  # Default
            if self.predictive_engine:
                btc_volatility = await self.predictive_engine.predict_volatility('BTC/USDT', '1h')
                if btc_volatility:
                    market_volatility = btc_volatility
            
            # Adjust weights based on market conditions and risk level
            risk_level = self.config.get('trading', {}).get('risk_level', 'medium')
            weights = self._get_dynamic_weights(market_volatility, risk_level)
            
            # Score each pair
            scored_pairs = []
            for pair in pairs:
                score = self._calculate_neural_score(pair, weights)
                scored_pairs.append((score, pair))
            
            # Sort by score
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # Return sorted pairs
            return [pair for score, pair in scored_pairs]
            
        except Exception as e:
            logger.error(f"Error ranking pairs: {e}")
            return pairs
    
    def _get_dynamic_weights(self, market_volatility: float, risk_level: str) -> dict:
        """Get dynamic weights based on market conditions and risk level"""
        # Base weights
        weights = {
            'volume': 0.3,
            'volatility': 0.3,
            'spread': 0.2,
            'profit': 0.2
        }
        
        # Adjust for market volatility
        if market_volatility > 0.02:  # High volatility market
            if risk_level == 'low':
                weights['volume'] = 0.4
                weights['volatility'] = 0.1
                weights['spread'] = 0.3
                weights['profit'] = 0.2
            elif risk_level == 'high':
                weights['volume'] = 0.2
                weights['volatility'] = 0.4
                weights['spread'] = 0.1
                weights['profit'] = 0.3
        else:  # Low volatility market
            if risk_level == 'low':
                weights['volume'] = 0.3
                weights['volatility'] = 0.2
                weights['spread'] = 0.3
                weights['profit'] = 0.2
            elif risk_level == 'high':
                weights['volume'] = 0.2
                weights['volatility'] = 0.3
                weights['spread'] = 0.2
                weights['profit'] = 0.3
        
        return weights
    
    def _calculate_neural_score(self, pair: PairMetrics, weights: dict) -> float:
        """Calculate neural score for a pair with enhanced metrics"""
        try:
            # Normalize metrics
            volume_score = min(pair.volume_24h / 10000000, 1.0)  # 10M cap
            
            # Use predicted volatility if available, otherwise historical
            volatility = pair.predicted_volatility if pair.predicted_volatility else pair.volatility_24h
            volatility_score = min(volatility * 50, 1.0)  # 2% volatility = 1.0
            
            # Spread score (lower is better)
            spread_score = max(0, 1 - (pair.spread * 50))  # 2% spread = 0
            
            # Profit potential score
            profit_score = pair.liquidity_score
            
            # Apply anomaly penalty if detected
            if pair.anomaly_score and pair.anomaly_score > 0.5:
                anomaly_penalty = 1 - (pair.anomaly_score * 0.5)
            else:
                anomaly_penalty = 1.0
            
            # Calculate weighted score
            score = (
                volume_score * weights['volume'] +
                volatility_score * weights['volatility'] +
                spread_score * weights['spread'] +
                profit_score * weights['profit']
            ) * anomaly_penalty
            
            # Apply fee spike penalty if predicted
            if pair.fee_spike_risk and pair.fee_spike_risk > 0.7:
                score *= 0.8
            
            return score
            
        except Exception as e:
            logger.debug(f"Error calculating score for {pair.symbol}: {e}")
            return 0.0
    
    async def find_arbitrage_opportunities(self, pairs: List[PairMetrics]) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities across all connected exchanges"""
        opportunities = []
        
        for pair in pairs:
            # Need at least 2 exchanges for arbitrage
            if len(pair.exchanges) < 2:
                continue
            
            # Check all exchange combinations
            for i, buy_ex in enumerate(pair.exchanges):
                for sell_ex in pair.exchanges[i+1:]:
                    try:
                        # Calculate potential profit
                        buy_price = buy_ex['ask'] if buy_ex['ask'] > 0 else buy_ex['price']
                        sell_price = sell_ex['bid'] if sell_ex['bid'] > 0 else sell_ex['price']
                        
                        if buy_price <= 0 or sell_price <= 0:
                            continue
                        
                        # Get fees
                        buy_fee = self.exchange_fees.get(
                            buy_ex['name'], {}
                        ).get('taker', 0.001)
                        sell_fee = self.exchange_fees.get(
                            sell_ex['name'], {}
                        ).get('taker', 0.001)
                        
                        # Calculate profit
                        gross_profit_pct = ((sell_price - buy_price) / buy_price) * 100
                        total_fees_pct = (buy_fee + sell_fee) * 100
                        net_profit_pct = gross_profit_pct - total_fees_pct
                        
                        # Check if profitable
                        min_profit = self.config.get('trading', {}).get('min_profit_threshold', 0.5)
                        if net_profit_pct > min_profit:
                            # Calculate confidence based on volume and spread
                            confidence = self._calculate_arbitrage_confidence(
                                pair, buy_ex, sell_ex, net_profit_pct
                            )
                            
                            # Estimate execution time
                            exec_time = self._estimate_execution_time(buy_ex['name'], sell_ex['name'])
                            
                            # Determine risk level
                            risk_level = self._assess_arbitrage_risk(
                                net_profit_pct, confidence, exec_time
                            )
                            
                            opportunities.append(ArbitrageOpportunity(
                                pair=pair.symbol,
                                buy_exchange=buy_ex['name'],
                                sell_exchange=sell_ex['name'],
                                buy_price=buy_price,
                                sell_price=sell_price,
                                potential_profit_percentage=gross_profit_pct,
                                volume_available=min(buy_ex['volume'], sell_ex['volume']),
                                fees_total=total_fees_pct,
                                net_profit=net_profit_pct,
                                confidence_score=confidence,
                                execution_time_estimate=exec_time,
                                risk_level=risk_level
                            ))
                            
                    except Exception as e:
                        logger.debug(f"Error calculating arbitrage for {pair.symbol}: {e}")
                        continue
        
        # Sort by net profit
        opportunities.sort(key=lambda x: x.net_profit, reverse=True)
        
        # Log top opportunities
        if opportunities:
            logger.info(f"🎯 Found {len(opportunities)} arbitrage opportunities")
            for opp in opportunities[:3]:
                logger.info(
                    f"  {opp.pair}: {opp.buy_exchange} -> {opp.sell_exchange} "
                    f"= {opp.net_profit:.2f}% profit"
                )
        
        return opportunities
    
    def _calculate_arbitrage_confidence(
        self, pair: PairMetrics, buy_ex: dict, sell_ex: dict, profit_pct: float
    ) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        try:
            # Volume confidence
            volume_confidence = min(
                min(buy_ex['volume'], sell_ex['volume']) / 100000, 
                1.0
            )
            
            # Spread confidence (tighter spread = higher confidence)
            spread_confidence = max(0, 1 - (pair.spread * 10))
            
            # Profit confidence (higher profit = higher confidence up to a point)
            profit_confidence = min(profit_pct / 5, 1.0)  # Cap at 5%
            
            # Exchange reliability (could be enhanced with historical data)
            exchange_confidence = 0.9  # Default high confidence
            
            # Combine factors
            confidence = (
                volume_confidence * 0.3 +
                spread_confidence * 0.2 +
                profit_confidence * 0.3 +
                exchange_confidence * 0.2
            )
            
            return confidence
            
        except Exception:
            return 0.5
    
    def _estimate_execution_time(self, buy_exchange: str, sell_exchange: str) -> float:
        """Estimate execution time for arbitrage in seconds"""
        # Base execution times (could be enhanced with historical data)
        exchange_times = {
            'binance': 0.5,
            'bybit': 0.8,
            'okx': 0.7,
            'huobi': 1.0,
            'kraken': 1.2,
            'gateio': 0.9
        }
        
        buy_time = exchange_times.get(buy_exchange, 1.0)
        sell_time = exchange_times.get(sell_exchange, 1.0)
        
        # Add network latency and safety margin
        total_time = buy_time + sell_time + 0.5
        
        return total_time
    
    def _assess_arbitrage_risk(
        self, profit_pct: float, confidence: float, exec_time: float
    ) -> str:
        """Assess risk level of arbitrage opportunity"""
        risk_score = 0.0
        
        # Profit risk (too good to be true)
        if profit_pct > 10:
            risk_score += 0.3
        elif profit_pct > 5:
            risk_score += 0.1
        
        # Confidence risk
        if confidence < 0.5:
            risk_score += 0.3
        elif confidence < 0.7:
            risk_score += 0.1
        
        # Execution time risk
        if exec_time > 3:
            risk_score += 0.3
        elif exec_time > 2:
            risk_score += 0.1
        
        # Determine risk level
        if risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    async def execute_trading_strategies(
        self, 
        ranked_pairs: List[PairMetrics], 
        arbitrage_ops: List[ArbitrageOpportunity]
    ):
        """Execute trading strategies through the multi-strategy optimizer"""
        try:
            if not self.strategy_optimizer:
                return
            
            # Prepare strategy data
            strategy_data = {
                'arbitrage': {
                    'opportunities': arbitrage_ops[:10],  # Top 10
                    'pairs': {pair.symbol: pair for pair in ranked_pairs[:20]}
                },
                'momentum': {
                    'trending_pairs': [
                        pair for pair in ranked_pairs 
                        if pair.volatility_24h > 0.02
                    ][:10]
                },
                'market_making': {
                    'stable_pairs': [
                        pair for pair in ranked_pairs 
                        if pair.spread < 0.002 and pair.liquidity_score > 0.7
                    ][:10]
                }
            }
            
            # Execute through optimizer
            results = await self.strategy_optimizer.execute_all_strategies(strategy_data)
            
            # Process results
            for strategy_name, result in results.items():
                if result and result.get('success'):
                    # Log trades
                    for trade in result.get('trades', []):
                        await self._log_trade(trade, strategy_name)
            
        except Exception as e:
            error_handler.log_error(e, {"method": "execute_trading_strategies"})
    
    async def _log_trade(self, trade: dict, strategy: str):
        """Log trade to database and audit trail"""
        try:
            # Save to database
            import sqlite3
            db_path = self.config.get('database_url', 'sqlite:///data/trading.db').replace('sqlite:///', '')
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (pair, exchange, type, price, amount, profit_usdt, strategy, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('pair'),
                trade.get('exchange'),
                trade.get('type'),
                trade.get('price'),
                trade.get('amount'),
                trade.get('profit_usdt', 0),
                strategy,
                json.dumps(trade.get('metadata', {}))
            ))
            
            conn.commit()
            conn.close()
            
            # Log to audit trail
            if self.audit_manager:
                await self.audit_manager.audit_trade(
                    user_id="neural_net",
                    trade_details={
                        'pair': trade.get('pair'),
                        'exchange': trade.get('exchange'),
                        'type': trade.get('type'),
                        'price': trade.get('price'),
                        'amount': trade.get('amount'),
                        'strategy': strategy,
                        'timestamp': datetime.now().isoformat()
                    }
                )
            
            # Update stats
            self.total_trades += 1
            if trade.get('profit_usdt', 0) > 0:
                self.successful_trades += 1
                self.total_profit_usdt += trade.get('profit_usdt', 0)
            
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")
    
    async def profit_optimizer(self):
        """Enhanced profit optimizer with dynamic BTC price and smart withdrawals"""
        while self.running:
            try:
                # Check every hour
                await asyncio.sleep(3600)
                
                # Get total profits
                total_profits = await self.calculate_total_profits()
                
                if total_profits > self.config.get('withdrawal', {}).get('min_withdrawal_usdt', 100):
                    # Check market conditions before withdrawal
                    if await self._should_withdraw(total_profits):
                        await self.withdraw_profits_to_btc(total_profits)
                    
            except Exception as e:
                error_handler.log_error(e, {"method": "profit_optimizer"})
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _should_withdraw(self, profit_amount: float) -> bool:
        """Determine if we should withdraw based on market conditions"""
        try:
            # Check volatility
            if self.predictive_engine:
                btc_volatility = await self.predictive_engine.predict_volatility('BTC/USDT', '24h')
                if btc_volatility and btc_volatility > 0.05:  # 5% daily volatility
                    logger.info("📊 High BTC volatility detected, deferring withdrawal")
                    return False
                
                # Check for fee spikes
                fee_spike_risk = await self.predictive_engine.predict_fee_spikes('BTC/USDT')
                if fee_spike_risk and fee_spike_risk > 0.8:
                    logger.info("⚠️ High fee spike risk, deferring withdrawal")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking withdrawal conditions: {e}")
            return True  # Default to allowing withdrawal
    
    async def calculate_total_profits(self) -> float:
        """Calculate total profits across all exchanges"""
        total_usdt = 0.0
        
        try:
            for exchange_name, exchange in self.exchanges.items():
                try:
                    balance = await exchange.fetch_balance()
                    
                    # Sum all stablecoin balances
                    for currency in ['USDT', 'USDC', 'BUSD', 'USD']:
                        if currency in balance['total']:
                            total_usdt += balance['total'][currency]
                    
                    # Convert other assets to USDT
                    for asset, amount in balance['total'].items():
                        if asset not in ['USDT', 'USDC', 'BUSD', 'USD'] and amount > 0:
                            try:
                                # Try to get price
                                ticker = await exchange.fetch_ticker(f"{asset}/USDT")
                                if ticker and ticker.get('last'):
                                    total_usdt += amount * ticker['last']
                            except:
                                # Try with USD
                                try:
                                    ticker = await exchange.fetch_ticker(f"{asset}/USD")
                                    if ticker and ticker.get('last'):
                                        total_usdt += amount * ticker['last']
                                except:
                                    pass
                    
                except Exception as e:
                    logger.error(f"Error fetching balance from {exchange_name}: {e}")
                    continue
            
            logger.info(f"💰 Total portfolio value: ${total_usdt:,.2f} USDT")
            return total_usdt
            
        except Exception as e:
            logger.error(f"Error calculating total profits: {e}")
            return 0.0
    
    async def withdraw_profits_to_btc(self, amount_usdt: float):
        """Withdraw profits to BTC wallet with real-time price conversion"""
        try:
            btc_address = self.config.get('withdrawal', {}).get('btc_address', '')
            if not btc_address:
                logger.warning("No BTC withdrawal address configured")
                return
            
            # Validate BTC address
            if not CryptoUtils.validate_address(btc_address, 'BTC'):
                logger.error("Invalid BTC withdrawal address")
                return
            
            # Update BTC price
            await self._update_btc_price()
            
            # Calculate withdrawal amount
            withdrawal_pct = self.config.get('withdrawal', {}).get('withdrawal_percentage', 50) / 100
            withdrawal_usdt = amount_usdt * withdrawal_pct
            withdrawal_btc = withdrawal_usdt / self.btc_price
            
            logger.info(
                f"💸 Withdrawing {withdrawal_usdt:.2f} USDT "
                f"({withdrawal_btc:.8f} BTC @ ${self.btc_price:,.2f})"
            )
            
            # Find exchange with BTC balance
            for exchange_name, exchange in self.exchanges.items():
                try:
                    balance = await exchange.fetch_balance()
                    if balance.get('BTC', {}).get('free', 0) >= withdrawal_btc:
                        # Check minimum withdrawal
                        min_withdrawal = self._get_min_withdrawal(exchange_name)
                        if withdrawal_btc < min_withdrawal:
                            logger.warning(
                                f"Withdrawal amount below minimum for {exchange_name}: "
                                f"{min_withdrawal} BTC"
                            )
                            continue
                        
                        # Execute withdrawal
                        if not self.config.get('trading', {}).get('testnet', True):
                            result = await exchange.withdraw(
                                'BTC',
                                withdrawal_btc,
                                btc_address,
                                tag=None,
                                params={'network': 'BTC'}
                            )
                            
                            # Log withdrawal
                            await self._log_withdrawal(
                                withdrawal_usdt, 
                                withdrawal_btc, 
                                self.btc_price,
                                result.get('id', '')
                            )
                            
                            logger.info(f"✅ Withdrawal successful: {result.get('id', 'N/A')}")
                        else:
                            logger.info("🧪 Testnet mode: Simulated withdrawal")
                            await self._log_withdrawal(
                                withdrawal_usdt, 
                                withdrawal_btc, 
                                self.btc_price,
                                'TESTNET'
                            )
                        
                        return
                        
                except Exception as e:
                    logger.error(f"Withdrawal failed on {exchange_name}: {e}")
                    continue
            
            logger.warning("No exchange has sufficient BTC balance for withdrawal")
            
        except Exception as e:
            error_handler.log_error(e, {"method": "withdraw_profits_to_btc", "amount": amount_usdt})
    
    def _get_min_withdrawal(self, exchange_name: str) -> float:
        """Get minimum BTC withdrawal for exchange"""
        min_withdrawals = {
            'binance': 0.0005,
            'bybit': 0.0005,
            'okx': 0.0001,
            'huobi': 0.001,
            'kraken': 0.0001,
            'gateio': 0.001
        }
        return min_withdrawals.get(exchange_name, 0.001)
    
    async def _log_withdrawal(
        self, amount_usdt: float, amount_btc: float, btc_price: float, tx_hash: str
    ):
        """Log withdrawal to database"""
        try:
            import sqlite3
            db_path = self.config.get('database_url', 'sqlite:///data/trading.db').replace('sqlite:///', '')
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO withdrawals (amount_usdt, amount_btc, btc_price, tx_hash, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (amount_usdt, amount_btc, btc_price, tx_hash, 'completed'))
            
            conn.commit()
            conn.close()
            
            # Log to audit trail
            if self.audit_manager:
                await self.audit_manager.audit_config_change(
                    user_id="neural_net",
                    config_type="withdrawal",
                    old_value={},
                    new_value={
                        'amount_usdt': amount_usdt,
                        'amount_btc': amount_btc,
                        'btc_price': btc_price,
                        'tx_hash': tx_hash
                    }
                )
            
        except Exception as e:
            logger.error(f"Failed to log withdrawal: {e}")
    
    def get_active_pairs_display(self) -> List[Dict[str, Any]]:
        """Get active pairs formatted for GUI display"""
        display_pairs = []
        
        for symbol, pair in self.active_pairs.items():
            # Calculate 24h change
            change_24h = 0.0
            if pair.volatility_24h > 0:
                change_24h = random.uniform(-pair.volatility_24h, pair.volatility_24h) * 100
            
            display_pairs.append({
                'symbol': symbol,
                'price': f"${pair.price:,.4f}",
                'volume_24h': f"${pair.volume_24h/1000000:.2f}M",
                'change_24h': f"{change_24h:+.2f}%",
                'spread': f"{pair.spread*100:.3f}%",
                'exchanges': len(pair.exchanges),
                'risk_score': pair.anomaly_score or 0.0,
                'predicted_volatility': pair.predicted_volatility or pair.volatility_24h
            })
        
        return display_pairs
    
    async def shutdown(self):
        """Graceful shutdown of neural net"""
        try:
            logger.info("🛑 Shutting down Nexlify Neural Net...")
            
            self.running = False
            
            # Cancel tasks
            if self.scanner_task:
                self.scanner_task.cancel()
            if self.optimizer_task:
                self.optimizer_task.cancel()
            
            # Shutdown modules
            if self.strategy_optimizer:
                await self.strategy_optimizer.stop()
            
            # Close exchange connections
            for exchange_name, exchange in self.exchanges.items():
                try:
                    await exchange.close()
                except:
                    pass
            
            # Final audit log
            if self.audit_manager:
                await self.audit_manager.audit_system_event(
                    event_type="shutdown",
                    severity="info",
                    details={"total_trades": self.total_trades, "total_profit": self.total_profit_usdt}
                )
            
            logger.info("✅ Neural Net shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage
async def main():
    """Example usage of Nexlify Neural Net"""
    neural_net = NexlifyNeuralNet()
    
    try:
        # Initialize
        await neural_net.initialize()
        
        # Let it run for a while
        await asyncio.sleep(300)  # 5 minutes
        
        # Get active pairs for display
        active_pairs = neural_net.get_active_pairs_display()
        for pair in active_pairs[:5]:
            print(f"{pair['symbol']}: {pair['price']} ({pair['change_24h']})")
        
    finally:
        await neural_net.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
