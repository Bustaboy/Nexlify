"""
Nexlify Neural Net Trading Engine - Enhanced Core Trading System
Orchestrates multi-exchange trading with ML predictions and risk management
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from decimal import Decimal, ROUND_DOWN
import traceback
from pathlib import Path

import ccxt
import ccxt.async_support as ccxt_async
import numpy as np
import pandas as pd

# Import our enhanced modules
from utils_module import (
    FileUtils, NetworkUtils, ValidationUtils, CryptoUtils,
    TimeUtils, MathUtils, AsyncUtils, DataUtils, get_error_handler
)
from error_handler import ErrorContext
from nexlify_advanced_security import SecurityManager, APIKeyRotation
from nexlify_audit_trail import AuditManager

# Optional module imports
try:
    from nexlify_predictive_features import PredictiveEngine
    HAS_PREDICTIVE = True
except ImportError:
    HAS_PREDICTIVE = False
    logging.warning("Predictive features not available")

try:
    from nexlify_multi_strategy import MultiStrategyOptimizer
    HAS_MULTI_STRATEGY = True
except ImportError:
    HAS_MULTI_STRATEGY = False
    logging.warning("Multi-strategy optimizer not available")


@dataclass
class TradingPair:
    """Enhanced trading pair with multi-exchange support"""
    symbol: str
    base: str
    quote: str
    exchanges: List[str] = field(default_factory=list)
    prices: Dict[str, float] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    spreads: Dict[str, float] = field(default_factory=dict)
    fees: Dict[str, float] = field(default_factory=dict)
    liquidity_scores: Dict[str, float] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid_exchange(self) -> Optional[str]:
        """Get exchange with best bid price"""
        if not self.prices:
            return None
        return max(self.prices.items(), key=lambda x: x[1])[0]
    
    @property
    def best_ask_exchange(self) -> Optional[str]:
        """Get exchange with best ask price"""
        if not self.prices:
            return None
        return min(self.prices.items(), key=lambda x: x[1])[0]
    
    @property
    def arbitrage_opportunity(self) -> float:
        """Calculate potential arbitrage profit percentage"""
        if len(self.prices) < 2:
            return 0.0
        
        prices = list(self.prices.values())
        min_price = min(prices)
        max_price = max(prices)
        
        # Account for fees
        buy_exchange = self.best_ask_exchange
        sell_exchange = self.best_bid_exchange
        
        if buy_exchange and sell_exchange:
            buy_fee = self.fees.get(buy_exchange, 0.001)
            sell_fee = self.fees.get(sell_exchange, 0.001)
            
            net_profit = (max_price * (1 - sell_fee)) - (min_price * (1 + buy_fee))
            profit_percent = (net_profit / min_price) * 100
            
            return max(0, profit_percent)
        
        return 0.0


@dataclass
class NeuralScore:
    """ML-based scoring for trading decisions"""
    symbol: str
    score: float  # 0-100
    components: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    signals: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class NexlifyNeuralNet:
    """Enhanced Neural Trading Engine with cyberpunk aesthetics"""
    
    def __init__(self, config_path: str = "config/enhanced_config.json"):
        """Initialize the neural trading engine"""
        self.logger = self._setup_logger()
        self.config = self._load_config(config_path)
        
        # Core components
        self.error_handler = get_error_handler()
        self.security_manager = SecurityManager(self.config)
        self.audit_manager = AuditManager()
        self.api_key_rotation = APIKeyRotation()
        
        # Optional components
        self.predictive_engine = PredictiveEngine() if HAS_PREDICTIVE else None
        self.strategy_optimizer = MultiStrategyOptimizer() if HAS_MULTI_STRATEGY else None
        
        # Exchange management
        self.exchanges: Dict[str, ccxt_async.Exchange] = {}
        self.exchange_info: Dict[str, Dict[str, Any]] = {}
        self.active_pairs: Dict[str, TradingPair] = {}
        
        # Trading state
        self.running = False
        self.neural_memory = {
            'pair_history': defaultdict(lambda: deque(maxlen=1000)),
            'trade_performance': defaultdict(float),
            'error_counts': defaultdict(int),
            'last_scan': datetime.now()
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'total_volume': 0.0,
            'start_time': datetime.now()
        }
        
        # Risk management
        self.risk_params = self._load_risk_params()
        self.daily_loss = 0.0
        self.daily_loss_reset = datetime.now().date()
        
        # Tasks
        self.scanner_task = None
        self.optimizer_task = None
        self.health_check_task = None
        
        self.logger.info("🧠 Nexlify Neural Net initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup enhanced logger"""
        logger = logging.getLogger("NexlifyNeuralNet")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        config = FileUtils.safe_json_load(config_path, {})
        
        # Set defaults if missing
        defaults = {
            'trading': {
                'scan_interval': 300,  # 5 minutes
                'min_profit_threshold': 0.1,  # 0.1%
                'max_pairs': 20,
                'enable_paper_trading': True
            },
            'exchanges': {
                'enabled': ['binance', 'kraken', 'coinbase'],
                'testnet': True
            },
            'risk': {
                'max_position_size': 0.1,
                'max_daily_loss': 0.02,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 5.0
            },
            'withdrawal': {
                'enabled': False,
                'min_amount': 100,
                'wallet_address': '',
                'auto_withdraw_threshold': 1000
            }
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in config[key]:
                        config[key][subkey] = subvalue
        
        return config
    
    def _load_risk_params(self) -> Dict[str, float]:
        """Load risk management parameters"""
        risk_config = self.config.get('risk', {})
        return {
            'max_position_size': risk_config.get('max_position_size', 0.1),
            'max_daily_loss': risk_config.get('max_daily_loss', 0.02),
            'stop_loss': risk_config.get('stop_loss_percent', 2.0) / 100,
            'take_profit': risk_config.get('take_profit_percent', 5.0) / 100,
            'max_trades_per_day': risk_config.get('max_trades_per_day', 50),
            'risk_per_trade': risk_config.get('risk_per_trade', 0.01)
        }
    
    async def initialize(self) -> bool:
        """Initialize exchanges and validate configuration"""
        self.logger.info("🚀 Initializing Neural Net...")
        
        try:
            # Validate security
            if not await self._validate_security():
                return False
            
            # Initialize exchanges
            await self._initialize_exchanges()
            
            # Load market data
            await self._load_markets()
            
            # Initialize optional components
            if self.predictive_engine:
                self.logger.info("Initializing predictive engine...")
                # Initialize predictive models
            
            if self.strategy_optimizer:
                self.logger.info("Initializing strategy optimizer...")
                await self.strategy_optimizer.initialize()
            
            # Audit initialization
            self.audit_manager.audit_system_event(
                "system_start",
                {"component": "NexlifyNeuralNet", "version": "2.0.8"}
            )
            
            self.logger.info("✅ Neural Net initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.error_handler.log_error(e, {"component": "initialization"})
            return False
    
    async def _validate_security(self) -> bool:
        """Validate security configuration"""
        try:
            # Check if master password is required
            if self.config.get('security', {}).get('master_password_required', False):
                if not self.security_manager.is_authenticated():
                    self.logger.error("Master password authentication required")
                    return False
            
            # Validate API keys
            for exchange in self.config['exchanges']['enabled']:
                api_key = self.config.get('credentials', {}).get(exchange, {}).get('api_key')
                api_secret = self.config.get('credentials', {}).get(exchange, {}).get('api_secret')
                
                if not api_key or not api_secret:
                    self.logger.warning(f"No credentials for {exchange}")
                    continue
                
                # Use enhanced validation
                result = ValidationUtils.validate_api_credentials(api_key, api_secret, exchange)
                if not result['valid']:
                    self.logger.error(f"{exchange} validation failed: {result['errors']}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Security validation failed: {e}")
            return False
    
    async def _initialize_exchanges(self):
        """Initialize exchange connections with rate limiting"""
        enabled_exchanges = self.config['exchanges']['enabled']
        testnet = self.config['exchanges']['testnet']
        
        for exchange_name in enabled_exchanges:
            try:
                # Get credentials (possibly rotated)
                creds = await self._get_exchange_credentials(exchange_name)
                if not creds:
                    continue
                
                # Create exchange instance
                exchange_class = getattr(ccxt_async, exchange_name)
                
                exchange_config = {
                    'apiKey': creds['api_key'],
                    'secret': creds['api_secret'],
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                }
                
                # Set testnet if enabled
                if testnet and exchange_name == 'binance':
                    exchange_config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api',
                            'private': 'https://testnet.binance.vision/api'
                        }
                    }
                
                exchange = exchange_class(exchange_config)
                
                # Test connection and validate permissions
                await self._validate_exchange_permissions(exchange, exchange_name)
                
                self.exchanges[exchange_name] = exchange
                
                # Store exchange info
                self.exchange_info[exchange_name] = {
                    'name': exchange_name,
                    'has_trading': True,
                    'has_withdrawal': False,  # Check permissions
                    'rate_limit': NetworkUtils.detect_exchange_rate_limit(exchange_name),
                    'fees': await self._get_exchange_fees(exchange)
                }
                
                self.logger.info(f"✅ Connected to {exchange_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {e}")
                self.error_handler.log_error(e, {"exchange": exchange_name})
    
    async def _get_exchange_credentials(self, exchange: str) -> Optional[Dict[str, str]]:
        """Get exchange credentials with rotation support"""
        try:
            # Check for rotated keys first
            rotated_keys = self.api_key_rotation.get_active_keys(exchange)
            if rotated_keys:
                return rotated_keys
            
            # Fall back to config
            creds = self.config.get('credentials', {}).get(exchange, {})
            if creds.get('api_key') and creds.get('api_secret'):
                return creds
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get credentials for {exchange}: {e}")
            return None
    
    async def _validate_exchange_permissions(self, exchange: ccxt_async.Exchange, name: str):
        """Validate exchange API permissions"""
        try:
            # Test balance endpoint (requires trading permission)
            balance = await exchange.fetch_balance()
            
            # Check if we can fetch orders (trading permission)
            try:
                await exchange.fetch_open_orders(limit=1)
                self.exchange_info[name] = self.exchange_info.get(name, {})
                self.exchange_info[name]['has_trading'] = True
            except:
                self.logger.warning(f"{name}: No trading permission")
            
            # Log successful validation
            self.audit_manager.audit_system_event(
                "exchange_connected",
                {"exchange": name, "testnet": self.config['exchanges']['testnet']}
            )
            
        except Exception as e:
            self.logger.error(f"Permission validation failed for {name}: {e}")
            raise
    
    async def _get_exchange_fees(self, exchange: ccxt_async.Exchange) -> Dict[str, float]:
        """Get dynamic exchange fees"""
        try:
            # Try to get trading fees
            if hasattr(exchange, 'fetch_trading_fees'):
                fees = await exchange.fetch_trading_fees()
                # Extract maker/taker fees
                return {
                    'maker': fees.get('maker', 0.001),
                    'taker': fees.get('taker', 0.001)
                }
        except:
            pass
        
        # Fall back to defaults
        fee_defaults = {
            'binance': {'maker': 0.001, 'taker': 0.001},
            'kraken': {'maker': 0.0016, 'taker': 0.0026},
            'coinbase': {'maker': 0.005, 'taker': 0.005}
        }
        
        return fee_defaults.get(exchange.id, {'maker': 0.001, 'taker': 0.001})
    
    async def _load_markets(self):
        """Load market data from all exchanges"""
        for name, exchange in self.exchanges.items():
            try:
                await exchange.load_markets()
                self.logger.info(f"Loaded {len(exchange.symbols)} markets from {name}")
            except Exception as e:
                self.logger.error(f"Failed to load markets from {name}: {e}")
    
    async def start(self):
        """Start the neural trading engine"""
        if self.running:
            self.logger.warning("Neural Net already running")
            return
        
        self.running = True
        self.logger.info("🎯 Starting Neural Net trading engine...")
        
        # Start background tasks
        self.scanner_task = asyncio.create_task(self.neural_scanner())
        self.optimizer_task = asyncio.create_task(self.profit_optimizer())
        self.health_check_task = asyncio.create_task(self.health_monitor())
        
        # Start strategy optimizer if available
        if self.strategy_optimizer:
            await self.strategy_optimizer.start()
        
        self.logger.info("✅ Neural Net is active")
    
    async def stop(self):
        """Stop the neural trading engine"""
        self.logger.info("🛑 Stopping Neural Net...")
        self.running = False
        
        # Cancel tasks
        tasks = [self.scanner_task, self.optimizer_task, self.health_check_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        # Stop strategy optimizer
        if self.strategy_optimizer:
            await self.strategy_optimizer.stop()
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        # Final audit
        self.audit_manager.audit_system_event(
            "system_stop",
            {"component": "NexlifyNeuralNet", "performance": self.performance_metrics}
        )
        
        self.logger.info("✅ Neural Net stopped")
    
    async def neural_scanner(self):
        """Main scanning loop for finding trading opportunities"""
        scan_interval = self.config['trading']['scan_interval']
        
        while self.running:
            try:
                with ErrorContext("neural_scanner", self.error_handler):
                    self.logger.info("🔍 Scanning for opportunities...")
                    
                    # Discover trading pairs
                    all_pairs = await self.discover_all_pairs()
                    
                    # Fetch metrics for each pair
                    await self.fetch_pair_metrics(all_pairs)
                    
                    # Apply ML scoring
                    scored_pairs = await self.rank_pairs_by_profit(all_pairs)
                    
                    # Update active pairs
                    self.active_pairs = {
                        pair.symbol: pair 
                        for pair in scored_pairs[:self.config['trading']['max_pairs']]
                    }
                    
                    # Execute trading strategies
                    if self.strategy_optimizer:
                        await self.strategy_optimizer.execute_all_strategies(self.active_pairs)
                    else:
                        # Simple arbitrage execution
                        await self.execute_arbitrage_opportunities()
                    
                    self.neural_memory['last_scan'] = datetime.now()
                    
                    # Audit scan
                    self.audit_manager.audit_system_event(
                        "market_scan",
                        {
                            "pairs_discovered": len(all_pairs),
                            "pairs_active": len(self.active_pairs),
                            "top_opportunity": scored_pairs[0].symbol if scored_pairs else None
                        }
                    )
                    
            except Exception as e:
                self.logger.error(f"Scanner error: {e}")
                self.error_handler.log_error(e, {"component": "neural_scanner"})
            
            await asyncio.sleep(scan_interval)
    
    async def discover_all_pairs(self) -> List[TradingPair]:
        """Discover all tradeable pairs across exchanges"""
        pairs_map: Dict[str, TradingPair] = {}
        
        # Quote currencies to focus on
        target_quotes = ['USDT', 'USDC', 'BUSD', 'USD', 'EUR', 'BTC', 'ETH']
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Get all symbols
                for symbol in exchange.symbols:
                    # Parse symbol
                    if '/' not in symbol:
                        continue
                    
                    base, quote = symbol.split('/')
                    
                    # Filter by quote currency
                    if quote not in target_quotes:
                        continue
                    
                    # Skip derivatives
                    if any(x in symbol for x in ['-PERP', 'SWAP', 'FUTURE']):
                        continue
                    
                    # Normalize symbol
                    normalized = CryptoUtils.normalize_symbol(symbol)
                    
                    # Create or update pair
                    if normalized not in pairs_map:
                        pairs_map[normalized] = TradingPair(
                            symbol=normalized,
                            base=base,
                            quote=quote
                        )
                    
                    pairs_map[normalized].exchanges.append(exchange_name)
                    
            except Exception as e:
                self.logger.error(f"Error discovering pairs on {exchange_name}: {e}")
        
        # Filter pairs available on multiple exchanges for arbitrage
        multi_exchange_pairs = [
            pair for pair in pairs_map.values() 
            if len(pair.exchanges) >= 2
        ]
        
        self.logger.info(f"Discovered {len(multi_exchange_pairs)} multi-exchange pairs")
        return multi_exchange_pairs
    
    async def fetch_pair_metrics(self, pairs: List[TradingPair]):
        """Fetch current metrics for all pairs"""
        tasks = []
        
        for pair in pairs:
            for exchange_name in pair.exchanges:
                task = self._fetch_single_pair_metrics(pair, exchange_name)
                tasks.append(task)
        
        # Limit concurrent requests
        results = await AsyncUtils.gather_with_limit(tasks, limit=20)
        
        # Process results
        for result in results:
            if result and not isinstance(result, Exception):
                pair, exchange_name, metrics = result
                
                # Update pair data
                pair.prices[exchange_name] = metrics.get('price', 0)
                pair.volumes[exchange_name] = metrics.get('volume', 0)
                pair.spreads[exchange_name] = metrics.get('spread', 0)
                pair.fees[exchange_name] = self.exchange_info[exchange_name]['fees']['taker']
                pair.liquidity_scores[exchange_name] = metrics.get('liquidity_score', 0)
                pair.last_update = datetime.now()
    
    async def _fetch_single_pair_metrics(self, pair: TradingPair, exchange_name: str) -> Optional[Tuple]:
        """Fetch metrics for a single pair on one exchange"""
        try:
            exchange = self.exchanges[exchange_name]
            
            # Rate-limited fetch
            ticker = await NetworkUtils.async_rate_limited_request(
                exchange.fetch_ticker,
                pair.symbol,
                exchange=exchange_name
            )
            
            # Calculate metrics
            metrics = {
                'price': ticker.get('last', 0),
                'volume': ticker.get('quoteVolume', 0),
                'spread': abs(ticker.get('ask', 0) - ticker.get('bid', 0)),
                'liquidity_score': self._calculate_liquidity_score(ticker)
            }
            
            # Get enhanced metrics if predictive engine available
            if self.predictive_engine and ticker.get('last'):
                # Add volatility prediction
                try:
                    volatility = await self.predictive_engine.predict_volatility(
                        pair.symbol,
                        exchange_name
                    )
                    metrics['predicted_volatility'] = volatility
                except:
                    pass
            
            return (pair, exchange_name, metrics)
            
        except Exception as e:
            self.logger.debug(f"Failed to fetch {pair.symbol} from {exchange_name}: {e}")
            return None
    
    def _calculate_liquidity_score(self, ticker: Dict[str, Any]) -> float:
        """Calculate liquidity score based on volume and spread"""
        volume = ticker.get('quoteVolume', 0)
        spread = abs(ticker.get('ask', 0) - ticker.get('bid', 0))
        price = ticker.get('last', 1)
        
        if price <= 0 or volume <= 0:
            return 0.0
        
        # Spread percentage
        spread_pct = (spread / price) * 100 if price > 0 else 100
        
        # Liquidity score (higher volume, lower spread = better liquidity)
        if spread_pct > 0:
            score = (volume / 1000) / spread_pct  # Normalize volume to thousands
        else:
            score = volume / 1000
        
        return min(100, score)  # Cap at 100
    
    async def rank_pairs_by_profit(self, pairs: List[TradingPair]) -> List[TradingPair]:
        """Rank pairs using ML-based scoring"""
        scored_pairs = []
        
        for pair in pairs:
            # Skip if insufficient data
            if len(pair.prices) < 2:
                continue
            
            # Calculate base score
            score = self._calculate_neural_score(pair)
            
            # Store in memory
            self.neural_memory['pair_history'][pair.symbol].append({
                'score': score.score,
                'timestamp': datetime.now(),
                'arbitrage': pair.arbitrage_opportunity
            })
            
            # Add to scored pairs
            scored_pairs.append((score.score, pair))
        
        # Sort by score descending
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        return [pair for _, pair in scored_pairs]
    
    def _calculate_neural_score(self, pair: TradingPair) -> NeuralScore:
        """Calculate ML-based score for a trading pair"""
        score = NeuralScore(symbol=pair.symbol)
        
        # Component weights (adaptive based on market conditions)
        weights = {
            'arbitrage': 0.35,
            'volume': 0.25,
            'liquidity': 0.20,
            'volatility': 0.10,
            'trend': 0.10
        }
        
        # 1. Arbitrage opportunity
        arb_score = min(100, pair.arbitrage_opportunity * 20)  # Scale to 0-100
        score.components['arbitrage'] = arb_score
        
        # 2. Volume score
        avg_volume = np.mean(list(pair.volumes.values()))
        volume_score = min(100, (avg_volume / 1_000_000) * 10)  # Scale by millions
        score.components['volume'] = volume_score
        
        # 3. Liquidity score
        avg_liquidity = np.mean(list(pair.liquidity_scores.values()))
        score.components['liquidity'] = avg_liquidity
        
        # 4. Volatility score (if available)
        if self.predictive_engine:
            # Use predicted volatility
            volatility_score = 50  # Default medium volatility
        else:
            # Calculate from price history
            prices = list(pair.prices.values())
            if len(prices) > 1:
                volatility = np.std(prices) / np.mean(prices) * 100
                volatility_score = min(100, volatility * 10)
            else:
                volatility_score = 50
        
        score.components['volatility'] = volatility_score
        
        # 5. Trend score from history
        history = self.neural_memory['pair_history'][pair.symbol]
        if len(history) >= 5:
            recent_scores = [h['score'] for h in list(history)[-5:]]
            trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            trend_score = 50 + (trend * 10)  # Center at 50
            trend_score = max(0, min(100, trend_score))
        else:
            trend_score = 50
        
        score.components['trend'] = trend_score
        
        # Calculate weighted score
        total_score = sum(
            score.components[component] * weight 
            for component, weight in weights.items()
        )
        
        score.score = total_score
        score.confidence = self._calculate_confidence(pair)
        
        # Generate signals
        if arb_score > 50:
            score.signals.append("HIGH_ARBITRAGE")
        if volume_score > 70:
            score.signals.append("HIGH_VOLUME")
        if avg_liquidity > 80:
            score.signals.append("HIGH_LIQUIDITY")
        
        return score
    
    def _calculate_confidence(self, pair: TradingPair) -> float:
        """Calculate confidence in the scoring"""
        factors = []
        
        # Data freshness
        age = (datetime.now() - pair.last_update).total_seconds()
        freshness = max(0, 1 - (age / 300))  # 5 minute decay
        factors.append(freshness)
        
        # Data completeness
        expected_exchanges = len(pair.exchanges)
        actual_prices = len(pair.prices)
        completeness = actual_prices / expected_exchanges if expected_exchanges > 0 else 0
        factors.append(completeness)
        
        # Historical consistency
        history = self.neural_memory['pair_history'][pair.symbol]
        if len(history) >= 10:
            scores = [h['score'] for h in list(history)[-10:]]
            consistency = 1 - (np.std(scores) / 100)  # Lower variance = higher confidence
            factors.append(consistency)
        
        return np.mean(factors) if factors else 0.5
    
    async def execute_arbitrage_opportunities(self):
        """Execute arbitrage trades on best opportunities"""
        min_profit = self.config['trading']['min_profit_threshold']
        
        for symbol, pair in self.active_pairs.items():
            try:
                # Check arbitrage opportunity
                if pair.arbitrage_opportunity < min_profit:
                    continue
                
                # Check risk limits
                if not self._check_risk_limits():
                    self.logger.warning("Risk limits exceeded, skipping trades")
                    break
                
                # Calculate optimal trade size
                trade_size = await self._calculate_trade_size(pair)
                if trade_size <= 0:
                    continue
                
                # Execute arbitrage
                result = await self._execute_arbitrage_trade(pair, trade_size)
                
                if result['success']:
                    # Update performance
                    self.performance_metrics['total_trades'] += 1
                    if result['profit'] > 0:
                        self.performance_metrics['winning_trades'] += 1
                    self.performance_metrics['total_profit'] += result['profit']
                    self.performance_metrics['total_volume'] += result['volume']
                    
                    # Audit trade
                    self.audit_manager.audit_trade({
                        'symbol': symbol,
                        'type': 'arbitrage',
                        'size': trade_size,
                        'profit': result['profit'],
                        'buy_exchange': result['buy_exchange'],
                        'sell_exchange': result['sell_exchange']
                    })
                    
            except Exception as e:
                self.logger.error(f"Arbitrage execution error for {symbol}: {e}")
                self.error_handler.log_error(e, {"symbol": symbol, "component": "arbitrage"})
    
    def _check_risk_limits(self) -> bool:
        """Check if we're within risk limits"""
        # Reset daily loss if new day
        if datetime.now().date() > self.daily_loss_reset:
            self.daily_loss = 0.0
            self.daily_loss_reset = datetime.now().date()
        
        # Check daily loss limit
        if abs(self.daily_loss) >= self.risk_params['max_daily_loss']:
            return False
        
        # Check trade count
        if self.performance_metrics['total_trades'] >= self.risk_params.get('max_trades_per_day', 50):
            return False
        
        return True
    
    async def _calculate_trade_size(self, pair: TradingPair) -> float:
        """Calculate optimal trade size based on risk management"""
        # Get account balances
        total_balance = 0
        available_balance = {}
        
        for exchange_name in pair.exchanges:
            try:
                exchange = self.exchanges[exchange_name]
                balance = await exchange.fetch_balance()
                
                # Get quote currency balance
                quote_balance = balance.get(pair.quote, {}).get('free', 0)
                available_balance[exchange_name] = quote_balance
                total_balance += quote_balance
                
            except Exception as e:
                self.logger.error(f"Failed to fetch balance from {exchange_name}: {e}")
        
        if total_balance <= 0:
            return 0
        
        # Calculate position size based on risk
        risk_amount = total_balance * self.risk_params['risk_per_trade']
        
        # Consider exchange minimums
        min_trade_size = max([
            self.exchange_info[ex].get('min_trade_size', 10)
            for ex in pair.exchanges
        ])
        
        # Calculate based on available balance and risk
        max_size = total_balance * self.risk_params['max_position_size']
        trade_size = min(risk_amount, max_size)
        
        # Ensure minimum size
        if trade_size < min_trade_size:
            return 0
        
        return trade_size
    
    async def _execute_arbitrage_trade(self, pair: TradingPair, size: float) -> Dict[str, Any]:
        """Execute an arbitrage trade"""
        result = {
            'success': False,
            'profit': 0,
            'volume': 0,
            'buy_exchange': None,
            'sell_exchange': None
        }
        
        try:
            # Identify best exchanges
            buy_exchange = pair.best_ask_exchange
            sell_exchange = pair.best_bid_exchange
            
            if not buy_exchange or not sell_exchange or buy_exchange == sell_exchange:
                return result
            
            buy_price = pair.prices[buy_exchange]
            sell_price = pair.prices[sell_exchange]
            
            # Calculate amount in base currency
            amount = size / buy_price
            
            # Execute buy order
            buy_order = await self._place_order(
                buy_exchange,
                pair.symbol,
                'buy',
                amount,
                buy_price
            )
            
            if not buy_order:
                return result
            
            # Execute sell order
            sell_order = await self._place_order(
                sell_exchange,
                pair.symbol,
                'sell',
                amount,
                sell_price
            )
            
            if not sell_order:
                # Cancel buy order if sell fails
                await self._cancel_order(buy_exchange, buy_order['id'], pair.symbol)
                return result
            
            # Calculate profit
            buy_cost = buy_order['cost'] * (1 + pair.fees[buy_exchange])
            sell_revenue = sell_order['cost'] * (1 - pair.fees[sell_exchange])
            profit = sell_revenue - buy_cost
            
            result.update({
                'success': True,
                'profit': profit,
                'volume': size,
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange
            })
            
            # Update daily P&L
            self.daily_loss -= profit  # Negative because profit reduces loss
            
            self.logger.info(
                f"✅ Arbitrage executed: {pair.symbol} "
                f"Buy@{buy_exchange} ${buy_price:.2f} -> "
                f"Sell@{sell_exchange} ${sell_price:.2f} "
                f"Profit: ${profit:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Arbitrage execution failed: {e}")
            self.error_handler.log_error(e, {"pair": pair.symbol})
        
        return result
    
    async def _place_order(self, exchange_name: str, symbol: str, 
                          side: str, amount: float, price: float) -> Optional[Dict]:
        """Place an order with error handling"""
        try:
            exchange = self.exchanges[exchange_name]
            
            # Use market orders for faster execution
            order = await NetworkUtils.async_rate_limited_request(
                exchange.create_order,
                symbol,
                'market',
                side,
                amount,
                exchange=exchange_name
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Order placement failed on {exchange_name}: {e}")
            return None
    
    async def _cancel_order(self, exchange_name: str, order_id: str, symbol: str):
        """Cancel an order"""
        try:
            exchange = self.exchanges[exchange_name]
            await exchange.cancel_order(order_id, symbol)
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
    
    async def profit_optimizer(self):
        """Optimize profits and handle withdrawals"""
        while self.running:
            try:
                # Check every hour
                await asyncio.sleep(3600)
                
                with ErrorContext("profit_optimizer", self.error_handler):
                    # Calculate total profits
                    total_profits = await self.calculate_total_profits()
                    
                    # Check withdrawal threshold
                    if self.config['withdrawal']['enabled']:
                        threshold = self.config['withdrawal']['auto_withdraw_threshold']
                        if total_profits >= threshold:
                            await self.withdraw_profits()
                    
                    # Optimize strategy allocations if available
                    if self.strategy_optimizer:
                        await self.strategy_optimizer.optimize_allocation()
                    
                    # Log performance
                    self.logger.info(
                        f"💰 Performance Update: "
                        f"Trades: {self.performance_metrics['total_trades']} "
                        f"Win Rate: {self._calculate_win_rate():.1f}% "
                        f"Total Profit: ${self.performance_metrics['total_profit']:.2f}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Profit optimizer error: {e}")
                self.error_handler.log_error(e, {"component": "profit_optimizer"})
    
    async def calculate_total_profits(self) -> float:
        """Calculate total profits across all exchanges"""
        total_profit_usd = 0
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                balance = await exchange.fetch_balance()
                
                # Get BTC price for conversion (with real-time fetch)
                btc_price = await self._get_btc_price(exchange)
                
                # Calculate USD value of all non-stablecoin holdings
                for currency, amount in balance['total'].items():
                    if amount <= 0:
                        continue
                    
                    # Skip stablecoins
                    if currency in ['USDT', 'USDC', 'BUSD', 'USD', 'EUR']:
                        total_profit_usd += amount
                        continue
                    
                    # Convert to USD
                    try:
                        if currency == 'BTC':
                            total_profit_usd += amount * btc_price
                        else:
                            # Get conversion rate
                            ticker = await exchange.fetch_ticker(f"{currency}/USDT")
                            price = ticker.get('last', 0)
                            total_profit_usd += amount * price
                    except:
                        pass
                        
            except Exception as e:
                self.logger.error(f"Failed to calculate profits for {exchange_name}: {e}")
        
        return total_profit_usd
    
    async def _get_btc_price(self, exchange: ccxt_async.Exchange) -> float:
        """Get current BTC price"""
        try:
            ticker = await exchange.fetch_ticker('BTC/USDT')
            return ticker.get('last', 40000)  # Fallback price
        except:
            # Try alternative pairs
            for pair in ['BTC/USD', 'BTC/USDC']:
                try:
                    ticker = await exchange.fetch_ticker(pair)
                    return ticker.get('last', 40000)
                except:
                    continue
            
            return 40000  # Fallback if all fail
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        total = self.performance_metrics['total_trades']
        if total == 0:
            return 0.0
        
        wins = self.performance_metrics['winning_trades']
        return (wins / total) * 100
    
    async def withdraw_profits(self):
        """Withdraw profits to configured wallet"""
        if not self.config['withdrawal']['enabled']:
            return
        
        wallet_address = self.config['withdrawal']['wallet_address']
        if not wallet_address:
            self.logger.error("No withdrawal wallet configured")
            return
        
        # Validate wallet address
        if not CryptoUtils.validate_address(wallet_address, 'BTC'):
            self.logger.error("Invalid BTC wallet address")
            return
        
        try:
            min_amount = self.config['withdrawal']['min_amount']
            
            # Find exchange with BTC balance
            for exchange_name, exchange in self.exchanges.items():
                balance = await exchange.fetch_balance()
                btc_balance = balance.get('BTC', {}).get('free', 0)
                
                # Get current BTC price
                btc_price = await self._get_btc_price(exchange)
                btc_value_usd = btc_balance * btc_price
                
                if btc_value_usd >= min_amount:
                    # Calculate withdrawal amount
                    withdrawal_amount = btc_balance * 0.9  # Keep 10% as buffer
                    
                    # Execute withdrawal
                    result = await exchange.withdraw(
                        'BTC',
                        withdrawal_amount,
                        wallet_address,
                        tag=None,
                        params={}
                    )
                    
                    self.logger.info(
                        f"💸 Withdrawn {withdrawal_amount:.6f} BTC "
                        f"(~${withdrawal_amount * btc_price:.2f}) to {wallet_address}"
                    )
                    
                    # Audit withdrawal
                    self.audit_manager.audit_withdrawal({
                        'exchange': exchange_name,
                        'currency': 'BTC',
                        'amount': withdrawal_amount,
                        'address': wallet_address,
                        'value_usd': withdrawal_amount * btc_price
                    })
                    
                    break
                    
        except Exception as e:
            self.logger.error(f"Withdrawal failed: {e}")
            self.error_handler.log_error(e, {"component": "withdrawal"})
    
    async def health_monitor(self):
        """Monitor system health and exchange connections"""
        check_interval = 300  # 5 minutes
        
        while self.running:
            try:
                await asyncio.sleep(check_interval)
                
                health_status = {
                    'timestamp': datetime.now().isoformat(),
                    'exchanges': {},
                    'performance': self.performance_metrics,
                    'active_pairs': len(self.active_pairs),
                    'memory_usage': self._get_memory_usage()
                }
                
                # Check each exchange
                for name, exchange in self.exchanges.items():
                    try:
                        # Simple connectivity check
                        await asyncio.wait_for(
                            exchange.fetch_ticker('BTC/USDT'),
                            timeout=10
                        )
                        health_status['exchanges'][name] = 'online'
                    except:
                        health_status['exchanges'][name] = 'offline'
                        self.logger.warning(f"Exchange {name} appears offline")
                
                # Log health status
                self.logger.debug(f"Health check: {health_status}")
                
                # Save health status
                FileUtils.safe_json_save(
                    health_status,
                    "logs/health_status.json"
                )
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
    
    def _get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        import sys
        
        return {
            'pair_history': sum(
                sys.getsizeof(history) 
                for history in self.neural_memory['pair_history'].values()
            ),
            'active_pairs': sys.getsizeof(self.active_pairs),
            'total_objects': len(self.neural_memory['pair_history'])
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'running': self.running,
            'exchanges': list(self.exchanges.keys()),
            'active_pairs': len(self.active_pairs),
            'performance': self.performance_metrics,
            'last_scan': self.neural_memory['last_scan'].isoformat(),
            'daily_pnl': -self.daily_loss  # Convert back to profit
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        await self.stop()
        
        # Close all exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        # Save final state
        state = {
            'performance': self.performance_metrics,
            'neural_memory': {
                'pair_history': dict(self.neural_memory['pair_history']),
                'trade_performance': dict(self.neural_memory['trade_performance'])
            }
        }
        
        FileUtils.safe_json_save(state, "data/neural_state.json")
        
        self.logger.info("💤 Neural Net shutdown complete")


# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize neural net
        neural_net = NexlifyNeuralNet()
        
        # Initialize system
        if await neural_net.initialize():
            # Start trading
            await neural_net.start()
            
            # Run for some time
            await asyncio.sleep(3600)  # 1 hour
            
            # Shutdown
            await neural_net.shutdown()
        else:
            print("Failed to initialize Neural Net")
    
    # Run
    asyncio.run(main())
