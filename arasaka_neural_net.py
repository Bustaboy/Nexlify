#!/usr/bin/env python3
"""
Nexlify - Arasaka Neural-Net Trading Matrix
Main autonomous trading engine with AI-driven pair selection
Cyberpunk-themed cryptocurrency arbitrage system
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from decimal import Decimal
import ccxt.async_support as ccxt
import aiohttp
from dataclasses import dataclass
import json

# Load environment settings from config
config_path = "config/neural_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        full_config = json.load(f)
        env_config = full_config.get('environment', {})
else:
    env_config = {}

# Configure cyber-logging based on config
log_level = getattr(logging, env_config.get('log_level', 'INFO'))
logging.basicConfig(
    level=log_level,
    format='[%(asctime)s] [NEURAL-NET] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/neural_net.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Apply debug mode
if env_config.get('debug', False):
    logger.setLevel(logging.DEBUG)
    logger.debug("🐛 Debug mode enabled")

@dataclass
class CyberPair:
    """Cyberpunk trading pair data structure"""
    symbol: str
    base: str
    quote: str
    profit_score: float
    volume_24h: float
    volatility: float
    neural_confidence: float
    exchanges: List[str]
    last_scan: datetime

class ArasakaNeuralNet:
    """
    Main AI-driven trading engine with autonomous pair selection
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.exchanges = {}
        self.active_pairs = {}
        self.neural_memory = {}
        self.btc_wallet = config.get('btc_wallet_address', '')
        self.is_active = True
        
        # Fee configurations per exchange
        self.exchange_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001, 'withdrawal': 0.0005},
            'kraken': {'maker': 0.0016, 'taker': 0.0026, 'withdrawal': 0.0005},
            'coinbase': {'maker': 0.005, 'taker': 0.005, 'withdrawal': 0.0006}
        }
        
        # Gas fee estimator (for ETH/BSC/MATIC pairs)
        self.gas_fees = {
            'ETH': 0.003,  # ~$10 at current prices
            'BSC': 0.0001,  # ~$0.30
            'MATIC': 0.00001  # ~$0.01
        }
        
    async def initialize(self):
        """Initialize the Neural-Net and connect to exchanges"""
        logger.info("🧠 Initializing Arasaka Neural-Net Trading Matrix...")
        
        # Load configuration from file
        config_path = "config/neural_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                self.config.update(file_config)
        
        # Initialize exchanges
        exchanges_config = self.config.get('exchanges', {})
        if not exchanges_config:
            logger.warning("⚠️ No exchanges configured! Please use the GUI to add exchange API keys.")
            return
        
        for exchange_id, api_config in exchanges_config.items():
            try:
                # Skip if no API key
                if not api_config.get('api_key') or api_config['api_key'] == 'YOUR_API_KEY_HERE':
                    logger.warning(f"⏭️ Skipping {exchange_id} - no API key configured")
                    continue
                
                exchange_class = getattr(ccxt, exchange_id)
                exchange_config = {
                    'apiKey': api_config['api_key'],
                    'secret': api_config['secret'],
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
                
                # Use testnet if configured
                if api_config.get('testnet', True):
                    if exchange_id == 'binance':
                        exchange_config['urls'] = {
                            'api': {
                                'public': 'https://testnet.binance.vision/api',
                                'private': 'https://testnet.binance.vision/api',
                            }
                        }
                    logger.info(f"🧪 Using {exchange_id} TESTNET")
                
                self.exchanges[exchange_id] = exchange_class(exchange_config)
                await self.exchanges[exchange_id].load_markets()
                logger.info(f"✅ Connected to {exchange_id.upper()} exchange")
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to {exchange_id}: {e}")
        
        if not self.exchanges:
            logger.error("❌ No exchanges connected! Please check your API configuration.")
            return
        
        # Update BTC wallet from config
        self.btc_wallet = self.config.get('btc_wallet_address', '') or self.config.get('btc_wallet', '')
        
        # Check for emergency notifications
        env_config = self.config.get('environment', {})
        if env_config.get('emergency_contact'):
            logger.info(f"📧 Emergency notifications enabled: {env_config['emergency_contact'][:5]}...")
        if env_config.get('telegram_bot_token'):
            logger.info("🤖 Telegram bot integration enabled")
        
        # Start autonomous systems
        asyncio.create_task(self.neural_scanner())
        asyncio.create_task(self.profit_optimizer())
        logger.info("🚀 Neural-Net fully operational - Welcome to Nexlify")
    
    async def neural_scanner(self):
        """Continuously scan and rank trading pairs using AI logic"""
        while self.is_active:
            try:
                logger.info("🔍 Neural scan initiated...")
                all_pairs = await self.discover_all_pairs()
                ranked_pairs = await self.rank_pairs_by_profit(all_pairs)
                
                # Update active pairs (keep top 20)
                new_active = ranked_pairs[:20]
                
                # Log changes
                for pair in new_active:
                    if pair.symbol not in self.active_pairs:
                        logger.info(f"➕ Adding profitable pair: {pair.symbol} "
                                  f"(Score: {pair.profit_score:.2f}%)")
                
                for symbol in list(self.active_pairs.keys()):
                    if symbol not in [p.symbol for p in new_active]:
                        logger.info(f"➖ Removing underperforming pair: {symbol}")
                
                # Update active pairs
                self.active_pairs = {p.symbol: p for p in new_active}
                
                # Store in neural memory
                self.neural_memory['last_scan'] = datetime.now()
                self.neural_memory['pair_history'] = ranked_pairs
                
                await asyncio.sleep(300)  # Scan every 5 minutes
                
            except Exception as e:
                logger.error(f"Neural scan error: {e}")
                await asyncio.sleep(60)
    
    async def discover_all_pairs(self) -> List[CyberPair]:
        """Discover all available trading pairs across exchanges"""
        all_pairs = {}
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                markets = exchange.markets
                for symbol, market in markets.items():
                    if market['active'] and market['type'] == 'spot':
                        base = market['base']
                        quote = market['quote']
                        
                        # Filter for quality pairs
                        if quote in ['USDT', 'USDC', 'BUSD', 'USD', 'EUR']:
                            if symbol not in all_pairs:
                                all_pairs[symbol] = CyberPair(
                                    symbol=symbol,
                                    base=base,
                                    quote=quote,
                                    profit_score=0,
                                    volume_24h=0,
                                    volatility=0,
                                    neural_confidence=0,
                                    exchanges=[exchange_id],
                                    last_scan=datetime.now()
                                )
                            else:
                                all_pairs[symbol].exchanges.append(exchange_id)
            except Exception as e:
                logger.error(f"Error discovering pairs on {exchange_id}: {e}")
        
        return list(all_pairs.values())
    
    async def rank_pairs_by_profit(self, pairs: List[CyberPair]) -> List[CyberPair]:
        """AI-driven ranking of pairs by profit potential"""
        ranked_pairs = []
        
        for pair in pairs:
            try:
                # Multi-exchange arbitrage check
                if len(pair.exchanges) >= 2:
                    arb_profit = await self.calculate_arbitrage_profit(pair)
                    pair.profit_score = max(pair.profit_score, arb_profit)
                
                # Get market data
                market_data = await self.fetch_pair_metrics(pair)
                
                # Neural scoring algorithm
                volume_score = min(market_data['volume_24h'] / 1000000, 10)  # Cap at 10M
                volatility_score = market_data['volatility'] * 100  # Convert to percentage
                spread_score = (1 - market_data['spread']) * 10
                
                # Calculate neural confidence
                pair.neural_confidence = (
                    volume_score * 0.3 +
                    volatility_score * 0.3 +
                    spread_score * 0.2 +
                    pair.profit_score * 0.2
                ) / 10
                
                # Update metrics
                pair.volume_24h = market_data['volume_24h']
                pair.volatility = market_data['volatility']
                
                # Overall profit score (includes fees)
                pair.profit_score = self.calculate_net_profit(
                    pair.profit_score,
                    pair.symbol,
                    pair.exchanges[0]
                )
                
                if pair.profit_score > 0.1:  # Minimum 0.1% profit
                    ranked_pairs.append(pair)
                    
            except Exception as e:
                logger.debug(f"Error ranking {pair.symbol}: {e}")
        
        # Sort by profit potential
        ranked_pairs.sort(key=lambda x: x.profit_score * x.neural_confidence, reverse=True)
        return ranked_pairs
    
    async def calculate_arbitrage_profit(self, pair: CyberPair) -> float:
        """Calculate potential arbitrage profit between exchanges"""
        prices = {}
        
        for exchange_id in pair.exchanges[:3]:  # Check max 3 exchanges
            try:
                ticker = await self.exchanges[exchange_id].fetch_ticker(pair.symbol)
                prices[exchange_id] = {
                    'bid': ticker['bid'],
                    'ask': ticker['ask']
                }
            except:
                continue
        
        if len(prices) < 2:
            return 0
        
        # Find best arbitrage opportunity
        max_profit = 0
        for buy_ex, buy_price in prices.items():
            for sell_ex, sell_price in prices.items():
                if buy_ex != sell_ex and buy_price['ask'] and sell_price['bid']:
                    # Calculate gross profit
                    gross_profit = (sell_price['bid'] / buy_price['ask'] - 1) * 100
                    
                    # Subtract fees
                    buy_fee = self.exchange_fees[buy_ex]['taker'] * 100
                    sell_fee = self.exchange_fees[sell_ex]['taker'] * 100
                    
                    net_profit = gross_profit - buy_fee - sell_fee
                    max_profit = max(max_profit, net_profit)
        
        return max_profit
    
    def calculate_net_profit(self, gross_profit: float, symbol: str, exchange: str) -> float:
        """Calculate net profit after all fees"""
        # Exchange fees
        trade_fee = self.exchange_fees[exchange]['taker'] * 100 * 2  # Buy + Sell
        
        # Gas fees for blockchain-based assets
        gas_fee = 0
        if 'ETH' in symbol:
            gas_fee = self.gas_fees['ETH'] * 100
        elif 'BNB' in symbol or 'BSC' in symbol:
            gas_fee = self.gas_fees['BSC'] * 100
        elif 'MATIC' in symbol:
            gas_fee = self.gas_fees['MATIC'] * 100
        
        # Withdrawal fee (if profit taking)
        withdrawal_fee = self.exchange_fees[exchange]['withdrawal'] * 100
        
        # Calculate net profit
        net_profit = gross_profit - trade_fee - gas_fee - withdrawal_fee
        
        return max(0, net_profit)
    
    async def fetch_pair_metrics(self, pair: CyberPair) -> Dict:
        """Fetch detailed metrics for a trading pair"""
        metrics = {
            'volume_24h': 0,
            'volatility': 0,
            'spread': 0
        }
        
        try:
            # Use first available exchange
            exchange = self.exchanges[pair.exchanges[0]]
            
            # Get ticker data
            ticker = await exchange.fetch_ticker(pair.symbol)
            metrics['volume_24h'] = ticker.get('quoteVolume', 0)
            
            # Calculate spread
            if ticker['bid'] and ticker['ask']:
                metrics['spread'] = (ticker['ask'] - ticker['bid']) / ticker['ask']
            
            # Get OHLCV for volatility
            ohlcv = await exchange.fetch_ohlcv(pair.symbol, '1h', limit=24)
            if ohlcv:
                closes = [x[4] for x in ohlcv]
                returns = pd.Series(closes).pct_change().dropna()
                metrics['volatility'] = returns.std()
        
        except Exception as e:
            logger.debug(f"Error fetching metrics for {pair.symbol}: {e}")
        
        return metrics
    
    async def profit_optimizer(self):
        """Continuously optimize profits and execute withdrawal strategies"""
        while self.is_active:
            try:
                await asyncio.sleep(3600)  # Check hourly
                
                # Check total profits
                total_profits = await self.calculate_total_profits()
                
                if total_profits > self.config.get('min_withdrawal', 100):
                    logger.info(f"💰 Profit threshold reached: ${total_profits:.2f}")
                    
                    if self.btc_wallet:
                        # Execute BTC withdrawal
                        success = await self.withdraw_profits_to_btc(total_profits)
                        if success:
                            logger.info(f"✅ Profits withdrawn to BTC wallet: {self.btc_wallet}")
                    else:
                        logger.warning("⚠️ No BTC wallet configured for withdrawals")
                
            except Exception as e:
                logger.error(f"Profit optimizer error: {e}")
    
    async def calculate_total_profits(self) -> float:
        """Calculate total unrealized profits across all exchanges"""
        total = 0
        
        for exchange_id, exchange in self.exchanges.items():
            try:
                balance = await exchange.fetch_balance()
                
                # Sum up non-base currency balances
                for currency, amount in balance['total'].items():
                    if currency not in ['USD', 'USDT', 'USDC', 'BUSD'] and amount > 0:
                        # Convert to USD value
                        try:
                            ticker = await exchange.fetch_ticker(f"{currency}/USDT")
                            usd_value = amount * ticker['last']
                            total += usd_value
                        except:
                            pass
            except Exception as e:
                logger.error(f"Error calculating profits on {exchange_id}: {e}")
        
        return total
    
    async def withdraw_profits_to_btc(self, amount: float) -> bool:
        """Withdraw profits to configured BTC wallet"""
        try:
            # Find exchange with BTC
            for exchange_id, exchange in self.exchanges.items():
                balance = await exchange.fetch_balance()
                
                if balance['BTC']['free'] >= 0.001:  # Min withdrawal
                    # Execute withdrawal
                    result = await exchange.withdraw(
                        code='BTC',
                        amount=min(balance['BTC']['free'], amount / 40000),  # Assume BTC ~$40k
                        address=self.btc_wallet,
                        params={'network': 'BTC'}
                    )
                    return True
            
            # If no BTC, convert USDT to BTC first
            logger.info("No BTC balance found, attempting to convert USDT to BTC...")

            for exchange_id, exchange in self.exchanges.items():
                try:
                    balance = await exchange.fetch_balance()

                    # Check USDT balance
                    usdt_balance = balance.get('USDT', {}).get('free', 0)
                    if usdt_balance < 10:  # Need at least $10 USDT
                        continue

                    # Get BTC/USDT ticker for current price
                    ticker = await exchange.fetch_ticker('BTC/USDT')
                    btc_price = ticker['last']

                    # Calculate how much BTC we can buy
                    btc_to_buy = min(usdt_balance / btc_price, amount / btc_price)

                    # Need at least minimum withdrawal amount
                    if btc_to_buy < 0.001:
                        logger.warning(f"Calculated BTC amount too small: {btc_to_buy}")
                        continue

                    # Place market buy order for BTC
                    logger.info(f"Converting {usdt_balance:.2f} USDT to BTC at ${btc_price:.2f}")
                    order = await exchange.create_market_buy_order('BTC/USDT', btc_to_buy)

                    if order and order.get('status') in ['closed', 'filled']:
                        logger.info(f"✅ Successfully converted to {btc_to_buy:.6f} BTC")

                        # Wait a moment for balance to update
                        await asyncio.sleep(2)

                        # Now withdraw BTC
                        updated_balance = await exchange.fetch_balance()
                        btc_available = updated_balance.get('BTC', {}).get('free', 0)

                        if btc_available >= 0.001:
                            result = await exchange.withdraw(
                                code='BTC',
                                amount=btc_available,
                                address=self.btc_wallet,
                                params={'network': 'BTC'}
                            )

                            if result:
                                logger.info(f"✅ Withdrawal initiated: {btc_available:.6f} BTC to {self.btc_wallet[:8]}...")
                                return True
                        else:
                            logger.warning("BTC balance still insufficient after conversion")

                except Exception as conv_error:
                    logger.error(f"Conversion error on {exchange_id}: {conv_error}")
                    continue

            logger.warning("Could not complete BTC conversion on any exchange")
            return False

        except Exception as e:
            logger.error(f"Withdrawal error: {e}")
            return False
    
    def get_active_pairs_display(self) -> List[Dict]:
        """Get active pairs formatted for display"""
        display_pairs = []
        
        for symbol, pair in self.active_pairs.items():
            display_pairs.append({
                'symbol': symbol,
                'profit_score': f"{pair.profit_score:.2f}%",
                'volume': f"${pair.volume_24h/1000000:.1f}M",
                'volatility': f"{pair.volatility*100:.1f}%",
                'confidence': f"{pair.neural_confidence*100:.0f}%",
                'exchanges': ', '.join(pair.exchanges),
                'status': '🟢 ACTIVE' if pair.profit_score > 0.5 else '🟡 MONITORING'
            })
        
        return display_pairs
    
    async def shutdown(self):
        """Gracefully shutdown the Neural-Net"""
        logger.info("🔌 Shutting down Arasaka Neural-Net...")
        self.is_active = False
        
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        logger.info("👋 Neural-Net offline. Stay safe in the Matrix!")

# Usage example
if __name__ == "__main__":
    # Load config from file or use defaults
    config_path = "config/neural_config.json"
    config = {}
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config if no file exists
        config = {
            'exchanges': {},
            'btc_wallet_address': '',
            'min_withdrawal': 100,
            'environment': {
                'debug': False,
                'log_level': 'INFO',
                'api_port': 8000,
                'database_url': 'sqlite:///data/trading.db'
            }
        }
        logger.warning("No config file found. Please use the GUI to configure exchanges.")
    
    neural_net = ArasakaNeuralNet(config)
    
    # Run the neural net
    try:
        asyncio.run(neural_net.initialize())
        # Keep running
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Neural-Net...")
        asyncio.run(neural_net.shutdown())