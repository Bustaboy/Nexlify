#!/usr/bin/env python3
"""
Nexlify Auto-Execution Engine
Fully autonomous trading system with risk management
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class TradeExecution:
    """Represents an executed trade"""
    trade_id: str
    symbol: str
    exchange: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    timestamp: datetime
    profit_target: float
    stop_loss: float
    strategy: str
    status: str  # 'open', 'closed', 'failed'


class RiskManager:
    """Manages trading risk and position limits"""

    def __init__(self, config: Dict):
        self.config = config
        self.max_position_size = config.get('max_position_size', 100)  # USD
        self.max_concurrent_trades = config.get('max_concurrent_trades', 5)
        self.max_daily_loss = config.get('max_daily_loss', 100)  # USD
        self.min_profit_threshold = config.get('min_profit_percent', 0.5)
        self.min_confidence = config.get('min_confidence', 0.7)

        # Track daily statistics
        self.daily_profit = 0.0
        self.daily_trades = 0
        self.last_reset = datetime.now().date()

    def reset_daily_stats(self):
        """Reset daily statistics at midnight"""
        today = datetime.now().date()
        if today > self.last_reset:
            self.daily_profit = 0.0
            self.daily_trades = 0
            self.last_reset = today
            logger.info("Daily statistics reset")

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit exceeded"""
        self.reset_daily_stats()
        if self.daily_profit < -self.max_daily_loss:
            logger.warning(f"Daily loss limit reached: ${abs(self.daily_profit):.2f}")
            return False
        return True

    def check_concurrent_trades(self, active_count: int) -> bool:
        """Check if max concurrent trades limit exceeded"""
        if active_count >= self.max_concurrent_trades:
            logger.warning(f"Max concurrent trades reached: {active_count}/{self.max_concurrent_trades}")
            return False
        return True

    def calculate_position_size(self, balance: float, risk_percent: float = 2.0) -> float:
        """Calculate position size based on risk management"""
        # Use Kelly Criterion simplified
        risk_amount = balance * (risk_percent / 100)
        position_size = min(risk_amount, self.max_position_size)
        return position_size

    def should_trade(self, pair_data: Dict, balance: float, active_trades: int) -> tuple[bool, str]:
        """
        Determine if a trade should be executed

        Returns:
            (should_trade: bool, reason: str)
        """
        # Check daily loss limit
        if not self.check_daily_loss_limit():
            return False, "Daily loss limit exceeded"

        # Check concurrent trades
        if not self.check_concurrent_trades(active_trades):
            return False, "Max concurrent trades reached"

        # Check profit threshold
        profit_score = pair_data.get('profit_score', 0)
        if profit_score < self.min_profit_threshold:
            return False, f"Profit too low: {profit_score:.2f}% < {self.min_profit_threshold}%"

        # Check confidence threshold
        confidence = pair_data.get('neural_confidence', 0)
        if confidence < self.min_confidence:
            return False, f"Confidence too low: {confidence:.2f} < {self.min_confidence}"

        # Check sufficient balance
        required_size = self.calculate_position_size(balance)
        if balance < required_size:
            return False, f"Insufficient balance: ${balance:.2f} < ${required_size:.2f}"

        return True, "All checks passed"


class PositionManager:
    """Manages open positions and exit strategies"""

    def __init__(self, config: Dict):
        self.config = config
        self.take_profit_percent = config.get('take_profit', 5.0)  # 5%
        self.stop_loss_percent = config.get('stop_loss', 2.0)  # 2%
        self.trailing_stop_percent = config.get('trailing_stop', 3.0)  # 3%
        self.max_hold_time_hours = config.get('max_hold_time_hours', 24)

    async def should_close_position(self, trade: TradeExecution, current_price: float) -> tuple[bool, str]:
        """
        Determine if position should be closed

        Returns:
            (should_close: bool, reason: str)
        """
        entry_price = trade.price
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        # Check take profit
        if pnl_percent >= self.take_profit_percent:
            return True, f"Take profit hit: {pnl_percent:.2f}%"

        # Check stop loss
        if pnl_percent <= -self.stop_loss_percent:
            return True, f"Stop loss hit: {pnl_percent:.2f}%"

        # Check trailing stop (for profitable positions)
        if pnl_percent > 0 and pnl_percent < self.trailing_stop_percent:
            # If price drops below trailing stop from peak
            if pnl_percent <= -(self.trailing_stop_percent - pnl_percent):
                return True, f"Trailing stop hit: {pnl_percent:.2f}%"

        # Check max hold time
        hold_time = datetime.now() - trade.timestamp
        if hold_time > timedelta(hours=self.max_hold_time_hours):
            return True, f"Max hold time exceeded: {hold_time.total_seconds()/3600:.1f}h"

        return False, "Position still valid"

    def calculate_exit_levels(self, entry_price: float) -> Dict:
        """Calculate take-profit and stop-loss levels"""
        return {
            'take_profit': entry_price * (1 + self.take_profit_percent / 100),
            'stop_loss': entry_price * (1 - self.stop_loss_percent / 100),
            'trailing_stop': entry_price * (1 + self.trailing_stop_percent / 100)
        }


class AutoExecutionEngine:
    """
    Main autonomous trading execution engine
    """

    def __init__(self, neural_net, audit_manager=None, config: Dict = None):
        self.neural_net = neural_net
        self.audit_manager = audit_manager
        self.config = config or {}

        # Initialize managers
        self.risk_manager = RiskManager(self.config.get('trading', {}))
        self.position_manager = PositionManager(self.config.get('trading', {}))

        # State tracking
        self.active_trades: Dict[str, TradeExecution] = {}
        self.is_active = False
        self.auto_trade_enabled = self.config.get('auto_trade', False)

        # Phase 1 & 2 Integration Manager (will be set by neural_net)
        self.integration_manager = None

        # RL Agent integration
        self.rl_agent = None
        self.use_rl = self.config.get('use_rl_agent', False)
        if self.use_rl:
            self._load_rl_agent()

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0

        logger.info("🤖 Auto-Execution Engine initialized")

    def _load_rl_agent(self):
        """Load trained RL agent"""
        try:
            from nexlify_rl_agent import DQNAgent
            from pathlib import Path

            model_path = Path("models/rl_agent_trained.pth")

            if not model_path.exists():
                logger.warning("⚠️ RL model not found, using rule-based trading")
                self.use_rl = False
                return

            # Create agent (state_size=8, action_size=3)
            self.rl_agent = DQNAgent(state_size=8, action_size=3)
            self.rl_agent.load(str(model_path))

            logger.info("🧠 RL Agent loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
            self.use_rl = False

    def _get_rl_state(self, pair_data: Dict, balance: float) -> np.ndarray:
        """Convert market data to RL state representation"""
        current_price = pair_data.get('current_price', 0)

        # State: [balance, position, position_price, current_price,
        #         price_change, RSI, MACD, volume_ratio]
        state = np.array([
            balance / 10000,  # Normalized balance
            0,  # No current position for new trade
            0,  # No entry price yet
            current_price / 10000,  # Normalized price
            pair_data.get('price_change', 0),
            pair_data.get('rsi', 0.5),
            pair_data.get('macd', 0),
            pair_data.get('volume_ratio', 0.5)
        ], dtype=np.float32)

        return state

    def enable(self):
        """Enable auto-trading"""
        self.auto_trade_enabled = True
        self.is_active = True
        mode = "RL-powered" if self.use_rl else "rule-based"
        logger.info(f"✅ Auto-trading ENABLED ({mode})")

    def disable(self):
        """Disable auto-trading"""
        self.auto_trade_enabled = False
        logger.warning("⚠️ Auto-trading DISABLED")

    async def start(self):
        """Start the auto-execution engine"""
        if not self.neural_net:
            logger.error("Neural net not available")
            return

        self.is_active = True
        logger.info("🚀 Auto-Execution Engine started")

        # Start background tasks
        await asyncio.gather(
            self.opportunity_monitor(),
            self.position_monitor(),
            self.performance_reporter()
        )

    async def stop(self):
        """Stop the auto-execution engine"""
        self.is_active = False
        logger.info("🛑 Auto-Execution Engine stopped")

        # Close all open positions
        if self.active_trades:
            logger.warning(f"Closing {len(self.active_trades)} open positions...")
            await self.close_all_positions()

    @handle_errors("Opportunity Monitor", reraise=False)
    async def opportunity_monitor(self):
        """Monitor for trading opportunities and execute automatically"""
        logger.info("👀 Opportunity monitor started")

        while self.is_active:
            try:
                # Check if auto-trading is enabled
                if not self.auto_trade_enabled:
                    await asyncio.sleep(10)
                    continue

                # Get opportunities from neural net
                if not hasattr(self.neural_net, 'active_pairs'):
                    await asyncio.sleep(30)
                    continue

                active_pairs = self.neural_net.active_pairs

                # Get account balance
                balance = await self.get_available_balance()

                # Check each opportunity
                for symbol, pair in active_pairs.items():
                    # Skip if already have position in this pair
                    if symbol in self.active_trades:
                        continue

                    # Convert pair to dict format
                    pair_data = {
                        'symbol': pair.symbol,
                        'profit_score': pair.profit_score,
                        'neural_confidence': pair.neural_confidence,
                        'exchanges': pair.exchanges
                    }

                    # Check with RL agent first (if enabled)
                    rl_approved = True
                    if self.use_rl and self.rl_agent:
                        state = self._get_rl_state(pair_data, balance)
                        action = self.rl_agent.act(state, training=False)

                        # Action: 0=Hold, 1=Buy, 2=Sell
                        # Only proceed if RL says Buy (action=1)
                        if action != 1:
                            rl_approved = False
                            logger.debug(f"🤖 RL Agent: Skip {symbol} (action={action})")

                    if rl_approved:
                        # Check with risk manager
                        should_trade, reason = self.risk_manager.should_trade(
                            pair_data,
                            balance,
                            len(self.active_trades)
                        )

                        if should_trade:
                            rl_tag = "🤖 RL+" if self.use_rl else ""
                            logger.info(f"🎯 {rl_tag} Trade opportunity: {symbol} ({reason})")
                            await self.execute_trade(pair)
                        else:
                            logger.debug(f"⏭️ Skipping {symbol}: {reason}")

                # Sleep before next check (30 seconds)
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in opportunity monitor: {e}")
                await asyncio.sleep(60)

    @handle_errors("Position Monitor", reraise=False)
    async def position_monitor(self):
        """Monitor open positions and manage exits"""
        logger.info("📊 Position monitor started")

        while self.is_active:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                if not self.active_trades:
                    continue

                # Check each position
                for trade_id, trade in list(self.active_trades.items()):
                    try:
                        # Get current price
                        current_price = await self.get_current_price(
                            trade.exchange,
                            trade.symbol
                        )

                        # Check if should close
                        should_close, reason = await self.position_manager.should_close_position(
                            trade,
                            current_price
                        )

                        if should_close:
                            logger.info(f"🔄 Closing position: {trade.symbol} - {reason}")
                            await self.close_position(trade_id, reason)

                    except Exception as e:
                        logger.error(f"Error monitoring position {trade_id}: {e}")

            except Exception as e:
                logger.error(f"Error in position monitor: {e}")
                await asyncio.sleep(60)

    async def execute_trade(self, pair):
        """Execute a trade automatically"""
        try:
            symbol = pair.symbol
            exchange_id = pair.exchanges[0]  # Use first available exchange

            # Get balance
            balance = await self.get_available_balance(exchange_id)

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(balance)

            # Get current price
            ticker = await self.neural_net.exchanges[exchange_id].fetch_ticker(symbol)
            current_price = ticker['last']

            # Calculate amount in base currency
            amount = position_size / current_price

            # Execute buy order
            logger.info(f"📈 Executing BUY: {amount:.6f} {symbol} @ ${current_price:.2f}")

            order = await self.neural_net.exchanges[exchange_id].create_market_buy_order(
                symbol,
                amount
            )

            if order and order.get('status') in ['closed', 'filled']:
                # Calculate exit levels
                exit_levels = self.position_manager.calculate_exit_levels(current_price)

                # Create trade record
                trade = TradeExecution(
                    trade_id=order['id'],
                    symbol=symbol,
                    exchange=exchange_id,
                    side='buy',
                    amount=amount,
                    price=current_price,
                    timestamp=datetime.now(),
                    profit_target=exit_levels['take_profit'],
                    stop_loss=exit_levels['stop_loss'],
                    strategy='auto_execution',
                    status='open'
                )

                self.active_trades[trade.trade_id] = trade
                self.total_trades += 1

                logger.info(f"✅ Trade executed: {symbol} - TP: ${exit_levels['take_profit']:.2f}, SL: ${exit_levels['stop_loss']:.2f}")

                # Audit log
                if self.audit_manager:
                    await self.audit_manager.audit_trade(
                        'auto_trader',
                        exchange_id,
                        symbol,
                        'buy',
                        amount,
                        current_price,
                        'market',
                        True
                    )

                # Notify integration manager (Phase 1 & 2)
                if self.integration_manager:
                    asyncio.create_task(self.integration_manager.on_trade_executed({
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': amount,
                        'price': current_price,
                        'exchange': exchange_id,
                        'timestamp': datetime.now(),
                        'fees': order.get('fee', {}).get('cost', 0) if isinstance(order.get('fee'), dict) else 0
                    }))

                return True
            else:
                logger.error(f"❌ Trade failed: {order}")
                return False

        except Exception as e:
            error_handler.log_error(e, f"Trade execution failed: {symbol}", severity="error")
            return False

    async def close_position(self, trade_id: str, reason: str):
        """Close an open position"""
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                return False

            # Get current price
            current_price = await self.get_current_price(trade.exchange, trade.symbol)

            # Calculate PnL
            pnl = (current_price - trade.price) * trade.amount
            pnl_percent = ((current_price - trade.price) / trade.price) * 100

            # Execute sell order
            logger.info(f"📉 Executing SELL: {trade.amount:.6f} {trade.symbol} @ ${current_price:.2f}")

            order = await self.neural_net.exchanges[trade.exchange].create_market_sell_order(
                trade.symbol,
                trade.amount
            )

            if order and order.get('status') in ['closed', 'filled']:
                # Update statistics
                self.risk_manager.daily_profit += pnl
                self.total_profit += pnl

                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                # Remove from active trades
                trade.status = 'closed'
                del self.active_trades[trade_id]

                logger.info(f"✅ Position closed: {trade.symbol} - PnL: ${pnl:.2f} ({pnl_percent:+.2f}%) - {reason}")

                # Audit log
                if self.audit_manager:
                    await self.audit_manager.audit_trade(
                        'auto_trader',
                        trade.exchange,
                        trade.symbol,
                        'sell',
                        trade.amount,
                        current_price,
                        'market',
                        True
                    )

                # Notify integration manager (Phase 1 & 2)
                if self.integration_manager:
                    # Record the sell transaction
                    asyncio.create_task(self.integration_manager.on_trade_executed({
                        'symbol': trade.symbol,
                        'side': 'sell',
                        'quantity': trade.amount,
                        'price': current_price,
                        'exchange': trade.exchange,
                        'timestamp': datetime.now(),
                        'fees': order.get('fee', {}).get('cost', 0) if isinstance(order.get('fee'), dict) else 0
                    }))

                    # Notify position closure
                    asyncio.create_task(self.integration_manager.on_position_closed({
                        'trade_id': trade_id,
                        'symbol': trade.symbol,
                        'pnl': pnl,
                        'exit_price': current_price
                    }))

                return True
            else:
                logger.error(f"❌ Close failed: {order}")
                return False

        except Exception as e:
            error_handler.log_error(e, f"Close position failed: {trade_id}", severity="error")
            return False

    async def close_all_positions(self):
        """Close all open positions (emergency stop)"""
        for trade_id in list(self.active_trades.keys()):
            await self.close_position(trade_id, "Emergency close")

    async def get_available_balance(self, exchange_id: str = None) -> float:
        """Get available trading balance"""
        try:
            if not exchange_id:
                # Use first available exchange
                exchange_id = list(self.neural_net.exchanges.keys())[0]

            balance = await self.neural_net.exchanges[exchange_id].fetch_balance()

            # Get USDT balance
            usdt_free = balance.get('USDT', {}).get('free', 0)
            return float(usdt_free)

        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0

    async def get_current_price(self, exchange_id: str, symbol: str) -> float:
        """Get current market price for a symbol"""
        try:
            ticker = await self.neural_net.exchanges[exchange_id].fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    async def performance_reporter(self):
        """Periodically report performance statistics"""
        while self.is_active:
            try:
                await asyncio.sleep(3600)  # Every hour

                if self.total_trades > 0:
                    win_rate = (self.winning_trades / self.total_trades) * 100
                    avg_profit = self.total_profit / self.total_trades

                    logger.info("="*50)
                    logger.info("📊 AUTO-TRADER PERFORMANCE REPORT")
                    logger.info("="*50)
                    logger.info(f"Total Trades: {self.total_trades}")
                    logger.info(f"Win Rate: {win_rate:.2f}%")
                    logger.info(f"Total Profit: ${self.total_profit:.2f}")
                    logger.info(f"Avg Profit/Trade: ${avg_profit:.2f}")
                    logger.info(f"Active Positions: {len(self.active_trades)}")
                    logger.info("="*50)

            except Exception as e:
                logger.error(f"Error in performance reporter: {e}")

    def get_statistics(self) -> Dict:
        """Get current trading statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'active_positions': len(self.active_trades),
            'daily_profit': self.risk_manager.daily_profit,
            'auto_trade_enabled': self.auto_trade_enabled
        }
