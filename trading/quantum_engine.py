# nexlify/trading/quantum_engine.py
"""
Nexlify Quantum Trading Engine - The Digital Market Executor
Executes trades with the precision of a corpo assassin and the speed of a netrunner
This is where signals become profits
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import numpy as np
from sortedcontainers import SortedDict
import aiohttp
import websockets
import hmac
import hashlib
import time

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

# Our chrome
from config.config_manager import get_config
from database.models import (
    Market, Symbol, Order, Position, Portfolio,
    OrderStatus, OrderType, OrderSide, Candle
)
from monitoring.sentinel import get_sentinel, MetricType, AlertSeverity
from ml.neural_trader import TradingSignal

logger = logging.getLogger("nexlify.trading")

class RiskLevel(Enum):
    """Risk levels - from careful to YOLO"""
    CONSERVATIVE = "conservative"  # 1% risk per trade
    MODERATE = "moderate"  # 2% risk per trade
    AGGRESSIVE = "aggressive"  # 3% risk per trade
    DEGEN = "degen"  # 5%+ risk - living on the edge

class ExecutionMode(Enum):
    """Execution modes - how we interact with the market"""
    PAPER = "paper"  # Simulated trading
    LIVE = "live"  # Real money, real consequences
    HYBRID = "hybrid"  # Paper trade with live data

@dataclass
class MarketTick:
    """Real-time market data tick - the pulse of the market"""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume: Decimal
    
    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid
    
    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

@dataclass
class OrderBook:
    """Order book snapshot - see the matrix of supply and demand"""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[Decimal, Decimal]]  # [(price, quantity), ...]
    asks: List[Tuple[Decimal, Decimal]]
    
    def get_market_impact(self, side: OrderSide, quantity: Decimal) -> Decimal:
        """Calculate market impact of an order - how much we move the market"""
        book = self.asks if side == OrderSide.BUY else self.bids
        remaining = quantity
        total_cost = Decimal(0)
        
        for price, available in book:
            if remaining <= 0:
                break
            
            filled = min(remaining, available)
            total_cost += filled * price
            remaining -= filled
        
        if remaining > 0:
            # Not enough liquidity
            return Decimal('inf')
        
        avg_price = total_cost / quantity
        return avg_price

@dataclass
class ExecutionReport:
    """Trade execution report - the receipt of digital warfare"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    filled_quantity: Decimal
    average_price: Decimal
    status: OrderStatus
    timestamp: datetime
    fees: Decimal
    slippage: Decimal
    execution_time_ms: int

class RiskManager:
    """
    Risk Management System - Your digital bodyguard
    Prevents you from YOLOing your entire stack
    """
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.config = get_config()
        self.sentinel = get_sentinel()
        
        # Risk parameters
        self.max_position_size = portfolio.max_position_size / 100  # Convert to decimal
        self.max_correlation = 0.7  # Max correlation between positions
        self.max_sector_exposure = 0.4  # Max exposure to one sector
        self.max_drawdown = 0.2  # 20% max drawdown
        
        # Risk tracking
        self.daily_loss = Decimal(0)
        self.daily_loss_limit = Decimal('0.05')  # 5% daily loss limit
        self.open_risk = Decimal(0)
        self.correlation_matrix = {}
    
    def calculate_position_size(
        self,
        signal: TradingSignal,
        account_balance: Decimal,
        current_price: Decimal,
        risk_level: RiskLevel = RiskLevel.MODERATE
    ) -> Decimal:
        """
        Calculate position size using Kelly Criterion with safety limits
        Like calculating how much chrome you can afford to lose
        """
        # Base risk per trade
        risk_percentages = {
            RiskLevel.CONSERVATIVE: Decimal('0.01'),
            RiskLevel.MODERATE: Decimal('0.02'),
            RiskLevel.AGGRESSIVE: Decimal('0.03'),
            RiskLevel.DEGEN: Decimal('0.05')
        }
        
        base_risk = risk_percentages[risk_level]
        
        # Adjust for confidence (Kelly Criterion simplified)
        confidence_adj = Decimal(str(signal.confidence))
        win_rate = Decimal('0.6')  # Assumed from backtesting
        avg_win_loss_ratio = Decimal('2.0')  # Risk:Reward ratio
        
        # Kelly percentage = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        kelly_pct = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        kelly_pct = max(Decimal(0), min(kelly_pct, Decimal('0.25')))  # Cap at 25%
        
        # Combine base risk with Kelly
        risk_amount = account_balance * base_risk * confidence_adj * kelly_pct
        
        # Calculate position size based on stop loss
        stop_distance = abs(current_price - Decimal(str(signal.stop_loss)))
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = Decimal(0)
        
        # Apply portfolio limits
        max_position_value = account_balance * self.max_position_size
        max_position_units = max_position_value / current_price
        
        # Final position size
        position_size = min(position_size, max_position_units)
        
        # Round down to avoid exceeding limits
        return position_size.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
    
    def check_risk_limits(self, proposed_order: Dict) -> Tuple[bool, Optional[str]]:
        """
        Check if order passes risk limits - the bouncer at the club
        Returns (allowed, rejection_reason)
        """
        # Check daily loss limit
        if self.daily_loss >= self.daily_loss_limit:
            return False, "Daily loss limit reached - trading halted"
        
        # Check position concentration
        position_value = proposed_order['quantity'] * proposed_order['price']
        total_value = self._get_total_portfolio_value()
        
        if total_value > 0:
            position_pct = position_value / total_value
            if position_pct > self.max_position_size:
                return False, f"Position too large: {position_pct:.1%} of portfolio"
        
        # Check correlation with existing positions
        correlation_risk = self._check_correlation_risk(proposed_order['symbol'])
        if correlation_risk > self.max_correlation:
            return False, f"High correlation with existing positions: {correlation_risk:.2f}"
        
        # Check sector exposure
        sector_exposure = self._get_sector_exposure(proposed_order['symbol'])
        if sector_exposure > self.max_sector_exposure:
            return False, f"Sector exposure too high: {sector_exposure:.1%}"
        
        # All checks passed
        return True, None
    
    def update_risk_metrics(self, execution: ExecutionReport):
        """Update risk metrics after trade execution"""
        if execution.status == OrderStatus.FILLED:
            # Update daily P&L
            if execution.side == OrderSide.SELL:
                # Assuming closing position
                pnl = self._calculate_position_pnl(execution)
                self.daily_loss += pnl if pnl < 0 else 0
            
            # Log risk metrics
            self.sentinel.record_metric(
                MetricType.TRADING,
                "risk_metrics",
                {
                    'daily_loss': float(self.daily_loss),
                    'open_risk': float(self.open_risk),
                    'positions_count': self._get_open_positions_count()
                }
            )
    
    def _get_total_portfolio_value(self) -> Decimal:
        """Get total portfolio value including positions"""
        # Simplified - in production would calculate from actual positions
        return self.portfolio.initial_balance + self.portfolio.total_pnl
    
    def _check_correlation_risk(self, symbol: str) -> float:
        """Check correlation with existing positions"""
        # Simplified correlation check
        # In production, would use actual price correlation data
        return 0.5  # Placeholder
    
    def _get_sector_exposure(self, symbol: str) -> float:
        """Get exposure to symbol's sector"""
        # Simplified sector check
        # In production, would categorize symbols by sector
        return 0.2  # Placeholder
    
    def _get_open_positions_count(self) -> int:
        """Get count of open positions"""
        # In production, would query database
        return 3  # Placeholder
    
    def _calculate_position_pnl(self, execution: ExecutionReport) -> Decimal:
        """Calculate P&L for a position"""
        # Simplified P&L calculation
        return Decimal('100')  # Placeholder

class OrderManager:
    """
    Order Management System - Handles order lifecycle
    Like a fixer handling your deals in Night City
    """
    
    def __init__(self, exchange_connector):
        self.exchange = exchange_connector
        self.pending_orders: Dict[str, Order] = {}
        self.order_queue: asyncio.Queue = asyncio.Queue()
        self.config = get_config()
        self.sentinel = get_sentinel()
        
        # Order tracking
        self.order_history: deque = deque(maxlen=1000)
        self.fill_tracker: Dict[str, List[Decimal]] = defaultdict(list)
    
    async def submit_order(
        self,
        order: Order,
        execution_mode: ExecutionMode = ExecutionMode.LIVE
    ) -> ExecutionReport:
        """
        Submit order to exchange - send it into the matrix
        """
        start_time = time.time()
        
        try:
            if execution_mode == ExecutionMode.PAPER:
                # Simulate execution
                execution = await self._simulate_execution(order)
            else:
                # Real execution
                execution = await self._execute_live_order(order)
            
            # Update metrics
            execution_time = int((time.time() - start_time) * 1000)
            execution.execution_time_ms = execution_time
            
            # Record metrics
            self.sentinel.orders_placed.labels(
                type=order.type.value,
                side=order.side.value
            ).inc()
            
            # Track order
            self.order_history.append(execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            self.sentinel.orders_failed.labels(reason="submission_error").inc()
            
            return ExecutionReport(
                order_id=str(order.id),
                symbol=order.symbol.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=Decimal(0),
                average_price=Decimal(0),
                status=OrderStatus.REJECTED,
                timestamp=datetime.now(timezone.utc),
                fees=Decimal(0),
                slippage=Decimal(0),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _simulate_execution(self, order: Order) -> ExecutionReport:
        """Simulate order execution for paper trading"""
        # Get current market price
        market_price = await self._get_market_price(order.symbol.symbol)
        
        # Simulate slippage
        slippage_pct = Decimal('0.001')  # 0.1% slippage
        if order.side == OrderSide.BUY:
            execution_price = market_price * (1 + slippage_pct)
        else:
            execution_price = market_price * (1 - slippage_pct)
        
        # Simulate fees
        fee_rate = Decimal('0.001')  # 0.1% fee
        fees = order.quantity * execution_price * fee_rate
        
        # Create execution report
        return ExecutionReport(
            order_id=str(order.id),
            symbol=order.symbol.symbol,
            side=order.side,
            quantity=order.quantity,
            filled_quantity=order.quantity,
            average_price=execution_price,
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc),
            fees=fees,
            slippage=execution_price - market_price,
            execution_time_ms=50  # Simulated latency
        )
    
    async def _execute_live_order(self, order: Order) -> ExecutionReport:
        """Execute real order on exchange"""
        # This would integrate with actual exchange APIs
        # For now, placeholder
        return await self._simulate_execution(order)
    
    async def _get_market_price(self, symbol: str) -> Decimal:
        """Get current market price"""
        # In production, would fetch from exchange
        return Decimal('50000')  # Placeholder
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order - abort the mission"""
        try:
            if order_id in self.pending_orders:
                order = self.pending_orders[order_id]
                
                # Send cancellation to exchange
                success = await self.exchange.cancel_order(order_id)
                
                if success:
                    del self.pending_orders[order_id]
                    logger.info(f"Order {order_id} cancelled")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return list(self.pending_orders.values())

class PositionManager:
    """
    Position Management System - Track your stakes in the game
    Manages entries, exits, and everything in between
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.positions: Dict[str, Position] = {}
        self.config = get_config()
        self.sentinel = get_sentinel()
        
        # Position tracking
        self.position_history = deque(maxlen=1000)
        self.pnl_tracker = defaultdict(Decimal)
    
    def open_position(
        self,
        portfolio_id: str,
        symbol_id: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None
    ) -> Position:
        """
        Open a new position - stake your claim
        """
        position = Position(
            portfolio_id=portfolio_id,
            symbol_id=symbol_id,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            is_open=True
        )
        
        self.db.add(position)
        self.db.commit()
        
        # Cache position
        self.positions[str(position.id)] = position
        
        # Update metrics
        self.sentinel.active_positions.inc()
        
        logger.info(f"Position opened: {side} {quantity} @ {entry_price}")
        
        return position
    
    def update_position(
        self,
        position_id: str,
        current_price: Decimal,
        update_stops: bool = False
    ):
        """
        Update position with current market data
        Track unrealized P&L
        """
        position = self.positions.get(position_id)
        if not position:
            position = self.db.query(Position).filter(Position.id == position_id).first()
        
        if position and position.is_open:
            # Update current price
            position.current_price = current_price
            
            # Calculate unrealized P&L
            if position.side == "long":
                position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
            else:  # short
                position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
            
            # Update trailing stop if enabled
            if update_stops and position.trailing_stop_distance:
                self._update_trailing_stop(position, current_price)
            
            self.db.commit()
            
            # Update metrics
            total_unrealized = sum(p.unrealized_pnl for p in self.positions.values() if p.is_open)
            self.sentinel.total_pnl.set(float(total_unrealized))
    
    def close_position(
        self,
        position_id: str,
        exit_price: Decimal,
        exit_reason: str = "manual"
    ) -> Decimal:
        """
        Close a position - cash out or cut losses
        """
        position = self.positions.get(position_id)
        if not position:
            position = self.db.query(Position).filter(Position.id == position_id).first()
        
        if position and position.is_open:
            # Calculate realized P&L
            if position.side == "long":
                realized_pnl = (exit_price - position.entry_price) * position.quantity
            else:  # short
                realized_pnl = (position.entry_price - exit_price) * position.quantity
            
            # Update position
            position.is_open = False
            position.closed_at = datetime.now(timezone.utc)
            position.realized_pnl = realized_pnl
            position.current_price = exit_price
            
            # Update portfolio stats
            portfolio = self.db.query(Portfolio).filter(
                Portfolio.id == position.portfolio_id
            ).first()
            
            if portfolio:
                portfolio.total_trades += 1
                if realized_pnl > 0:
                    portfolio.winning_trades += 1
                portfolio.total_pnl += realized_pnl
            
            self.db.commit()
            
            # Remove from cache
            if position_id in self.positions:
                del self.positions[position_id]
            
            # Update metrics
            self.sentinel.active_positions.dec()
            
            # Log closure
            logger.info(
                f"Position closed: {position.side} {position.quantity} @ {exit_price} "
                f"P&L: {realized_pnl:.2f} ({exit_reason})"
            )
            
            return realized_pnl
        
        return Decimal(0)
    
    def _update_trailing_stop(self, position: Position, current_price: Decimal):
        """Update trailing stop loss"""
        if position.side == "long":
            # For long positions, trail below the high
            new_stop = current_price - position.trailing_stop_distance
            if position.stop_loss is None or new_stop > position.stop_loss:
                position.stop_loss = new_stop
        else:  # short
            # For short positions, trail above the low
            new_stop = current_price + position.trailing_stop_distance
            if position.stop_loss is None or new_stop < position.stop_loss:
                position.stop_loss = new_stop
    
    def check_stop_levels(self, market_tick: MarketTick) -> List[Position]:
        """
        Check if any positions hit stop loss or take profit
        Returns positions that should be closed
        """
        positions_to_close = []
        
        for position in self.positions.values():
            if not position.is_open:
                continue
            
            if position.symbol.symbol != market_tick.symbol:
                continue
            
            # Check stop loss
            if position.stop_loss:
                if position.side == "long" and market_tick.bid <= position.stop_loss:
                    positions_to_close.append(position)
                    logger.warning(f"Stop loss triggered for position {position.id}")
                elif position.side == "short" and market_tick.ask >= position.stop_loss:
                    positions_to_close.append(position)
                    logger.warning(f"Stop loss triggered for position {position.id}")
            
            # Check take profit
            if position.take_profit:
                if position.side == "long" and market_tick.bid >= position.take_profit:
                    positions_to_close.append(position)
                    logger.info(f"Take profit triggered for position {position.id}")
                elif position.side == "short" and market_tick.ask <= position.take_profit:
                    positions_to_close.append(position)
                    logger.info(f"Take profit triggered for position {position.id}")
        
        return positions_to_close

class QuantumTradingEngine:
    """
    The main trading engine - orchestrates all trading operations
    Like a master netrunner coordinating multiple daemons
    """
    
    def __init__(
        self,
        db_session: Session,
        portfolio: Portfolio,
        execution_mode: ExecutionMode = ExecutionMode.PAPER
    ):
        self.db = db_session
        self.portfolio = portfolio
        self.execution_mode = execution_mode
        self.config = get_config()
        self.sentinel = get_sentinel()
        
        # Initialize components
        self.risk_manager = RiskManager(portfolio)
        self.order_manager = OrderManager(None)  # Exchange connector would go here
        self.position_manager = PositionManager(db_session)
        
        # Market data
        self.market_data: Dict[str, MarketTick] = {}
        self.order_books: Dict[str, OrderBook] = {}
        
        # Engine state
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.trade_log = deque(maxlen=1000)
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': Decimal(0),
            'max_drawdown': Decimal(0),
            'sharpe_ratio': 0.0
        }
    
    async def start(self):
        """
        Start the trading engine - boot up the neural matrix
        """
        if self.is_running:
            logger.warning("Trading engine already running")
            return
        
        logger.info(f"Starting Quantum Trading Engine in {self.execution_mode.value} mode")
        self.is_running = True
        
        # Start component tasks
        self.tasks = [
            asyncio.create_task(self._market_data_loop()),
            asyncio.create_task(self._order_execution_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._position_management_loop()),
            asyncio.create_task(self._performance_tracking_loop())
        ]
        
        # Log startup
        await self.sentinel.raise_alert(
            severity=AlertSeverity.INFO,
            metric_type=MetricType.TRADING,
            title="Trading Engine Started",
            description=f"Quantum engine online in {self.execution_mode.value} mode",
            metric_value=1,
            threshold=0,
            component="trading_engine"
        )
    
    async def stop(self):
        """
        Stop the trading engine - graceful shutdown
        """
        logger.info("Stopping Quantum Trading Engine")
        self.is_running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close all positions if in paper mode
        if self.execution_mode == ExecutionMode.PAPER:
            await self._close_all_positions("engine_shutdown")
        
        logger.info("Trading engine stopped")
    
    async def execute_signal(
        self,
        signal: TradingSignal,
        risk_level: RiskLevel = RiskLevel.MODERATE
    ) -> Optional[ExecutionReport]:
        """
        Execute a trading signal - turn AI wisdom into market action
        """
        try:
            # Get symbol
            symbol = self.db.query(Symbol).filter(
                Symbol.symbol == signal.symbol
            ).first()
            
            if not symbol:
                logger.error(f"Symbol not found: {signal.symbol}")
                return None
            
            # Get account balance
            account_balance = self._get_account_balance()
            current_price = Decimal(str(signal.entry_price))
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal,
                account_balance,
                current_price,
                risk_level
            )
            
            if position_size <= 0:
                logger.warning("Position size too small, skipping trade")
                return None
            
            # Create order
            order = Order(
                portfolio_id=self.portfolio.id,
                symbol_id=symbol.id,
                type=OrderType.LIMIT,
                side=OrderSide.BUY if signal.action == "buy" else OrderSide.SELL,
                quantity=position_size,
                price=current_price,
                ai_signal_strength=signal.action,
                ai_confidence=Decimal(str(signal.confidence))
            )
            
            # Check risk limits
            order_dict = {
                'symbol': signal.symbol,
                'quantity': position_size,
                'price': current_price
            }
            
            allowed, reason = self.risk_manager.check_risk_limits(order_dict)
            if not allowed:
                logger.warning(f"Order rejected by risk manager: {reason}")
                return None
            
            # Submit order
            self.db.add(order)
            self.db.commit()
            
            execution = await self.order_manager.submit_order(order, self.execution_mode)
            
            # Handle execution result
            if execution.status == OrderStatus.FILLED:
                # Open position
                position = self.position_manager.open_position(
                    portfolio_id=self.portfolio.id,
                    symbol_id=symbol.id,
                    side="long" if signal.action == "buy" else "short",
                    quantity=execution.filled_quantity,
                    entry_price=execution.average_price,
                    stop_loss=Decimal(str(signal.stop_loss)),
                    take_profit=Decimal(str(signal.take_profit[0]))
                )
                
                # Update order with position
                order.position_id = position.id
                self.db.commit()
                
                # Log successful trade
                logger.info(
                    f"Signal executed: {signal.action} {execution.filled_quantity} "
                    f"{signal.symbol} @ {execution.average_price}"
                )
            
            # Update risk metrics
            self.risk_manager.update_risk_metrics(execution)
            
            return execution
            
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            return None
    
    async def _market_data_loop(self):
        """
        Market data processing loop - stay connected to the pulse
        """
        while self.is_running:
            try:
                # In production, this would connect to exchange websockets
                # For now, simulate market updates
                
                for symbol in self.get_active_symbols():
                    # Simulate tick
                    tick = self._generate_mock_tick(symbol)
                    self.market_data[symbol] = tick
                    
                    # Check stop levels
                    positions_to_close = self.position_manager.check_stop_levels(tick)
                    
                    for position in positions_to_close:
                        await self._close_position_market(position, tick)
                
                await asyncio.sleep(1)  # 1 second updates
                
            except Exception as e:
                logger.error(f"Market data error: {e}")
                await asyncio.sleep(5)
    
    async def _order_execution_loop(self):
        """
        Order execution loop - process the order queue
        """
        while self.is_running:
            try:
                # Process any pending orders
                # In production, would handle order state management
                
                await asyncio.sleep(0.1)  # 100ms loop
                
            except Exception as e:
                logger.error(f"Order execution error: {e}")
                await asyncio.sleep(1)
    
    async def _risk_monitoring_loop(self):
        """
        Risk monitoring loop - the guardian angel
        """
        while self.is_running:
            try:
                # Monitor portfolio risk metrics
                total_value = self._get_portfolio_value()
                open_positions = len(self.position_manager.positions)
                
                # Check for risk warnings
                if self.risk_manager.daily_loss > self.risk_manager.daily_loss_limit * 0.8:
                    await self.sentinel.raise_alert(
                        severity=AlertSeverity.WARNING,
                        metric_type=MetricType.TRADING,
                        title="Approaching Daily Loss Limit",
                        description=f"Daily loss at {float(self.risk_manager.daily_loss):.2%}",
                        metric_value=float(self.risk_manager.daily_loss),
                        threshold=float(self.risk_manager.daily_loss_limit),
                        component="risk_manager"
                    )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _position_management_loop(self):
        """
        Position management loop - tend to your garden
        """
        while self.is_running:
            try:
                # Update all open positions
                for position in self.position_manager.positions.values():
                    if position.is_open and position.symbol.symbol in self.market_data:
                        tick = self.market_data[position.symbol.symbol]
                        self.position_manager.update_position(
                            str(position.id),
                            tick.mid,
                            update_stops=True
                        )
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Position management error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_tracking_loop(self):
        """
        Performance tracking loop - know your stats
        """
        while self.is_running:
            try:
                # Calculate performance metrics
                self._update_performance_metrics()
                
                # Log to monitoring
                self.sentinel.record_metric(
                    MetricType.TRADING,
                    "performance",
                    {
                        'total_pnl': float(self.performance_metrics['total_pnl']),
                        'win_rate': self._calculate_win_rate(),
                        'sharpe_ratio': self.performance_metrics['sharpe_ratio'],
                        'open_positions': len(self.position_manager.positions)
                    }
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(120)
    
    def _get_account_balance(self) -> Decimal:
        """Get current account balance"""
        return self.portfolio.initial_balance + self.portfolio.total_pnl
    
    def _get_portfolio_value(self) -> Decimal:
        """Get total portfolio value including positions"""
        cash = self._get_account_balance()
        position_value = sum(
            p.quantity * p.current_price
            for p in self.position_manager.positions.values()
            if p.is_open
        )
        return cash + position_value
    
    def get_active_symbols(self) -> List[str]:
        """Get list of symbols we're actively trading"""
        # In production, would be configurable
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    def _generate_mock_tick(self, symbol: str) -> MarketTick:
        """Generate mock market tick for testing"""
        base_prices = {
            "BTC/USDT": Decimal("50000"),
            "ETH/USDT": Decimal("3000"),
            "SOL/USDT": Decimal("100")
        }
        
        base = base_prices.get(symbol, Decimal("100"))
        spread = base * Decimal("0.0001")  # 0.01% spread
        
        # Add some randomness
        import random
        change = Decimal(str(random.uniform(-0.001, 0.001)))
        base = base * (1 + change)
        
        return MarketTick(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bid=base - spread/2,
            ask=base + spread/2,
            last=base,
            volume=Decimal(str(random.uniform(100, 1000)))
        )
    
    async def _close_position_market(self, position: Position, tick: MarketTick):
        """Close position at market price"""
        exit_price = tick.bid if position.side == "long" else tick.ask
        
        pnl = self.position_manager.close_position(
            str(position.id),
            exit_price,
            "stop_triggered"
        )
        
        logger.info(f"Position closed at market: P&L {pnl:.2f}")
    
    async def _close_all_positions(self, reason: str):
        """Close all open positions"""
        for position in list(self.position_manager.positions.values()):
            if position.is_open:
                tick = self.market_data.get(position.symbol.symbol)
                if tick:
                    await self._close_position_market(position, tick)
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        # Calculate from position history
        if self.portfolio.total_trades > 0:
            self.performance_metrics['total_trades'] = self.portfolio.total_trades
            self.performance_metrics['winning_trades'] = self.portfolio.winning_trades
            self.performance_metrics['total_pnl'] = self.portfolio.total_pnl
            
            # Simple Sharpe calculation (annualized)
            if len(self.trade_log) > 30:
                returns = [float(t['pnl']) for t in self.trade_log]
                if returns:
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        self.performance_metrics['sharpe_ratio'] = (avg_return / std_return) * np.sqrt(252)
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate"""
        if self.performance_metrics['total_trades'] > 0:
            return self.performance_metrics['winning_trades'] / self.performance_metrics['total_trades']
        return 0.0
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status - full diagnostics"""
        return {
            'is_running': self.is_running,
            'execution_mode': self.execution_mode.value,
            'portfolio_id': str(self.portfolio.id),
            'account_balance': float(self._get_account_balance()),
            'portfolio_value': float(self._get_portfolio_value()),
            'open_positions': len(self.position_manager.positions),
            'daily_pnl': float(self.risk_manager.daily_loss),
            'performance': {
                'total_trades': self.performance_metrics['total_trades'],
                'win_rate': self._calculate_win_rate(),
                'total_pnl': float(self.performance_metrics['total_pnl']),
                'sharpe_ratio': self.performance_metrics['sharpe_ratio']
            }
        }

# Global engine registry
_trading_engines: Dict[str, QuantumTradingEngine] = {}

def get_trading_engine(portfolio_id: str) -> Optional[QuantumTradingEngine]:
    """Get trading engine for portfolio"""
    return _trading_engines.get(portfolio_id)

def create_trading_engine(
    db_session: Session,
    portfolio: Portfolio,
    execution_mode: ExecutionMode = ExecutionMode.PAPER
) -> QuantumTradingEngine:
    """Create new trading engine for portfolio"""
    engine = QuantumTradingEngine(db_session, portfolio, execution_mode)
    _trading_engines[str(portfolio.id)] = engine
    return engine
