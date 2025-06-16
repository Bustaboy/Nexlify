# nexlify/database/models.py
"""
Nexlify Database Models - The Data Fortress
PostgreSQL-powered, built to handle the heat of Night City markets
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, List, Dict, Any
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, 
    ForeignKey, JSON, DECIMAL, Index, UniqueConstraint,
    Text, ARRAY, CheckConstraint, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

Base = declarative_base()

class OrderStatus(str, Enum):
    """Order status - track your deals through the digital maze"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(str, Enum):
    """Order types - different ways to jack into the market"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(str, Enum):
    """Buy or sell - are you going long or burning it down?"""
    BUY = "buy"
    SELL = "sell"

class SignalStrength(str, Enum):
    """AI signal strength - how confident is our neural net?"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"

class Market(Base):
    """Market/Exchange configuration - where the action happens"""
    __tablename__ = "markets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    exchange = Column(String(50), nullable=False)
    
    # Connection details
    api_url = Column(String(255), nullable=False)
    websocket_url = Column(String(255), nullable=True)
    
    # Trading parameters
    maker_fee = Column(DECIMAL(5, 4), default=Decimal("0.001"))
    taker_fee = Column(DECIMAL(5, 4), default=Decimal("0.001"))
    min_order_size = Column(DECIMAL(20, 8), default=Decimal("0.001"))
    
    # Status tracking
    is_active = Column(Boolean, default=True)
    maintenance_mode = Column(Boolean, default=False)
    last_health_check = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    # Relationships
    symbols = relationship("Symbol", back_populates="market")
    
    __table_args__ = (
        Index("idx_market_active", "is_active"),
    )

class Symbol(Base):
    """Trading symbols - your targets in the digital battlefield"""
    __tablename__ = "symbols"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    market_id = Column(UUID(as_uuid=True), ForeignKey("markets.id"), nullable=False)
    
    symbol = Column(String(20), nullable=False)  # BTC/USDT
    base_asset = Column(String(10), nullable=False)  # BTC
    quote_asset = Column(String(10), nullable=False)  # USDT
    
    # Trading constraints
    min_quantity = Column(DECIMAL(20, 8), nullable=False)
    max_quantity = Column(DECIMAL(20, 8), nullable=False)
    step_size = Column(DECIMAL(20, 8), nullable=False)
    
    min_price = Column(DECIMAL(20, 8), nullable=False)
    max_price = Column(DECIMAL(20, 8), nullable=False)
    tick_size = Column(DECIMAL(20, 8), nullable=False)
    
    # Status
    is_trading = Column(Boolean, default=True)
    is_margin_trading = Column(Boolean, default=False)
    
    # Performance metrics
    volume_24h = Column(DECIMAL(30, 8), nullable=True)
    price_change_24h = Column(DECIMAL(10, 4), nullable=True)
    
    # Relationships
    market = relationship("Market", back_populates="symbols")
    candles = relationship("Candle", back_populates="symbol")
    orders = relationship("Order", back_populates="symbol")
    
    __table_args__ = (
        UniqueConstraint("market_id", "symbol", name="uq_market_symbol"),
        Index("idx_symbol_trading", "is_trading"),
        Index("idx_symbol_base_quote", "base_asset", "quote_asset"),
    )

class Candle(Base):
    """OHLCV data - the digital heartbeat of the market"""
    __tablename__ = "candles"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("symbols.id"), nullable=False)
    
    # Time and interval
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    interval = Column(String(10), nullable=False)  # 1m, 5m, 1h, etc.
    
    # OHLCV data
    open = Column(DECIMAL(20, 8), nullable=False)
    high = Column(DECIMAL(20, 8), nullable=False)
    low = Column(DECIMAL(20, 8), nullable=False)
    close = Column(DECIMAL(20, 8), nullable=False)
    volume = Column(DECIMAL(30, 8), nullable=False)
    
    # Additional metrics
    trades_count = Column(Integer, default=0)
    quote_volume = Column(DECIMAL(30, 8), nullable=True)
    
    # Technical indicators (cached)
    indicators = Column(JSONB, default=dict)  # RSI, MACD, etc.
    
    # Relationships
    symbol = relationship("Symbol", back_populates="candles")
    
    __table_args__ = (
        UniqueConstraint("symbol_id", "timestamp", "interval", name="uq_candle_time"),
        Index("idx_candle_lookup", "symbol_id", "interval", "timestamp"),
        CheckConstraint("high >= low", name="check_high_low"),
        CheckConstraint("high >= open", name="check_high_open"),
        CheckConstraint("high >= close", name="check_high_close"),
        CheckConstraint("low <= open", name="check_low_open"),
        CheckConstraint("low <= close", name="check_low_close"),
    )

class Portfolio(Base):
    """Portfolio tracking - your digital wallet in the sprawl"""
    __tablename__ = "portfolios"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String(100), nullable=False)
    
    # Portfolio settings
    is_active = Column(Boolean, default=True)
    is_paper_trading = Column(Boolean, default=False)
    initial_balance = Column(DECIMAL(20, 8), default=Decimal("10000"))
    
    # Risk management
    max_position_size = Column(DECIMAL(5, 2), default=Decimal("10"))  # % of portfolio
    stop_loss_default = Column(DECIMAL(5, 2), default=Decimal("2"))  # %
    take_profit_default = Column(DECIMAL(5, 2), default=Decimal("5"))  # %
    
    # Performance tracking
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(DECIMAL(20, 8), default=Decimal("0"))
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    
    # Relationships
    positions = relationship("Position", back_populates="portfolio")
    orders = relationship("Order", back_populates="portfolio")
    
    @hybrid_property
    def win_rate(self):
        if self.total_trades > 0:
            return float(self.winning_trades) / float(self.total_trades)
        return 0.0

class Position(Base):
    """Active positions - your stakes in the digital casino"""
    __tablename__ = "positions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("symbols.id"), nullable=False)
    
    # Position details
    side = Column(String(10), nullable=False)  # long/short
    quantity = Column(DECIMAL(20, 8), nullable=False)
    entry_price = Column(DECIMAL(20, 8), nullable=False)
    current_price = Column(DECIMAL(20, 8), nullable=True)
    
    # Risk management
    stop_loss = Column(DECIMAL(20, 8), nullable=True)
    take_profit = Column(DECIMAL(20, 8), nullable=True)
    trailing_stop_distance = Column(DECIMAL(10, 4), nullable=True)
    
    # P&L tracking
    realized_pnl = Column(DECIMAL(20, 8), default=Decimal("0"))
    unrealized_pnl = Column(DECIMAL(20, 8), default=Decimal("0"))
    fees_paid = Column(DECIMAL(20, 8), default=Decimal("0"))
    
    # Status
    is_open = Column(Boolean, default=True)
    opened_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    closed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # AI tracking
    ai_confidence = Column(DECIMAL(5, 4), nullable=True)
    strategy_name = Column(String(100), nullable=True)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="positions")
    orders = relationship("Order", back_populates="position")
    
    __table_args__ = (
        Index("idx_position_open", "portfolio_id", "is_open"),
        Index("idx_position_symbol", "symbol_id", "is_open"),
    )

class Order(Base):
    """Orders - your commands to the market matrix"""
    __tablename__ = "orders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("symbols.id"), nullable=False)
    position_id = Column(UUID(as_uuid=True), ForeignKey("positions.id"), nullable=True)
    
    # Order details
    exchange_order_id = Column(String(100), nullable=True, unique=True)
    type = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)
    status = Column(String(20), default=OrderStatus.PENDING)
    
    # Quantities and prices
    quantity = Column(DECIMAL(20, 8), nullable=False)
    filled_quantity = Column(DECIMAL(20, 8), default=Decimal("0"))
    price = Column(DECIMAL(20, 8), nullable=True)  # For limit orders
    stop_price = Column(DECIMAL(20, 8), nullable=True)  # For stop orders
    average_fill_price = Column(DECIMAL(20, 8), nullable=True)
    
    # Fees
    fee = Column(DECIMAL(20, 8), default=Decimal("0"))
    fee_asset = Column(String(10), nullable=True)
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), onupdate=func.now())
    filled_at = Column(TIMESTAMP(timezone=True), nullable=True)
    cancelled_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # AI metadata
    ai_signal_strength = Column(String(20), nullable=True)
    ai_confidence = Column(DECIMAL(5, 4), nullable=True)
    strategy_metadata = Column(JSONB, default=dict)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="orders")
    symbol = relationship("Symbol", back_populates="orders")
    position = relationship("Position", back_populates="orders")
    
    __table_args__ = (
        Index("idx_order_status", "portfolio_id", "status"),
        Index("idx_order_created", "created_at"),
        Index("idx_order_exchange", "exchange_order_id"),
    )

class TradingSignal(Base):
    """AI trading signals - wisdom from the machine spirits"""
    __tablename__ = "trading_signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("symbols.id"), nullable=False)
    
    # Signal details
    signal_type = Column(String(20), nullable=False)  # buy/sell/hold
    strength = Column(String(20), nullable=False)
    confidence = Column(DECIMAL(5, 4), nullable=False)
    
    # Price targets
    entry_price = Column(DECIMAL(20, 8), nullable=False)
    stop_loss = Column(DECIMAL(20, 8), nullable=True)
    take_profit_1 = Column(DECIMAL(20, 8), nullable=True)
    take_profit_2 = Column(DECIMAL(20, 8), nullable=True)
    take_profit_3 = Column(DECIMAL(20, 8), nullable=True)
    
    # Model metadata
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20), nullable=False)
    features_used = Column(JSONB, default=list)
    
    # Validation
    is_executed = Column(Boolean, default=False)
    executed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    execution_price = Column(DECIMAL(20, 8), nullable=True)
    
    # Performance tracking
    actual_outcome = Column(String(20), nullable=True)  # profit/loss/neutral
    pnl_percentage = Column(DECIMAL(10, 4), nullable=True)
    
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    __table_args__ = (
        Index("idx_signal_symbol_created", "symbol_id", "created_at"),
        Index("idx_signal_executed", "is_executed", "created_at"),
    )

class BacktestResult(Base):
    """Backtest results - testing strategies in the digital past"""
    __tablename__ = "backtest_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    strategy_name = Column(String(100), nullable=False)
    
    # Test parameters
    start_date = Column(TIMESTAMP(timezone=True), nullable=False)
    end_date = Column(TIMESTAMP(timezone=True), nullable=False)
    initial_capital = Column(DECIMAL(20, 8), nullable=False)
    
    # Results
    final_capital = Column(DECIMAL(20, 8), nullable=False)
    total_return = Column(DECIMAL(10, 4), nullable=False)
    sharpe_ratio = Column(DECIMAL(10, 4), nullable=True)
    sortino_ratio = Column(DECIMAL(10, 4), nullable=True)
    max_drawdown = Column(DECIMAL(10, 4), nullable=True)
    
    # Trade statistics
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    avg_win = Column(DECIMAL(10, 4), nullable=True)
    avg_loss = Column(DECIMAL(10, 4), nullable=True)
    
    # Detailed results
    equity_curve = Column(JSONB, default=list)
    trade_log = Column(JSONB, default=list)
    parameters = Column(JSONB, default=dict)
    
    # Metadata
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    execution_time = Column(DECIMAL(10, 2), nullable=True)  # seconds
    
    __table_args__ = (
        Index("idx_backtest_strategy", "strategy_name", "created_at"),
    )

class SystemMetric(Base):
    """System performance metrics - keeping tabs on the chrome"""
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(DECIMAL(20, 4), nullable=False)
    metric_unit = Column(String(20), nullable=True)
    
    # Context
    component = Column(String(50), nullable=False)  # api, gui, ml, trading
    tags = Column(JSONB, default=dict)
    
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_metric_lookup", "component", "metric_name", "timestamp"),
    )
