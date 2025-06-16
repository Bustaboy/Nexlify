# nexlify/api/main.py
"""
Nexlify API - The Digital Trading Hub
FastAPI-powered, async to the core, secure as a Militech data vault
This is where the street meets the elite
"""

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import json
import logging
from pathlib import Path

# Pydantic models for API
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_
import redis
import websockets

# Our modules - the chrome we built
from config.config_manager import get_config
from security.auth_manager import get_auth_manager, require_auth, JWTBearer
from database.models import (
    Base, Market, Symbol, Candle, Portfolio, Position, Order,
    TradingSignal, BacktestResult, OrderStatus, OrderType, OrderSide
)
from database.migration_manager import get_migration_engine
from monitoring.sentinel import get_sentinel, AlertSeverity, MetricType
from ml.neural_trader import get_neural_orchestrator, NeuralTrader

# Logging setup
logger = logging.getLogger("nexlify.api")

# --- Pydantic Models ---
class UserRegistration(BaseModel):
    """User registration model - your entry to the grid"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    enable_2fa: bool = Field(default=True)

class UserLogin(BaseModel):
    """Login credentials - your access codes"""
    username: str
    password: str
    pin: str
    totp_code: Optional[str] = None

class MarketData(BaseModel):
    """Market data response - the pulse of the market"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades_count: Optional[int] = 0

class PortfolioCreate(BaseModel):
    """Create a new portfolio - your digital wallet"""
    name: str = Field(..., min_length=1, max_length=100)
    initial_balance: float = Field(default=10000, ge=0)
    is_paper_trading: bool = Field(default=True)
    max_position_size: float = Field(default=10, ge=1, le=100)
    stop_loss_default: float = Field(default=2, ge=0.1, le=50)
    take_profit_default: float = Field(default=5, ge=0.1, le=100)

class OrderRequest(BaseModel):
    """Order request - your market command"""
    portfolio_id: str
    symbol: str
    type: OrderType
    side: OrderSide
    quantity: float = Field(..., gt=0)
    price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    
    @validator('price')
    def validate_price(cls, v, values):
        if values.get('type') in [OrderType.LIMIT, OrderType.STOP_LIMIT] and v is None:
            raise ValueError('Price required for limit orders')
        return v

class SignalResponse(BaseModel):
    """AI trading signal response - wisdom from the machine"""
    symbol: str
    action: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    risk_reward_ratio: float
    expected_return: float
    reasoning: Dict[str, Any]
    timestamp: datetime

class BacktestRequest(BaseModel):
    """Backtest request - test your strategies in the past"""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=10000, gt=0)
    position_size: float = Field(default=0.1, gt=0, le=1)

class SystemStatus(BaseModel):
    """System status - health check of the chrome"""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, bool]
    metrics: Dict[str, float]

# --- API Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage API lifecycle - boot up and shutdown sequences
    Like a netrunner's pre-run ritual
    """
    # Startup
    logger.info("Nexlify API initializing - jacking into the market matrix...")
    
    # Initialize services
    config = get_config()
    sentinel = get_sentinel()
    
    # Start monitoring
    await sentinel.start_monitoring()
    
    # Initialize database
    engine = get_migration_engine()
    if not engine._check_database_connection():
        await engine.initialize_database()
    
    # Initialize ML orchestrator
    orchestrator = get_neural_orchestrator()
    
    # WebSocket connection manager
    app.state.ws_manager = ConnectionManager()
    
    logger.info("Nexlify API online - ready to trade in the digital frontier")
    
    yield
    
    # Shutdown
    logger.info("Nexlify API shutting down - going dark...")
    await sentinel.stop_monitoring()
    logger.info("Nexlify API offline - see you in the Net, choom")

# --- FastAPI App ---
app = FastAPI(
    title="Nexlify Trading Platform",
    description="Cyberpunk-themed algorithmic trading platform - where chrome meets the market",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# --- WebSocket Connection Manager ---
class ConnectionManager:
    """
    Manage WebSocket connections - the neural links to our traders
    """
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, channel: str = "general"):
        await websocket.accept()
        
        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        
        self.active_connections[channel].add(websocket)
        self.user_connections[user_id] = websocket
        
        logger.info(f"User {user_id} connected to channel {channel}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        # Remove from all channels
        for channel in self.active_connections:
            self.active_connections[channel].discard(websocket)
        
        self.user_connections.pop(user_id, None)
        logger.info(f"User {user_id} disconnected")
    
    async def send_to_channel(self, message: dict, channel: str):
        """Broadcast to all connections in a channel"""
        if channel in self.active_connections:
            dead_connections = set()
            
            for connection in self.active_connections[channel]:
                try:
                    await connection.send_json(message)
                except:
                    dead_connections.add(connection)
            
            # Clean up dead connections
            self.active_connections[channel] -= dead_connections
    
    async def send_to_user(self, message: dict, user_id: str):
        """Send to specific user"""
        if user_id in self.user_connections:
            try:
                await self.user_connections[user_id].send_json(message)
            except:
                self.user_connections.pop(user_id, None)

# --- Dependency Injection ---
async def get_db() -> Session:
    """Database session dependency - your connection to the data vault"""
    engine = get_migration_engine()
    session = engine.SessionLocal()
    try:
        yield session
    finally:
        session.close()

async def get_current_user(
    request: Request,
    db: Session = Depends(get_db)
) -> str:
    """Get current user from JWT - identity verification"""
    auth = JWTBearer(get_auth_manager(db))
    user_id = await auth(request)
    return user_id

# --- Health & Status Endpoints ---
@app.get("/", response_model=SystemStatus)
async def root():
    """Root endpoint - system status check"""
    config = get_config()
    sentinel = get_sentinel()
    
    return SystemStatus(
        status="online",
        timestamp=datetime.now(timezone.utc),
        version="2.0.0",
        services={
            "api": True,
            "database": get_migration_engine()._check_database_connection(),
            "monitoring": sentinel.is_running,
            "ml_engine": True
        },
        metrics={
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "active_alerts": len(sentinel.active_alerts)
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint - quick pulse check"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

# --- Authentication Endpoints ---
@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegistration,
    db: Session = Depends(get_db)
):
    """
    Register new user - welcome to the grid, choom
    """
    auth_manager = get_auth_manager(db)
    
    try:
        user, pin, totp_uri = await auth_manager.register_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            enable_2fa=user_data.enable_2fa
        )
        
        response = {
            "user_id": user.id,
            "username": user.username,
            "pin": pin,  # Only shown once!
            "message": "Registration successful - save your PIN, it won't be shown again!"
        }
        
        if totp_uri:
            response["totp_uri"] = totp_uri
            response["backup_codes"] = user.backup_codes[:5]  # Show first 5
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed - the matrix rejected you"
        )

@app.post("/auth/login")
async def login(
    credentials: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Login - jack into the system
    """
    auth_manager = get_auth_manager(db)
    
    # Get client IP
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "Unknown")
    
    try:
        tokens = await auth_manager.authenticate(
            username=credentials.username,
            password=credentials.password,
            pin=credentials.pin,
            totp_code=credentials.totp_code,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        return {
            **tokens,
            "message": "Welcome back to the Net"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed - access denied"
        )

@app.post("/auth/refresh")
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """Refresh access token - extend your session"""
    auth_manager = get_auth_manager(db)
    
    try:
        payload = await auth_manager.verify_token(refresh_token)
        
        if payload.get('type') != 'refresh':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Generate new access token
        new_access_token = auth_manager._generate_access_token(payload['user_id'])
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

# --- Market Data Endpoints ---
@app.get("/markets", response_model=List[Dict])
async def get_markets(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get available markets - the trading grounds"""
    markets = db.query(Market).filter(Market.is_active == True).all()
    
    return [
        {
            "id": str(market.id),
            "name": market.name,
            "exchange": market.exchange,
            "maker_fee": float(market.maker_fee),
            "taker_fee": float(market.taker_fee),
            "maintenance_mode": market.maintenance_mode
        }
        for market in markets
    ]

@app.get("/symbols/{market_id}", response_model=List[Dict])
async def get_symbols(
    market_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get trading symbols for a market"""
    symbols = db.query(Symbol).filter(
        and_(
            Symbol.market_id == market_id,
            Symbol.is_trading == True
        )
    ).all()
    
    return [
        {
            "id": str(symbol.id),
            "symbol": symbol.symbol,
            "base_asset": symbol.base_asset,
            "quote_asset": symbol.quote_asset,
            "min_quantity": float(symbol.min_quantity),
            "max_quantity": float(symbol.max_quantity),
            "step_size": float(symbol.step_size),
            "volume_24h": float(symbol.volume_24h) if symbol.volume_24h else 0,
            "price_change_24h": float(symbol.price_change_24h) if symbol.price_change_24h else 0
        }
        for symbol in symbols
    ]

@app.get("/market-data/{symbol_id}/candles", response_model=List[MarketData])
async def get_candles(
    symbol_id: str,
    interval: str = "1h",
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get historical candle data - market memories"""
    candles = db.query(Candle).filter(
        and_(
            Candle.symbol_id == symbol_id,
            Candle.interval == interval
        )
    ).order_by(Candle.timestamp.desc()).limit(limit).all()
    
    # Get symbol for response
    symbol = db.query(Symbol).filter(Symbol.id == symbol_id).first()
    
    return [
        MarketData(
            symbol=symbol.symbol,
            timestamp=candle.timestamp,
            open=float(candle.open),
            high=float(candle.high),
            low=float(candle.low),
            close=float(candle.close),
            volume=float(candle.volume),
            trades_count=candle.trades_count
        )
        for candle in reversed(candles)  # Return in chronological order
    ]

# --- Portfolio Management ---
@app.get("/portfolios", response_model=List[Dict])
async def get_portfolios(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get user portfolios - your digital wallets"""
    portfolios = db.query(Portfolio).filter(
        Portfolio.user_id == user_id
    ).all()
    
    sentinel = get_sentinel()
    
    return [
        {
            "id": str(portfolio.id),
            "name": portfolio.name,
            "is_active": portfolio.is_active,
            "is_paper_trading": portfolio.is_paper_trading,
            "initial_balance": float(portfolio.initial_balance),
            "total_trades": portfolio.total_trades,
            "win_rate": portfolio.win_rate,
            "total_pnl": float(portfolio.total_pnl),
            "created_at": portfolio.created_at
        }
        for portfolio in portfolios
    ]

@app.post("/portfolios", status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Create new portfolio - expand your empire"""
    portfolio = Portfolio(
        user_id=user_id,
        name=portfolio_data.name,
        initial_balance=Decimal(str(portfolio_data.initial_balance)),
        is_paper_trading=portfolio_data.is_paper_trading,
        max_position_size=Decimal(str(portfolio_data.max_position_size)),
        stop_loss_default=Decimal(str(portfolio_data.stop_loss_default)),
        take_profit_default=Decimal(str(portfolio_data.take_profit_default))
    )
    
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    
    logger.info(f"Portfolio created: {portfolio.id} for user {user_id}")
    
    return {
        "id": str(portfolio.id),
        "name": portfolio.name,
        "message": "Portfolio created - ready to trade"
    }

@app.get("/portfolios/{portfolio_id}/positions")
async def get_positions(
    portfolio_id: str,
    include_closed: bool = False,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get portfolio positions - your market stakes"""
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        and_(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id
        )
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    query = db.query(Position).filter(Position.portfolio_id == portfolio_id)
    
    if not include_closed:
        query = query.filter(Position.is_open == True)
    
    positions = query.all()
    
    return [
        {
            "id": str(position.id),
            "symbol": position.symbol.symbol,
            "side": position.side,
            "quantity": float(position.quantity),
            "entry_price": float(position.entry_price),
            "current_price": float(position.current_price) if position.current_price else None,
            "stop_loss": float(position.stop_loss) if position.stop_loss else None,
            "take_profit": float(position.take_profit) if position.take_profit else None,
            "unrealized_pnl": float(position.unrealized_pnl),
            "realized_pnl": float(position.realized_pnl),
            "is_open": position.is_open,
            "opened_at": position.opened_at
        }
        for position in positions
    ]

# --- Trading Endpoints ---
@app.post("/orders")
async def place_order(
    order_request: OrderRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Place order - execute your market command"""
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        and_(
            Portfolio.id == order_request.portfolio_id,
            Portfolio.user_id == user_id
        )
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    # Get symbol
    symbol = db.query(Symbol).filter(Symbol.symbol == order_request.symbol).first()
    
    if not symbol:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Symbol not found"
        )
    
    # Create order
    order = Order(
        portfolio_id=portfolio.id,
        symbol_id=symbol.id,
        type=order_request.type,
        side=order_request.side,
        quantity=Decimal(str(order_request.quantity)),
        price=Decimal(str(order_request.price)) if order_request.price else None,
        stop_price=Decimal(str(order_request.stop_price)) if order_request.stop_price else None,
        status=OrderStatus.PENDING
    )
    
    db.add(order)
    db.commit()
    db.refresh(order)
    
    # In a real system, this would send to the exchange
    # For now, simulate order processing
    asyncio.create_task(_process_order(order.id, db))
    
    # Log order
    sentinel = get_sentinel()
    sentinel.orders_placed.labels(
        type=order_request.type.value,
        side=order_request.side.value
    ).inc()
    
    # Send WebSocket update
    if hasattr(app.state, 'ws_manager'):
        await app.state.ws_manager.send_to_user(
            {
                "type": "order_update",
                "order_id": str(order.id),
                "status": order.status.value,
                "message": f"Order placed: {order_request.side.value} {order_request.quantity} {order_request.symbol}"
            },
            user_id
        )
    
    return {
        "order_id": str(order.id),
        "status": order.status.value,
        "message": "Order placed - executing in the digital market"
    }

async def _process_order(order_id: str, db: Session):
    """Process order - simulate exchange execution"""
    await asyncio.sleep(1)  # Simulate network delay
    
    # In production, this would interact with exchange APIs
    # For now, simulate execution
    order = db.query(Order).filter(Order.id == order_id).first()
    
    if order:
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = order.price or Decimal("50000")  # Mock price
        order.filled_at = datetime.now(timezone.utc)
        
        db.commit()

@app.get("/orders/{portfolio_id}")
async def get_orders(
    portfolio_id: str,
    status: Optional[OrderStatus] = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get portfolio orders - your market commands history"""
    # Verify portfolio ownership
    portfolio = db.query(Portfolio).filter(
        and_(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id
        )
    ).first()
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    query = db.query(Order).filter(Order.portfolio_id == portfolio_id)
    
    if status:
        query = query.filter(Order.status == status)
    
    orders = query.order_by(Order.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": str(order.id),
            "symbol": order.symbol.symbol,
            "type": order.type.value,
            "side": order.side.value,
            "status": order.status.value,
            "quantity": float(order.quantity),
            "filled_quantity": float(order.filled_quantity),
            "price": float(order.price) if order.price else None,
            "average_fill_price": float(order.average_fill_price) if order.average_fill_price else None,
            "created_at": order.created_at,
            "filled_at": order.filled_at
        }
        for order in orders
    ]

# --- AI Trading Endpoints ---
@app.post("/ai/signal/{symbol}", response_model=SignalResponse)
async def get_ai_signal(
    symbol: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Get AI trading signal - consult the oracle"""
    # Get recent candles
    symbol_obj = db.query(Symbol).filter(Symbol.symbol == symbol).first()
    
    if not symbol_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Symbol not found"
        )
    
    # Get recent candles for prediction
    candles = db.query(Candle).filter(
        and_(
            Candle.symbol_id == symbol_obj.id,
            Candle.interval == "1h"
        )
    ).order_by(Candle.timestamp.desc()).limit(200).all()
    
    if len(candles) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient historical data for prediction"
        )
    
    # Get neural trader
    orchestrator = get_neural_orchestrator()
    trader = orchestrator.get_trader(symbol)
    
    if not trader or not trader.is_trained:
        # Try to load pre-trained model
        trader = NeuralTrader(symbol)
        try:
            trader.load_checkpoint()
        except:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI model not available for this symbol"
            )
    
    # Generate prediction
    try:
        current_price = float(candles[0].close)
        signal = await trader.predict(list(reversed(candles)), current_price)
        
        # Store signal in database
        db_signal = TradingSignal(
            symbol_id=symbol_obj.id,
            signal_type=signal.action,
            strength="strong" if signal.confidence > 0.8 else "moderate",
            confidence=Decimal(str(signal.confidence)),
            entry_price=Decimal(str(signal.entry_price)),
            stop_loss=Decimal(str(signal.stop_loss)),
            take_profit_1=Decimal(str(signal.take_profit[0])),
            take_profit_2=Decimal(str(signal.take_profit[1])) if len(signal.take_profit) > 1 else None,
            take_profit_3=Decimal(str(signal.take_profit[2])) if len(signal.take_profit) > 2 else None,
            model_name=trader.model_name,
            model_version="1.0",
            features_used=signal.reasoning
        )
        
        db.add(db_signal)
        db.commit()
        
        return SignalResponse(
            symbol=symbol,
            action=signal.action,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward_ratio=signal.risk_reward_ratio,
            expected_return=signal.expected_return,
            reasoning=signal.reasoning,
            timestamp=signal.timestamp
        )
        
    except Exception as e:
        logger.error(f"AI prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI prediction failed - the oracle is silent"
        )

@app.post("/ai/backtest")
async def run_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """Run AI backtest - test strategies in the digital past"""
    # Get historical data
    symbol_obj = db.query(Symbol).filter(Symbol.symbol == request.symbol).first()
    
    if not symbol_obj:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Symbol not found"
        )
    
    # Get candles in date range
    candles = db.query(Candle).filter(
        and_(
            Candle.symbol_id == symbol_obj.id,
            Candle.interval == "1h",
            Candle.timestamp >= request.start_date,
            Candle.timestamp <= request.end_date
        )
    ).order_by(Candle.timestamp).all()
    
    if len(candles) < 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient data for backtesting"
        )
    
    # Get neural trader
    trader = NeuralTrader(request.symbol)
    try:
        trader.load_checkpoint()
    except:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI model not available for backtesting"
        )
    
    # Run backtest
    try:
        results = await trader.backtest(
            candles,
            request.initial_capital,
            request.position_size
        )
        
        # Store results
        backtest = BacktestResult(
            strategy_name=f"NeuralTrader_{request.symbol}",
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=Decimal(str(request.initial_capital)),
            final_capital=Decimal(str(results['final_capital'])),
            total_return=Decimal(str(results['total_return'])),
            sharpe_ratio=Decimal(str(results['sharpe_ratio'])),
            sortino_ratio=Decimal(str(results['sortino_ratio'])),
            max_drawdown=Decimal(str(results['max_drawdown'])),
            total_trades=results['total_trades'],
            winning_trades=results['winning_trades'],
            losing_trades=results['losing_trades'],
            avg_win=Decimal(str(results['avg_win'])),
            avg_loss=Decimal(str(results['avg_loss'])),
            equity_curve=results['equity_curve'],
            trade_log=results['trades']
        )
        
        db.add(backtest)
        db.commit()
        
        return {
            "backtest_id": str(backtest.id),
            "total_return": f"{results['total_return']:.2%}",
            "sharpe_ratio": results['sharpe_ratio'],
            "max_drawdown": f"{results['max_drawdown']:.2%}",
            "win_rate": f"{results['win_rate']:.2%}",
            "total_trades": results['total_trades'],
            "profit_factor": results['profit_factor']
        }
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backtest failed - the time machine malfunctioned"
        )

# --- Monitoring Endpoints ---
@app.get("/monitoring/metrics")
async def get_metrics(
    user_id: str = Depends(get_current_user)
):
    """Get system metrics - the vital signs"""
    sentinel = get_sentinel()
    
    return {
        "system": {
            "cpu_usage": sentinel.cpu_usage._value.get(),
            "memory_usage": sentinel.memory_usage._value.get(),
            "disk_usage": sentinel.disk_usage._value.get()
        },
        "trading": {
            "active_positions": sentinel.active_positions._value.get(),
            "total_pnl": sentinel.total_pnl._value.get()
        },
        "alerts": [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "title": alert.title,
                "timestamp": alert.timestamp
            }
            for alert in sentinel.active_alerts.values()
        ]
    }

@app.get("/monitoring/alerts")
async def get_alerts(
    include_resolved: bool = False,
    user_id: str = Depends(get_current_user)
):
    """Get system alerts - the warning signals"""
    sentinel = get_sentinel()
    
    alerts = list(sentinel.active_alerts.values())
    
    return [
        {
            "id": alert.id,
            "severity": alert.severity.value,
            "type": alert.metric_type.value,
            "title": alert.title,
            "description": alert.description,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp,
            "component": alert.component,
            "resolved": alert.resolved
        }
        for alert in alerts
        if include_resolved or not alert.resolved
    ]

# --- WebSocket Endpoints ---
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    db: Session = Depends(get_db)
):
    """
    WebSocket connection - real-time neural link
    Stream market data, alerts, and updates
    """
    # Verify user exists
    auth_manager = get_auth_manager(db)
    
    # Connect
    await app.state.ws_manager.connect(websocket, user_id, "general")
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "subscribe":
                channel = data.get("channel", "general")
                await app.state.ws_manager.connect(websocket, user_id, channel)
                
                await websocket.send_json({
                    "type": "subscribed",
                    "channel": channel,
                    "message": f"Subscribed to {channel}"
                })
            
            elif data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    
    except WebSocketDisconnect:
        app.state.ws_manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")

# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions - when things go sideways"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions - when the matrix glitches"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error - the matrix has you",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url)
        }
    )

# --- Startup Tasks ---
@app.on_event("startup")
async def startup_tasks():
    """Startup tasks - boot sequence"""
    logger.info("Running startup tasks...")
    
    # Initialize monitoring metrics
    sentinel = get_sentinel()
    
    # Start background tasks
    asyncio.create_task(_market_data_streamer())
    asyncio.create_task(_alert_broadcaster())

async def _market_data_streamer():
    """Stream market data to WebSocket clients"""
    while True:
        try:
            # In production, this would connect to real market data feeds
            # For now, simulate updates
            await asyncio.sleep(5)
            
            # Broadcast mock market update
            if hasattr(app.state, 'ws_manager'):
                await app.state.ws_manager.send_to_channel(
                    {
                        "type": "market_update",
                        "symbol": "BTC/USDT",
                        "price": 50000 + random.uniform(-1000, 1000),
                        "volume": random.uniform(100, 1000),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    "market_data"
                )
        
        except Exception as e:
            logger.error(f"Market data streamer error: {e}")
            await asyncio.sleep(30)

async def _alert_broadcaster():
    """Broadcast alerts to WebSocket clients"""
    sentinel = get_sentinel()
    
    async def broadcast_alert(alert):
        if hasattr(app.state, 'ws_manager'):
            await app.state.ws_manager.send_to_channel(
                {
                    "type": "alert",
                    "alert": {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat()
                    }
                },
                "alerts"
            )
    
    sentinel.add_alert_handler(broadcast_alert)

# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
