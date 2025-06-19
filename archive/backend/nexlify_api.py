"""
Nexlify API Backend - Neural Cortex for Electron Frontend
FastAPI with WebSocket support for real-time trading
"""

import os
import sys
import json
import asyncio
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import socketio
from pydantic import BaseModel, Field, validator
import jwt
from passlib.context import CryptContext
import pyotp
import aioredis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Import existing Nexlify modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.nexlify_neural_net import NexlifyNeuralNet
from src.security.nexlify_advanced_security import SecurityManager
from src.audit.nexlify_audit_trail import AuditManager
from src.strategies.nexlify_multi_strategy import MultiStrategyOptimizer
from src.ml.nexlify_predictive_features import PredictiveEngine
from src.trading.nexlify_dex_integration import DexIntegration
from src.utils.error_handler import get_error_handler
from src.utils.utils_module import ValidationUtils, NetworkUtils

# Configuration
API_VERSION = "v1"
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
PIN_MAX_ATTEMPTS = 5
PIN_LOCKOUT_MINUTES = 5

# Logging
logger = logging.getLogger("nexlify.api")
error_handler = get_error_handler()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Socket.io server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",  # Configure properly in production
    logger=True,
    engineio_logger=True
)

# Database setup (async SQLAlchemy)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./data/nexlify.db")
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Redis for caching and rate limiting
redis_client = None

# Core components (initialized on startup)
neural_net = None
security_manager = None
audit_manager = None
strategy_optimizer = None
predictive_engine = None
dex_integration = None

# Active connections tracking
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        
    async def connect(self, websocket: WebSocket, user_id: str, connection_id: str):
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(connection_id)
        
        logger.info(f"User {user_id} connected with {connection_id}")
        
    def disconnect(self, user_id: str, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(connection_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
                
        logger.info(f"User {user_id} disconnected {connection_id}")
        
    async def send_to_user(self, user_id: str, message: dict):
        """Send message to all connections of a user"""
        if user_id in self.user_sessions:
            for conn_id in self.user_sessions[user_id]:
                if conn_id in self.active_connections:
                    try:
                        await self.active_connections[conn_id].send_json(message)
                    except Exception as e:
                        logger.error(f"Error sending to {conn_id}: {e}")
                        
    async def broadcast(self, message: dict, exclude_user: Optional[str] = None):
        """Broadcast to all connected users"""
        for user_id, connections in self.user_sessions.items():
            if exclude_user and user_id == exclude_user:
                continue
            await self.send_to_user(user_id, message)

manager = ConnectionManager()

# Pydantic models
class PinAuthRequest(BaseModel):
    pin: str = Field(..., min_length=4, max_length=8)
    device_id: str
    enable_2fa: bool = False

class TwoFactorRequest(BaseModel):
    token: str
    code: str = Field(..., min_length=6, max_length=6)

class UserSession(BaseModel):
    user_id: str
    device_id: str
    pin_hash: str
    totp_secret: Optional[str] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

class TradingSignal(BaseModel):
    strategy: str
    symbol: str
    action: str  # buy/sell/hold
    confidence: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

class OrderRequest(BaseModel):
    exchange: str
    symbol: str
    side: str  # buy/sell
    order_type: str  # market/limit
    amount: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class SystemStatus(BaseModel):
    status: str  # online/degraded/offline
    uptime: float
    active_strategies: List[str]
    connected_exchanges: List[str]
    websocket_connections: int
    cpu_usage: float
    memory_usage: float
    last_trade: Optional[datetime]
    profit_24h: float
    active_positions: int

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    logger.info("🚀 Nexlify API starting up...")
    
    global neural_net, security_manager, audit_manager
    global strategy_optimizer, predictive_engine, dex_integration, redis_client
    
    try:
        # Initialize Redis
        redis_client = await aioredis.create_redis_pool(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            encoding="utf-8"
        )
        
        # Initialize core components
        neural_net = NexlifyNeuralNet()
        security_manager = SecurityManager()
        audit_manager = AuditManager()
        strategy_optimizer = MultiStrategyOptimizer()
        predictive_engine = PredictiveEngine()
        dex_integration = DexIntegration()
        
        # Load configuration
        await neural_net.initialize()
        
        logger.info("✅ All systems operational")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        error_handler.log_critical_error(e, "api_startup")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Nexlify API shutting down...")
    
    try:
        # Close connections
        if redis_client:
            redis_client.close()
            await redis_client.wait_closed()
            
        # Stop trading
        if neural_net:
            await neural_net.shutdown()
            
        logger.info("✅ Shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI app
app = FastAPI(
    title="Nexlify Trading Matrix API",
    description="Neural-net powered trading system with real-time capabilities",
    version="3.0.0",
    lifespan=lifespan
)

# Add Socket.io to FastAPI
socket_app = socketio.ASGIApp(sio, app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "app://nexlify"],  # Electron app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
            
        # Check if session exists in Redis
        if redis_client:
            session_data = await redis_client.get(f"session:{user_id}")
            if not session_data:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Session expired"
                )
                
        return user_id
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Authentication endpoints
@app.post("/api/v1/auth/pin", response_model=dict)
async def authenticate_pin(request: PinAuthRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate with PIN"""
    try:
        # Check rate limiting
        if redis_client:
            attempts_key = f"pin_attempts:{request.device_id}"
            attempts = await redis_client.get(attempts_key)
            
            if attempts and int(attempts) >= PIN_MAX_ATTEMPTS:
                lockout_key = f"pin_lockout:{request.device_id}"
                lockout = await redis_client.get(lockout_key)
                
                if lockout:
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Too many attempts. Locked for {PIN_LOCKOUT_MINUTES} minutes"
                    )
        
        # Verify PIN (this would check against stored user sessions)
        # For now, using a secure default that should be changed
        correct_pin_hash = pwd_context.hash("2077")  # This should come from DB
        
        if not pwd_context.verify(request.pin, correct_pin_hash):
            # Increment failed attempts
            if redis_client:
                attempts = await redis_client.incr(attempts_key)
                await redis_client.expire(attempts_key, 300)  # 5 min expiry
                
                if attempts >= PIN_MAX_ATTEMPTS:
                    await redis_client.setex(
                        f"pin_lockout:{request.device_id}",
                        PIN_LOCKOUT_MINUTES * 60,
                        "locked"
                    )
                    
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid PIN"
            )
        
        # Generate session
        user_id = f"user_{request.device_id}"  # In production, lookup real user
        
        # Create JWT token
        expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
        token_data = {
            "sub": user_id,
            "exp": expire,
            "device_id": request.device_id
        }
        
        token = jwt.encode(token_data, JWT_SECRET, algorithm=JWT_ALGORITHM)
        
        # Store session in Redis
        if redis_client:
            session_data = {
                "device_id": request.device_id,
                "created_at": datetime.utcnow().isoformat(),
                "2fa_enabled": request.enable_2fa
            }
            await redis_client.setex(
                f"session:{user_id}",
                JWT_EXPIRATION_HOURS * 3600,
                json.dumps(session_data)
            )
            
            # Clear failed attempts
            await redis_client.delete(attempts_key)
            await redis_client.delete(f"pin_lockout:{request.device_id}")
        
        # Audit log
        await audit_manager.log_event(
            "auth.pin_success",
            user_id,
            {"device_id": request.device_id}
        )
        
        response = {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": JWT_EXPIRATION_HOURS * 3600,
            "requires_2fa": request.enable_2fa
        }
        
        if request.enable_2fa:
            # Generate TOTP secret
            totp_secret = pyotp.random_base32()
            totp_uri = pyotp.totp.TOTP(totp_secret).provisioning_uri(
                name=user_id,
                issuer_name="Nexlify Trading"
            )
            
            # Store secret temporarily
            if redis_client:
                await redis_client.setex(
                    f"2fa_setup:{user_id}",
                    300,  # 5 min to complete 2FA setup
                    totp_secret
                )
            
            response["totp_uri"] = totp_uri
            response["totp_secret"] = totp_secret
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.log_error(e, "auth_pin")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@app.post("/api/v1/auth/2fa/verify", response_model=dict)
async def verify_2fa(request: TwoFactorRequest):
    """Verify 2FA code"""
    try:
        # Decode token to get user_id
        payload = jwt.decode(request.token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Get TOTP secret from Redis
        if redis_client:
            totp_secret = await redis_client.get(f"2fa_setup:{user_id}")
            
            if not totp_secret:
                # Check if user already has 2FA setup (would be in DB)
                # For now, using a test secret
                totp_secret = "JBSWY3DPEHPK3PXP"  # Should come from DB
        
        # Verify TOTP code
        totp = pyotp.TOTP(totp_secret)
        if not totp.verify(request.code, valid_window=1):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA code"
            )
        
        # Mark session as 2FA verified
        if redis_client:
            session_key = f"session:{user_id}"
            session_data = await redis_client.get(session_key)
            
            if session_data:
                session = json.loads(session_data)
                session["2fa_verified"] = True
                session["2fa_verified_at"] = datetime.utcnow().isoformat()
                
                await redis_client.setex(
                    session_key,
                    JWT_EXPIRATION_HOURS * 3600,
                    json.dumps(session)
                )
                
            # Clean up setup key
            await redis_client.delete(f"2fa_setup:{user_id}")
        
        # Audit log
        await audit_manager.log_event(
            "auth.2fa_success",
            user_id,
            {}
        )
        
        return {
            "status": "verified",
            "message": "2FA verification successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.log_error(e, "verify_2fa")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="2FA verification failed"
        )

# System status endpoint
@app.get("/api/v1/system/status", response_model=SystemStatus)
async def get_system_status(user_id: str = Depends(verify_token)):
    """Get system status and health metrics"""
    try:
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Get trading metrics
        active_strategies = strategy_optimizer.get_active_strategies() if strategy_optimizer else []
        connected_exchanges = neural_net.get_connected_exchanges() if neural_net else []
        
        # Get WebSocket connections
        total_connections = sum(len(conns) for conns in manager.user_sessions.values())
        
        # Get trading performance (would come from DB)
        last_trade = None
        profit_24h = 0.0
        active_positions = 0
        
        if neural_net:
            stats = await neural_net.get_stats_24h()
            last_trade = stats.get("last_trade")
            profit_24h = stats.get("profit_24h", 0.0)
            active_positions = stats.get("active_positions", 0)
        
        return SystemStatus(
            status="online" if neural_net else "offline",
            uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
            active_strategies=active_strategies,
            connected_exchanges=connected_exchanges,
            websocket_connections=total_connections,
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            last_trade=last_trade,
            profit_24h=profit_24h,
            active_positions=active_positions
        )
        
    except Exception as e:
        error_handler.log_error(e, "system_status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system status"
        )

# Trading endpoints
@app.post("/api/v1/trading/order", response_model=dict)
async def place_order(
    order: OrderRequest,
    user_id: str = Depends(verify_token),
    db: AsyncSession = Depends(get_db)
):
    """Place a trading order"""
    try:
        # Validate order
        if not ValidationUtils.validate_symbol(order.symbol):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid trading symbol"
            )
        
        # Check risk limits
        risk_check = await neural_net.check_risk_limits(
            order.symbol,
            order.amount,
            order.side
        )
        
        if not risk_check["allowed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Order rejected: {risk_check['reason']}"
            )
        
        # Place order
        result = await neural_net.place_order(
            exchange=order.exchange,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            amount=order.amount,
            price=order.price,
            params={
                "stop_loss": order.stop_loss,
                "take_profit": order.take_profit
            }
        )
        
        # Audit log
        await audit_manager.log_event(
            "trading.order_placed",
            user_id,
            {
                "order_id": result["id"],
                "exchange": order.exchange,
                "symbol": order.symbol,
                "side": order.side,
                "amount": order.amount,
                "price": order.price or result.get("price")
            }
        )
        
        # Send real-time update
        await manager.send_to_user(user_id, {
            "event": "order_placed",
            "data": result
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_handler.log_error(e, "place_order", {"order": order.dict()})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to place order"
        )

@app.get("/api/v1/trading/positions", response_model=list)
async def get_positions(user_id: str = Depends(verify_token)):
    """Get current open positions"""
    try:
        positions = await neural_net.get_open_positions()
        
        # Enrich with P&L calculations
        for position in positions:
            current_price = await neural_net.get_current_price(
                position["symbol"],
                position["exchange"]
            )
            
            # Calculate unrealized P&L
            if position["side"] == "long":
                position["unrealized_pnl"] = (current_price - position["entry_price"]) * position["amount"]
                position["unrealized_pnl_percent"] = ((current_price - position["entry_price"]) / position["entry_price"]) * 100
            else:
                position["unrealized_pnl"] = (position["entry_price"] - current_price) * position["amount"]
                position["unrealized_pnl_percent"] = ((position["entry_price"] - current_price) / position["entry_price"]) * 100
            
            position["current_price"] = current_price
        
        return positions
        
    except Exception as e:
        error_handler.log_error(e, "get_positions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get positions"
        )

@app.get("/api/v1/trading/signals", response_model=List[TradingSignal])
async def get_trading_signals(
    limit: int = 10,
    user_id: str = Depends(verify_token)
):
    """Get latest trading signals from strategies"""
    try:
        signals = await strategy_optimizer.get_latest_signals(limit)
        
        return [
            TradingSignal(
                strategy=signal["strategy"],
                symbol=signal["symbol"],
                action=signal["action"],
                confidence=signal["confidence"],
                price=signal["price"],
                timestamp=signal["timestamp"],
                metadata=signal.get("metadata", {})
            )
            for signal in signals
        ]
        
    except Exception as e:
        error_handler.log_error(e, "get_signals")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trading signals"
        )

# Market data endpoints
@app.get("/api/v1/market/prices", response_model=dict)
async def get_market_prices(
    symbols: str,  # Comma-separated list
    user_id: str = Depends(verify_token)
):
    """Get current market prices"""
    try:
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        prices = {}
        for symbol in symbol_list:
            price_data = await neural_net.get_ticker(symbol)
            prices[symbol] = {
                "bid": price_data["bid"],
                "ask": price_data["ask"],
                "last": price_data["last"],
                "volume_24h": price_data["volume"],
                "change_24h": price_data["percentage"]
            }
        
        return prices
        
    except Exception as e:
        error_handler.log_error(e, "get_prices")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get market prices"
        )

# WebSocket endpoint for real-time data
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket connection for real-time updates"""
    connection_id = f"{user_id}_{secrets.token_urlsafe(8)}"
    
    try:
        # Verify token from query params
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Verify token
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            if payload.get("sub") != user_id:
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
        except jwt.JWTError:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Connect
        await manager.connect(websocket, user_id, connection_id)
        
        # Send initial data
        await websocket.send_json({
            "event": "connected",
            "data": {
                "connection_id": connection_id,
                "server_time": datetime.utcnow().isoformat()
            }
        })
        
        # Start market data stream
        async def stream_market_data():
            while True:
                try:
                    # Get active symbols from user preferences
                    symbols = ["BTC/USDT", "ETH/USDT"]  # Would come from user settings
                    
                    market_data = {}
                    for symbol in symbols:
                        ticker = await neural_net.get_ticker(symbol)
                        market_data[symbol] = {
                            "price": ticker["last"],
                            "change": ticker["percentage"],
                            "volume": ticker["volume"]
                        }
                    
                    await websocket.send_json({
                        "event": "market_update",
                        "data": market_data
                    })
                    
                    await asyncio.sleep(1)  # Update every second
                    
                except Exception as e:
                    logger.error(f"Market data stream error: {e}")
                    break
        
        # Start streaming task
        stream_task = asyncio.create_task(stream_market_data())
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data["type"] == "subscribe":
                # Handle subscription requests
                pass
            elif data["type"] == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if 'stream_task' in locals():
            stream_task.cancel()
        manager.disconnect(user_id, connection_id)

# Socket.io events for alternative real-time communication
@sio.event
async def connect(sid, environ, auth):
    """Socket.io connection handler"""
    logger.info(f"Socket.io client connected: {sid}")

@sio.event
async def disconnect(sid):
    """Socket.io disconnection handler"""
    logger.info(f"Socket.io client disconnected: {sid}")

@sio.event
async def subscribe_market(sid, data):
    """Subscribe to market data updates"""
    symbols = data.get("symbols", [])
    
    # Add to room for each symbol
    for symbol in symbols:
        sio.enter_room(sid, f"market:{symbol}")
    
    await sio.emit("subscribed", {"symbols": symbols}, to=sid)

# Background tasks
async def broadcast_market_updates():
    """Broadcast market updates to Socket.io rooms"""
    while True:
        try:
            # Get all active symbols
            active_symbols = ["BTC/USDT", "ETH/USDT"]  # Would be dynamic
            
            for symbol in active_symbols:
                ticker = await neural_net.get_ticker(symbol)
                
                await sio.emit(
                    "market_update",
                    {
                        "symbol": symbol,
                        "price": ticker["last"],
                        "change": ticker["percentage"],
                        "volume": ticker["volume"],
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    room=f"market:{symbol}"
                )
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Market broadcast error: {e}")
            await asyncio.sleep(5)

# Start background tasks on startup
@app.on_event("startup")
async def startup_tasks():
    app.state.start_time = time.time()
    asyncio.create_task(broadcast_market_updates())

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "nexlify_api:socket_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
