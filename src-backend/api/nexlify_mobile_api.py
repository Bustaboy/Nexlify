"""
Nexlify Enhanced - Mobile Companion API
Implements Feature 6: Mobile app backend with real-time updates and remote control
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
from datetime import datetime, timedelta
import jwt
import qrcode
import io
import base64
from dataclasses import dataclass
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mobile", tags=["mobile"])
security = HTTPBearer()

# WebSocket connection manager
class MobileConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.device_sessions: Dict[str, Dict] = {}
        
    async def connect(self, websocket: WebSocket, device_id: str):
        await websocket.accept()
        self.active_connections[device_id] = websocket
        logger.info(f"Mobile device connected: {device_id}")
        
    def disconnect(self, device_id: str):
        if device_id in self.active_connections:
            del self.active_connections[device_id]
        logger.info(f"Mobile device disconnected: {device_id}")
        
    async def send_to_device(self, device_id: str, message: dict):
        if device_id in self.active_connections:
            await self.active_connections[device_id].send_json(message)
            
    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        disconnected = []
        for device_id, connection in self.active_connections.items():
            if device_id != exclude:
                try:
                    await connection.send_json(message)
                except:
                    disconnected.append(device_id)
                    
        # Clean up disconnected
        for device_id in disconnected:
            self.disconnect(device_id)

manager = MobileConnectionManager()

@dataclass
class MobileNotification:
    """Push notification structure"""
    id: str
    type: str  # trade, alert, achievement, system
    title: str
    message: str
    data: Dict
    timestamp: datetime
    priority: str = "normal"  # low, normal, high, critical

class MobileAuthManager:
    """Handles mobile device authentication and pairing"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.paired_devices = {}
        self.pending_pairs = {}
        
    def generate_pairing_qr(self, desktop_id: str) -> str:
        """Generate QR code for mobile pairing"""
        # Create pairing token
        pairing_data = {
            'desktop_id': desktop_id,
            'timestamp': datetime.now().isoformat(),
            'pairing_code': str(uuid.uuid4())[:8]
        }
        
        self.pending_pairs[pairing_data['pairing_code']] = pairing_data
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(json.dumps(pairing_data))
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        # Return base64 encoded image
        return base64.b64encode(buffer.getvalue()).decode()
        
    def complete_pairing(self, pairing_code: str, device_info: Dict) -> str:
        """Complete device pairing and return auth token"""
        if pairing_code not in self.pending_pairs:
            raise ValueError("Invalid pairing code")
            
        pairing_data = self.pending_pairs[pairing_code]
        
        # Check if pairing is still valid (5 minutes)
        created = datetime.fromisoformat(pairing_data['timestamp'])
        if datetime.now() - created > timedelta(minutes=5):
            del self.pending_pairs[pairing_code]
            raise ValueError("Pairing code expired")
            
        # Create device session
        device_id = str(uuid.uuid4())
        device_session = {
            'device_id': device_id,
            'desktop_id': pairing_data['desktop_id'],
            'device_info': device_info,
            'paired_at': datetime.now().isoformat()
        }
        
        self.paired_devices[device_id] = device_session
        del self.pending_pairs[pairing_code]
        
        # Generate JWT token
        token_data = {
            'device_id': device_id,
            'desktop_id': pairing_data['desktop_id'],
            'exp': datetime.utcnow() + timedelta(days=30)
        }
        
        token = jwt.encode(token_data, self.secret_key, algorithm='HS256')
        return token
        
    def verify_token(self, token: str) -> Dict:
        """Verify mobile auth token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

# Initialize auth manager
auth_manager = MobileAuthManager(secret_key="your-secret-key-here")

# Dependency to verify mobile authentication
async def verify_mobile_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    return auth_manager.verify_token(token)

# REST API Endpoints

@router.get("/pairing/qr")
async def generate_pairing_qr(desktop_id: str):
    """Generate QR code for mobile device pairing"""
    qr_image = auth_manager.generate_pairing_qr(desktop_id)
    return {
        "qr_code": f"data:image/png;base64,{qr_image}",
        "expires_in": 300  # 5 minutes
    }

@router.post("/pairing/complete")
async def complete_pairing(pairing_code: str, device_info: Dict):
    """Complete mobile device pairing"""
    try:
        token = auth_manager.complete_pairing(pairing_code, device_info)
        return {
            "success": True,
            "token": token,
            "device_id": auth_manager.paired_devices[list(auth_manager.paired_devices.keys())[-1]]['device_id']
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboard")
async def get_mobile_dashboard(auth: Dict = Depends(verify_mobile_auth)):
    """Get dashboard data for mobile app"""
    # This would fetch from the trading engine
    return {
        "portfolio": {
            "total_value": 50000.00,
            "daily_pnl": 1250.50,
            "daily_pnl_percent": 2.51,
            "total_pnl": 15000.00,
            "total_pnl_percent": 30.00
        },
        "active_positions": [
            {
                "symbol": "BTC/USDT",
                "side": "long",
                "size": 0.5,
                "entry_price": 45000,
                "current_price": 46500,
                "pnl": 750.00,
                "pnl_percent": 3.33
            }
        ],
        "recent_trades": [
            {
                "symbol": "ETH/USDT",
                "side": "buy",
                "price": 3200,
                "size": 2.0,
                "timestamp": datetime.now().isoformat(),
                "profit": 150.00
            }
        ],
        "alerts": [
            {
                "type": "price",
                "message": "BTC approaching resistance at $47,000",
                "severity": "medium"
            }
        ]
    }

@router.get("/positions")
async def get_positions(auth: Dict = Depends(verify_mobile_auth)):
    """Get current positions"""
    return {
        "positions": [
            {
                "id": "pos_123",
                "symbol": "BTC/USDT",
                "side": "long",
                "size": 0.5,
                "entry_price": 45000,
                "current_price": 46500,
                "stop_loss": 44000,
                "take_profit": 48000,
                "pnl": 750.00,
                "pnl_percent": 3.33,
                "opened_at": datetime.now().isoformat()
            }
        ]
    }

@router.post("/trade/execute")
async def execute_trade(
    trade_request: Dict,
    auth: Dict = Depends(verify_mobile_auth)
):
    """Execute trade from mobile"""
    # Validate trade request
    required_fields = ['symbol', 'side', 'size']
    if not all(field in trade_request for field in required_fields):
        raise HTTPException(status_code=400, detail="Missing required fields")
        
    # This would execute through the trading engine
    logger.info(f"Mobile trade request: {trade_request}")
    
    # Simulate execution
    return {
        "success": True,
        "trade_id": str(uuid.uuid4()),
        "executed_at": datetime.now().isoformat(),
        "details": {
            "symbol": trade_request['symbol'],
            "side": trade_request['side'],
            "size": trade_request['size'],
            "price": 45000,  # Would be actual execution price
            "commission": 5.00
        }
    }

@router.post("/emergency/stop")
async def emergency_stop(auth: Dict = Depends(verify_mobile_auth)):
    """Emergency stop all trading"""
    logger.warning(f"Emergency stop triggered from mobile device: {auth['device_id']}")
    
    # This would trigger the kill switch in the main engine
    return {
        "success": True,
        "message": "Emergency stop activated",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/settings")
async def get_settings(auth: Dict = Depends(verify_mobile_auth)):
    """Get current trading settings"""
    return {
        "risk_level": "moderate",
        "max_positions": 10,
        "auto_trading": True,
        "notifications": {
            "trades": True,
            "alerts": True,
            "achievements": True
        }
    }

@router.put("/settings")
async def update_settings(
    settings: Dict,
    auth: Dict = Depends(verify_mobile_auth)
):
    """Update trading settings from mobile"""
    logger.info(f"Settings update from mobile: {settings}")
    
    # This would update the main configuration
    return {
        "success": True,
        "updated_settings": settings
    }

# WebSocket endpoint for real-time updates
@router.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """WebSocket connection for real-time updates"""
    await manager.connect(websocket, device_id)
    
    try:
        # Send initial connection success
        await manager.send_to_device(device_id, {
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive messages from mobile
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "ping":
                await manager.send_to_device(device_id, {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                
            elif data.get("type") == "subscribe":
                # Subscribe to specific data streams
                channels = data.get("channels", [])
                logger.info(f"Mobile device {device_id} subscribed to: {channels}")
                
            elif data.get("type") == "command":
                # Handle commands from mobile
                command = data.get("command")
                logger.info(f"Mobile command: {command} from {device_id}")
                
    except WebSocketDisconnect:
        manager.disconnect(device_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(device_id)

# Function to send push notifications
async def send_push_notification(
    device_id: str,
    notification: MobileNotification
):
    """Send push notification to mobile device"""
    message = {
        "type": "notification",
        "notification": {
            "id": notification.id,
            "type": notification.type,
            "title": notification.title,
            "message": notification.message,
            "data": notification.data,
            "timestamp": notification.timestamp.isoformat(),
            "priority": notification.priority
        }
    }
    
    await manager.send_to_device(device_id, message)

# Function to broadcast market updates
async def broadcast_market_update(update: Dict):
    """Broadcast market update to all connected mobile devices"""
    message = {
        "type": "market_update",
        "data": update,
        "timestamp": datetime.now().isoformat()
    }
    
    await manager.broadcast(message)

# Function to send trade execution updates
async def send_trade_update(device_id: str, trade: Dict):
    """Send trade execution update to specific device"""
    message = {
        "type": "trade_update",
        "trade": trade,
        "timestamp": datetime.now().isoformat()
    }
    
    await manager.send_to_device(device_id, message)

# Mobile-specific data formatters
def format_price_for_mobile(price: float) -> str:
    """Format price for mobile display"""
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"

def format_percentage_for_mobile(percent: float) -> Dict:
    """Format percentage with color coding"""
    return {
        "value": f"{percent:+.2f}%",
        "color": "#00ff00" if percent >= 0 else "#ff0000",
        "direction": "up" if percent >= 0 else "down"
    }

# Background task to send periodic updates
async def mobile_update_loop():
    """Send periodic updates to mobile devices"""
    while True:
        try:
            # Get current market data
            market_data = {
                "btc_price": 45000,
                "eth_price": 3200,
                "total_market_cap": 2.1e12,
                "fear_greed_index": 65
            }
            
            # Broadcast to all mobile devices
            await broadcast_market_update(market_data)
            
            # Wait before next update
            await asyncio.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Mobile update loop error: {e}")
            await asyncio.sleep(60)
