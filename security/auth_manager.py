# nexlify/security/auth_manager.py
"""
Nexlify Authentication Manager - Your Digital Bouncer
Handles auth, 2FA, rate limiting, and keeps the riff-raff out
"""

import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import pyotp
import qrcode
from io import BytesIO
import base64
import jwt
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import Column, String, Integer, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
import redis
from passlib.context import CryptContext
import ipaddress
from collections import defaultdict
import asyncio
from functools import wraps
import logging

from config.config_manager import get_config

# Crypto context - military grade
pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

# Database models
Base = declarative_base()

class User(Base):
    """User model - your digital identity in the Nexlify matrix"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: secrets.token_urlsafe(16))
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    pin_hash = Column(String, nullable=False)
    
    # 2FA fields
    totp_secret = Column(String, nullable=True)
    backup_codes = Column(JSON, default=list)
    two_fa_enabled = Column(Boolean, default=False)
    
    # Security tracking
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)
    last_ip = Column(String, nullable=True)
    failed_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)
    
    # Session management
    refresh_tokens = Column(JSON, default=dict)  # token_id: expiry
    active_sessions = Column(JSON, default=list)

class SecurityEvent(Base):
    """Security event logging - know who's knocking"""
    __tablename__ = "security_events"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=True)
    event_type = Column(String, nullable=False)  # login, logout, failed_auth, 2fa_fail, etc.
    ip_address = Column(String, nullable=False)
    user_agent = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    details = Column(JSON, default=dict)

class NexlifyAuthManager:
    """Main authentication brain - handles all security ops"""
    
    def __init__(self, db_session: Session, redis_client: Optional[redis.Redis] = None):
        self.config = get_config()
        self.db = db_session
        self.redis = redis_client or self._init_redis()
        self.logger = logging.getLogger("nexlify.security")
        
        # JWT settings
        self.jwt_algorithm = "HS256"
        self.jwt_secret = self.config.security.master_key
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            self.redis,
            self.config.security.rate_limit_requests,
            self.config.security.rate_limit_window
        )
        
        # IP manager
        self.ip_manager = IPSecurityManager(self.config.security.allowed_ips)
    
    def _init_redis(self) -> redis.Redis:
        """Initialize Redis connection for caching and rate limiting"""
        return redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True,
            socket_connect_timeout=5
        )
    
    async def register_user(
        self, 
        username: str, 
        email: str, 
        password: str,
        enable_2fa: bool = True
    ) -> Tuple[User, str, Optional[str]]:
        """
        Register new user - welcome to the grid, choom
        Returns: (user, pin, totp_uri)
        """
        # Check if user exists
        existing = self.db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists in the system"
            )
        
        # Generate secure PIN
        pin, pin_hash = self.config.generate_pin()
        
        # Create user
        user = User(
            username=username,
            email=email,
            password_hash=pwd_context.hash(password),
            pin_hash=pin_hash,
            two_fa_enabled=enable_2fa
        )
        
        # Setup 2FA if enabled
        totp_uri = None
        if enable_2fa:
            totp_secret = pyotp.random_base32()
            user.totp_secret = totp_secret
            user.backup_codes = self._generate_backup_codes()
            
            # Generate QR code
            totp = pyotp.TOTP(totp_secret)
            totp_uri = totp.provisioning_uri(
                name=email,
                issuer_name=self.config.security.totp_issuer
            )
        
        self.db.add(user)
        self.db.commit()
        
        # Log security event
        self._log_security_event(
            user_id=user.id,
            event_type="user_registered",
            ip_address="127.0.0.1",
            details={"2fa_enabled": enable_2fa}
        )
        
        return user, pin, totp_uri
    
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for 2FA recovery"""
        codes = []
        for _ in range(count):
            code = f"{secrets.randbelow(1000):03d}-{secrets.randbelow(1000):03d}"
            codes.append(pwd_context.hash(code))
        return codes
    
    async def authenticate(
        self,
        username: str,
        password: str,
        pin: str,
        totp_code: Optional[str] = None,
        ip_address: str = "127.0.0.1",
        user_agent: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Authenticate user - multi-factor like a Militech secure facility
        Returns JWT tokens
        """
        # Check rate limit
        if not await self.rate_limiter.check_limit(f"auth:{ip_address}"):
            self._log_security_event(
                user_id=None,
                event_type="rate_limit_exceeded",
                ip_address=ip_address,
                details={"endpoint": "authenticate"}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many authentication attempts. Try again later."
            )
        
        # Check IP whitelist if enabled
        if self.config.security.ip_whitelist_enabled:
            if not self.ip_manager.is_allowed(ip_address):
                self._log_security_event(
                    user_id=None,
                    event_type="ip_blocked",
                    ip_address=ip_address,
                    details={"username_attempt": username}
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied from this location"
                )
        
        # Find user
        user = self.db.query(User).filter(User.username == username).first()
        if not user:
            await asyncio.sleep(secrets.randbelow(2000) / 1000)  # Timing attack prevention
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Account locked until {user.locked_until.isoformat()}"
            )
        
        # Verify password
        if not pwd_context.verify(password, user.password_hash):
            user.failed_attempts += 1
            if user.failed_attempts >= 5:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            self.db.commit()
            
            self._log_security_event(
                user_id=user.id,
                event_type="failed_password",
                ip_address=ip_address,
                details={"attempts": user.failed_attempts}
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Verify PIN
        if not self.config.verify_pin(pin, user.pin_hash):
            self._log_security_event(
                user_id=user.id,
                event_type="failed_pin",
                ip_address=ip_address
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid PIN"
            )
        
        # Verify 2FA if enabled
        if user.two_fa_enabled:
            if not totp_code:
                raise HTTPException(
                    status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                    detail="2FA code required"
                )
            
            totp = pyotp.TOTP(user.totp_secret)
            if not totp.verify(totp_code, valid_window=1):
                # Check backup codes
                backup_valid = False
                for i, backup_hash in enumerate(user.backup_codes):
                    if pwd_context.verify(totp_code, backup_hash):
                        # Remove used backup code
                        user.backup_codes.pop(i)
                        backup_valid = True
                        break
                
                if not backup_valid:
                    self._log_security_event(
                        user_id=user.id,
                        event_type="failed_2fa",
                        ip_address=ip_address
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid 2FA code"
                    )
        
        # Reset failed attempts on successful auth
        user.failed_attempts = 0
        user.last_login = datetime.now(timezone.utc)
        user.last_ip = ip_address
        
        # Generate tokens
        access_token = self._generate_access_token(user.id)
        refresh_token = self._generate_refresh_token(user.id)
        
        # Store refresh token
        if not user.refresh_tokens:
            user.refresh_tokens = {}
        
        token_id = secrets.token_urlsafe(16)
        user.refresh_tokens[token_id] = (
            datetime.now(timezone.utc) + 
            timedelta(seconds=self.config.security.refresh_token_expiry)
        ).isoformat()
        
        # Add active session
        if not user.active_sessions:
            user.active_sessions = []
        
        user.active_sessions.append({
            'id': secrets.token_urlsafe(8),
            'ip': ip_address,
            'user_agent': user_agent,
            'login_time': datetime.now(timezone.utc).isoformat()
        })
        
        # Limit active sessions
        if len(user.active_sessions) > 5:
            user.active_sessions = user.active_sessions[-5:]
        
        self.db.commit()
        
        # Cache session in Redis
        self.redis.setex(
            f"session:{user.id}:{token_id}",
            self.config.security.session_timeout,
            "active"
        )
        
        self._log_security_event(
            user_id=user.id,
            event_type="successful_login",
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'token_type': 'bearer',
            'expires_in': self.config.security.session_timeout
        }
    
    def _generate_access_token(self, user_id: str) -> str:
        """Generate JWT access token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.now(timezone.utc) + timedelta(seconds=self.config.security.session_timeout),
            'iat': datetime.now(timezone.utc),
            'jti': secrets.token_urlsafe(16)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _generate_refresh_token(self, user_id: str) -> str:
        """Generate JWT refresh token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.now(timezone.utc) + timedelta(seconds=self.config.security.refresh_token_expiry),
            'iat': datetime.now(timezone.utc),
            'type': 'refresh',
            'jti': secrets.token_urlsafe(16)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def verify_token(self, token: str) -> Dict[str, any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check if session is still active in Redis
            user_id = payload.get('user_id')
            if user_id:
                session_key = f"session:{user_id}:*"
                if not self.redis.keys(session_key):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Session expired"
                    )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def _log_security_event(
        self,
        user_id: Optional[str],
        event_type: str,
        ip_address: str,
        user_agent: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Log security events for audit trail"""
        event = SecurityEvent(
            user_id=user_id,
            event_type=event_type,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        self.db.add(event)
        self.db.commit()
        
        # Also log to monitoring system
        self.logger.info(f"Security Event: {event_type}", extra={
            'user_id': user_id,
            'ip': ip_address,
            'details': details
        })

class RateLimiter:
    """Rate limiting - keep the DDoS kiddies at bay"""
    
    def __init__(self, redis_client: redis.Redis, max_requests: int, window_seconds: int):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds
    
    async def check_limit(self, key: str) -> bool:
        """Check if request is within rate limit"""
        pipe = self.redis.pipeline()
        now = time.time()
        window_start = now - self.window
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(now): now})
        
        # Set expiry
        pipe.expire(key, self.window)
        
        results = pipe.execute()
        current_requests = results[1]
        
        return current_requests < self.max_requests

class IPSecurityManager:
    """IP-based security - geofencing for the digital age"""
    
    def __init__(self, allowed_ips: List[str]):
        self.allowed_networks = []
        for ip in allowed_ips:
            try:
                # Support CIDR notation
                self.allowed_networks.append(ipaddress.ip_network(ip, strict=False))
            except ValueError:
                logging.warning(f"Invalid IP/CIDR: {ip}")
    
    def is_allowed(self, ip: str) -> bool:
        """Check if IP is in whitelist"""
        try:
            addr = ipaddress.ip_address(ip)
            return any(addr in network for network in self.allowed_networks)
        except ValueError:
            return False

class JWTBearer(HTTPBearer):
    """FastAPI JWT Bearer for route protection"""
    
    def __init__(self, auth_manager: NexlifyAuthManager, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.auth_manager = auth_manager
    
    async def __call__(self, request: Request) -> Optional[str]:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if credentials:
            if credentials.scheme != "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme"
                )
            
            payload = await self.auth_manager.verify_token(credentials.credentials)
            return payload.get('user_id')
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authorization code"
        )

def require_auth(f):
    """Decorator for protecting routes - no ticket, no ride"""
    @wraps(f)
    async def decorated_function(request: Request, *args, **kwargs):
        auth = JWTBearer(get_auth_manager())
        user_id = await auth(request)
        request.state.user_id = user_id
        return await f(request, *args, **kwargs)
    return decorated_function

# Global auth manager instance
_auth_manager: Optional[NexlifyAuthManager] = None

def get_auth_manager(db_session: Optional[Session] = None) -> NexlifyAuthManager:
    """Get or create auth manager instance"""
    global _auth_manager
    if _auth_manager is None and db_session:
        _auth_manager = NexlifyAuthManager(db_session)
    return _auth_manager
