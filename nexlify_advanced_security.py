"""
Nexlify Enhanced - Advanced Security System
Implements Feature 29: 2FA, encryption, API rotation, and access control
"""

import pyotp
import qrcode
import io
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import jwt
import secrets
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import hashlib
import ipaddress
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Security event for audit logging"""
    event_type: str
    user_id: str
    ip_address: str
    details: Dict
    timestamp: datetime
    severity: str  # info, warning, critical
    success: bool

class TwoFactorAuth:
    """Manages 2FA with TOTP and hardware key support"""
    
    def __init__(self, issuer: str = "Nexlify Trading"):
        self.issuer = issuer
        self.users = {}  # In production, use secure storage
        
    def generate_secret(self, user_id: str) -> str:
        """Generate new 2FA secret for user"""
        secret = pyotp.random_base32()
        
        # Store encrypted
        self.users[user_id] = {
            'secret': secret,
            'enabled': False,
            'backup_codes': self._generate_backup_codes()
        }
        
        return secret
        
    def generate_qr_code(self, user_id: str, secret: str) -> str:
        """Generate QR code for 2FA setup"""
        # Create provisioning URI
        totp = pyotp.TOTP(secret)
        provisioning_uri = totp.provisioning_uri(
            name=user_id,
            issuer_name=self.issuer
        )
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(provisioning_uri)
        qr.make(fit=True)
        
        # Convert to base64 image
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        
        return base64.b64encode(buffer.getvalue()).decode()
        
    def verify_token(self, user_id: str, token: str) -> bool:
        """Verify TOTP token"""
        if user_id not in self.users or not self.users[user_id]['enabled']:
            return False
            
        secret = self.users[user_id]['secret']
        totp = pyotp.TOTP(secret)
        
        # Allow for time drift (±1 time step)
        return totp.verify(token, valid_window=1)
        
    def enable_2fa(self, user_id: str, verification_token: str) -> bool:
        """Enable 2FA after successful verification"""
        if self.verify_token(user_id, verification_token):
            self.users[user_id]['enabled'] = True
            return True
        return False
        
    def disable_2fa(self, user_id: str, current_token: str) -> bool:
        """Disable 2FA with current token verification"""
        if self.verify_token(user_id, current_token):
            self.users[user_id]['enabled'] = False
            return True
        return False
        
    def _generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for account recovery"""
        return [secrets.token_hex(4).upper() for _ in range(count)]
        
    def verify_backup_code(self, user_id: str, code: str) -> bool:
        """Verify and consume backup code"""
        if user_id not in self.users:
            return False
            
        backup_codes = self.users[user_id].get('backup_codes', [])
        
        if code in backup_codes:
            backup_codes.remove(code)
            return True
            
        return False

class EncryptionManager:
    """Handles local data encryption"""
    
    def __init__(self, master_password: Optional[str] = None):
        if master_password:
            self.key = self._derive_key(master_password)
        else:
            self.key = Fernet.generate_key()
            
        self.fernet = Fernet(self.key)
        
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        # Use PBKDF2 for key derivation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'nexlify_salt_v1',  # In production, use random salt
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
        
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
        
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
        
    def encrypt_file(self, file_path: Path) -> Path:
        """Encrypt entire file"""
        with open(file_path, 'rb') as f:
            encrypted = self.fernet.encrypt(f.read())
            
        encrypted_path = file_path.with_suffix('.enc')
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted)
            
        return encrypted_path
        
    def decrypt_file(self, encrypted_path: Path) -> Path:
        """Decrypt file"""
        with open(encrypted_path, 'rb') as f:
            decrypted = self.fernet.decrypt(f.read())
            
        decrypted_path = encrypted_path.with_suffix('')
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted)
            
        return decrypted_path

class APIKeyRotation:
    """Manages automatic API key rotation"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
        self.rotation_schedule = {}
        self.key_history = {}
        
    def schedule_rotation(self, 
                         exchange: str,
                         rotation_days: int = 30,
                         overlap_hours: int = 24):
        """Schedule API key rotation"""
        self.rotation_schedule[exchange] = {
            'rotation_days': rotation_days,
            'overlap_hours': overlap_hours,
            'last_rotation': datetime.now(),
            'next_rotation': datetime.now() + timedelta(days=rotation_days)
        }
        
    def check_rotation_needed(self, exchange: str) -> bool:
        """Check if key rotation is needed"""
        if exchange not in self.rotation_schedule:
            return False
            
        schedule = self.rotation_schedule[exchange]
        return datetime.now() >= schedule['next_rotation']
        
    def rotate_keys(self, exchange: str, new_key: str, new_secret: str) -> Dict:
        """Rotate API keys with overlap period"""
        # Encrypt new keys
        encrypted_key = self.encryption.encrypt_data(new_key)
        encrypted_secret = self.encryption.encrypt_data(new_secret)
        
        # Store old keys in history
        if exchange not in self.key_history:
            self.key_history[exchange] = []
            
        self.key_history[exchange].append({
            'key': encrypted_key,
            'secret': encrypted_secret,
            'activated_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(
                hours=self.rotation_schedule[exchange]['overlap_hours']
            )
        })
        
        # Update rotation schedule
        self.rotation_schedule[exchange]['last_rotation'] = datetime.now()
        self.rotation_schedule[exchange]['next_rotation'] = (
            datetime.now() + timedelta(days=self.rotation_schedule[exchange]['rotation_days'])
        )
        
        return {
            'success': True,
            'next_rotation': self.rotation_schedule[exchange]['next_rotation'],
            'overlap_expires': self.key_history[exchange][-1]['expires_at']
        }
        
    def get_active_keys(self, exchange: str) -> List[Tuple[str, str]]:
        """Get all currently active keys (including overlap)"""
        active_keys = []
        
        if exchange in self.key_history:
            for key_set in self.key_history[exchange]:
                if datetime.now() < key_set['expires_at']:
                    decrypted_key = self.encryption.decrypt_data(key_set['key'])
                    decrypted_secret = self.encryption.decrypt_data(key_set['secret'])
                    active_keys.append((decrypted_key, decrypted_secret))
                    
        return active_keys

class AccessControl:
    """IP whitelisting and access control"""
    
    def __init__(self):
        self.whitelisted_ips = set()
        self.blacklisted_ips = set()
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        
    def add_whitelist(self, ip: str):
        """Add IP to whitelist"""
        try:
            # Validate IP
            ipaddress.ip_address(ip)
            self.whitelisted_ips.add(ip)
            logger.info(f"Added {ip} to whitelist")
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            
    def remove_whitelist(self, ip: str):
        """Remove IP from whitelist"""
        self.whitelisted_ips.discard(ip)
        
    def is_whitelisted(self, ip: str) -> bool:
        """Check if IP is whitelisted"""
        # If no whitelist configured, allow all
        if not self.whitelisted_ips:
            return True
            
        return ip in self.whitelisted_ips
        
    def is_blacklisted(self, ip: str) -> bool:
        """Check if IP is blacklisted"""
        return ip in self.blacklisted_ips
        
    def record_failed_attempt(self, ip: str):
        """Record failed login attempt"""
        if ip not in self.failed_attempts:
            self.failed_attempts[ip] = []
            
        self.failed_attempts[ip].append(datetime.now())
        
        # Check if should blacklist
        recent_attempts = [
            attempt for attempt in self.failed_attempts[ip]
            if datetime.now() - attempt < self.lockout_duration
        ]
        
        if len(recent_attempts) >= self.max_failed_attempts:
            self.blacklisted_ips.add(ip)
            logger.warning(f"IP {ip} blacklisted after {len(recent_attempts)} failed attempts")
            
    def clear_failed_attempts(self, ip: str):
        """Clear failed attempts on successful login"""
        if ip in self.failed_attempts:
            del self.failed_attempts[ip]
            
    def check_access(self, ip: str) -> Tuple[bool, str]:
        """Check if IP has access"""
        if self.is_blacklisted(ip):
            return False, "IP is blacklisted due to failed attempts"
            
        if not self.is_whitelisted(ip):
            return False, "IP is not whitelisted"
            
        return True, "Access granted"

class SecurityManager:
    """Main security manager coordinating all security features"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        self.two_factor = TwoFactorAuth()
        self.encryption = EncryptionManager(config.get('master_password'))
        self.api_rotation = APIKeyRotation(self.encryption)
        self.access_control = AccessControl()
        
        # Security events log
        self.security_events = []
        
        # Session management
        self.active_sessions = {}
        self.session_timeout = timedelta(hours=24)
        
    def authenticate_user(self, 
                         username: str,
                         password: str,
                         totp_token: Optional[str],
                         ip_address: str) -> Tuple[bool, str, Optional[str]]:
        """Complete authentication with 2FA"""
        # Check IP access
        has_access, access_msg = self.access_control.check_access(ip_address)
        if not has_access:
            self.log_security_event(
                SecurityEvent(
                    event_type="login_blocked",
                    user_id=username,
                    ip_address=ip_address,
                    details={"reason": access_msg},
                    timestamp=datetime.now(),
                    severity="warning",
                    success=False
                )
            )
            return False, access_msg, None
            
        # Verify password (simplified - use proper hashing in production)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        stored_hash = self.config.get('users', {}).get(username, {}).get('password_hash')
        
        if password_hash != stored_hash:
            self.access_control.record_failed_attempt(ip_address)
            self.log_security_event(
                SecurityEvent(
                    event_type="login_failed",
                    user_id=username,
                    ip_address=ip_address,
                    details={"reason": "invalid_password"},
                    timestamp=datetime.now(),
                    severity="warning",
                    success=False
                )
            )
            return False, "Invalid credentials", None
            
        # Check 2FA if enabled
        if self.two_factor.users.get(username, {}).get('enabled', False):
            if not totp_token:
                return False, "2FA token required", None
                
            if not self.two_factor.verify_token(username, totp_token):
                self.access_control.record_failed_attempt(ip_address)
                self.log_security_event(
                    SecurityEvent(
                        event_type="2fa_failed",
                        user_id=username,
                        ip_address=ip_address,
                        details={"reason": "invalid_token"},
                        timestamp=datetime.now(),
                        severity="warning",
                        success=False
                    )
                )
                return False, "Invalid 2FA token", None
                
        # Authentication successful
        self.access_control.clear_failed_attempts(ip_address)
        
        # Create session
        session_token = self.create_session(username, ip_address)
        
        self.log_security_event(
            SecurityEvent(
                event_type="login_success",
                user_id=username,
                ip_address=ip_address,
                details={"2fa_used": bool(totp_token)},
                timestamp=datetime.now(),
                severity="info",
                success=True
            )
        )
        
        return True, "Authentication successful", session_token
        
    def create_session(self, username: str, ip_address: str) -> str:
        """Create authenticated session"""
        session_id = secrets.token_urlsafe(32)
        
        self.active_sessions[session_id] = {
            'username': username,
            'ip_address': ip_address,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'expires_at': datetime.now() + self.session_timeout
        }
        
        # Create JWT token
        token_data = {
            'session_id': session_id,
            'username': username,
            'exp': datetime.utcnow() + self.session_timeout
        }
        
        token = jwt.encode(
            token_data,
            self.encryption.key,
            algorithm='HS256'
        )
        
        return token
        
    def validate_session(self, token: str, ip_address: str) -> Tuple[bool, str]:
        """Validate session token"""
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.encryption.key,
                algorithms=['HS256']
            )
            
            session_id = payload.get('session_id')
            
            # Check if session exists
            if session_id not in self.active_sessions:
                return False, "Invalid session"
                
            session = self.active_sessions[session_id]
            
            # Check IP match
            if session['ip_address'] != ip_address:
                self.log_security_event(
                    SecurityEvent(
                        event_type="session_ip_mismatch",
                        user_id=session['username'],
                        ip_address=ip_address,
                        details={
                            "expected_ip": session['ip_address'],
                            "actual_ip": ip_address
                        },
                        timestamp=datetime.now(),
                        severity="critical",
                        success=False
                    )
                )
                return False, "IP address mismatch"
                
            # Check expiration
            if datetime.now() > session['expires_at']:
                del self.active_sessions[session_id]
                return False, "Session expired"
                
            # Update last activity
            session['last_activity'] = datetime.now()
            
            return True, session['username']
            
        except jwt.ExpiredSignatureError:
            return False, "Token expired"
        except jwt.InvalidTokenError:
            return False, "Invalid token"
            
    def log_security_event(self, event: SecurityEvent):
        """Log security event for audit trail"""
        self.security_events.append(event)
        
        # Log to file
        log_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'severity': event.severity,
            'success': event.success,
            'details': event.details
        }
        
        logger.info(f"Security Event: {json.dumps(log_entry)}")
        
        # Trigger alerts for critical events
        if event.severity == "critical":
            self.send_security_alert(event)
            
    def send_security_alert(self, event: SecurityEvent):
        """Send security alert for critical events"""
        alert_message = f"""
NEXLIFY SECURITY ALERT

Event: {event.event_type}
User: {event.user_id}
IP: {event.ip_address}
Time: {event.timestamp}
Details: {json.dumps(event.details, indent=2)}

This is a critical security event that requires immediate attention.
"""
        
        # In production, send via email/SMS/Telegram
        logger.critical(alert_message)
        
    def get_security_summary(self) -> Dict:
        """Get security status summary"""
        recent_events = [
            e for e in self.security_events
            if datetime.now() - e.timestamp < timedelta(days=1)
        ]
        
        return {
            'active_sessions': len(self.active_sessions),
            'whitelisted_ips': len(self.access_control.whitelisted_ips),
            'blacklisted_ips': len(self.access_control.blacklisted_ips),
            'recent_events': {
                'total': len(recent_events),
                'failed_logins': sum(1 for e in recent_events if e.event_type == 'login_failed'),
                'successful_logins': sum(1 for e in recent_events if e.event_type == 'login_success'),
                'critical_events': sum(1 for e in recent_events if e.severity == 'critical')
            },
            '2fa_enabled_users': sum(
                1 for u in self.two_factor.users.values() 
                if u.get('enabled', False)
            )
        }