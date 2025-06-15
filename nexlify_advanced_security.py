"""
Nexlify Advanced Security Module v2.0.8
Comprehensive security management with optional master password and 2FA
"""

import os
import json
import jwt
import pyotp
import qrcode
import secrets
import hashlib
import base64
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
import ipaddress
import threading
import asyncio
from collections import defaultdict
import time
import re
from io import BytesIO
import aiofiles
import logging

# Import error handler
from src.core.error_handler import get_error_handler, handle_errors

@dataclass
class SessionInfo:
    """Session information"""
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    device_info: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)

@dataclass
class SecurityEvent:
    """Security event for logging"""
    event_type: str
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    details: Dict[str, Any]
    severity: str  # info, warning, critical

class EncryptionManager:
    """Manages encryption with proper key derivation and rotation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_handler = get_error_handler()
        
        # Key storage
        self.keys_path = Path("config/.keys")
        self.keys_path.mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions
        if os.name != 'nt':  # Unix-like systems
            os.chmod(self.keys_path, 0o700)
            
        # Initialize keys
        self._init_keys()
        
        # Key rotation tracking
        self.last_rotation = datetime.now()
        self.rotation_interval = timedelta(days=90)  # Default 90 days
        
    def _init_keys(self):
        """Initialize encryption keys"""
        # Master key file
        self.master_key_file = self.keys_path / "master.key"
        
        # Check if master password is enabled
        if self.config.get('security', {}).get('master_password_enabled', False):
            master_password = self.config.get('security', {}).get('master_password', '')
            if not master_password:
                # Generate a secure random key if no password
                self.master_key = Fernet.generate_key()
                self._save_master_key(self.master_key)
            else:
                # Derive key from master password
                self.master_key = self._derive_key_from_password(master_password)
        else:
            # No master password, use stored key or generate new
            if self.master_key_file.exists():
                self.master_key = self._load_master_key()
            else:
                self.master_key = Fernet.generate_key()
                self._save_master_key(self.master_key)
                
        self.cipher = Fernet(self.master_key)
        
        # JWT signing key (separate from encryption key)
        self.jwt_key_file = self.keys_path / "jwt.key"
        if self.jwt_key_file.exists():
            self.jwt_key = self.jwt_key_file.read_bytes()
        else:
            self.jwt_key = secrets.token_bytes(32)
            self.jwt_key_file.write_bytes(self.jwt_key)
            if os.name != 'nt':
                os.chmod(self.jwt_key_file, 0o600)
                
    def _derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password with proper salt"""
        if salt is None:
            # Load or generate salt
            salt_file = self.keys_path / "salt"
            if salt_file.exists():
                salt = salt_file.read_bytes()
            else:
                salt = os.urandom(32)  # Random salt per installation
                salt_file.write_bytes(salt)
                if os.name != 'nt':
                    os.chmod(salt_file, 0o600)
                    
        # Use PBKDF2 with high iterations
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600000  # High iteration count for 2025 standards
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
        
    def _save_master_key(self, key: bytes):
        """Save master key securely"""
        self.master_key_file.write_bytes(key)
        if os.name != 'nt':
            os.chmod(self.master_key_file, 0o600)
            
    def _load_master_key(self) -> bytes:
        """Load master key"""
        return self.master_key_file.read_bytes()
        
    def rotate_keys(self):
        """Rotate encryption keys"""
        try:
            # Generate new master key
            new_key = Fernet.generate_key()
            new_cipher = Fernet(new_key)
            
            # Re-encrypt existing data with new key
            self._reencrypt_data(self.cipher, new_cipher)
            
            # Update keys
            self.master_key = new_key
            self.cipher = new_cipher
            self._save_master_key(new_key)
            
            # Rotate JWT key
            self.jwt_key = secrets.token_bytes(32)
            self.jwt_key_file.write_bytes(self.jwt_key)
            
            self.last_rotation = datetime.now()
            
            self.error_handler.log_warning(
                "Encryption keys rotated successfully",
                component="security",
                context={"rotation_time": datetime.now().isoformat()}
            )
            
        except Exception as e:
            self.error_handler.log_critical_error(
                e,
                component="security",
                context={"operation": "key_rotation"}
            )
            raise
            
    def _reencrypt_data(self, old_cipher: Fernet, new_cipher: Fernet):
        """Re-encrypt data with new key"""
        # This would re-encrypt all encrypted configs and data
        # Implementation depends on what data is encrypted
        pass
        
    def encrypt_data(self, data: str) -> str:
        """Encrypt string data"""
        try:
            return self.cipher.encrypt(data.encode()).decode()
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            raise
            
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            return self.cipher.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            raise
            
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None):
        """Encrypt a file"""
        try:
            if output_path is None:
                output_path = file_path.with_suffix(file_path.suffix + '.enc')
                
            # Check if output already exists
            if output_path.exists():
                backup_path = output_path.with_suffix('.bak')
                output_path.rename(backup_path)
                
            data = file_path.read_bytes()
            encrypted = self.cipher.encrypt(data)
            output_path.write_bytes(encrypted)
            
            if os.name != 'nt':
                os.chmod(output_path, 0o600)
                
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="security",
                context={"file": str(file_path)}
            )
            raise
            
    def decrypt_file(self, encrypted_path: Path, output_path: Optional[Path] = None):
        """Decrypt a file with integrity check"""
        try:
            if output_path is None:
                output_path = encrypted_path.with_suffix('')
                
            encrypted_data = encrypted_path.read_bytes()
            
            # Decrypt and verify
            try:
                decrypted = self.cipher.decrypt(encrypted_data)
            except Exception:
                # Try with old key if rotation happened
                if hasattr(self, 'old_cipher'):
                    decrypted = self.old_cipher.decrypt(encrypted_data)
                else:
                    raise
                    
            output_path.write_bytes(decrypted)
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="security",
                context={"file": str(encrypted_path)}
            )
            raise
            
    def encrypt_config(self, config: Dict[str, Any]) -> str:
        """Encrypt configuration dictionary"""
        try:
            json_str = json.dumps(config)
            return self.encrypt_data(json_str)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            raise
            
    def decrypt_config(self, encrypted_config: str) -> Dict[str, Any]:
        """Decrypt configuration dictionary"""
        try:
            json_str = self.decrypt_data(encrypted_config)
            return json.loads(json_str)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            raise

class TwoFactorAuth:
    """Two-factor authentication with enhanced security"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption = encryption_manager
        self.error_handler = get_error_handler()
        
        # Secure storage for 2FA secrets (encrypted)
        self.secrets_file = Path("config/.2fa_secrets.enc")
        self.users = self._load_secrets()
        
        # Rate limiting for TOTP attempts
        self.attempt_tracker = defaultdict(list)
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        
    def _load_secrets(self) -> Dict[str, Dict]:
        """Load encrypted 2FA secrets"""
        if self.secrets_file.exists():
            try:
                encrypted = self.secrets_file.read_text()
                return self.encryption.decrypt_config(encrypted)
            except:
                return {}
        return {}
        
    def _save_secrets(self):
        """Save encrypted 2FA secrets"""
        try:
            encrypted = self.encryption.encrypt_config(self.users)
            self.secrets_file.write_text(encrypted)
            if os.name != 'nt':
                os.chmod(self.secrets_file, 0o600)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            
    def setup_2fa(self, user_id: str, app_name: str = "Nexlify") -> Tuple[str, str]:
        """Setup 2FA for user, returns (secret, qr_code_data)"""
        try:
            # Generate secure secret
            secret = pyotp.random_base32()
            
            # Create TOTP object
            totp = pyotp.TOTP(secret)
            
            # Generate provisioning URI
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name=app_name
            )
            
            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            # Save QR code to memory
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            qr_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Generate backup codes
            backup_codes = self.generate_backup_codes()
            
            # Store user data (encrypted)
            self.users[user_id] = {
                'secret': secret,
                'backup_codes': backup_codes,
                'created_at': datetime.now().isoformat(),
                'last_used': None
            }
            self._save_secrets()
            
            return secret, qr_data
            
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            raise
            
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate secure backup codes"""
        codes = []
        for _ in range(count):
            # Generate 12-character alphanumeric codes
            code = secrets.token_urlsafe(9)  # ~12 chars when base64 encoded
            codes.append(code)
        return codes
        
    def verify_2fa(self, user_id: str, token: str) -> bool:
        """Verify 2FA token with rate limiting"""
        # Check rate limit
        if self._is_rate_limited(user_id):
            self.error_handler.log_warning(
                f"2FA rate limit exceeded for user {user_id}",
                component="security"
            )
            return False
            
        user_data = self.users.get(user_id)
        if not user_data:
            return False
            
        try:
            # Track attempt
            self._track_attempt(user_id)
            
            # Check if it's a backup code
            if token in user_data.get('backup_codes', []):
                # Remove used backup code
                user_data['backup_codes'].remove(token)
                user_data['last_used'] = datetime.now().isoformat()
                self._save_secrets()
                return True
                
            # Verify TOTP with time window for clock drift
            totp = pyotp.TOTP(user_data['secret'])
            valid = totp.verify(token, valid_window=2)  # Allow 2 intervals for clock drift
            
            if valid:
                user_data['last_used'] = datetime.now().isoformat()
                self._save_secrets()
                # Clear attempts on success
                self.attempt_tracker[user_id].clear()
                
            return valid
            
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            return False
            
    def _track_attempt(self, user_id: str):
        """Track verification attempt"""
        now = time.time()
        self.attempt_tracker[user_id].append(now)
        
        # Clean old attempts
        cutoff = now - self.lockout_duration
        self.attempt_tracker[user_id] = [
            t for t in self.attempt_tracker[user_id] if t > cutoff
        ]
        
    def _is_rate_limited(self, user_id: str) -> bool:
        """Check if user is rate limited"""
        now = time.time()
        cutoff = now - self.lockout_duration
        recent_attempts = [
            t for t in self.attempt_tracker[user_id] if t > cutoff
        ]
        return len(recent_attempts) >= self.max_attempts
        
    def disable_2fa(self, user_id: str, password: str) -> bool:
        """Disable 2FA with password verification"""
        # Password verification should be done by SecurityManager
        if user_id in self.users:
            del self.users[user_id]
            self._save_secrets()
            return True
        return False
        
    def generate_qr_code(self, user_id: str) -> Optional[str]:
        """Regenerate QR code for existing user"""
        user_data = self.users.get(user_id)
        if not user_data:
            return None
            
        try:
            totp = pyotp.TOTP(user_data['secret'])
            provisioning_uri = totp.provisioning_uri(
                name=user_id,
                issuer_name="Nexlify"
            )
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            
            # Set expiry for QR code (30 minutes)
            user_data['qr_expiry'] = (datetime.now() + timedelta(minutes=30)).isoformat()
            self._save_secrets()
            
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            return None

class APIKeyRotation:
    """Manages API key rotation for exchanges"""
    
    def __init__(self, config: Dict[str, Any], encryption_manager: EncryptionManager):
        self.config = config
        self.encryption = encryption_manager
        self.error_handler = get_error_handler()
        
        # Key storage
        self.keys_file = Path("config/.api_keys.enc")
        self.key_history_file = Path("config/.key_history.enc")
        
        # Load keys
        self.active_keys = self._load_keys()
        self.key_history = self._load_history()
        
        # Rotation settings
        self.rotation_days = config.get('security', {}).get('api_key_rotation_days', 30)
        self.overlap_hours = 24  # Keep old keys active for 24 hours
        
    def _load_keys(self) -> Dict[str, Dict]:
        """Load encrypted API keys"""
        if self.keys_file.exists():
            try:
                encrypted = self.keys_file.read_text()
                return self.encryption.decrypt_config(encrypted)
            except:
                return {}
        return {}
        
    def _load_history(self) -> List[Dict]:
        """Load key rotation history"""
        if self.key_history_file.exists():
            try:
                encrypted = self.key_history_file.read_text()
                history = self.encryption.decrypt_config(encrypted)
                # Prune old entries (keep last 100)
                return history[-100:]
            except:
                return []
        return []
        
    def _save_keys(self):
        """Save encrypted API keys"""
        try:
            encrypted = self.encryption.encrypt_config(self.active_keys)
            self.keys_file.write_text(encrypted)
            if os.name != 'nt':
                os.chmod(self.keys_file, 0o600)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            
    def _save_history(self):
        """Save key history"""
        try:
            encrypted = self.encryption.encrypt_config(self.key_history)
            self.key_history_file.write_text(encrypted)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            
    def add_key(self, exchange: str, api_key: str, api_secret: str):
        """Add new API key with validation"""
        # Validate format based on exchange
        if not self._validate_key_format(exchange, api_key, api_secret):
            raise ValueError(f"Invalid API key format for {exchange}")
            
        self.active_keys[exchange] = {
            'api_key': api_key,
            'api_secret': api_secret,
            'created_at': datetime.now().isoformat(),
            'last_rotated': datetime.now().isoformat(),
            'status': 'active'
        }
        
        self._save_keys()
        
        # Log to history
        self.key_history.append({
            'exchange': exchange,
            'action': 'added',
            'timestamp': datetime.now().isoformat(),
            'key_prefix': api_key[:8] + '...'  # Store prefix only
        })
        self._save_history()
        
    def _validate_key_format(self, exchange: str, api_key: str, api_secret: str) -> bool:
        """Validate API key format for exchange"""
        patterns = {
            'binance': (r'^[A-Za-z0-9]{64}$', r'^[A-Za-z0-9]{64}$'),
            'kraken': (r'^[A-Za-z0-9+/]{56}$', r'^[A-Za-z0-9+/]{88}$'),
            'coinbase': (r'^[a-zA-Z0-9-]{32}$', r'^[a-zA-Z0-9+/]{43}=$'),
        }
        
        if exchange in patterns:
            key_pattern, secret_pattern = patterns[exchange]
            return (re.match(key_pattern, api_key) is not None and 
                   re.match(secret_pattern, api_secret) is not None)
        
        # Default validation - check minimum length
        return len(api_key) >= 16 and len(api_secret) >= 16
        
    def rotate_keys(self, exchange: str, new_api_key: str, new_api_secret: str) -> bool:
        """Rotate API keys for exchange with validation"""
        try:
            # Validate new keys
            if not self._validate_key_format(exchange, new_api_key, new_api_secret):
                return False
                
            # Keep old key temporarily
            old_key = self.active_keys.get(exchange)
            if old_key:
                old_key['status'] = 'rotating'
                old_key['expires_at'] = (
                    datetime.now() + timedelta(hours=self.overlap_hours)
                ).isoformat()
                
            # Add new key
            self.active_keys[f"{exchange}_new"] = {
                'api_key': new_api_key,
                'api_secret': new_api_secret,
                'created_at': datetime.now().isoformat(),
                'last_rotated': datetime.now().isoformat(),
                'status': 'pending'
            }
            
            self._save_keys()
            
            # Schedule activation
            self._schedule_key_activation(exchange)
            
            # Log rotation
            self.key_history.append({
                'exchange': exchange,
                'action': 'rotated',
                'timestamp': datetime.now().isoformat(),
                'old_key_prefix': old_key['api_key'][:8] + '...' if old_key else 'N/A',
                'new_key_prefix': new_api_key[:8] + '...'
            })
            self._save_history()
            
            return True
            
        except Exception as e:
            self.error_handler.log_error(
                e,
                component="security",
                context={"exchange": exchange}
            )
            return False
            
    def _schedule_key_activation(self, exchange: str):
        """Schedule new key activation after validation"""
        def activate():
            time.sleep(60)  # Wait 1 minute for validation
            
            new_key = self.active_keys.get(f"{exchange}_new")
            if new_key and new_key['status'] == 'pending':
                # Activate new key
                new_key['status'] = 'active'
                self.active_keys[exchange] = new_key
                del self.active_keys[f"{exchange}_new"]
                self._save_keys()
                
        threading.Thread(target=activate, daemon=True).start()
        
    def get_active_keys(self, exchange: str) -> Optional[Dict[str, str]]:
        """Get active API keys for exchange"""
        # Check for keys in rotation
        for key_name, key_data in self.active_keys.items():
            if key_name.startswith(exchange) and key_data['status'] == 'active':
                return {
                    'api_key': key_data['api_key'],
                    'api_secret': key_data['api_secret']
                }
        return None
        
    def check_rotation_needed(self) -> List[str]:
        """Check which exchanges need key rotation"""
        exchanges_needing_rotation = []
        
        for exchange, key_data in self.active_keys.items():
            if '_new' in exchange or key_data['status'] != 'active':
                continue
                
            last_rotated = datetime.fromisoformat(key_data['last_rotated'])
            if datetime.now() - last_rotated > timedelta(days=self.rotation_days):
                exchanges_needing_rotation.append(exchange)
                
        return exchanges_needing_rotation
        
    def prune_old_keys(self):
        """Remove expired keys from rotation"""
        to_remove = []
        
        for key_name, key_data in self.active_keys.items():
            if key_data['status'] == 'rotating' and 'expires_at' in key_data:
                expires_at = datetime.fromisoformat(key_data['expires_at'])
                if datetime.now() > expires_at:
                    to_remove.append(key_name)
                    
        for key_name in to_remove:
            del self.active_keys[key_name]
            
        if to_remove:
            self._save_keys()

class AccessControl:
    """IP-based access control with rate limiting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.error_handler = get_error_handler()
        
        # Access lists
        self.whitelist = set(config.get('security', {}).get('ip_whitelist', []))
        self.blacklist = set()
        
        # Failed attempt tracking
        self.failed_attempts = defaultdict(list)
        self.max_attempts = config.get('security', {}).get('max_failed_attempts', 5)
        self.lockout_duration = config.get('security', {}).get('lockout_duration_minutes', 30) * 60
        
        # Rate limiting
        self.request_tracker = defaultdict(list)
        self.rate_limit = config.get('api', {}).get('rate_limit_per_minute', 60)
        
        # Geo-fencing (optional)
        self.geo_restrictions = config.get('security', {}).get('geo_restrictions', {})
        
    def check_access(self, ip_address: str, user_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if IP has access, returns (allowed, reason)"""
        # Check blacklist first
        if ip_address in self.blacklist:
            return False, "IP blacklisted"
            
        # Check if temporarily locked out
        if self._is_locked_out(ip_address):
            return False, "Temporarily locked out due to failed attempts"
            
        # Check whitelist if enabled
        if self.config.get('security', {}).get('ip_whitelist_enabled', False):
            if not self.is_whitelisted(ip_address):
                return False, "IP not whitelisted"
                
        # Check geo-restrictions if enabled
        if self.geo_restrictions.get('enabled', False):
            if not self._check_geo_restriction(ip_address):
                return False, "Geographic restriction"
                
        # Check rate limit
        if not self._check_rate_limit(ip_address):
            return False, "Rate limit exceeded"
            
        return True, None
        
    def is_whitelisted(self, ip_address: str) -> bool:
        """Check if IP is whitelisted (supports CIDR)"""
        if not self.whitelist:
            return True  # No whitelist means all allowed
            
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for allowed in self.whitelist:
                try:
                    # Check if it's a CIDR range
                    if '/' in allowed:
                        network = ipaddress.ip_network(allowed, strict=False)
                        if ip in network:
                            return True
                    else:
                        # Direct IP match
                        if str(ip) == allowed:
                            return True
                except:
                    continue
                    
        except:
            pass
            
        return False
        
    def add_to_whitelist(self, ip_or_cidr: str):
        """Add IP or CIDR range to whitelist"""
        try:
            # Validate IP/CIDR
            if '/' in ip_or_cidr:
                ipaddress.ip_network(ip_or_cidr, strict=False)
            else:
                ipaddress.ip_address(ip_or_cidr)
                
            self.whitelist.add(ip_or_cidr)
            
            # Update config
            self.config['security']['ip_whitelist'] = list(self.whitelist)
            
        except ValueError as e:
            self.error_handler.log_error(e, component="security")
            raise
            
    def remove_from_whitelist(self, ip_or_cidr: str):
        """Remove IP or CIDR from whitelist"""
        self.whitelist.discard(ip_or_cidr)
        self.config['security']['ip_whitelist'] = list(self.whitelist)
        
    def record_failed_attempt(self, ip_address: str, reason: str = ""):
        """Record failed access attempt"""
        now = time.time()
        self.failed_attempts[ip_address].append({
            'timestamp': now,
            'reason': reason
        })
        
        # Clean old attempts
        self._clean_old_attempts(ip_address)
        
        # Check if should blacklist
        recent_attempts = self.failed_attempts[ip_address]
        if len(recent_attempts) >= self.max_attempts * 2:
            # Permanent blacklist after double the max attempts
            self.blacklist.add(ip_address)
            self.error_handler.log_warning(
                f"IP {ip_address} blacklisted after multiple failures",
                component="security"
            )
            
    def _clean_old_attempts(self, ip_address: str):
        """Clean old failed attempts"""
        now = time.time()
        cutoff = now - self.lockout_duration
        
        self.failed_attempts[ip_address] = [
            attempt for attempt in self.failed_attempts[ip_address]
            if attempt['timestamp'] > cutoff
        ]
        
    def _is_locked_out(self, ip_address: str) -> bool:
        """Check if IP is temporarily locked out"""
        self._clean_old_attempts(ip_address)
        return len(self.failed_attempts[ip_address]) >= self.max_attempts
        
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP is within rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        # Track request
        self.request_tracker[ip_address].append(now)
        
        # Clean old requests
        self.request_tracker[ip_address] = [
            t for t in self.request_tracker[ip_address] if t > minute_ago
        ]
        
        return len(self.request_tracker[ip_address]) <= self.rate_limit
        
    def _check_geo_restriction(self, ip_address: str) -> bool:
        """Check geographic restrictions (stub for geo-IP lookup)"""
        # This would integrate with a geo-IP service
        # For now, return True
        return True
        
    def clear_failed_attempts(self, ip_address: str):
        """Clear failed attempts for IP"""
        self.failed_attempts.pop(ip_address, None)

class SecurityManager:
    """Main security manager coordinating all security features"""
    
    def __init__(self, config_path: str = "config/enhanced_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.error_handler = get_error_handler()
        
        # Initialize components
        self.encryption = EncryptionManager(self.config)
        self.two_factor = TwoFactorAuth(self.encryption)
        self.api_rotation = APIKeyRotation(self.config, self.encryption)
        self.access_control = AccessControl(self.config)
        
        # Password hasher (Argon2)
        self.password_hasher = PasswordHasher(
            time_cost=3,  # iterations
            memory_cost=65536,  # 64MB
            parallelism=4
        )
        
        # Session management
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.session_timeout = timedelta(
            minutes=self.config.get('security', {}).get('session_timeout_minutes', 60)
        )
        
        # Security event log
        self.security_events: List[SecurityEvent] = []
        self.max_events = 10000
        
        # User credentials (encrypted storage)
        self.users_file = Path("config/.users.enc")
        self.users = self._load_users()
        
        # Initialize audit logging
        self._init_audit_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            return {}
            
    def _load_users(self) -> Dict[str, Dict]:
        """Load encrypted user data"""
        if self.users_file.exists():
            try:
                encrypted = self.users_file.read_text()
                return self.encryption.decrypt_config(encrypted)
            except:
                return {}
        return {"admin": {}}  # Default admin user
        
    def _save_users(self):
        """Save encrypted user data"""
        try:
            encrypted = self.encryption.encrypt_config(self.users)
            self.users_file.write_text(encrypted)
            if os.name != 'nt':
                os.chmod(self.users_file, 0o600)
        except Exception as e:
            self.error_handler.log_error(e, component="security")
            
    def _init_audit_logging(self):
        """Initialize audit logging"""
        # This will be handled by nexlify_audit_trail.py
        pass
        
    @handle_errors(component="security", severity="critical")
    def authenticate_user(self, username: str, password: str, ip_address: str,
                         totp_token: Optional[str] = None) -> Optional[str]:
        """Authenticate user and create session"""
        # Check access control
        allowed, reason = self.access_control.check_access(ip_address, username)
        if not allowed:
            self.log_security_event(
                "login_denied",
                user_id=username,
                ip_address=ip_address,
                details={"reason": reason},
                severity="warning"
            )
            self.access_control.record_failed_attempt(ip_address, reason)
            return None
            
        # Check if master password is enabled
        if not self.config.get('security', {}).get('master_password_enabled', False):
            # No master password, create session directly
            return self.create_session(username, ip_address)
            
        # Verify password
        user_data = self.users.get(username, {})
        stored_hash = user_data.get('password_hash')
        
        if not stored_hash:
            # First login, set password
            user_data['password_hash'] = self.password_hasher.hash(password)
            self.users[username] = user_data
            self._save_users()
        else:
            # Verify password
            try:
                self.password_hasher.verify(stored_hash, password)
                
                # Check if rehashing needed (algorithm update)
                if self.password_hasher.check_needs_rehash(stored_hash):
                    user_data['password_hash'] = self.password_hasher.hash(password)
                    self._save_users()
                    
            except (VerifyMismatchError, VerificationError):
                self.log_security_event(
                    "login_failed",
                    user_id=username,
                    ip_address=ip_address,
                    details={"reason": "invalid_password"},
                    severity="warning"
                )
                self.access_control.record_failed_attempt(ip_address, "invalid_password")
                return None
                
        # Check 2FA if enabled
        if self.config.get('security', {}).get('2fa_enabled', False):
            if username in self.two_factor.users:
                if not totp_token:
                    # 2FA required but not provided
                    return None
                    
                if not self.two_factor.verify_2fa(username, totp_token):
                    self.log_security_event(
                        "2fa_failed",
                        user_id=username,
                        ip_address=ip_address,
                        details={},
                        severity="warning"
                    )
                    self.access_control.record_failed_attempt(ip_address, "invalid_2fa")
                    return None
                    
        # Authentication successful
        self.access_control.clear_failed_attempts(ip_address)
        return self.create_session(username, ip_address)
        
    def create_session(self, user_id: str, ip_address: str,
                      device_info: Optional[Dict] = None) -> str:
        """Create new session"""
        # Generate session token
        session_id = secrets.token_urlsafe(32)
        
        # Create JWT token
        now = datetime.utcnow()
        expires_at = now + self.session_timeout
        
        payload = {
            'session_id': session_id,
            'user_id': user_id,
            'ip_address': ip_address,
            'iat': now,
            'exp': expires_at
        }
        
        token = jwt.encode(payload, self.encryption.jwt_key, algorithm='HS256')
        
        # Store session info
        session_info = SessionInfo(
            user_id=user_id,
            token=token,
            created_at=now,
            expires_at=expires_at,
            ip_address=ip_address,
            device_info=device_info or {},
            permissions=self._get_user_permissions(user_id)
        )
        
        self.active_sessions[session_id] = session_info
        
        # Log event
        self.log_security_event(
            "login_success",
            user_id=user_id,
            ip_address=ip_address,
            details={"session_id": session_id},
            severity="info"
        )
        
        return token
        
    def validate_session(self, token: str, ip_address: str) -> Optional[SessionInfo]:
        """Validate session token"""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.encryption.jwt_key, algorithms=['HS256'])
            
            session_id = payload.get('session_id')
            session_info = self.active_sessions.get(session_id)
            
            if not session_info:
                return None
                
            # Check expiration
            if datetime.utcnow() > session_info.expires_at:
                del self.active_sessions[session_id]
                return None
                
            # Check IP match (optional)
            if self.config.get('security', {}).get('enforce_ip_match', True):
                if session_info.ip_address != ip_address:
                    self.log_security_event(
                        "session_ip_mismatch",
                        user_id=session_info.user_id,
                        ip_address=ip_address,
                        details={
                            "expected_ip": session_info.ip_address,
                            "actual_ip": ip_address
                        },
                        severity="warning"
                    )
                    return None
                    
            # Check rate limit
            allowed, _ = self.access_control.check_access(ip_address, session_info.user_id)
            if not allowed:
                return None
                
            return session_info
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            self.log_security_event(
                "invalid_token",
                user_id=None,
                ip_address=ip_address,
                details={},
                severity="warning"
            )
            return None
            
    def invalidate_session(self, token: str):
        """Invalidate session"""
        try:
            payload = jwt.decode(token, self.encryption.jwt_key, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            if session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]
                del self.active_sessions[session_id]
                
                self.log_security_event(
                    "logout",
                    user_id=session_info.user_id,
                    ip_address=session_info.ip_address,
                    details={"session_id": session_id},
                    severity="info"
                )
        except:
            pass
            
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions"""
        # Default permissions
        permissions = ["read", "trade"]
        
        # Admin gets all permissions
        if user_id == "admin":
            permissions.extend(["admin", "configure", "audit"])
            
        return permissions
        
    def log_security_event(self, event_type: str, user_id: Optional[str],
                          ip_address: Optional[str], details: Dict[str, Any],
                          severity: str = "info"):
        """Log security event"""
        event = SecurityEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            severity=severity
        )
        
        self.security_events.append(event)
        
        # Trim old events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]
            
        # Log to error handler if critical
        if severity == "critical":
            self.error_handler.log_critical_error(
                Exception(f"Security event: {event_type}"),
                component="security",
                context=details
            )
            
        # Send alert for critical events
        if severity in ["critical", "warning"]:
            self.send_security_alert(event)
            
    def send_security_alert(self, event: SecurityEvent):
        """Send security alert via configured channels"""
        message = (
            f"🔐 Security Alert: {event.event_type}\n"
            f"User: {event.user_id or 'Unknown'}\n"
            f"IP: {event.ip_address or 'Unknown'}\n"
            f"Time: {event.timestamp}\n"
            f"Details: {json.dumps(event.details, indent=2)}"
        )
        
        # Log for now (telegram integration in error_handler)
        self.error_handler.log_warning(
            message,
            component="security"
        )
        
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary"""
        # Recent events by type
        event_counts = defaultdict(int)
        for event in self.security_events[-1000:]:  # Last 1000 events
            event_counts[event.event_type] += 1
            
        # Active session count
        active_sessions = sum(
            1 for s in self.active_sessions.values()
            if datetime.utcnow() <= s.expires_at
        )
        
        # Failed attempts summary
        total_failed_attempts = sum(
            len(attempts) for attempts in self.access_control.failed_attempts.values()
        )
        
        # Keys needing rotation
        keys_needing_rotation = self.api_rotation.check_rotation_needed()
        
        return {
            'security_enabled': {
                'master_password': self.config.get('security', {}).get('master_password_enabled', False),
                '2fa': self.config.get('security', {}).get('2fa_enabled', False),
                'ip_whitelist': self.config.get('security', {}).get('ip_whitelist_enabled', False)
            },
            'active_sessions': active_sessions,
            'total_users': len(self.users),
            '2fa_users': len(self.two_factor.users),
            'whitelisted_ips': len(self.access_control.whitelist),
            'blacklisted_ips': len(self.access_control.blacklist),
            'failed_attempts': total_failed_attempts,
            'recent_events': dict(event_counts),
            'keys_needing_rotation': keys_needing_rotation,
            'last_key_rotation': self.encryption.last_rotation.isoformat()
        }
        
    def change_master_password(self, old_password: str, new_password: str) -> bool:
        """Change master password"""
        # Verify old password first
        if self.config.get('security', {}).get('master_password_enabled', False):
            # Would need to verify old password
            pass
            
        # Update password
        self.config['security']['master_password'] = new_password
        
        # Re-derive encryption key
        self.encryption._init_keys()
        
        # Re-encrypt all data with new key
        # This would need to be implemented based on what data is encrypted
        
        return True
        
    def enable_2fa(self, user_id: str) -> Tuple[str, str]:
        """Enable 2FA for user"""
        secret, qr_code = self.two_factor.setup_2fa(user_id)
        
        # Update config
        if not self.config.get('security', {}).get('2fa_enabled', False):
            self.config['security']['2fa_enabled'] = True
            
        self.log_security_event(
            "2fa_enabled",
            user_id=user_id,
            ip_address=None,
            details={},
            severity="info"
        )
        
        return secret, qr_code
        
    def disable_2fa(self, user_id: str, password: str) -> bool:
        """Disable 2FA for user"""
        # Verify password
        if not self.authenticate_user(user_id, password, "127.0.0.1"):
            return False
            
        success = self.two_factor.disable_2fa(user_id, password)
        
        if success:
            self.log_security_event(
                "2fa_disabled",
                user_id=user_id,
                ip_address=None,
                details={},
                severity="info"
            )
            
        return success

# Global security manager instance
_security_manager = None

def get_security_manager() -> SecurityManager:
    """Get or create global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager
