#!/usr/bin/env python3
"""
Nexlify Advanced Security Module
Enhanced security features including 2FA, encryption, and session management
"""

import os
import hashlib
import secrets
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import json
from pathlib import Path

# Security libraries
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import pyotp
import base64

from error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class EncryptionManager:
    """Handles encryption/decryption of sensitive data"""

    def __init__(self, master_password: str = None):
        self.master_password = master_password or self._generate_master_key()
        self.cipher_suite = self._init_cipher()

    def _generate_master_key(self) -> str:
        """Generate a secure master key"""
        return Fernet.generate_key().decode('utf-8')

    def _init_cipher(self) -> Fernet:
        """Initialize the encryption cipher"""
        try:
            # Derive a key from master password
            if isinstance(self.master_password, str):
                password = self.master_password.encode('utf-8')
            else:
                password = self.master_password

            # Use PBKDF2 to derive a key
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'nexlify_salt_v1',  # Should be random and stored
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            return Fernet(key)
        except Exception as e:
            logger.error(f"Error initializing cipher: {e}")
            # Fallback to simple key
            return Fernet(Fernet.generate_key())

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            if not data:
                return ""
            encrypted = self.cipher_suite.encrypt(data.encode('utf-8'))
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return ""

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            if not encrypted_data:
                return ""
            decoded = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted = self.cipher_suite.decrypt(decoded)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return ""


class TwoFactorAuth:
    """Two-Factor Authentication management"""

    def __init__(self):
        self.users_file = Path("config/2fa_users.json")
        self.users = self._load_users()

    def _load_users(self) -> Dict:
        """Load 2FA user data"""
        try:
            if self.users_file.exists():
                with open(self.users_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load 2FA users: {e}")
        return {}

    def _save_users(self):
        """Save 2FA user data"""
        try:
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.users_file, 'w') as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving 2FA users: {e}")

    def setup_2fa(self, username: str) -> Dict:
        """
        Setup 2FA for a user

        Returns:
            Dictionary with secret, provisioning_uri, and backup_codes
        """
        try:
            # Generate secret
            secret = pyotp.random_base32()

            # Generate provisioning URI for QR code
            totp = pyotp.TOTP(secret)
            provisioning_uri = totp.provisioning_uri(
                name=username,
                issuer_name="Nexlify Trading"
            )

            # Generate backup codes
            backup_codes = [secrets.token_hex(4) for _ in range(10)]

            # Save user data
            self.users[username] = {
                'secret': secret,
                'enabled': True,
                'backup_codes': backup_codes,
                'setup_date': datetime.now().isoformat()
            }
            self._save_users()

            return {
                'secret': secret,
                'provisioning_uri': provisioning_uri,
                'backup_codes': backup_codes
            }

        except Exception as e:
            error_handler.log_error(e, f"2FA setup failed for {username}", severity="error")
            return {}

    def verify_token(self, username: str, token: str) -> bool:
        """Verify a 2FA token"""
        try:
            if username not in self.users:
                return False

            user_data = self.users[username]

            # Check if it's a backup code
            if token in user_data.get('backup_codes', []):
                # Remove used backup code
                user_data['backup_codes'].remove(token)
                self._save_users()
                logger.info(f"Backup code used for {username}")
                return True

            # Verify TOTP token
            secret = user_data.get('secret')
            if not secret:
                return False

            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=1)  # Allow 1 period window

        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False

    def generate_qr_code(self, username: str) -> Optional[bytes]:
        """Generate QR code for 2FA setup"""
        try:
            import qrcode
            from io import BytesIO

            if username not in self.users:
                setup_data = self.setup_2fa(username)
                provisioning_uri = setup_data['provisioning_uri']
            else:
                user_data = self.users[username]
                secret = user_data['secret']
                totp = pyotp.TOTP(secret)
                provisioning_uri = totp.provisioning_uri(
                    name=username,
                    issuer_name="Nexlify Trading"
                )

            # Generate QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)

            img = qr.make_image(fill_color="black", back_color="white")

            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return buffer.getvalue()

        except ImportError:
            logger.warning("qrcode library not installed, cannot generate QR code")
            return None
        except Exception as e:
            logger.error(f"QR code generation error: {e}")
            return None

    def disable_2fa(self, username: str) -> bool:
        """Disable 2FA for a user"""
        try:
            if username in self.users:
                self.users[username]['enabled'] = False
                self._save_users()
                return True
            return False
        except Exception as e:
            logger.error(f"Error disabling 2FA: {e}")
            return False


class SessionManager:
    """Secure session management"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_timeout = timedelta(minutes=30)

    def create_session(self, username: str, ip_address: str) -> str:
        """Create a new session"""
        session_token = secrets.token_urlsafe(32)

        self.sessions[session_token] = {
            'username': username,
            'ip_address': ip_address,
            'created_at': datetime.now(),
            'last_activity': datetime.now()
        }

        logger.info(f"Session created for {username} from {ip_address}")
        return session_token

    def validate_session(self, session_token: str, ip_address: str) -> bool:
        """Validate a session token"""
        if session_token not in self.sessions:
            return False

        session = self.sessions[session_token]

        # Check IP address match
        if session['ip_address'] != ip_address:
            logger.warning(f"IP mismatch for session: expected {session['ip_address']}, got {ip_address}")
            return False

        # Check timeout
        if datetime.now() - session['last_activity'] > self.session_timeout:
            logger.info(f"Session expired for {session['username']}")
            del self.sessions[session_token]
            return False

        # Update last activity
        session['last_activity'] = datetime.now()
        return True

    def destroy_session(self, session_token: str):
        """Destroy a session"""
        if session_token in self.sessions:
            username = self.sessions[session_token]['username']
            del self.sessions[session_token]
            logger.info(f"Session destroyed for {username}")


class SecurityManager:
    """Main security manager coordinating all security features"""

    def __init__(self, master_password: str = None):
        self.encryption_manager = EncryptionManager(master_password)
        self.two_factor_auth = TwoFactorAuth()
        self.session_manager = SessionManager()
        self.failed_login_attempts: Dict[str, List[datetime]] = {}
        self.lockout_duration = timedelta(minutes=15)
        self.max_attempts = 5

    @handle_errors("User Authentication", reraise=False)
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[str]:
        """
        Authenticate a user and create session

        Returns:
            Session token if successful, None otherwise
        """
        # Check if account is locked
        if self._is_account_locked(username):
            logger.warning(f"Login attempt for locked account: {username}")
            return None

        # Verify password (in production, use proper password hashing)
        # For now, this is a placeholder
        is_valid = self._verify_password(username, password)

        if not is_valid:
            self._record_failed_attempt(username)
            logger.warning(f"Failed login attempt for {username} from {ip_address}")
            return None

        # Reset failed attempts on successful login
        if username in self.failed_login_attempts:
            del self.failed_login_attempts[username]

        # Create session
        session_token = self.session_manager.create_session(username, ip_address)
        return session_token

    def _verify_password(self, username: str, password: str) -> bool:
        """Verify user password (placeholder implementation)"""
        # In production, use argon2-cffi for password hashing
        # For now, we'll use a simple check
        # This should load from encrypted storage

        # Load from config
        config_path = Path("config/neural_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                stored_pin = config.get('security', {}).get('pin', '2077')
                return password == stored_pin

        return False

    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_login_attempts:
            return False

        attempts = self.failed_login_attempts[username]

        # Remove old attempts
        cutoff_time = datetime.now() - self.lockout_duration
        attempts = [a for a in attempts if a > cutoff_time]
        self.failed_login_attempts[username] = attempts

        # Check if still locked
        return len(attempts) >= self.max_attempts

    def _record_failed_attempt(self, username: str):
        """Record a failed login attempt"""
        if username not in self.failed_login_attempts:
            self.failed_login_attempts[username] = []

        self.failed_login_attempts[username].append(datetime.now())

        # Check if account should be locked
        if len(self.failed_login_attempts[username]) >= self.max_attempts:
            logger.warning(f"Account locked due to failed attempts: {username}")

    def validate_session(self, session_token: str, ip_address: str) -> bool:
        """Validate a session"""
        return self.session_manager.validate_session(session_token, ip_address)

    def logout(self, session_token: str):
        """Logout a user"""
        self.session_manager.destroy_session(session_token)

    def encrypt_api_key(self, api_key: str) -> str:
        """Encrypt an API key"""
        return self.encryption_manager.encrypt_data(api_key)

    def decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt an API key"""
        return self.encryption_manager.decrypt_data(encrypted_key)
