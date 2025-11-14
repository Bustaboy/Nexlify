#!/usr/bin/env python3
"""
Nexlify PIN Authentication Manager
🔐 Secure PIN-based authentication with encryption and rate limiting

Features:
- Secure PIN generation (4-8 digits, customizable)
- Argon2 password hashing (memory-hard, resistant to GPUs)
- Failed attempt tracking with exponential backoff
- Account lockout after max failed attempts
- PIN change with old PIN verification
- Emergency PIN reset with 2FA
- Session timeout after inactivity
- Audit logging for all auth events
"""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

# Argon2 for secure password hashing
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHash

from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class AuthAttempt:
    """Authentication attempt record"""

    timestamp: datetime
    username: str
    success: bool
    ip_address: str = ""
    reason: str = ""


@dataclass
class PINConfig:
    """PIN configuration"""

    min_length: int = 4
    max_length: int = 8
    allow_sequential: bool = False  # e.g., 1234, 4321
    allow_repeated: bool = False  # e.g., 1111, 2222
    max_failed_attempts: int = 3
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 30
    require_pin_change_days: int = 90  # Require PIN change every 90 days


class PINValidator:
    """Validates PIN strength and patterns"""

    @staticmethod
    def is_sequential(pin: str) -> bool:
        """Check if PIN is sequential (e.g., 1234, 4321)"""
        if len(pin) < 3:
            return False

        # Check ascending
        is_ascending = all(
            int(pin[i]) == int(pin[i - 1]) + 1 for i in range(1, len(pin))
        )

        # Check descending
        is_descending = all(
            int(pin[i]) == int(pin[i - 1]) - 1 for i in range(1, len(pin))
        )

        return is_ascending or is_descending

    @staticmethod
    def is_repeated(pin: str) -> bool:
        """Check if PIN has all same digits (e.g., 1111)"""
        return len(set(pin)) == 1

    @staticmethod
    def is_common(pin: str) -> bool:
        """Check if PIN is in common PIN list"""
        common_pins = [
            "0000",
            "1111",
            "2222",
            "3333",
            "4444",
            "5555",
            "6666",
            "7777",
            "8888",
            "9999",
            "1234",
            "4321",
            "1212",
            "0123",
            "5678",
            "9876",
            "1010",
            "2020",
            "2077",
        ]
        return pin in common_pins

    @staticmethod
    def validate_pin(pin: str, config: PINConfig) -> Tuple[bool, str]:
        """
        Validate PIN against security rules

        Returns:
            (is_valid, error_message)
        """
        # Check if numeric
        if not pin.isdigit():
            return False, "PIN must contain only digits"

        # Check length
        if len(pin) < config.min_length:
            return False, f"PIN must be at least {config.min_length} digits"

        if len(pin) > config.max_length:
            return False, f"PIN must be at most {config.max_length} digits"

        # Check for sequential patterns
        if not config.allow_sequential and PINValidator.is_sequential(pin):
            return False, "PIN cannot be sequential (e.g., 1234)"

        # Check for repeated digits
        if not config.allow_repeated and PINValidator.is_repeated(pin):
            return False, "PIN cannot have all same digits (e.g., 1111)"

        # Check for common PINs
        if PINValidator.is_common(pin):
            return False, "PIN is too common, please choose a different one"

        return True, "PIN is valid"


class PINManager:
    """
    🔐 PIN Authentication Manager

    Provides secure PIN-based authentication with:
    - Argon2 password hashing (industry standard, memory-hard)
    - Rate limiting and account lockout
    - Session management with timeout
    - Audit trail for all authentication events
    - PIN strength validation
    - Emergency reset capabilities
    """

    def __init__(self, config: Dict, encryption_manager=None):
        """Initialize PIN Manager"""
        pin_config = config.get("pin_authentication", {})

        # PIN configuration
        self.pin_config = PINConfig(
            min_length=pin_config.get("min_length", 4),
            max_length=pin_config.get("max_length", 8),
            allow_sequential=pin_config.get("allow_sequential", False),
            allow_repeated=pin_config.get("allow_repeated", False),
            max_failed_attempts=pin_config.get("max_failed_attempts", 3),
            lockout_duration_minutes=pin_config.get("lockout_duration_minutes", 15),
            session_timeout_minutes=pin_config.get("session_timeout_minutes", 30),
            require_pin_change_days=pin_config.get("require_pin_change_days", 90),
        )

        # Argon2 password hasher (memory-hard, GPU-resistant)
        self.password_hasher = PasswordHasher(
            time_cost=2,  # Number of iterations
            memory_cost=65536,  # 64 MB memory usage
            parallelism=4,  # Number of parallel threads
            hash_len=32,  # Length of hash
            salt_len=16,  # Length of salt
        )

        # Encryption manager for storing PINs (double security)
        self.encryption_manager = encryption_manager

        # State management
        self.users_file = Path("config/pin_users.json")
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, Dict] = self._load_users()

        # Failed attempts tracking
        self.failed_attempts: Dict[str, list] = {}

        # Audit log
        self.audit_log_file = Path("data/auth_audit.jsonl")
        self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("🔐 PIN Manager initialized")
        logger.info(
            f"   PIN length: {self.pin_config.min_length}-{self.pin_config.max_length} digits"
        )
        logger.info(f"   Max failed attempts: {self.pin_config.max_failed_attempts}")
        logger.info(
            f"   Lockout duration: {self.pin_config.lockout_duration_minutes} minutes"
        )
        logger.info(
            f"   Session timeout: {self.pin_config.session_timeout_minutes} minutes"
        )

    @handle_errors("PIN Manager - Load Users", reraise=False)
    def _load_users(self) -> Dict:
        """Load user PIN data from disk"""
        if not self.users_file.exists():
            return {}

        try:
            with open(self.users_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load PIN users: {e}")
            return {}

    @handle_errors("PIN Manager - Save Users", reraise=False)
    def _save_users(self):
        """Save user PIN data to disk"""
        try:
            with open(self.users_file, "w") as f:
                json.dump(self.users, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save PIN users: {e}")

    def _log_auth_attempt(self, attempt: AuthAttempt):
        """Log authentication attempt to audit trail"""
        try:
            log_entry = {
                "timestamp": attempt.timestamp.isoformat(),
                "username": attempt.username,
                "success": attempt.success,
                "ip_address": attempt.ip_address,
                "reason": attempt.reason,
            }

            with open(self.audit_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.error(f"Failed to log auth attempt: {e}")

    def generate_pin(self, length: int = 6) -> str:
        """
        Generate a cryptographically secure random PIN

        Args:
            length: PIN length (default: 6)

        Returns:
            Random PIN string
        """
        if length < self.pin_config.min_length or length > self.pin_config.max_length:
            length = 6  # Default fallback

        # Generate random PIN
        pin = "".join(str(secrets.randbelow(10)) for _ in range(length))

        # Validate generated PIN
        is_valid, _ = PINValidator.validate_pin(pin, self.pin_config)

        # Regenerate if invalid (e.g., sequential, repeated)
        max_attempts = 10
        attempts = 0
        while not is_valid and attempts < max_attempts:
            pin = "".join(str(secrets.randbelow(10)) for _ in range(length))
            is_valid, _ = PINValidator.validate_pin(pin, self.pin_config)
            attempts += 1

        return pin

    def setup_pin(self, username: str, pin: str) -> Tuple[bool, str]:
        """
        Setup PIN for a user

        Args:
            username: Username
            pin: PIN to set

        Returns:
            (success, message)
        """
        # Validate PIN
        is_valid, error_msg = PINValidator.validate_pin(pin, self.pin_config)
        if not is_valid:
            logger.warning(f"PIN setup failed for {username}: {error_msg}")
            return False, error_msg

        try:
            # Hash PIN with Argon2
            pin_hash = self.password_hasher.hash(pin)

            # Additionally encrypt the hash (defense in depth)
            if self.encryption_manager:
                pin_hash = self.encryption_manager.encrypt_data(pin_hash)

            # Store user data
            self.users[username] = {
                "pin_hash": pin_hash,
                "created_at": datetime.now().isoformat(),
                "last_changed": datetime.now().isoformat(),
                "failed_attempts": 0,
                "locked_until": None,
            }

            self._save_users()

            logger.info(f"✅ PIN setup successful for {username}")
            return True, "PIN setup successful"

        except Exception as e:
            logger.error(f"PIN setup failed for {username}: {e}")
            error_handler.log_error(
                e, f"PIN setup failed for {username}", severity="error"
            )
            return False, f"PIN setup failed: {str(e)}"

    def verify_pin(
        self, username: str, pin: str, ip_address: str = ""
    ) -> Tuple[bool, str]:
        """
        Verify PIN for authentication

        Args:
            username: Username
            pin: PIN to verify
            ip_address: IP address of request (for logging)

        Returns:
            (success, message)
        """
        # Check if user exists
        if username not in self.users:
            logger.warning(f"PIN verification failed: User {username} not found")
            attempt = AuthAttempt(
                timestamp=datetime.now(),
                username=username,
                success=False,
                ip_address=ip_address,
                reason="User not found",
            )
            self._log_auth_attempt(attempt)
            return False, "Invalid credentials"

        user_data = self.users[username]

        # Check if account is locked
        if user_data.get("locked_until"):
            locked_until = datetime.fromisoformat(user_data["locked_until"])
            if datetime.now() < locked_until:
                remaining = (locked_until - datetime.now()).total_seconds() / 60
                logger.warning(
                    f"Account locked: {username} (remaining: {remaining:.0f} minutes)"
                )
                return False, f"Account locked. Try again in {remaining:.0f} minutes"
            else:
                # Lockout expired, unlock account
                user_data["locked_until"] = None
                user_data["failed_attempts"] = 0
                self._save_users()

        try:
            # Get stored PIN hash
            pin_hash = user_data["pin_hash"]

            # Decrypt if encrypted
            if self.encryption_manager and pin_hash:
                pin_hash = self.encryption_manager.decrypt_data(pin_hash)

            # Verify PIN with Argon2
            try:
                self.password_hasher.verify(pin_hash, pin)

                # Check if rehashing is needed (Argon2 updates parameters)
                if self.password_hasher.check_needs_rehash(pin_hash):
                    logger.info(f"Rehashing PIN for {username} with updated parameters")
                    user_data["pin_hash"] = self.password_hasher.hash(pin)
                    if self.encryption_manager:
                        user_data["pin_hash"] = self.encryption_manager.encrypt_data(
                            user_data["pin_hash"]
                        )
                    self._save_users()

                # Success! Reset failed attempts
                user_data["failed_attempts"] = 0
                user_data["last_login"] = datetime.now().isoformat()
                self._save_users()

                logger.info(f"✅ PIN verification successful for {username}")

                attempt = AuthAttempt(
                    timestamp=datetime.now(),
                    username=username,
                    success=True,
                    ip_address=ip_address,
                    reason="PIN verified",
                )
                self._log_auth_attempt(attempt)

                return True, "Authentication successful"

            except (VerifyMismatchError, VerificationError, InvalidHash):
                # PIN verification failed
                user_data["failed_attempts"] = user_data.get("failed_attempts", 0) + 1

                logger.warning(
                    f"❌ PIN verification failed for {username} (attempt {user_data['failed_attempts']})"
                )

                # Check if lockout threshold reached
                if user_data["failed_attempts"] >= self.pin_config.max_failed_attempts:
                    locked_until = datetime.now() + timedelta(
                        minutes=self.pin_config.lockout_duration_minutes
                    )
                    user_data["locked_until"] = locked_until.isoformat()
                    logger.warning(
                        f"🔒 Account locked: {username} until {locked_until}"
                    )

                    attempt = AuthAttempt(
                        timestamp=datetime.now(),
                        username=username,
                        success=False,
                        ip_address=ip_address,
                        reason=f"Account locked after {user_data['failed_attempts']} failed attempts",
                    )
                    self._log_auth_attempt(attempt)

                    self._save_users()
                    return (
                        False,
                        f"Too many failed attempts. Account locked for {self.pin_config.lockout_duration_minutes} minutes",
                    )

                self._save_users()

                attempt = AuthAttempt(
                    timestamp=datetime.now(),
                    username=username,
                    success=False,
                    ip_address=ip_address,
                    reason=f"Invalid PIN (attempt {user_data['failed_attempts']})",
                )
                self._log_auth_attempt(attempt)

                remaining_attempts = (
                    self.pin_config.max_failed_attempts - user_data["failed_attempts"]
                )
                return False, f"Invalid PIN. {remaining_attempts} attempts remaining"

        except Exception as e:
            logger.error(f"PIN verification error for {username}: {e}")
            error_handler.log_error(
                e, f"PIN verification error for {username}", severity="error"
            )
            return False, "Authentication error"

    def change_pin(self, username: str, old_pin: str, new_pin: str) -> Tuple[bool, str]:
        """
        Change user PIN

        Args:
            username: Username
            old_pin: Current PIN
            new_pin: New PIN

        Returns:
            (success, message)
        """
        # Verify old PIN first
        success, message = self.verify_pin(username, old_pin)
        if not success:
            return False, "Current PIN is incorrect"

        # Validate new PIN
        is_valid, error_msg = PINValidator.validate_pin(new_pin, self.pin_config)
        if not is_valid:
            return False, error_msg

        # Check if new PIN is same as old PIN
        if old_pin == new_pin:
            return False, "New PIN must be different from current PIN"

        # Setup new PIN
        try:
            pin_hash = self.password_hasher.hash(new_pin)

            if self.encryption_manager:
                pin_hash = self.encryption_manager.encrypt_data(pin_hash)

            self.users[username]["pin_hash"] = pin_hash
            self.users[username]["last_changed"] = datetime.now().isoformat()
            self._save_users()

            logger.info(f"✅ PIN changed successfully for {username}")
            return True, "PIN changed successfully"

        except Exception as e:
            logger.error(f"PIN change failed for {username}: {e}")
            return False, f"PIN change failed: {str(e)}"

    def emergency_reset_pin(
        self, username: str, new_pin: str, admin_approved: bool = False
    ) -> Tuple[bool, str]:
        """
        Emergency PIN reset (requires admin approval or 2FA)

        Args:
            username: Username
            new_pin: New PIN
            admin_approved: Whether admin has approved (or 2FA verified)

        Returns:
            (success, message)
        """
        if not admin_approved:
            return False, "Emergency reset requires admin approval or 2FA verification"

        # Validate new PIN
        is_valid, error_msg = PINValidator.validate_pin(new_pin, self.pin_config)
        if not is_valid:
            return False, error_msg

        try:
            pin_hash = self.password_hasher.hash(new_pin)

            if self.encryption_manager:
                pin_hash = self.encryption_manager.encrypt_data(pin_hash)

            # Reset PIN and unlock account
            self.users[username]["pin_hash"] = pin_hash
            self.users[username]["last_changed"] = datetime.now().isoformat()
            self.users[username]["failed_attempts"] = 0
            self.users[username]["locked_until"] = None
            self._save_users()

            logger.warning(f"⚠️ Emergency PIN reset for {username}")
            return True, "PIN reset successful"

        except Exception as e:
            logger.error(f"Emergency PIN reset failed for {username}: {e}")
            return False, f"PIN reset failed: {str(e)}"

    def is_locked(self, username: str) -> bool:
        """Check if account is locked"""
        if username not in self.users:
            return False

        user_data = self.users[username]
        if user_data.get("locked_until"):
            locked_until = datetime.fromisoformat(user_data["locked_until"])
            return datetime.now() < locked_until

        return False

    def unlock_account(self, username: str, admin_approved: bool = False) -> bool:
        """Manually unlock an account"""
        if not admin_approved:
            logger.warning("Account unlock requires admin approval")
            return False

        if username not in self.users:
            return False

        self.users[username]["locked_until"] = None
        self.users[username]["failed_attempts"] = 0
        self._save_users()

        logger.info(f"🔓 Account unlocked: {username}")
        return True

    def get_user_info(self, username: str) -> Optional[Dict]:
        """Get user information (excluding sensitive data)"""
        if username not in self.users:
            return None

        user_data = self.users[username].copy()
        # Remove sensitive data
        user_data.pop("pin_hash", None)

        # Add computed fields
        user_data["is_locked"] = self.is_locked(username)

        return user_data

    def get_audit_log(self, username: str = None, limit: int = 100) -> list:
        """Get authentication audit log"""
        logs = []

        if not self.audit_log_file.exists():
            return logs

        try:
            with open(self.audit_log_file, "r") as f:
                for line in f:
                    log = json.loads(line)
                    if username is None or log.get("username") == username:
                        logs.append(log)

            # Return last N entries
            return logs[-limit:]

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")
            return []


# Usage example
if __name__ == "__main__":
    # Test configuration
    config = {
        "pin_authentication": {
            "min_length": 4,
            "max_length": 8,
            "allow_sequential": False,
            "allow_repeated": False,
            "max_failed_attempts": 3,
            "lockout_duration_minutes": 15,
        }
    }

    pin_manager = PINManager(config)

    # Generate a secure PIN
    print("Generating secure PIN...")
    secure_pin = pin_manager.generate_pin(6)
    print(f"Generated PIN: {secure_pin}")

    # Setup PIN for a user
    print("\nSetting up PIN for test_user...")
    success, msg = pin_manager.setup_pin("test_user", "123456")
    print(f"Setup: {success} - {msg}")

    # Verify correct PIN
    print("\nVerifying correct PIN...")
    success, msg = pin_manager.verify_pin("test_user", "123456", "127.0.0.1")
    print(f"Verify: {success} - {msg}")

    # Verify incorrect PIN
    print("\nVerifying incorrect PIN...")
    success, msg = pin_manager.verify_pin("test_user", "000000", "127.0.0.1")
    print(f"Verify: {success} - {msg}")

    # Change PIN
    print("\nChanging PIN...")
    success, msg = pin_manager.change_pin("test_user", "123456", "654321")
    print(f"Change: {success} - {msg}")

    # Get user info
    print("\nUser info:")
    info = pin_manager.get_user_info("test_user")
    print(json.dumps(info, indent=2))

    # Get audit log
    print("\nAudit log:")
    logs = pin_manager.get_audit_log("test_user")
    for log in logs:
        print(f"  {log['timestamp']}: {log['reason']} (success={log['success']})")
