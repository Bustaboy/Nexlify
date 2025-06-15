#!/usr/bin/env python3
"""
Basic tests for Nexlify Security Module
Run with: python -m pytest tests/test_security.py
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime, timedelta

# Mock config for testing
TEST_CONFIG = {
    "version": "2.0.8",
    "security": {
        "master_password_enabled": False,
        "master_password": "",
        "2fa_enabled": False,
        "2fa_secret": "",
        "session_timeout_minutes": 60,
        "ip_whitelist_enabled": False,
        "ip_whitelist": [],
        "max_failed_attempts": 5,
        "lockout_duration_minutes": 30,
        "api_key_rotation_days": 30
    }
}

@pytest.fixture
def temp_config_dir():
    """Create temporary config directory"""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir) / "config"
    config_dir.mkdir()
    
    # Write test config
    config_file = config_dir / "enhanced_config.json"
    with open(config_file, 'w') as f:
        json.dump(TEST_CONFIG, f)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def security_manager(temp_config_dir, monkeypatch):
    """Create security manager with test config"""
    monkeypatch.chdir(temp_config_dir)
    
    from src.security.nexlify_advanced_security import SecurityManager
    return SecurityManager(str(Path(temp_config_dir) / "config" / "enhanced_config.json"))

class TestEncryption:
    """Test encryption functionality"""
    
    def test_encrypt_decrypt_string(self, security_manager):
        """Test basic string encryption"""
        original = "secret_api_key_12345"
        
        encrypted = security_manager.encryption.encrypt_data(original)
        assert encrypted != original
        assert len(encrypted) > len(original)
        
        decrypted = security_manager.encryption.decrypt_data(encrypted)
        assert decrypted == original
    
    def test_encrypt_decrypt_config(self, security_manager):
        """Test config encryption"""
        config = {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "nested": {
                "value": 12345
            }
        }
        
        encrypted = security_manager.encryption.encrypt_config(config)
        decrypted = security_manager.encryption.decrypt_config(encrypted)
        
        assert decrypted == config
        assert decrypted["nested"]["value"] == 12345

class TestAuthentication:
    """Test authentication functionality"""
    
    def test_no_master_password_login(self, security_manager):
        """Test login without master password"""
        # Should succeed with any password when master password disabled
        token = security_manager.authenticate_user(
            username="testuser",
            password="anypassword",
            ip_address="127.0.0.1"
        )
        
        assert token is not None
        assert len(token) > 50  # JWT tokens are long
    
    def test_session_validation(self, security_manager):
        """Test session validation"""
        # Create session
        token = security_manager.authenticate_user(
            username="testuser",
            password="password",
            ip_address="127.0.0.1"
        )
        
        # Validate from same IP
        session = security_manager.validate_session(token, "127.0.0.1")
        assert session is not None
        assert session.user_id == "testuser"
        
        # Invalidate session
        security_manager.invalidate_session(token)
        
        # Should fail after invalidation
        session = security_manager.validate_session(token, "127.0.0.1")
        assert session is None
    
    def test_failed_login_lockout(self, security_manager):
        """Test lockout after failed attempts"""
        # Enable master password for this test
        security_manager.config['security']['master_password_enabled'] = True
        
        # First login sets password
        token = security_manager.authenticate_user(
            username="testuser",
            password="correct_password",
            ip_address="192.168.1.100"
        )
        assert token is not None
        
        # Logout
        security_manager.invalidate_session(token)
        
        # Try wrong password multiple times
        for i in range(6):
            token = security_manager.authenticate_user(
                username="testuser",
                password="wrong_password",
                ip_address="192.168.1.100"
            )
            assert token is None
        
        # Should be locked out now
        token = security_manager.authenticate_user(
            username="testuser",
            password="correct_password",
            ip_address="192.168.1.100"
        )
        assert token is None  # Locked out

class TestTwoFactor:
    """Test 2FA functionality"""
    
    def test_2fa_setup(self, security_manager):
        """Test 2FA setup"""
        secret, qr_code = security_manager.two_factor.setup_2fa("testuser")
        
        assert len(secret) == 32  # Base32 secret
        assert qr_code.startswith("iVBOR")  # PNG image header in base64
        
        # Check backup codes
        user_data = security_manager.two_factor.users["testuser"]
        assert len(user_data["backup_codes"]) == 10
    
    def test_2fa_verification(self, security_manager):
        """Test 2FA token verification"""
        import pyotp
        
        # Setup 2FA
        secret, _ = security_manager.two_factor.setup_2fa("testuser")
        
        # Generate valid token
        totp = pyotp.TOTP(secret)
        valid_token = totp.now()
        
        # Verify valid token
        assert security_manager.two_factor.verify_2fa("testuser", valid_token) is True
        
        # Verify invalid token
        assert security_manager.two_factor.verify_2fa("testuser", "000000") is False
    
    def test_backup_codes(self, security_manager):
        """Test backup code usage"""
        # Setup 2FA
        security_manager.two_factor.setup_2fa("testuser")
        
        # Get a backup code
        backup_code = security_manager.two_factor.users["testuser"]["backup_codes"][0]
        
        # Use backup code
        assert security_manager.two_factor.verify_2fa("testuser", backup_code) is True
        
        # Backup code should be removed after use
        assert backup_code not in security_manager.two_factor.users["testuser"]["backup_codes"]

class TestAPIKeyRotation:
    """Test API key rotation"""
    
    def test_add_api_key(self, security_manager):
        """Test adding API keys"""
        # Binance format
        security_manager.api_rotation.add_key(
            "binance",
            "A" * 64,  # 64 char key
            "B" * 64   # 64 char secret
        )
        
        keys = security_manager.api_rotation.get_active_keys("binance")
        assert keys is not None
        assert keys["api_key"] == "A" * 64
    
    def test_invalid_api_key_format(self, security_manager):
        """Test API key validation"""
        with pytest.raises(ValueError):
            security_manager.api_rotation.add_key(
                "binance",
                "short",  # Too short
                "B" * 64
            )
    
    def test_key_rotation(self, security_manager):
        """Test key rotation process"""
        # Add initial key
        security_manager.api_rotation.add_key(
            "binance",
            "A" * 64,
            "B" * 64
        )
        
        # Rotate to new key
        success = security_manager.api_rotation.rotate_keys(
            "binance",
            "C" * 64,
            "D" * 64
        )
        
        assert success is True
        
        # Check history
        assert len(security_manager.api_rotation.key_history) > 0
        assert security_manager.api_rotation.key_history[-1]["action"] == "rotated"

class TestAccessControl:
    """Test access control features"""
    
    def test_ip_whitelist(self, security_manager):
        """Test IP whitelisting"""
        # Enable whitelist
        security_manager.config['security']['ip_whitelist_enabled'] = True
        
        # Add IPs
        security_manager.access_control.add_to_whitelist("192.168.1.100")
        security_manager.access_control.add_to_whitelist("10.0.0.0/24")
        
        # Test access
        allowed, reason = security_manager.access_control.check_access("192.168.1.100")
        assert allowed is True
        
        allowed, reason = security_manager.access_control.check_access("10.0.0.50")
        assert allowed is True
        
        allowed, reason = security_manager.access_control.check_access("8.8.8.8")
        assert allowed is False
        assert reason == "IP not whitelisted"
    
    def test_rate_limiting(self, security_manager):
        """Test rate limiting"""
        ip = "192.168.1.200"
        
        # Make requests up to limit
        for i in range(60):
            allowed, _ = security_manager.access_control.check_access(ip)
            assert allowed is True
        
        # Next request should be rate limited
        allowed, reason = security_manager.access_control.check_access(ip)
        assert allowed is False
        assert reason == "Rate limit exceeded"

class TestSecurityEvents:
    """Test security event logging"""
    
    def test_security_event_logging(self, security_manager):
        """Test event logging"""
        # Generate some events
        security_manager.log_security_event(
            "test_event",
            user_id="testuser",
            ip_address="127.0.0.1",
            details={"test": True},
            severity="info"
        )
        
        # Check event was logged
        assert len(security_manager.security_events) > 0
        
        last_event = security_manager.security_events[-1]
        assert last_event.event_type == "test_event"
        assert last_event.user_id == "testuser"
        assert last_event.details["test"] is True
    
    def test_security_summary(self, security_manager):
        """Test security summary generation"""
        # Create some test data
        security_manager.authenticate_user("user1", "pass", "127.0.0.1")
        security_manager.authenticate_user("user2", "pass", "127.0.0.2")
        
        summary = security_manager.get_security_summary()
        
        assert "active_sessions" in summary
        assert "security_enabled" in summary
        assert summary["active_sessions"] >= 2
        assert summary["security_enabled"]["master_password"] is False

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
