#!/usr/bin/env python3
"""
Unit tests for Nexlify Security Modules
Testing security suite, PIN manager, audit trail, and integrity monitoring
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.security.nexlify_security_suite import SecuritySuite
from nexlify.security.nexlify_pin_manager import PINManager
from nexlify.security.nexlify_audit_trail import AuditTrail
from nexlify.security.nexlify_integrity_monitor import IntegrityMonitor


class TestSecuritySuite:
    """Test Security Suite functionality"""

    @pytest.fixture
    def security_suite(self, tmp_path):
        """Create security suite"""
        config = {
            "security": {
                "enabled": True,
                "encryption_enabled": True,
                "audit_enabled": True,
            }
        }
        suite = SecuritySuite(config)
        suite.data_path = tmp_path
        return suite

    def test_initialization(self, security_suite):
        """Test security suite initialization"""
        assert security_suite is not None
        assert security_suite.enabled is True

    def test_encrypt_decrypt_data(self, security_suite):
        """Test data encryption and decryption"""
        plaintext = "sensitive trading data"

        encrypted = security_suite.encrypt(plaintext)
        assert encrypted != plaintext

        decrypted = security_suite.decrypt(encrypted)
        assert decrypted == plaintext

    def test_hash_data(self, security_suite):
        """Test data hashing"""
        data = "test data"

        hash1 = security_suite.hash_data(data)
        hash2 = security_suite.hash_data(data)

        # Same data should produce same hash
        assert hash1 == hash2

        # Different data should produce different hash
        hash3 = security_suite.hash_data("different data")
        assert hash1 != hash3

    def test_verify_hash(self, security_suite):
        """Test hash verification"""
        data = "test data"
        hash_value = security_suite.hash_data(data)

        # Correct data
        assert security_suite.verify_hash(data, hash_value) is True

        # Wrong data
        assert security_suite.verify_hash("wrong data", hash_value) is False

    def test_generate_api_key(self, security_suite):
        """Test API key generation"""
        api_key = security_suite.generate_api_key()

        assert isinstance(api_key, str)
        assert len(api_key) > 0

    def test_validate_api_key(self, security_suite):
        """Test API key validation"""
        valid_key = security_suite.generate_api_key()
        security_suite.register_api_key(valid_key)

        assert security_suite.validate_api_key(valid_key) is True
        assert security_suite.validate_api_key("invalid_key") is False


class TestPINManager:
    """Test PIN Manager functionality"""

    @pytest.fixture
    def pin_manager(self, tmp_path):
        """Create PIN manager"""
        config = {
            "pin_security": {
                "enabled": True,
                "max_attempts": 3,
                "lockout_duration": 300,
            }
        }
        manager = PINManager(config)
        manager.data_path = tmp_path
        return manager

    def test_initialization(self, pin_manager):
        """Test PIN manager initialization"""
        assert pin_manager is not None
        assert pin_manager.max_attempts == 3

    def test_set_pin(self, pin_manager):
        """Test setting PIN"""
        pin = "1234"
        success = pin_manager.set_pin(pin)

        assert success is True

    def test_verify_correct_pin(self, pin_manager):
        """Test verifying correct PIN"""
        pin = "1234"
        pin_manager.set_pin(pin)

        result = pin_manager.verify_pin(pin)

        assert result is True
        assert pin_manager.failed_attempts == 0

    def test_verify_incorrect_pin(self, pin_manager):
        """Test verifying incorrect PIN"""
        pin_manager.set_pin("1234")

        result = pin_manager.verify_pin("0000")

        assert result is False
        assert pin_manager.failed_attempts == 1

    def test_lockout_after_max_attempts(self, pin_manager):
        """Test account lockout after max failed attempts"""
        pin_manager.set_pin("1234")

        # Fail max times
        for i in range(3):
            pin_manager.verify_pin("0000")

        # Should be locked out
        assert pin_manager.is_locked_out() is True

        # Even correct PIN should fail when locked
        result = pin_manager.verify_pin("1234")
        assert result is False

    def test_reset_attempts(self, pin_manager):
        """Test resetting failed attempts"""
        pin_manager.set_pin("1234")
        pin_manager.verify_pin("0000")
        pin_manager.verify_pin("0000")

        assert pin_manager.failed_attempts == 2

        pin_manager.reset_attempts()

        assert pin_manager.failed_attempts == 0

    def test_change_pin(self, pin_manager):
        """Test changing PIN"""
        old_pin = "1234"
        new_pin = "5678"

        pin_manager.set_pin(old_pin)

        success = pin_manager.change_pin(old_pin, new_pin)
        assert success is True

        # Old PIN should no longer work
        assert pin_manager.verify_pin(old_pin) is False

        # New PIN should work
        pin_manager.reset_attempts()  # Reset failed attempts
        assert pin_manager.verify_pin(new_pin) is True


class TestAuditTrail:
    """Test Audit Trail functionality"""

    @pytest.fixture
    def audit_trail(self, tmp_path):
        """Create audit trail"""
        config = {"audit": {"enabled": True, "log_path": str(tmp_path / "audit.log")}}
        return AuditTrail(config)

    def test_initialization(self, audit_trail):
        """Test audit trail initialization"""
        assert audit_trail is not None
        assert audit_trail.enabled is True

    def test_log_event(self, audit_trail):
        """Test logging audit event"""
        event = {
            "action": "trade",
            "user": "system",
            "details": {"symbol": "BTC/USDT", "amount": 0.1},
        }

        success = audit_trail.log_event(event)
        assert success is True

    def test_log_security_event(self, audit_trail):
        """Test logging security event"""
        event = {
            "action": "login_attempt",
            "user": "admin",
            "success": False,
            "ip_address": "192.168.1.1",
        }

        success = audit_trail.log_security_event(event)
        assert success is True

    def test_get_events(self, audit_trail):
        """Test retrieving audit events"""
        # Log some events
        audit_trail.log_event({"action": "test1"})
        audit_trail.log_event({"action": "test2"})

        events = audit_trail.get_events(limit=10)

        assert isinstance(events, list)
        assert len(events) >= 2

    def test_get_events_by_action(self, audit_trail):
        """Test filtering events by action"""
        audit_trail.log_event({"action": "trade", "symbol": "BTC/USDT"})
        audit_trail.log_event({"action": "login", "user": "admin"})
        audit_trail.log_event({"action": "trade", "symbol": "ETH/USDT"})

        events = audit_trail.get_events_by_action("trade")

        assert len(events) == 2
        assert all(e["action"] == "trade" for e in events)

    def test_get_events_by_user(self, audit_trail):
        """Test filtering events by user"""
        audit_trail.log_event({"action": "trade", "user": "system"})
        audit_trail.log_event({"action": "login", "user": "admin"})
        audit_trail.log_event({"action": "trade", "user": "system"})

        events = audit_trail.get_events_by_user("system")

        assert len(events) == 2
        assert all(e["user"] == "system" for e in events)

    def test_export_audit_log(self, audit_trail, tmp_path):
        """Test exporting audit log"""
        audit_trail.log_event({"action": "test"})

        export_path = tmp_path / "export.csv"
        success = audit_trail.export_log(str(export_path))

        assert success is True
        assert export_path.exists()


class TestIntegrityMonitor:
    """Test Integrity Monitor functionality"""

    @pytest.fixture
    def integrity_monitor(self, tmp_path):
        """Create integrity monitor"""
        config = {"integrity": {"enabled": True, "check_interval": 60}}
        monitor = IntegrityMonitor(config)
        monitor.data_path = tmp_path
        return monitor

    def test_initialization(self, integrity_monitor):
        """Test integrity monitor initialization"""
        assert integrity_monitor is not None
        assert integrity_monitor.enabled is True

    def test_calculate_file_hash(self, integrity_monitor, tmp_path):
        """Test calculating file hash"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        hash1 = integrity_monitor.calculate_file_hash(str(test_file))
        hash2 = integrity_monitor.calculate_file_hash(str(test_file))

        # Same file should produce same hash
        assert hash1 == hash2

    def test_register_file(self, integrity_monitor, tmp_path):
        """Test registering file for monitoring"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        success = integrity_monitor.register_file(str(test_file))
        assert success is True

    def test_verify_file_integrity_unchanged(self, integrity_monitor, tmp_path):
        """Test verifying unchanged file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        integrity_monitor.register_file(str(test_file))

        is_valid = integrity_monitor.verify_file(str(test_file))
        assert is_valid is True

    def test_verify_file_integrity_changed(self, integrity_monitor, tmp_path):
        """Test detecting changed file"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        integrity_monitor.register_file(str(test_file))

        # Modify file
        test_file.write_text("modified content")

        is_valid = integrity_monitor.verify_file(str(test_file))
        assert is_valid is False

    def test_verify_all_files(self, integrity_monitor, tmp_path):
        """Test verifying all registered files"""
        # Create and register multiple files
        file1 = tmp_path / "file1.txt"
        file1.write_text("content1")
        file2 = tmp_path / "file2.txt"
        file2.write_text("content2")

        integrity_monitor.register_file(str(file1))
        integrity_monitor.register_file(str(file2))

        results = integrity_monitor.verify_all_files()

        assert len(results) == 2
        assert all(r["valid"] for r in results)

    def test_detect_tampering(self, integrity_monitor, tmp_path):
        """Test tampering detection"""
        test_file = tmp_path / "critical.txt"
        test_file.write_text("critical data")

        integrity_monitor.register_file(str(test_file))

        # Tamper with file
        test_file.write_text("tampered data")

        tampering = integrity_monitor.detect_tampering()

        assert len(tampering) > 0
        assert str(test_file) in [t["file"] for t in tampering]


class TestIntegration:
    """Integration tests for security modules"""

    def test_full_security_workflow(self, tmp_path):
        """Test complete security workflow"""
        config = {
            "security": {"enabled": True},
            "pin_security": {"enabled": True, "max_attempts": 3},
            "audit": {"enabled": True, "log_path": str(tmp_path / "audit.log")},
        }

        # Initialize components
        security = SecuritySuite(config)
        pin_mgr = PINManager(config)
        audit = AuditTrail(config)

        # Set PIN
        pin_mgr.set_pin("1234")
        audit.log_event({"action": "pin_set", "user": "system"})

        # Encrypt sensitive data
        sensitive_data = "api_key_12345"
        encrypted = security.encrypt(sensitive_data)
        audit.log_event({"action": "data_encrypted"})

        # Verify PIN
        if pin_mgr.verify_pin("1234"):
            audit.log_security_event({"action": "pin_verified", "success": True})

            # Decrypt data
            decrypted = security.decrypt(encrypted)
            assert decrypted == sensitive_data


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_encrypt_empty_string(self):
        """Test encrypting empty string"""
        config = {"security": {"enabled": True}}
        security = SecuritySuite(config)

        encrypted = security.encrypt("")
        assert isinstance(encrypted, str)

    def test_verify_pin_before_set(self):
        """Test verifying PIN before setting it"""
        config = {"pin_security": {"enabled": True, "max_attempts": 3}}
        pin_mgr = PINManager(config)

        result = pin_mgr.verify_pin("1234")
        assert result is False

    def test_hash_none_value(self):
        """Test hashing None"""
        config = {"security": {"enabled": True}}
        security = SecuritySuite(config)

        try:
            hash_val = security.hash_data(None)
            assert isinstance(hash_val, str)
        except Exception:
            pass  # Acceptable to raise error

    def test_verify_nonexistent_file(self, tmp_path):
        """Test verifying non-existent file"""
        config = {"integrity": {"enabled": True}}
        monitor = IntegrityMonitor(config)

        result = monitor.verify_file(str(tmp_path / "nonexistent.txt"))
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
