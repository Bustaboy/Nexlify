#!/usr/bin/env python3
"""
Nexlify Security Suite - Phase 1 Integration
🛡️ Centralized security and safety management

Integrates:
- Emergency Kill Switch
- Flash Crash Protection
- PIN Authentication
- System Integrity Monitoring

This module provides a unified interface for all Phase 1 security features.
"""

import logging
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Import Phase 1 modules
from nexlify.risk.nexlify_emergency_kill_switch import EmergencyKillSwitch, KillSwitchTrigger
from nexlify.risk.nexlify_flash_crash_protection import FlashCrashProtection, CrashSeverity
from nexlify.security.nexlify_pin_manager import PINManager
from nexlify.security.nexlify_integrity_monitor import IntegrityMonitor
from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class SecuritySuite:
    """
    🛡️ Nexlify Security Suite

    Central coordinator for all security and safety features.
    Provides unified interface for:
    - Authentication (PIN-based)
    - Emergency shutdown (Kill Switch)
    - Market crash protection (Flash Crash)
    - System integrity monitoring

    Usage:
        config = load_config()
        suite = SecuritySuite(config)

        # Initialize
        await suite.initialize()

        # Authenticate
        success = suite.authenticate("user", "1234")

        # Monitor prices
        suite.update_market_price("BTC/USDT", 50000, 1000)

        # Check status
        status = suite.get_comprehensive_status()
    """

    def __init__(self, config: Dict):
        """Initialize Security Suite"""
        self.config = config

        # Initialize all Phase 1 components
        logger.info("=" * 80)
        logger.info("🛡️ INITIALIZING NEXLIFY SECURITY SUITE - PHASE 1")
        logger.info("=" * 80)

        # 1. PIN Authentication
        from nexlify.security.nexlify_advanced_security import EncryptionManager
        self.encryption_manager = EncryptionManager(
            master_password=config.get('security', {}).get('master_password', 'nexlify_2077')
        )
        self.pin_manager = PINManager(config, self.encryption_manager)
        logger.info("✅ PIN Authentication initialized")

        # 2. Emergency Kill Switch
        self.kill_switch = EmergencyKillSwitch(config)
        logger.info("✅ Emergency Kill Switch initialized")

        # 3. Flash Crash Protection
        self.flash_protection = FlashCrashProtection(config)
        logger.info("✅ Flash Crash Protection initialized")

        # 4. System Integrity Monitor
        self.integrity_monitor = IntegrityMonitor(config)
        logger.info("✅ System Integrity Monitor initialized")

        # Inject cross-dependencies
        self._inject_dependencies()

        # State
        self.is_initialized = False
        self.authenticated_user: Optional[str] = None
        self.monitoring_active = False

        logger.info("=" * 80)
        logger.info("🛡️ SECURITY SUITE READY")
        logger.info("=" * 80)

    def _inject_dependencies(self):
        """Inject cross-dependencies between components"""
        # Kill switch needs telegram bot (will be injected later)
        # Flash protection needs kill switch
        self.flash_protection.inject_dependencies(
            kill_switch=self.kill_switch
        )

        # Integrity monitor needs kill switch
        self.integrity_monitor.inject_dependencies(
            kill_switch=self.kill_switch
        )

        logger.info("✅ Cross-dependencies injected")

    def inject_external_dependencies(
        self,
        risk_manager=None,
        exchange_manager=None,
        telegram_bot=None
    ):
        """
        Inject external dependencies from main application

        Args:
            risk_manager: Risk manager instance
            exchange_manager: Exchange manager instance
            telegram_bot: Telegram bot instance
        """
        # Inject into kill switch
        self.kill_switch.inject_dependencies(
            exchange_manager=exchange_manager,
            risk_manager=risk_manager,
            telegram_bot=telegram_bot,
            security_manager=self
        )

        # Inject into flash protection
        self.flash_protection.inject_dependencies(
            kill_switch=self.kill_switch,
            risk_manager=risk_manager,
            exchange_manager=exchange_manager,
            telegram_bot=telegram_bot
        )

        # Inject into integrity monitor
        self.integrity_monitor.inject_dependencies(
            kill_switch=self.kill_switch,
            telegram_bot=telegram_bot
        )

        logger.info("✅ External dependencies injected")

    async def initialize(self):
        """Initialize the security suite"""
        if self.is_initialized:
            logger.warning("Security suite already initialized")
            return

        logger.info("🔐 Initializing security suite...")

        # Setup default user if not exists
        default_user = self.config.get('security', {}).get('default_user', 'nexlify_user')
        default_pin = self.config.get('security', {}).get('pin', '2077')

        user_info = self.pin_manager.get_user_info(default_user)
        if user_info is None:
            logger.info(f"Setting up default user: {default_user}")
            success, msg = self.pin_manager.setup_pin(default_user, default_pin)
            if success:
                logger.info(f"✅ Default user created: {default_user}")
            else:
                logger.warning(f"⚠️ Could not create default user: {msg}")

        # Create integrity baseline if not exists
        if not self.integrity_monitor.baseline:
            logger.info("Creating integrity baseline...")
            self.integrity_monitor.create_baseline()

        self.is_initialized = True
        logger.info("✅ Security suite initialized")

    def authenticate(self, username: str, pin: str, ip_address: str = "127.0.0.1") -> tuple:
        """
        Authenticate user with PIN

        Args:
            username: Username
            pin: PIN
            ip_address: IP address

        Returns:
            (success: bool, message: str)
        """
        success, message = self.pin_manager.verify_pin(username, pin, ip_address)

        if success:
            self.authenticated_user = username
            logger.info(f"✅ User authenticated: {username}")
        else:
            logger.warning(f"❌ Authentication failed: {username} - {message}")

        return success, message

    def is_authenticated(self) -> bool:
        """Check if a user is authenticated"""
        return self.authenticated_user is not None

    def logout(self):
        """Logout current user"""
        if self.authenticated_user:
            logger.info(f"👋 User logged out: {self.authenticated_user}")
            self.authenticated_user = None

    async def trigger_emergency_shutdown(
        self,
        reason: str = "Manual activation",
        trigger_type: KillSwitchTrigger = KillSwitchTrigger.MANUAL
    ) -> Dict:
        """
        Trigger emergency shutdown

        Args:
            reason: Reason for shutdown
            trigger_type: Type of trigger

        Returns:
            Result dictionary
        """
        logger.critical(f"🚨 EMERGENCY SHUTDOWN TRIGGERED: {reason}")

        result = await self.kill_switch.trigger(
            trigger_type=trigger_type,
            reason=reason,
            auto_trigger=False
        )

        return result

    def update_market_price(
        self,
        symbol: str,
        price: float,
        volume: float = 0.0
    ):
        """
        Update market price for flash crash monitoring

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            price: Current price
            volume: Trading volume
        """
        self.flash_protection.add_price_update(symbol, price, volume)

    async def check_flash_crash(self, symbol: str) -> tuple:
        """
        Check for flash crash on a symbol

        Args:
            symbol: Symbol to check

        Returns:
            (severity, details)
        """
        severity, details = self.flash_protection.detect_crash(symbol)

        # Auto-trigger protection if crash detected
        if severity != CrashSeverity.NONE and not self.flash_protection.is_protecting:
            logger.warning(f"⚡ Flash crash detected: {symbol} - {severity.value}")
            await self.flash_protection.trigger_protection(symbol, severity, details)

        return severity, details

    async def start_monitoring(self, symbols: list = None):
        """
        Start all monitoring tasks

        Args:
            symbols: List of symbols to monitor for flash crashes
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        logger.info("🔍 Starting security monitoring...")

        # Start flash crash monitoring
        if symbols:
            self.flash_protection.start_monitoring(symbols)
            logger.info(f"   ✅ Flash crash monitoring: {len(symbols)} symbols")

        # Start integrity monitoring
        self.integrity_monitor.start_monitoring()
        logger.info("   ✅ Integrity monitoring active")

        self.monitoring_active = True
        logger.info("✅ All monitoring active")

    def stop_monitoring(self):
        """Stop all monitoring tasks"""
        logger.info("⛔ Stopping security monitoring...")

        self.flash_protection.stop_monitoring()
        self.integrity_monitor.stop_monitoring()

        self.monitoring_active = False
        logger.info("✅ All monitoring stopped")

    def get_comprehensive_status(self) -> Dict:
        """Get comprehensive status of all security features"""
        return {
            'timestamp': datetime.now().isoformat(),
            'authenticated': self.is_authenticated(),
            'authenticated_user': self.authenticated_user,
            'monitoring_active': self.monitoring_active,
            'components': {
                'pin_auth': {
                    'enabled': self.config.get('pin_authentication', {}).get('enabled', True),
                    'user_info': self.pin_manager.get_user_info(self.authenticated_user) if self.authenticated_user else None
                },
                'kill_switch': self.kill_switch.get_status(),
                'flash_protection': self.flash_protection.get_status(),
                'integrity_monitor': self.integrity_monitor.get_status()
            }
        }

    def get_security_dashboard(self) -> Dict:
        """Get security dashboard data for GUI"""
        status = self.get_comprehensive_status()

        # Simplify for dashboard
        dashboard = {
            'authentication': {
                'status': 'authenticated' if status['authenticated'] else 'not_authenticated',
                'user': status['authenticated_user']
            },
            'kill_switch': {
                'active': status['components']['kill_switch']['is_active'],
                'locked': status['components']['kill_switch']['is_locked']
            },
            'flash_protection': {
                'enabled': status['components']['flash_protection']['enabled'],
                'protecting': status['components']['flash_protection']['is_protecting'],
                'severity': status['components']['flash_protection']['current_severity']
            },
            'integrity': {
                'enabled': status['components']['integrity_monitor']['enabled'],
                'last_check': status['components']['integrity_monitor']['last_check'],
                'total_violations': status['components']['integrity_monitor']['total_violations']
            },
            'monitoring': {
                'active': status['monitoring_active']
            }
        }

        return dashboard

    def get_recent_events(self, limit: int = 10) -> Dict:
        """Get recent security events"""
        return {
            'kill_switch_events': self.kill_switch.get_event_history(limit),
            'flash_crashes': self.flash_protection.get_event_history(limit),
            'integrity_violations': self.integrity_monitor.get_violation_history(limit),
            'auth_logs': self.pin_manager.get_audit_log(self.authenticated_user, limit) if self.authenticated_user else []
        }

    async def run_health_check(self) -> Dict:
        """
        Run comprehensive health check

        Returns:
            Health check results
        """
        logger.info("🏥 Running security health check...")

        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'checks': {}
        }

        # Check 1: Integrity
        logger.info("   Checking file integrity...")
        violations = self.integrity_monitor.verify_all_files()
        results['checks']['integrity'] = {
            'status': 'healthy' if len(violations) == 0 else 'unhealthy',
            'violations': len(violations)
        }
        if violations:
            results['overall_health'] = 'degraded'

        # Check 2: Kill switch
        logger.info("   Checking kill switch...")
        ks_status = self.kill_switch.get_status()
        results['checks']['kill_switch'] = {
            'status': 'armed' if not ks_status['is_active'] else 'triggered',
            'locked': ks_status['is_locked']
        }
        if ks_status['is_active']:
            results['overall_health'] = 'critical'

        # Check 3: Flash protection
        logger.info("   Checking flash protection...")
        fp_status = self.flash_protection.get_status()
        results['checks']['flash_protection'] = {
            'status': 'monitoring' if not fp_status['is_protecting'] else 'protecting',
            'severity': fp_status['current_severity']
        }
        if fp_status['is_protecting']:
            results['overall_health'] = 'warning'

        # Check 4: Authentication
        logger.info("   Checking authentication...")
        results['checks']['authentication'] = {
            'status': 'authenticated' if self.is_authenticated() else 'unauthenticated',
            'user': self.authenticated_user
        }

        logger.info(f"✅ Health check complete: {results['overall_health']}")

        return results


# Usage example and testing
if __name__ == "__main__":
    async def test_security_suite():
        """Test security suite functionality"""

        # Load config
        config_path = Path("config/neural_config.json")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Use default config for testing
            config = {
                'security': {
                    'pin': '2077',
                    'default_user': 'test_user'
                },
                'pin_authentication': {
                    'enabled': True,
                    'min_length': 4,
                    'max_length': 8
                },
                'emergency_kill_switch': {
                    'enabled': True,
                    'auto_backup': True
                },
                'flash_crash_protection': {
                    'enabled': True,
                    'critical_threshold': -0.15
                },
                'integrity_monitor': {
                    'enabled': True,
                    'critical_files': ['config/neural_config.json']
                }
            }

        # Initialize suite
        print("Initializing Security Suite...")
        suite = SecuritySuite(config)
        await suite.initialize()

        # Test authentication
        print("\nTesting authentication...")
        success, msg = suite.authenticate('test_user', '2077')
        print(f"Auth result: {success} - {msg}")

        # Test price updates
        print("\nTesting flash crash detection...")
        suite.update_market_price("BTC/USDT", 50000, 1000)
        suite.update_market_price("BTC/USDT", 48000, 1200)
        suite.update_market_price("BTC/USDT", 45000, 1500)
        severity, details = await suite.check_flash_crash("BTC/USDT")
        print(f"Flash crash severity: {severity.value}")

        # Get status
        print("\nGetting comprehensive status...")
        status = suite.get_comprehensive_status()
        print(json.dumps(status, indent=2, default=str))

        # Run health check
        print("\nRunning health check...")
        health = await suite.run_health_check()
        print(json.dumps(health, indent=2))

        print("\n✅ Security Suite test complete!")

    asyncio.run(test_security_suite())
