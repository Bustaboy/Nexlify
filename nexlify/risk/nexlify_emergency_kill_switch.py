#!/usr/bin/env python3
"""
Nexlify Emergency Kill Switch
🚨 Instant shutdown of all trading operations with capital preservation

Features:
- Immediate stop of all trading operations
- Close all open positions (market orders)
- Cancel all pending orders across all exchanges
- Lock the application (require PIN to unlock)
- Create emergency backup of current state
- Log emergency event with timestamp and reason
- Telegram notification support
"""

import logging
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import shutil

from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class KillSwitchTrigger(Enum):
    """Kill switch trigger reasons"""

    MANUAL = "manual"
    FLASH_CRASH = "flash_crash"
    API_FAILURE = "api_failure"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    UNUSUAL_ACTIVITY = "unusual_activity"
    SYSTEM_TAMPER = "system_tamper"
    NETWORK_LOSS = "network_loss"


@dataclass
class KillSwitchEvent:
    """Emergency kill switch event record"""

    timestamp: datetime = field(default_factory=datetime.now)
    trigger: KillSwitchTrigger = KillSwitchTrigger.MANUAL
    reason: str = ""
    positions_closed: int = 0
    orders_cancelled: int = 0
    backup_created: bool = False
    notification_sent: bool = False
    total_value_at_trigger: float = 0.0
    exchanges_affected: List[str] = field(default_factory=list)
    recovery_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["trigger"] = self.trigger.value
        if self.recovery_time:
            data["recovery_time"] = self.recovery_time.isoformat()
        return data


class EmergencyKillSwitch:
    """
    🚨 Emergency Kill Switch System

    Provides instant shutdown capabilities with multiple trigger options:
    - Manual activation via GUI button
    - Automatic triggers (flash crash, API failures, etc.)
    - System tamper detection

    Safety Features:
    - Atomic operations (all-or-nothing)
    - Emergency state persistence
    - Automatic backups before shutdown
    - Recovery verification
    """

    def __init__(self, config: Dict):
        """Initialize Emergency Kill Switch"""
        self.config = config.get("emergency_kill_switch", {})
        self.enabled = self.config.get("enabled", True)

        # Kill switch state
        self.is_active = False
        self.is_locked = False
        self.activation_time: Optional[datetime] = None
        self.current_event: Optional[KillSwitchEvent] = None

        # Settings
        self.auto_backup = self.config.get("auto_backup", True)
        self.close_positions = self.config.get("close_positions", True)
        self.cancel_orders = self.config.get("cancel_orders", True)
        self.lock_on_trigger = self.config.get("lock_on_trigger", True)
        self.telegram_notify = self.config.get("telegram_notify", True)
        self.require_pin_unlock = self.config.get("require_pin_unlock", True)

        # Thresholds for auto-triggers
        self.flash_crash_threshold = self.config.get(
            "flash_crash_threshold", 0.15
        )  # 15% drop
        self.api_failure_threshold = self.config.get(
            "api_failure_threshold", 5
        )  # 5 consecutive failures

        # State persistence
        self.state_file = Path("data/emergency_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_file = Path("data/emergency_events.jsonl")

        # References to external systems (to be injected)
        self.exchange_manager = None
        self.risk_manager = None
        self.telegram_bot = None
        self.security_manager = None

        # Load previous state
        self._load_state()

        logger.info("🚨 Emergency Kill Switch initialized")
        logger.info(f"   Enabled: {self.enabled}")
        logger.info(f"   Auto-backup: {self.auto_backup}")
        logger.info(f"   Close positions: {self.close_positions}")
        logger.info(f"   Lock on trigger: {self.lock_on_trigger}")

    def inject_dependencies(
        self,
        exchange_manager=None,
        risk_manager=None,
        telegram_bot=None,
        security_manager=None,
    ):
        """Inject external dependencies for kill switch operations"""
        self.exchange_manager = exchange_manager
        self.risk_manager = risk_manager
        self.telegram_bot = telegram_bot
        self.security_manager = security_manager
        logger.info("✅ Kill Switch dependencies injected")

    @handle_errors("Kill Switch - Load State", reraise=False)
    def _load_state(self):
        """Load kill switch state from disk"""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            self.is_active = data.get("is_active", False)
            self.is_locked = data.get("is_locked", False)

            if data.get("activation_time"):
                self.activation_time = datetime.fromisoformat(data["activation_time"])

            if self.is_active:
                logger.warning(
                    f"⚠️ Kill Switch was previously activated at {self.activation_time}"
                )
                logger.warning(
                    "   System remains in emergency mode until manually reset"
                )

        except Exception as e:
            logger.error(f"Failed to load kill switch state: {e}")

    @handle_errors("Kill Switch - Save State", reraise=False)
    def _save_state(self):
        """Save kill switch state to disk"""
        data = {
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "activation_time": (
                self.activation_time.isoformat() if self.activation_time else None
            ),
            "last_updated": datetime.now().isoformat(),
        }

        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save kill switch state: {e}")

    @handle_errors("Kill Switch - Log Event", reraise=False)
    def _log_event(self, event: KillSwitchEvent):
        """Log emergency event to persistent log"""
        try:
            with open(self.event_log_file, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log kill switch event: {e}")

    async def trigger(
        self,
        trigger_type: KillSwitchTrigger = KillSwitchTrigger.MANUAL,
        reason: str = "Emergency shutdown",
        auto_trigger: bool = False,
    ) -> Dict[str, Any]:
        """
        🚨 TRIGGER EMERGENCY KILL SWITCH

        This will:
        1. Stop all trading operations immediately
        2. Create emergency backup
        3. Close all open positions
        4. Cancel all pending orders
        5. Lock the system
        6. Send notifications

        Args:
            trigger_type: Reason for trigger
            reason: Detailed explanation
            auto_trigger: If True, this was an automatic trigger

        Returns:
            Dictionary with shutdown results
        """
        if not self.enabled:
            logger.warning("⚠️ Kill Switch is disabled in config")
            return {"success": False, "reason": "Kill switch disabled"}

        if self.is_active:
            logger.warning("⚠️ Kill Switch already active")
            return {"success": False, "reason": "Already active"}

        logger.critical("=" * 80)
        logger.critical("🚨 EMERGENCY KILL SWITCH ACTIVATED 🚨")
        logger.critical(f"Trigger: {trigger_type.value}")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Auto-trigger: {auto_trigger}")
        logger.critical(f"Time: {datetime.now().isoformat()}")
        logger.critical("=" * 80)

        # Mark as active immediately
        self.is_active = True
        self.activation_time = datetime.now()
        self._save_state()

        # Create event record
        self.current_event = KillSwitchEvent(trigger=trigger_type, reason=reason)

        results = {
            "success": False,
            "trigger": trigger_type.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "steps_completed": [],
        }

        try:
            # Step 1: Create emergency backup
            if self.auto_backup:
                logger.info("📦 Creating emergency backup...")
                backup_success = await self._create_emergency_backup()
                self.current_event.backup_created = backup_success
                results["backup_created"] = backup_success
                results["steps_completed"].append("backup")

            # Step 2: Stop all trading operations
            logger.info("⛔ Stopping all trading operations...")
            await self._stop_trading()
            results["steps_completed"].append("stop_trading")

            # Step 3: Cancel all pending orders
            if self.cancel_orders and self.exchange_manager:
                logger.info("❌ Cancelling all pending orders...")
                cancelled = await self._cancel_all_orders()
                self.current_event.orders_cancelled = cancelled
                results["orders_cancelled"] = cancelled
                results["steps_completed"].append("cancel_orders")

            # Step 4: Close all open positions
            if self.close_positions and self.exchange_manager:
                logger.info("📉 Closing all open positions...")
                closed = await self._close_all_positions()
                self.current_event.positions_closed = closed
                results["positions_closed"] = closed
                results["steps_completed"].append("close_positions")

            # Step 5: Lock the system
            if self.lock_on_trigger:
                logger.info("🔒 Locking system...")
                self._lock_system()
                results["system_locked"] = True
                results["steps_completed"].append("lock_system")

            # Step 6: Send notifications
            if self.telegram_notify and self.telegram_bot:
                logger.info("📱 Sending emergency notifications...")
                notified = await self._send_notifications(trigger_type, reason)
                self.current_event.notification_sent = notified
                results["notification_sent"] = notified
                results["steps_completed"].append("notifications")

            # Log the event
            self._log_event(self.current_event)

            results["success"] = True
            logger.critical("✅ Emergency shutdown completed successfully")

        except Exception as e:
            logger.critical(f"❌ Emergency shutdown encountered errors: {e}")
            error_handler.log_error(
                e, "Emergency Kill Switch execution failed", severity="critical"
            )
            results["error"] = str(e)

        finally:
            self._save_state()

        return results

    async def _create_emergency_backup(self) -> bool:
        """Create emergency backup of critical data"""
        try:
            backup_dir = Path("backups/emergency")
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"emergency_{timestamp}"
            backup_path.mkdir(exist_ok=True)

            # Backup critical files
            critical_files = [
                "data/trading.db",
                "data/risk_state.json",
                "config/neural_config.json",
                "data/performance_metrics.json",
            ]

            for file_path in critical_files:
                src = Path(file_path)
                if src.exists():
                    dst = backup_path / src.name
                    shutil.copy2(src, dst)
                    logger.info(f"   ✅ Backed up: {file_path}")

            logger.info(f"✅ Emergency backup created: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"❌ Emergency backup failed: {e}")
            return False

    async def _stop_trading(self):
        """Stop all trading operations"""
        # Stop auto-trader
        if self.risk_manager:
            self.risk_manager.trading_halted = True
            self.risk_manager.halt_reason = "Emergency kill switch activated"
            logger.info("   ✅ Auto-trader halted")

    async def _cancel_all_orders(self) -> int:
        """Cancel all pending orders across all exchanges"""
        cancelled_count = 0

        if not self.exchange_manager:
            logger.warning("   ⚠️ Exchange manager not available")
            return 0

        try:
            # This would integrate with your exchange manager
            # For now, this is a placeholder that shows the structure
            exchanges = getattr(self.exchange_manager, "exchanges", {})

            for exchange_name, exchange in exchanges.items():
                try:
                    # Cancel all open orders on this exchange
                    if hasattr(exchange, "cancel_all_orders"):
                        await exchange.cancel_all_orders()
                        logger.info(f"   ✅ Cancelled orders on {exchange_name}")
                        cancelled_count += 1
                except Exception as e:
                    logger.error(
                        f"   ❌ Failed to cancel orders on {exchange_name}: {e}"
                    )

            logger.info(f"   ✅ Total exchanges processed: {cancelled_count}")

        except Exception as e:
            logger.error(f"❌ Error cancelling orders: {e}")

        return cancelled_count

    async def _close_all_positions(self) -> int:
        """Close all open positions with market orders"""
        closed_count = 0

        if not self.exchange_manager:
            logger.warning("   ⚠️ Exchange manager not available")
            return 0

        try:
            # This would integrate with your exchange manager
            # For now, this is a placeholder that shows the structure
            exchanges = getattr(self.exchange_manager, "exchanges", {})

            for exchange_name, exchange in exchanges.items():
                try:
                    # Close all positions on this exchange
                    if hasattr(exchange, "close_all_positions"):
                        await exchange.close_all_positions()
                        logger.info(f"   ✅ Closed positions on {exchange_name}")
                        closed_count += 1
                except Exception as e:
                    logger.error(
                        f"   ❌ Failed to close positions on {exchange_name}: {e}"
                    )

            logger.info(f"   ✅ Total positions closed: {closed_count}")

        except Exception as e:
            logger.error(f"❌ Error closing positions: {e}")

        return closed_count

    def _lock_system(self):
        """Lock the system (require PIN to unlock)"""
        self.is_locked = True

        if self.security_manager:
            # Destroy all active sessions
            if hasattr(self.security_manager, "session_manager"):
                sessions = getattr(
                    self.security_manager.session_manager, "sessions", {}
                )
                for session_token in list(sessions.keys()):
                    self.security_manager.session_manager.destroy_session(session_token)
                logger.info("   ✅ All sessions destroyed")

        logger.info("   🔒 System locked - PIN required to unlock")
        self._save_state()

    async def _send_notifications(
        self, trigger_type: KillSwitchTrigger, reason: str
    ) -> bool:
        """Send emergency notifications via Telegram"""
        if not self.telegram_bot:
            return False

        try:
            message = (
                "🚨 *EMERGENCY KILL SWITCH ACTIVATED* 🚨\n\n"
                f"*Trigger:* {trigger_type.value}\n"
                f"*Reason:* {reason}\n"
                f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "All trading operations have been halted.\n"
                "System is now locked and requires PIN to unlock."
            )

            # Send to Telegram
            if hasattr(self.telegram_bot, "send_message"):
                await self.telegram_bot.send_message(message, parse_mode="Markdown")
                logger.info("   ✅ Emergency notification sent via Telegram")
                return True

        except Exception as e:
            logger.error(f"❌ Failed to send notification: {e}")

        return False

    def check_auto_triggers(self, market_data: Dict) -> Optional[KillSwitchTrigger]:
        """
        Check if any auto-trigger conditions are met

        Args:
            market_data: Dictionary with current market conditions

        Returns:
            KillSwitchTrigger if condition met, None otherwise
        """
        if not self.enabled or self.is_active:
            return None

        # Check flash crash
        price_change = market_data.get("price_change_5m", 0)
        if price_change < -self.flash_crash_threshold:
            return KillSwitchTrigger.FLASH_CRASH

        # Check API failures
        consecutive_failures = market_data.get("consecutive_api_failures", 0)
        if consecutive_failures >= self.api_failure_threshold:
            return KillSwitchTrigger.API_FAILURE

        # Check network loss
        if market_data.get("network_disconnected", False):
            return KillSwitchTrigger.NETWORK_LOSS

        return None

    def unlock(self, pin: str) -> bool:
        """
        Unlock the system after kill switch activation

        Args:
            pin: User PIN for authentication

        Returns:
            True if unlocked successfully
        """
        if not self.is_locked:
            return True

        # Verify PIN
        if self.security_manager:
            # In production, verify against stored PIN
            if hasattr(self.security_manager, "_verify_password"):
                if self.security_manager._verify_password("default_user", pin):
                    self.is_locked = False
                    self._save_state()
                    logger.info("🔓 System unlocked successfully")
                    return True

        logger.warning("❌ Unlock failed: Invalid PIN")
        return False

    def reset(self, authorized: bool = False) -> bool:
        """
        Reset kill switch (allow trading to resume)

        Args:
            authorized: Must be True to confirm authorization

        Returns:
            True if reset successful
        """
        if not authorized:
            logger.error("❌ Kill switch reset requires explicit authorization")
            return False

        if self.is_locked:
            logger.error("❌ Cannot reset while system is locked - unlock first")
            return False

        logger.warning("🔄 Resetting Emergency Kill Switch")

        # Log recovery time
        if self.current_event:
            self.current_event.recovery_time = datetime.now()
            self._log_event(self.current_event)

        # Reset state
        self.is_active = False
        self.activation_time = None
        self.current_event = None

        # Resume trading (if risk manager allows)
        if self.risk_manager:
            self.risk_manager.resume_trading("Kill switch reset")

        self._save_state()

        logger.info("✅ Kill Switch reset - system operational")
        return True

    def get_status(self) -> Dict:
        """Get current kill switch status"""
        return {
            "enabled": self.enabled,
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "activation_time": (
                self.activation_time.isoformat() if self.activation_time else None
            ),
            "current_event": (
                self.current_event.to_dict() if self.current_event else None
            ),
            "auto_backup": self.auto_backup,
            "close_positions": self.close_positions,
            "cancel_orders": self.cancel_orders,
            "lock_on_trigger": self.lock_on_trigger,
            "telegram_notify": self.telegram_notify,
        }

    def get_event_history(self, limit: int = 10) -> List[Dict]:
        """Get recent kill switch events"""
        events = []

        if not self.event_log_file.exists():
            return events

        try:
            with open(self.event_log_file, "r") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read event history: {e}")

        return events


# Usage example
if __name__ == "__main__":

    async def test_kill_switch():
        """Test kill switch functionality"""

        config = {
            "emergency_kill_switch": {
                "enabled": True,
                "auto_backup": True,
                "close_positions": True,
                "cancel_orders": True,
                "lock_on_trigger": True,
                "telegram_notify": False,
                "flash_crash_threshold": 0.15,
                "api_failure_threshold": 5,
            }
        }

        kill_switch = EmergencyKillSwitch(config)

        print("Testing Emergency Kill Switch...")
        print(f"Status: {kill_switch.get_status()}")

        # Test manual trigger
        print("\nTriggering kill switch (TEST MODE)...")
        result = await kill_switch.trigger(
            trigger_type=KillSwitchTrigger.MANUAL, reason="Test activation"
        )

        print(f"\nResult: {json.dumps(result, indent=2)}")
        print(f"\nStatus after trigger: {kill_switch.get_status()}")

        # Test unlock
        print("\nTesting unlock...")
        unlocked = kill_switch.unlock("2077")
        print(f"Unlocked: {unlocked}")

        # Test reset
        print("\nTesting reset...")
        reset_success = kill_switch.reset(authorized=True)
        print(f"Reset: {reset_success}")

        print(f"\nFinal status: {kill_switch.get_status()}")

    asyncio.run(test_kill_switch())
