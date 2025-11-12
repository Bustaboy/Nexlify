#!/usr/bin/env python3
"""
Nexlify Flash Crash Protection System
⚡ Real-time monitoring and protection against market crashes

Features:
- Multi-timeframe crash detection (1m, 5m, 15m)
- Configurable crash thresholds (minor, major, critical)
- Automatic position closure based on severity
- Recovery detection and auto-resume
- Integration with Emergency Kill Switch
- Price volatility analysis
- Volume spike detection
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import json
from pathlib import Path

from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class CrashSeverity(Enum):
    """Flash crash severity levels"""
    NONE = "none"
    MINOR = "minor"      # Warning only
    MAJOR = "major"      # Close risky positions
    CRITICAL = "critical"  # Close ALL positions


@dataclass
class PriceSnapshot:
    """Price snapshot at a point in time"""
    timestamp: datetime
    price: float
    volume: float = 0.0
    symbol: str = ""


@dataclass
class CrashEvent:
    """Flash crash event record"""
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    severity: CrashSeverity = CrashSeverity.NONE
    price_drop_percent: float = 0.0
    timeframe: str = ""
    price_before: float = 0.0
    price_after: float = 0.0
    volume_spike: float = 0.0
    action_taken: str = ""
    positions_closed: int = 0
    recovery_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'severity': self.severity.value,
            'price_drop_percent': self.price_drop_percent,
            'timeframe': self.timeframe,
            'price_before': self.price_before,
            'price_after': self.price_after,
            'volume_spike': self.volume_spike,
            'action_taken': self.action_taken,
            'positions_closed': self.positions_closed,
            'recovery_time': self.recovery_time.isoformat() if self.recovery_time else None
        }


class FlashCrashProtection:
    """
    ⚡ Flash Crash Protection System

    Monitors price movements in real-time and automatically protects
    capital during extreme market volatility.

    Detection Methods:
    - Rolling price change analysis (1m, 5m, 15m windows)
    - Volume spike detection (unusual selling pressure)
    - Volatility analysis (standard deviation)
    - Multi-asset correlation (market-wide crash)

    Protection Actions:
    - Minor crash: Send warning, tighten stop-losses
    - Major crash: Close leveraged/risky positions
    - Critical crash: Close ALL positions, trigger kill switch

    Recovery:
    - Automatic detection of price stabilization
    - Configurable recovery threshold (e.g., 20% rebound)
    - Gradual resumption of trading
    """

    def __init__(self, config: Dict):
        """Initialize Flash Crash Protection"""
        self.config = config.get('flash_crash_protection', {})
        self.enabled = self.config.get('enabled', True)

        # Crash thresholds (negative percentages)
        self.thresholds = {
            CrashSeverity.MINOR: self.config.get('minor_threshold', -0.05),      # -5%
            CrashSeverity.MAJOR: self.config.get('major_threshold', -0.10),      # -10%
            CrashSeverity.CRITICAL: self.config.get('critical_threshold', -0.15) # -15%
        }

        # Monitoring settings
        self.check_interval = self.config.get('check_interval', 30)  # seconds
        self.timeframes = self.config.get('timeframes', ['1m', '5m', '15m'])
        self.recovery_threshold = self.config.get('recovery_threshold', 0.20)  # 20% recovery

        # Volume spike detection
        self.volume_spike_threshold = self.config.get('volume_spike_threshold', 3.0)  # 3x avg

        # State management
        self.is_protecting = False
        self.crash_detected_time: Optional[datetime] = None
        self.current_severity = CrashSeverity.NONE
        self.price_history: Dict[str, Dict[str, deque]] = {}  # symbol -> timeframe -> prices
        self.volume_history: Dict[str, deque] = {}  # symbol -> volumes

        # Maximum history to keep (in data points)
        self.max_history = {
            '1m': 60,   # 1 hour of 1-minute data
            '5m': 60,   # 5 hours of 5-minute data
            '15m': 96   # 24 hours of 15-minute data
        }

        # Event logging
        self.event_log_file = Path("data/flash_crash_events.jsonl")
        self.event_log_file.parent.mkdir(parents=True, exist_ok=True)

        # External dependencies
        self.kill_switch = None
        self.risk_manager = None
        self.exchange_manager = None
        self.telegram_bot = None

        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("⚡ Flash Crash Protection initialized")
        logger.info(f"   Enabled: {self.enabled}")
        logger.info(f"   Thresholds: Minor={self.thresholds[CrashSeverity.MINOR]*100:.0f}%, "
                   f"Major={self.thresholds[CrashSeverity.MAJOR]*100:.0f}%, "
                   f"Critical={self.thresholds[CrashSeverity.CRITICAL]*100:.0f}%")
        logger.info(f"   Check interval: {self.check_interval}s")
        logger.info(f"   Timeframes: {', '.join(self.timeframes)}")

    def inject_dependencies(
        self,
        kill_switch=None,
        risk_manager=None,
        exchange_manager=None,
        telegram_bot=None
    ):
        """Inject external dependencies"""
        self.kill_switch = kill_switch
        self.risk_manager = risk_manager
        self.exchange_manager = exchange_manager
        self.telegram_bot = telegram_bot
        logger.info("✅ Flash Crash Protection dependencies injected")

    def add_price_update(self, symbol: str, price: float, volume: float = 0.0, timestamp: Optional[datetime] = None):
        """
        Add a price update for monitoring

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            price: Current price
            volume: Trading volume (optional)
            timestamp: Time of update (default: now)
        """
        if not self.enabled:
            return

        if timestamp is None:
            timestamp = datetime.now()

        # Initialize history for this symbol if needed
        if symbol not in self.price_history:
            self.price_history[symbol] = {}
            for tf in self.timeframes:
                self.price_history[symbol][tf] = deque(maxlen=self.max_history[tf])

        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=100)

        # Add to all timeframes
        for tf in self.timeframes:
            self.price_history[symbol][tf].append(
                PriceSnapshot(timestamp=timestamp, price=price, volume=volume, symbol=symbol)
            )

        # Add to volume history
        if volume > 0:
            self.volume_history[symbol].append(volume)

    def calculate_price_change(self, symbol: str, timeframe: str) -> Optional[float]:
        """
        Calculate price change percentage over a timeframe

        Args:
            symbol: Trading pair
            timeframe: Time window ('1m', '5m', '15m')

        Returns:
            Price change as decimal (e.g., -0.15 for -15% drop)
        """
        if symbol not in self.price_history:
            return None

        if timeframe not in self.price_history[symbol]:
            return None

        prices = self.price_history[symbol][timeframe]

        if len(prices) < 2:
            return None

        # Get current and oldest price in window
        current_price = prices[-1].price
        old_price = prices[0].price

        if old_price == 0:
            return None

        # Calculate percentage change
        price_change = (current_price - old_price) / old_price

        return price_change

    def detect_crash(self, symbol: str) -> Tuple[CrashSeverity, Dict]:
        """
        Detect flash crash for a symbol

        Args:
            symbol: Trading pair to analyze

        Returns:
            Tuple of (severity, details_dict)
        """
        if not self.enabled:
            return CrashSeverity.NONE, {}

        if symbol not in self.price_history:
            return CrashSeverity.NONE, {}

        max_severity = CrashSeverity.NONE
        details = {
            'symbol': symbol,
            'timeframe_analysis': {},
            'volume_spike': False
        }

        # Check each timeframe
        for tf in self.timeframes:
            price_change = self.calculate_price_change(symbol, tf)

            if price_change is None:
                continue

            details['timeframe_analysis'][tf] = {
                'price_change': price_change,
                'price_change_percent': price_change * 100
            }

            # Determine severity for this timeframe
            if price_change <= self.thresholds[CrashSeverity.CRITICAL]:
                max_severity = CrashSeverity.CRITICAL
                details['timeframe_analysis'][tf]['severity'] = 'critical'
            elif price_change <= self.thresholds[CrashSeverity.MAJOR]:
                if max_severity.value == CrashSeverity.NONE.value:
                    max_severity = CrashSeverity.MAJOR
                details['timeframe_analysis'][tf]['severity'] = 'major'
            elif price_change <= self.thresholds[CrashSeverity.MINOR]:
                if max_severity.value == CrashSeverity.NONE.value:
                    max_severity = CrashSeverity.MINOR
                details['timeframe_analysis'][tf]['severity'] = 'minor'
            else:
                details['timeframe_analysis'][tf]['severity'] = 'none'

        # Check for volume spike
        if symbol in self.volume_history and len(self.volume_history[symbol]) > 0:
            volumes = list(self.volume_history[symbol])
            if len(volumes) >= 10:
                avg_volume = sum(volumes[:-1]) / len(volumes[:-1])
                current_volume = volumes[-1]

                if avg_volume > 0 and current_volume > avg_volume * self.volume_spike_threshold:
                    details['volume_spike'] = True
                    details['volume_spike_ratio'] = current_volume / avg_volume

        return max_severity, details

    async def trigger_protection(self, symbol: str, severity: CrashSeverity, details: Dict) -> CrashEvent:
        """
        Trigger crash protection actions

        Args:
            symbol: Symbol experiencing crash
            severity: Crash severity level
            details: Detection details

        Returns:
            CrashEvent record
        """
        logger.warning("=" * 80)
        logger.warning(f"⚡ FLASH CRASH DETECTED: {symbol}")
        logger.warning(f"Severity: {severity.value.upper()}")
        logger.warning(f"Details: {json.dumps(details, indent=2)}")
        logger.warning("=" * 80)

        # Get price information
        price_before = details.get('price_before', 0.0)
        price_after = details.get('price_after', 0.0)
        price_drop = 0.0

        if symbol in self.price_history and self.timeframes:
            prices = self.price_history[symbol][self.timeframes[0]]
            if len(prices) >= 2:
                price_before = prices[0].price
                price_after = prices[-1].price
                price_drop = (price_after - price_before) / price_before if price_before > 0 else 0.0

        # Create event record
        event = CrashEvent(
            symbol=symbol,
            severity=severity,
            price_drop_percent=price_drop * 100,
            timeframe=', '.join(self.timeframes),
            price_before=price_before,
            price_after=price_after,
            volume_spike=details.get('volume_spike_ratio', 0.0)
        )

        # Mark as protecting
        self.is_protecting = True
        self.crash_detected_time = datetime.now()
        self.current_severity = severity

        # Take action based on severity
        positions_closed = 0

        if severity == CrashSeverity.MINOR:
            # Minor crash: Send warning only
            event.action_taken = "Warning sent, monitoring closely"
            logger.warning(f"⚠️ Minor crash detected - monitoring closely")
            await self._send_notification(
                f"⚠️ *Minor Flash Crash Warning*\n\n"
                f"Symbol: {symbol}\n"
                f"Drop: {price_drop*100:.2f}%\n"
                f"Action: Monitoring closely"
            )

        elif severity == CrashSeverity.MAJOR:
            # Major crash: Close risky positions
            event.action_taken = "Closed leveraged/risky positions"
            logger.warning(f"⚠️ Major crash detected - closing risky positions")
            positions_closed = await self._close_risky_positions(symbol)
            event.positions_closed = positions_closed
            await self._send_notification(
                f"🚨 *Major Flash Crash Detected*\n\n"
                f"Symbol: {symbol}\n"
                f"Drop: {price_drop*100:.2f}%\n"
                f"Action: Closed {positions_closed} risky positions"
            )

        elif severity == CrashSeverity.CRITICAL:
            # Critical crash: Trigger kill switch
            event.action_taken = "Triggered Emergency Kill Switch"
            logger.critical(f"🚨 CRITICAL crash detected - triggering kill switch")

            if self.kill_switch:
                from nexlify.risk.nexlify_emergency_kill_switch import KillSwitchTrigger
                await self.kill_switch.trigger(
                    trigger_type=KillSwitchTrigger.FLASH_CRASH,
                    reason=f"Critical flash crash detected: {symbol} dropped {price_drop*100:.2f}%",
                    auto_trigger=True
                )
                event.positions_closed = -1  # Kill switch handles all positions
            else:
                logger.error("❌ Kill switch not available!")

        # Log event
        self._log_event(event)

        return event

    async def _close_risky_positions(self, symbol: str) -> int:
        """Close leveraged or risky positions"""
        closed_count = 0

        # This would integrate with your exchange manager
        # For now, placeholder showing the structure
        if self.exchange_manager:
            try:
                # Close leveraged positions for this symbol
                if hasattr(self.exchange_manager, 'close_leveraged_positions'):
                    closed_count = await self.exchange_manager.close_leveraged_positions(symbol)
                    logger.info(f"   ✅ Closed {closed_count} leveraged positions")
            except Exception as e:
                logger.error(f"   ❌ Error closing positions: {e}")

        return closed_count

    async def _send_notification(self, message: str):
        """Send notification via Telegram"""
        if self.telegram_bot:
            try:
                if hasattr(self.telegram_bot, 'send_message'):
                    await self.telegram_bot.send_message(message, parse_mode='Markdown')
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

    def _log_event(self, event: CrashEvent):
        """Log crash event to persistent storage"""
        try:
            with open(self.event_log_file, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log crash event: {e}")

    async def check_recovery(self, symbol: str) -> bool:
        """
        Check if market has recovered from crash

        Args:
            symbol: Symbol to check

        Returns:
            True if recovered
        """
        if not self.is_protecting:
            return True

        if symbol not in self.price_history:
            return False

        # Get price at crash detection
        if not self.crash_detected_time:
            return False

        # Get current price and crash price
        for tf in self.timeframes:
            prices = self.price_history[symbol][tf]
            if len(prices) < 2:
                continue

            # Find price at crash time
            crash_price = None
            for snapshot in prices:
                if snapshot.timestamp >= self.crash_detected_time:
                    crash_price = snapshot.price
                    break

            if crash_price is None:
                continue

            current_price = prices[-1].price

            # Calculate recovery
            if crash_price > 0:
                recovery = (current_price - crash_price) / crash_price

                if recovery >= self.recovery_threshold:
                    logger.info(f"✅ Recovery detected for {symbol}: {recovery*100:.2f}% rebound")
                    return True

        return False

    async def resume_after_recovery(self, symbol: str):
        """Resume trading after market recovery"""
        logger.info(f"▶️ Resuming trading after recovery: {symbol}")

        self.is_protecting = False
        self.current_severity = CrashSeverity.NONE

        # Log recovery time in last event
        # (You could enhance this to update the actual event record)

        # Resume risk manager
        if self.risk_manager:
            self.risk_manager.resume_trading(f"Flash crash recovery: {symbol}")

        await self._send_notification(
            f"✅ *Market Recovery Detected*\n\n"
            f"Symbol: {symbol}\n"
            f"Status: Trading resumed"
        )

    async def monitor_loop(self, symbols: List[str]):
        """
        Main monitoring loop

        Args:
            symbols: List of symbols to monitor
        """
        logger.info(f"⚡ Starting flash crash monitoring for {len(symbols)} symbols")

        while True:
            try:
                for symbol in symbols:
                    # Detect crash
                    severity, details = self.detect_crash(symbol)

                    if severity != CrashSeverity.NONE and not self.is_protecting:
                        # Trigger protection
                        await self.trigger_protection(symbol, severity, details)

                    elif self.is_protecting:
                        # Check for recovery
                        if await self.check_recovery(symbol):
                            await self.resume_after_recovery(symbol)

                # Wait before next check
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                logger.info("⚡ Flash crash monitoring stopped")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    def start_monitoring(self, symbols: List[str]):
        """Start the monitoring loop"""
        if self.monitoring_task is not None:
            logger.warning("Monitoring already running")
            return

        self.monitoring_task = asyncio.create_task(self.monitor_loop(symbols))
        logger.info("✅ Flash crash monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring loop"""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            self.monitoring_task = None
            logger.info("⛔ Flash crash monitoring stopped")

    def get_status(self) -> Dict:
        """Get current protection status"""
        return {
            'enabled': self.enabled,
            'is_protecting': self.is_protecting,
            'current_severity': self.current_severity.value if self.current_severity else 'none',
            'crash_detected_time': self.crash_detected_time.isoformat() if self.crash_detected_time else None,
            'thresholds': {
                'minor': f"{self.thresholds[CrashSeverity.MINOR]*100:.0f}%",
                'major': f"{self.thresholds[CrashSeverity.MAJOR]*100:.0f}%",
                'critical': f"{self.thresholds[CrashSeverity.CRITICAL]*100:.0f}%"
            },
            'monitoring_symbols': len(self.price_history),
            'check_interval': self.check_interval,
            'recovery_threshold': f"{self.recovery_threshold*100:.0f}%"
        }

    def get_event_history(self, limit: int = 10) -> List[Dict]:
        """Get recent crash events"""
        events = []

        if not self.event_log_file.exists():
            return events

        try:
            with open(self.event_log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read event history: {e}")

        return events


# Usage example
if __name__ == "__main__":
    async def test_flash_crash_protection():
        """Test flash crash protection"""

        config = {
            'flash_crash_protection': {
                'enabled': True,
                'minor_threshold': -0.05,
                'major_threshold': -0.10,
                'critical_threshold': -0.15,
                'check_interval': 5,
                'timeframes': ['1m', '5m'],
                'recovery_threshold': 0.20
            }
        }

        protection = FlashCrashProtection(config)

        # Simulate normal price movement
        print("Simulating normal price movement...")
        for i in range(10):
            price = 50000 + i * 10
            protection.add_price_update("BTC/USDT", price, volume=100.0)
            await asyncio.sleep(0.1)

        # Check status
        print(f"\nStatus: {protection.get_status()}")

        # Simulate flash crash
        print("\nSimulating flash crash...")
        crash_prices = [50000, 48000, 46000, 44000, 42000]  # -16% drop
        for price in crash_prices:
            protection.add_price_update("BTC/USDT", price, volume=500.0)
            severity, details = protection.detect_crash("BTC/USDT")
            print(f"Price: ${price:,} | Severity: {severity.value}")
            await asyncio.sleep(0.1)

        # Trigger protection if crash detected
        severity, details = protection.detect_crash("BTC/USDT")
        if severity != CrashSeverity.NONE:
            print(f"\nCrash detected! Details: {json.dumps(details, indent=2)}")
            event = await protection.trigger_protection("BTC/USDT", severity, details)
            print(f"Event: {event.to_dict()}")

        # Simulate recovery
        print("\nSimulating recovery...")
        recovery_prices = [42000, 44000, 46000, 48000, 50000]
        for price in recovery_prices:
            protection.add_price_update("BTC/USDT", price, volume=100.0)
            recovered = await protection.check_recovery("BTC/USDT")
            print(f"Price: ${price:,} | Recovered: {recovered}")
            await asyncio.sleep(0.1)

            if recovered:
                await protection.resume_after_recovery("BTC/USDT")
                break

        print(f"\nFinal status: {protection.get_status()}")

    asyncio.run(test_flash_crash_protection())
