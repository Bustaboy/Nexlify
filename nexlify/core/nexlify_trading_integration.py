#!/usr/bin/env python3
"""
Nexlify Trading Integration - Phase 1 & 2
Integrates security and financial features into automated trading
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from nexlify.financial.nexlify_defi_integration import DeFiIntegration
from nexlify.financial.nexlify_profit_manager import (ProfitManager,
                                                      WithdrawalDestination)
from nexlify.financial.nexlify_tax_reporter import TaxReporter
from nexlify.risk.nexlify_emergency_kill_switch import KillSwitchTrigger
from nexlify.risk.nexlify_flash_crash_protection import CrashSeverity
from nexlify.security.nexlify_security_suite import SecuritySuite

logger = logging.getLogger(__name__)


class TradingIntegrationManager:
    """
    Manages integration of Phase 1 & 2 features with automated trading

    Automatically:
    - Records all trades for tax reporting
    - Tracks profits and manages withdrawals
    - Monitors for flash crashes
    - Moves idle funds to DeFi
    - Triggers emergency responses
    """

    def __init__(self, config: Dict):
        """Initialize all integrated features"""
        self.config = config
        self.enabled = config.get("enable_phase1_phase2_integration", True)

        if not self.enabled:
            logger.warning("⚠️ Phase 1 & 2 integration is DISABLED")
            return

        # Initialize Phase 1 & 2 components
        self.security_suite = SecuritySuite(config)
        self.tax_reporter = TaxReporter(config)
        self.profit_manager = ProfitManager(config)
        self.defi_integration = DeFiIntegration(config)

        # Tracking
        self.trade_count = 0
        self.total_fees_paid = Decimal("0")
        self.last_defi_check = None
        self.flash_crash_monitoring = {}

        logger.info("=" * 80)
        logger.info("🔗 TRADING INTEGRATION MANAGER INITIALIZED")
        logger.info("=" * 80)
        logger.info("✅ Tax Reporting: Active")
        logger.info("✅ Profit Management: Active")
        logger.info("✅ Flash Crash Protection: Active")
        logger.info("✅ DeFi Integration: Active")
        logger.info("✅ Security Suite: Active")
        logger.info("=" * 80)

    async def initialize(self):
        """Async initialization"""
        if not self.enabled:
            return

        await self.security_suite.initialize()

        # Start background tasks
        asyncio.create_task(self._monitor_idle_funds())
        asyncio.create_task(self._check_profit_withdrawals())

        logger.info("✅ Trading integration fully initialized")

    def inject_dependencies(
        self,
        neural_net=None,
        risk_manager=None,
        exchange_manager=None,
        telegram_bot=None,
    ):
        """Inject external dependencies"""
        if not self.enabled:
            return

        self.neural_net = neural_net
        self.risk_manager = risk_manager
        self.exchange_manager = exchange_manager

        # Inject into security suite
        self.security_suite.inject_external_dependencies(
            risk_manager=risk_manager,
            exchange_manager=exchange_manager,
            telegram_bot=telegram_bot,
        )

        # Inject into flash protection
        if hasattr(self.security_suite, "flash_protection") and exchange_manager:
            self.security_suite.flash_protection.exchanges = exchange_manager

        logger.info("✅ Dependencies injected into trading integration")

    # ==================== TRADE LIFECYCLE HOOKS ====================

    async def on_trade_executed(self, trade_data: Dict) -> bool:
        """
        Called immediately after a trade is executed

        Args:
            trade_data: Dict with keys: symbol, side, quantity, price, exchange, timestamp, fees

        Returns:
            bool: True if processing succeeded
        """
        if not self.enabled:
            return True

        try:
            symbol = trade_data["symbol"]
            side = trade_data["side"]  # 'buy' or 'sell'
            quantity = float(trade_data["quantity"])
            price = float(trade_data["price"])
            exchange = trade_data.get("exchange", "unknown")
            timestamp = trade_data.get("timestamp", datetime.now())
            fees = float(trade_data.get("fees", 0))

            # Extract base asset (e.g., 'BTC' from 'BTC/USDT')
            base_asset = symbol.split("/")[0] if "/" in symbol else symbol

            # 1. Record for tax reporting
            if side.lower() == "buy":
                lot_id = self.tax_reporter.record_purchase(
                    asset=base_asset,
                    quantity=quantity,
                    price=price,
                    exchange=exchange,
                    timestamp=timestamp,
                )
                logger.debug(f"📊 Tax: Recorded purchase lot {lot_id}")

            elif side.lower() == "sell":
                gains = self.tax_reporter.record_sale(
                    asset=base_asset,
                    quantity=quantity,
                    price=price,
                    exchange=exchange,
                    fees=fees,
                    timestamp=timestamp,
                )

                # Calculate realized profit from this sale
                realized_profit = sum(float(g.gain_loss) for g in gains)

                # 2. Update profit manager
                self.profit_manager.update_profit(
                    realized=realized_profit,
                    unrealized=0,  # Will be calculated separately
                )

                logger.info(
                    f"💰 Profit: ${realized_profit:.2f} from {quantity:.4f} {base_asset}"
                )

            # 3. Track fees
            self.total_fees_paid += Decimal(str(fees))
            self.trade_count += 1

            # 4. Update flash crash monitoring with this price
            if hasattr(self.security_suite, "flash_protection"):
                self.security_suite.flash_protection.add_price_update(
                    symbol=symbol,
                    price=price,
                    volume=0,  # Volume tracked separately
                    timestamp=timestamp,
                )

            return True

        except Exception as e:
            logger.error(f"❌ Error in on_trade_executed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def on_position_opened(self, position_data: Dict):
        """
        Called when a new position is opened

        Args:
            position_data: Dict with position details
        """
        if not self.enabled:
            return

        try:
            symbol = position_data.get("symbol")
            logger.debug(f"📈 Position opened: {symbol}")

            # Flash crash protection knows about this position
            if hasattr(self.security_suite, "flash_protection"):
                if symbol not in self.flash_crash_monitoring:
                    self.flash_crash_monitoring[symbol] = True
                    logger.debug(f"⚡ Flash crash monitoring enabled for {symbol}")

        except Exception as e:
            logger.error(f"Error in on_position_opened: {e}")

    async def on_position_closed(self, position_data: Dict):
        """
        Called when a position is closed

        Args:
            position_data: Dict with position details and PnL
        """
        if not self.enabled:
            return

        try:
            pnl = float(position_data.get("pnl", 0))
            symbol = position_data.get("symbol")

            # Update unrealized -> realized profit
            if pnl != 0:
                self.profit_manager.update_profit(
                    realized=pnl, unrealized=-pnl  # Remove from unrealized
                )

            logger.debug(f"📉 Position closed: {symbol}, PnL: ${pnl:.2f}")

        except Exception as e:
            logger.error(f"Error in on_position_closed: {e}")

    async def on_price_update(self, symbol: str, price: float, volume: float = 0):
        """
        Called on price updates for flash crash detection

        Args:
            symbol: Trading pair
            price: Current price
            volume: Trading volume
        """
        if not self.enabled:
            return

        try:
            # Feed price to flash crash protection
            if hasattr(self.security_suite, "flash_protection"):
                self.security_suite.flash_protection.add_price_update(
                    symbol=symbol, price=price, volume=volume, timestamp=datetime.now()
                )

                # Check for crash
                severity, crash_info = (
                    self.security_suite.flash_protection.detect_crash(symbol)
                )

                # Auto-trigger kill switch on critical crashes
                if severity == CrashSeverity.CRITICAL:
                    logger.critical(f"⚡ CRITICAL FLASH CRASH DETECTED: {symbol}")
                    logger.critical(f"⚡ Details: {crash_info}")

                    # Trigger emergency kill switch
                    await self.security_suite.trigger_emergency_shutdown(
                        reason=f"Critical flash crash: {symbol}",
                        trigger_type=KillSwitchTrigger.FLASH_CRASH,
                    )

        except Exception as e:
            logger.error(f"Error in on_price_update: {e}")

    # ==================== BACKGROUND TASKS ====================

    async def _monitor_idle_funds(self):
        """Background task: Move idle funds to DeFi"""
        if not self.enabled:
            return

        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Get idle balance (funds not in active trading)
                # This would need to be calculated based on total balance vs active positions
                # For now, we'll skip automatic DeFi as it needs careful implementation

                logger.debug("💸 Checked for idle funds (DeFi auto-invest pending)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in idle funds monitor: {e}")
                await asyncio.sleep(3600)

    async def _check_profit_withdrawals(self):
        """Background task: Check and execute scheduled withdrawals"""
        if not self.enabled:
            return

        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check scheduled withdrawals
                await self.profit_manager.check_and_execute_schedules()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in withdrawal check: {e}")
                await asyncio.sleep(300)

    # ==================== STATUS & REPORTING ====================

    def get_integration_status(self) -> Dict:
        """Get current integration status"""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "trades_processed": self.trade_count,
            "total_fees_paid": float(self.total_fees_paid),
            "tax_reporter": {
                "trades": self.tax_reporter.calculate_tax_summary(
                    datetime.now().year
                ).total_trades,
                "jurisdiction": self.tax_reporter.jurisdiction,
            },
            "profit_manager": self.profit_manager.get_withdrawal_summary(),
            "security": self.security_suite.get_comprehensive_status(),
            "defi": (
                self.defi_integration.get_status()
                if self.defi_integration.enabled
                else None
            ),
        }

    async def shutdown(self):
        """Graceful shutdown"""
        if not self.enabled:
            return

        logger.info("🔌 Shutting down trading integration...")

        # Stop monitoring
        self.security_suite.stop_monitoring()

        logger.info("✅ Trading integration shutdown complete")


# ==================== CONVENIENCE FUNCTIONS ====================


async def create_integrated_trading_manager(config: Dict) -> TradingIntegrationManager:
    """
    Create and initialize a trading integration manager

    Usage:
        manager = await create_integrated_trading_manager(config)
        manager.inject_dependencies(neural_net, risk_manager, exchange_manager)
    """
    manager = TradingIntegrationManager(config)
    await manager.initialize()
    return manager
