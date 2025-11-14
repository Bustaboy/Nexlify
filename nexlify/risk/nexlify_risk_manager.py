#!/usr/bin/env python3
"""
Nexlify - Advanced Risk Management System
Professional-grade risk controls for cryptocurrency trading
🛡️ Risk management is the difference between survival and extinction
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class TradeValidation:
    """Result of trade validation"""

    approved: bool
    reason: str
    adjusted_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class RiskMetrics:
    """Current risk metrics"""

    daily_loss: float = 0.0
    daily_profit: float = 0.0
    open_positions: int = 0
    total_exposure: float = 0.0
    largest_position: float = 0.0
    trades_today: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    🛡️ Advanced Risk Management System

    Features:
    - Position size limits
    - Daily loss limits with auto-cutoff
    - Stop-loss and take-profit automation
    - Kelly Criterion position sizing
    - Exposure monitoring
    - Risk state persistence
    """

    def __init__(self, config: Dict):
        """Initialize the Risk Manager"""
        self.config = config.get("risk_management", {})
        self.enabled = self.config.get("enabled", True)

        # Risk parameters
        self.max_position_size = self.config.get(
            "max_position_size", 0.05
        )  # 5% default
        self.max_daily_loss = self.config.get("max_daily_loss", 0.05)  # 5% default
        self.stop_loss_percent = self.config.get(
            "stop_loss_percent", 0.02
        )  # 2% default
        self.take_profit_percent = self.config.get(
            "take_profit_percent", 0.05
        )  # 5% default
        self.use_kelly = self.config.get("use_kelly_criterion", True)
        self.kelly_fraction = self.config.get(
            "kelly_fraction", 0.5
        )  # Conservative Kelly
        self.min_kelly_confidence = self.config.get("min_kelly_confidence", 0.6)
        self.max_concurrent_trades = self.config.get("max_concurrent_trades", 3)

        # Risk state
        self.metrics = RiskMetrics()
        self.trading_halted = False
        self.halt_reason = ""

        # State persistence
        self.state_file = Path("data/risk_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load previous state if exists
        self._load_state()

        logger.info("🛡️ Risk Manager initialized")
        logger.info(f"   Max position size: {self.max_position_size*100:.1f}%")
        logger.info(f"   Max daily loss: {self.max_daily_loss*100:.1f}%")
        logger.info(f"   Stop loss: {self.stop_loss_percent*100:.1f}%")
        logger.info(f"   Take profit: {self.take_profit_percent*100:.1f}%")
        logger.info(
            f"   Kelly Criterion: {'✅ Enabled' if self.use_kelly else '❌ Disabled'}"
        )

    @handle_errors("Risk Manager - Load State", reraise=False)
    def _load_state(self):
        """Load risk state from disk"""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            # Check if we need to reset (new day)
            last_reset = datetime.fromisoformat(
                data.get("last_reset", datetime.now().isoformat())
            )
            if last_reset.date() < datetime.now().date():
                logger.info("🔄 New trading day - resetting daily metrics")
                self._reset_daily_metrics()
            else:
                # Restore state
                self.metrics.daily_loss = data.get("daily_loss", 0.0)
                self.metrics.daily_profit = data.get("daily_profit", 0.0)
                self.metrics.trades_today = data.get("trades_today", 0)
                self.metrics.last_reset = last_reset
                self.trading_halted = data.get("trading_halted", False)
                self.halt_reason = data.get("halt_reason", "")

                if self.trading_halted:
                    logger.warning(f"⚠️ Trading halted: {self.halt_reason}")

        except Exception as e:
            logger.error(f"Failed to load risk state: {e}")

    @handle_errors("Risk Manager - Save State", reraise=False)
    def _save_state(self):
        """Save risk state to disk"""
        data = {
            "daily_loss": self.metrics.daily_loss,
            "daily_profit": self.metrics.daily_profit,
            "trades_today": self.metrics.trades_today,
            "last_reset": self.metrics.last_reset.isoformat(),
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
        }

        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save risk state: {e}")

    def _reset_daily_metrics(self):
        """Reset daily tracking metrics"""
        self.metrics.daily_loss = 0.0
        self.metrics.daily_profit = 0.0
        self.metrics.trades_today = 0
        self.metrics.last_reset = datetime.now()
        self.trading_halted = False
        self.halt_reason = ""
        self._save_state()
        logger.info("✅ Daily risk metrics reset")

    def _check_daily_reset(self):
        """Check if we need to reset daily metrics"""
        if self.metrics.last_reset.date() < datetime.now().date():
            self._reset_daily_metrics()

    async def validate_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        balance: float,
        confidence: float = 0.7,
    ) -> TradeValidation:
        """
        🔍 Validate trade against risk parameters

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            side: "buy" or "sell"
            quantity: Amount to trade
            price: Current price
            balance: Available capital
            confidence: Neural net confidence (0-1)

        Returns:
            TradeValidation with approval status and recommendations
        """
        # Check if risk management is enabled
        if not self.enabled:
            return TradeValidation(approved=True, reason="Risk management disabled")

        # Check for daily reset
        self._check_daily_reset()

        # Check if trading is halted
        if self.trading_halted:
            return TradeValidation(
                approved=False, reason=f"Trading halted: {self.halt_reason}"
            )

        warnings = []

        # Calculate trade value
        trade_value = quantity * price

        # Check for zero balance (edge case)
        if balance <= 0:
            return TradeValidation(approved=False, reason="Insufficient balance")

        # Calculate stop-loss and take-profit first (needed for all validations)
        if side == "buy":
            stop_loss = price * (1 - self.stop_loss_percent)
            take_profit = price * (1 + self.take_profit_percent)
        else:  # sell
            stop_loss = price * (1 + self.stop_loss_percent)
            take_profit = price * (1 - self.take_profit_percent)

        # 1. Check position size limit
        position_size_ratio = trade_value / balance
        if position_size_ratio > self.max_position_size:
            # Calculate maximum allowed quantity
            max_value = balance * self.max_position_size
            adjusted_quantity = max_value / price

            logger.warning(
                f"⚠️ Trade rejected: Position size {position_size_ratio*100:.2f}% "
                f"exceeds limit {self.max_position_size*100:.1f}%"
            )

            return TradeValidation(
                approved=False,
                reason=f"Position size {position_size_ratio*100:.2f}% exceeds limit {self.max_position_size*100:.1f}%",
                adjusted_size=adjusted_quantity,
                stop_loss=stop_loss,
                take_profit=take_profit,
                warnings=[
                    f"Suggested size: {adjusted_quantity:.6f} {symbol.split('/')[0]}"
                ],
            )

        # 2. Check daily loss limit
        if self.metrics.daily_loss >= self.max_daily_loss:
            self.trading_halted = True
            self.halt_reason = (
                f"Daily loss limit reached: {self.metrics.daily_loss*100:.2f}%"
            )
            self._save_state()

            logger.error(f"❌ {self.halt_reason}")

            return TradeValidation(approved=False, reason=self.halt_reason)

        # 3. Check max concurrent trades
        if self.metrics.open_positions >= self.max_concurrent_trades:
            return TradeValidation(
                approved=False,
                reason=f"Max concurrent trades reached ({self.max_concurrent_trades})",
            )

        # 4. Apply Kelly Criterion if enabled
        final_quantity = quantity
        if self.use_kelly and confidence >= self.min_kelly_confidence:
            kelly_quantity = self._calculate_kelly_size(balance, price, confidence)

            # Use the smaller of Kelly and max position size
            max_quantity = (balance * self.max_position_size) / price
            final_quantity = min(kelly_quantity, max_quantity, quantity)

            if final_quantity < quantity:
                warnings.append(
                    f"Kelly Criterion recommends {final_quantity:.6f} "
                    f"(confidence: {confidence*100:.0f}%)"
                )

        # 5. Check remaining daily loss allowance
        remaining_loss_allowance = (
            self.max_daily_loss - self.metrics.daily_loss
        ) * balance
        potential_loss = trade_value * self.stop_loss_percent

        if potential_loss > remaining_loss_allowance:
            warnings.append(
                f"Potential loss ${potential_loss:.2f} exceeds daily allowance ${remaining_loss_allowance:.2f}"
            )

        # Log validation
        logger.info(
            f"✅ Trade validated: {symbol} {side} {final_quantity:.6f} @ ${price:.2f}"
        )
        logger.info(f"   Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}")

        return TradeValidation(
            approved=True,
            reason="All risk checks passed",
            adjusted_size=final_quantity if final_quantity != quantity else None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            warnings=warnings,
        )

    def _calculate_kelly_size(
        self, balance: float, price: float, confidence: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Kelly % = (W * R - L) / R
        where:
        - W = Winning probability (confidence)
        - R = Win/Loss ratio (take_profit / stop_loss)
        - L = Losing probability (1 - confidence)
        """
        win_prob = confidence
        loss_prob = 1 - confidence
        win_loss_ratio = self.take_profit_percent / self.stop_loss_percent

        # Kelly formula
        kelly_percent = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio

        # Apply conservative fraction
        kelly_percent = max(0, kelly_percent * self.kelly_fraction)

        # Calculate quantity
        kelly_value = balance * kelly_percent
        kelly_quantity = kelly_value / price

        logger.debug(
            f"📊 Kelly Criterion: {kelly_percent*100:.2f}% "
            f"(confidence: {confidence*100:.0f}%, W/L ratio: {win_loss_ratio:.2f})"
        )

        return kelly_quantity

    def record_trade_result(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        balance: float,
    ):
        """
        📝 Record trade result and update risk metrics

        Args:
            symbol: Trading pair
            side: "buy" or "sell"
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            balance: Current balance (for calculating P&L ratio)
        """
        # Calculate P&L
        if side == "buy":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        pnl_percent = pnl / balance if balance > 0 else 0

        # Update metrics
        self.metrics.trades_today += 1

        if pnl < 0:
            self.metrics.daily_loss += abs(pnl_percent)
            logger.warning(f"📉 Trade loss: ${pnl:.2f} ({pnl_percent*100:.2f}%)")
        else:
            self.metrics.daily_profit += pnl_percent
            logger.info(f"📈 Trade profit: ${pnl:.2f} ({pnl_percent*100:.2f}%)")

        # Log daily status
        net_pnl = self.metrics.daily_profit - self.metrics.daily_loss
        logger.info(
            f"📊 Daily P&L: {net_pnl*100:.2f}% "
            f"(Profit: {self.metrics.daily_profit*100:.2f}%, "
            f"Loss: {self.metrics.daily_loss*100:.2f}%)"
        )

        # Check if we hit daily loss limit
        if self.metrics.daily_loss >= self.max_daily_loss:
            self.trading_halted = True
            self.halt_reason = (
                f"Daily loss limit reached: {self.metrics.daily_loss*100:.2f}%"
            )
            logger.error(f"🔴 {self.halt_reason}")

        # Save state
        self._save_state()

    def update_open_positions(self, count: int, total_exposure: float):
        """Update open position tracking"""
        self.metrics.open_positions = count
        self.metrics.total_exposure = total_exposure

    def get_risk_status(self) -> Dict:
        """
        📊 Get current risk status for display

        Returns:
            Dictionary with current risk metrics
        """
        self._check_daily_reset()

        net_pnl = self.metrics.daily_profit - self.metrics.daily_loss
        loss_remaining = max(0, self.max_daily_loss - self.metrics.daily_loss)

        return {
            "enabled": self.enabled,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "daily_profit": f"{self.metrics.daily_profit*100:.2f}%",
            "daily_loss": f"{self.metrics.daily_loss*100:.2f}%",
            "net_pnl": f"{net_pnl*100:.2f}%",
            "loss_remaining": f"{loss_remaining*100:.2f}%",
            "trades_today": self.metrics.trades_today,
            "open_positions": self.metrics.open_positions,
            "max_position_size": f"{self.max_position_size*100:.1f}%",
            "max_daily_loss": f"{self.max_daily_loss*100:.1f}%",
            "kelly_enabled": self.use_kelly,
            "last_reset": self.metrics.last_reset.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def force_reset(self):
        """🔄 Force reset risk metrics (use with caution!)"""
        logger.warning("⚠️ Forcing risk metrics reset")
        self._reset_daily_metrics()

    def resume_trading(self, reason: str = "Manual override"):
        """▶️ Resume trading after halt"""
        if self.trading_halted:
            logger.warning(f"▶️ Resuming trading: {reason}")
            self.trading_halted = False
            self.halt_reason = ""
            self._save_state()


# Usage example
if __name__ == "__main__":
    # Test configuration
    test_config = {
        "risk_management": {
            "enabled": True,
            "max_position_size": 0.05,
            "max_daily_loss": 0.05,
            "stop_loss_percent": 0.02,
            "take_profit_percent": 0.05,
            "use_kelly_criterion": True,
        }
    }

    async def test_risk_manager():
        rm = RiskManager(test_config)

        # Test trade validation
        balance = 10000.0
        validation = await rm.validate_trade(
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000,
            balance=balance,
            confidence=0.75,
        )

        print(f"Trade approved: {validation.approved}")
        print(f"Reason: {validation.reason}")
        if validation.stop_loss:
            print(f"Stop Loss: ${validation.stop_loss:.2f}")
        if validation.take_profit:
            print(f"Take Profit: ${validation.take_profit:.2f}")

    asyncio.run(test_risk_manager())
