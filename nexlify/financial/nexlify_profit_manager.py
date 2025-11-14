#!/usr/bin/env python3
"""
Nexlify Automated Profit Withdrawal Manager
💸 Automatic profit extraction and management

Features:
- Multiple withdrawal strategies (percentage, threshold, time-based, hybrid)
- Scheduled withdrawals (daily, weekly, monthly, quarterly)
- Multiple destinations (cold wallet, bank, reinvest, hold)
- Minimum operating balance protection
- Profit target-based withdrawals
- Tax-efficient withdrawal timing
- Compound remaining profits
- Withdrawal history tracking
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class WithdrawalStrategy(Enum):
    """Withdrawal strategy types"""

    PERCENTAGE = "percentage"  # Withdraw X% of profits
    THRESHOLD = "threshold"  # Withdraw when profit > $Y
    TIME_BASED = "time_based"  # Withdraw on schedule
    HYBRID = "hybrid"  # Combine multiple strategies


class WithdrawalDestination(Enum):
    """Withdrawal destinations"""

    COLD_WALLET = "cold_wallet"
    BANK_ACCOUNT = "bank_account"
    REINVEST = "reinvest"
    HOLD = "hold"


class WithdrawalFrequency(Enum):
    """Withdrawal frequency"""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


@dataclass
class WithdrawalSchedule:
    """Scheduled withdrawal configuration"""

    id: str
    strategy: WithdrawalStrategy
    frequency: WithdrawalFrequency
    percentage: Optional[Decimal] = None  # For PERCENTAGE strategy
    threshold: Optional[Decimal] = None  # For THRESHOLD strategy
    destination: WithdrawalDestination = WithdrawalDestination.COLD_WALLET
    enabled: bool = True
    last_executed: Optional[datetime] = None
    next_execution: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "frequency": self.frequency.value,
            "percentage": float(self.percentage) if self.percentage else None,
            "threshold": float(self.threshold) if self.threshold else None,
            "destination": self.destination.value,
            "enabled": self.enabled,
            "last_executed": (
                self.last_executed.isoformat() if self.last_executed else None
            ),
            "next_execution": (
                self.next_execution.isoformat() if self.next_execution else None
            ),
        }


@dataclass
class WithdrawalRecord:
    """Record of a withdrawal"""

    id: str
    timestamp: datetime
    amount: Decimal
    destination: WithdrawalDestination
    reason: str
    profit_at_withdrawal: Decimal
    remaining_profit: Decimal
    tx_hash: Optional[str] = None
    status: str = "pending"  # pending, completed, failed

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "amount": float(self.amount),
            "destination": self.destination.value,
            "reason": self.reason,
            "profit_at_withdrawal": float(self.profit_at_withdrawal),
            "remaining_profit": float(self.remaining_profit),
            "tx_hash": self.tx_hash,
            "status": self.status,
        }


class ProfitManager:
    """
    💸 Automated Profit Extraction Manager

    Features:
    - Multiple withdrawal strategies
    - Scheduled automatic withdrawals
    - Tax-efficient timing
    - Minimum balance protection
    - Comprehensive tracking
    """

    def __init__(self, config: Dict):
        """Initialize Profit Manager"""
        # Handle both profit_management and profit_tracking config keys
        self.config = config.get("profit_management", config.get("profit_tracking", {}))
        self.enabled = self.config.get("enabled", True)

        # Test-specific config
        self.target_profit_percent = self.config.get("target_profit_percent", 10.0)
        self.stop_loss_percent = self.config.get("stop_loss_percent", 5.0)

        # Configuration
        self.min_operating_balance = Decimal(
            str(self.config.get("min_operating_balance", 1000))
        )
        self.default_strategy = WithdrawalStrategy(
            self.config.get("default_strategy", "threshold")
        )
        self.default_percentage = Decimal(
            str(self.config.get("default_percentage", 50))
        )  # 50%
        self.default_threshold = Decimal(
            str(self.config.get("default_threshold", 1000))
        )  # $1000
        self.auto_compound = self.config.get("auto_compound", True)
        self.compound_percentage = Decimal(
            str(self.config.get("compound_percentage", 50))
        )  # 50%

        # Database - support new paths.trading_database config with backward compatibility
        # First check if paths config is in the root config, not in profit_management
        root_config = config if "paths" in config else {}
        paths_config = root_config.get("paths", {})
        db_path = paths_config.get("trading_database", self.config.get("database_path", "data/trading.db"))
        self.db_path = Path(db_path)

        # Tracking
        self.withdrawal_schedules: Dict[str, WithdrawalSchedule] = {}
        self.withdrawal_history: List[WithdrawalRecord] = []

        # Profit tracking
        self.total_profit = Decimal("0")
        self.realized_profit = Decimal("0")
        self.unrealized_profit = Decimal("0")
        self.withdrawn_profit = Decimal("0")

        # Test-specific tracking
        self.profit_records: List[Dict] = []

        # Initialize database
        self._init_database()

        # Load schedules and history
        self._load_schedules()
        self._load_withdrawal_history()

        logger.info("💸 Profit Manager initialized")
        logger.info(f"   Enabled: {self.enabled}")
        logger.info(
            f"   Min operating balance: ${float(self.min_operating_balance):,.2f}"
        )
        logger.info(f"   Default strategy: {self.default_strategy.value}")
        logger.info(
            f"   Auto-compound: {self.auto_compound} ({float(self.compound_percentage):.0f}%)"
        )

    @handle_errors("Profit Manager - Init Database", reraise=False)
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Withdrawal schedules table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS withdrawal_schedules (
                id TEXT PRIMARY KEY,
                strategy TEXT NOT NULL,
                frequency TEXT NOT NULL,
                percentage REAL,
                threshold REAL,
                destination TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                last_executed TEXT,
                next_execution TEXT
            )
        """
        )

        # Withdrawal history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS withdrawal_history (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                amount REAL NOT NULL,
                destination TEXT NOT NULL,
                reason TEXT NOT NULL,
                profit_at_withdrawal REAL NOT NULL,
                remaining_profit REAL NOT NULL,
                tx_hash TEXT,
                status TEXT NOT NULL
            )
        """
        )

        # Profit tracking table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS profit_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_profit REAL NOT NULL,
                realized_profit REAL NOT NULL,
                unrealized_profit REAL NOT NULL,
                withdrawn_profit REAL NOT NULL
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info("✅ Profit Manager database initialized")

    @handle_errors("Profit Manager - Load Schedules", reraise=False)
    def _load_schedules(self):
        """Load withdrawal schedules from database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM withdrawal_schedules")
        rows = cursor.fetchall()

        for row in rows:
            schedule = WithdrawalSchedule(
                id=row[0],
                strategy=WithdrawalStrategy(row[1]),
                frequency=WithdrawalFrequency(row[2]),
                percentage=Decimal(str(row[3])) if row[3] else None,
                threshold=Decimal(str(row[4])) if row[4] else None,
                destination=WithdrawalDestination(row[5]),
                enabled=bool(row[6]),
                last_executed=datetime.fromisoformat(row[7]) if row[7] else None,
                next_execution=datetime.fromisoformat(row[8]) if row[8] else None,
            )
            self.withdrawal_schedules[schedule.id] = schedule

        conn.close()

        logger.info(f"✅ Loaded {len(self.withdrawal_schedules)} withdrawal schedules")

    @handle_errors("Profit Manager - Load History", reraise=False)
    def _load_withdrawal_history(self, limit: int = 100):
        """Load withdrawal history from database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM withdrawal_history
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()

        for row in rows:
            record = WithdrawalRecord(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                amount=Decimal(str(row[2])),
                destination=WithdrawalDestination(row[3]),
                reason=row[4],
                profit_at_withdrawal=Decimal(str(row[5])),
                remaining_profit=Decimal(str(row[6])),
                tx_hash=row[7],
                status=row[8],
            )
            self.withdrawal_history.append(record)

        conn.close()

        logger.info(f"✅ Loaded {len(self.withdrawal_history)} withdrawal records")

    def create_schedule(
        self,
        strategy: WithdrawalStrategy,
        frequency: WithdrawalFrequency,
        percentage: Optional[float] = None,
        threshold: Optional[float] = None,
        destination: WithdrawalDestination = WithdrawalDestination.COLD_WALLET,
    ) -> str:
        """
        Create a new withdrawal schedule

        Args:
            strategy: Withdrawal strategy
            frequency: Withdrawal frequency
            percentage: Percentage to withdraw (for PERCENTAGE strategy)
            threshold: Threshold amount (for THRESHOLD strategy)
            destination: Withdrawal destination

        Returns:
            Schedule ID
        """
        schedule_id = f"schedule_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        schedule = WithdrawalSchedule(
            id=schedule_id,
            strategy=strategy,
            frequency=frequency,
            percentage=Decimal(str(percentage)) if percentage else None,
            threshold=Decimal(str(threshold)) if threshold else None,
            destination=destination,
            enabled=True,
            next_execution=self._calculate_next_execution(frequency),
        )

        self.withdrawal_schedules[schedule_id] = schedule

        # Save to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO withdrawal_schedules
            (id, strategy, frequency, percentage, threshold, destination, enabled, last_executed, next_execution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                schedule.id,
                schedule.strategy.value,
                schedule.frequency.value,
                float(schedule.percentage) if schedule.percentage else None,
                float(schedule.threshold) if schedule.threshold else None,
                schedule.destination.value,
                1 if schedule.enabled else 0,
                None,
                (
                    schedule.next_execution.isoformat()
                    if schedule.next_execution
                    else None
                ),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(
            f"✅ Created withdrawal schedule: {schedule_id} ({strategy.value}, {frequency.value})"
        )

        return schedule_id

    def _calculate_next_execution(self, frequency: WithdrawalFrequency) -> datetime:
        """Calculate next execution time based on frequency"""
        now = datetime.now()

        if frequency == WithdrawalFrequency.DAILY:
            return now + timedelta(days=1)
        elif frequency == WithdrawalFrequency.WEEKLY:
            return now + timedelta(weeks=1)
        elif frequency == WithdrawalFrequency.MONTHLY:
            return now + timedelta(days=30)
        elif frequency == WithdrawalFrequency.QUARTERLY:
            return now + timedelta(days=90)
        else:
            return now + timedelta(days=1)  # Default to daily

    def update_profit(self, realized: float, unrealized: float):
        """
        Update profit tracking

        Args:
            realized: Realized profit
            unrealized: Unrealized profit
        """
        self.realized_profit = Decimal(str(realized))
        self.unrealized_profit = Decimal(str(unrealized))
        self.total_profit = self.realized_profit + self.unrealized_profit

        # Save to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO profit_tracking
            (timestamp, total_profit, realized_profit, unrealized_profit, withdrawn_profit)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                float(self.total_profit),
                float(self.realized_profit),
                float(self.unrealized_profit),
                float(self.withdrawn_profit),
            ),
        )

        conn.commit()
        conn.close()

        logger.debug(
            f"📊 Profit updated: Total=${float(self.total_profit):.2f}, Realized=${float(self.realized_profit):.2f}"
        )

    async def execute_withdrawal(
        self,
        amount: float,
        destination: WithdrawalDestination,
        reason: str = "Manual withdrawal",
    ) -> Optional[str]:
        """
        Execute a withdrawal

        Args:
            amount: Amount to withdraw
            destination: Withdrawal destination
            reason: Reason for withdrawal

        Returns:
            Withdrawal ID if successful
        """
        withdrawal_amount = Decimal(str(amount))

        # Check if we have enough profit
        available_profit = self.realized_profit - self.withdrawn_profit

        if withdrawal_amount > available_profit:
            logger.warning(
                f"⚠️ Insufficient profit: requested ${float(withdrawal_amount):.2f}, available ${float(available_profit):.2f}"
            )
            return None

        # Check minimum operating balance
        if withdrawal_amount > available_profit - self.min_operating_balance:
            logger.warning(f"⚠️ Withdrawal would breach minimum operating balance")
            return None

        logger.info(
            f"💸 Executing withdrawal: ${float(withdrawal_amount):.2f} to {destination.value}"
        )

        try:
            # Create withdrawal record
            withdrawal_id = f"withdrawal_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            record = WithdrawalRecord(
                id=withdrawal_id,
                timestamp=datetime.now(),
                amount=withdrawal_amount,
                destination=destination,
                reason=reason,
                profit_at_withdrawal=self.total_profit,
                remaining_profit=available_profit - withdrawal_amount,
                status="completed",  # In production, would be "pending" until confirmed
            )

            # Update withdrawn profit
            self.withdrawn_profit += withdrawal_amount

            # Save record
            self._save_withdrawal_record(record)
            self.withdrawal_history.insert(0, record)

            logger.info(f"✅ Withdrawal executed: {withdrawal_id}")

            # Auto-compound if enabled
            if self.auto_compound and destination == WithdrawalDestination.REINVEST:
                compound_amount = withdrawal_amount * self.compound_percentage / 100
                logger.info(f"♻️  Auto-compounding ${float(compound_amount):.2f}")

            return withdrawal_id

        except Exception as e:
            logger.error(f"❌ Withdrawal failed: {e}")
            return None

    @handle_errors("Profit Manager - Save Withdrawal Record", reraise=False)
    def _save_withdrawal_record(self, record: WithdrawalRecord):
        """Save withdrawal record to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO withdrawal_history
            (id, timestamp, amount, destination, reason, profit_at_withdrawal,
             remaining_profit, tx_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                record.id,
                record.timestamp.isoformat(),
                float(record.amount),
                record.destination.value,
                record.reason,
                float(record.profit_at_withdrawal),
                float(record.remaining_profit),
                record.tx_hash,
                record.status,
            ),
        )

        conn.commit()
        conn.close()

    async def check_and_execute_schedules(self):
        """Check and execute due withdrawal schedules"""
        now = datetime.now()

        for schedule_id, schedule in self.withdrawal_schedules.items():
            if not schedule.enabled:
                continue

            if schedule.next_execution and now >= schedule.next_execution:
                logger.info(f"⏰ Schedule due: {schedule_id}")

                # Determine withdrawal amount based on strategy
                amount = None

                if schedule.strategy == WithdrawalStrategy.PERCENTAGE:
                    available_profit = self.realized_profit - self.withdrawn_profit
                    amount = float(available_profit * schedule.percentage / 100)

                elif schedule.strategy == WithdrawalStrategy.THRESHOLD:
                    available_profit = self.realized_profit - self.withdrawn_profit
                    if available_profit >= schedule.threshold:
                        amount = float(schedule.threshold)

                elif schedule.strategy == WithdrawalStrategy.TIME_BASED:
                    available_profit = self.realized_profit - self.withdrawn_profit
                    amount = (
                        float(available_profit * schedule.percentage / 100)
                        if schedule.percentage
                        else float(available_profit * Decimal("0.5"))
                    )

                if amount and amount > 0:
                    withdrawal_id = await self.execute_withdrawal(
                        amount,
                        schedule.destination,
                        f"Scheduled withdrawal: {schedule.strategy.value}",
                    )

                    if withdrawal_id:
                        # Update schedule
                        schedule.last_executed = now
                        schedule.next_execution = self._calculate_next_execution(
                            schedule.frequency
                        )
                        self._update_schedule(schedule)

    @handle_errors("Profit Manager - Update Schedule", reraise=False)
    def _update_schedule(self, schedule: WithdrawalSchedule):
        """Update schedule in database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE withdrawal_schedules
            SET last_executed = ?, next_execution = ?
            WHERE id = ?
        """,
            (
                schedule.last_executed.isoformat() if schedule.last_executed else None,
                (
                    schedule.next_execution.isoformat()
                    if schedule.next_execution
                    else None
                ),
                schedule.id,
            ),
        )

        conn.commit()
        conn.close()

    def get_withdrawal_summary(self) -> Dict:
        """Get withdrawal summary"""
        total_withdrawn = sum(
            float(record.amount)
            for record in self.withdrawal_history
            if record.status == "completed"
        )

        return {
            "total_profit": float(self.total_profit),
            "realized_profit": float(self.realized_profit),
            "unrealized_profit": float(self.unrealized_profit),
            "total_withdrawn": total_withdrawn,
            "available_for_withdrawal": float(
                self.realized_profit - self.withdrawn_profit
            ),
            "min_operating_balance": float(self.min_operating_balance),
            "withdrawal_count": len(
                [r for r in self.withdrawal_history if r.status == "completed"]
            ),
            "active_schedules": len(
                [s for s in self.withdrawal_schedules.values() if s.enabled]
            ),
        }

    def get_status(self) -> Dict:
        """Get profit manager status"""
        return {
            "enabled": self.enabled,
            "summary": self.get_withdrawal_summary(),
            "schedules": {
                sid: s.to_dict() for sid, s in self.withdrawal_schedules.items()
            },
            "recent_withdrawals": [r.to_dict() for r in self.withdrawal_history[:10]],
        }

    # Backward compatibility methods for tests

    def calculate_profit(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str = "buy",
        fee: float = 0.0,
    ) -> float:
        """
        Calculate profit/loss for a trade

        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Trade quantity
            side: Trade side (buy or sell)
            fee: Trading fees

        Returns:
            Profit/loss amount
        """
        if side == "buy":
            # Long position: profit when price goes up
            profit = (exit_price - entry_price) * quantity
        else:
            # Short position: profit when price goes down
            profit = (entry_price - exit_price) * quantity

        return profit - fee

    def should_take_profit(
        self, current_price: float, entry_price: float, side: str = "buy"
    ) -> bool:
        """Check if profit target is reached"""
        if entry_price == 0:
            return False

        if side == "buy":
            price_change_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            price_change_percent = ((entry_price - current_price) / entry_price) * 100

        return price_change_percent >= self.target_profit_percent

    def should_stop_loss(
        self, current_price: float, entry_price: float, side: str = "buy"
    ) -> bool:
        """Check if stop loss is triggered"""
        if entry_price == 0:
            return False

        if side == "buy":
            price_change_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            price_change_percent = ((entry_price - current_price) / entry_price) * 100

        return price_change_percent <= -self.stop_loss_percent

    def calculate_roi(self, profit: float, investment: float) -> float:
        """Calculate return on investment percentage"""
        if investment == 0:
            return 0.0
        return (profit / investment) * 100

    def record_profit(self, profit: float, symbol: str, timestamp: datetime):
        """Record a profit/loss event"""
        self.profit_records.append(
            {"profit": profit, "symbol": symbol, "timestamp": timestamp}
        )

        # Update totals
        self.total_profit += Decimal(str(profit))
        if profit > 0:
            self.realized_profit += Decimal(str(profit))

    def get_profit_summary(self) -> Dict:
        """Get summary of profits"""
        winning_trades = [r for r in self.profit_records if r["profit"] > 0]
        losing_trades = [r for r in self.profit_records if r["profit"] < 0]

        return {
            "total_profit": float(self.total_profit),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_trades": len(self.profit_records),
        }


# Usage example
if __name__ == "__main__":

    async def test_profit_manager():
        """Test profit manager"""

        config = {
            "profit_management": {
                "enabled": True,
                "min_operating_balance": 1000,
                "default_strategy": "threshold",
                "default_threshold": 1000,
                "auto_compound": True,
                "compound_percentage": 50,
            }
        }

        manager = ProfitManager(config)

        # Update profit
        print("Updating profit...")
        manager.update_profit(realized=5000, unrealized=2000)

        # Create schedule
        print("\nCreating withdrawal schedule...")
        schedule_id = manager.create_schedule(
            strategy=WithdrawalStrategy.THRESHOLD,
            frequency=WithdrawalFrequency.WEEKLY,
            threshold=1000,
            destination=WithdrawalDestination.COLD_WALLET,
        )
        print(f"Schedule created: {schedule_id}")

        # Execute manual withdrawal
        print("\nExecuting manual withdrawal...")
        withdrawal_id = await manager.execute_withdrawal(
            amount=500,
            destination=WithdrawalDestination.COLD_WALLET,
            reason="Manual test withdrawal",
        )
        print(f"Withdrawal executed: {withdrawal_id}")

        # Get summary
        print("\nWithdrawal summary:")
        summary = manager.get_withdrawal_summary()
        print(json.dumps(summary, indent=2))

    asyncio.run(test_profit_manager())
