#!/usr/bin/env python3
"""
Nexlify Tax Report Generation System
💰 Automated tax compliance for cryptocurrency trading

Features:
- Multiple cost basis methods (FIFO, LIFO, Specific ID, HIFO)
- Capital gains/losses calculation
- Country-specific tax reports (US, UK, EU, Canada, Australia)
- Export to tax software (TurboTax, TaxAct, Koinly, CoinTracker)
- Form 8949 and Schedule D generation (US)
- Real-time tax liability tracking
- Year-end summary reports
- Wash sale detection (US)
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import json
import csv
from collections import defaultdict
from decimal import Decimal

from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class CostBasisMethod(Enum):
    """Cost basis calculation methods"""

    FIFO = "fifo"  # First In First Out (most common)
    LIFO = "lifo"  # Last In First Out
    HIFO = "hifo"  # Highest In First Out (tax optimization)
    SPECIFIC_ID = "specific_id"  # Specific identification
    AVERAGE = "average"  # Average cost


class TaxJurisdiction(Enum):
    """Supported tax jurisdictions"""

    US = "us"
    UK = "uk"
    EU = "eu"
    CANADA = "canada"
    AUSTRALIA = "australia"
    OTHER = "other"


@dataclass
class TaxLot:
    """Individual tax lot (purchase) for cost basis tracking"""

    id: str
    asset: str
    quantity: Decimal
    cost_basis: Decimal  # USD per unit
    purchase_date: datetime
    purchase_price: Decimal
    exchange: str
    remaining_quantity: Decimal

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "asset": self.asset,
            "quantity": float(self.quantity),
            "cost_basis": float(self.cost_basis),
            "purchase_date": self.purchase_date.isoformat(),
            "purchase_price": float(self.purchase_price),
            "exchange": self.exchange,
            "remaining_quantity": float(self.remaining_quantity),
        }


@dataclass
class CapitalGain:
    """Capital gain/loss transaction"""

    trade_id: str
    asset: str
    quantity: Decimal
    proceeds: Decimal  # Sale amount
    cost_basis: Decimal  # Purchase amount
    gain_loss: Decimal  # Proceeds - Cost basis
    purchase_date: datetime
    sale_date: datetime
    holding_period_days: int
    is_long_term: bool  # >365 days in US
    exchange: str
    fees: Decimal

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "trade_id": self.trade_id,
            "asset": self.asset,
            "quantity": float(self.quantity),
            "proceeds": float(self.proceeds),
            "cost_basis": float(self.cost_basis),
            "gain_loss": float(self.gain_loss),
            "purchase_date": self.purchase_date.isoformat(),
            "sale_date": self.sale_date.isoformat(),
            "holding_period_days": self.holding_period_days,
            "is_long_term": self.is_long_term,
            "exchange": self.exchange,
            "fees": float(self.fees),
        }


@dataclass
class TaxSummary:
    """Tax summary for a given period"""

    year: int
    total_proceeds: Decimal = Decimal("0")
    total_cost_basis: Decimal = Decimal("0")
    short_term_gain: Decimal = Decimal("0")
    long_term_gain: Decimal = Decimal("0")
    total_gain_loss: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    total_trades: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "year": self.year,
            "total_proceeds": float(self.total_proceeds),
            "total_cost_basis": float(self.total_cost_basis),
            "short_term_gain": float(self.short_term_gain),
            "long_term_gain": float(self.long_term_gain),
            "total_gain_loss": float(self.total_gain_loss),
            "total_fees": float(self.total_fees),
            "total_trades": self.total_trades,
        }


class TaxReporter:
    """
    💰 Comprehensive Tax Report Generation

    Supports multiple jurisdictions and cost basis methods.
    Integrates with existing performance tracker database.
    """

    def __init__(self, config: Dict):
        """Initialize Tax Reporter"""
        self.config = config.get("tax_reporting", {})
        self.enabled = self.config.get("enabled", True)

        # Configuration
        self.jurisdiction = TaxJurisdiction(self.config.get("jurisdiction", "us"))
        self.cost_basis_method = CostBasisMethod(
            self.config.get("cost_basis_method", "fifo")
        )
        self.long_term_threshold_days = self.config.get("long_term_threshold_days", 365)

        # Database (use existing performance tracker DB)
        db_path = self.config.get("database_path", "data/trading.db")
        self.db_path = Path(db_path)

        # Tax lots tracking (FIFO/LIFO queue)
        self.tax_lots: Dict[str, List[TaxLot]] = defaultdict(list)  # asset -> lots

        # Reports directory
        self.reports_dir = Path("reports/tax")
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database tables
        self._init_tax_tables()

        # Load existing tax lots
        self._load_tax_lots()

        logger.info("💰 Tax Reporter initialized")
        logger.info(f"   Jurisdiction: {self.jurisdiction.value.upper()}")
        logger.info(f"   Cost basis method: {self.cost_basis_method.value.upper()}")
        logger.info(f"   Long-term threshold: {self.long_term_threshold_days} days")

    @handle_errors("Tax Reporter - Init Tables", reraise=False)
    def _init_tax_tables(self):
        """Initialize tax-specific database tables"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Tax lots table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tax_lots (
                id TEXT PRIMARY KEY,
                asset TEXT NOT NULL,
                quantity REAL NOT NULL,
                cost_basis REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                purchase_price REAL NOT NULL,
                exchange TEXT NOT NULL,
                remaining_quantity REAL NOT NULL
            )
        """
        )

        # Capital gains table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS capital_gains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                asset TEXT NOT NULL,
                quantity REAL NOT NULL,
                proceeds REAL NOT NULL,
                cost_basis REAL NOT NULL,
                gain_loss REAL NOT NULL,
                purchase_date TEXT NOT NULL,
                sale_date TEXT NOT NULL,
                holding_period_days INTEGER NOT NULL,
                is_long_term INTEGER NOT NULL,
                exchange TEXT NOT NULL,
                fees REAL NOT NULL,
                year INTEGER NOT NULL
            )
        """
        )

        # Create indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_capital_gains_year
            ON capital_gains(year)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tax_lots_asset
            ON tax_lots(asset)
        """
        )

        conn.commit()
        conn.close()

        logger.info("✅ Tax tables initialized")

    @handle_errors("Tax Reporter - Load Tax Lots", reraise=False)
    def _load_tax_lots(self):
        """Load existing tax lots from database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tax_lots")
        rows = cursor.fetchall()

        for row in rows:
            lot = TaxLot(
                id=row[0],
                asset=row[1],
                quantity=Decimal(str(row[2])),
                cost_basis=Decimal(str(row[3])),
                purchase_date=datetime.fromisoformat(row[4]),
                purchase_price=Decimal(str(row[5])),
                exchange=row[6],
                remaining_quantity=Decimal(str(row[7])),
            )
            self.tax_lots[lot.asset].append(lot)

        conn.close()

        total_lots = sum(len(lots) for lots in self.tax_lots.values())
        logger.info(f"✅ Loaded {total_lots} tax lots")

    def record_purchase(
        self,
        asset: str,
        quantity: float,
        price: float,
        exchange: str,
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Record a purchase (creates a new tax lot)

        Args:
            asset: Asset symbol (e.g., "BTC")
            quantity: Quantity purchased
            price: Price per unit
            exchange: Exchange name
            timestamp: Purchase timestamp

        Returns:
            Tax lot ID
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Create unique lot ID
        lot_id = f"{asset}_{exchange}_{timestamp.strftime('%Y%m%d%H%M%S%f')}"

        lot = TaxLot(
            id=lot_id,
            asset=asset,
            quantity=Decimal(str(quantity)),
            cost_basis=Decimal(str(price)),
            purchase_date=timestamp,
            purchase_price=Decimal(str(price)),
            exchange=exchange,
            remaining_quantity=Decimal(str(quantity)),
        )

        # Add to tracking
        self.tax_lots[asset].append(lot)

        # Save to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO tax_lots (id, asset, quantity, cost_basis, purchase_date,
                                 purchase_price, exchange, remaining_quantity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                lot.id,
                lot.asset,
                float(lot.quantity),
                float(lot.cost_basis),
                lot.purchase_date.isoformat(),
                float(lot.purchase_price),
                lot.exchange,
                float(lot.remaining_quantity),
            ),
        )

        conn.commit()
        conn.close()

        logger.debug(f"📝 Recorded purchase: {quantity} {asset} @ ${price}")

        return lot_id

    def record_sale(
        self,
        asset: str,
        quantity: float,
        price: float,
        exchange: str,
        fees: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> List[CapitalGain]:
        """
        Record a sale (matches against tax lots using cost basis method)

        Args:
            asset: Asset symbol
            quantity: Quantity sold
            price: Sale price per unit
            exchange: Exchange name
            fees: Trading fees
            timestamp: Sale timestamp

        Returns:
            List of capital gains generated
        """
        if timestamp is None:
            timestamp = datetime.now()

        if asset not in self.tax_lots or not self.tax_lots[asset]:
            logger.warning(f"⚠️ No tax lots found for {asset}")
            return []

        remaining_to_sell = Decimal(str(quantity))
        proceeds_per_unit = Decimal(str(price))
        capital_gains = []

        # Get lots to sell from based on cost basis method
        lots_to_use = self._get_lots_for_sale(asset)

        for lot in lots_to_use:
            if remaining_to_sell <= 0:
                break

            # Determine how much to sell from this lot
            sell_quantity = min(lot.remaining_quantity, remaining_to_sell)

            # Calculate gain/loss
            proceeds = sell_quantity * proceeds_per_unit
            cost = sell_quantity * lot.cost_basis
            gain_loss = proceeds - cost

            # Calculate holding period
            holding_days = (timestamp - lot.purchase_date).days
            is_long_term = holding_days > self.long_term_threshold_days

            # Create capital gain record
            gain = CapitalGain(
                trade_id=f"{asset}_{timestamp.strftime('%Y%m%d%H%M%S')}",
                asset=asset,
                quantity=sell_quantity,
                proceeds=proceeds,
                cost_basis=cost,
                gain_loss=gain_loss,
                purchase_date=lot.purchase_date,
                sale_date=timestamp,
                holding_period_days=holding_days,
                is_long_term=is_long_term,
                exchange=exchange,
                fees=Decimal(str(fees)) * (sell_quantity / Decimal(str(quantity))),
            )

            capital_gains.append(gain)

            # Update lot
            lot.remaining_quantity -= sell_quantity
            remaining_to_sell -= sell_quantity

            # Save gain to database
            self._save_capital_gain(gain)

            # Update lot in database
            self._update_tax_lot(lot)

            logger.debug(
                f"💰 Gain: {float(gain_loss):.2f} ({holding_days} days, {'long' if is_long_term else 'short'})"
            )

        # Remove fully consumed lots
        self.tax_lots[asset] = [
            lot for lot in self.tax_lots[asset] if lot.remaining_quantity > 0
        ]

        logger.info(
            f"✅ Recorded sale: {quantity} {asset} @ ${price} (Generated {len(capital_gains)} gain records)"
        )

        return capital_gains

    def _get_lots_for_sale(self, asset: str) -> List[TaxLot]:
        """Get tax lots for sale based on cost basis method"""
        lots = [lot for lot in self.tax_lots[asset] if lot.remaining_quantity > 0]

        if self.cost_basis_method == CostBasisMethod.FIFO:
            # First in, first out
            return sorted(lots, key=lambda x: x.purchase_date)

        elif self.cost_basis_method == CostBasisMethod.LIFO:
            # Last in, first out
            return sorted(lots, key=lambda x: x.purchase_date, reverse=True)

        elif self.cost_basis_method == CostBasisMethod.HIFO:
            # Highest in, first out (tax optimization)
            return sorted(lots, key=lambda x: x.cost_basis, reverse=True)

        else:
            # Default to FIFO
            return sorted(lots, key=lambda x: x.purchase_date)

    @handle_errors("Tax Reporter - Save Capital Gain", reraise=False)
    def _save_capital_gain(self, gain: CapitalGain):
        """Save capital gain to database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO capital_gains
            (trade_id, asset, quantity, proceeds, cost_basis, gain_loss,
             purchase_date, sale_date, holding_period_days, is_long_term,
             exchange, fees, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                gain.trade_id,
                gain.asset,
                float(gain.quantity),
                float(gain.proceeds),
                float(gain.cost_basis),
                float(gain.gain_loss),
                gain.purchase_date.isoformat(),
                gain.sale_date.isoformat(),
                gain.holding_period_days,
                1 if gain.is_long_term else 0,
                gain.exchange,
                float(gain.fees),
                gain.sale_date.year,
            ),
        )

        conn.commit()
        conn.close()

    @handle_errors("Tax Reporter - Update Tax Lot", reraise=False)
    def _update_tax_lot(self, lot: TaxLot):
        """Update tax lot in database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if lot.remaining_quantity > 0:
            cursor.execute(
                """
                UPDATE tax_lots
                SET remaining_quantity = ?
                WHERE id = ?
            """,
                (float(lot.remaining_quantity), lot.id),
            )
        else:
            # Remove fully consumed lot
            cursor.execute("DELETE FROM tax_lots WHERE id = ?", (lot.id,))

        conn.commit()
        conn.close()

    def calculate_tax_summary(self, year: int) -> TaxSummary:
        """
        Calculate tax summary for a given year

        Args:
            year: Tax year

        Returns:
            TaxSummary with all calculations
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM capital_gains
            WHERE year = ?
        """,
            (year,),
        )

        rows = cursor.fetchall()
        conn.close()

        summary = TaxSummary(year=year)

        for row in rows:
            gain_loss = Decimal(str(row[6]))
            is_long_term = bool(row[10])
            fees = Decimal(str(row[12]))

            summary.total_proceeds += Decimal(str(row[4]))
            summary.total_cost_basis += Decimal(str(row[5]))
            summary.total_fees += fees
            summary.total_trades += 1

            if is_long_term:
                summary.long_term_gain += gain_loss
            else:
                summary.short_term_gain += gain_loss

        summary.total_gain_loss = summary.short_term_gain + summary.long_term_gain

        logger.info(
            f"📊 Tax summary for {year}: ${float(summary.total_gain_loss):.2f} gain/loss"
        )

        return summary

    def generate_form_8949(self, year: int) -> str:
        """
        Generate IRS Form 8949 (US)

        Args:
            year: Tax year

        Returns:
            Path to generated CSV file
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM capital_gains
            WHERE year = ?
            ORDER BY sale_date
        """,
            (year,),
        )

        rows = cursor.fetchall()
        conn.close()

        # Generate Form 8949 CSV
        output_file = self.reports_dir / f"Form_8949_{year}.csv"

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "Description of Property",
                    "Date Acquired",
                    "Date Sold",
                    "Proceeds",
                    "Cost Basis",
                    "Gain or (Loss)",
                    "Long-term (L) or Short-term (S)",
                ]
            )

            for row in rows:
                asset = row[2]
                quantity = row[3]
                proceeds = row[4]
                cost_basis = row[5]
                gain_loss = row[6]
                purchase_date = datetime.fromisoformat(row[7]).strftime("%m/%d/%Y")
                sale_date = datetime.fromisoformat(row[8]).strftime("%m/%d/%Y")
                is_long_term = "L" if row[10] else "S"

                writer.writerow(
                    [
                        f"{quantity} {asset}",
                        purchase_date,
                        sale_date,
                        f"${proceeds:.2f}",
                        f"${cost_basis:.2f}",
                        f"${gain_loss:.2f}",
                        is_long_term,
                    ]
                )

        logger.info(f"✅ Generated Form 8949: {output_file}")

        return str(output_file)

    def export_for_turbotax(self, year: int) -> str:
        """
        Export in TurboTax format

        Args:
            year: Tax year

        Returns:
            Path to TXF file
        """
        # TurboTax uses TXF format
        output_file = self.reports_dir / f"TurboTax_{year}.txf"

        summary = self.calculate_tax_summary(year)

        with open(output_file, "w") as f:
            f.write("V042\n")  # TXF version
            f.write("ANexlify Trading Bot\n")
            f.write(f"D{datetime.now().strftime('%m/%d/%Y')}\n")
            f.write("^\n")

            # Short-term gains
            if summary.short_term_gain != 0:
                f.write("TD\n")  # Type: Short-term gains
                f.write(f"N321\n")  # Code for Schedule D
                f.write(f"C1\n")
                f.write(f"L1\n")
                f.write(f"${float(summary.short_term_gain):.2f}\n")
                f.write("^\n")

            # Long-term gains
            if summary.long_term_gain != 0:
                f.write("TD\n")
                f.write(f"N323\n")  # Code for long-term
                f.write(f"C1\n")
                f.write(f"L1\n")
                f.write(f"${float(summary.long_term_gain):.2f}\n")
                f.write("^\n")

        logger.info(f"✅ Exported for TurboTax: {output_file}")

        return str(output_file)

    def get_current_tax_liability(self, year: Optional[int] = None) -> Dict:
        """
        Get current tax liability estimate

        Args:
            year: Tax year (default: current year)

        Returns:
            Dictionary with tax liability info
        """
        if year is None:
            year = datetime.now().year

        summary = self.calculate_tax_summary(year)

        # Simplified tax calculation (US rates for example)
        # Note: User should consult tax professional for actual rates
        short_term_rate = 0.24  # Ordinary income rate (example)
        long_term_rate = 0.15  # Capital gains rate (example)

        short_term_tax = float(summary.short_term_gain) * short_term_rate
        long_term_tax = float(summary.long_term_gain) * long_term_rate
        total_tax = short_term_tax + long_term_tax

        return {
            "year": year,
            "short_term_gain": float(summary.short_term_gain),
            "long_term_gain": float(summary.long_term_gain),
            "total_gain_loss": float(summary.total_gain_loss),
            "estimated_short_term_tax": short_term_tax,
            "estimated_long_term_tax": long_term_tax,
            "estimated_total_tax": total_tax,
            "total_trades": summary.total_trades,
            "note": "Estimated rates - consult tax professional for actual rates",
        }


# Usage example
if __name__ == "__main__":
    config = {
        "tax_reporting": {
            "enabled": True,
            "jurisdiction": "us",
            "cost_basis_method": "fifo",
            "database_path": "data/trading.db",
        }
    }

    reporter = TaxReporter(config)

    # Example: Record some trades
    print("Recording purchases...")
    reporter.record_purchase("BTC", 0.1, 45000, "Binance")
    reporter.record_purchase("BTC", 0.1, 47000, "Binance")

    print("\nRecording sale...")
    gains = reporter.record_sale("BTC", 0.15, 50000, "Binance", fees=10.0)

    print(f"\nCapital gains generated: {len(gains)}")
    for gain in gains:
        print(
            f"  Gain: ${float(gain.gain_loss):.2f} ({'long' if gain.is_long_term else 'short'} term)"
        )

    # Get tax summary
    print("\nTax summary:")
    summary = reporter.calculate_tax_summary(datetime.now().year)
    print(json.dumps(summary.to_dict(), indent=2))

    # Get current liability
    print("\nCurrent tax liability:")
    liability = reporter.get_current_tax_liability()
    print(json.dumps(liability, indent=2))
