#!/usr/bin/env python3
"""
Nexlify - Performance Metrics Calculator
Professional-grade trading performance analytics
📊 Know your numbers: Sharpe ratio, max drawdown, profit factor, and more
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import csv
import math

from nexlify.utils.error_handler import handle_errors, get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class Trade:
    """Trade record"""
    id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    exchange: str = ""
    symbol: str = ""
    side: str = ""  # "buy" or "sell"
    quantity: float = 0.0
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    fee: float = 0.0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "open"  # "open", "closed", "cancelled"
    strategy: str = ""
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        if isinstance(data['timestamp'], datetime):
            data['timestamp'] = data['timestamp'].isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """Performance metrics calculation result"""
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0

    # Profit metrics
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # Advanced metrics
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    current_drawdown: float = 0.0

    # Daily metrics
    daily_pnl: float = 0.0
    daily_trades: int = 0

    # Additional stats
    avg_trade_duration: float = 0.0  # Minutes
    total_fees: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class PerformanceTracker:
    """
    📊 Professional Trading Performance Tracker

    Features:
    - Comprehensive trade recording
    - Real-time performance metrics
    - Sharpe ratio calculation
    - Max drawdown tracking
    - Profit factor analysis
    - Export to JSON/CSV
    - SQLite database storage
    """

    def __init__(self, config: Dict):
        """Initialize Performance Tracker"""
        self.config = config.get('performance_tracking', {})
        self.enabled = self.config.get('enabled', True)

        # Database setup
        db_path = self.config.get('database_path', 'data/trading.db')
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.calculate_sharpe = self.config.get('calculate_sharpe_ratio', True)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)  # 2% annual
        self.track_drawdown = self.config.get('track_drawdown', True)

        # Initialize database
        self._init_database()

        # In-memory cache for performance
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

        logger.info("📊 Performance Tracker initialized")
        logger.info(f"   Database: {self.db_path}")
        logger.info(f"   Sharpe calculation: {'✅ Enabled' if self.calculate_sharpe else '❌ Disabled'}")
        logger.info(f"   Drawdown tracking: {'✅ Enabled' if self.track_drawdown else '❌ Disabled'}")

    @handle_errors("Performance Tracker - Database Init", reraise=False)
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                exchange TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                fee REAL DEFAULT 0,
                pnl REAL DEFAULT 0,
                pnl_percent REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                strategy TEXT,
                notes TEXT
            )
        """)

        # Create equity_curve table for drawdown calculation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity_curve (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL
            )
        """)

        # Create indexes for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp
            ON trades(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_status
            ON trades(status)
        """)

        conn.commit()
        conn.close()

        logger.info("✅ Database initialized successfully")

    @handle_errors("Performance Tracker - Record Trade", reraise=False)
    def record_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        exit_price: Optional[float] = None,
        fee: float = 0.0,
        strategy: str = "",
        notes: str = ""
    ) -> Optional[int]:
        """
        📝 Record a trade in the database

        Args:
            exchange: Exchange name
            symbol: Trading pair
            side: "buy" or "sell"
            quantity: Trade quantity
            entry_price: Entry price
            exit_price: Exit price (None if still open)
            fee: Trading fees
            strategy: Strategy name
            notes: Additional notes

        Returns:
            Trade ID or None if failed
        """
        if not self.enabled:
            return None

        # Calculate P&L if trade is closed
        pnl = 0.0
        pnl_percent = 0.0
        status = "open"

        if exit_price is not None:
            status = "closed"
            if side == "buy":
                pnl = (exit_price - entry_price) * quantity - fee
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # sell
                pnl = (entry_price - exit_price) * quantity - fee
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        trade = Trade(
            timestamp=datetime.now(),
            exchange=exchange,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            fee=fee,
            pnl=pnl,
            pnl_percent=pnl_percent,
            status=status,
            strategy=strategy,
            notes=notes
        )

        # Insert into database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO trades
            (timestamp, exchange, symbol, side, quantity, entry_price,
             exit_price, fee, pnl, pnl_percent, status, strategy, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.timestamp.isoformat(),
            trade.exchange,
            trade.symbol,
            trade.side,
            trade.quantity,
            trade.entry_price,
            trade.exit_price,
            trade.fee,
            trade.pnl,
            trade.pnl_percent,
            trade.status,
            trade.strategy,
            trade.notes
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(
            f"📝 Trade recorded: {symbol} {side} {quantity:.6f} @ ${entry_price:.2f} "
            f"(P&L: ${pnl:.2f})"
        )

        return trade_id

    @handle_errors("Performance Tracker - Update Trade", reraise=False)
    def update_trade(
        self,
        trade_id: int,
        exit_price: float,
        status: str = "closed"
    ):
        """
        🔄 Update an open trade with exit information

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            status: New status (default: "closed")
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get trade details
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = cursor.fetchone()

        if not row:
            logger.error(f"Trade {trade_id} not found")
            conn.close()
            return

        # Calculate P&L
        entry_price = row[6]
        quantity = row[5]
        side = row[4]
        fee = row[8]

        if side == "buy":
            pnl = (exit_price - entry_price) * quantity - fee
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl = (entry_price - exit_price) * quantity - fee
            pnl_percent = ((entry_price - exit_price) / entry_price) * 100

        # Update trade
        cursor.execute("""
            UPDATE trades
            SET exit_price = ?, pnl = ?, pnl_percent = ?, status = ?
            WHERE id = ?
        """, (exit_price, pnl, pnl_percent, status, trade_id))

        conn.commit()
        conn.close()

        logger.info(f"🔄 Trade {trade_id} updated: Exit ${exit_price:.2f}, P&L ${pnl:.2f}")

    def get_performance_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        exchange: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> PerformanceMetrics:
        """
        📊 Calculate comprehensive performance metrics

        Args:
            start_date: Start date filter
            end_date: End date filter
            exchange: Exchange filter
            symbol: Symbol filter

        Returns:
            PerformanceMetrics object with all calculations
        """
        metrics = PerformanceMetrics()

        if not self.enabled:
            return metrics

        # Build query
        query = "SELECT * FROM trades WHERE status = 'closed'"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " ORDER BY timestamp ASC"

        # Fetch trades
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(query, params)
        trades = cursor.fetchall()
        conn.close()

        if not trades:
            return metrics

        # Parse trades
        wins = []
        losses = []
        all_pnl = []
        all_pnl_percent = []
        total_fees = 0.0

        for trade in trades:
            pnl = trade[9]
            pnl_percent = trade[10]
            fee = trade[8]

            all_pnl.append(pnl)
            all_pnl_percent.append(pnl_percent)
            total_fees += fee

            if pnl > 0:
                wins.append(pnl)
            elif pnl < 0:
                losses.append(abs(pnl))

        # Basic metrics
        metrics.total_trades = len(trades)
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = (len(wins) / len(trades)) * 100 if trades else 0.0

        # Profit metrics
        metrics.total_pnl = sum(all_pnl)
        metrics.total_pnl_percent = sum(all_pnl_percent)
        metrics.average_win = sum(wins) / len(wins) if wins else 0.0
        metrics.average_loss = sum(losses) / len(losses) if losses else 0.0
        metrics.profit_factor = (sum(wins) / sum(losses)) if losses else float('inf')
        metrics.best_trade = max(all_pnl) if all_pnl else 0.0
        metrics.worst_trade = min(all_pnl) if all_pnl else 0.0
        metrics.total_fees = total_fees

        # Sharpe ratio calculation
        if self.calculate_sharpe and len(all_pnl_percent) > 1:
            metrics.sharpe_ratio = self._calculate_sharpe_ratio(all_pnl_percent)

        # Max drawdown calculation
        if self.track_drawdown:
            metrics.max_drawdown, metrics.max_drawdown_percent = self._calculate_max_drawdown(all_pnl)

        # Daily metrics
        today = datetime.now().date()
        daily_query = "SELECT * FROM trades WHERE date(timestamp) = ? AND status = 'closed'"

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(daily_query, (today.isoformat(),))
        daily_trades = cursor.fetchall()
        conn.close()

        metrics.daily_trades = len(daily_trades)
        metrics.daily_pnl = sum(trade[9] for trade in daily_trades)

        return metrics

    def _calculate_sharpe_ratio(self, returns: List[float], timeframe: str = '1h') -> float:
        """
        Calculate annualized Sharpe ratio

        Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns * sqrt(periods_per_year)
        """
        if len(returns) < 2:
            return 0.0

        # Calculate periods per year for annualization
        timeframe_to_periods = {
            '1m': 525600, '5m': 105120, '15m': 35040,
            '1h': 8760, '4h': 2190, '1d': 365
        }
        periods_per_year = timeframe_to_periods.get(timeframe, 8760)

        # Convert percentage returns to decimal
        returns_decimal = [r / 100 for r in returns]

        # Calculate mean and std dev
        mean_return = sum(returns_decimal) / len(returns_decimal)
        variance = sum((r - mean_return) ** 2 for r in returns_decimal) / len(returns_decimal)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return 0.0

        # Period risk-free rate
        period_rf = self.risk_free_rate / periods_per_year

        # Calculate Sharpe ratio and annualize
        sharpe = ((mean_return - period_rf) / std_dev) * math.sqrt(periods_per_year)

        return sharpe

    def _calculate_max_drawdown(self, pnl_series: List[float]) -> Tuple[float, float]:
        """
        Calculate maximum drawdown (peak to trough)

        Returns:
            (max_drawdown_dollars, max_drawdown_percent)
        """
        if not pnl_series:
            return 0.0, 0.0

        # Build equity curve
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        max_dd_percent = 0.0

        for pnl in pnl_series:
            equity += pnl

            # Update peak
            if equity > peak:
                peak = equity

            # Calculate drawdown
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
                if peak != 0:
                    max_dd_percent = (drawdown / peak) * 100

        return max_dd, max_dd_percent

    def export_trades(
        self,
        filepath: str,
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bool:
        """
        📤 Export trades to file

        Args:
            filepath: Output file path
            format: "json" or "csv"
            start_date: Start date filter
            end_date: End date filter

        Returns:
            True if successful
        """
        # Build query
        query = "SELECT * FROM trades"
        params = []

        if start_date or end_date:
            query += " WHERE"

        if start_date:
            query += " timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            if start_date:
                query += " AND"
            query += " timestamp <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp ASC"

        # Fetch trades
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(query, params)
        trades = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        if format == "json":
            # Export as JSON
            trades_data = [dict(zip(columns, trade)) for trade in trades]
            with open(filepath, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)

        elif format == "csv":
            # Export as CSV
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                writer.writerows(trades)

        else:
            logger.error(f"Unknown export format: {format}")
            return False

        logger.info(f"📤 Exported {len(trades)} trades to {filepath}")
        return True

    def get_trades_summary(self, limit: int = 10) -> List[Dict]:
        """Get recent trades summary"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, exchange, symbol, side, quantity,
                   entry_price, exit_price, pnl, status
            FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        trades = cursor.fetchall()
        conn.close()

        return [
            {
                'timestamp': t[0],
                'exchange': t[1],
                'symbol': t[2],
                'side': t[3],
                'quantity': t[4],
                'entry_price': t[5],
                'exit_price': t[6],
                'pnl': t[7],
                'status': t[8]
            }
            for t in trades
        ]


# Usage example
if __name__ == "__main__":
    # Test configuration
    config = {
        'performance_tracking': {
            'enabled': True,
            'database_path': 'data/test_trading.db',
            'calculate_sharpe_ratio': True,
            'risk_free_rate': 0.02
        }
    }

    tracker = PerformanceTracker(config)

    # Record some test trades
    print("\n📝 Recording test trades...")

    # Winning trade
    trade_id1 = tracker.record_trade(
        exchange="binance",
        symbol="BTC/USDT",
        side="buy",
        quantity=0.1,
        entry_price=50000,
        exit_price=51000,
        fee=10.0
    )

    # Losing trade
    trade_id2 = tracker.record_trade(
        exchange="binance",
        symbol="ETH/USDT",
        side="buy",
        quantity=1.0,
        entry_price=3000,
        exit_price=2900,
        fee=3.0
    )

    # Get metrics
    print("\n📊 Performance Metrics:")
    metrics = tracker.get_performance_metrics()

    print(f"Total Trades: {metrics.total_trades}")
    print(f"Win Rate: {metrics.win_rate:.1f}%")
    print(f"Total P&L: ${metrics.total_pnl:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_percent:.1f}%)")

    print("\n✅ Performance Tracker test completed!")
