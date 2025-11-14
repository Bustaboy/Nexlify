#!/usr/bin/env python3
"""
Nexlify Paper Trading Engine
Simulated trading with real market data for risk-free testing
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


@dataclass
class PaperPosition:
    """Paper trading position"""

    id: str
    symbol: str
    side: str
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    fees_paid: float = 0.0


@dataclass
class PaperTrade:
    """Completed paper trade"""

    id: str
    symbol: str
    side: str
    amount: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percent: float
    fees: float
    strategy: str = ""


class PaperTradingEngine:
    """
    Paper trading engine that simulates real trading
    Uses live market data but no real money
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Initial paper balance
        self.initial_balance = self.config.get("paper_balance", 10000.0)
        self.current_balance = self.initial_balance

        # Trading parameters
        self.fee_rate = self.config.get("fee_rate", 0.001)  # 0.1%
        self.slippage = self.config.get("slippage", 0.0005)  # 0.05%

        # State
        self.positions: Dict[str, PaperPosition] = {}
        self.completed_trades: List[PaperTrade] = []
        self.equity_curve: List[float] = [self.initial_balance]
        self.equity_timestamps: List[datetime] = [datetime.now()]

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_fees_paid = 0.0

        logger.info(
            f"📄 Paper Trading Engine initialized with ${self.initial_balance:,.2f}"
        )

    @handle_errors("Paper Trading", reraise=False)
    async def place_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        order_type: str = "market",
        strategy: str = "",
    ) -> Dict:
        """
        Place a paper trading order

        Args:
            symbol: Trading pair
            side: 'buy' or 'sell'
            amount: Amount to trade
            price: Current market price
            order_type: 'market' or 'limit'
            strategy: Strategy name for tracking

        Returns:
            Order result dict
        """
        # Apply slippage
        if side == "buy":
            execution_price = price * (1 + self.slippage)
        else:
            execution_price = price * (1 - self.slippage)

        if side == "buy":
            return await self._execute_buy(symbol, amount, execution_price, strategy)
        else:
            return await self._execute_sell(symbol, amount, execution_price, strategy)

    async def _execute_buy(
        self, symbol: str, amount: float, price: float, strategy: str
    ) -> Dict:
        """Execute paper buy order"""
        # Calculate cost
        cost = amount * price
        fees = cost * self.fee_rate
        total_cost = cost + fees

        # Check balance
        if total_cost > self.current_balance:
            logger.warning(
                f"❌ Insufficient paper balance: ${self.current_balance:.2f} < ${total_cost:.2f}"
            )
            return {
                "success": False,
                "error": "Insufficient balance",
                "required": total_cost,
                "available": self.current_balance,
            }

        # Create position
        position_id = str(uuid.uuid4())[:8]
        position = PaperPosition(
            id=position_id,
            symbol=symbol,
            side="long",
            amount=amount,
            entry_price=price,
            entry_time=datetime.now(),
            current_price=price,
            fees_paid=fees,
        )

        # Update balances
        self.current_balance -= total_cost
        self.total_fees_paid += fees
        self.positions[position_id] = position

        logger.info(
            f"📝 Paper BUY: {amount:.4f} {symbol} @ ${price:.2f} (Cost: ${total_cost:.2f})"
        )

        return {
            "success": True,
            "position_id": position_id,
            "symbol": symbol,
            "side": "buy",
            "amount": amount,
            "price": price,
            "cost": cost,
            "fees": fees,
            "total_cost": total_cost,
            "remaining_balance": self.current_balance,
        }

    async def _execute_sell(
        self, symbol: str, amount: float, price: float, strategy: str
    ) -> Dict:
        """Execute paper sell order"""
        # Find matching position
        position = None
        for pos_id, pos in self.positions.items():
            if pos.symbol == symbol and pos.side == "long":
                position = pos
                break

        if not position:
            logger.warning(f"❌ No open position found for {symbol}")
            return {"success": False, "error": f"No open position for {symbol}"}

        # Calculate proceeds
        proceeds = amount * price
        fees = proceeds * self.fee_rate
        net_proceeds = proceeds - fees

        # Calculate PnL
        entry_cost = position.amount * position.entry_price
        gross_pnl = proceeds - entry_cost
        net_pnl = gross_pnl - position.fees_paid - fees
        pnl_percent = (net_pnl / entry_cost) * 100

        # Update balances
        self.current_balance += net_proceeds
        self.total_fees_paid += fees

        # Record trade
        trade = PaperTrade(
            id=position.id,
            symbol=symbol,
            side="long",
            amount=amount,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=net_pnl,
            pnl_percent=pnl_percent,
            fees=position.fees_paid + fees,
            strategy=strategy,
        )

        self.completed_trades.append(trade)
        self.total_trades += 1

        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        # Remove position
        del self.positions[position.id]

        logger.info(
            f"📝 Paper SELL: {amount:.4f} {symbol} @ ${price:.2f} "
            f"(PnL: ${net_pnl:.2f} / {pnl_percent:.2f}%)"
        )

        return {
            "success": True,
            "trade_id": trade.id,
            "symbol": symbol,
            "side": "sell",
            "amount": amount,
            "price": price,
            "proceeds": proceeds,
            "fees": fees,
            "net_proceeds": net_proceeds,
            "pnl": net_pnl,
            "pnl_percent": pnl_percent,
            "remaining_balance": self.current_balance,
        }

    async def update_positions(self, market_prices: Dict[str, float]):
        """Update all open positions with current market prices"""
        for position in self.positions.values():
            if position.symbol in market_prices:
                position.current_price = market_prices[position.symbol]

                # Calculate unrealized PnL
                entry_cost = position.amount * position.entry_price
                current_value = position.amount * position.current_price
                position.unrealized_pnl = (
                    current_value - entry_cost - position.fees_paid
                )

        # Update equity curve
        total_equity = self.get_total_equity(market_prices)
        self.equity_curve.append(total_equity)
        self.equity_timestamps.append(datetime.now())

    def get_total_equity(self, market_prices: Dict[str, float]) -> float:
        """Calculate total equity including unrealized positions"""
        equity = self.current_balance

        for position in self.positions.values():
            if position.symbol in market_prices:
                current_value = position.amount * market_prices[position.symbol]
                equity += current_value

        return equity

    def get_statistics(self) -> Dict:
        """Get comprehensive paper trading statistics"""
        total_equity = self.current_balance + sum(
            p.amount * p.current_price for p in self.positions.values()
        )

        total_return = total_equity - self.initial_balance
        total_return_percent = (total_return / self.initial_balance) * 100

        win_rate = (
            (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0
        )

        # Average win/loss
        wins = [t.pnl for t in self.completed_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in self.completed_trades if t.pnl <= 0]

        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Profit factor
        total_wins = sum(wins)
        total_losses = sum(losses)
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        return {
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "total_equity": total_equity,
            "total_return": total_return,
            "total_return_percent": total_return_percent,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_fees_paid": self.total_fees_paid,
            "open_positions": len(self.positions),
            "unrealized_pnl": sum(p.unrealized_pnl for p in self.positions.values()),
        }

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return [
            {
                "id": pos.id,
                "symbol": pos.symbol,
                "side": pos.side,
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "entry_time": pos.entry_time.isoformat(),
            }
            for pos in self.positions.values()
        ]

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get completed trade history"""
        trades = self.completed_trades[-limit:]

        return [
            {
                "id": t.id,
                "symbol": t.symbol,
                "amount": t.amount,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": t.pnl,
                "pnl_percent": t.pnl_percent,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "duration": str(t.exit_time - t.entry_time),
            }
            for t in trades
        ]

    def save_session(self, filepath: str = "paper_trading/session.json"):
        """Save paper trading session to file"""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            stats = self.get_statistics()
            positions = self.get_open_positions()
            history = self.get_trade_history()

            data = {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats,
                "open_positions": positions,
                "trade_history": history,
                "equity_curve": self.equity_curve,
                "equity_timestamps": [ts.isoformat() for ts in self.equity_timestamps],
            }

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"💾 Paper trading session saved: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def generate_report(self) -> str:
        """Generate formatted performance report"""
        stats = self.get_statistics()

        report = f"""
{'='*70}
PAPER TRADING PERFORMANCE REPORT
{'='*70}

BALANCE
  Initial Balance:      ${stats['initial_balance']:>12,.2f}
  Current Cash:         ${stats['current_balance']:>12,.2f}
  Total Equity:         ${stats['total_equity']:>12,.2f}
  Total Return:         ${stats['total_return']:>12,.2f} ({stats['total_return_percent']:>6.2f}%)

TRADING STATISTICS
  Total Trades:         {stats['total_trades']:>12}
  Winning Trades:       {stats['winning_trades']:>12}
  Losing Trades:        {stats['losing_trades']:>12}
  Win Rate:             {stats['win_rate']:>11.2f}%

PERFORMANCE METRICS
  Average Win:          ${stats['avg_win']:>12,.2f}
  Average Loss:         ${stats['avg_loss']:>12,.2f}
  Profit Factor:        {stats['profit_factor']:>12.2f}
  Total Fees Paid:      ${stats['total_fees_paid']:>12,.2f}

CURRENT POSITIONS
  Open Positions:       {stats['open_positions']:>12}
  Unrealized P&L:       ${stats['unrealized_pnl']:>12,.2f}

{'='*70}
"""
        return report


if __name__ == "__main__":

    async def main():
        print("=" * 70)
        print("NEXLIFY PAPER TRADING DEMO")
        print("=" * 70)

        # Initialize
        engine = PaperTradingEngine(
            {"paper_balance": 10000, "fee_rate": 0.001, "slippage": 0.0005}
        )

        # Simulate some trades
        print("\n📝 Executing paper trades...\n")

        # Buy
        result1 = await engine.place_order("BTC/USDT", "buy", 0.1, 45000)
        print(f"Trade 1: {result1}")

        # Update price and sell
        await engine.update_positions({"BTC/USDT": 47000})

        result2 = await engine.place_order("BTC/USDT", "sell", 0.1, 47000)
        print(f"Trade 2: {result2}")

        # Generate report
        print(engine.generate_report())

        # Save session
        engine.save_session()

    asyncio.run(main())
