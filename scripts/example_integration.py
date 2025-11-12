#!/usr/bin/env python3
"""
Nexlify - Complete Integration Example
Demonstrates all three advanced features working together
"""

import asyncio
import logging
from datetime import datetime
import json

from nexlify.risk.nexlify_risk_manager import RiskManager
from nexlify.risk.nexlify_circuit_breaker import CircuitBreakerManager
from nexlify.analytics.nexlify_performance_tracker import PerformanceTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedTradingEngine:
    """
    Example trading engine with all three advanced features integrated
    """

    def __init__(self, config_path: str = "config/neural_config.json"):
        """Initialize with all advanced features"""

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize all three features
        self.risk_manager = RiskManager(self.config)
        self.circuit_manager = CircuitBreakerManager(self.config)
        self.performance_tracker = PerformanceTracker(self.config)

        logger.info("🚀 Enhanced Trading Engine initialized")
        logger.info("=" * 70)

    async def execute_protected_trade(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        balance: float,
        confidence: float = 0.7
    ) -> bool:
        """
        Execute a trade with full protection:
        1. Risk validation
        2. Circuit breaker protection
        3. Performance tracking
        """

        logger.info(f"\n📊 Attempting trade: {symbol} {side} {quantity} @ ${price}")

        # STEP 1: Risk Management Validation
        logger.info("🛡️ Step 1: Risk validation...")

        validation = await self.risk_manager.validate_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=confidence
        )

        if not validation.approved:
            logger.error(f"❌ Trade rejected by risk manager: {validation.reason}")
            return False

        # Use adjusted size if recommended
        final_quantity = validation.adjusted_size or quantity
        if validation.adjusted_size:
            logger.info(f"   Adjusted quantity: {quantity} → {final_quantity}")

        logger.info(f"   ✅ Risk check passed")
        logger.info(f"   Stop Loss: ${validation.stop_loss:.2f}")
        logger.info(f"   Take Profit: ${validation.take_profit:.2f}")

        # STEP 2: Execute with Circuit Breaker Protection
        logger.info(f"\n🔌 Step 2: Execute with circuit breaker...")

        breaker = self.circuit_manager.get_or_create(exchange)
        breaker_status = breaker.get_status()

        logger.info(f"   Circuit state: {breaker_status['state']}")
        logger.info(f"   Success rate: {breaker_status['success_rate']}")

        try:
            # Simulate exchange API call with circuit breaker
            result = await breaker.call(
                self._simulate_exchange_order,
                exchange, symbol, side, final_quantity, price
            )

            logger.info(f"   ✅ Order executed successfully")

        except Exception as e:
            logger.error(f"   ❌ Order failed: {e}")
            return False

        # STEP 3: Record in Performance Tracker
        logger.info(f"\n📝 Step 3: Record trade...")

        # For demo, simulate immediate close with small profit
        exit_price = price * 1.02 if side == "buy" else price * 0.98

        trade_id = self.performance_tracker.record_trade(
            exchange=exchange,
            symbol=symbol,
            side=side,
            quantity=final_quantity,
            entry_price=price,
            exit_price=exit_price,
            fee=price * final_quantity * 0.001,  # 0.1% fee
            strategy="demo"
        )

        logger.info(f"   ✅ Trade recorded (ID: {trade_id})")

        # STEP 4: Update Risk Manager
        logger.info(f"\n📊 Step 4: Update risk metrics...")

        self.risk_manager.record_trade_result(
            symbol=symbol,
            side=side,
            entry_price=price,
            exit_price=exit_price,
            quantity=final_quantity,
            balance=balance
        )

        logger.info(f"   ✅ Risk metrics updated")

        return True

    async def _simulate_exchange_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ):
        """Simulate an exchange order (for demo purposes)"""
        await asyncio.sleep(0.1)  # Simulate network delay

        # Simulate occasional failures (10% failure rate)
        import random
        if random.random() < 0.1:
            raise Exception(f"{exchange} API error: Rate limit exceeded")

        return {
            'id': f"order_{int(datetime.now().timestamp())}",
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'status': 'filled'
        }

    def print_comprehensive_status(self):
        """Print status of all systems"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 COMPREHENSIVE STATUS REPORT")
        logger.info("=" * 70)

        # Risk Management Status
        logger.info("\n🛡️ RISK MANAGEMENT:")
        risk_status = self.risk_manager.get_risk_status()
        logger.info(f"   Trading Halted: {risk_status['trading_halted']}")
        logger.info(f"   Daily P&L: {risk_status['net_pnl']}")
        logger.info(f"   Trades Today: {risk_status['trades_today']}")
        logger.info(f"   Loss Remaining: {risk_status['loss_remaining']}")

        # Circuit Breaker Status
        logger.info("\n🔌 CIRCUIT BREAKERS:")
        health = self.circuit_manager.get_health_summary()
        logger.info(f"   Overall Health: {health['overall_health'].upper()}")
        logger.info(f"   Healthy: {health['healthy']}/{health['total_breakers']}")
        logger.info(f"   Failed: {health['failed']}")

        all_breakers = self.circuit_manager.get_all_status()
        for name, status in all_breakers.items():
            logger.info(f"   {name}: {status['state']} ({status['success_rate']})")

        # Performance Metrics
        logger.info("\n📈 PERFORMANCE METRICS:")
        metrics = self.performance_tracker.get_performance_metrics()
        logger.info(f"   Total Trades: {metrics.total_trades}")
        logger.info(f"   Win Rate: {metrics.win_rate:.1f}%")
        logger.info(f"   Total P&L: ${metrics.total_pnl:.2f}")
        logger.info(f"   Profit Factor: {metrics.profit_factor:.2f}")
        if metrics.sharpe_ratio != 0:
            logger.info(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        logger.info(f"   Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_percent:.1f}%)")

        logger.info("\n" + "=" * 70)


async def main():
    """Main demonstration"""

    logger.info("🌆 Nexlify Advanced Features Integration Demo")
    logger.info("=" * 70)

    # Initialize engine
    engine = EnhancedTradingEngine()

    # Simulate a trading session
    logger.info("\n🔥 Starting trading simulation...\n")

    balance = 10000.0  # Starting balance

    # Execute several trades
    trades = [
        ("binance", "BTC/USDT", "buy", 0.01, 50000, 0.75),
        ("binance", "ETH/USDT", "buy", 0.5, 3000, 0.80),
        ("kraken", "BTC/USDT", "buy", 0.01, 50100, 0.65),
        ("binance", "BTC/USDT", "sell", 0.01, 50200, 0.70),
    ]

    for i, (exchange, symbol, side, quantity, price, confidence) in enumerate(trades, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"TRADE #{i}/{len(trades)}")
        logger.info(f"{'='*70}")

        success = await engine.execute_protected_trade(
            exchange=exchange,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            balance=balance,
            confidence=confidence
        )

        if success:
            logger.info(f"\n✅ Trade #{i} completed successfully")
        else:
            logger.info(f"\n❌ Trade #{i} failed")

        await asyncio.sleep(1)  # Pause between trades

    # Print final status
    logger.info("\n")
    engine.print_comprehensive_status()

    # Export trades
    logger.info("\n📤 Exporting trades...")
    engine.performance_tracker.export_trades(
        filepath="demo_trades.json",
        format="json"
    )
    logger.info("   ✅ Trades exported to demo_trades.json")

    logger.info("\n🎉 Demo completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
