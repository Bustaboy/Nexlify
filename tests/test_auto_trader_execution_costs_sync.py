#!/usr/bin/env python3

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from nexlify.core.nexlify_auto_trader import AutoTrader, TradeExecution


def test_get_exchange_execution_costs_uses_active_exchange_data_sync():
    trader = AutoTrader({"require_actual_execution_costs": False})
    exchange = AsyncMock()
    exchange.fetch_trading_fee = AsyncMock(return_value={"taker": 0.0012})
    exchange.fetch_order_book = AsyncMock(
        return_value={"asks": [[100.0, 0.5], [100.2, 0.5]], "bids": [[99.8, 1.0]]}
    )
    exchange.load_markets = AsyncMock(return_value={"BTC/USDT": {"taker": 0.0015}})

    trader.neural_net = Mock()
    trader.neural_net.exchanges = {"binance": exchange}

    fee_rate, slippage = asyncio.run(
        trader._get_exchange_execution_costs(
            exchange_id="binance",
            symbol="BTC/USDT",
            side="buy",
            amount=1.0,
            reference_price=100.0,
        )
    )

    assert fee_rate == 0.0012
    assert slippage >= 0


def test_execute_trade_skips_when_edge_below_costs_sync():
    trader = AutoTrader({"require_actual_execution_costs": False})
    exchange = AsyncMock()
    exchange.fetch_ticker = AsyncMock(return_value={"last": 100.0})
    exchange.create_market_buy_order = AsyncMock(return_value={"status": "closed", "id": "o1"})
    exchange.fetch_trading_fee = AsyncMock(return_value={"taker": 0.002})
    exchange.fetch_order_book = AsyncMock(return_value={"asks": [[100.5, 100]], "bids": [[99.5, 100]]})
    exchange.load_markets = AsyncMock(return_value={"BTC/USDT": {"taker": 0.002}})

    trader.neural_net = Mock()
    trader.neural_net.exchanges = {"binance": exchange}
    trader.get_available_balance = AsyncMock(return_value=1000.0)

    pair = SimpleNamespace(symbol="BTC/USDT", exchanges=["binance"], profit_score=0.10)
    result = asyncio.run(trader.execute_trade(pair))

    assert result is False


def test_close_position_applies_live_exit_costs_sync():
    trader = AutoTrader({"require_actual_execution_costs": False})
    exchange = AsyncMock()
    exchange.create_market_sell_order = AsyncMock(return_value={"status": "closed", "id": "s1"})
    exchange.fetch_trading_fee = AsyncMock(return_value={"taker": 0.001})
    exchange.fetch_order_book = AsyncMock(return_value={"bids": [[99.0, 10]], "asks": [[101.0, 10]]})
    exchange.load_markets = AsyncMock(return_value={"BTC/USDT": {"taker": 0.001}})

    trader.neural_net = Mock()
    trader.neural_net.exchanges = {"binance": exchange}
    trader.get_current_price = AsyncMock(return_value=100.0)

    trade = TradeExecution(
        trade_id="t1",
        symbol="BTC/USDT",
        exchange="binance",
        side="buy",
        amount=1.0,
        price=100.0,
        timestamp=datetime.now(),
        profit_target=110.0,
        stop_loss=90.0,
        strategy="test",
        status="open",
    )
    trade.execution_fee_rate = 0.001
    trader.active_trades = {"t1": trade}

    result = asyncio.run(trader.close_position("t1", "test-close"))

    assert result is True
    assert "t1" not in trader.active_trades


def test_get_exchange_execution_costs_strict_raises_when_live_data_missing_sync():
    trader = AutoTrader({"require_actual_execution_costs": True})
    exchange = AsyncMock()
    exchange.fetch_trading_fee = AsyncMock(side_effect=Exception("not supported"))
    exchange.load_markets = AsyncMock(side_effect=Exception("not supported"))
    exchange.fetch_order_book = AsyncMock(side_effect=Exception("not supported"))

    trader.neural_net = Mock()
    trader.neural_net.exchanges = {"binance": exchange}

    try:
        asyncio.run(
            trader._get_exchange_execution_costs(
                exchange_id="binance",
                symbol="BTC/USDT",
                side="buy",
                amount=1.0,
                reference_price=100.0,
            )
        )
        assert False, "Expected RuntimeError in strict mode when live costs are unavailable"
    except RuntimeError as exc:
        assert "TRADING BLOCKED" in str(exc)
