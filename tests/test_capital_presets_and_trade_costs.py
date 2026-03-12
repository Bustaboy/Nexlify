#!/usr/bin/env python3

from nexlify.config.crypto_trading_config import (
    CryptoTradingConfig,
    get_capital_scaling_preset,
)


def test_capital_scaling_presets_match_expected_buckets():
    micro = get_capital_scaling_preset(100)
    small = get_capital_scaling_preset(250)
    scaled = get_capital_scaling_preset(1000)

    assert micro.name == "micro_capital"
    assert small.name == "small_capital"
    assert scaled.name == "scaled_capital"


def test_round_trip_cost_and_profitability_gate_include_costs():
    config = CryptoTradingConfig(
        trading_network="static",
        use_dynamic_fees=False,
        fee_rate=0.001,
        slippage=0.0005,
    )

    trade_size = 100.0
    cost = config.estimate_total_round_trip_cost(trade_size)

    # 0.1% entry + 0.1% exit + 0.05% entry slip + 0.05% exit slip => ~0.30%
    assert 0.25 <= (cost / trade_size) * 100 <= 0.35

    # 0.2% edge should fail net profitability gate with 0.1% safety buffer
    assert not config.is_trade_net_profitable(expected_move_pct=0.002, trade_size_usd=trade_size)

    # 0.5% edge should pass under same conditions
    assert config.is_trade_net_profitable(expected_move_pct=0.005, trade_size_usd=trade_size)


def test_config_auto_scales_with_capital_growth():
    config = CryptoTradingConfig(initial_balance=100, trading_network="static", use_dynamic_fees=False)

    # Initialized from balance bucket
    assert config.capital_preset_name == "micro_capital"
    assert config.max_concurrent_trades == 1

    # Simulate growth: upgrade to next preset
    config.refresh_capital_scaling(300)
    assert config.capital_preset_name == "small_capital"
    assert config.max_concurrent_trades == 2

    # Simulate more growth: upgrade again
    config.refresh_capital_scaling(1200)
    assert config.capital_preset_name == "scaled_capital"
    assert config.max_concurrent_trades == 3


def test_calculate_expected_actual_cost_requires_live_inputs_when_requested():
    config = CryptoTradingConfig(trading_network="static", use_dynamic_fees=False, fee_rate=0.001, slippage=0.0005)

    try:
        config.calculate_expected_actual_cost(
            trade_size_usd=100.0,
            require_actual_inputs=True,
        )
        assert False, "Expected RuntimeError when strict actual inputs are missing"
    except RuntimeError:
        assert True


def test_calculate_expected_actual_cost_uses_explicit_inputs_over_estimates():
    config = CryptoTradingConfig(trading_network="static", use_dynamic_fees=False, fee_rate=0.001, slippage=0.0005)

    baseline = config.estimate_total_round_trip_cost(100.0)
    actual = config.calculate_expected_actual_cost(
        trade_size_usd=100.0,
        actual_entry_fee_rate=0.002,
        actual_exit_fee_rate=0.002,
        actual_entry_slippage=0.001,
        actual_exit_slippage=0.001,
    )

    assert actual > baseline
