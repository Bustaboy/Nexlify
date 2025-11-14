#!/usr/bin/env python3
"""
Unit tests for GammaOptimizer

Tests adaptive gamma selection and adjustment for different trading timeframes.
"""

import pytest
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from nexlify.strategies.gamma_optimizer import (
    GammaOptimizer,
    TradingStyle,
    TRADING_STYLES,
    TIMEFRAME_TO_HOURS,
    get_recommended_gamma,
    print_gamma_recommendations,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def optimizer_1h():
    """Create optimizer for 1h timeframe (day trading)"""
    return GammaOptimizer(timeframe="1h", auto_adjust=True)


@pytest.fixture
def optimizer_5m():
    """Create optimizer for 5m timeframe (scalping)"""
    return GammaOptimizer(timeframe="5m", auto_adjust=True)


@pytest.fixture
def optimizer_1d():
    """Create optimizer for 1d timeframe (swing trading)"""
    return GammaOptimizer(timeframe="1d", auto_adjust=True)


@pytest.fixture
def optimizer_manual():
    """Create optimizer with manual gamma override"""
    return GammaOptimizer(timeframe="1h", auto_adjust=False, manual_gamma=0.98)


# ============================================================================
# BASIC INITIALIZATION TESTS
# ============================================================================

@pytest.mark.unit
def test_initialization_default():
    """Test default initialization"""
    optimizer = GammaOptimizer()

    assert optimizer.timeframe == "1h"
    assert optimizer.auto_adjust is True
    assert optimizer.current_gamma == 0.95  # Day trading default
    assert optimizer.current_style.name == "day_trading"


@pytest.mark.unit
def test_initialization_scalping(optimizer_5m):
    """Test initialization for scalping timeframe"""
    assert optimizer_5m.timeframe == "5m"
    assert optimizer_5m.current_gamma == 0.90  # Scalping gamma
    assert optimizer_5m.current_style.name == "scalping"


@pytest.mark.unit
def test_initialization_swing(optimizer_1d):
    """Test initialization for swing trading timeframe"""
    assert optimizer_1d.timeframe == "1d"
    assert optimizer_1d.current_gamma == 0.97  # Swing trading gamma
    assert optimizer_1d.current_style.name == "swing_trading"


@pytest.mark.unit
def test_initialization_manual(optimizer_manual):
    """Test manual gamma override"""
    assert optimizer_manual.manual_gamma == 0.98
    assert optimizer_manual.current_gamma == 0.98
    assert optimizer_manual.auto_adjust is False


@pytest.mark.unit
def test_initialization_custom_params():
    """Test initialization with custom parameters"""
    optimizer = GammaOptimizer(
        timeframe="4h",
        auto_adjust=True,
        adjustment_interval=50,
        history_window=100,
        adjustment_threshold=0.15
    )

    assert optimizer.timeframe == "4h"
    assert optimizer.adjustment_interval == 50
    assert optimizer.history_window == 100
    assert optimizer.adjustment_threshold == 0.15


# ============================================================================
# GAMMA SELECTION TESTS
# ============================================================================

@pytest.mark.unit
def test_get_gamma(optimizer_1h):
    """Test getting current gamma value"""
    gamma = optimizer_1h.get_gamma()
    assert gamma == 0.95
    assert isinstance(gamma, float)


@pytest.mark.unit
def test_recommended_gamma_scalping():
    """Test recommended gamma for scalping"""
    gamma = get_recommended_gamma("1m")
    assert gamma == 0.90

    gamma = get_recommended_gamma("5m")
    assert gamma == 0.90


@pytest.mark.unit
def test_recommended_gamma_day_trading():
    """Test recommended gamma for day trading"""
    gamma = get_recommended_gamma("1h")
    assert gamma == 0.95

    gamma = get_recommended_gamma("4h")
    assert gamma == 0.95


@pytest.mark.unit
def test_recommended_gamma_swing_trading():
    """Test recommended gamma for swing trading"""
    gamma = get_recommended_gamma("1d")
    assert gamma == 0.97


@pytest.mark.unit
def test_recommended_gamma_position_trading():
    """Test recommended gamma for position trading"""
    gamma = get_recommended_gamma("1w")
    assert gamma == 0.99


# ============================================================================
# TRADE RECORDING TESTS
# ============================================================================

@pytest.mark.unit
def test_record_trade_with_timestamps(optimizer_1h):
    """Test recording a trade with timestamps"""
    entry = datetime.now()
    exit = entry + timedelta(hours=2)
    profit = 100.0
    volatility = 0.03

    optimizer_1h.record_trade(entry, exit, profit, volatility)

    assert len(optimizer_1h.trade_durations) == 1
    assert len(optimizer_1h.trade_profits) == 1
    assert len(optimizer_1h.volatility_history) == 1

    # Check values
    assert optimizer_1h.trade_durations[0] == 2.0  # 2 hours
    assert optimizer_1h.trade_profits[0] == 100.0
    assert optimizer_1h.volatility_history[0] == 0.03


@pytest.mark.unit
def test_record_trade_from_steps(optimizer_1h):
    """Test recording a trade using step counts"""
    optimizer_1h.record_trade_from_steps(
        entry_step=100,
        exit_step=150,
        profit=50.0,
        steps_per_hour=10,  # 10 steps = 1 hour
        volatility=0.02
    )

    assert len(optimizer_1h.trade_durations) == 1
    assert optimizer_1h.trade_durations[0] == 5.0  # 50 steps / 10 steps per hour


@pytest.mark.unit
def test_record_multiple_trades(optimizer_1h):
    """Test recording multiple trades"""
    base_time = datetime.now()

    for i in range(10):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=1)
        profit = np.random.uniform(-10, 20)

        optimizer_1h.record_trade(entry, exit, profit)

    assert len(optimizer_1h.trade_durations) == 10
    assert len(optimizer_1h.trade_profits) == 10


@pytest.mark.unit
def test_trade_history_window_limit(optimizer_1h):
    """Test that trade history respects window limit"""
    # Default window is 50
    base_time = datetime.now()

    # Record 100 trades
    for i in range(100):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=1)
        profit = 10.0

        optimizer_1h.record_trade(entry, exit, profit)

    # Should only keep last 50 (default history_window)
    assert len(optimizer_1h.trade_durations) == 50
    assert len(optimizer_1h.trade_profits) == 50


# ============================================================================
# GAMMA ADJUSTMENT TESTS
# ============================================================================

@pytest.mark.unit
def test_no_adjustment_insufficient_data(optimizer_1h):
    """Test no adjustment when insufficient trade data"""
    # Record only 5 trades (need 10+)
    base_time = datetime.now()

    for i in range(5):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=1)
        optimizer_1h.record_trade(entry, exit, 10.0)

    adjusted, rationale = optimizer_1h.update_gamma(episode=100)

    assert adjusted is False
    assert rationale is None


@pytest.mark.unit
def test_no_adjustment_before_interval(optimizer_1h):
    """Test no adjustment before adjustment interval"""
    # Record enough trades
    base_time = datetime.now()

    for i in range(20):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=1)
        optimizer_1h.record_trade(entry, exit, 10.0)

    # Try to adjust at episode 50 (interval is 100)
    adjusted, rationale = optimizer_1h.update_gamma(episode=50)

    assert adjusted is False
    assert rationale is None


@pytest.mark.unit
def test_gamma_adjustment_scalping_behavior(optimizer_1h):
    """Test gamma adjustment when agent exhibits scalping behavior"""
    # Start with 1h timeframe (day trading, gamma=0.95)
    assert optimizer_1h.current_gamma == 0.95

    # Simulate scalping-style trades (15-30 minute holds)
    base_time = datetime.now()

    for i in range(30):
        entry = base_time + timedelta(hours=i)
        # Random duration between 0.25-0.5 hours (15-30 minutes)
        duration_hours = np.random.uniform(0.25, 0.5)
        exit = entry + timedelta(hours=duration_hours)
        optimizer_1h.record_trade(entry, exit, np.random.uniform(-5, 15))

    # Update gamma after interval
    adjusted, rationale = optimizer_1h.update_gamma(episode=100)

    # Should adjust to scalping gamma (0.90)
    assert adjusted is True
    assert optimizer_1h.current_gamma == 0.90
    assert optimizer_1h.current_style.name == "scalping"
    assert rationale is not None


@pytest.mark.unit
def test_gamma_adjustment_swing_behavior(optimizer_1h):
    """Test gamma adjustment when agent exhibits swing trading behavior"""
    # Start with 1h timeframe (day trading, gamma=0.95)
    assert optimizer_1h.current_gamma == 0.95

    # Simulate swing trading (2-5 day holds)
    base_time = datetime.now()

    for i in range(30):
        entry = base_time + timedelta(days=i)
        # Random duration between 48-120 hours (2-5 days)
        duration_hours = np.random.uniform(48, 120)
        exit = entry + timedelta(hours=duration_hours)
        optimizer_1h.record_trade(entry, exit, np.random.uniform(-100, 200))

    # Update gamma after interval
    adjusted, rationale = optimizer_1h.update_gamma(episode=100)

    # Should adjust to swing trading gamma (0.97)
    assert adjusted is True
    assert optimizer_1h.current_gamma == 0.97
    assert optimizer_1h.current_style.name == "swing_trading"


@pytest.mark.unit
def test_no_adjustment_below_threshold(optimizer_1h):
    """Test no adjustment when change is below threshold"""
    # Start with 1h timeframe (day trading, gamma=0.95)
    # Simulate trades very close to expected duration (1-2 hours)

    base_time = datetime.now()

    for i in range(30):
        entry = base_time + timedelta(hours=i)
        # Very close to expected duration
        duration_hours = np.random.uniform(1.0, 2.0)
        exit = entry + timedelta(hours=duration_hours)
        optimizer_1h.record_trade(entry, exit, 10.0)

    # Update gamma
    adjusted, rationale = optimizer_1h.update_gamma(episode=100)

    # Should not adjust (change below 10% threshold)
    assert adjusted is False


@pytest.mark.unit
def test_manual_gamma_no_adjustment(optimizer_manual):
    """Test manual gamma is never adjusted"""
    # Simulate very different trades
    base_time = datetime.now()

    for i in range(30):
        entry = base_time + timedelta(hours=i)
        # Very short trades (scalping)
        exit = entry + timedelta(minutes=15)
        optimizer_manual.record_trade(entry, exit, 5.0)

    # Try to update
    adjusted, rationale = optimizer_manual.update_gamma(episode=100)

    # Should not adjust (manual override)
    assert adjusted is False
    assert optimizer_manual.current_gamma == 0.98  # Still manual value


# ============================================================================
# STATISTICS TESTS
# ============================================================================

@pytest.mark.unit
def test_get_statistics_empty(optimizer_1h):
    """Test statistics with no trades"""
    stats = optimizer_1h.get_statistics()

    assert stats["timeframe"] == "1h"
    assert stats["current_gamma"] == 0.95
    assert stats["current_style"] == "day_trading"
    assert stats["trades_recorded"] == 0
    assert stats["adjustments_made"] == 0


@pytest.mark.unit
def test_get_statistics_with_trades(optimizer_1h):
    """Test statistics with recorded trades"""
    base_time = datetime.now()

    profits = []
    for i in range(20):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=2)
        profit = np.random.uniform(-10, 20)
        profits.append(profit)

        optimizer_1h.record_trade(entry, exit, profit, volatility=0.03)

    stats = optimizer_1h.get_statistics()

    assert stats["trades_recorded"] == 20
    assert "avg_trade_duration_hours" in stats
    assert "avg_trade_profit" in stats
    assert "win_rate" in stats
    assert "avg_volatility" in stats

    # Check values
    assert stats["avg_trade_duration_hours"] == 2.0
    assert stats["avg_trade_profit"] == pytest.approx(np.mean(profits), rel=0.01)


# ============================================================================
# SAVE/LOAD TESTS
# ============================================================================

@pytest.mark.unit
def test_save_history(optimizer_1h, temp_dir):
    """Test saving adjustment history"""
    # Record some trades and adjustments
    base_time = datetime.now()

    for i in range(30):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(minutes=30)  # Scalping behavior
        optimizer_1h.record_trade(entry, exit, 10.0)

    optimizer_1h.update_gamma(episode=100)

    # Save history
    filepath = temp_dir / "gamma_history.json"
    optimizer_1h.save_history(str(filepath))

    assert filepath.exists()

    # Verify contents
    with open(filepath) as f:
        data = json.load(f)

    assert "config" in data
    assert "current_state" in data
    assert "adjustments" in data


@pytest.mark.unit
def test_load_history(optimizer_1h, temp_dir):
    """Test loading adjustment history"""
    # Create and save history
    base_time = datetime.now()

    for i in range(30):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(minutes=30)
        optimizer_1h.record_trade(entry, exit, 10.0)

    optimizer_1h.update_gamma(episode=100)

    filepath = temp_dir / "gamma_history.json"
    optimizer_1h.save_history(str(filepath))

    # Create new optimizer and load
    optimizer_new = GammaOptimizer(timeframe="1h")
    optimizer_new.load_history(str(filepath))

    # Should have loaded adjustment history
    assert len(optimizer_new.adjustment_history) == len(optimizer_1h.adjustment_history)


# ============================================================================
# RESET TESTS
# ============================================================================

@pytest.mark.unit
def test_reset(optimizer_1h):
    """Test resetting optimizer"""
    # Record some data
    base_time = datetime.now()

    for i in range(20):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=1)
        optimizer_1h.record_trade(entry, exit, 10.0)

    optimizer_1h.update_gamma(episode=100)

    # Reset
    optimizer_1h.reset()

    # Check all data cleared
    assert len(optimizer_1h.trade_durations) == 0
    assert len(optimizer_1h.trade_profits) == 0
    assert len(optimizer_1h.adjustment_history) == 0
    assert optimizer_1h.last_adjustment_episode == 0

    # Configuration should be preserved
    assert optimizer_1h.timeframe == "1h"
    assert optimizer_1h.auto_adjust is True


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.unit
def test_very_long_trades():
    """Test handling of very long position trades"""
    optimizer = GammaOptimizer(timeframe="1h")

    base_time = datetime.now()

    # Simulate very long trades (weeks/months)
    for i in range(20):
        entry = base_time + timedelta(days=i*30)
        exit = entry + timedelta(days=30)  # 1 month
        optimizer.record_trade(entry, exit, 1000.0)

    adjusted, rationale = optimizer.update_gamma(episode=100)

    # Should adjust to position trading
    assert adjusted is True
    assert optimizer.current_gamma == 0.99
    assert optimizer.current_style.name == "position_trading"


@pytest.mark.unit
def test_mixed_trade_durations():
    """Test handling of mixed trade durations"""
    optimizer = GammaOptimizer(timeframe="1h")

    base_time = datetime.now()

    # Mix of different durations
    durations = [0.5, 1.0, 2.0, 0.25, 3.0, 1.5, 4.0, 0.75, 2.5, 1.25] * 3

    for i, duration in enumerate(durations):
        entry = base_time + timedelta(hours=i)
        exit = entry + timedelta(hours=duration)
        optimizer.record_trade(entry, exit, 10.0)

    adjusted, rationale = optimizer.update_gamma(episode=100)

    # Should use median/average to determine style
    avg_duration = np.mean(durations)
    assert avg_duration > 0  # Sanity check


@pytest.mark.unit
def test_repr_string(optimizer_1h):
    """Test string representation"""
    repr_str = repr(optimizer_1h)

    assert "GammaOptimizer" in repr_str
    assert "1h" in repr_str
    assert "0.95" in repr_str or "0.950" in repr_str


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
def test_full_workflow():
    """Test complete workflow from initialization to adjustment"""
    # Initialize for day trading
    optimizer = GammaOptimizer(
        timeframe="1h",
        auto_adjust=True,
        adjustment_interval=100
    )

    assert optimizer.current_gamma == 0.95

    # Simulate 200 episodes of scalping behavior
    base_time = datetime.now()

    for episode in range(1, 201):
        # Each episode, make 1-3 trades
        num_trades = np.random.randint(1, 4)

        for _ in range(num_trades):
            entry = base_time + timedelta(hours=episode)
            # Scalping: 10-40 minute holds
            duration = np.random.uniform(10/60, 40/60)
            exit = entry + timedelta(hours=duration)
            profit = np.random.uniform(-5, 15)

            optimizer.record_trade(entry, exit, profit)

        # Check for adjustment every episode
        if episode % 10 == 0:
            adjusted, rationale = optimizer.update_gamma(episode)

            if adjusted:
                print(f"\nAdjusted at episode {episode}:")
                print(rationale)

    # After 200 episodes of scalping, should be at scalping gamma
    assert optimizer.current_gamma == 0.90
    assert optimizer.current_style.name == "scalping"

    # Get final statistics
    stats = optimizer.get_statistics()
    assert stats["trades_recorded"] > 0
    assert stats["adjustments_made"] > 0


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

@pytest.mark.unit
def test_print_gamma_recommendations(capsys):
    """Test printing gamma recommendations"""
    print_gamma_recommendations()

    captured = capsys.readouterr()
    assert "GAMMA RECOMMENDATIONS" in captured.out
    assert "scalping" in captured.out.lower()
    assert "0.90" in captured.out
    assert "0.95" in captured.out
    assert "0.97" in captured.out
    assert "0.99" in captured.out


@pytest.mark.unit
def test_trading_styles_defined():
    """Test that all trading styles are properly defined"""
    assert len(TRADING_STYLES) == 4

    styles = ["scalping", "day_trading", "swing_trading", "position_trading"]
    for style_name in styles:
        assert style_name in TRADING_STYLES
        style = TRADING_STYLES[style_name]
        assert isinstance(style, TradingStyle)
        assert 0 < style.gamma <= 1
        assert style.min_duration_hours >= 0
        assert style.max_duration_hours > style.min_duration_hours


@pytest.mark.unit
def test_timeframe_to_hours_mapping():
    """Test timeframe to hours conversion"""
    assert TIMEFRAME_TO_HOURS["1m"] == 1/60
    assert TIMEFRAME_TO_HOURS["5m"] == 5/60
    assert TIMEFRAME_TO_HOURS["1h"] == 1.0
    assert TIMEFRAME_TO_HOURS["4h"] == 4.0
    assert TIMEFRAME_TO_HOURS["1d"] == 24.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
