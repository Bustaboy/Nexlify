#!/usr/bin/env python3
"""
Unit tests for GammaSelector and MultiGammaTrainer

Tests static gamma selection and multi-gamma parallel training.
"""

import pytest
import numpy as np

from nexlify.strategies.gamma_selector import (
    GammaSelector,
    TradingStyle,
    TRADING_STYLES,
    TIMEFRAME_TO_HOURS,
    get_recommended_gamma,
    print_gamma_recommendations,
)
from nexlify.strategies.multi_gamma_trainer import (
    MultiGammaTrainer,
    HardwareBenchmark,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def selector_1h():
    """Create selector for 1h timeframe (day trading)"""
    return GammaSelector(timeframe="1h")


@pytest.fixture
def selector_5m():
    """Create selector for 5m timeframe (scalping)"""
    return GammaSelector(timeframe="5m")


@pytest.fixture
def selector_1d():
    """Create selector for 1d timeframe (swing trading)"""
    return GammaSelector(timeframe="1d")


@pytest.fixture
def selector_manual():
    """Create selector with manual gamma override"""
    return GammaSelector(manual_gamma=0.98)


# ============================================================================
# GAMMA SELECTOR TESTS
# ============================================================================

@pytest.mark.unit
def test_initialization_default():
    """Test default initialization"""
    selector = GammaSelector()
    assert selector.timeframe == "1h"
    assert selector.gamma == 0.95  # Day trading default
    assert selector.get_style_name() == "day_trading"


@pytest.mark.unit
def test_initialization_scalping(selector_5m):
    """Test initialization for scalping timeframe"""
    assert selector_5m.timeframe == "5m"
    assert selector_5m.gamma == 0.90  # Scalping gamma
    assert selector_5m.get_style_name() == "scalping"


@pytest.mark.unit
def test_initialization_swing(selector_1d):
    """Test initialization for swing trading timeframe"""
    assert selector_1d.timeframe == "1d"
    assert selector_1d.gamma == 0.97  # Swing trading gamma
    assert selector_1d.get_style_name() == "swing_trading"


@pytest.mark.unit
def test_initialization_manual(selector_manual):
    """Test manual gamma override"""
    assert selector_manual.manual_gamma == 0.98
    assert selector_manual.gamma == 0.98


@pytest.mark.unit
def test_get_gamma(selector_1h):
    """Test getting current gamma value"""
    gamma = selector_1h.get_gamma()
    assert gamma == 0.95
    assert isinstance(gamma, float)


@pytest.mark.unit
def test_recommended_gamma_scalping():
    """Test recommended gamma for scalping"""
    assert get_recommended_gamma("1m") == 0.90
    assert get_recommended_gamma("5m") == 0.90


@pytest.mark.unit
def test_recommended_gamma_day_trading():
    """Test recommended gamma for day trading"""
    assert get_recommended_gamma("1h") == 0.95
    assert get_recommended_gamma("4h") == 0.95


@pytest.mark.unit
def test_recommended_gamma_swing_trading():
    """Test recommended gamma for swing trading"""
    assert get_recommended_gamma("1d") == 0.97


@pytest.mark.unit
def test_recommended_gamma_position_trading():
    """Test recommended gamma for position trading"""
    assert get_recommended_gamma("1w") == 0.99


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


@pytest.mark.unit
def test_print_gamma_recommendations(capsys):
    """Test printing gamma recommendations"""
    print_gamma_recommendations()
    captured = capsys.readouterr()
    assert "GAMMA RECOMMENDATIONS" in captured.out
    assert "scalping" in captured.out.lower()
    assert "0.90" in captured.out
    assert "0.95" in captured.out


# ============================================================================
# HARDWARE BENCHMARK TESTS
# ============================================================================

@pytest.mark.unit
def test_hardware_benchmark_init():
    """Test hardware benchmark initialization"""
    benchmark = HardwareBenchmark()
    assert benchmark.cpu_count > 0
    assert benchmark.memory_total_gb > 0
    assert benchmark.system in ["Linux", "Darwin", "Windows"]


@pytest.mark.unit
def test_hardware_check_requirements():
    """Test hardware requirements check"""
    benchmark = HardwareBenchmark()
    meets_req, reason = benchmark.check_requirements(
        num_agents=3,
        state_size=12,
        action_size=3
    )

    assert isinstance(meets_req, bool)
    assert isinstance(reason, str)
    assert len(reason) > 0


@pytest.mark.slow
def test_hardware_quick_benchmark():
    """Test quick hardware benchmark"""
    benchmark = HardwareBenchmark()
    result = benchmark.run_quick_benchmark()

    assert "iterations_per_second" in result
    assert "estimated_slowdown" in result
    assert "can_handle_multi_gamma" in result
    assert result["iterations_per_second"] > 0


# ============================================================================
# MULTI-GAMMA TRAINER TESTS
# ============================================================================

@pytest.mark.unit
def test_multi_gamma_trainer_init():
    """Test MultiGammaTrainer initialization"""
    # Disable hardware check for testing
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    assert trainer.state_size == 12
    assert trainer.action_size == 3
    assert len(trainer.agents) > 0
    assert trainer.active_gamma is not None


@pytest.mark.unit
def test_multi_gamma_trainer_fallback():
    """Test fallback to single agent when hardware insufficient"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    # Should have fallback agent
    assert len(trainer.agents) == 1 or len(trainer.agents) == 3
    assert trainer.active_gamma in trainer.agents


@pytest.mark.unit
def test_multi_gamma_trainer_custom_gammas():
    """Test MultiGammaTrainer with custom gamma values"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        gammas=[0.85, 0.90, 0.95],
        enable_if_hardware_sufficient=False
    )

    assert trainer.gammas == [0.85, 0.90, 0.95]


@pytest.mark.unit
def test_multi_gamma_get_active_agent():
    """Test getting active agent"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    agent = trainer.get_active_agent()
    assert agent is not None
    assert agent.state_size == 12
    assert agent.action_size == 3


@pytest.mark.unit
def test_multi_gamma_record_trade_duration():
    """Test recording trade durations"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    trainer.record_trade_duration(2.5)
    trainer.record_trade_duration(3.0)

    assert len(trainer.trade_durations) == 2
    assert trainer.trade_durations[0] == 2.5
    assert trainer.trade_durations[1] == 3.0


@pytest.mark.unit
def test_multi_gamma_get_optimal_gamma():
    """Test optimal gamma calculation from trades"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    # Record scalping-style trades (< 1h)
    for _ in range(20):
        trainer.record_trade_duration(np.random.uniform(0.2, 0.8))

    optimal = trainer.get_optimal_gamma_from_trades()
    assert optimal == 0.90  # Should recommend scalping gamma


@pytest.mark.unit
def test_multi_gamma_statistics():
    """Test getting statistics"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    stats = trainer.get_statistics()

    assert "enabled" in stats
    assert "active_gamma" in stats
    assert "gammas" in stats
    assert "hardware_sufficient" in stats


@pytest.mark.integration
def test_multi_gamma_remember_and_replay():
    """Test remembering experiences and training"""
    trainer = MultiGammaTrainer(
        state_size=12,
        action_size=3,
        enable_if_hardware_sufficient=False
    )

    # Create fake experience
    state = np.random.randn(12).astype(np.float32)
    action = 1
    reward = 10.0
    next_state = np.random.randn(12).astype(np.float32)
    done = False

    # Remember experience
    trainer.remember(state, action, reward, next_state, done)

    # All agents should have the experience
    for agent in trainer.agents.values():
        assert len(agent.memory) == 1

    # Replay (train)
    losses = trainer.replay(batch_size=1)
    assert isinstance(losses, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
