#!/usr/bin/env python3
"""
Tests for Advanced Epsilon Decay Manager
"""

import pytest
import numpy as np
from pathlib import Path
import json
import tempfile

from nexlify.strategies.epsilon_decay import (
    LinearEpsilonDecay,
    ScheduledEpsilonDecay,
    ExponentialEpsilonDecay,
    EpsilonDecayFactory
)


class TestLinearEpsilonDecay:
    """Test LinearEpsilonDecay strategy"""

    def test_initialization(self):
        """Test strategy initialization"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        assert strategy.epsilon_start == 1.0
        assert strategy.epsilon_end == 0.05
        assert strategy.current_epsilon == 1.0
        assert strategy.current_step == 0

    def test_linear_decay(self):
        """Test linear decay formula"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        # Test expected values
        assert abs(strategy.get_epsilon(0) - 1.0) < 0.01
        assert abs(strategy.get_epsilon(100) - 0.9525) < 0.01
        assert abs(strategy.get_epsilon(500) - 0.7625) < 0.01
        assert abs(strategy.get_epsilon(1000) - 0.525) < 0.01
        assert abs(strategy.get_epsilon(2000) - 0.05) < 0.01

    def test_epsilon_never_below_end(self):
        """Test epsilon never goes below epsilon_end"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        # Test beyond decay_steps
        epsilon = strategy.get_epsilon(3000)
        assert epsilon >= 0.05
        assert epsilon == 0.05

    def test_step_method(self):
        """Test step() method updates correctly"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        epsilon1 = strategy.step()
        assert strategy.current_step == 1
        assert len(strategy.epsilon_history) == 1

        epsilon2 = strategy.step()
        assert strategy.current_step == 2
        assert len(strategy.epsilon_history) == 2
        assert epsilon1 > epsilon2

    def test_threshold_logging(self):
        """Test threshold crossing detection"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=1000
        )

        # Run through decay
        for _ in range(1000):
            strategy.step()

        # All thresholds should be crossed
        expected_thresholds = {0.9, 0.7, 0.5, 0.3, 0.1}
        assert strategy.thresholds_crossed == expected_thresholds

    def test_reset(self):
        """Test reset functionality"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        # Run a few steps
        for _ in range(100):
            strategy.step()

        # Reset
        strategy.reset()

        assert strategy.current_epsilon == 1.0
        assert strategy.current_step == 0
        assert len(strategy.epsilon_history) == 0
        assert len(strategy.thresholds_crossed) == 0


class TestScheduledEpsilonDecay:
    """Test ScheduledEpsilonDecay strategy"""

    def test_default_schedule(self):
        """Test default schedule"""
        strategy = ScheduledEpsilonDecay()

        assert strategy.get_epsilon(0) == 1.0
        assert abs(strategy.get_epsilon(300) - 0.7) < 0.01
        assert abs(strategy.get_epsilon(1000) - 0.3) < 0.01
        assert abs(strategy.get_epsilon(2000) - 0.05) < 0.01

    def test_custom_schedule(self):
        """Test custom schedule"""
        schedule = {
            0: 1.0,
            500: 0.5,
            1000: 0.1
        }

        strategy = ScheduledEpsilonDecay(schedule=schedule)

        assert strategy.get_epsilon(0) == 1.0
        assert abs(strategy.get_epsilon(250) - 0.75) < 0.01  # Midpoint
        assert abs(strategy.get_epsilon(500) - 0.5) < 0.01
        assert abs(strategy.get_epsilon(750) - 0.3) < 0.01  # Midpoint
        assert abs(strategy.get_epsilon(1000) - 0.1) < 0.01

    def test_interpolation(self):
        """Test linear interpolation between schedule points"""
        schedule = {0: 1.0, 1000: 0.5}
        strategy = ScheduledEpsilonDecay(schedule=schedule)

        # Test interpolation at 25%, 50%, 75%
        assert abs(strategy.get_epsilon(250) - 0.875) < 0.01
        assert abs(strategy.get_epsilon(500) - 0.75) < 0.01
        assert abs(strategy.get_epsilon(750) - 0.625) < 0.01

    def test_beyond_schedule(self):
        """Test epsilon after last scheduled step"""
        schedule = {0: 1.0, 1000: 0.1}
        strategy = ScheduledEpsilonDecay(schedule=schedule, epsilon_end=0.05)

        # Beyond schedule should return last value, respecting epsilon_end
        epsilon = strategy.get_epsilon(2000)
        assert epsilon == 0.1


class TestExponentialEpsilonDecay:
    """Test ExponentialEpsilonDecay strategy"""

    def test_auto_decay_rate(self):
        """Test automatic decay rate calculation"""
        strategy = ExponentialEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        # Should reach epsilon_end at decay_steps
        epsilon = strategy.get_epsilon(2000)
        assert abs(epsilon - 0.05) < 0.01

    def test_exponential_curve(self):
        """Test exponential decay curve"""
        strategy = ExponentialEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        # Exponential decay should be steeper early on
        eps_100 = strategy.get_epsilon(100)
        eps_500 = strategy.get_epsilon(500)
        eps_1000 = strategy.get_epsilon(1000)

        # Verify decay is happening
        assert eps_100 > eps_500 > eps_1000
        assert eps_1000 > 0.05

    def test_custom_decay_rate(self):
        """Test custom decay rate"""
        strategy = ExponentialEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.01,
            decay_rate=0.99
        )

        eps_0 = strategy.get_epsilon(0)
        eps_100 = strategy.get_epsilon(100)

        assert eps_0 == 1.0
        assert abs(eps_100 - 1.0 * (0.99 ** 100)) < 0.01

    def test_epsilon_floor(self):
        """Test epsilon never goes below epsilon_end"""
        strategy = ExponentialEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=100
        )

        # Run far beyond decay steps
        epsilon = strategy.get_epsilon(10000)
        assert epsilon >= 0.05
        assert epsilon == 0.05


class TestEpsilonDecayFactory:
    """Test EpsilonDecayFactory"""

    def test_create_linear(self):
        """Test creating linear strategy"""
        strategy = EpsilonDecayFactory.create(
            'linear',
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        assert isinstance(strategy, LinearEpsilonDecay)
        assert strategy.epsilon_start == 1.0
        assert strategy.epsilon_end == 0.05

    def test_create_scheduled(self):
        """Test creating scheduled strategy"""
        strategy = EpsilonDecayFactory.create('scheduled')

        assert isinstance(strategy, ScheduledEpsilonDecay)

    def test_create_exponential(self):
        """Test creating exponential strategy"""
        strategy = EpsilonDecayFactory.create(
            'exponential',
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        assert isinstance(strategy, ExponentialEpsilonDecay)

    def test_invalid_strategy_type(self):
        """Test invalid strategy type raises error"""
        with pytest.raises(ValueError) as exc_info:
            EpsilonDecayFactory.create('invalid_type')

        assert 'Unknown strategy type' in str(exc_info.value)

    def test_create_from_config_linear(self):
        """Test creating from config dictionary"""
        config = {
            'epsilon_decay_type': 'linear',
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 2000
        }

        strategy = EpsilonDecayFactory.create_from_config(config)

        assert isinstance(strategy, LinearEpsilonDecay)
        assert strategy.epsilon_start == 1.0
        assert strategy.epsilon_end == 0.05
        assert strategy.decay_steps == 2000

    def test_create_from_config_scheduled(self):
        """Test creating scheduled from config"""
        config = {
            'epsilon_decay_type': 'scheduled',
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_schedule': {0: 1.0, 500: 0.5, 1000: 0.05}
        }

        strategy = EpsilonDecayFactory.create_from_config(config)

        assert isinstance(strategy, ScheduledEpsilonDecay)

    def test_create_from_config_defaults(self):
        """Test creating from config with defaults"""
        config = {}

        strategy = EpsilonDecayFactory.create_from_config(config)

        # Should default to linear
        assert isinstance(strategy, LinearEpsilonDecay)


class TestEpsilonHistorySaving:
    """Test epsilon history saving functionality"""

    def test_save_history(self):
        """Test saving epsilon history to JSON"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=100
        )

        # Run some steps
        for _ in range(50):
            strategy.step()

        # Save history
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            strategy.save_history(filepath)

            # Verify file exists and has correct content
            assert Path(filepath).exists()

            with open(filepath, 'r') as f:
                data = json.load(f)

            assert data['strategy'] == 'LinearEpsilonDecay'
            assert len(data['history']) == 50
            assert data['config']['epsilon_start'] == 1.0
            assert data['config']['epsilon_end'] == 0.05

        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)


class TestExpectedOutputs:
    """Test expected outputs match requirements"""

    def test_episode_100_epsilon(self):
        """Test epsilon at episode 100"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        epsilon = strategy.get_epsilon(100)
        # Expected: ≈ 0.95
        assert abs(epsilon - 0.9525) < 0.01

    def test_episode_500_epsilon(self):
        """Test epsilon at episode 500"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        epsilon = strategy.get_epsilon(500)
        # Expected: ≈ 0.75
        assert abs(epsilon - 0.7625) < 0.01

    def test_episode_1000_epsilon(self):
        """Test epsilon at episode 1000"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        epsilon = strategy.get_epsilon(1000)
        # Expected: ≈ 0.50
        assert abs(epsilon - 0.525) < 0.01

    def test_episode_2000_epsilon(self):
        """Test epsilon at episode 2000"""
        strategy = LinearEpsilonDecay(
            epsilon_start=1.0,
            epsilon_end=0.05,
            decay_steps=2000
        )

        epsilon = strategy.get_epsilon(2000)
        # Expected: ≈ 0.05
        assert abs(epsilon - 0.05) < 0.01


class TestDQNAgentIntegration:
    """Test integration with DQNAgent"""

    def test_agent_uses_epsilon_decay_strategy(self):
        """Test DQNAgent integrates with epsilon decay strategy"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {
            'epsilon_decay_type': 'linear',
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 2000
        }

        agent = DQNAgent(state_size=8, action_size=3, config=config)

        assert hasattr(agent, 'epsilon_decay_strategy')
        assert isinstance(agent.epsilon_decay_strategy, LinearEpsilonDecay)
        assert agent.epsilon == 1.0

    def test_agent_decay_epsilon(self):
        """Test agent's decay_epsilon method"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {
            'epsilon_decay_type': 'linear',
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay_steps': 100
        }

        agent = DQNAgent(state_size=8, action_size=3, config=config)

        initial_epsilon = agent.epsilon

        # Decay a few times
        for _ in range(10):
            agent.decay_epsilon()

        assert agent.epsilon < initial_epsilon

    def test_legacy_config_support(self):
        """Test legacy config is converted to new system"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        # Old style config with epsilon_decay (multiplicative)
        config = {
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995
        }

        agent = DQNAgent(state_size=8, action_size=3, config=config)

        # Should create exponential strategy when epsilon_decay is provided
        assert hasattr(agent, 'epsilon_decay_strategy')
        assert isinstance(agent.epsilon_decay_strategy, ExponentialEpsilonDecay)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
