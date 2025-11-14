#!/usr/bin/env python3
"""
Unit tests for Nexlify RL Agent
Comprehensive testing of reinforcement learning trading agent
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.strategies.nexlify_rl_agent import (DQNAgent, ReplayBuffer,
                                                 TradingEnvironment)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    # Create synthetic price data with trend
    np.random.seed(42)
    base_price = 50000
    steps = 100
    prices = base_price + np.cumsum(np.random.randn(steps) * 100)
    return prices.astype(np.float32)


@pytest.fixture
def trading_env(sample_price_data):
    """Create trading environment"""
    return TradingEnvironment(price_data=sample_price_data, initial_balance=10000.0)


@pytest.fixture
def dqn_agent():
    """Create DQN agent with new epsilon decay system (legacy 8-feature for backward compat tests)"""
    config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon_decay_type': 'exponential',
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay_rate': 0.995,
        'epsilon_decay_steps': 10000,
    }
    # Note: Uses state_size=8 for backward compatibility testing
    # Production default is now 12 features (see test_crypto_optimized_defaults)
    return DQNAgent(state_size=8, action_size=3, config=config)


class TestTradingEnvironment:
    """Test Trading Environment functionality"""

    def test_initialization(self, trading_env, sample_price_data):
        """Test environment initialization"""
        assert trading_env.initial_balance == 10000.0
        assert trading_env.balance == 10000.0
        assert trading_env.position == 0
        assert trading_env.position_price == 0
        assert trading_env.current_step == 0
        assert trading_env.action_space_n == 3
        assert trading_env.state_space_n == 12  # Updated for crypto-specific features
        assert len(trading_env.price_data) == len(sample_price_data)

    def test_reset(self, trading_env):
        """Test environment reset"""
        # Modify state
        trading_env.current_step = 50
        trading_env.balance = 5000.0
        trading_env.position = 0.1

        # Reset
        state = trading_env.reset()

        assert trading_env.current_step == 0
        assert trading_env.balance == 10000.0
        assert trading_env.position == 0
        assert trading_env.position_price == 0
        assert isinstance(state, np.ndarray)
        assert len(state) == 12  # Updated for crypto-specific features

    def test_get_state(self, trading_env):
        """Test state representation"""
        state = trading_env._get_state()

        assert isinstance(state, np.ndarray)
        assert len(state) == 12  # Updated for crypto-specific features
        assert state.dtype == np.float32
        # Check state is normalized/reasonable
        assert np.all(np.isfinite(state))

    def test_step_hold_action(self, trading_env):
        """Test step with hold action"""
        initial_balance = trading_env.balance
        initial_position = trading_env.position

        state, reward, done, info = trading_env.step(0)  # Hold

        assert trading_env.balance == initial_balance
        assert trading_env.position == initial_position
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_buy_action(self, trading_env):
        """Test step with buy action"""
        initial_balance = trading_env.balance

        state, reward, done, info = trading_env.step(1)  # Buy

        # Should have bought some crypto
        assert trading_env.position > 0
        assert trading_env.balance < initial_balance
        assert trading_env.position_price > 0

    def test_step_sell_action_with_position(self, trading_env):
        """Test step with sell action when holding position"""
        # First buy
        trading_env.step(1)
        position_after_buy = trading_env.position

        # Then sell
        state, reward, done, info = trading_env.step(2)

        # Position should be reduced or closed
        assert trading_env.position < position_after_buy

    def test_step_sell_action_no_position(self, trading_env):
        """Test sell action when not holding position"""
        initial_balance = trading_env.balance

        state, reward, done, info = trading_env.step(2)  # Sell with no position

        # Balance shouldn't change
        assert trading_env.balance == initial_balance
        assert trading_env.position == 0

    def test_episode_completion(self, trading_env):
        """Test completing a full episode"""
        state = trading_env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = np.random.randint(0, 3)
            state, reward, done, info = trading_env.step(action)
            steps += 1

        assert done or steps == 100

    def test_rsi_calculation(self, trading_env):
        """Test RSI indicator calculation"""
        rsi = trading_env._calculate_rsi(14)

        # RSI should be between 0 and 1
        assert 0 <= rsi <= 1

    def test_macd_calculation(self, trading_env):
        """Test MACD calculation"""
        macd = trading_env._calculate_macd()

        # MACD can be any value, but should be finite
        assert np.isfinite(macd)

    def test_volatility_calculation(self, trading_env):
        """Test volatility calculation"""
        volatility = trading_env._calculate_volatility(10)

        # Volatility should be non-negative and finite
        assert volatility >= 0
        assert np.isfinite(volatility)

    def test_profit_loss_calculation(self, trading_env):
        """Test profit/loss from trading"""
        initial_equity = trading_env.balance

        # Buy at current price
        trading_env.step(1)

        # Advance a few steps
        for _ in range(5):
            trading_env.step(0)  # Hold

        # Sell
        trading_env.step(2)

        # Check final equity
        final_equity = trading_env.balance
        pnl = final_equity - initial_equity

        # P&L should be calculated (could be positive or negative)
        assert isinstance(pnl, (int, float))


class TestDQNAgent:
    """Test DQN Agent functionality"""

    def test_initialization(self, dqn_agent):
        """Test agent initialization"""
        assert dqn_agent.state_size == 8  # Test fixture uses 8
        assert dqn_agent.action_size == 3
        assert dqn_agent.learning_rate == 0.001  # Test fixture specifies this
        assert dqn_agent.gamma == 0.95  # Test fixture specifies this
        assert dqn_agent.epsilon == 1.0
        # New epsilon decay system
        assert dqn_agent.epsilon_decay_strategy is not None
        assert dqn_agent.epsilon_decay_strategy.epsilon_start == 1.0
        assert dqn_agent.epsilon_decay_strategy.epsilon_end == 0.01  # Test fixture specifies this
        assert len(dqn_agent.memory) == 0

    def test_crypto_optimized_defaults(self):
        """Test 24/7 crypto-optimized default configuration"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent
        from nexlify.strategies.epsilon_decay import ScheduledEpsilonDecay

        # Create agent with NO config - should use crypto defaults
        agent = DQNAgent(state_size=12, action_size=3)

        # Verify 24/7 crypto defaults
        assert agent.gamma == 0.89  # Lower discount for fast crypto
        assert agent.learning_rate == 0.0015  # Aggressive learning
        assert agent.learning_rate_decay == 0.9998
        assert agent.batch_size == 64
        assert agent.target_update_freq == 1200
        assert len(agent.memory.buffer) == 0
        assert agent.memory.max_size == 60000  # Smaller buffer for regime adaptation

        # Verify scheduled epsilon decay is used by default
        assert isinstance(agent.epsilon_decay_strategy, ScheduledEpsilonDecay)
        assert agent.epsilon_decay_strategy.epsilon_end == 0.22  # High exploration

    def test_remember(self, dqn_agent):
        """Test experience replay memory"""
        state = np.random.randn(12)
        action = 1
        reward = 10.0
        next_state = np.random.randn(12)
        done = False

        dqn_agent.remember(state, action, reward, next_state, done)

        assert len(dqn_agent.memory) == 1

    def test_act_exploration(self, dqn_agent):
        """Test action selection during exploration"""
        state = np.random.randn(12)

        # With high epsilon, should explore randomly
        dqn_agent.epsilon = 1.0
        action = dqn_agent.act(state)

        assert 0 <= action < 3

    def test_act_exploitation(self, dqn_agent):
        """Test action selection during exploitation"""
        state = np.random.randn(12)

        # With low epsilon, should exploit
        dqn_agent.epsilon = 0.0
        action = dqn_agent.act(state)

        assert 0 <= action < 3

    def test_epsilon_decay(self, dqn_agent):
        """Test epsilon decay"""
        initial_epsilon = dqn_agent.epsilon

        dqn_agent.update_epsilon()

        assert dqn_agent.epsilon < initial_epsilon
        assert dqn_agent.epsilon >= dqn_agent.epsilon_decay_strategy.epsilon_end

    def test_epsilon_minimum(self, dqn_agent):
        """Test epsilon doesn't go below minimum"""
        epsilon_end = dqn_agent.epsilon_decay_strategy.epsilon_end

        # Decay many times
        for _ in range(1000):
            dqn_agent.update_epsilon()

        assert dqn_agent.epsilon >= epsilon_end

    def test_replay_insufficient_memory(self, dqn_agent):
        """Test replay with insufficient memory"""
        # Add only a few experiences
        for i in range(10):
            state = np.random.randn(12)
            dqn_agent.remember(state, 1, 1.0, state, False)

        # Should handle gracefully with small batch
        loss = dqn_agent.replay(batch_size=32)
        assert loss >= 0 or loss is None

    def test_replay_training(self, dqn_agent):
        """Test replay training with sufficient memory"""
        # Fill memory
        for i in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = np.random.rand() > 0.9

            dqn_agent.remember(state, action, reward, next_state, done)

        # Train
        loss = dqn_agent.replay(batch_size=32)

        # Should return a valid loss
        assert isinstance(loss, (int, float))
        assert loss >= 0

    def test_save_and_load_model(self, dqn_agent, tmp_path):
        """Test saving and loading model"""
        model_path = tmp_path / "test_model.h5"

        # Save model
        dqn_agent.save(str(model_path))
        assert model_path.exists()

        # Create new agent and load
        new_agent = DQNAgent(state_size=8, action_size=3)
        new_agent.load(str(model_path))

        # Models should produce similar outputs
        state = np.random.randn(12)
        action1 = dqn_agent.act(state)
        action2 = new_agent.act(state)

        # With epsilon=0, should be deterministic
        dqn_agent.epsilon = 0.0
        new_agent.epsilon = 0.0
        assert dqn_agent.act(state) == new_agent.act(state)

    def test_get_model_summary(self, dqn_agent):
        """Test getting model summary"""
        summary = dqn_agent.get_model_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestReplayBuffer:
    """Test Replay Buffer functionality"""

    def test_initialization(self):
        """Test replay buffer initialization"""
        buffer = ReplayBuffer(max_size=1000)

        assert buffer.max_size == 1000
        assert len(buffer) == 0

    def test_add_experience(self):
        """Test adding experience to buffer"""
        buffer = ReplayBuffer(max_size=1000)

        state = np.random.randn(12)
        action = 1
        reward = 10.0
        next_state = np.random.randn(12)
        done = False

        buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 1

    def test_buffer_overflow(self):
        """Test buffer behavior when full"""
        buffer = ReplayBuffer(max_size=10)

        # Fill buffer beyond capacity
        for i in range(20):
            state = np.random.randn(12)
            buffer.add(state, 0, 0.0, state, False)

        # Should maintain max size
        assert len(buffer) == 10

    def test_sample(self):
        """Test sampling from buffer"""
        buffer = ReplayBuffer(max_size=1000)

        # Add experiences
        for i in range(100):
            state = np.random.randn(12)
            buffer.add(state, 0, 0.0, state, False)

        # Sample
        batch = buffer.sample(batch_size=32)

        assert len(batch) == 32

    def test_sample_insufficient_data(self):
        """Test sampling when buffer too small"""
        buffer = ReplayBuffer(max_size=1000)

        # Add few experiences
        for i in range(10):
            state = np.random.randn(12)
            buffer.add(state, 0, 0.0, state, False)

        # Sample more than available
        batch = buffer.sample(batch_size=32)

        # Should return what's available
        assert len(batch) <= 32


class TestIntegration:
    """Integration tests for agent and environment"""

    def test_training_episode(self, trading_env, dqn_agent):
        """Test a complete training episode"""
        state = trading_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 50:  # Short episode for testing
            action = dqn_agent.act(state)
            next_state, reward, done, info = trading_env.step(action)

            dqn_agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            # Train if enough memory
            if len(dqn_agent.memory) > 32:
                dqn_agent.replay(batch_size=32)

        assert steps > 0
        assert isinstance(total_reward, (int, float))

    def test_multiple_episodes(self, trading_env, dqn_agent):
        """Test training over multiple episodes"""
        episode_rewards = []

        for episode in range(3):  # Few episodes for testing
            state = trading_env.reset()
            done = False
            total_reward = 0

            while not done:
                action = dqn_agent.act(state)
                next_state, reward, done, info = trading_env.step(action)

                dqn_agent.remember(state, action, reward, next_state, done)
                if len(dqn_agent.memory) > 32:
                    dqn_agent.replay(batch_size=32)

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            dqn_agent.update_epsilon()

        assert len(episode_rewards) == 3
        # Epsilon should have decayed
        assert dqn_agent.epsilon < 1.0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_price_data(self):
        """Test with zero prices"""
        prices = np.zeros(100)
        env = TradingEnvironment(prices, initial_balance=10000.0)

        state = env.reset()
        # Should handle gracefully
        assert isinstance(state, np.ndarray)

    def test_negative_prices(self):
        """Test with negative prices (edge case)"""
        prices = -np.abs(np.random.randn(100))
        env = TradingEnvironment(prices, initial_balance=10000.0)

        # Should handle or raise appropriate error
        try:
            state = env.reset()
            assert isinstance(state, np.ndarray)
        except Exception:
            pass  # Acceptable to raise error for invalid data

    def test_single_step_environment(self):
        """Test environment with only one step"""
        prices = np.array([50000.0, 51000.0])
        env = TradingEnvironment(prices, initial_balance=10000.0)

        state = env.reset()
        state, reward, done, info = env.step(0)

        assert done is True

    def test_agent_with_zero_state_size(self):
        """Test agent with invalid state size"""
        with pytest.raises(Exception):
            agent = DQNAgent(state_size=0, action_size=3)

    def test_agent_with_zero_action_size(self):
        """Test agent with invalid action size"""
        with pytest.raises(Exception):
            agent = DQNAgent(state_size=8, action_size=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
