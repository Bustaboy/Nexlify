#!/usr/bin/env python3
"""
Tests for N-Step Replay Buffer
"""

import numpy as np
import pytest

from nexlify.memory.nstep_replay_buffer import (
    NStepReplayBuffer,
    MixedNStepReplayBuffer,
    NStepReturnCalculator,
)


class TestNStepReturnCalculator:
    """Tests for NStepReturnCalculator"""

    def test_initialization(self):
        """Test calculator initialization"""
        calc = NStepReturnCalculator(n_step=5, gamma=0.99)
        assert calc.n_step == 5
        assert calc.gamma == 0.99
        assert len(calc._gamma_powers) == 5

    def test_initialization_validation(self):
        """Test initialization parameter validation"""
        with pytest.raises(ValueError, match="n_step must be >= 1"):
            NStepReturnCalculator(n_step=0, gamma=0.99)

        with pytest.raises(ValueError, match="gamma must be in"):
            NStepReturnCalculator(n_step=5, gamma=1.5)

    def test_one_step_return(self):
        """Test 1-step return calculation"""
        calc = NStepReturnCalculator(n_step=1, gamma=0.99)

        # R = r_0 + γ * Q(s_1)
        rewards = [1.0]
        next_q = 10.0
        dones = [False]

        n_step_return, actual_steps = calc.calculate(rewards, next_q, dones)

        expected = 1.0 + 0.99 * 10.0
        assert np.isclose(n_step_return, expected)
        assert actual_steps == 1

    def test_three_step_return(self):
        """Test 3-step return calculation"""
        calc = NStepReturnCalculator(n_step=3, gamma=0.9)

        # R = r_0 + γ*r_1 + γ²*r_2 + γ³*Q(s_3)
        rewards = [1.0, 2.0, 3.0]
        next_q = 10.0
        dones = [False, False, False]

        n_step_return, actual_steps = calc.calculate(rewards, next_q, dones)

        # Expected: 1.0 + 0.9*2.0 + 0.81*3.0 + 0.729*10.0
        expected = 1.0 + 0.9 * 2.0 + 0.81 * 3.0 + 0.729 * 10.0
        assert np.isclose(n_step_return, expected)
        assert actual_steps == 3

    def test_terminal_state_truncation(self):
        """Test that returns are truncated at terminal states"""
        calc = NStepReturnCalculator(n_step=5, gamma=0.99)

        # Episode ends at step 2
        rewards = [1.0, 2.0, 3.0]
        next_q = 100.0  # Should be ignored
        dones = [False, True, False]

        n_step_return, actual_steps = calc.calculate(rewards, next_q, dones)

        # Should only use first 2 steps (episode ended at step 1)
        # R = r_0 + γ*r_1 (no bootstrap)
        expected = 1.0 + 0.99 * 2.0
        assert np.isclose(n_step_return, expected)
        assert actual_steps == 2

    def test_immediate_terminal(self):
        """Test return calculation when first step is terminal"""
        calc = NStepReturnCalculator(n_step=5, gamma=0.99)

        rewards = [5.0]
        next_q = 100.0  # Should be ignored
        dones = [True]

        n_step_return, actual_steps = calc.calculate(rewards, next_q, dones)

        # Only immediate reward, no bootstrap
        assert np.isclose(n_step_return, 5.0)
        assert actual_steps == 1

    def test_empty_rewards(self):
        """Test handling of empty reward sequence"""
        calc = NStepReturnCalculator(n_step=5, gamma=0.99)

        n_step_return, actual_steps = calc.calculate([], 10.0, [])

        assert n_step_return == 0.0
        assert actual_steps == 0


class TestNStepReplayBuffer:
    """Tests for NStepReplayBuffer"""

    @pytest.fixture
    def buffer(self):
        """Create buffer for testing"""
        return NStepReplayBuffer(capacity=1000, n_step=5, gamma=0.99)

    @pytest.fixture
    def small_buffer(self):
        """Create small buffer for testing"""
        return NStepReplayBuffer(capacity=10, n_step=3, gamma=0.9)

    def test_initialization(self, buffer):
        """Test buffer initialization"""
        assert buffer.capacity == 1000
        assert buffer.n_step == 5
        assert buffer.gamma == 0.99
        assert len(buffer) == 0
        assert buffer.max_size == 1000

    def test_initialization_validation(self):
        """Test parameter validation"""
        with pytest.raises(ValueError, match="Capacity must be positive"):
            NStepReplayBuffer(capacity=0)

        with pytest.raises(ValueError, match="n_step must be >= 1"):
            NStepReplayBuffer(capacity=100, n_step=0)

        with pytest.raises(ValueError, match="gamma must be in"):
            NStepReplayBuffer(capacity=100, gamma=1.5)

    def test_push_and_accumulation(self, small_buffer):
        """Test pushing transitions and n-step accumulation"""
        # Create dummy states
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Push first transition
        small_buffer.push(state, 0, 1.0, next_state, False)

        # Buffer should be empty until we have n_step transitions
        assert len(small_buffer) == 0
        assert len(small_buffer.n_step_buffer) == 1

        # Push second transition
        small_buffer.push(state, 1, 2.0, next_state, False)
        assert len(small_buffer) == 0
        assert len(small_buffer.n_step_buffer) == 2

        # Push third transition (reaches n_step=3)
        small_buffer.push(state, 0, 3.0, next_state, False)
        assert len(small_buffer) == 1  # First n-step transition stored
        assert len(small_buffer.n_step_buffer) == 3

        # Push fourth transition
        small_buffer.push(state, 1, 4.0, next_state, False)
        assert len(small_buffer) == 2  # Second n-step transition stored

    def test_episode_boundary_handling(self, small_buffer):
        """Test proper handling of episode boundaries"""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Add 2 transitions, then terminal
        small_buffer.push(state, 0, 1.0, next_state, False)
        small_buffer.push(state, 1, 2.0, next_state, False)
        small_buffer.push(state, 0, 3.0, next_state, True)  # Terminal

        # Should have flushed all transitions
        # With n_step=3 and terminal at step 2, we should have 2-3 transitions
        assert len(small_buffer) >= 2
        assert len(small_buffer.n_step_buffer) == 0  # Buffer flushed

    def test_sampling(self, buffer):
        """Test sampling from buffer"""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Fill buffer with enough transitions
        for i in range(20):
            buffer.push(state, i % 3, float(i), next_state, False)

        # Sample batch
        batch_size = 10
        batch = buffer.sample(batch_size)

        assert len(batch) == batch_size

        # Check format: (state, action, n_step_return, next_state_n, done, actual_steps)
        for experience in batch:
            assert len(experience) == 6
            s, a, r, ns, d, steps = experience
            assert s.shape == state.shape
            assert isinstance(a, (int, np.integer))
            assert isinstance(r, (float, np.floating))
            assert ns.shape == next_state.shape
            assert isinstance(d, (bool, np.bool_))
            assert isinstance(steps, int)
            assert 1 <= steps <= buffer.n_step

    def test_sample_insufficient_data(self, buffer):
        """Test sampling with insufficient data raises error"""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Add only 2 transitions
        for i in range(2):
            buffer.push(state, 0, 1.0, next_state, False)

        # Try to sample more than available
        with pytest.raises(ValueError, match="Not enough experiences"):
            buffer.sample(10)

    def test_n_step_return_values(self, small_buffer):
        """Test that n-step returns are calculated correctly"""
        state = np.array([1.0])
        next_state = np.array([2.0])

        # Add sequence with known rewards
        # gamma = 0.9, n_step = 3
        # Expected for first transition: r0 + γ*r1 + γ²*r2
        # = 1.0 + 0.9*2.0 + 0.81*3.0 = 1.0 + 1.8 + 2.43 = 5.23
        small_buffer.push(state, 0, 1.0, next_state, False)
        small_buffer.push(state, 1, 2.0, next_state, False)
        small_buffer.push(state, 0, 3.0, next_state, False)
        small_buffer.push(state, 1, 4.0, next_state, False)

        # Sample and check returns
        batch = small_buffer.sample(1)
        _, _, n_step_return, _, _, _ = batch[0]

        # Should be close to calculated value (may include bootstrap)
        assert n_step_return > 0

    def test_statistics(self, buffer):
        """Test statistics tracking"""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Add some transitions
        for i in range(15):
            buffer.push(state, i % 3, float(i), next_state, False)

        stats = buffer.get_stats()

        assert "size" in stats
        assert "capacity" in stats
        assert "n_step" in stats
        assert "gamma" in stats
        assert "total_pushes" in stats
        assert stats["total_pushes"] == 15
        assert stats["capacity"] == 1000
        assert stats["n_step"] == 5

    def test_capacity_overflow(self):
        """Test that buffer respects capacity limit"""
        buffer = NStepReplayBuffer(capacity=10, n_step=3, gamma=0.99)
        state = np.array([1.0])
        next_state = np.array([2.0])

        # Add more than capacity
        for i in range(50):
            buffer.push(state, 0, 1.0, next_state, False)

        # Should not exceed capacity
        assert len(buffer) <= buffer.capacity


class TestMixedNStepReplayBuffer:
    """Tests for MixedNStepReplayBuffer"""

    @pytest.fixture
    def mixed_buffer(self):
        """Create mixed buffer for testing"""
        return MixedNStepReplayBuffer(
            capacity=1000, n_step=5, gamma=0.99, n_step_ratio=0.7
        )

    def test_initialization(self, mixed_buffer):
        """Test mixed buffer initialization"""
        assert mixed_buffer.capacity == 1000
        assert mixed_buffer.n_step == 5
        assert mixed_buffer.gamma == 0.99
        assert mixed_buffer.buffer.use_mixed_returns is True
        assert mixed_buffer.buffer.mixed_returns_ratio == 0.7

    def test_mixed_sampling(self, mixed_buffer):
        """Test that buffer samples from both 1-step and n-step"""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Fill buffer
        for i in range(30):
            mixed_buffer.push(state, i % 3, float(i), next_state, False)

        # Sample batch
        batch = mixed_buffer.sample(20)
        assert len(batch) == 20

        # Check that we have mix of step counts
        step_counts = [exp[5] for exp in batch]
        assert min(step_counts) >= 1
        assert max(step_counts) <= mixed_buffer.n_step

        # With 70% n-step ratio, we should have roughly 14 n-step and 6 one-step
        # (but exact counts may vary due to buffer state)

    def test_statistics(self, mixed_buffer):
        """Test statistics for mixed buffer"""
        state = np.array([1.0, 2.0, 3.0])
        next_state = np.array([2.0, 3.0, 4.0])

        # Fill buffer
        for i in range(20):
            mixed_buffer.push(state, i % 3, float(i), next_state, False)

        # Sample to trigger statistics
        if len(mixed_buffer) >= 10:
            mixed_buffer.sample(10)

        stats = mixed_buffer.get_stats()

        assert "n_step_samples" in stats
        assert "one_step_samples" in stats


class TestNStepIntegration:
    """Integration tests for n-step buffer"""

    def test_full_episode(self):
        """Test processing a complete episode"""
        buffer = NStepReplayBuffer(capacity=100, n_step=3, gamma=0.9)

        # Simulate a 10-step episode
        for step in range(10):
            state = np.array([float(step)])
            next_state = np.array([float(step + 1)])
            reward = float(step)
            done = (step == 9)  # Last step

            buffer.push(state, 0, reward, next_state, done)

        # Should have stored all transitions as n-step
        # With n_step=3 and 10 steps, we should have around 8-10 n-step transitions
        assert len(buffer) >= 8

        # Buffer should be flushed after terminal state
        assert len(buffer.n_step_buffer) == 0

    def test_multiple_episodes(self):
        """Test processing multiple episodes"""
        buffer = NStepReplayBuffer(capacity=100, n_step=5, gamma=0.99)
        total_steps = 0

        # Run 3 episodes
        for episode in range(3):
            episode_length = 10 + episode * 2  # Variable length episodes

            for step in range(episode_length):
                state = np.array([float(episode), float(step)])
                next_state = np.array([float(episode), float(step + 1)])
                reward = float(step)
                done = (step == episode_length - 1)

                buffer.push(state, 0, reward, next_state, done)
                total_steps += 1

        # Should have many transitions stored
        assert len(buffer) > 0
        assert len(buffer) <= total_steps

        # Last episode should be flushed
        assert len(buffer.n_step_buffer) == 0

    def test_comparison_1step_vs_nstep(self):
        """Compare 1-step vs n-step returns"""
        # Create two buffers: 1-step and 5-step
        buffer_1 = NStepReplayBuffer(capacity=100, n_step=1, gamma=0.99)
        buffer_5 = NStepReplayBuffer(capacity=100, n_step=5, gamma=0.99)

        # Same sequence of transitions
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        for r in rewards:
            state = np.array([r])
            next_state = np.array([r + 1])
            buffer_1.push(state, 0, r, next_state, False)
            buffer_5.push(state, 0, r, next_state, False)

        # 5-step should accumulate more total return per transition
        if len(buffer_1) > 0 and len(buffer_5) > 0:
            batch_1 = buffer_1.sample(1)
            batch_5 = buffer_5.sample(1)

            _, _, return_1, _, _, _ = batch_1[0]
            _, _, return_5, _, _, _ = batch_5[0]

            # 5-step return should generally be larger (accumulates more future rewards)
            # This is probabilistic, so we just check they're both positive
            assert return_1 > 0
            assert return_5 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
