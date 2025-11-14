#!/usr/bin/env python3
"""
Tests for Prioritized Experience Replay (PER)
Tests SumTree, PrioritizedReplayBuffer, and integration with DQNAgent
"""

import numpy as np
import pytest

from nexlify.memory.sumtree import SumTree
from nexlify.memory.prioritized_replay_buffer import PrioritizedReplayBuffer


class TestSumTree:
    """Test cases for SumTree data structure"""

    def test_initialization(self):
        """Test SumTree initialization"""
        tree = SumTree(capacity=100)
        assert tree.capacity == 100
        assert len(tree) == 0
        assert tree.total() == 0

    def test_initialization_invalid_capacity(self):
        """Test initialization with invalid capacity"""
        with pytest.raises(ValueError):
            SumTree(capacity=0)
        with pytest.raises(ValueError):
            SumTree(capacity=-10)

    def test_add_single_item(self):
        """Test adding single item"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")

        assert len(tree) == 1
        assert tree.total() == 1.0
        assert tree.get_max_priority() == 1.0

    def test_add_multiple_items(self):
        """Test adding multiple items"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")
        tree.add(2.0, "data2")
        tree.add(3.0, "data3")

        assert len(tree) == 3
        assert tree.total() == 6.0
        assert tree.get_max_priority() == 3.0

    def test_add_negative_priority(self):
        """Test adding item with negative priority"""
        tree = SumTree(capacity=10)
        with pytest.raises(ValueError):
            tree.add(-1.0, "data")

    def test_circular_buffer(self):
        """Test circular buffer behavior when capacity exceeded"""
        tree = SumTree(capacity=3)
        tree.add(1.0, "data1")
        tree.add(2.0, "data2")
        tree.add(3.0, "data3")
        tree.add(4.0, "data4")  # Should overwrite data1

        assert len(tree) == 3  # Max capacity
        assert tree.total() == 9.0  # 2 + 3 + 4

    def test_update_priority(self):
        """Test updating priority"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")

        # Get the tree index (first leaf)
        idx = tree.capacity - 1

        # Update priority
        tree.update(idx, 5.0)

        assert tree.total() == 5.0
        assert tree.get_max_priority() == 5.0

    def test_update_negative_priority(self):
        """Test updating with negative priority"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")
        idx = tree.capacity - 1

        with pytest.raises(ValueError):
            tree.update(idx, -1.0)

    def test_sample_single_item(self):
        """Test sampling single item"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")

        idx, priority, data = tree.sample(0.5)
        assert data == "data1"
        assert priority == 1.0

    def test_sample_multiple_items(self):
        """Test sampling from multiple items"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")
        tree.add(2.0, "data2")
        tree.add(3.0, "data3")

        # Sample from first item
        idx1, priority1, data1 = tree.sample(0.5)
        assert data1 == "data1"

        # Sample from second item
        idx2, priority2, data2 = tree.sample(1.5)
        assert data2 == "data2"

        # Sample from third item
        idx3, priority3, data3 = tree.sample(4.0)
        assert data3 == "data3"

    def test_sample_out_of_range(self):
        """Test sampling with out of range value"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")

        with pytest.raises(ValueError):
            tree.sample(2.0)  # Total is only 1.0

        with pytest.raises(ValueError):
            tree.sample(-1.0)

    def test_get_leaf(self):
        """Test getting leaf data"""
        tree = SumTree(capacity=10)
        tree.add(1.5, "data1")

        idx = tree.capacity - 1
        priority, data = tree.get_leaf(idx)

        assert priority == 1.5
        assert data == "data1"

    def test_get_leaf_invalid_index(self):
        """Test getting leaf with invalid index"""
        tree = SumTree(capacity=10)
        tree.add(1.0, "data1")

        # Test out of range
        with pytest.raises(IndexError):
            tree.get_leaf(100)

        # Test internal node (not leaf)
        with pytest.raises(IndexError):
            tree.get_leaf(0)  # Root node


class TestPrioritizedReplayBuffer:
    """Test cases for PrioritizedReplayBuffer"""

    def test_initialization(self):
        """Test buffer initialization"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert buffer.alpha == 0.6
        assert buffer.beta_start == 0.4

    def test_initialization_invalid_params(self):
        """Test initialization with invalid parameters"""
        with pytest.raises(ValueError):
            PrioritizedReplayBuffer(capacity=0)

        with pytest.raises(ValueError):
            PrioritizedReplayBuffer(capacity=100, alpha=1.5)

        with pytest.raises(ValueError):
            PrioritizedReplayBuffer(capacity=100, beta_start=-0.1)

    def test_push_experience(self):
        """Test pushing experience to buffer"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        state = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_state = np.array([4, 5, 6])
        done = False

        buffer.push(state, action, reward, next_state, done)

        assert len(buffer) == 1

    def test_push_with_td_error(self):
        """Test pushing experience with TD error"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        state = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_state = np.array([4, 5, 6])
        done = False
        td_error = 2.5

        buffer.push(state, action, reward, next_state, done, td_error=td_error)

        assert len(buffer) == 1

    def test_sample_batch(self):
        """Test sampling batch of experiences"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add some experiences
        for i in range(10):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = float(i)
            next_state = np.array([i + 1, i + 2, i + 3])
            done = False
            buffer.push(state, action, reward, next_state, done)

        # Sample batch
        batch_size = 5
        experiences, indices, weights = buffer.sample(batch_size)

        assert len(experiences) == batch_size
        assert len(indices) == batch_size
        assert len(weights) == batch_size
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0)

    def test_sample_insufficient_data(self):
        """Test sampling when buffer has insufficient data"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add only 3 experiences
        for i in range(3):
            state = np.array([i])
            buffer.push(state, 0, 0.0, state, False)

        # Try to sample batch of 5
        with pytest.raises(ValueError):
            buffer.sample(5)

    def test_update_priorities(self):
        """Test updating priorities"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # Add experiences
        for i in range(5):
            state = np.array([i])
            buffer.push(state, 0, 0.0, state, False)

        # Sample and update
        experiences, indices, weights = buffer.sample(3)
        td_errors = np.array([1.0, 2.0, 0.5])

        buffer.update_priorities(indices, td_errors)

        # Check that priorities were updated
        stats = buffer.get_stats()
        assert stats["priority_updates"] == 3

    def test_update_priorities_mismatch(self):
        """Test updating priorities with mismatched arrays"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for i in range(5):
            state = np.array([i])
            buffer.push(state, 0, 0.0, state, False)

        experiences, indices, weights = buffer.sample(3)
        td_errors = np.array([1.0, 2.0])  # Wrong size

        with pytest.raises(ValueError):
            buffer.update_priorities(indices, td_errors)

    def test_beta_annealing(self):
        """Test beta annealing over time"""
        buffer = PrioritizedReplayBuffer(
            capacity=100, beta_start=0.4, beta_end=1.0, beta_annealing_steps=100
        )

        # Add experiences
        for i in range(10):
            state = np.array([i])
            buffer.push(state, 0, 0.0, state, False)

        # Sample multiple times and track beta
        betas = []
        for _ in range(150):  # More than annealing steps
            experiences, indices, weights = buffer.sample(5)
            stats = buffer.get_stats()
            betas.append(stats["beta"])

        # Beta should increase from 0.4 to 1.0
        assert betas[0] == pytest.approx(0.4, abs=0.1)
        assert betas[-1] == pytest.approx(1.0, abs=0.01)
        assert np.all(np.diff(betas) >= -1e-6)  # Should be non-decreasing

    def test_priority_clipping(self):
        """Test priority clipping"""
        buffer = PrioritizedReplayBuffer(capacity=100, priority_clip=10.0)

        state = np.array([1])

        # Add experience with very high TD error
        buffer.push(state, 0, 0.0, state, False, td_error=1000.0)

        stats = buffer.get_stats()
        assert stats["max_priority"] <= 10.0

    def test_get_stats(self):
        """Test getting buffer statistics"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for i in range(5):
            state = np.array([i])
            buffer.push(state, 0, 0.0, state, False)

        stats = buffer.get_stats()

        assert "size" in stats
        assert "capacity" in stats
        assert "beta" in stats
        assert "alpha" in stats
        assert "mean_priority" in stats
        assert stats["size"] == 5
        assert stats["capacity"] == 100

    def test_backward_compatibility(self):
        """Test backward compatibility with ReplayBuffer API"""
        buffer = PrioritizedReplayBuffer(capacity=100)

        state = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_state = np.array([4, 5, 6])
        done = False

        # Test add() alias
        buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == 1
        assert buffer.max_size == 100  # Backward compatibility attribute


class TestPERIntegration:
    """Test PER integration with DQNAgent"""

    def test_agent_with_per_enabled(self):
        """Test DQNAgent with PER enabled"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {
            "use_prioritized_replay": True,
            "per_alpha": 0.6,
            "per_beta_start": 0.4,
            "per_beta_end": 1.0,
            "replay_buffer_size": 1000,
        }

        agent = DQNAgent(state_size=12, action_size=3, config=config)

        # Should be using PER
        assert agent.use_per is True
        assert hasattr(agent.memory, "tree")  # PrioritizedReplayBuffer has tree

    def test_agent_with_per_disabled(self):
        """Test DQNAgent with PER disabled"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {
            "use_prioritized_replay": False,
            "replay_buffer_size": 1000,
        }

        agent = DQNAgent(state_size=12, action_size=3, config=config)

        # Should be using standard replay buffer
        assert agent.use_per is False
        assert not hasattr(agent.memory, "tree")

    def test_agent_training_with_per(self):
        """Test agent training loop with PER"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {
            "use_prioritized_replay": True,
            "replay_buffer_size": 1000,
            "batch_size": 32,
        }

        agent = DQNAgent(state_size=12, action_size=3, config=config)

        # Add experiences
        for i in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = bool(np.random.randint(0, 2))

            agent.remember(state, action, reward, next_state, done)

        # Train (replay)
        loss = agent.replay(batch_size=32)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_get_per_stats(self):
        """Test getting PER stats from agent"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {"use_prioritized_replay": True}

        agent = DQNAgent(state_size=12, action_size=3, config=config)

        # Add some experiences
        for i in range(10):
            state = np.random.randn(12)
            agent.remember(state, 0, 0.0, state, False)

        stats = agent.get_per_stats()

        assert stats is not None
        assert "size" in stats
        assert "beta" in stats
        assert stats["size"] == 10

    def test_get_per_stats_disabled(self):
        """Test getting PER stats when PER is disabled"""
        from nexlify.strategies.nexlify_rl_agent import DQNAgent

        config = {"use_prioritized_replay": False}

        agent = DQNAgent(state_size=12, action_size=3, config=config)

        stats = agent.get_per_stats()

        assert stats is None


class TestPERVisualization:
    """Test PER visualization tools"""

    def test_stats_tracker_initialization(self):
        """Test PERStatsTracker initialization"""
        from nexlify.memory.per_visualization import PERStatsTracker

        tracker = PERStatsTracker()

        assert len(tracker.history["beta"]) == 0

    def test_stats_tracker_record(self):
        """Test recording stats"""
        from nexlify.memory.per_visualization import PERStatsTracker

        tracker = PERStatsTracker()

        stats = {
            "beta": 0.5,
            "mean_priority": 1.5,
            "max_priority": 3.0,
            "min_priority": 0.1,
            "total_priority": 100.0,
            "size": 50,
            "capacity": 100,
            "total_samples": 1000,
            "priority_updates": 500,
        }

        tracker.record(stats, episode=10)

        assert len(tracker.history["beta"]) == 1
        assert tracker.history["beta"][0] == 0.5
        assert tracker.history["episodes"][0] == 10

    def test_stats_tracker_summary(self):
        """Test getting summary statistics"""
        from nexlify.memory.per_visualization import PERStatsTracker

        tracker = PERStatsTracker()

        # Record multiple stats
        for i in range(10):
            stats = {
                "beta": 0.4 + i * 0.06,
                "mean_priority": 1.0 + i * 0.1,
                "max_priority": 2.0 + i * 0.2,
                "min_priority": 0.1,
                "total_priority": 100.0,
                "size": 50,
                "capacity": 100,
                "total_samples": 1000 + i * 100,
                "priority_updates": 500 + i * 50,
            }
            tracker.record(stats, episode=i)

        summary = tracker.get_summary()

        assert "final_beta" in summary
        assert "mean_priority_avg" in summary
        # Final beta should be 0.4 + 9*0.06 = 0.94
        assert summary["final_beta"] == pytest.approx(0.94, abs=0.01)

    def test_create_report(self):
        """Test creating PER report"""
        from nexlify.memory.per_visualization import create_per_report

        stats = {
            "beta": 0.5,
            "mean_priority": 1.5,
            "max_priority": 3.0,
            "min_priority": 0.1,
            "total_priority": 100.0,
            "size": 50,
            "capacity": 100,
            "total_samples": 1000,
            "priority_updates": 500,
            "alpha": 0.6,
        }

        report = create_per_report(stats)

        assert "PRIORITIZED EXPERIENCE REPLAY" in report
        assert "Beta (IS weight)" in report
        assert "0.5" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
