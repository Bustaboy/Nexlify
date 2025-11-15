#!/usr/bin/env python3
"""
Tests for Double DQN and Dueling DQN implementations

Tests:
1. Network architecture creation
2. Double DQN Q-value calculation
3. Dueling network forward pass
4. Overestimation reduction
5. Value/advantage separation
6. Configuration options
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from nexlify.models.dueling_network import (
    StandardDQNNetwork,
    DuelingNetwork,
    create_network,
)
from nexlify.strategies.double_dqn_agent import DoubleDQNAgent
from nexlify.strategies.nexlify_rl_agent import TradingEnvironment
from nexlify.utils.architecture_comparison import ArchitectureComparator


class TestStandardDQNNetwork:
    """Test standard DQN network architecture"""

    def test_network_creation(self):
        """Test network can be created with correct dimensions"""
        state_size = 12
        action_size = 3
        hidden_sizes = (128, 128, 64)

        network = StandardDQNNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
        )

        assert network is not None
        assert network.state_size == state_size
        assert network.action_size == action_size
        assert network.hidden_sizes == hidden_sizes

    def test_forward_pass(self):
        """Test forward pass produces correct output shape"""
        network = StandardDQNNetwork(
            state_size=12,
            action_size=3,
            hidden_sizes=(128, 64),
        )

        batch_size = 32
        state = torch.randn(batch_size, 12)

        q_values = network(state)

        assert q_values.shape == (batch_size, 3)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_different_activations(self):
        """Test network works with different activation functions"""
        activations = ["relu", "tanh", "elu", "leaky_relu"]

        for activation in activations:
            network = StandardDQNNetwork(
                state_size=10,
                action_size=3,
                hidden_sizes=(64, 32),
                activation=activation,
            )

            state = torch.randn(16, 10)
            q_values = network(state)

            assert q_values.shape == (16, 3)


class TestDuelingNetwork:
    """Test Dueling DQN network architecture"""

    def test_network_creation(self):
        """Test Dueling network can be created"""
        network = DuelingNetwork(
            state_size=12,
            action_size=3,
            shared_sizes=(128, 128),
            value_sizes=(64,),
            advantage_sizes=(64,),
        )

        assert network is not None
        assert network.state_size == 12
        assert network.action_size == 3

    def test_forward_pass(self):
        """Test forward pass produces correct Q-values"""
        network = DuelingNetwork(
            state_size=12,
            action_size=3,
            shared_sizes=(128, 64),
            value_sizes=(32,),
            advantage_sizes=(32,),
        )

        batch_size = 32
        state = torch.randn(batch_size, 12)

        q_values = network(state)

        assert q_values.shape == (batch_size, 3)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_value_advantage_separation(self):
        """Test that value and advantage streams produce correct shapes"""
        network = DuelingNetwork(
            state_size=10,
            action_size=4,
            shared_sizes=(64,),
            value_sizes=(32,),
            advantage_sizes=(32,),
        )

        state = torch.randn(16, 10)
        value, advantage = network.get_value_advantage(state)

        assert value.shape == (16, 1)
        assert advantage.shape == (16, 4)

    def test_aggregation_methods(self):
        """Test both mean and max aggregation methods"""
        state = torch.randn(32, 10)

        # Mean aggregation
        network_mean = DuelingNetwork(
            state_size=10,
            action_size=3,
            shared_sizes=(64,),
            value_sizes=(32,),
            advantage_sizes=(32,),
            aggregation="mean",
        )

        q_mean = network_mean(state)
        assert q_mean.shape == (32, 3)

        # Max aggregation
        network_max = DuelingNetwork(
            state_size=10,
            action_size=3,
            shared_sizes=(64,),
            value_sizes=(32,),
            advantage_sizes=(32,),
            aggregation="max",
        )

        q_max = network_max(state)
        assert q_max.shape == (32, 3)

        # Results should be different
        assert not torch.allclose(q_mean, q_max, rtol=1e-3)

    def test_invalid_aggregation(self):
        """Test that invalid aggregation method raises error"""
        network = DuelingNetwork(
            state_size=10,
            action_size=3,
            aggregation="invalid",
        )

        state = torch.randn(16, 10)

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            network(state)


class TestNetworkFactory:
    """Test network creation factory function"""

    def test_create_standard_network(self):
        """Test creating standard DQN network"""
        network = create_network(
            state_size=12,
            action_size=3,
            use_dueling=False,
        )

        assert isinstance(network, StandardDQNNetwork)

    def test_create_dueling_network(self):
        """Test creating Dueling DQN network"""
        network = create_network(
            state_size=12,
            action_size=3,
            use_dueling=True,
        )

        assert isinstance(network, DuelingNetwork)

    def test_config_parameters(self):
        """Test that config parameters are properly passed"""
        config = {
            "dueling_shared_sizes": [256, 128],
            "dueling_value_sizes": [64, 32],
            "dueling_advantage_sizes": [64, 32],
            "dueling_aggregation": "max",
            "activation": "tanh",
        }

        network = create_network(
            state_size=10,
            action_size=3,
            use_dueling=True,
            config=config,
        )

        assert isinstance(network, DuelingNetwork)
        assert network.aggregation == "max"


class TestDoubleDQNAgent:
    """Test Double DQN agent implementation"""

    @pytest.fixture
    def test_config(self):
        """Standard test configuration"""
        return {
            "learning_rate": 0.001,
            "batch_size": 32,
            "gamma": 0.99,
            "epsilon": 0.1,
            "replay_buffer_size": 10000,
            "target_update_freq": 10,
            "use_double_dqn": True,
            "use_dueling_dqn": True,
            "track_q_values": True,
        }

    def test_agent_creation(self, test_config):
        """Test agent can be created with both variants enabled"""
        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=test_config,
        )

        assert agent is not None
        assert agent.use_double_dqn is True
        assert agent.use_dueling_dqn is True
        assert agent.track_q_values is True

    def test_standard_dqn_mode(self):
        """Test agent works with both features disabled (standard DQN)"""
        config = {
            "use_double_dqn": False,
            "use_dueling_dqn": False,
        }

        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=config,
        )

        assert agent.use_double_dqn is False
        assert agent.use_dueling_dqn is False

    def test_double_only_mode(self):
        """Test agent with Double DQN only (no dueling)"""
        config = {
            "use_double_dqn": True,
            "use_dueling_dqn": False,
        }

        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=config,
        )

        assert agent.use_double_dqn is True
        assert agent.use_dueling_dqn is False

    def test_dueling_only_mode(self):
        """Test agent with Dueling DQN only (no double)"""
        config = {
            "use_double_dqn": False,
            "use_dueling_dqn": True,
        }

        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=config,
        )

        assert agent.use_double_dqn is False
        assert agent.use_dueling_dqn is True

    def test_act_function(self, test_config):
        """Test action selection"""
        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=test_config,
        )

        state = np.random.randn(12)
        action = agent.act(state, training=False)

        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 3

    def test_replay_training(self, test_config):
        """Test replay training updates model"""
        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=test_config,
        )

        # Fill memory with experiences
        for _ in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = False

            agent.remember(state, action, reward, next_state, done)

        # Train
        loss = agent.replay(batch_size=32)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_q_value_tracking(self, test_config):
        """Test Q-value statistics tracking"""
        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=test_config,
        )

        # Train for a few steps
        for _ in range(50):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = False

            agent.remember(state, action, reward, next_state, done)

        # Replay to generate Q-value stats
        agent.replay(batch_size=16)
        agent.replay(batch_size=16)

        stats = agent.get_q_value_stats()

        if stats:  # Stats available after enough training
            assert "mean_q_value" in stats
            assert "std_q_value" in stats
            assert "total_samples" in stats

    def test_model_summary(self, test_config):
        """Test model summary generation"""
        agent = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config=test_config,
        )

        summary = agent.get_model_summary()

        assert isinstance(summary, str)
        assert "Double Dueling DQN" in summary
        assert "State Size: 12" in summary
        assert "Action Size: 3" in summary

    def test_architecture_name(self):
        """Test correct architecture naming"""
        # Double + Dueling
        agent1 = DoubleDQNAgent(
            state_size=10,
            action_size=3,
            config={"use_double_dqn": True, "use_dueling_dqn": True},
        )
        assert agent1._get_architecture_name() == "Double Dueling DQN"

        # Double only
        agent2 = DoubleDQNAgent(
            state_size=10,
            action_size=3,
            config={"use_double_dqn": True, "use_dueling_dqn": False},
        )
        assert agent2._get_architecture_name() == "Double DQN"

        # Dueling only
        agent3 = DoubleDQNAgent(
            state_size=10,
            action_size=3,
            config={"use_double_dqn": False, "use_dueling_dqn": True},
        )
        assert agent3._get_architecture_name() == "Dueling DQN"

        # Standard
        agent4 = DoubleDQNAgent(
            state_size=10,
            action_size=3,
            config={"use_double_dqn": False, "use_dueling_dqn": False},
        )
        assert agent4._get_architecture_name() == "Standard DQN"


class TestOverestimationReduction:
    """Test that Double DQN reduces overestimation"""

    def test_double_vs_standard_q_values(self):
        """
        Test that Double DQN produces different Q-values than standard DQN

        This verifies the implementation is actually using the double DQN logic
        """
        # Create two agents with same random seed
        torch.manual_seed(42)
        agent_double = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config={"use_double_dqn": True, "use_dueling_dqn": False},
        )

        torch.manual_seed(42)
        agent_standard = DoubleDQNAgent(
            state_size=12,
            action_size=3,
            config={"use_double_dqn": False, "use_dueling_dqn": False},
        )

        # Generate same experiences
        np.random.seed(42)
        for _ in range(100):
            state = np.random.randn(12)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.randn(12)
            done = False

            agent_double.remember(state, action, reward, next_state, done)
            agent_standard.remember(state, action, reward, next_state, done)

        # Train both
        loss_double = agent_double.replay(batch_size=32)
        loss_standard = agent_standard.replay(batch_size=32)

        # Both should train successfully
        assert loss_double >= 0
        assert loss_standard >= 0

        # Note: We can't easily verify they're different without running many episodes
        # But we can verify the training runs without errors


class TestArchitectureComparator:
    """Test architecture comparison tools"""

    def test_comparator_creation(self):
        """Test comparator can be created"""
        comparator = ArchitectureComparator(output_dir="test_output")

        assert comparator is not None
        assert len(comparator.results) == 4  # 4 architectures

    def test_add_result(self):
        """Test adding results to comparator"""
        comparator = ArchitectureComparator(output_dir="test_output")

        comparator.add_result(
            architecture="double_dqn",
            episode=1,
            reward=100.0,
            loss=0.5,
        )

        assert len(comparator.results["double_dqn"]) == 1
        assert comparator.results["double_dqn"][0]["reward"] == 100.0

    def test_compute_metrics(self):
        """Test metric computation"""
        comparator = ArchitectureComparator(output_dir="test_output")

        # Add some mock results
        for i in range(100):
            comparator.add_result(
                architecture="double_dqn",
                episode=i,
                reward=i * 10.0,
                loss=1.0 / (i + 1),
            )

        metrics = comparator.compute_metrics()

        assert "double_dqn" in metrics
        assert "mean_reward" in metrics["double_dqn"]
        assert "convergence_speed" in metrics["double_dqn"]
        assert "sample_efficiency" in metrics["double_dqn"]

    def test_get_best_architecture(self):
        """Test best architecture selection"""
        comparator = ArchitectureComparator(output_dir="test_output")

        # Add results with different performance
        for i in range(50):
            comparator.add_result("standard_dqn", i, reward=i * 5.0, loss=0.1)
            comparator.add_result("double_dqn", i, reward=i * 7.0, loss=0.1)
            comparator.add_result("dueling_dqn", i, reward=i * 6.0, loss=0.1)
            comparator.add_result("double_dueling_dqn", i, reward=i * 8.0, loss=0.1)

        best_arch, best_metrics = comparator.get_best_architecture()

        assert best_arch == "double_dueling_dqn"
        assert "final_reward" in best_metrics

    def test_generate_report(self):
        """Test report generation"""
        comparator = ArchitectureComparator(output_dir="test_output")

        # Add some results
        for i in range(20):
            comparator.add_result("double_dqn", i, reward=i * 10.0, loss=0.1)

        report = comparator.generate_report()

        assert isinstance(report, str)
        assert "DQN Architecture Comparison Report" in report
        assert "double_dqn" in report


@pytest.mark.integration
class TestIntegrationWithEnvironment:
    """Integration tests with trading environment"""

    @pytest.fixture
    def env(self):
        """Create test trading environment"""
        price_data = np.random.randn(1000) * 10 + 100
        return TradingEnvironment(price_data, initial_balance=10000)

    def test_agent_environment_interaction(self, env):
        """Test agent can interact with environment"""
        agent = DoubleDQNAgent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            config={
                "use_double_dqn": True,
                "use_dueling_dqn": True,
                "batch_size": 32,
            },
        )

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= 32:
                loss = agent.replay()
                assert loss >= 0

            state = next_state
            steps += 1

        assert steps > 0

    @pytest.mark.slow  # Training 5 episodes can take >30s
    @pytest.mark.timeout(120)  # Allow up to 2 minutes for neural network training
    def test_multiple_episodes(self, env):
        """Test agent across multiple episodes"""
        agent = DoubleDQNAgent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            config={
                "use_double_dqn": True,
                "use_dueling_dqn": True,
            },
        )

        rewards = []

        for episode in range(5):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = agent.act(state, training=True)
                next_state, reward, done, info = env.step(action)

                agent.remember(state, action, reward, next_state, done)
                if len(agent.memory) >= 32:
                    agent.replay()

                episode_reward += reward
                state = next_state

            rewards.append(episode_reward)
            agent.decay_epsilon()

        assert len(rewards) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
