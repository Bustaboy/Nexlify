#!/usr/bin/env python3
"""
Double DQN and Dueling DQN Example
Demonstrates how to use the new architectures for trading

This example shows:
1. Creating agents with different configurations
2. Training and comparing architectures
3. Analyzing overestimation reduction
4. Running ablation studies
"""

import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexlify.strategies.double_dqn_agent import DoubleDQNAgent
from nexlify.strategies.nexlify_rl_agent import TradingEnvironment
from nexlify.utils.architecture_comparison import ArchitectureComparator, run_ablation_study


def example_1_basic_usage():
    """Example 1: Basic usage of Double DQN and Dueling DQN"""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Create test environment
    price_data = np.random.randn(1000) * 10 + 100
    env = TradingEnvironment(price_data, initial_balance=10000)

    # Create agent with Double + Dueling DQN
    agent = DoubleDQNAgent(
        state_size=env.state_space_n,
        action_size=env.action_space_n,
        config={
            "use_double_dqn": True,
            "use_dueling_dqn": True,
            "learning_rate": 0.001,
            "batch_size": 64,
            "track_q_values": True,
        },
    )

    print(agent.get_model_summary())

    # Train for a few episodes
    for episode in range(10):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= 64:
                loss = agent.replay()

            episode_reward += reward
            state = next_state

        agent.decay_epsilon()
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # Get Q-value statistics
    q_stats = agent.get_q_value_stats()
    if q_stats:
        print("\nQ-Value Statistics:")
        print(f"  Mean Q: {q_stats['mean_q_value']:.4f}")
        print(f"  Std Q: {q_stats['std_q_value']:.4f}")


def example_2_architecture_variants():
    """Example 2: Compare different architecture variants"""
    print("\n" + "=" * 80)
    print("Example 2: Architecture Variants")
    print("=" * 80)

    # Create environment
    price_data = np.random.randn(500) * 10 + 100
    env = TradingEnvironment(price_data, initial_balance=10000)

    # Test different configurations
    configs = [
        ("Standard DQN", {"use_double_dqn": False, "use_dueling_dqn": False}),
        ("Double DQN", {"use_double_dqn": True, "use_dueling_dqn": False}),
        ("Dueling DQN", {"use_double_dqn": False, "use_dueling_dqn": True}),
        ("Double + Dueling", {"use_double_dqn": True, "use_dueling_dqn": True}),
    ]

    for name, config in configs:
        agent = DoubleDQNAgent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            config=config,
        )

        print(f"\n{name}:")
        print(f"  Architecture: {agent._get_architecture_name()}")
        print(f"  Double DQN: {agent.use_double_dqn}")
        print(f"  Dueling DQN: {agent.use_dueling_dqn}")


def example_3_comparison_tool():
    """Example 3: Using the architecture comparison tool"""
    print("\n" + "=" * 80)
    print("Example 3: Architecture Comparison Tool")
    print("=" * 80)

    # Create comparator
    comparator = ArchitectureComparator(output_dir="comparison_results")

    # Simulate results from training
    print("\nSimulating training results...")

    # Standard DQN (baseline)
    for i in range(100):
        reward = i * 5.0 + np.random.randn() * 10
        loss = 1.0 / (i + 1)
        comparator.add_result("standard_dqn", i, reward, loss)

    # Double DQN (better)
    for i in range(100):
        reward = i * 6.0 + np.random.randn() * 8
        loss = 0.8 / (i + 1)
        comparator.add_result("double_dqn", i, reward, loss)

    # Dueling DQN (better)
    for i in range(100):
        reward = i * 6.5 + np.random.randn() * 8
        loss = 0.85 / (i + 1)
        comparator.add_result("dueling_dqn", i, reward, loss)

    # Double + Dueling (best)
    for i in range(100):
        reward = i * 7.0 + np.random.randn() * 7
        loss = 0.7 / (i + 1)
        comparator.add_result("double_dueling_dqn", i, reward, loss)

    # Compute metrics
    metrics = comparator.compute_metrics()

    print("\nPerformance Metrics:")
    for arch_name, arch_metrics in sorted(
        metrics.items(), key=lambda x: x[1]["final_reward"], reverse=True
    ):
        print(f"\n{arch_name}:")
        print(f"  Final Reward: {arch_metrics['final_reward']:.2f}")
        print(f"  Convergence Speed: {arch_metrics['convergence_speed']} episodes")
        print(f"  Sample Efficiency: {arch_metrics['sample_efficiency']:.4f}")

    # Get best architecture
    best_arch, best_metrics = comparator.get_best_architecture()
    print(f"\n🏆 Best Architecture: {best_arch}")
    print(f"   Final Reward: {best_metrics['final_reward']:.2f}")

    # Generate and print report
    print("\n" + comparator.generate_report())


def example_4_configuration_options():
    """Example 4: Advanced configuration options"""
    print("\n" + "=" * 80)
    print("Example 4: Advanced Configuration")
    print("=" * 80)

    # Custom Dueling architecture
    config = {
        "use_double_dqn": True,
        "use_dueling_dqn": True,
        # Network architecture
        "dueling_shared_sizes": [256, 128],  # Shared feature extractor
        "dueling_value_sizes": [64, 32],  # Value stream
        "dueling_advantage_sizes": [64, 32],  # Advantage stream
        "dueling_aggregation": "mean",  # 'mean' or 'max'
        # Training
        "learning_rate": 0.0005,
        "batch_size": 128,
        "replay_buffer_size": 50000,
        # Q-value tracking
        "track_q_values": True,
    }

    # Create environment
    price_data = np.random.randn(500) * 10 + 100
    env = TradingEnvironment(price_data, initial_balance=10000)

    agent = DoubleDQNAgent(
        state_size=env.state_space_n,
        action_size=env.action_space_n,
        config=config,
    )

    print("Custom Configuration:")
    print(f"  Shared layers: {config['dueling_shared_sizes']}")
    print(f"  Value stream: {config['dueling_value_sizes']}")
    print(f"  Advantage stream: {config['dueling_advantage_sizes']}")
    print(f"  Aggregation: {config['dueling_aggregation']}")
    print(f"\n{agent.get_model_summary()}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("Double DQN and Dueling DQN Examples")
    print("=" * 80)

    try:
        example_1_basic_usage()
        example_2_architecture_variants()
        example_3_comparison_tool()
        example_4_configuration_options()

        print("\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
