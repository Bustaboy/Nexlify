#!/usr/bin/env python3
"""
Advanced Epsilon Decay Manager - Usage Examples
Demonstrates different decay strategies for RL exploration-exploitation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexlify.strategies.epsilon_decay import (
    LinearEpsilonDecay,
    ScheduledEpsilonDecay,
    ExponentialEpsilonDecay,
    EpsilonDecayFactory
)


def demo_linear_decay():
    """Demonstrate linear epsilon decay"""
    print("\n" + "=" * 60)
    print("LINEAR EPSILON DECAY")
    print("=" * 60)

    strategy = LinearEpsilonDecay(
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_steps=2000
    )

    print(f"\nConfiguration:")
    print(f"  Start: {strategy.epsilon_start}")
    print(f"  End: {strategy.epsilon_end}")
    print(f"  Decay Steps: {strategy.decay_steps}")

    print(f"\nExpected Outputs:")
    milestones = [0, 100, 500, 1000, 2000]
    for step in milestones:
        epsilon = strategy.get_epsilon(step)
        print(f"  Episode {step:4d}: ε = {epsilon:.4f}")

    return strategy


def demo_scheduled_decay():
    """Demonstrate scheduled epsilon decay"""
    print("\n" + "=" * 60)
    print("SCHEDULED EPSILON DECAY")
    print("=" * 60)

    schedule = {
        0: 1.0,
        300: 0.7,
        1000: 0.3,
        2000: 0.05
    }

    strategy = ScheduledEpsilonDecay(schedule=schedule)

    print(f"\nConfiguration:")
    print(f"  Schedule: {schedule}")

    print(f"\nExpected Outputs:")
    milestones = [0, 150, 300, 650, 1000, 1500, 2000]
    for step in milestones:
        epsilon = strategy.get_epsilon(step)
        print(f"  Episode {step:4d}: ε = {epsilon:.4f}")

    return strategy


def demo_exponential_decay():
    """Demonstrate exponential epsilon decay"""
    print("\n" + "=" * 60)
    print("EXPONENTIAL EPSILON DECAY")
    print("=" * 60)

    strategy = ExponentialEpsilonDecay(
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_steps=2000
    )

    print(f"\nConfiguration:")
    print(f"  Start: {strategy.epsilon_start}")
    print(f"  End: {strategy.epsilon_end}")
    print(f"  Decay Steps: {strategy.decay_steps}")
    print(f"  Auto-calculated decay rate: {strategy.decay_rate:.6f}")

    print(f"\nExpected Outputs:")
    milestones = [0, 100, 500, 1000, 1500, 2000]
    for step in milestones:
        epsilon = strategy.get_epsilon(step)
        print(f"  Episode {step:4d}: ε = {epsilon:.4f}")

    return strategy


def demo_factory():
    """Demonstrate using the factory"""
    print("\n" + "=" * 60)
    print("EPSILON DECAY FACTORY")
    print("=" * 60)

    # Create from config
    config = {
        'epsilon_decay_type': 'linear',
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 2000
    }

    print(f"\nConfig: {config}")

    strategy = EpsilonDecayFactory.create_from_config(config)

    print(f"\nCreated: {type(strategy).__name__}")
    print(f"  Epsilon at step 0: {strategy.get_epsilon(0):.4f}")
    print(f"  Epsilon at step 1000: {strategy.get_epsilon(1000):.4f}")


def demo_dqn_agent_integration():
    """Demonstrate DQNAgent integration"""
    print("\n" + "=" * 60)
    print("DQN AGENT INTEGRATION")
    print("=" * 60)

    from nexlify.strategies.nexlify_rl_agent import DQNAgent

    config = {
        'epsilon_decay_type': 'linear',
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 2000
    }

    print(f"\nAgent Config: {config}")

    agent = DQNAgent(state_size=8, action_size=3, config=config)

    print(f"\nAgent created with:")
    print(f"  Strategy: {type(agent.epsilon_decay_strategy).__name__}")
    print(f"  Initial epsilon: {agent.epsilon:.4f}")

    # Simulate some training episodes
    print(f"\nSimulating decay over episodes:")
    for episode in [0, 10, 50, 100, 500, 1000]:
        # Set the strategy to the correct step
        agent.epsilon_decay_strategy.current_step = episode
        agent.epsilon = agent.epsilon_decay_strategy.get_epsilon(episode)
        print(f"  Episode {episode:4d}: ε = {agent.epsilon:.4f}")


def plot_comparison():
    """Plot all strategies for comparison"""
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON PLOT")
    print("=" * 60)

    steps = np.arange(0, 2001, 10)

    # Create strategies
    linear = LinearEpsilonDecay(1.0, 0.05, 2000)
    scheduled = ScheduledEpsilonDecay(schedule={0: 1.0, 300: 0.7, 1000: 0.3, 2000: 0.05})
    exponential = ExponentialEpsilonDecay(1.0, 0.05, 2000)

    # Get epsilon values
    linear_epsilons = [linear.get_epsilon(step) for step in steps]
    scheduled_epsilons = [scheduled.get_epsilon(step) for step in steps]
    exponential_epsilons = [exponential.get_epsilon(step) for step in steps]

    # Create plot
    plt.figure(figsize=(12, 8))

    plt.plot(steps, linear_epsilons, label='Linear', linewidth=2)
    plt.plot(steps, scheduled_epsilons, label='Scheduled', linewidth=2, linestyle='--')
    plt.plot(steps, exponential_epsilons, label='Exponential', linewidth=2, linestyle='-.')

    # Mark key thresholds
    thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
    for threshold in thresholds:
        plt.axhline(y=threshold, color='gray', linestyle=':', alpha=0.3)
        plt.text(2050, threshold, f'ε={threshold}', va='center', fontsize=8)

    # Mark key episodes
    key_episodes = [100, 500, 1000, 2000]
    for episode in key_episodes:
        plt.axvline(x=episode, color='gray', linestyle=':', alpha=0.3)
        plt.text(episode, 0.02, f'{episode}', ha='center', fontsize=8)

    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Epsilon (Exploration Rate)', fontsize=12)
    plt.title('Epsilon Decay Strategy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2100)
    plt.ylim(0, 1.05)

    # Save plot
    output_path = Path(__file__).parent / 'epsilon_decay_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")


def main():
    """Run all demos"""
    print("=" * 60)
    print("ADVANCED EPSILON DECAY MANAGER - DEMO")
    print("=" * 60)

    # Run demos
    demo_linear_decay()
    demo_scheduled_decay()
    demo_exponential_decay()
    demo_factory()
    demo_dqn_agent_integration()

    # Generate comparison plot
    try:
        plot_comparison()
    except Exception as e:
        print(f"\n⚠️  Could not generate plot: {e}")

    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("=" * 60)
    print("\nKey Features:")
    print("  ✓ Multiple decay strategies (Linear, Scheduled, Exponential)")
    print("  ✓ Configurable parameters")
    print("  ✓ Automatic threshold logging")
    print("  ✓ Epsilon history tracking")
    print("  ✓ Factory pattern for easy creation")
    print("  ✓ Seamless DQNAgent integration")
    print("\nNext Steps:")
    print("  1. Choose a decay strategy in your config")
    print("  2. Set epsilon_decay_type: 'linear', 'scheduled', or 'exponential'")
    print("  3. Configure epsilon_decay_steps based on training length")
    print("  4. Monitor epsilon thresholds during training")
    print()


if __name__ == '__main__':
    main()
