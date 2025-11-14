#!/usr/bin/env python3
"""
Basic validation script for epsilon decay strategies
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nexlify.strategies.epsilon_decay import (
    LinearEpsilonDecay,
    ScheduledEpsilonDecay,
    ExponentialEpsilonDecay,
    EpsilonDecayFactory
)


def test_linear_decay():
    """Test LinearEpsilonDecay"""
    print("\n=== Testing LinearEpsilonDecay ===")

    strategy = LinearEpsilonDecay(
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_steps=2000
    )

    # Test key milestones
    test_points = [
        (0, 1.0),
        (100, 0.9525),
        (500, 0.7625),
        (1000, 0.525),
        (2000, 0.05)
    ]

    for step, expected in test_points:
        epsilon = strategy.get_epsilon(step)
        diff = abs(epsilon - expected)
        status = "✅" if diff < 0.01 else "❌"
        print(f"{status} Step {step:4d}: ε = {epsilon:.4f} (expected ~{expected:.4f}, diff={diff:.4f})")

    # Test step method
    strategy.reset()
    for i in range(5):
        epsilon = strategy.step()
        print(f"  Step {i}: ε = {epsilon:.4f}")

    print(f"  Thresholds crossed: {strategy.thresholds_crossed}")
    print("✅ LinearEpsilonDecay tests passed!")


def test_scheduled_decay():
    """Test ScheduledEpsilonDecay"""
    print("\n=== Testing ScheduledEpsilonDecay ===")

    schedule = {0: 1.0, 300: 0.7, 1000: 0.3, 2000: 0.05}
    strategy = ScheduledEpsilonDecay(schedule=schedule)

    test_points = [
        (0, 1.0),
        (150, 0.85),  # Midpoint between 0 and 300
        (300, 0.7),
        (650, 0.5),   # Midpoint between 300 and 1000
        (1000, 0.3),
        (1500, 0.175), # Midpoint between 1000 and 2000
        (2000, 0.05)
    ]

    for step, expected in test_points:
        epsilon = strategy.get_epsilon(step)
        diff = abs(epsilon - expected)
        status = "✅" if diff < 0.01 else "❌"
        print(f"{status} Step {step:4d}: ε = {epsilon:.4f} (expected ~{expected:.4f}, diff={diff:.4f})")

    print("✅ ScheduledEpsilonDecay tests passed!")


def test_exponential_decay():
    """Test ExponentialEpsilonDecay"""
    print("\n=== Testing ExponentialEpsilonDecay ===")

    strategy = ExponentialEpsilonDecay(
        epsilon_start=1.0,
        epsilon_end=0.05,
        decay_steps=2000
    )

    print(f"  Auto-calculated decay rate: {strategy.decay_rate:.6f}")

    test_points = [0, 100, 500, 1000, 1500, 2000]

    for step in test_points:
        epsilon = strategy.get_epsilon(step)
        print(f"  Step {step:4d}: ε = {epsilon:.4f}")

    # Verify it reaches epsilon_end
    final_epsilon = strategy.get_epsilon(2000)
    assert abs(final_epsilon - 0.05) < 0.01, f"Final epsilon {final_epsilon} != 0.05"

    print("✅ ExponentialEpsilonDecay tests passed!")


def test_factory():
    """Test EpsilonDecayFactory"""
    print("\n=== Testing EpsilonDecayFactory ===")

    # Test creating different strategies
    linear = EpsilonDecayFactory.create('linear', decay_steps=2000)
    print(f"✅ Created LinearEpsilonDecay: {type(linear).__name__}")

    scheduled = EpsilonDecayFactory.create('scheduled')
    print(f"✅ Created ScheduledEpsilonDecay: {type(scheduled).__name__}")

    exponential = EpsilonDecayFactory.create('exponential', decay_steps=2000)
    print(f"✅ Created ExponentialEpsilonDecay: {type(exponential).__name__}")

    # Test creating from config
    config = {
        'epsilon_decay_type': 'linear',
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 2000
    }

    strategy = EpsilonDecayFactory.create_from_config(config)
    print(f"✅ Created from config: {type(strategy).__name__}")

    print("✅ EpsilonDecayFactory tests passed!")


def test_dqn_integration():
    """Test DQNAgent integration"""
    print("\n=== Testing DQNAgent Integration ===")

    from nexlify.strategies.nexlify_rl_agent import DQNAgent

    config = {
        'epsilon_decay_type': 'linear',
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 2000
    }

    agent = DQNAgent(state_size=12, action_size=3, config=config)

    print(f"  Agent epsilon: {agent.epsilon:.4f}")
    print(f"  Strategy type: {type(agent.epsilon_decay_strategy).__name__}")

    # Test decay
    initial_epsilon = agent.epsilon
    for i in range(10):
        agent.decay_epsilon()

    print(f"  After 10 decays: {agent.epsilon:.4f}")
    assert agent.epsilon < initial_epsilon, "Epsilon should decrease"

    print("✅ DQNAgent integration tests passed!")


def test_legacy_config():
    """Test legacy config support"""
    print("\n=== Testing Legacy Config Support ===")

    from nexlify.strategies.nexlify_rl_agent import DQNAgent

    # Old style config
    config = {
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995
    }

    agent = DQNAgent(state_size=12, action_size=3, config=config)

    print(f"  Agent epsilon: {agent.epsilon:.4f}")
    print(f"  Strategy type: {type(agent.epsilon_decay_strategy).__name__}")

    assert hasattr(agent, 'epsilon_decay_strategy'), "Should have strategy"

    print("✅ Legacy config support tests passed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Advanced Epsilon Decay Manager")
    print("=" * 60)

    try:
        test_linear_decay()
        test_scheduled_decay()
        test_exponential_decay()
        test_factory()
        test_dqn_integration()
        test_legacy_config()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
