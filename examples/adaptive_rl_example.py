#!/usr/bin/env python3
"""
Nexlify Adaptive RL Agent - Usage Examples

This file demonstrates how to use the adaptive RL agent with different
hardware configurations and scenarios.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from nexlify.strategies.nexlify_adaptive_rl_agent import (
    create_optimized_agent,
    HardwareProfiler,
    AdaptiveDQNAgent
)
from nexlify.strategies.nexlify_rl_agent import TradingEnvironment


def example_1_auto_detection():
    """
    Example 1: Automatic hardware detection and optimization
    The agent automatically detects your hardware and configures itself
    """
    print("=" * 70)
    print("EXAMPLE 1: Auto-Detection")
    print("=" * 70)

    # Create agent with automatic hardware detection
    agent = create_optimized_agent(
        state_size=8,
        action_size=3,
        auto_detect=True  # This is the default
    )

    print(f"\n✅ Agent created with auto-detected configuration")
    print(f"   Model: {agent.config.get('model_size', 'unknown')}")
    print(f"   Batch size: {agent.batch_size}")
    print(f"   Buffer: {agent.buffer_capacity:,}")
    print(f"   Device: {agent.device}")


def example_2_manual_override():
    """
    Example 2: Manual configuration override
    Override automatic detection for specific requirements
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Manual Override")
    print("=" * 70)

    # Force specific configuration
    custom_config = {
        'model_size': 'large',
        'batch_size': 128,
        'use_mixed_precision': True
    }

    agent = create_optimized_agent(
        state_size=8,
        action_size=3,
        auto_detect=True,
        config_override=custom_config
    )

    print(f"\n✅ Agent created with custom configuration")
    print(f"   Model: {agent.config.get('model_size')}")
    print(f"   Batch size: {agent.batch_size}")
    print(f"   Mixed precision: {agent.use_mixed_precision}")


def example_3_hardware_profiling():
    """
    Example 3: Detailed hardware profiling
    See what hardware capabilities are detected
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Hardware Profiling")
    print("=" * 70)

    profiler = HardwareProfiler()

    print("\n📊 Detected Hardware:")
    print(f"   CPU: {profiler.profile['cpu']['cores_physical']} cores, "
          f"{profiler.profile['cpu']['tier']}")
    print(f"   RAM: {profiler.profile['ram']['total_gb']:.1f} GB, "
          f"{profiler.profile['ram']['tier']}")

    if profiler.profile['gpu']['available']:
        print(f"   GPU: {profiler.profile['gpu']['name']}")
        print(f"        VRAM: {profiler.profile['gpu']['vram_gb']:.1f} GB")
        print(f"        Tier: {profiler.profile['gpu']['tier']}")
    else:
        print(f"   GPU: None detected")

    print("\n⚡ Performance Benchmark:")
    bench = profiler.profile.get('benchmark', {})
    print(f"   CPU: {bench.get('cpu_gflops', 0):.1f} GFLOPS")
    print(f"   GPU: {bench.get('gpu_gflops', 0):.1f} GFLOPS")
    print(f"   Memory: {bench.get('memory_bandwidth_gbps', 0):.1f} GB/s")

    print("\n⚙️  Recommended Configuration:")
    config = profiler.optimal_config
    print(f"   Model: {config.get('model_size', 'unknown').upper()}")
    print(f"   Batch: {config.get('batch_size', 0)}")
    print(f"   Buffer: {config.get('buffer_size', 0):,}")
    print(f"   Optimization: {config.get('optimization_level', 'unknown')}")


def example_4_training_simple():
    """
    Example 4: Simple training loop
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Simple Training")
    print("=" * 70)

    # Generate simple price data
    np.random.seed(42)
    prices = 30000 + np.cumsum(np.random.randn(1000) * 100)

    # Create environment
    env = TradingEnvironment(prices, initial_balance=10000)

    # Create agent
    agent = create_optimized_agent(
        state_size=env.state_space_n,
        action_size=env.action_space_n
    )

    print(f"\n🚀 Training for 10 episodes...")

    for episode in range(10):
        state = env.reset()
        episode_reward = 0

        for step in range(env.max_steps):
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.batch_size:
                agent.replay(iteration=step)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()

        profit = env.get_portfolio_value() - 10000
        print(f"   Episode {episode + 1:2d}: Profit=${profit:+8.2f}, "
              f"Reward={episode_reward:7.2f}, ε={agent.epsilon:.3f}")

    print("\n✅ Training complete")


def example_5_save_load():
    """
    Example 5: Save and load models
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Save/Load Model")
    print("=" * 70)

    # Create agent
    agent = create_optimized_agent(state_size=8, action_size=3)

    # Save model
    save_path = "models/example_model.pth"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)
    print(f"\n💾 Model saved to {save_path}")

    # Load model
    new_agent = create_optimized_agent(state_size=8, action_size=3)
    new_agent.load(save_path)
    print(f"✅ Model loaded from {save_path}")


def example_6_performance_monitoring():
    """
    Example 6: Real-time performance monitoring
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Performance Monitoring")
    print("=" * 70)

    # Generate price data
    np.random.seed(42)
    prices = 30000 + np.cumsum(np.random.randn(500) * 100)

    env = TradingEnvironment(prices, initial_balance=10000)
    agent = create_optimized_agent(
        state_size=env.state_space_n,
        action_size=env.action_space_n
    )

    print(f"\n🔍 Running training with performance monitoring...")

    # Run one episode
    state = env.reset()
    for step in range(min(200, env.max_steps)):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        if len(agent.memory) >= agent.batch_size:
            agent.replay(iteration=step)

        state = next_state
        if done:
            break

    # Get performance stats
    stats = agent.get_performance_stats()

    print(f"\n📊 Performance Statistics:")
    print(f"   Avg batch time: {stats['avg_batch_time_ms']:.2f} ms")
    print(f"   Buffer size: {stats['buffer_size']:,}")
    print(f"   Recent loss: {stats['recent_loss']:.4f}")
    print(f"   Epsilon: {stats['epsilon']:.3f}")

    if stats['avg_memory_usage_gb'] > 0:
        print(f"   GPU memory: {stats['avg_memory_usage_gb']:.2f} GB")


def example_7_different_hardware_scenarios():
    """
    Example 7: Simulating different hardware configurations
    Shows how the agent adapts to various hardware setups
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Hardware Scenarios")
    print("=" * 70)

    scenarios = [
        {
            'name': 'Budget System (Dual-core, 4GB RAM, No GPU)',
            'config': {'model_size': 'tiny', 'batch_size': 16, 'buffer_size': 25000}
        },
        {
            'name': 'Mid-range (i5, 16GB RAM, GTX 1050)',
            'config': {'model_size': 'small', 'batch_size': 64, 'buffer_size': 100000}
        },
        {
            'name': 'High-end (i7, 32GB RAM, RTX 3080)',
            'config': {'model_size': 'large', 'batch_size': 256, 'buffer_size': 250000, 'use_mixed_precision': True}
        },
        {
            'name': 'Enthusiast (Threadripper, 64GB RAM, RTX 4090)',
            'config': {'model_size': 'xlarge', 'batch_size': 512, 'buffer_size': 500000, 'use_mixed_precision': True}
        }
    ]

    print("\n📊 Agent configurations for different hardware:\n")

    for scenario in scenarios:
        print(f"{scenario['name']}:")
        config = scenario['config']
        print(f"   Model: {config['model_size'].upper()}")
        print(f"   Batch: {config['batch_size']}")
        print(f"   Buffer: {config['buffer_size']:,}")
        print(f"   FP16: {'Yes' if config.get('use_mixed_precision') else 'No'}")
        print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("NEXLIFY ADAPTIVE RL AGENT - USAGE EXAMPLES")
    print("=" * 70)
    print("\nThese examples show how to use the adaptive RL agent")
    print("that automatically optimizes for your hardware.\n")

    try:
        example_1_auto_detection()
        example_2_manual_override()
        example_3_hardware_profiling()
        example_4_training_simple()
        example_5_save_load()
        example_6_performance_monitoring()
        example_7_different_hardware_scenarios()

        print("\n" + "=" * 70)
        print("✅ ALL EXAMPLES COMPLETED")
        print("=" * 70)
        print("\nTo train your own model, run:")
        print("   python scripts/train_adaptive_rl_agent.py --episodes 1000")
        print("\nFor help:")
        print("   python scripts/train_adaptive_rl_agent.py --help")
        print()

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
