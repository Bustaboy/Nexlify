#!/usr/bin/env python3
"""
Simple GPU Training Example

Demonstrates how to use GPU-accelerated training with Nexlify's
ultra-optimized RL agent.

Features:
- Automatic GPU detection
- Hardware-aware optimization
- CPU fallback (backward compatibility)
- Mixed precision training
- Thermal monitoring

Usage:
    python examples/gpu_training_example.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main example"""

    print("\n" + "="*80)
    print("  NEXLIFY GPU TRAINING EXAMPLE")
    print("="*80 + "\n")

    # Import Nexlify modules
    from nexlify.strategies.nexlify_ultra_optimized_rl_agent import (
        create_ultra_optimized_agent
    )
    from nexlify.ml.nexlify_optimization_manager import OptimizationProfile
    from nexlify.strategies.nexlify_rl_agent import TradingEnvironment

    # Step 1: Create training data
    print("📊 Step 1: Creating training data...")
    price_data = np.cumsum(np.random.randn(1000)) + 40000  # Synthetic BTC prices
    print(f"   Generated {len(price_data)} price points")
    print(f"   Price range: ${price_data.min():.2f} - ${price_data.max():.2f}\n")

    # Step 2: Create environment
    print("🌍 Step 2: Creating trading environment...")
    env = TradingEnvironment(price_data, initial_balance=10000)
    print(f"   State space: {env.state_space_n}")
    print(f"   Action space: {env.action_space_n}")
    print(f"   Max steps: {env.max_steps}\n")

    # Step 3: Create GPU-optimized agent
    print("🤖 Step 3: Creating GPU-optimized agent...")
    print("   Using AUTO profile (automatically selects best optimizations)")

    agent = create_ultra_optimized_agent(
        state_size=env.state_space_n,
        action_size=env.action_space_n,
        profile=OptimizationProfile.BALANCED,  # or AUTO for automatic optimization
        enable_sentiment=False  # Set to True if you have API keys
    )

    print(f"\n   ✅ Agent created!")
    print(f"   Device: {agent.device}")
    print(f"   Architecture: {agent.architecture}")
    print(f"   Batch Size: {agent.batch_size}")
    print(f"   Mixed Precision: {'✅ Enabled' if agent.use_mixed_precision else '❌ Disabled'}")

    # Display GPU info if available
    gpu_info = agent.monitor.get_gpu_info_summary()
    if gpu_info['available']:
        print(f"\n   🎮 GPU Information:")
        print(f"   Name: {gpu_info['name']}")
        print(f"   VRAM: {gpu_info['vram_gb']:.1f} GB")
        print(f"   Tensor Cores: {'✅' if gpu_info['has_tensor_cores'] else '❌'}")
    else:
        print(f"\n   ⚠️  No GPU detected - using CPU")

    # Step 4: Train for a few episodes
    print(f"\n🎯 Step 4: Training for 5 episodes...")

    num_episodes = 5
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        losses = []

        for step in range(env.max_steps):
            # Select action
            action = agent.act(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Update target network
        if episode % 5 == 0:
            agent.update_target_model()

        # Decay epsilon
        agent.epsilon *= agent.epsilon_decay

        # Calculate profit
        final_value = env.get_portfolio_value()
        profit = final_value - 10000
        profit_pct = (profit / 10000) * 100

        avg_loss = np.mean(losses) if losses else 0

        print(f"\n   Episode {episode + 1}/{num_episodes}:")
        print(f"     Profit: ${profit:+.2f} ({profit_pct:+.2f}%)")
        print(f"     Reward: {episode_reward:.2f}")
        print(f"     Loss: {avg_loss:.6f}")
        print(f"     Epsilon: {agent.epsilon:.4f}")
        print(f"     Memory: {len(agent.memory):,} experiences")

    # Step 5: Get statistics
    print(f"\n📊 Step 5: Agent Statistics")
    stats = agent.get_statistics()

    print(f"   Training steps: {stats['training_steps']}")
    print(f"   Memory size: {stats['memory_size']}")
    print(f"   Architecture: {stats['architecture']}")
    print(f"   Device: {stats['device']}")

    if 'gpu_name' in stats:
        print(f"   GPU: {stats['gpu_name']} ({stats['gpu_vram']})")
        if 'gpu_temp' in stats and stats['gpu_temp']:
            print(f"   GPU Temperature: {stats['gpu_temp']:.1f}°C")

    # Step 6: Save model
    print(f"\n💾 Step 6: Saving model...")
    model_dir = Path("models/examples")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "gpu_example_model.pth"
    agent.save(str(model_path))
    print(f"   ✅ Model saved to {model_path}")

    # Cleanup
    print(f"\n🧹 Cleaning up...")
    agent.shutdown()

    print("\n" + "="*80)
    print("  ✅ GPU TRAINING EXAMPLE COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("  1. GPU training is enabled automatically if GPU is available")
    print("  2. CPU fallback works seamlessly if no GPU is detected")
    print("  3. Mixed precision training is enabled on compatible GPUs")
    print("  4. Thermal monitoring prevents GPU throttling")
    print("  5. Architecture scales based on available VRAM")
    print("\nTo use GPU training in your own code:")
    print("  - Import: from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent")
    print("  - Create: agent = create_ultra_optimized_agent(state_size, action_size)")
    print("  - Train: Use agent.act(), agent.remember(), agent.replay() as usual")
    print("  - That's it! GPU acceleration is automatic.")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
