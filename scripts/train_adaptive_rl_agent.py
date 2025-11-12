#!/usr/bin/env python3
"""
Adaptive RL Agent Training Script for Nexlify
Automatically optimizes training for any consumer hardware configuration

This script:
- Auto-detects hardware (CPU, RAM, GPU VRAM)
- Selects optimal model architecture
- Configures batch sizes and buffer sizes
- Enables mixed precision when available
- Provides real-time performance monitoring
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import numpy as np
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Nexlify modules
try:
    from nexlify.strategies.nexlify_adaptive_rl_agent import (
        create_optimized_agent,
        HardwareProfiler
    )
    from nexlify.strategies.nexlify_rl_agent import TradingEnvironment
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please ensure Nexlify is properly installed")
    sys.exit(1)


def fetch_training_data(symbol: str = "BTC/USDT", days: int = 180) -> np.ndarray:
    """
    Fetch historical price data for training

    Args:
        symbol: Trading pair
        days: Number of days of historical data

    Returns:
        NumPy array of prices
    """
    logger.info(f"📊 Fetching {days} days of {symbol} data...")

    try:
        import ccxt

        # Use Binance as data source
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        # Calculate timestamps
        now = exchange.milliseconds()
        since = now - (days * 24 * 60 * 60 * 1000)

        # Fetch OHLCV data (1 hour candles)
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=days * 24)

        # Extract close prices
        prices = np.array([candle[4] for candle in ohlcv])

        logger.info(f"✅ Fetched {len(prices)} price points")
        return prices

    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Generating synthetic data for training...")
        return generate_synthetic_data(days)


def generate_synthetic_data(days: int = 180) -> np.ndarray:
    """
    Generate synthetic price data for training

    Args:
        days: Number of days

    Returns:
        Synthetic price array
    """
    num_points = days * 24  # Hourly data
    base_price = 30000

    # Random walk with trend and volatility
    returns = np.random.normal(0.0001, 0.02, num_points)
    trend = np.linspace(0, 0.3, num_points)  # 30% uptrend over period

    prices = base_price * np.exp(np.cumsum(returns) + trend)

    logger.info(f"✅ Generated {len(prices)} synthetic price points")
    return prices


def train_adaptive_agent(
    episodes: int = 1000,
    data_days: int = 180,
    initial_balance: float = 10000,
    checkpoint_dir: str = "models/adaptive_rl",
    save_interval: int = 50,
    config_override: dict = None
):
    """
    Train adaptive RL agent with automatic hardware optimization

    Args:
        episodes: Number of training episodes
        data_days: Days of historical data
        initial_balance: Starting capital
        checkpoint_dir: Directory for model checkpoints
        save_interval: Episodes between checkpoints
        config_override: Optional configuration overrides
    """
    logger.info("=" * 70)
    logger.info("NEXLIFY ADAPTIVE RL AGENT TRAINING")
    logger.info("=" * 70)

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Hardware profiling
    logger.info("\n🔍 HARDWARE PROFILING")
    logger.info("-" * 70)
    profiler = HardwareProfiler()

    # Print hardware summary
    hw = profiler.profile
    print(f"\n📊 Hardware Configuration:")
    print(f"   CPU: {hw['cpu']['cores_physical']} cores @ {hw['cpu']['frequency_mhz']:.0f} MHz ({hw['cpu']['tier']})")
    print(f"   RAM: {hw['ram']['total_gb']:.1f} GB total, {hw['ram']['available_gb']:.1f} GB available ({hw['ram']['tier']})")

    if hw['gpu']['available']:
        print(f"   GPU: {hw['gpu']['name']}")
        print(f"        {hw['gpu']['vram_gb']:.1f} GB VRAM ({hw['gpu']['tier']})")
        print(f"        Compute: {hw['gpu']['compute_capability']}")
        print(f"        FP16: {'✅' if hw['gpu']['supports_fp16'] else '❌'}")
    else:
        print(f"   GPU: None (CPU-only mode)")

    config = profiler.optimal_config
    print(f"\n⚙️  Optimal Configuration:")
    print(f"   Model Size: {config['model_size'].upper()}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Buffer Size: {config['buffer_size']:,}")
    print(f"   Mixed Precision: {'✅' if config['use_mixed_precision'] else '❌'}")
    print(f"   Gradient Accumulation: {config['gradient_accumulation_steps']}x")
    print(f"   CPU Workers: {config['num_workers']}")
    print(f"   Optimization Level: {config['optimization_level']}")

    # Save hardware profile
    profile_path = checkpoint_path / "hardware_profile.json"
    with open(profile_path, 'w') as f:
        json.dump(profiler.get_hardware_summary(), f, indent=2)
    logger.info(f"\n💾 Hardware profile saved to {profile_path}")

    # Fetch training data
    logger.info("\n📊 PREPARING TRAINING DATA")
    logger.info("-" * 70)
    price_data = fetch_training_data(days=data_days)

    # Create environment
    env = TradingEnvironment(price_data, initial_balance=initial_balance)
    logger.info(f"✅ Environment created: {env.max_steps} steps per episode")

    # Create adaptive agent
    logger.info("\n🤖 CREATING ADAPTIVE AGENT")
    logger.info("-" * 70)

    agent = create_optimized_agent(
        state_size=env.state_space_n,
        action_size=env.action_space_n,
        auto_detect=True,
        config_override=config_override
    )

    # Training loop
    logger.info("\n🚀 STARTING TRAINING")
    logger.info("-" * 70)
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")

    training_results = {
        'episodes': [],
        'rewards': [],
        'profits': [],
        'epsilons': [],
        'performance_stats': []
    }

    best_profit = -float('inf')
    start_time = datetime.now()

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []

        for step in range(env.max_steps):
            # Select action
            action = agent.act(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay(iteration=step)
                if loss is not None:
                    episode_loss.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_model()

        # Decay epsilon
        agent.decay_epsilon()

        # Calculate profit
        final_value = env.get_portfolio_value()
        profit = final_value - initial_balance
        profit_percent = (profit / initial_balance) * 100

        # Track best model
        if profit > best_profit:
            best_profit = profit
            best_model_path = checkpoint_path / "best_model.pth"
            agent.save(str(best_model_path))

        # Record results
        training_results['episodes'].append(episode + 1)
        training_results['rewards'].append(episode_reward)
        training_results['profits'].append(profit)
        training_results['epsilons'].append(agent.epsilon)

        # Get performance stats
        perf_stats = agent.get_performance_stats()
        training_results['performance_stats'].append(perf_stats)

        # Log progress
        if (episode + 1) % 10 == 0:
            avg_loss = np.mean(episode_loss) if episode_loss else 0
            avg_recent_profit = np.mean(training_results['profits'][-10:])

            logger.info(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Profit: ${profit:+8.2f} ({profit_percent:+6.2f}%) | "
                f"Reward: {episode_reward:7.2f} | "
                f"Loss: {avg_loss:6.4f} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Batch: {perf_stats['avg_batch_time_ms']:.1f}ms"
            )

            if hw['gpu']['available']:
                logger.info(
                    f"           GPU Memory: {perf_stats['avg_memory_usage_gb']:.2f} GB | "
                    f"Buffer: {perf_stats['buffer_size']:,} | "
                    f"Avg-10: ${avg_recent_profit:+.2f}"
                )

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_ep{episode + 1}.pth"
            agent.save(str(checkpoint_file))
            logger.info(f"💾 Checkpoint saved: {checkpoint_file.name}")

    # Training complete
    duration = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Duration: {duration}")
    logger.info(f"Best Profit: ${best_profit:,.2f} ({(best_profit/initial_balance)*100:+.2f}%)")
    logger.info(f"Final Epsilon: {agent.epsilon:.4f}")
    logger.info(f"Total Experiences: {len(agent.memory):,}")

    # Save final model
    final_model_path = checkpoint_path / "final_model.pth"
    agent.save(str(final_model_path))
    logger.info(f"\n💾 Final model saved to {final_model_path}")

    # Save training results
    results_path = checkpoint_path / "training_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        results_serializable = {
            'episodes': [int(x) for x in training_results['episodes']],
            'rewards': [float(x) for x in training_results['rewards']],
            'profits': [float(x) for x in training_results['profits']],
            'epsilons': [float(x) for x in training_results['epsilons']],
            'best_profit': float(best_profit),
            'duration_seconds': duration.total_seconds(),
            'hardware_config': config,
            'training_params': {
                'episodes': episodes,
                'initial_balance': initial_balance,
                'data_days': data_days
            }
        }
        json.dump(results_serializable, f, indent=2)

    logger.info(f"📊 Training results saved to {results_path}")

    # Generate training report
    generate_training_report(training_results, profiler, checkpoint_path)

    return agent, training_results


def generate_training_report(results: dict, profiler: HardwareProfiler, output_dir: Path):
    """
    Generate visual training report

    Args:
        results: Training results dictionary
        profiler: Hardware profiler
        output_dir: Output directory
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        logger.info("\n📈 Generating training report...")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Nexlify Adaptive RL Training Report', fontsize=16, fontweight='bold')

        # Plot 1: Profit over episodes
        axes[0, 0].plot(results['episodes'], results['profits'], linewidth=2)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Profit ($)')
        axes[0, 0].set_title('Profit per Episode')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cumulative reward
        axes[0, 1].plot(results['episodes'], np.cumsum(results['rewards']), color='green', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_title('Cumulative Reward Over Time')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Epsilon decay
        axes[1, 0].plot(results['episodes'], results['epsilons'], color='orange', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate (Epsilon) Decay')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Moving average profit
        window = 50
        if len(results['profits']) >= window:
            moving_avg = np.convolve(results['profits'], np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(results['profits'])), moving_avg, color='purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Profit ($)')
            axes[1, 1].set_title(f'Moving Average Profit (window={window})')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        report_path = output_dir / "training_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Training report saved to {report_path}")

        # Generate text report
        text_report_path = output_dir / "training_summary.txt"
        with open(text_report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("NEXLIFY ADAPTIVE RL TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("HARDWARE CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            hw = profiler.profile
            f.write(f"CPU: {hw['cpu']['cores_physical']} cores @ {hw['cpu']['frequency_mhz']:.0f} MHz ({hw['cpu']['tier']})\n")
            f.write(f"RAM: {hw['ram']['total_gb']:.1f} GB ({hw['ram']['tier']})\n")
            if hw['gpu']['available']:
                f.write(f"GPU: {hw['gpu']['name']} with {hw['gpu']['vram_gb']:.1f} GB VRAM ({hw['gpu']['tier']})\n")
            else:
                f.write("GPU: None (CPU-only)\n")

            config = profiler.optimal_config
            f.write(f"\nMODEL CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Model Size: {config['model_size'].upper()}\n")
            f.write(f"Batch Size: {config['batch_size']}\n")
            f.write(f"Buffer Size: {config['buffer_size']:,}\n")
            f.write(f"Mixed Precision: {'Yes' if config['use_mixed_precision'] else 'No'}\n")
            f.write(f"Optimization: {config['optimization_level']}\n")

            f.write(f"\nTRAINING RESULTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Episodes: {len(results['episodes'])}\n")
            f.write(f"Best Profit: ${max(results['profits']):,.2f}\n")
            f.write(f"Average Profit: ${np.mean(results['profits']):,.2f}\n")
            f.write(f"Final Profit: ${results['profits'][-1]:,.2f}\n")
            f.write(f"Final Epsilon: {results['epsilons'][-1]:.4f}\n")

        logger.info(f"✅ Text summary saved to {text_report_path}")

    except ImportError:
        logger.warning("matplotlib not available - skipping visual report")
    except Exception as e:
        logger.error(f"Error generating report: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Nexlify Adaptive RL Agent with automatic hardware optimization"
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Number of training episodes (default: 1000)'
    )

    parser.add_argument(
        '--data-days',
        type=int,
        default=180,
        help='Days of historical data (default: 180)'
    )

    parser.add_argument(
        '--balance',
        type=float,
        default=10000,
        help='Initial balance (default: 10000)'
    )

    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='models/adaptive_rl',
        help='Checkpoint directory (default: models/adaptive_rl)'
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=50,
        help='Episodes between checkpoints (default: 50)'
    )

    parser.add_argument(
        '--model-size',
        type=str,
        choices=['tiny', 'small', 'medium', 'large', 'xlarge'],
        help='Force specific model size (overrides auto-detection)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        help='Force specific batch size (overrides auto-detection)'
    )

    args = parser.parse_args()

    # Build config override
    config_override = {}
    if args.model_size:
        config_override['model_size'] = args.model_size
        logger.info(f"⚠️  Forcing model size: {args.model_size}")

    if args.batch_size:
        config_override['batch_size'] = args.batch_size
        logger.info(f"⚠️  Forcing batch size: {args.batch_size}")

    # Train agent
    try:
        agent, results = train_adaptive_agent(
            episodes=args.episodes,
            data_days=args.data_days,
            initial_balance=args.balance,
            checkpoint_dir=args.checkpoint_dir,
            save_interval=args.save_interval,
            config_override=config_override if config_override else None
        )

        logger.info("\n✅ Training completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
