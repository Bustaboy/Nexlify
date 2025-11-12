#!/usr/bin/env python3
"""
Multi-Mode RL Training Script

Trains a comprehensive RL agent that can trade across multiple modes:
- Spot trading
- Futures (long/short)
- Margin trading
- DeFi operations (liquidity pools, staking, DEX swaps, yield farming)
- Partial position sizing
- Risk management
- Liquidity modeling with slippage
- Comprehensive fee tracking

Features:
- 30 actions (vs 3 in basic agent)
- 31 state features (vs 8 in basic agent)
- GPU-accelerated training
- Supports all optimization profiles
- Comprehensive tracking and reporting
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import numpy as np
from datetime import datetime
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/multi_mode_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from nexlify.strategies.nexlify_enhanced_rl_agent import EnhancedTradingEnvironment
from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
from nexlify.ml.nexlify_optimization_manager import OptimizationProfile


def print_banner():
    """Print training banner"""
    print("\n" + "="*80)
    print("  NEXLIFY MULTI-MODE RL TRAINING")
    print("="*80)
    print("  Comprehensive training across all trading modes")
    print("  Spot | Futures Long/Short | Margin | Position Sizing")
    print("="*80 + "\n")


def fetch_training_data(symbol: str = "BTC/USDT", days: int = 180) -> np.ndarray:
    """
    Fetch historical price data for training

    Args:
        symbol: Trading pair
        days: Number of days

    Returns:
        Price data array
    """
    logger.info(f"📊 Fetching {days} days of {symbol} data...")

    try:
        import ccxt

        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

        now = exchange.milliseconds()
        since = now - (days * 24 * 60 * 60 * 1000)

        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=days * 24)
        prices = np.array([candle[4] for candle in ohlcv])

        logger.info(f"✅ Fetched {len(prices)} price points")
        logger.info(f"   Range: ${prices.min():.2f} - ${prices.max():.2f}")

        return prices

    except Exception as e:
        logger.warning(f"Failed to fetch real data: {e}")
        logger.info("Generating synthetic data...")
        return generate_synthetic_data(days)


def generate_synthetic_data(days: int = 180) -> np.ndarray:
    """Generate synthetic price data with realistic patterns"""
    num_points = days * 24

    # Base price
    base_price = 40000

    # Random walk with trend and volatility
    returns = np.random.normal(0.0002, 0.02, num_points)

    # Add trend
    trend = np.linspace(0, 0.4, num_points)

    # Add cyclical patterns
    cycle1 = 0.1 * np.sin(np.linspace(0, 8 * np.pi, num_points))  # Long cycle
    cycle2 = 0.05 * np.sin(np.linspace(0, 20 * np.pi, num_points))  # Short cycle

    # Construct prices
    prices = base_price * np.exp(np.cumsum(returns) + trend + cycle1 + cycle2)

    logger.info(f"✅ Generated {len(prices)} synthetic price points")
    logger.info(f"   Range: ${prices.min():.2f} - ${prices.max():.2f}")

    return prices


def train_multi_mode_agent(
    num_episodes: int = 100,
    data_days: int = 180,
    initial_balance: float = 10000,
    max_leverage: float = 10.0,
    optimization_profile: OptimizationProfile = OptimizationProfile.BALANCED,
    checkpoint_dir: str = "models/multi_mode",
    save_interval: int = 10
):
    """
    Train multi-mode RL agent

    Args:
        num_episodes: Number of training episodes
        data_days: Days of price data
        initial_balance: Starting capital
        max_leverage: Maximum leverage allowed
        optimization_profile: GPU optimization profile
        checkpoint_dir: Where to save checkpoints
        save_interval: Save every N episodes
    """
    print_banner()

    # Create directories
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Fetch training data
    price_data = fetch_training_data(days=data_days)

    # Save training data
    data_file = checkpoint_path / "training_data.npy"
    np.save(data_file, price_data)
    logger.info(f"💾 Training data saved to {data_file}")

    # Create enhanced environment
    logger.info("\n🌍 CREATING ENHANCED TRADING ENVIRONMENT")
    logger.info("-" * 80)

    env = EnhancedTradingEnvironment(
        price_data=price_data,
        initial_balance=initial_balance,
        max_leverage=max_leverage,
        trading_fee=0.001,  # 0.1%
        funding_rate=0.0001,  # 0.01% per 8h
        margin_interest=0.0002  # 0.02% per day
    )

    logger.info(f"✅ Enhanced environment created")
    logger.info(f"   State space: {env.state_space_n} features")
    logger.info(f"   Action space: {env.action_space_n} actions")
    logger.info(f"   Max steps: {env.max_steps}")
    logger.info(f"   Max leverage: {max_leverage}x")

    # Create GPU-optimized agent
    logger.info("\n🤖 CREATING MULTI-MODE RL AGENT")
    logger.info("-" * 80)

    agent = create_ultra_optimized_agent(
        state_size=env.state_space_n,  # 31 features
        action_size=env.action_space_n,  # 30 actions
        profile=optimization_profile,
        enable_sentiment=False
    )

    logger.info(f"✅ Agent created successfully")
    logger.info(f"   Device: {agent.device}")
    logger.info(f"   Architecture: {agent.architecture}")
    logger.info(f"   Batch size: {agent.batch_size}")
    logger.info(f"   Mixed precision: {'✅' if agent.use_mixed_precision else '❌'}")

    # Training loop
    logger.info(f"\n🚀 STARTING MULTI-MODE TRAINING")
    logger.info("=" * 80)
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Initial balance: ${initial_balance:,.2f}")
    logger.info(f"Optimization: {optimization_profile.value}")
    logger.info("=" * 80 + "\n")

    training_results = {
        'episodes': [],
        'rewards': [],
        'final_values': [],
        'returns': [],
        'num_trades': [],
        'liquidations': [],
        'epsilons': [],
        'losses': [],
        'timestamps': []
    }

    best_return = -float('inf')
    start_time = datetime.now()

    for episode in range(num_episodes):
        episode_start = time.time()
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        num_trades = 0

        for step in range(env.max_steps):
            # Select action
            action = agent.act(state, training=True)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Track trades
            if info.get('trade', False):
                num_trades += 1

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train agent
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        # Update target network
        if episode % 10 == 0:
            agent.update_target_model()

        # Decay epsilon
        agent.epsilon *= agent.epsilon_decay

        # Calculate performance
        perf = env.get_performance_summary()
        final_value = perf['final_value']
        total_return = perf['total_return_%']
        liquidations = perf['liquidations']

        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # Track best model
        if total_return > best_return:
            best_return = total_return
            best_model_path = checkpoint_path / "best_model.pth"
            agent.save(str(best_model_path))
            logger.info(f"🏆 New best model! Return: {total_return:+.2f}%")

        # Record results
        training_results['episodes'].append(episode + 1)
        training_results['rewards'].append(episode_reward)
        training_results['final_values'].append(final_value)
        training_results['returns'].append(total_return)
        training_results['num_trades'].append(num_trades)
        training_results['liquidations'].append(liquidations)
        training_results['epsilons'].append(agent.epsilon)
        training_results['losses'].append(avg_loss)
        training_results['timestamps'].append(datetime.now().isoformat())

        # Log progress
        if (episode + 1) % 10 == 0 or episode == 0:
            episode_time = time.time() - episode_start
            elapsed = datetime.now() - start_time
            eta = (elapsed / (episode + 1)) * (num_episodes - episode - 1)

            avg_return = np.mean(training_results['returns'][-10:])
            avg_trades = np.mean(training_results['num_trades'][-10:])
            total_liquidations = sum(training_results['liquidations'])

            logger.info(f"\n{'='*80}")
            logger.info(f"Episode {episode + 1}/{num_episodes}")
            logger.info(f"{'-'*80}")
            logger.info(f"Current Episode:")
            logger.info(f"  Return: {total_return:+7.2f}%")
            logger.info(f"  Final Value: ${final_value:,.2f}")
            logger.info(f"  Reward: {episode_reward:9.2f}")
            logger.info(f"  Trades: {num_trades:3d}")
            logger.info(f"  Liquidations: {liquidations}")
            logger.info(f"  Loss: {avg_loss:8.4f}")
            logger.info(f"  Epsilon: {agent.epsilon:.4f}")
            logger.info(f"Recent Performance (last 10):")
            logger.info(f"  Avg Return: {avg_return:+7.2f}%")
            logger.info(f"  Avg Trades: {avg_trades:.1f}")
            logger.info(f"Best Performance:")
            logger.info(f"  Best Return: {best_return:+7.2f}%")
            logger.info(f"  Total Liquidations: {total_liquidations}")
            logger.info(f"Progress:")
            logger.info(f"  Elapsed: {str(elapsed).split('.')[0]}")
            logger.info(f"  ETA: {str(eta).split('.')[0]}")
            logger.info(f"  Episode Time: {episode_time:.2f}s")
            logger.info(f"{'='*80}\n")

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_ep{episode + 1}.pth"
            agent.save(str(checkpoint_file))
            logger.info(f"💾 Checkpoint saved: {checkpoint_file.name}\n")

            # Save results
            results_file = checkpoint_path / f"results_ep{episode + 1}.json"
            save_results(training_results, results_file)

    # Training complete
    duration = datetime.now() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("✅ MULTI-MODE TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Duration: {str(duration).split('.')[0]}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Best Return: {best_return:+.2f}%")
    logger.info(f"Final Return: {training_results['returns'][-1]:+.2f}%")
    logger.info(f"Total Liquidations: {sum(training_results['liquidations'])}")
    logger.info(f"Avg Last 10 Returns: {np.mean(training_results['returns'][-10:]):+.2f}%")
    logger.info("=" * 80 + "\n")

    # Save final model
    final_model_path = checkpoint_path / f"final_model_{num_episodes}.pth"
    agent.save(str(final_model_path))
    logger.info(f"💾 Final model saved to {final_model_path}")

    # Save complete results
    results_path = checkpoint_path / f"training_results_{num_episodes}.json"
    save_results(training_results, results_path)

    # Generate report
    generate_report(training_results, checkpoint_path)

    # Cleanup
    agent.shutdown()

    return agent, training_results


def save_results(results: dict, output_path: Path):
    """Save training results to JSON"""
    results_serializable = {
        'episodes': [int(x) for x in results['episodes']],
        'rewards': [float(x) for x in results['rewards']],
        'final_values': [float(x) for x in results['final_values']],
        'returns': [float(x) for x in results['returns']],
        'num_trades': [int(x) for x in results['num_trades']],
        'liquidations': [int(x) for x in results['liquidations']],
        'epsilons': [float(x) for x in results['epsilons']],
        'losses': [float(x) for x in results['losses']],
        'timestamps': results['timestamps']
    }

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"📊 Results saved to {output_path}")


def generate_report(results: dict, output_dir: Path):
    """Generate training report"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        logger.info("\n📈 Generating training report...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Multi-Mode RL Training Report', fontsize=16, fontweight='bold')

        # Returns
        axes[0, 0].plot(results['episodes'], results['returns'], linewidth=2, color='#2E86AB')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Return (%)')
        axes[0, 0].set_title('Returns per Episode')
        axes[0, 0].grid(True, alpha=0.3)

        # Cumulative rewards
        axes[0, 1].plot(results['episodes'], np.cumsum(results['rewards']), color='#06A77D', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Cumulative Reward')
        axes[0, 1].set_title('Cumulative Reward')
        axes[0, 1].grid(True, alpha=0.3)

        # Epsilon decay
        axes[0, 2].plot(results['episodes'], results['epsilons'], color='#F18F01', linewidth=2)
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Epsilon')
        axes[0, 2].set_title('Exploration Rate')
        axes[0, 2].grid(True, alpha=0.3)

        # Moving average returns
        window = min(20, len(results['returns']) // 2)
        if len(results['returns']) >= window:
            moving_avg = np.convolve(results['returns'], np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(results['returns'])), moving_avg,
                          color='#C73E1D', linewidth=2.5)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Return (%)')
            axes[1, 0].set_title(f'Moving Avg Returns (window={window})')
            axes[1, 0].grid(True, alpha=0.3)

        # Trades per episode
        axes[1, 1].plot(results['episodes'], results['num_trades'], color='#1B998B', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Number of Trades')
        axes[1, 1].set_title('Trades per Episode')
        axes[1, 1].grid(True, alpha=0.3)

        # Liquidations
        cumulative_liq = np.cumsum(results['liquidations'])
        axes[1, 2].plot(results['episodes'], cumulative_liq, color='#E63946', linewidth=2)
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Cumulative Liquidations')
        axes[1, 2].set_title('Liquidation Events')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        report_path = output_dir / "training_report.png"
        plt.savefig(report_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Report saved to {report_path}")

    except ImportError:
        logger.warning("matplotlib not available - skipping visual report")
    except Exception as e:
        logger.error(f"Error generating report: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train multi-mode RL agent"
    )

    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--days', type=int, default=180,
                       help='Days of price data (default: 180)')
    parser.add_argument('--balance', type=float, default=10000,
                       help='Initial balance (default: 10000)')
    parser.add_argument('--leverage', type=float, default=10.0,
                       help='Max leverage (default: 10.0)')
    parser.add_argument('--profile', type=str, default='balanced',
                       choices=['auto', 'balanced', 'ultra_low_overhead', 'maximum'],
                       help='Optimization profile (default: balanced)')
    parser.add_argument('--checkpoint-dir', type=str, default='models/multi_mode',
                       help='Checkpoint directory')
    parser.add_argument('--save-interval', type=int, default=10,
                       help='Save checkpoint every N episodes')

    args = parser.parse_args()

    # Map profile string to enum
    profile_map = {
        'auto': OptimizationProfile.AUTO,
        'balanced': OptimizationProfile.BALANCED,
        'ultra_low_overhead': OptimizationProfile.ULTRA_LOW_OVERHEAD,
        'maximum': OptimizationProfile.MAXIMUM_PERFORMANCE
    }

    try:
        agent, results = train_multi_mode_agent(
            num_episodes=args.episodes,
            data_days=args.days,
            initial_balance=args.balance,
            max_leverage=args.leverage,
            optimization_profile=profile_map[args.profile],
            checkpoint_dir=args.checkpoint_dir,
            save_interval=args.save_interval
        )

        print("\n" + "=" * 80)
        print("  ✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"  Models: {args.checkpoint_dir}/")
        print(f"  Best model: {args.checkpoint_dir}/best_model.pth")
        print(f"  Report: {args.checkpoint_dir}/training_report.png")
        print("=" * 80 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
