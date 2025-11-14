#!/usr/bin/env python3
"""
Nexlify ML/RL 1000-Round Training Script
========================================
Comprehensive training script for 1000 episodes with hardware optimization,
detailed tracking, and comprehensive reporting.

Features:
- Automatic hardware detection and optimization
- 1000 training episodes with progress tracking
- Checkpointing every 50 episodes
- Comprehensive performance metrics
- Visual and text reports
- Best model tracking
- Resume from checkpoint support
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import Tuple
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    # Try to set UTF-8 encoding for Windows console
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_rl_1000_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import Nexlify modules
try:
    from nexlify.strategies.nexlify_adaptive_rl_agent import (
        create_optimized_agent,
        HardwareProfiler
    )
    from nexlify.strategies.nexlify_rl_agent import TradingEnvironment
    ADAPTIVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Adaptive agent not available: {e}")
    ADAPTIVE_AVAILABLE = False

# Try to import ultra-optimized agent
try:
    from nexlify.strategies.nexlify_ultra_optimized_rl_agent import (
        UltraOptimizedDQNAgent,
        UltraOptimizedConfig
    )
    ULTRA_AVAILABLE = True
except ImportError:
    logger.warning("Ultra-optimized agent not available")
    ULTRA_AVAILABLE = False

# Import validation and early stopping
try:
    from nexlify.training.validation_monitor import (
        ValidationMonitor,
        ValidationDataSplitter
    )
    from nexlify.training.early_stopping import (
        EarlyStopping,
        EarlyStoppingConfig,
        TrainingPhaseDetector,
        OverfittingDetector
    )
    VALIDATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Validation and early stopping not available: {e}")
    VALIDATION_AVAILABLE = False


def print_banner():
    """Print training banner"""
    print("\n" + "="*80)
    print("  🚀 NEXLIFY ML/RL 1000-ROUND TRAINING SCRIPT")
    print("="*80)
    print("  Training Episodes: 1000")
    print("  Hardware Optimization: Auto")
    print("  Checkpointing: Every 50 episodes")
    print("  Best Model Tracking: Enabled")
    print("="*80 + "\n")


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
        logger.info(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
        logger.info(f"   Average price: ${prices.mean():.2f}")

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
    base_price = 40000

    # Random walk with trend and volatility
    returns = np.random.normal(0.0002, 0.02, num_points)
    trend = np.linspace(0, 0.4, num_points)  # 40% uptrend over period

    # Add some cycles
    cycle = 0.1 * np.sin(np.linspace(0, 8 * np.pi, num_points))

    prices = base_price * np.exp(np.cumsum(returns) + trend + cycle)

    logger.info(f"✅ Generated {len(prices)} synthetic price points")
    logger.info(f"   Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    logger.info(f"   Average price: ${prices.mean():.2f}")

    return prices


def train_1000_rounds(
    agent_type: str = "adaptive",
    data_days: int = 180,
    initial_balance: float = 10000,
    checkpoint_dir: str = "models/ml_rl_1000",
    resume_from: str = None,
    symbol: str = "BTC/USDT",
    # Validation parameters
    use_validation: bool = True,
    validation_frequency: int = 50,
    validation_metric: str = 'val_sharpe',
    train_val_test_split: Tuple = (0.70, 0.15, 0.15),
    # Early stopping parameters
    use_early_stopping: bool = True,
    early_stopping_patience: int = 30,
    early_stopping_min_delta: float = 0.01,
    early_stopping_mode: str = 'max'
):
    """
    Train ML/RL agent for exactly 1000 rounds with validation and early stopping

    Args:
        agent_type: Type of agent ('adaptive', 'ultra', 'basic')
        data_days: Days of historical data
        initial_balance: Starting capital
        checkpoint_dir: Directory for model checkpoints
        resume_from: Path to checkpoint to resume from
        symbol: Trading symbol
        use_validation: Enable validation monitoring (default: True)
        validation_frequency: Run validation every N episodes (default: 50)
        validation_metric: Metric to track ('val_sharpe', 'val_return', 'val_win_rate')
        train_val_test_split: Train/val/test split ratios (default: 0.70, 0.15, 0.15)
        use_early_stopping: Enable early stopping (default: True)
        early_stopping_patience: Episodes without improvement before stopping (default: 30)
        early_stopping_min_delta: Minimum improvement threshold (default: 0.01)
        early_stopping_mode: 'max' for metrics to maximize, 'min' for minimize
    """
    TOTAL_EPISODES = 1000
    SAVE_INTERVAL = 50

    print_banner()

    # Create checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Create logs directory if needed
    Path("logs").mkdir(exist_ok=True)

    logger.info(f"📁 Checkpoint directory: {checkpoint_path}")

    # Hardware profiling
    if ADAPTIVE_AVAILABLE:
        logger.info("\n🔍 HARDWARE PROFILING")
        logger.info("-" * 80)
        profiler = HardwareProfiler()

        # Print hardware summary
        hw = profiler.profile
        print(f"\n📊 Hardware Configuration:")
        print(f"   CPU: {hw['cpu']['cores_physical']} physical cores, "
              f"{hw['cpu']['cores_logical']} logical @ {hw['cpu']['frequency_mhz']:.0f} MHz")
        print(f"        Tier: {hw['cpu']['tier'].upper()}")
        print(f"   RAM: {hw['ram']['total_gb']:.1f} GB total, "
              f"{hw['ram']['available_gb']:.1f} GB available")
        print(f"        Tier: {hw['ram']['tier'].upper()}")

        if hw['gpu']['available']:
            print(f"   GPU: {hw['gpu']['name']}")
            print(f"        VRAM: {hw['gpu']['vram_gb']:.1f} GB ({hw['gpu']['tier'].upper()})")
            print(f"        Compute: {hw['gpu'].get('compute_capability', 'Unknown')}")
            fp16_support = '✅' if hw['gpu'].get('supports_fp16', False) else '❌'
            tensor_cores = '✅' if hw['gpu'].get('has_tensor_cores', False) else '❌'
            print(f"        Features: FP16={fp16_support}, Tensor Cores={tensor_cores}")
        else:
            print(f"   GPU: Not available (CPU-only mode)")

        config = profiler.optimal_config
        print(f"\n⚙️  Optimal Training Configuration:")
        print(f"   Model Size: {config['model_size'].upper()}")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Buffer Size: {config['buffer_size']:,}")
        print(f"   Mixed Precision: {'✅ Enabled' if config['use_mixed_precision'] else '❌ Disabled'}")
        print(f"   Gradient Accumulation: {config['gradient_accumulation_steps']}x")
        print(f"   CPU Workers: {config['num_workers']}")
        print(f"   Optimization Level: {config['optimization_level'].upper()}")

        # Save hardware profile
        profile_path = checkpoint_path / "hardware_profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profiler.get_hardware_summary(), f, indent=2)
        logger.info(f"\n💾 Hardware profile saved to {profile_path}")

    # Fetch training data
    logger.info("\n📊 PREPARING TRAINING DATA")
    logger.info("-" * 80)
    price_data = fetch_training_data(symbol=symbol, days=data_days)

    # Save price data for reproducibility
    data_file = checkpoint_path / "training_data.npy"
    np.save(data_file, price_data)
    logger.info(f"💾 Price data saved to {data_file}")

    # Split data into train/val/test if validation enabled
    val_env = None
    test_data = None
    validation_monitor = None
    early_stopping = None

    if use_validation and VALIDATION_AVAILABLE:
        logger.info("\n🔍 VALIDATION SETUP")
        logger.info("-" * 80)

        # Split data temporally
        splitter = ValidationDataSplitter(
            train_ratio=train_val_test_split[0],
            val_ratio=train_val_test_split[1],
            test_ratio=train_val_test_split[2]
        )

        data_split = splitter.split(price_data)

        # Use training data for main environment
        train_data = data_split.train_data
        val_data = data_split.val_data
        test_data = data_split.test_data

        logger.info(f"✅ Data split completed:")
        logger.info(f"   Train: {len(train_data)} samples")
        logger.info(f"   Val: {len(val_data)} samples")
        logger.info(f"   Test: {len(test_data)} samples (held out for final evaluation)")

        # Create validation environment
        val_env = TradingEnvironment(val_data, initial_balance=initial_balance)
        logger.info(f"✅ Validation environment created: {val_env.max_steps} steps per episode")

        # Initialize validation monitor
        validation_monitor = ValidationMonitor(
            validation_frequency=validation_frequency,
            save_dir=checkpoint_path / "validation",
            cache_results=True
        )

        # Initialize early stopping
        if use_early_stopping:
            early_stopping_config = EarlyStoppingConfig(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode=early_stopping_mode,
                metric=validation_metric,
                restore_best_weights=True,
                save_best_model=True,
                model_save_path=str(checkpoint_path / "best_model.pth")
            )

            phase_detector = TrainingPhaseDetector()
            overfitting_detector = OverfittingDetector(
                overfitting_threshold=0.20,  # 20% difference threshold
                window_size=10
            )

            early_stopping = EarlyStopping(
                config=early_stopping_config,
                phase_detector=phase_detector,
                overfitting_detector=overfitting_detector
            )

            logger.info(f"✅ Early stopping configured:")
            logger.info(f"   Metric: {validation_metric}")
            logger.info(f"   Patience: {early_stopping_patience} (adaptive by phase)")
            logger.info(f"   Min delta: {early_stopping_min_delta}")

        # Use training split for training
        price_data_to_use = train_data
    else:
        # Use full dataset if validation disabled
        price_data_to_use = price_data
        logger.info("⚠️  Validation disabled - using full dataset for training")

    # Create training environment
    env = TradingEnvironment(price_data_to_use, initial_balance=initial_balance)
    logger.info(f"✅ Training environment created: {env.max_steps} steps per episode")

    # Create agent based on type
    logger.info("\n🤖 CREATING ML/RL AGENT")
    logger.info("-" * 80)

    if agent_type == "adaptive" and ADAPTIVE_AVAILABLE:
        logger.info("Using Adaptive RL Agent with hardware optimization")
        agent = create_optimized_agent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            auto_detect=True
        )
    elif agent_type == "ultra" and ULTRA_AVAILABLE:
        logger.info("Using Ultra-Optimized RL Agent")
        ultra_config = UltraOptimizedConfig(mode='auto')
        agent = UltraOptimizedDQNAgent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            config=ultra_config
        )
    else:
        logger.info("Using Basic DQN Agent")
        from nexlify.strategies.nexlify_rl_agent import DQNAgent
        agent = DQNAgent(
            state_size=env.state_space_n,
            action_size=env.action_space_n
        )

    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from and Path(resume_from).exists():
        logger.info(f"📂 Resuming from checkpoint: {resume_from}")
        agent.load(resume_from)
        # Extract episode number from filename if possible
        try:
            import re
            match = re.search(r'ep(\d+)', resume_from)
            if match:
                start_episode = int(match.group(1))
                logger.info(f"   Starting from episode {start_episode}")
        except:
            pass

    # Training loop
    logger.info("\n🚀 STARTING 1000-ROUND TRAINING")
    logger.info("=" * 80)
    logger.info(f"Total Episodes: {TOTAL_EPISODES}")
    logger.info(f"Starting Episode: {start_episode + 1}")
    logger.info(f"Initial Balance: ${initial_balance:,.2f}")
    logger.info(f"Symbol: {symbol}")
    logger.info("=" * 80 + "\n")

    training_results = {
        'episodes': [],
        'rewards': [],
        'profits': [],
        'profit_percentages': [],
        'epsilons': [],
        'losses': [],
        'trades': [],
        'win_rates': [],
        'timestamps': []
    }

    best_profit = -float('inf')
    best_profit_pct = -float('inf')
    start_time = datetime.now()

    for episode in range(start_episode, TOTAL_EPISODES):
        episode_start = time.time()
        state = env.reset()
        episode_reward = 0
        episode_losses = []
        episode_trades = []

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
                    episode_losses.append(loss)

            episode_reward += reward
            state = next_state

            # Track trades
            if 'trade' in info and info['trade']:
                episode_trades.append(info)

            if done:
                break

        # Update target network periodically
        if episode % 10 == 0:
            agent.update_target_model()

        # Decay epsilon
        agent.decay_epsilon()

        # Calculate metrics
        final_value = env.get_portfolio_value()
        profit = final_value - initial_balance
        profit_percent = (profit / initial_balance) * 100

        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # Calculate win rate
        winning_trades = sum(1 for t in episode_trades if t.get('profit', 0) > 0)
        win_rate = (winning_trades / len(episode_trades) * 100) if episode_trades else 0

        # Calculate training Sharpe ratio for comparison with validation
        if len(env.equity_curve) > 1:
            train_returns = np.diff(env.equity_curve) / env.equity_curve[:-1]
            train_sharpe = (
                np.mean(train_returns) / np.std(train_returns) * np.sqrt(252)
                if np.std(train_returns) > 0 else 0
            )
        else:
            train_sharpe = 0

        # Track best model (training profit)
        if profit_percent > best_profit_pct:
            best_profit_pct = profit_percent
            best_profit = profit
            best_model_path = checkpoint_path / "best_model.pth"
            agent.save(str(best_model_path))
            logger.info(f"🏆 New best model! Profit: ${profit:+,.2f} ({profit_percent:+.2f}%)")

        # Record results
        training_results['episodes'].append(episode + 1)
        training_results['rewards'].append(episode_reward)
        training_results['profits'].append(profit)
        training_results['profit_percentages'].append(profit_percent)
        training_results['epsilons'].append(agent.epsilon)
        training_results['losses'].append(avg_loss)
        training_results['trades'].append(len(episode_trades))
        training_results['win_rates'].append(win_rate)
        training_results['timestamps'].append(datetime.now().isoformat())

        # Run validation if enabled
        should_stop = False
        if validation_monitor and val_env:
            if validation_monitor.should_validate(episode + 1):
                # Run validation
                val_result = validation_monitor.run_validation(
                    agent=agent,
                    val_env=val_env,
                    current_episode=episode + 1,
                    num_episodes=5  # Average over 5 validation episodes
                )

                # Update best validation
                is_new_best = validation_monitor.update_best(val_result, metric=validation_metric)

                # Early stopping check
                if early_stopping and use_early_stopping:
                    # Get model weights for potential restoration
                    model_weights = None
                    if hasattr(agent, 'model') and hasattr(agent.model, 'state_dict'):
                        model_weights = agent.model.state_dict()
                    elif hasattr(agent, 'get_weights'):
                        model_weights = agent.get_weights()

                    # Get validation metric value
                    val_metric_value = getattr(val_result, validation_metric)

                    # Get training metric for overfitting detection
                    if validation_metric == 'val_sharpe':
                        train_metric_value = train_sharpe
                    elif validation_metric == 'val_return_pct':
                        train_metric_value = profit_percent
                    elif validation_metric == 'val_win_rate':
                        train_metric_value = win_rate
                    else:
                        train_metric_value = None

                    # Update early stopping
                    should_stop = early_stopping.update(
                        metric_value=val_metric_value,
                        episode=episode + 1,
                        epsilon=agent.epsilon,
                        model_weights=model_weights,
                        train_metric=train_metric_value
                    )

                    # Save checkpoint if this is the best episode
                    if is_new_best and early_stopping.should_save_checkpoint(episode + 1):
                        best_val_checkpoint = checkpoint_path / "best_validation_model.pth"
                        agent.save(str(best_val_checkpoint))
                        logger.info(f"💾 Best validation model saved to {best_val_checkpoint.name}")

        # Break if early stopping triggered
        if should_stop:
            logger.info(f"\n🛑 Training stopped early at episode {episode + 1}")
            logger.info(f"   Best validation episode was {early_stopping.best_episode}")

            # Restore best weights if configured
            if early_stopping.config.restore_best_weights:
                early_stopping.restore_best_weights(agent)

            break

        # Calculate ETA
        episode_time = time.time() - episode_start
        episodes_remaining = TOTAL_EPISODES - (episode + 1)
        eta_seconds = episode_time * episodes_remaining
        eta_minutes = eta_seconds / 60

        # Log progress
        if (episode + 1) % 10 == 0:
            avg_recent_reward = np.mean(training_results['rewards'][-10:])
            avg_recent_profit = np.mean(training_results['profits'][-10:])
            avg_recent_profit_pct = np.mean(training_results['profit_percentages'][-10:])
            avg_recent_win_rate = np.mean(training_results['win_rates'][-10:])

            elapsed = datetime.now() - start_time
            progress = ((episode + 1) / TOTAL_EPISODES) * 100

            logger.info(f"\n{'='*80}")
            logger.info(f"Episode {episode + 1}/{TOTAL_EPISODES} ({progress:.1f}% complete)")
            logger.info(f"{'-'*80}")
            logger.info(f"Current Episode:")
            logger.info(f"  Profit: ${profit:+9.2f} ({profit_percent:+7.2f}%)")
            logger.info(f"  Reward: {episode_reward:9.2f}")
            logger.info(f"  Loss: {avg_loss:8.4f}")
            logger.info(f"  Trades: {len(episode_trades):3d} (Win Rate: {win_rate:.1f}%)")
            logger.info(f"  Epsilon: {agent.epsilon:.4f}")
            logger.info(f"Recent Performance (last 10 episodes):")
            logger.info(f"  Avg Profit: ${avg_recent_profit:+9.2f} ({avg_recent_profit_pct:+7.2f}%)")
            logger.info(f"  Avg Reward: {avg_recent_reward:9.2f}")
            logger.info(f"  Avg Win Rate: {avg_recent_win_rate:.1f}%")
            logger.info(f"Best Performance:")
            logger.info(f"  Best Profit: ${best_profit:+9.2f} ({best_profit_pct:+7.2f}%)")
            logger.info(f"Progress:")
            logger.info(f"  Elapsed: {str(elapsed).split('.')[0]}")
            logger.info(f"  ETA: {eta_minutes:.1f} minutes ({eta_minutes/60:.1f} hours)")
            logger.info(f"  Ep Time: {episode_time:.2f}s")

            # Get performance stats if available
            if hasattr(agent, 'get_performance_stats'):
                perf_stats = agent.get_performance_stats()
                logger.info(f"System Performance:")
                logger.info(f"  Batch Time: {perf_stats.get('avg_batch_time_ms', 0):.1f}ms")
                logger.info(f"  Memory Usage: {perf_stats.get('avg_memory_usage_gb', 0):.2f} GB")
                logger.info(f"  Buffer Size: {perf_stats.get('buffer_size', 0):,}")

            logger.info(f"{'='*80}\n")

        # Save checkpoint
        if (episode + 1) % SAVE_INTERVAL == 0:
            checkpoint_file = checkpoint_path / f"checkpoint_ep{episode + 1}.pth"
            agent.save(str(checkpoint_file))
            logger.info(f"💾 Checkpoint saved: {checkpoint_file.name}\n")

            # Save intermediate results
            results_file = checkpoint_path / f"results_ep{episode + 1}.json"
            save_results(training_results, results_file)

    # Training complete
    duration = datetime.now() - start_time
    actual_episodes = len(training_results['episodes'])

    logger.info("\n" + "=" * 80)
    logger.info(f"✅ TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total Episodes: {actual_episodes}/{TOTAL_EPISODES}")
    logger.info(f"Total Duration: {str(duration).split('.')[0]}")
    logger.info(f"Average Episode Time: {duration.total_seconds() / actual_episodes:.2f}s")

    # Early stopping summary
    if early_stopping and early_stopping.stopped:
        logger.info(f"\n🛑 Early Stopping Summary:")
        early_stop_summary = early_stopping.get_summary()
        logger.info(f"   Stopped at Episode: {early_stop_summary['stop_episode']}")
        logger.info(f"   Best Episode: {early_stop_summary['best_episode']}")
        logger.info(f"   Best {validation_metric}: {early_stop_summary['best_metric']:.4f}")
        logger.info(f"   Episodes Saved: {early_stop_summary['episodes_saved']}")

        if 'overfitting' in early_stop_summary:
            ovf_summary = early_stop_summary['overfitting']
            if ovf_summary:
                logger.info(f"   Overfitting Score: {ovf_summary.get('avg_overfitting_score', 0)*100:.1f}%")
                logger.info(f"   Chronic Overfitting: {'YES' if ovf_summary.get('chronic_overfitting_detected') else 'NO'}")

    logger.info(f"\n📈 Training Results:")
    logger.info(f"  Best Profit: ${best_profit:+,.2f} ({best_profit_pct:+.2f}%)")
    logger.info(f"  Final Profit: ${training_results['profits'][-1]:+,.2f} "
                f"({training_results['profit_percentages'][-1]:+.2f}%)")
    logger.info(f"  Final Epsilon: {agent.epsilon:.4f}")
    logger.info(f"  Total Experiences: {len(agent.memory):,}")

    if actual_episodes >= 100:
        logger.info(f"  Avg Last 100 Episodes Profit: "
                    f"${np.mean(training_results['profits'][-100:]):+,.2f}")

    # Validation summary
    if validation_monitor and validation_monitor.validation_results:
        logger.info(f"\n🔍 Validation Summary:")
        val_summary = validation_monitor.get_metrics_summary()
        logger.info(f"  Total Validations: {val_summary['num_validations']}")
        logger.info(f"  Best Validation Sharpe: {val_summary['best_val_sharpe']:.3f} (Episode {val_summary['best_episode']})")
        logger.info(f"  Best Validation Return: {val_summary['best_val_return_pct']:+.2f}%")
        logger.info(f"  Avg Validation Sharpe: {val_summary['avg_val_sharpe']:.3f}")
        logger.info(f"  Avg Validation Return: {val_summary['avg_val_return_pct']:+.2f}%")
        logger.info(f"  Avg Win Rate: {val_summary['avg_win_rate']:.1f}%")

    logger.info("=" * 80 + "\n")

    # Save final model
    final_model_path = checkpoint_path / "final_model_1000.pth"
    agent.save(str(final_model_path))
    logger.info(f"💾 Final model saved to {final_model_path}")

    # Save complete training results
    results_path = checkpoint_path / "training_results_1000.json"
    save_results(training_results, results_path, duration, best_profit, best_profit_pct)

    # Generate validation report if available
    if validation_monitor and validation_monitor.validation_results:
        logger.info("\n📊 Generating validation report...")
        validation_monitor.generate_report(checkpoint_path / "validation_report.txt")

        # Save validation history as CSV for analysis
        val_history = validation_monitor.get_validation_history()
        val_history.to_csv(checkpoint_path / "validation_history.csv", index=False)
        logger.info(f"💾 Validation history saved to {checkpoint_path / 'validation_history.csv'}")

    # Generate early stopping plot if available
    if early_stopping and early_stopping.metric_history:
        logger.info("\n📈 Generating early stopping plot...")
        early_stopping.plot_metric_history(checkpoint_path / "early_stopping_plot.png")

    # Generate comprehensive report
    generate_training_report(training_results, checkpoint_path, hw if ADAPTIVE_AVAILABLE else None)

    return agent, training_results


def save_results(results: dict, output_path: Path, duration=None, best_profit=None, best_profit_pct=None):
    """Save training results to JSON"""
    results_serializable = {
        'episodes': [int(x) for x in results['episodes']],
        'rewards': [float(x) for x in results['rewards']],
        'profits': [float(x) for x in results['profits']],
        'profit_percentages': [float(x) for x in results['profit_percentages']],
        'epsilons': [float(x) for x in results['epsilons']],
        'losses': [float(x) for x in results['losses']],
        'trades': [int(x) for x in results['trades']],
        'win_rates': [float(x) for x in results['win_rates']],
        'timestamps': results['timestamps']
    }

    if duration:
        results_serializable['duration_seconds'] = duration.total_seconds()
        results_serializable['duration_str'] = str(duration).split('.')[0]

    if best_profit is not None:
        results_serializable['best_profit'] = float(best_profit)
        results_serializable['best_profit_pct'] = float(best_profit_pct)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    logger.info(f"📊 Results saved to {output_path}")


def generate_training_report(results: dict, output_dir: Path, hardware_profile=None):
    """Generate comprehensive visual training report"""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        logger.info("\n📈 Generating comprehensive training report...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Main title
        fig.suptitle('Nexlify ML/RL 1000-Round Training Report',
                     fontsize=18, fontweight='bold')

        # Plot 1: Profit over episodes
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['episodes'], results['profit_percentages'],
                linewidth=2, color='#2E86AB', alpha=0.7)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Episode', fontsize=10)
        ax1.set_ylabel('Profit (%)', fontsize=10)
        ax1.set_title('Profit % per Episode', fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cumulative reward
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_rewards = np.cumsum(results['rewards'])
        ax2.plot(results['episodes'], cumulative_rewards,
                color='#06A77D', linewidth=2)
        ax2.set_xlabel('Episode', fontsize=10)
        ax2.set_ylabel('Cumulative Reward', fontsize=10)
        ax2.set_title('Cumulative Reward', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Epsilon decay
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(results['episodes'], results['epsilons'],
                color='#F18F01', linewidth=2)
        ax3.set_xlabel('Episode', fontsize=10)
        ax3.set_ylabel('Epsilon', fontsize=10)
        ax3.set_title('Exploration Rate (Epsilon)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Moving average profit (50 episode window)
        ax4 = fig.add_subplot(gs[1, 0])
        window = 50
        if len(results['profit_percentages']) >= window:
            moving_avg = np.convolve(results['profit_percentages'],
                                    np.ones(window)/window, mode='valid')
            ax4.plot(range(window-1, len(results['profit_percentages'])),
                    moving_avg, color='#C73E1D', linewidth=2.5)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('Episode', fontsize=10)
            ax4.set_ylabel('Profit (%)', fontsize=10)
            ax4.set_title(f'Moving Avg Profit % (window={window})', fontweight='bold')
            ax4.grid(True, alpha=0.3)

        # Plot 5: Training loss
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results['episodes'], results['losses'],
                color='#6A4C93', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Episode', fontsize=10)
        ax5.set_ylabel('Loss', fontsize=10)
        ax5.set_title('Training Loss', fontweight='bold')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Win rate
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(results['episodes'], results['win_rates'],
                color='#1B998B', linewidth=2, alpha=0.7)
        ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax6.set_xlabel('Episode', fontsize=10)
        ax6.set_ylabel('Win Rate (%)', fontsize=10)
        ax6.set_title('Trade Win Rate', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Plot 7: Number of trades
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(results['episodes'], results['trades'],
                color='#E63946', linewidth=1.5, alpha=0.7)
        ax7.set_xlabel('Episode', fontsize=10)
        ax7.set_ylabel('Number of Trades', fontsize=10)
        ax7.set_title('Trades per Episode', fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Plot 8: Profit distribution histogram
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(results['profit_percentages'], bins=50,
                color='#457B9D', alpha=0.7, edgecolor='black')
        ax8.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax8.axvline(x=np.mean(results['profit_percentages']),
                   color='green', linestyle='--', linewidth=2, label='Mean')
        ax8.set_xlabel('Profit (%)', fontsize=10)
        ax8.set_ylabel('Frequency', fontsize=10)
        ax8.set_title('Profit Distribution', fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')

        # Plot 9: Reward distribution histogram
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.hist(results['rewards'], bins=50,
                color='#A8DADC', alpha=0.7, edgecolor='black')
        ax9.axvline(x=np.mean(results['rewards']),
                   color='red', linestyle='--', linewidth=2, label='Mean')
        ax9.set_xlabel('Reward', fontsize=10)
        ax9.set_ylabel('Frequency', fontsize=10)
        ax9.set_title('Reward Distribution', fontweight='bold')
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')

        # Save figure
        report_path = output_dir / "training_report_1000.png"
        plt.savefig(report_path, dpi=200, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Visual report saved to {report_path}")

        # Generate text summary
        text_report_path = output_dir / "training_summary_1000.txt"
        with open(text_report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NEXLIFY ML/RL 1000-ROUND TRAINING SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            if hardware_profile:
                f.write("HARDWARE CONFIGURATION\n")
                f.write("-" * 80 + "\n")
                f.write(f"CPU: {hardware_profile['cpu']['cores_physical']} cores @ "
                       f"{hardware_profile['cpu']['frequency_mhz']:.0f} MHz "
                       f"({hardware_profile['cpu']['tier']})\n")
                f.write(f"RAM: {hardware_profile['ram']['total_gb']:.1f} GB "
                       f"({hardware_profile['ram']['tier']})\n")
                if hardware_profile['gpu']['available']:
                    f.write(f"GPU: {hardware_profile['gpu']['name']} with "
                           f"{hardware_profile['gpu']['vram_gb']:.1f} GB VRAM "
                           f"({hardware_profile['gpu']['tier']})\n")
                else:
                    f.write("GPU: None (CPU-only)\n")
                f.write("\n")

            f.write("TRAINING RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Episodes: {len(results['episodes'])}\n")
            f.write(f"Best Profit: ${max(results['profits']):,.2f} "
                   f"({max(results['profit_percentages']):+.2f}%)\n")
            f.write(f"Average Profit: ${np.mean(results['profits']):,.2f} "
                   f"({np.mean(results['profit_percentages']):+.2f}%)\n")
            f.write(f"Final Profit: ${results['profits'][-1]:,.2f} "
                   f"({results['profit_percentages'][-1]:+.2f}%)\n")
            f.write(f"Median Profit: ${np.median(results['profits']):,.2f} "
                   f"({np.median(results['profit_percentages']):+.2f}%)\n")
            f.write(f"Std Dev Profit: ${np.std(results['profits']):,.2f} "
                   f"({np.std(results['profit_percentages']):.2f}%)\n\n")

            f.write(f"Profitable Episodes: "
                   f"{sum(1 for p in results['profit_percentages'] if p > 0)} "
                   f"({sum(1 for p in results['profit_percentages'] if p > 0)/10:.1f}%)\n")
            f.write(f"Average Win Rate: {np.mean(results['win_rates']):.2f}%\n")
            f.write(f"Average Trades per Episode: {np.mean(results['trades']):.1f}\n")
            f.write(f"Total Trades: {sum(results['trades'])}\n\n")

            f.write(f"Final Epsilon: {results['epsilons'][-1]:.4f}\n")
            f.write(f"Average Loss: {np.mean(results['losses']):.6f}\n")
            f.write(f"Final Loss: {results['losses'][-1]:.6f}\n\n")

            # Last 100 episodes performance
            f.write("LAST 100 EPISODES PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Profit: ${np.mean(results['profits'][-100:]):,.2f} "
                   f"({np.mean(results['profit_percentages'][-100:]):+.2f}%)\n")
            f.write(f"Average Reward: {np.mean(results['rewards'][-100:]):.2f}\n")
            f.write(f"Average Win Rate: {np.mean(results['win_rates'][-100:]):.2f}%\n")
            f.write(f"Profitable Episodes: "
                   f"{sum(1 for p in results['profit_percentages'][-100:] if p > 0)}/100\n")

        logger.info(f"✅ Text summary saved to {text_report_path}")

    except ImportError:
        logger.warning("matplotlib not available - skipping visual report")
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Nexlify ML/RL Agent for 1000 rounds"
    )

    parser.add_argument(
        '--agent-type',
        type=str,
        choices=['adaptive', 'ultra', 'basic'],
        default='adaptive',
        help='Type of agent to use (default: adaptive)'
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
        default='models/ml_rl_1000',
        help='Checkpoint directory (default: models/ml_rl_1000)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint file'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Trading symbol (default: BTC/USDT)'
    )

    # Validation arguments
    parser.add_argument(
        '--use-validation',
        action='store_true',
        default=True,
        help='Enable validation monitoring (default: True)'
    )

    parser.add_argument(
        '--no-validation',
        dest='use_validation',
        action='store_false',
        help='Disable validation monitoring'
    )

    parser.add_argument(
        '--validation-frequency',
        type=int,
        default=50,
        help='Run validation every N episodes (default: 50)'
    )

    parser.add_argument(
        '--validation-metric',
        type=str,
        choices=['val_sharpe', 'val_return_pct', 'val_win_rate'],
        default='val_sharpe',
        help='Metric for validation and early stopping (default: val_sharpe)'
    )

    parser.add_argument(
        '--train-split',
        type=float,
        default=0.70,
        help='Training data split ratio (default: 0.70)'
    )

    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation data split ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Test data split ratio (default: 0.15)'
    )

    # Early stopping arguments
    parser.add_argument(
        '--use-early-stopping',
        action='store_true',
        default=True,
        help='Enable early stopping (default: True)'
    )

    parser.add_argument(
        '--no-early-stopping',
        dest='use_early_stopping',
        action='store_false',
        help='Disable early stopping'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=30,
        help='Early stopping patience (default: 30)'
    )

    parser.add_argument(
        '--min-delta',
        type=float,
        default=0.01,
        help='Minimum improvement threshold for early stopping (default: 0.01)'
    )

    args = parser.parse_args()

    # Train agent
    try:
        agent, results = train_1000_rounds(
            agent_type=args.agent_type,
            data_days=args.data_days,
            initial_balance=args.balance,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.resume,
            symbol=args.symbol,
            # Validation parameters
            use_validation=args.use_validation,
            validation_frequency=args.validation_frequency,
            validation_metric=args.validation_metric,
            train_val_test_split=(args.train_split, args.val_split, args.test_split),
            # Early stopping parameters
            use_early_stopping=args.use_early_stopping,
            early_stopping_patience=args.patience,
            early_stopping_min_delta=args.min_delta,
            early_stopping_mode='max'  # Sharpe/return/win_rate should all be maximized
        )

        print("\n" + "=" * 80)
        print("  ✅ 1000-ROUND TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"  Models saved in: {args.checkpoint_dir}/")
        print(f"  Best model: {args.checkpoint_dir}/best_model.pth")
        print(f"  Final model: {args.checkpoint_dir}/final_model_1000.pth")
        print(f"  Report: {args.checkpoint_dir}/training_report_1000.png")
        print("=" * 80 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        logger.info("Progress has been saved in checkpoints")
        return 1

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
