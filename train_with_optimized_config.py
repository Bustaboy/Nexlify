#!/usr/bin/env python3
"""
Train RL agent with optimized hyperparameters

This script uses the optimized configuration from nexlify_rl_optimized_config.py
to train the agent with settings tuned for faster learning and better performance.

Usage:
    # Standard optimized training
    python train_with_optimized_config.py --pairs BTC/USD --exchange auto

    # Fast learning mode (for testing)
    python train_with_optimized_config.py --pairs BTC/USD --fast-mode

    # Custom configuration
    python train_with_optimized_config.py --pairs BTC/USD --gamma 0.95 --lr 0.0003
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from nexlify_advanced_dqn_agent import AgentConfig
from nexlify_rl_optimized_config import (
    OptimizedAgentConfig,
    get_optimized_config,
    get_fast_learning_config,
    print_config_comparison
)
from nexlify_training.nexlify_advanced_training_orchestrator import AdvancedTrainingOrchestrator
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_to_agent_config(opt_config: OptimizedAgentConfig) -> AgentConfig:
    """
    Convert OptimizedAgentConfig to AgentConfig for compatibility

    Args:
        opt_config: Optimized configuration

    Returns:
        AgentConfig compatible with existing training code
    """
    agent_config = AgentConfig()

    # Copy all matching fields
    for field in opt_config.__dataclass_fields__:
        if hasattr(agent_config, field):
            setattr(agent_config, field, getattr(opt_config, field))

    return agent_config


def main():
    parser = argparse.ArgumentParser(description="Train RL agent with optimized hyperparameters")

    # Data arguments
    parser.add_argument('--pairs', nargs='+', required=True,
                       help='Trading pairs (e.g., BTC/USD ETH/USD)')
    parser.add_argument('--exchange', type=str, default='auto',
                       help='Exchange to use (default: auto)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Timeframe (default: 1h)')
    parser.add_argument('--years', type=int, default=2,
                       help='Years of historical data (default: 2)')

    # Configuration mode
    parser.add_argument('--fast-mode', action='store_true',
                       help='Use fast learning config (for testing)')
    parser.add_argument('--show-comparison', action='store_true',
                       help='Show config comparison and exit')

    # Hyperparameter overrides
    parser.add_argument('--gamma', type=float, help='Discount factor (default: 0.95)')
    parser.add_argument('--lr', type=float, help='Learning rate (default: 0.0003)')
    parser.add_argument('--batch-size', type=int, help='Batch size (default: 128)')
    parser.add_argument('--epsilon-decay-steps', type=int, help='Epsilon decay steps (default: 2000)')

    # Training arguments
    parser.add_argument('--output-dir', type=str, default='./training_output_optimized',
                       help='Output directory')
    parser.add_argument('--automated', action='store_true', default=True,
                       help='Automated mode (default: True)')

    args = parser.parse_args()

    # Show comparison if requested
    if args.show_comparison:
        print_config_comparison()
        return

    # Get appropriate config
    if args.fast_mode:
        logger.info("🚀 Using FAST LEARNING config")
        config = get_fast_learning_config()
    else:
        logger.info("⚙️  Using OPTIMIZED config")
        config = get_optimized_config()

    # Apply command-line overrides
    if args.gamma is not None:
        config.gamma = args.gamma
        logger.info(f"   Override: gamma = {args.gamma}")

    if args.lr is not None:
        config.learning_rate = args.lr
        logger.info(f"   Override: learning_rate = {args.lr}")

    if args.batch_size is not None:
        config.batch_size = args.batch_size
        logger.info(f"   Override: batch_size = {args.batch_size}")

    if args.epsilon_decay_steps is not None:
        config.epsilon_decay_steps = args.epsilon_decay_steps
        logger.info(f"   Override: epsilon_decay_steps = {args.epsilon_decay_steps}")

    # Convert to AgentConfig
    agent_config = convert_to_agent_config(config)

    # Print key hyperparameters
    logger.info("\n" + "="*80)
    logger.info("KEY HYPERPARAMETERS")
    logger.info("="*80)
    logger.info(f"Gamma (discount): {agent_config.gamma}")
    logger.info(f"Learning rate: {agent_config.learning_rate}")
    logger.info(f"Batch size: {agent_config.batch_size}")
    logger.info(f"Epsilon decay: {'Linear/' + str(agent_config.epsilon_decay_steps) + ' steps' if agent_config.use_linear_epsilon_decay else 'Multiplicative/' + str(agent_config.epsilon_decay)}")
    logger.info(f"Target update freq: {agent_config.target_update_frequency}")
    logger.info(f"N-step returns: {agent_config.n_step}")
    logger.info(f"Hidden layers: {agent_config.hidden_layers}")
    logger.info("="*80 + "\n")

    # Initialize orchestrator
    orchestrator = AdvancedTrainingOrchestrator(
        output_dir=args.output_dir,
        cache_enabled=True
    )

    # Train each pair
    for pair in args.pairs:
        logger.info(f"\n{'#'*80}")
        logger.info(f"TRAINING: {pair}")
        logger.info(f"{'#'*80}\n")

        try:
            # Note: You would need to modify the orchestrator to accept custom agent_config
            # For now, this demonstrates how to use the optimized config

            # TODO: Modify orchestrator.train_comprehensive() to accept agent_config parameter
            logger.warning("⚠️  To use custom config, modify AdvancedTrainingOrchestrator.train_comprehensive() to accept agent_config parameter")
            logger.info("For now, you can manually edit nexlify_advanced_dqn_agent.py AgentConfig defaults")

            results = orchestrator.train_comprehensive(
                exchange=args.exchange,
                symbol=pair,
                timeframe=args.timeframe,
                total_years=args.years
            )

            logger.info(f"\n✅ Training complete for {pair}")
            logger.info(f"   Final return: {results.get('final_return', 'N/A')}")
            logger.info(f"   Best sharpe: {results.get('best_sharpe', 'N/A')}")

        except Exception as e:
            logger.error(f"❌ Training failed for {pair}: {e}")
            if not args.automated:
                raise

    logger.info("\n" + "="*80)
    logger.info("ALL TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
