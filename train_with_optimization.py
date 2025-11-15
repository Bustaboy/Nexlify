#!/usr/bin/env python3
"""
Training with Hyperparameter Optimization Integration
Demonstrates how to use Optuna optimization with Nexlify training infrastructure

This script shows two modes:
1. Optimize-then-train: Find best hyperparameters, then train with them
2. Load-and-train: Load previously optimized parameters and train

Features:
- Integrates Optuna offline optimization with AutoTuner online tuning
- Automatically applies optimized parameters to training config
- Saves optimized config for future use
- Comprehensive logging and reporting
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from nexlify.optimization import (
    OptimizationIntegration,
    create_optimized_agent,
    HyperparameterTuner,
    create_custom_search_space,
    create_balanced_objective
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def optimize_and_train(
    symbol: str = 'BTC/USDT',
    exchange: str = 'binance',
    optimization_trials: int = 50,
    training_episodes: int = 1000,
    optimization_objective: str = 'sharpe',
    search_space: str = 'compact',
    output_dir: str = './optimized_training',
    enable_dynamic_tuning: bool = True,
    save_config: bool = True
) -> Dict[str, Any]:
    """
    Complete workflow: optimize hyperparameters, then train with best params

    Args:
        symbol: Trading pair
        exchange: Exchange name
        optimization_trials: Number of optimization trials
        training_episodes: Episodes for final training
        optimization_objective: Objective for optimization ('sharpe', 'multi', etc.)
        search_space: Search space type ('compact', 'default', 'advanced')
        output_dir: Output directory
        enable_dynamic_tuning: Enable AutoTuner during training
        save_config: Save optimized config for future use

    Returns:
        Dict with optimization and training results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "="*80)
    logger.info("NEXLIFY TRAINING WITH HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Exchange: {exchange}")
    logger.info(f"Optimization trials: {optimization_trials}")
    logger.info(f"Training episodes: {training_episodes}")
    logger.info(f"Objective: {optimization_objective}")
    logger.info(f"Search space: {search_space}")
    logger.info(f"Dynamic tuning: {enable_dynamic_tuning}")
    logger.info("="*80 + "\n")

    # =========================================================================
    # PHASE 1: HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: OFFLINE HYPERPARAMETER OPTIMIZATION")
    logger.info("="*80)
    logger.info("Finding optimal hyperparameters using Bayesian optimization...")

    optimization_dir = output_path / 'optimization'

    # Create objective
    if optimization_objective == 'multi':
        objective = create_balanced_objective()
        logger.info("Using balanced multi-objective (40% Sharpe, 30% Return, 30% Drawdown)")
    else:
        from nexlify.optimization import create_objective
        objective = create_objective(optimization_objective)
        logger.info(f"Using {optimization_objective} objective")

    # Create search space
    space = create_custom_search_space(search_space)
    logger.info(f"Search space: {len(space.get_parameter_names())} parameters")

    # Create tuner
    tuner = HyperparameterTuner(
        objective=objective,
        search_space=space,
        n_trials=optimization_trials,
        sampler='tpe',
        pruner='median',
        output_dir=str(optimization_dir),
        verbose=True
    )

    # Training function for optimization
    def train_for_optimization(params, train_data, val_data):
        """
        Training function called during optimization

        This should be your actual training code.
        For demonstration, we'll use a simplified version.
        """
        logger.info(f"Training with params: {json.dumps(params, indent=2)}")

        # In a real implementation, this would:
        # 1. Create agent with params
        # 2. Train on train_data
        # 3. Evaluate on val_data
        # 4. Return metrics

        # For now, simulate training
        import numpy as np
        import time

        # Simulate training time
        time.sleep(1)

        # Simulate results (in reality, this comes from actual training)
        sharpe = np.random.uniform(-1, 3)
        total_return = sharpe * 0.3 + np.random.uniform(-0.1, 0.1)
        max_drawdown = np.random.uniform(-0.3, -0.05)

        return {
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'final_balance': 10000 * (1 + total_return),
            'initial_balance': 10000,
            'returns': np.random.randn(100) * 0.02,
            'balance_history': [10000 * (1 + i * total_return / 100) for i in range(100)],
        }

    # Run optimization
    logger.info(f"Running {optimization_trials} optimization trials...")
    opt_results = tuner.optimize(train_func=train_for_optimization)

    best_params = opt_results['best_params']
    best_value = opt_results['best_value']

    logger.info("\n" + "-"*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("-"*80)
    logger.info(f"Best {objective.name}: {best_value:.4f}")
    logger.info("\nBest hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"  {param:25s} = {value}")
    logger.info("-"*80 + "\n")

    # Generate optimization report
    report_path = optimization_dir / 'optimization_report.txt'
    tuner.generate_report(output_path=str(report_path))
    logger.info(f"Detailed optimization report saved to {report_path}")

    # Generate visualizations
    try:
        tuner.plot_optimization_history(save_path=str(optimization_dir / 'history.png'))
        tuner.plot_param_importances(save_path=str(optimization_dir / 'importance.png'))
        logger.info("Optimization visualizations saved")
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {e}")

    # =========================================================================
    # PHASE 2: APPLY OPTIMIZED PARAMETERS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: APPLYING OPTIMIZED HYPERPARAMETERS")
    logger.info("="*80)

    # Load base config if exists
    base_config_path = Path('config/neural_config.json')
    if base_config_path.exists():
        logger.info(f"Loading base config from {base_config_path}")
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)
    else:
        logger.info("No base config found, using defaults")
        base_config = None

    # Create training config with optimized params
    training_config = OptimizationIntegration.create_training_config_from_params(
        best_params,
        base_config
    )

    # Save optimized config if requested
    if save_config:
        config_path = output_path / 'optimized_config.json'
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        logger.info(f"Optimized configuration saved to {config_path}")

    # =========================================================================
    # PHASE 3: TRAINING WITH OPTIMIZED PARAMETERS
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    logger.info("="*80)
    logger.info(f"Training episodes: {training_episodes}")
    logger.info(f"Online dynamic tuning: {'ENABLED' if enable_dynamic_tuning else 'DISABLED'}")

    # This is where you would integrate with your actual training code
    logger.info("\nIntegration with existing training infrastructure:")
    logger.info("  Option 1: Use AdvancedTrainingOrchestrator with optimized config")
    logger.info("  Option 2: Create agent with create_optimized_agent()")
    logger.info("  Option 3: Manually apply params to your training script")

    logger.info("\nExample integration code:")
    logger.info("-"*80)
    logger.info("""
    from nexlify.optimization import OptimizationIntegration
    from nexlify_training.nexlify_advanced_training_orchestrator import AdvancedTrainingOrchestrator

    # Load optimized parameters
    best_params = OptimizationIntegration.load_best_params('./optimized_training/optimization')

    # Create training config
    config = OptimizationIntegration.create_training_config_from_params(best_params)

    # Create orchestrator with optimized config AND online tuning
    orchestrator = AdvancedTrainingOrchestrator(
        output_dir='./training_output',
        enable_auto_tuning=True  # Combines offline + online optimization!
    )

    # Train
    results = orchestrator.train(
        symbol='BTC/USDT',
        episodes=1000
    )
    """)
    logger.info("-"*80 + "\n")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"✓ Completed {len(opt_results['optimization_history'])} optimization trials")
    logger.info(f"✓ Best {objective.name}: {best_value:.4f}")
    logger.info(f"✓ Optimized {len(best_params)} hyperparameters")
    logger.info(f"✓ Configuration saved to {output_dir}/optimized_config.json")
    logger.info(f"✓ Ready for training with optimized parameters")
    logger.info("="*80 + "\n")

    return {
        'optimization_results': opt_results,
        'best_params': best_params,
        'best_value': best_value,
        'training_config': training_config,
        'output_dir': str(output_path)
    }


def load_and_train(
    optimization_dir: str,
    training_episodes: int = 1000,
    enable_dynamic_tuning: bool = True
) -> Dict[str, Any]:
    """
    Load previously optimized parameters and train

    Args:
        optimization_dir: Directory with optimization results
        training_episodes: Episodes for training
        enable_dynamic_tuning: Enable AutoTuner during training

    Returns:
        Training results
    """
    logger.info("\n" + "="*80)
    logger.info("TRAINING WITH PREVIOUSLY OPTIMIZED HYPERPARAMETERS")
    logger.info("="*80)

    # Load best parameters
    best_params = OptimizationIntegration.load_best_params(optimization_dir)

    logger.info("Loaded optimized hyperparameters:")
    for param, value in best_params.items():
        logger.info(f"  {param:25s} = {value}")

    # Create training config
    training_config = OptimizationIntegration.create_training_config_from_params(best_params)

    logger.info(f"\nTraining with {training_episodes} episodes")
    logger.info(f"Dynamic tuning: {'ENABLED' if enable_dynamic_tuning else 'DISABLED'}")

    # This would integrate with your actual training
    logger.info("\nUse this config with your training script or orchestrator")

    return {
        'best_params': best_params,
        'training_config': training_config
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Train Nexlify agent with hyperparameter optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize and train
  python train_with_optimization.py --mode optimize --trials 100 --episodes 1000

  # Load previous optimization and train
  python train_with_optimization.py --mode load --optimization-dir ./optimized_training/optimization

  # Quick test
  python train_with_optimization.py --mode optimize --trials 10 --episodes 100 --search-space compact

  # Advanced: multi-objective optimization
  python train_with_optimization.py --mode optimize --trials 200 --objective multi --search-space advanced
        """
    )

    parser.add_argument('--mode', type=str, default='optimize',
                       choices=['optimize', 'load'],
                       help='Mode: optimize (optimize then train) or load (load previous optimization)')

    # Optimization arguments
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials (default: 50)')
    parser.add_argument('--objective', type=str, default='sharpe',
                       choices=['sharpe', 'return', 'drawdown', 'multi'],
                       help='Optimization objective (default: sharpe)')
    parser.add_argument('--search-space', type=str, default='compact',
                       choices=['compact', 'default', 'advanced'],
                       help='Search space type (default: compact)')

    # Training arguments
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange name (default: binance)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Training episodes (default: 1000)')
    parser.add_argument('--disable-dynamic-tuning', action='store_true',
                       help='Disable online dynamic tuning (default: enabled)')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='./optimized_training',
                       help='Output directory (default: ./optimized_training)')
    parser.add_argument('--optimization-dir', type=str, default=None,
                       help='Optimization directory for load mode')
    parser.add_argument('--no-save-config', action='store_true',
                       help='Do not save optimized config')

    args = parser.parse_args()

    try:
        if args.mode == 'optimize':
            # Optimize and train
            results = optimize_and_train(
                symbol=args.symbol,
                exchange=args.exchange,
                optimization_trials=args.trials,
                training_episodes=args.episodes,
                optimization_objective=args.objective,
                search_space=args.search_space,
                output_dir=args.output_dir,
                enable_dynamic_tuning=not args.disable_dynamic_tuning,
                save_config=not args.no_save_config
            )

            logger.info("\n✓ Optimization and configuration complete!")
            logger.info(f"✓ Best parameters saved in {args.output_dir}/optimization/")
            logger.info(f"✓ Optimized config saved to {args.output_dir}/optimized_config.json")
            logger.info("\nNext steps:")
            logger.info("1. Review optimization results in the output directory")
            logger.info("2. Use optimized_config.json with your training scripts")
            logger.info("3. Or run with --mode load to use these parameters")

        elif args.mode == 'load':
            # Load and train
            opt_dir = args.optimization_dir or f"{args.output_dir}/optimization"

            results = load_and_train(
                optimization_dir=opt_dir,
                training_episodes=args.episodes,
                enable_dynamic_tuning=not args.disable_dynamic_tuning
            )

            logger.info("\n✓ Loaded optimized parameters successfully!")
            logger.info("✓ Ready to train with optimized configuration")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
