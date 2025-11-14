#!/usr/bin/env python3
"""
Train Ensemble of DQN Agents for Robust Trading

Trains multiple DQN agents with different initializations and combines them
into an ensemble for more robust and reliable trading predictions.

Features:
- Multi-model training with different seeds and epsilon strategies
- Parallel training support
- Automatic validation and model selection
- Ensemble performance analysis
- Uncertainty estimation

Usage:
    python train_ensemble.py [--num-models 5] [--episodes 1000] [--parallel]
    python train_ensemble.py --config ensemble_config.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add nexlify to path
sys.path.insert(0, str(Path(__file__).parent))

from nexlify.strategies.ensemble_agent import (EnsembleManager,
                                               EnsembleStrategy, create_ensemble)
from nexlify.training.ensemble_trainer import (EnsembleTrainer,
                                               EnsembleTrainingConfig,
                                               train_ensemble)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/neural_config.json"):
    """Load configuration file"""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        return {}

    with open(config_file) as f:
        return json.load(f)


def load_training_data(data_path: str = "data/historical_prices.npy"):
    """
    Load training data

    Returns:
        (training_data, validation_data) tuple
    """
    data_file = Path(data_path)

    if not data_file.exists():
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Generating synthetic data for demonstration...")

        # Generate synthetic price data for demonstration
        np.random.seed(42)
        num_steps = 10000
        base_price = 50000  # BTC starting price

        # Random walk with trend
        returns = np.random.normal(0.0001, 0.02, num_steps)
        prices = base_price * np.exp(np.cumsum(returns))

        # Add some volatility clustering
        for i in range(100, num_steps, 500):
            volatility_boost = np.random.uniform(1.5, 3.0)
            prices[i:i+100] += np.random.normal(0, prices[i] * 0.01 * volatility_boost, 100)

        prices = np.abs(prices)  # Ensure positive
    else:
        prices = np.load(data_path)

    # Split into training and validation (80/20)
    split_idx = int(len(prices) * 0.8)
    training_data = prices[:split_idx]
    validation_data = prices[split_idx:]

    logger.info(f"Loaded data: {len(training_data)} training, {len(validation_data)} validation")

    return training_data, validation_data


def main():
    parser = argparse.ArgumentParser(
        description="Train ensemble of DQN agents for trading"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/neural_config.json",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        help="Number of models to train (default: 5)"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Training episodes per model (default: 1000)"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train models in parallel"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Train models sequentially"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/ensemble",
        help="Output directory for trained models"
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/historical_prices.npy",
        help="Path to training data (numpy array)"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="weighted_avg",
        choices=["simple_avg", "weighted_avg", "voting", "stacking"],
        help="Ensemble strategy to use"
    )

    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="Number of models to include in final ensemble (default: 3)"
    )

    parser.add_argument(
        "--test-ensemble",
        action="store_true",
        help="Test ensemble after training"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Get ensemble config (use CLI args if provided, otherwise use config)
    ensemble_config = config.get('ensemble', {})
    training_config = ensemble_config.get('training', {})

    num_models = args.num_models or training_config.get('num_models', 5)
    episodes = args.episodes or training_config.get('episodes_per_model', 1000)
    parallel = args.parallel or (not args.no_parallel and training_config.get('parallel_training', True))
    output_dir = args.output_dir or training_config.get('output_dir', './models/ensemble')

    logger.info("="*80)
    logger.info("ENSEMBLE TRAINING")
    logger.info("="*80)
    logger.info(f"Number of models: {num_models}")
    logger.info(f"Episodes per model: {episodes}")
    logger.info(f"Parallel training: {parallel}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Ensemble strategy: {args.strategy}")
    logger.info("="*80)

    # Load training data
    training_data, validation_data = load_training_data(args.data)

    # Get agent configuration
    agent_config = config.get('rl_agent', {})

    # State and action sizes (default for crypto trading)
    state_size = 12  # From TradingEnvironment
    action_size = 3  # Hold, Buy, Sell

    # Train ensemble
    logger.info("\n🚀 Starting ensemble training...")

    trainer, results = train_ensemble(
        state_size=state_size,
        action_size=action_size,
        training_data=training_data,
        validation_data=validation_data,
        num_models=num_models,
        episodes_per_model=episodes,
        parallel=parallel,
        output_dir=output_dir,
        agent_config=agent_config
    )

    # Print comparison
    logger.info(trainer.compare_models())

    # Get best models
    best_models = trainer.get_best_models(top_k=args.ensemble_size)

    logger.info(f"\n🏆 Top {args.ensemble_size} models:")
    for i, result in enumerate(best_models, 1):
        logger.info(
            f"   {i}. Model {result.model_id} - "
            f"Validation: {result.validation_score:.4f}, "
            f"Avg Reward: {result.avg_reward:.2f}"
        )

    # Create ensemble manager
    logger.info(f"\n🎯 Creating ensemble manager ({args.strategy})...")

    validation_scores = trainer.get_validation_scores()

    ensemble = create_ensemble(
        state_size=state_size,
        action_size=action_size,
        models_dir=output_dir,
        strategy=args.strategy,
        ensemble_size=args.ensemble_size,
        validation_scores=validation_scores,
        config=config
    )

    # Save ensemble configuration
    ensemble_config_path = Path(output_dir) / "ensemble_config.json"
    ensemble.save_ensemble_config(str(ensemble_config_path))

    logger.info(f"💾 Ensemble configuration saved to {ensemble_config_path}")

    # Test ensemble if requested
    if args.test_ensemble:
        logger.info("\n🧪 Testing ensemble predictions...")
        test_ensemble_predictions(ensemble, validation_data)

    logger.info("\n✅ Ensemble training complete!")
    logger.info(f"   Models saved to: {output_dir}")
    logger.info(f"   Ensemble config: {ensemble_config_path}")


def test_ensemble_predictions(
    ensemble: EnsembleManager,
    test_data: np.ndarray,
    num_tests: int = 100
):
    """
    Test ensemble predictions and analyze uncertainty

    Args:
        ensemble: Trained ensemble manager
        test_data: Test price data
        num_tests: Number of test predictions
    """
    from nexlify.strategies.nexlify_rl_agent import TradingEnvironment

    # Create test environment
    env = TradingEnvironment(price_data=test_data)

    uncertainties = []
    actions = []

    for i in range(num_tests):
        state = env.reset()

        # Get ensemble prediction with uncertainty
        action, uncertainty = ensemble.predict(state, return_uncertainty=True)

        actions.append(action)
        if uncertainty is not None:
            uncertainties.append(uncertainty)

        # Take a few steps
        for _ in range(10):
            next_state, reward, done, info = env.step(action)
            if done:
                break
            action, uncertainty = ensemble.predict(next_state, return_uncertainty=True)
            if uncertainty is not None:
                uncertainties.append(uncertainty)

    # Analyze results
    logger.info("\n📊 Ensemble Test Results:")
    logger.info(f"   Total predictions: {len(actions)}")
    logger.info(f"   Action distribution:")
    logger.info(f"      Hold: {actions.count(0)} ({actions.count(0)/len(actions)*100:.1f}%)")
    logger.info(f"      Buy:  {actions.count(1)} ({actions.count(1)/len(actions)*100:.1f}%)")
    logger.info(f"      Sell: {actions.count(2)} ({actions.count(2)/len(actions)*100:.1f}%)")

    if uncertainties:
        logger.info(f"   Uncertainty statistics:")
        logger.info(f"      Mean: {np.mean(uncertainties):.4f}")
        logger.info(f"      Std:  {np.std(uncertainties):.4f}")
        logger.info(f"      Min:  {np.min(uncertainties):.4f}")
        logger.info(f"      Max:  {np.max(uncertainties):.4f}")

        # High uncertainty predictions
        high_uncertainty_threshold = 0.5
        high_uncertainty_count = sum(1 for u in uncertainties if u > high_uncertainty_threshold)
        logger.info(
            f"      High uncertainty (>{high_uncertainty_threshold}): "
            f"{high_uncertainty_count} ({high_uncertainty_count/len(uncertainties)*100:.1f}%)"
        )

    # Get ensemble statistics
    stats = ensemble.get_statistics()
    logger.info(f"\n📈 Ensemble Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")


if __name__ == "__main__":
    main()
