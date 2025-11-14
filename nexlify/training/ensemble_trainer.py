#!/usr/bin/env python3
"""
Nexlify Ensemble Trainer - Multi-Model Training for Ensemble Learning

Trains multiple DQN agents with different initializations to create diverse ensemble.

Features:
- Multi-start training with different random seeds
- Parallel training support (if hardware allows)
- Different epsilon strategies per model
- Automatic validation scoring
- Checkpoint saving for all models
- Performance comparison and analysis
"""

import json
import logging
import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from nexlify.strategies.nexlify_rl_agent import DQNAgent, TradingEnvironment
from nexlify.utils.error_handler import get_error_handler, handle_errors

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class EnsembleTrainingConfig:
    """Configuration for ensemble training"""

    def __init__(
        self,
        num_models: int = 5,
        episodes_per_model: int = 1000,
        parallel_training: bool = True,
        max_workers: Optional[int] = None,
        seed_start: int = 42,
        epsilon_variations: bool = True,
        save_all_checkpoints: bool = True,
        validation_episodes: int = 100,
        output_dir: str = "./models/ensemble"
    ):
        """
        Args:
            num_models: Number of models to train
            episodes_per_model: Training episodes per model
            parallel_training: Train models in parallel
            max_workers: Max parallel workers (None = auto-detect)
            seed_start: Starting random seed
            epsilon_variations: Use different epsilon schedules
            save_all_checkpoints: Save all model checkpoints
            validation_episodes: Episodes for validation scoring
            output_dir: Directory to save models
        """
        self.num_models = num_models
        self.episodes_per_model = episodes_per_model
        self.parallel_training = parallel_training
        self.max_workers = max_workers or min(num_models, mp.cpu_count())
        self.seed_start = seed_start
        self.epsilon_variations = epsilon_variations
        self.save_all_checkpoints = save_all_checkpoints
        self.validation_episodes = validation_episodes
        self.output_dir = output_dir


class ModelTrainingResult:
    """Results from training a single model"""

    def __init__(
        self,
        model_id: int,
        model_path: str,
        training_time: float,
        final_epsilon: float,
        avg_reward: float,
        validation_score: float,
        seed: int
    ):
        self.model_id = model_id
        self.model_path = model_path
        self.training_time = training_time
        self.final_epsilon = final_epsilon
        self.avg_reward = avg_reward
        self.validation_score = validation_score
        self.seed = seed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "training_time": self.training_time,
            "final_epsilon": self.final_epsilon,
            "avg_reward": self.avg_reward,
            "validation_score": self.validation_score,
            "seed": self.seed
        }


class EnsembleTrainer:
    """
    Trainer for ensemble of DQN agents

    Trains multiple models with different initializations to create
    a diverse ensemble for robust predictions.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        training_data: np.ndarray,
        validation_data: np.ndarray,
        config: EnsembleTrainingConfig,
        agent_config: Optional[Dict] = None
    ):
        """
        Initialize ensemble trainer

        Args:
            state_size: Size of state space
            action_size: Size of action space
            training_data: Price data for training
            validation_data: Price data for validation
            config: Ensemble training configuration
            agent_config: Configuration for individual agents
        """
        self.state_size = state_size
        self.action_size = action_size
        self.training_data = training_data
        self.validation_data = validation_data
        self.config = config
        self.agent_config = agent_config or {}

        # Results tracking
        self.training_results: List[ModelTrainingResult] = []

        # Ensure output directory exists
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🎓 Ensemble Trainer initialized")
        logger.info(f"   Number of models: {config.num_models}")
        logger.info(f"   Episodes per model: {config.episodes_per_model}")
        logger.info(f"   Parallel training: {config.parallel_training}")
        if config.parallel_training:
            logger.info(f"   Max workers: {config.max_workers}")
        logger.info(f"   Output directory: {config.output_dir}")

    def _create_model_config(self, model_id: int) -> Dict[str, Any]:
        """
        Create configuration for a specific model

        Varies epsilon schedules and other hyperparameters to encourage diversity

        Args:
            model_id: ID of model (0 to num_models-1)

        Returns:
            Configuration dictionary for this model
        """
        config = self.agent_config.copy()

        # Set random seed
        seed = self.config.seed_start + model_id
        config['random_seed'] = seed

        # Vary epsilon if enabled
        if self.config.epsilon_variations:
            # Different epsilon strategies for diversity
            strategies = [
                {
                    'epsilon_decay_type': 'scheduled',
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.01
                },
                {
                    'epsilon_decay_type': 'exponential',
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.01,
                    'decay_rate': 0.995
                },
                {
                    'epsilon_decay_type': 'exponential',
                    'epsilon_start': 0.8,  # Less exploration
                    'epsilon_end': 0.05,
                    'decay_rate': 0.998
                },
                {
                    'epsilon_decay_type': 'scheduled',
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.05,  # Higher final epsilon
                },
                {
                    'epsilon_decay_type': 'exponential',
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.001,  # More exploitation
                    'decay_rate': 0.992
                }
            ]

            # Select strategy based on model_id
            strategy = strategies[model_id % len(strategies)]
            config.update(strategy)

        return config

    @handle_errors
    def train_ensemble(self) -> List[ModelTrainingResult]:
        """
        Train all models in ensemble

        Returns:
            List of training results for each model
        """
        logger.info(f"🚀 Starting ensemble training ({self.config.num_models} models)...")

        start_time = time.time()

        if self.config.parallel_training:
            results = self._train_parallel()
        else:
            results = self._train_sequential()

        total_time = time.time() - start_time

        logger.info(f"✅ Ensemble training complete!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Models trained: {len(results)}")

        # Save ensemble summary
        self._save_ensemble_summary(results, total_time)

        self.training_results = results

        return results

    def _train_sequential(self) -> List[ModelTrainingResult]:
        """Train models sequentially"""
        results = []

        for model_id in range(self.config.num_models):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training Model {model_id + 1}/{self.config.num_models}")
            logger.info(f"{'='*60}")

            result = self._train_single_model(model_id)
            results.append(result)

            logger.info(f"✅ Model {model_id + 1} complete:")
            logger.info(f"   Validation score: {result.validation_score:.4f}")
            logger.info(f"   Training time: {result.training_time:.2f}s")

        return results

    def _train_parallel(self) -> List[ModelTrainingResult]:
        """Train models in parallel"""
        logger.info(f"🔀 Training {self.config.num_models} models in parallel "
                   f"({self.config.max_workers} workers)...")

        results = []

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all training jobs
            futures = {
                executor.submit(self._train_single_model, model_id): model_id
                for model_id in range(self.config.num_models)
            }

            # Collect results as they complete
            for future in as_completed(futures):
                model_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(
                        f"✅ Model {model_id + 1} complete "
                        f"(validation: {result.validation_score:.4f})"
                    )
                except Exception as e:
                    logger.error(f"❌ Model {model_id + 1} failed: {e}")

        # Sort by model_id
        results.sort(key=lambda r: r.model_id)

        return results

    def _train_single_model(self, model_id: int) -> ModelTrainingResult:
        """
        Train a single model

        Args:
            model_id: ID of model to train

        Returns:
            Training result
        """
        start_time = time.time()

        # Create model configuration
        model_config = self._create_model_config(model_id)

        # Set random seeds for reproducibility
        seed = model_config['random_seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create agent
        agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            config=model_config
        )

        # Create training environment
        env = TradingEnvironment(
            price_data=self.training_data,
            config=model_config
        )

        # Training loop
        episode_rewards = []

        for episode in range(self.config.episodes_per_model):
            state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Select action
                action = agent.act(state, training=True)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

                episode_reward += reward
                state = next_state

            # Decay epsilon
            agent.decay_epsilon()

            episode_rewards.append(episode_reward)

            # Update target network periodically
            if (episode + 1) % agent.target_update_freq == 0:
                agent.update_target_model()

            # Log progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.debug(
                    f"   Model {model_id + 1} | Episode {episode + 1}/{self.config.episodes_per_model} | "
                    f"Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}"
                )

        # Calculate average training reward
        avg_reward = float(np.mean(episode_rewards[-100:]))

        # Save model
        model_filename = f"model_{model_id:03d}_seed{seed}.pt"
        model_path = self.output_dir / model_filename
        agent.save(str(model_path))

        # Validate model
        validation_score = self._validate_model(agent)

        training_time = time.time() - start_time

        # Create result
        result = ModelTrainingResult(
            model_id=model_id,
            model_path=str(model_path),
            training_time=training_time,
            final_epsilon=agent.epsilon,
            avg_reward=avg_reward,
            validation_score=validation_score,
            seed=seed
        )

        return result

    def _validate_model(self, agent: DQNAgent) -> float:
        """
        Validate model performance

        Args:
            agent: Trained agent to validate

        Returns:
            Validation score (higher is better)
        """
        # Create validation environment
        env = TradingEnvironment(
            price_data=self.validation_data,
            config=self.agent_config
        )

        episode_returns = []

        for episode in range(self.config.validation_episodes):
            state = env.reset()
            done = False
            episode_return = 0

            while not done:
                # Select action (no exploration)
                action = agent.act(state, training=False)

                # Execute action
                next_state, reward, done, info = env.step(action)

                episode_return += reward
                state = next_state

            episode_returns.append(episode_return)

        # Validation score: mean return + Sharpe-like adjustment
        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)

        # Sharpe-like score (avoid division by zero)
        if std_return > 0:
            validation_score = mean_return / std_return
        else:
            validation_score = mean_return

        return float(validation_score)

    def _save_ensemble_summary(self, results: List[ModelTrainingResult], total_time: float):
        """Save summary of ensemble training"""
        summary = {
            "training_config": {
                "num_models": self.config.num_models,
                "episodes_per_model": self.config.episodes_per_model,
                "parallel_training": self.config.parallel_training,
                "epsilon_variations": self.config.epsilon_variations
            },
            "total_training_time": total_time,
            "models": [r.to_dict() for r in results],
            "statistics": {
                "avg_validation_score": float(np.mean([r.validation_score for r in results])),
                "best_validation_score": float(max([r.validation_score for r in results])),
                "worst_validation_score": float(min([r.validation_score for r in results])),
                "avg_training_time": float(np.mean([r.training_time for r in results])),
                "validation_score_std": float(np.std([r.validation_score for r in results]))
            },
            "timestamp": datetime.now().isoformat()
        }

        # Save summary
        summary_path = self.output_dir / "ensemble_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"💾 Ensemble summary saved to {summary_path}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ENSEMBLE TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Average validation score: {summary['statistics']['avg_validation_score']:.4f}")
        logger.info(f"Best validation score: {summary['statistics']['best_validation_score']:.4f}")
        logger.info(f"Validation score std: {summary['statistics']['validation_score_std']:.4f}")
        logger.info("="*60)

    def get_best_models(self, top_k: int = 3) -> List[ModelTrainingResult]:
        """
        Get top K models by validation score

        Args:
            top_k: Number of best models to return

        Returns:
            List of top K model results
        """
        if not self.training_results:
            return []

        sorted_results = sorted(
            self.training_results,
            key=lambda r: r.validation_score,
            reverse=True
        )

        return sorted_results[:top_k]

    def get_validation_scores(self) -> Dict[str, float]:
        """
        Get validation scores for all models

        Returns:
            Dict mapping model name to validation score
        """
        return {
            Path(r.model_path).stem: r.validation_score
            for r in self.training_results
        }

    def compare_models(self) -> str:
        """
        Generate comparison report of all models

        Returns:
            Formatted comparison report
        """
        if not self.training_results:
            return "No models trained yet"

        # Sort by validation score
        sorted_results = sorted(
            self.training_results,
            key=lambda r: r.validation_score,
            reverse=True
        )

        report = "\n" + "="*80 + "\n"
        report += "MODEL COMPARISON\n"
        report += "="*80 + "\n"
        report += f"{'Rank':<6} {'Model ID':<10} {'Val Score':<12} {'Avg Reward':<12} {'Training Time':<15} {'Seed':<8}\n"
        report += "-"*80 + "\n"

        for rank, result in enumerate(sorted_results, 1):
            report += (
                f"{rank:<6} "
                f"{result.model_id:<10} "
                f"{result.validation_score:<12.4f} "
                f"{result.avg_reward:<12.2f} "
                f"{result.training_time:<15.2f} "
                f"{result.seed:<8}\n"
            )

        report += "="*80 + "\n"

        return report


def train_ensemble(
    state_size: int,
    action_size: int,
    training_data: np.ndarray,
    validation_data: np.ndarray,
    num_models: int = 5,
    episodes_per_model: int = 1000,
    parallel: bool = True,
    output_dir: str = "./models/ensemble",
    agent_config: Optional[Dict] = None
) -> Tuple[EnsembleTrainer, List[ModelTrainingResult]]:
    """
    Convenience function to train ensemble

    Args:
        state_size: State space size
        action_size: Action space size
        training_data: Training price data
        validation_data: Validation price data
        num_models: Number of models to train
        episodes_per_model: Episodes per model
        parallel: Train in parallel
        output_dir: Output directory
        agent_config: Agent configuration

    Returns:
        (trainer, results) tuple
    """
    config = EnsembleTrainingConfig(
        num_models=num_models,
        episodes_per_model=episodes_per_model,
        parallel_training=parallel,
        output_dir=output_dir
    )

    trainer = EnsembleTrainer(
        state_size=state_size,
        action_size=action_size,
        training_data=training_data,
        validation_data=validation_data,
        config=config,
        agent_config=agent_config
    )

    results = trainer.train_ensemble()

    return trainer, results


__all__ = [
    "EnsembleTrainer",
    "EnsembleTrainingConfig",
    "ModelTrainingResult",
    "train_ensemble"
]
