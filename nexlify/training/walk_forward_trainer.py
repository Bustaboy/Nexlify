"""
Walk-Forward Training Integration

Integrates walk-forward validation with RL agent training for robust model development.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
import json
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from nexlify.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardResults,
    calculate_performance_metrics
)
from nexlify.strategies.nexlify_rl_agent import NexlifyRLAgent
from nexlify.environments.nexlify_trading_env import TradingEnvironment
from nexlify.utils.error_handler import get_error_handler

logger = logging.getLogger(__name__)


class WalkForwardTrainer:
    """
    Trainer that integrates RL agent training with walk-forward validation

    This class provides a complete training pipeline that uses walk-forward
    validation to ensure robust model performance estimates.

    Example:
        >>> config = load_config('config/neural_config.json')
        >>> trainer = WalkForwardTrainer(config)
        >>> results = await trainer.train()
        >>> print(results.summary())
    """

    def __init__(
        self,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        Initialize walk-forward trainer

        Args:
            config: Configuration dictionary with walk_forward and rl_agent sections
            progress_callback: Optional callback(message, progress_pct) for UI updates
        """
        self.config = config
        self.wf_config = config.get('walk_forward', {})
        self.rl_config = config.get('rl_agent', {})
        self.trading_config = config.get('trading', {})
        self.progress_callback = progress_callback
        self.error_handler = get_error_handler()

        # Training state
        self.current_fold = 0
        self.total_folds = 0
        self.training_history: List[Dict[str, Any]] = []
        self.best_model_path: Optional[Path] = None

        # Initialize environment
        self.env = self._create_environment()

        logger.info("WalkForwardTrainer initialized")

    def _create_environment(self) -> TradingEnvironment:
        """Create trading environment from configuration"""
        return TradingEnvironment(
            symbols=self.trading_config.get('symbols', ['BTC/USDT']),
            initial_balance=self.trading_config.get('initial_balance', 10000),
            timeframe=self.trading_config.get('timeframe', '1h'),
            config=self.config
        )

    def _update_progress(self, message: str, progress: float) -> None:
        """Update progress via callback if available"""
        logger.info(f"Progress {progress:.1f}%: {message}")
        if self.progress_callback:
            try:
                self.progress_callback(message, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    async def train_fold(
        self,
        train_start: int,
        train_end: int,
        fold_id: int
    ) -> NexlifyRLAgent:
        """
        Train RL agent on specified episode range

        Args:
            train_start: Starting episode index
            train_end: Ending episode index (exclusive)
            fold_id: Fold identifier

        Returns:
            Trained RL agent
        """
        self.current_fold = fold_id
        num_episodes = train_end - train_start

        self._update_progress(
            f"Training fold {fold_id + 1}/{self.total_folds} ({num_episodes} episodes)",
            (fold_id / self.total_folds) * 100
        )

        # Initialize fresh agent for this fold
        agent = NexlifyRLAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            config=self.rl_config
        )

        # Training metrics
        episode_rewards = []
        episode_profits = []

        # Training loop
        for i in range(num_episodes):
            episode_idx = train_start + i

            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            done = False
            steps = 0

            while not done:
                # Select action
                action = agent.act(state)

                # Take step
                next_state, reward, done, info = self.env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(agent.memory) > agent.batch_size:
                    loss = agent.replay()

                state = next_state
                episode_reward += reward
                episode_profit += info.get('profit', 0)
                steps += 1

            episode_rewards.append(episode_reward)
            episode_profits.append(episode_profit)

            # Log progress every 100 episodes
            if (i + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_profit = np.mean(episode_profits[-100:])
                progress = ((fold_id + (i + 1) / num_episodes) / self.total_folds) * 100

                self._update_progress(
                    f"Fold {fold_id + 1} - Episode {i + 1}/{num_episodes}: "
                    f"Avg Reward={avg_reward:.2f}, Avg Profit={avg_profit:.2f}",
                    progress
                )

        # Store fold training history
        self.training_history.append({
            'fold_id': fold_id,
            'episodes': num_episodes,
            'avg_reward': np.mean(episode_rewards),
            'avg_profit': np.mean(episode_profits),
            'final_epsilon': agent.epsilon
        })

        logger.info(
            f"Fold {fold_id} training complete: "
            f"Avg Reward={np.mean(episode_rewards):.2f}, "
            f"Epsilon={agent.epsilon:.3f}"
        )

        return agent

    async def evaluate_fold(
        self,
        agent: NexlifyRLAgent,
        test_start: int,
        test_end: int,
        fold_id: int
    ) -> Dict[str, float]:
        """
        Evaluate agent on test window

        Args:
            agent: Trained RL agent
            test_start: Starting episode index for testing
            test_end: Ending episode index for testing (exclusive)
            fold_id: Fold identifier

        Returns:
            Dictionary of performance metrics
        """
        num_episodes = test_end - test_start

        self._update_progress(
            f"Evaluating fold {fold_id + 1}/{self.total_folds} ({num_episodes} episodes)",
            ((fold_id + 0.9) / self.total_folds) * 100
        )

        # Set agent to evaluation mode (no exploration)
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Greedy policy

        # Collect metrics during evaluation
        returns_list = []
        rewards_list = []
        trades = []
        profits = []

        for i in range(num_episodes):
            episode_idx = test_start + i

            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_profit = 0
            episode_steps = 0
            done = False

            episode_trades = []

            while not done:
                # Select action (greedy)
                action = agent.act(state)

                # Take step
                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward
                episode_profit += info.get('profit', 0)
                episode_steps += 1

                # Track trades
                if info.get('trade_made', False):
                    episode_trades.append({
                        'profit': info.get('trade_profit', 0),
                        'duration': info.get('trade_duration', 0)
                    })

            # Calculate episode return from profit
            initial_balance = self.trading_config.get('initial_balance', 10000)
            episode_return = episode_profit / initial_balance if initial_balance > 0 else 0

            returns_list.append(episode_return)
            rewards_list.append(episode_reward)
            profits.append(episode_profit)
            trades.extend(episode_trades)

        # Restore epsilon
        agent.epsilon = original_epsilon

        # Calculate comprehensive metrics
        returns_array = np.array(returns_list)
        metrics = calculate_performance_metrics(
            returns=returns_array,
            trades=trades if trades else None,
            risk_free_rate=self.wf_config.get('risk_free_rate', 0.02)
        )

        # Add custom metrics
        metrics['avg_reward'] = float(np.mean(rewards_list))
        metrics['total_profit'] = float(np.sum(profits))
        metrics['avg_episode_profit'] = float(np.mean(profits))

        logger.info(
            f"Fold {fold_id} evaluation complete: "
            f"Return={metrics['total_return']:.2%}, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"Profit=${metrics['total_profit']:.2f}"
        )

        return metrics

    async def train(
        self,
        save_models: bool = True,
        generate_report: bool = True
    ) -> WalkForwardResults:
        """
        Execute walk-forward training

        Args:
            save_models: Whether to save models from each fold
            generate_report: Whether to generate validation report

        Returns:
            WalkForwardResults object with complete validation results
        """
        logger.info("Starting walk-forward training")
        self._update_progress("Initializing walk-forward validation", 0)

        # Initialize validator
        validator = WalkForwardValidator(
            total_episodes=self.wf_config.get('total_episodes', 2000),
            train_size=self.wf_config.get('train_size', 1000),
            test_size=self.wf_config.get('test_size', 200),
            step_size=self.wf_config.get('step_size', 200),
            mode=self.wf_config.get('mode', 'rolling'),
            min_train_size=self.wf_config.get('min_train_size', 500),
            config=self.config
        )

        self.total_folds = len(validator.folds)
        logger.info(f"Generated {self.total_folds} folds for validation")

        # Create wrapper functions that include fold_id
        async def train_fn(train_start: int, train_end: int) -> NexlifyRLAgent:
            # Calculate fold_id from train_start
            fold_id = len(self.training_history)
            return await self.train_fold(train_start, train_end, fold_id)

        async def eval_fn(
            agent: NexlifyRLAgent,
            test_start: int,
            test_end: int
        ) -> Dict[str, float]:
            # Calculate fold_id from test_start
            fold_id = len(self.training_history) - 1
            return await self.evaluate_fold(agent, test_start, test_end, fold_id)

        # Run validation
        try:
            results = await validator.validate(
                train_fn=train_fn,
                eval_fn=eval_fn,
                save_models=save_models,
                model_dir=Path(self.wf_config.get('model_dir', 'models/walk_forward'))
            )

            self._update_progress("Training complete", 100)

            # Select best model
            if self.wf_config.get('integration', {}).get('select_best_model', True):
                metric = self.wf_config.get('integration', {}).get('validation_metric', 'sharpe_ratio')
                best_fold_id = self._select_best_fold(results, metric)
                self.best_model_path = Path(
                    self.wf_config.get('model_dir', 'models/walk_forward')
                ) / f"fold_{best_fold_id}_model.pt"

                logger.info(f"Best model selected: Fold {best_fold_id} (by {metric})")

            # Generate report
            if generate_report:
                output_dir = Path(self.wf_config.get('output_dir', 'reports/walk_forward'))
                validator.generate_report(results, output_dir)
                logger.info(f"Report generated in {output_dir}")

            # Save training history
            self._save_training_history(results)

            return results

        except Exception as e:
            self.error_handler.log_error(
                e,
                context={'operation': 'walk_forward_training'}
            )
            self._update_progress(f"Training failed: {e}", 100)
            raise

    def _select_best_fold(
        self,
        results: WalkForwardResults,
        metric: str = 'sharpe_ratio'
    ) -> int:
        """
        Select best fold based on specified metric

        Args:
            results: Walk-forward validation results
            metric: Metric to use for selection

        Returns:
            Fold ID of best performing fold
        """
        if metric not in ['sharpe_ratio', 'total_return', 'sortino_ratio', 'calmar_ratio']:
            logger.warning(f"Unknown metric '{metric}', using sharpe_ratio")
            metric = 'sharpe_ratio'

        best_fold_id = max(
            range(len(results.fold_metrics)),
            key=lambda i: getattr(results.fold_metrics[i], metric)
        )

        logger.info(
            f"Best fold by {metric}: Fold {best_fold_id} "
            f"({metric}={getattr(results.fold_metrics[best_fold_id], metric):.3f})"
        )

        return best_fold_id

    def _save_training_history(self, results: WalkForwardResults) -> None:
        """Save training history to file"""
        history_file = Path('training_logs/walk_forward_history.json')
        history_file.parent.mkdir(parents=True, exist_ok=True)

        history_data = {
            'training_history': self.training_history,
            'validation_results': results.to_dict(),
            'best_model_path': str(self.best_model_path) if self.best_model_path else None,
            'config': {
                'walk_forward': self.wf_config,
                'rl_agent': self.rl_config
            }
        }

        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Training history saved to {history_file}")

    def load_best_model(self) -> Optional[NexlifyRLAgent]:
        """
        Load the best model from walk-forward validation

        Returns:
            Loaded RL agent or None if no best model available
        """
        if not self.best_model_path or not self.best_model_path.exists():
            logger.warning("No best model available to load")
            return None

        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available, cannot load model")
            return None

        try:
            agent = NexlifyRLAgent(
                state_size=self.env.state_size,
                action_size=self.env.action_size,
                config=self.rl_config
            )

            agent.model.load_state_dict(torch.load(self.best_model_path))
            agent.target_model.load_state_dict(torch.load(self.best_model_path))

            logger.info(f"Loaded best model from {self.best_model_path}")
            return agent

        except Exception as e:
            self.error_handler.log_error(
                e,
                context={'operation': 'load_best_model', 'path': str(self.best_model_path)}
            )
            return None


async def train_with_walk_forward(
    config_path: str = 'config/neural_config.json',
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> WalkForwardResults:
    """
    Convenience function to run walk-forward training

    Args:
        config_path: Path to configuration file
        progress_callback: Optional progress callback for UI

    Returns:
        WalkForwardResults object

    Example:
        >>> results = await train_with_walk_forward('config/neural_config.json')
        >>> print(results.summary())
    """
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)

    # Create trainer
    trainer = WalkForwardTrainer(config, progress_callback=progress_callback)

    # Run training
    results = await trainer.train()

    return results
