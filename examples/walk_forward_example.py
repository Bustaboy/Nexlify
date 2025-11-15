"""
Walk-Forward Validation Example

Demonstrates how to integrate walk-forward validation with RL agent training
for robust performance estimation and model selection.

Usage:
    python examples/walk_forward_example.py
    python examples/walk_forward_example.py --mode expanding --episodes 5000
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nexlify.validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardResults,
    calculate_performance_metrics
)
from nexlify.strategies.nexlify_rl_agent import NexlifyRLAgent
from nexlify.environments.nexlify_trading_env import TradingEnvironment
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/neural_config.json') -> Dict[str, Any]:
    """Load configuration from file"""
    config_file = Path(config_path)

    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    with open(config_file) as f:
        return json.load(f)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for walk-forward validation"""
    return {
        'walk_forward': {
            'enabled': True,
            'total_episodes': 2000,
            'train_size': 1000,
            'test_size': 200,
            'step_size': 200,
            'mode': 'rolling',
            'min_train_size': 500,
            'save_models': True,
            'risk_free_rate': 0.02,
            'output_dir': 'reports/walk_forward',
            'model_dir': 'models/walk_forward'
        },
        'rl_agent': {
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'replay_buffer_size': 100000,
            'target_update_frequency': 10
        },
        'trading': {
            'symbols': ['BTC/USDT'],
            'initial_balance': 10000,
            'timeframe': '1h'
        }
    }


class WalkForwardTrainer:
    """
    Trainer class that integrates RL agent training with walk-forward validation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.wf_config = config.get('walk_forward', {})
        self.rl_config = config.get('rl_agent', {})
        self.trading_config = config.get('trading', {})

        # Initialize environment (shared across folds)
        self.env = TradingEnvironment(
            symbols=self.trading_config.get('symbols', ['BTC/USDT']),
            initial_balance=self.trading_config.get('initial_balance', 10000),
            timeframe=self.trading_config.get('timeframe', '1h')
        )

        # Storage for episode data (simulated here)
        self.episode_data = self._generate_dummy_data()

        logger.info("WalkForwardTrainer initialized")

    def _generate_dummy_data(self) -> Dict[str, np.ndarray]:
        """
        Generate dummy episode data for demonstration

        In production, this would be replaced with real market data
        or preloaded historical data
        """
        total_episodes = self.wf_config.get('total_episodes', 2000)

        logger.info(f"Generating dummy data for {total_episodes} episodes")

        # Simulate returns with some drift and volatility
        np.random.seed(42)
        returns = np.random.randn(total_episodes) * 0.02 + 0.0001

        # Simulate rewards
        rewards = returns * 1000  # Scale to trading rewards

        return {
            'returns': returns,
            'rewards': rewards,
            'episode_ids': np.arange(total_episodes)
        }

    async def train_fold(
        self,
        train_start: int,
        train_end: int
    ) -> NexlifyRLAgent:
        """
        Train RL agent on specified episode range

        Args:
            train_start: Starting episode index
            train_end: Ending episode index (exclusive)

        Returns:
            Trained RL agent
        """
        logger.info(f"Training agent on episodes {train_start}-{train_end}")

        # Initialize fresh agent for this fold
        agent = NexlifyRLAgent(
            state_size=self.env.state_size,
            action_size=self.env.action_size,
            config=self.rl_config
        )

        # Training loop (simplified)
        num_train_episodes = train_end - train_start

        for i in range(num_train_episodes):
            episode_idx = train_start + i

            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action = agent.act(state)

                # Take step in environment
                next_state, reward, done, info = self.env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

                state = next_state
                episode_reward += reward

            # Log progress
            if (i + 1) % 100 == 0:
                logger.info(
                    f"  Episode {i+1}/{num_train_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Epsilon: {agent.epsilon:.3f}"
                )

        logger.info(f"Training completed for episodes {train_start}-{train_end}")

        return agent

    async def evaluate_fold(
        self,
        agent: NexlifyRLAgent,
        test_start: int,
        test_end: int
    ) -> Dict[str, float]:
        """
        Evaluate agent on test window

        Args:
            agent: Trained RL agent
            test_start: Starting episode index for testing
            test_end: Ending episode index for testing (exclusive)

        Returns:
            Dictionary of performance metrics
        """
        logger.info(f"Evaluating agent on episodes {test_start}-{test_end}")

        # Set agent to evaluation mode (no exploration)
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Greedy policy

        # Collect returns during evaluation
        returns_list = []
        rewards_list = []
        trades = []

        num_test_episodes = test_end - test_start

        for i in range(num_test_episodes):
            episode_idx = test_start + i

            # Reset environment
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                # Select action (greedy)
                action = agent.act(state)

                # Take step
                next_state, reward, done, info = self.env.step(action)

                state = next_state
                episode_reward += reward
                episode_steps += 1

            # Calculate episode return
            # In real implementation, this would come from portfolio value
            episode_return = self.episode_data['returns'][episode_idx]
            returns_list.append(episode_return)
            rewards_list.append(episode_reward)

            # Simulate trade recording
            if np.random.rand() > 0.3:  # ~70% of episodes have trades
                trades.append({
                    'profit': episode_reward,
                    'duration': episode_steps
                })

        # Restore epsilon
        agent.epsilon = original_epsilon

        # Calculate comprehensive metrics
        returns_array = np.array(returns_list)
        metrics = calculate_performance_metrics(
            returns=returns_array,
            trades=trades,
            risk_free_rate=self.wf_config.get('risk_free_rate', 0.02)
        )

        logger.info(
            f"Evaluation completed: "
            f"Return={metrics['total_return']:.2%}, "
            f"Sharpe={metrics['sharpe_ratio']:.2f}, "
            f"Win Rate={metrics['win_rate']:.2%}"
        )

        return metrics

    async def run_walk_forward_validation(self) -> WalkForwardResults:
        """
        Execute walk-forward validation

        Returns:
            WalkForwardResults object
        """
        logger.info("Starting walk-forward validation")

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

        # Run validation
        results = await validator.validate(
            train_fn=self.train_fold,
            eval_fn=self.evaluate_fold,
            save_models=self.wf_config.get('save_models', True),
            model_dir=Path(self.wf_config.get('model_dir', 'models/walk_forward'))
        )

        # Generate report
        output_dir = Path(self.wf_config.get('output_dir', 'reports/walk_forward'))
        validator.generate_report(results, output_dir)

        logger.info("Walk-forward validation completed successfully")

        return results


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Walk-forward validation example for RL trading agent'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/neural_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['rolling', 'expanding'],
        default='rolling',
        help='Walk-forward mode'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=2000,
        help='Total episodes for validation'
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=1000,
        help='Training window size'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=200,
        help='Test window size'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=200,
        help='Step size between folds'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if 'walk_forward' not in config:
        config['walk_forward'] = {}

    config['walk_forward'].update({
        'mode': args.mode,
        'total_episodes': args.episodes,
        'train_size': args.train_size,
        'test_size': args.test_size,
        'step_size': args.step_size
    })

    # Print configuration
    logger.info("Walk-Forward Validation Configuration:")
    logger.info(f"  Mode: {config['walk_forward']['mode']}")
    logger.info(f"  Total Episodes: {config['walk_forward']['total_episodes']}")
    logger.info(f"  Train Size: {config['walk_forward']['train_size']}")
    logger.info(f"  Test Size: {config['walk_forward']['test_size']}")
    logger.info(f"  Step Size: {config['walk_forward']['step_size']}")

    # Initialize trainer
    trainer = WalkForwardTrainer(config)

    # Run validation
    try:
        results = await trainer.run_walk_forward_validation()

        # Print summary
        print(results.summary())

        # Print model selection recommendation
        best_fold = results.fold_metrics[results.best_fold_id]
        print(f"\nRecommended Model: Fold {results.best_fold_id}")
        print(f"  Return: {best_fold.total_return:.2%}")
        print(f"  Sharpe Ratio: {best_fold.sharpe_ratio:.2f}")
        print(f"  Win Rate: {best_fold.win_rate:.2%}")
        print(f"  Max Drawdown: {best_fold.max_drawdown:.2%}")

        logger.info("Example completed successfully")

    except Exception as e:
        logger.error(f"Error during walk-forward validation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
