#!/usr/bin/env python3
"""
Nexlify Paper Trading Runner

Main script for running paper trading sessions with RL/ML agents.
Supports training, evaluation, and live paper trading with real market data.

Features:
- Train RL agents in paper trading mode
- Evaluate multiple agents simultaneously
- Real-time market data integration
- Performance tracking and visualization
- Continuous learning support
"""

import asyncio
import logging
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nexlify.backtesting.nexlify_paper_trading_orchestrator import (
    PaperTradingOrchestrator,
    AgentConfig
)
from nexlify.environments.nexlify_rl_training_env import TradingEnvironment
from nexlify.strategies.nexlify_adaptive_rl_agent import create_optimized_agent
from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """
    Main runner for paper trading sessions

    Handles setup, execution, and management of paper trading sessions
    with single or multiple RL/ML agents.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize runner

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.orchestrator = None
        self.training_env = None

        # Setup logging
        self._setup_logging()

        logger.info("🚀 Paper Trading Runner initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded config from: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'paper_trading': {
                'initial_balance': 10000.0,
                'fee_rate': 0.001,
                'slippage': 0.0005,
                'update_interval': 60
            },
            'training': {
                'episodes': 100,
                'max_steps': 1000,
                'save_frequency': 10
            },
            'agents': [
                {
                    'agent_id': 'adaptive_rl_1',
                    'agent_type': 'rl_adaptive',
                    'name': 'Adaptive RL Agent',
                    'enabled': True
                }
            ],
            'logging': {
                'level': 'INFO',
                'file': 'paper_trading/logs/session.log'
            }
        }

    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Setup file logging if specified
        log_file = log_config.get('file')
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            )
            logging.getLogger().addHandler(file_handler)

    async def train_agent(self, agent_type: str = 'adaptive', episodes: int = 100):
        """
        Train a single RL agent using paper trading environment

        Args:
            agent_type: Type of agent ('adaptive' or 'ultra')
            episodes: Number of training episodes
        """
        logger.info(f"🎓 Starting RL agent training: {agent_type}")
        logger.info(f"   Episodes: {episodes}")

        # Create training environment
        self.training_env = TradingEnvironment(
            initial_balance=self.config['paper_trading']['initial_balance'],
            fee_rate=self.config['paper_trading']['fee_rate'],
            slippage=self.config['paper_trading']['slippage'],
            max_steps=self.config['training']['max_steps'],
            use_paper_trading=True
        )

        # Create agent
        if agent_type == 'adaptive':
            agent = create_optimized_agent(
                state_size=self.training_env.state_size,
                action_size=self.training_env.action_size,
                auto_detect=True
            )
        elif agent_type == 'ultra':
            agent = create_ultra_optimized_agent(
                state_size=self.training_env.state_size,
                action_size=self.training_env.action_size,
                profile=OptimizationProfile.AUTO,
                enable_sentiment=False
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Training loop
        save_frequency = self.config['training']['save_frequency']

        for episode in range(1, episodes + 1):
            state = self.training_env.reset()
            total_reward = 0
            done = False

            while not done:
                # Agent selects action
                action = agent.act(state, training=True)

                # Execute action in environment
                next_state, reward, done, info = self.training_env.step(action)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if hasattr(agent, 'replay'):
                    loss = agent.replay()

                state = next_state
                total_reward += reward

            # Decay epsilon
            if hasattr(agent, 'decay_epsilon'):
                agent.decay_epsilon()
            elif hasattr(agent, 'update_epsilon'):
                agent.update_epsilon()

            # Get episode stats
            episode_stats = self.training_env.episode_history[-1]

            logger.info(f"Episode {episode}/{episodes} completed:")
            logger.info(f"  Total Reward: {total_reward:.2f}")
            logger.info(f"  Final Equity: ${episode_stats.final_equity:,.2f}")
            logger.info(f"  Return: {episode_stats.total_return_percent:.2f}%")
            logger.info(f"  Win Rate: {episode_stats.win_rate:.1f}%")
            logger.info(f"  Trades: {episode_stats.num_trades}")

            # Save checkpoint
            if episode % save_frequency == 0:
                checkpoint_path = f"models/paper_trading_{agent_type}_episode_{episode}.pt"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                agent.save(checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_path = f"models/paper_trading_{agent_type}_final.pt"
        agent.save(final_path)
        logger.info(f"✅ Training completed! Final model saved: {final_path}")

        return agent

    async def run_multi_agent_session(self, duration_hours: Optional[float] = None):
        """
        Run paper trading session with multiple agents

        Args:
            duration_hours: Session duration in hours (None = indefinite)
        """
        logger.info("🎯 Starting multi-agent paper trading session")

        # Create orchestrator
        self.orchestrator = PaperTradingOrchestrator(
            self.config.get('paper_trading', {})
        )

        # Register agents from config
        for agent_config_dict in self.config.get('agents', []):
            agent_config = AgentConfig(**agent_config_dict)
            self.orchestrator.register_agent(agent_config)

            # Load and initialize agent instance
            agent_instance = await self._load_agent_instance(agent_config)
            if agent_instance:
                self.orchestrator.load_agent_model(agent_config.agent_id, agent_instance)

        # Start session
        await self.orchestrator.start_session(duration_hours=duration_hours)

        # Get final report
        report = self.orchestrator.generate_final_report()
        print(report)

        return self.orchestrator

    async def _load_agent_instance(self, agent_config: AgentConfig) -> Optional[Any]:
        """
        Load agent instance based on configuration

        Args:
            agent_config: Agent configuration

        Returns:
            Initialized agent instance or None
        """
        try:
            if agent_config.agent_type == 'rl_adaptive':
                agent = create_optimized_agent(
                    state_size=8,
                    action_size=3,
                    auto_detect=True,
                    config_override=agent_config.config
                )
            elif agent_config.agent_type == 'rl_ultra':
                agent = create_ultra_optimized_agent(
                    state_size=8,
                    action_size=3,
                    profile=OptimizationProfile.AUTO,
                    enable_sentiment=agent_config.config.get('enable_sentiment', False)
                )
            else:
                logger.warning(f"Unknown agent type: {agent_config.agent_type}")
                return None

            # Load model weights if path provided
            if agent_config.model_path and Path(agent_config.model_path).exists():
                agent.load(agent_config.model_path)
                logger.info(f"✅ Loaded model from: {agent_config.model_path}")

            return agent

        except Exception as e:
            logger.error(f"Error loading agent {agent_config.agent_id}: {e}")
            return None

    async def evaluate_agents(self, model_paths: List[str], episodes: int = 10):
        """
        Evaluate multiple trained agents

        Args:
            model_paths: List of paths to trained models
            episodes: Number of evaluation episodes per agent
        """
        logger.info(f"📊 Evaluating {len(model_paths)} agents over {episodes} episodes")

        results = {}

        for model_path in model_paths:
            if not Path(model_path).exists():
                logger.warning(f"Model not found: {model_path}")
                continue

            logger.info(f"\nEvaluating: {model_path}")

            # Create environment
            env = TradingEnvironment(
                initial_balance=self.config['paper_trading']['initial_balance'],
                use_paper_trading=True
            )

            # Load agent
            agent = create_optimized_agent(
                state_size=env.state_size,
                action_size=env.action_size,
                auto_detect=True
            )
            agent.load(model_path)

            # Run evaluation episodes
            episode_returns = []
            episode_win_rates = []

            for episode in range(episodes):
                state = env.reset()
                done = False
                total_return = 0

                while not done:
                    action = agent.act(state, training=False)  # No exploration
                    next_state, reward, done, info = env.step(action)
                    state = next_state

                episode_stats = env.episode_history[-1]
                episode_returns.append(episode_stats.total_return_percent)
                episode_win_rates.append(episode_stats.win_rate)

                logger.debug(f"  Episode {episode + 1}: Return={episode_stats.total_return_percent:.2f}%")

            # Calculate statistics
            results[model_path] = {
                'mean_return': np.mean(episode_returns),
                'std_return': np.std(episode_returns),
                'mean_win_rate': np.mean(episode_win_rates),
                'best_return': np.max(episode_returns),
                'worst_return': np.min(episode_returns)
            }

            logger.info(f"Results for {Path(model_path).name}:")
            logger.info(f"  Mean Return: {results[model_path]['mean_return']:.2f}% ± {results[model_path]['std_return']:.2f}%")
            logger.info(f"  Mean Win Rate: {results[model_path]['mean_win_rate']:.1f}%")
            logger.info(f"  Best: {results[model_path]['best_return']:.2f}%, Worst: {results[model_path]['worst_return']:.2f}%")

        # Print comparison
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)

        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_return'], reverse=True)

        for rank, (model_path, stats) in enumerate(sorted_results, 1):
            logger.info(f"{rank}. {Path(model_path).name}")
            logger.info(f"   Mean Return: {stats['mean_return']:.2f}% ± {stats['std_return']:.2f}%")
            logger.info(f"   Win Rate: {stats['mean_win_rate']:.1f}%")

        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Nexlify Paper Trading Runner')

    parser.add_argument('mode', choices=['train', 'evaluate', 'multi-agent'],
                       help='Operation mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--agent-type', type=str, default='adaptive',
                       choices=['adaptive', 'ultra'],
                       help='Agent type for training')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--duration', type=float,
                       help='Session duration in hours (for multi-agent mode)')
    parser.add_argument('--models', nargs='+',
                       help='Model paths for evaluation')

    args = parser.parse_args()

    # Create runner
    runner = PaperTradingRunner(config_path=args.config)

    # Run based on mode
    if args.mode == 'train':
        asyncio.run(runner.train_agent(
            agent_type=args.agent_type,
            episodes=args.episodes
        ))

    elif args.mode == 'evaluate':
        if not args.models:
            print("Error: --models required for evaluate mode")
            sys.exit(1)

        import numpy as np
        asyncio.run(runner.evaluate_agents(
            model_paths=args.models,
            episodes=args.episodes
        ))

    elif args.mode == 'multi-agent':
        asyncio.run(runner.run_multi_agent_session(
            duration_hours=args.duration
        ))


if __name__ == "__main__":
    main()
