#!/usr/bin/env python3
"""
Nexlify RL Agent Training Script
Train DQN agent on historical cryptocurrency data
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

from nexlify.strategies.nexlify_rl_agent import TradingEnvironment, DQNAgent
from nexlify.utils.error_handler import get_error_handler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/rl_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class RLTrainer:
    """Manages RL agent training process"""

    def __init__(self, config: dict = None):
        self.config = config or self._default_config()

        # Paths
        self.models_dir = Path("models")
        self.data_dir = Path("data")
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Training parameters
        self.n_episodes = self.config.get('n_episodes', 1000)
        self.max_steps = self.config.get('max_steps', 1000)
        self.target_update_freq = self.config.get('target_update_freq', 10)
        self.save_freq = self.config.get('save_freq', 50)

        # Agent
        self.agent = None
        self.env = None

        # Training stats
        self.episode_rewards = []
        self.episode_profits = []
        self.episode_trades = []
        self.win_rates = []

        logger.info("🎓 RL Trainer initialized")

    def _default_config(self) -> dict:
        """Default training configuration"""
        return {
            'n_episodes': 1000,
            'max_steps': 1000,
            'target_update_freq': 10,
            'save_freq': 50,
            'agent': {
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'learning_rate': 0.001,
                'batch_size': 64
            }
        }

    async def fetch_historical_data(self, symbol: str = 'BTC/USDT',
                                   days: int = 180) -> np.ndarray:
        """
        Fetch historical price data for training

        Args:
            symbol: Trading pair
            days: Number of days of historical data

        Returns:
            numpy array of closing prices
        """
        try:
            import ccxt

            logger.info(f"📊 Fetching {days} days of {symbol} data...")

            exchange = ccxt.binance()
            since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())

            # Fetch OHLCV data (1 hour candles)
            ohlcv = await exchange.fetch_ohlcv(
                symbol,
                timeframe='1h',
                since=since,
                limit=days * 24
            )

            await exchange.close()

            # Extract closing prices
            prices = np.array([candle[4] for candle in ohlcv])

            logger.info(f"✅ Fetched {len(prices)} price points")

            # Save to file
            data_file = self.data_dir / f"training_data_{symbol.replace('/', '_')}.npy"
            np.save(data_file, prices)

            return prices

        except Exception as e:
            logger.error(f"Error fetching data: {e}")

            # Try to load from file
            data_file = self.data_dir / f"training_data_{symbol.replace('/', '_')}.npy"
            if data_file.exists():
                logger.warning("Loading cached data...")
                return np.load(data_file)

            # Generate synthetic data for testing
            logger.warning("Generating synthetic data for testing...")
            return self._generate_synthetic_data(days * 24)

    def _generate_synthetic_data(self, n_points: int) -> np.ndarray:
        """Generate synthetic price data for testing"""
        # Random walk with trend
        np.random.seed(42)

        prices = [45000]  # Starting BTC price
        for _ in range(n_points - 1):
            change = np.random.randn() * 500 + 10  # Slight upward trend
            prices.append(prices[-1] + change)

        return np.array(prices)

    def train(self, price_data: np.ndarray):
        """
        Train DQN agent on historical data

        Args:
            price_data: Array of historical prices
        """
        logger.info("="*70)
        logger.info("🚀 STARTING RL AGENT TRAINING")
        logger.info("="*70)

        # Create environment and agent
        self.env = TradingEnvironment(price_data)
        self.agent = DQNAgent(
            state_size=self.env.state_space_n,
            action_size=self.env.action_space_n,
            config=self.config.get('agent', {})
        )

        # Training loop
        for episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            trades = []

            for step in range(self.max_steps):
                # Agent chooses action
                action = self.agent.act(state, training=True)

                # Environment step
                next_state, reward, done, info = self.env.step(action)

                # Store experience
                self.agent.remember(state, action, reward, next_state, done)

                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.replay()

                episode_reward += reward
                state = next_state

                # Track trades
                if 'profit' in info:
                    trades.append(info)

                if done:
                    break

            # Decay exploration
            self.agent.decay_epsilon()

            # Update target network
            if episode % self.target_update_freq == 0:
                self.agent.update_target_model()

            # Calculate statistics
            portfolio_value = self.env.get_portfolio_value()
            profit = portfolio_value - self.env.initial_balance
            profit_percent = (profit / self.env.initial_balance) * 100

            winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
            win_rate = (winning_trades / len(trades) * 100) if trades else 0

            # Store stats
            self.episode_rewards.append(episode_reward)
            self.episode_profits.append(profit_percent)
            self.episode_trades.append(len(trades))
            self.win_rates.append(win_rate)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_profit = np.mean(self.episode_profits[-10:])
                avg_win_rate = np.mean(self.win_rates[-10:])

                logger.info(f"Episode {episode}/{self.n_episodes}")
                logger.info(f"  Avg Reward: {avg_reward:.4f}")
                logger.info(f"  Avg Profit: {avg_profit:+.2f}%")
                logger.info(f"  Avg Win Rate: {avg_win_rate:.1f}%")
                logger.info(f"  Trades: {len(trades)}")
                logger.info(f"  Epsilon: {self.agent.epsilon:.3f}")
                logger.info("-"*50)

            # Save checkpoint
            if episode % self.save_freq == 0:
                self.save_checkpoint(episode)

        logger.info("="*70)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("="*70)

        # Final save
        self.save_final_model()

        # Generate report
        self.generate_training_report()

    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_path = self.models_dir / f"rl_agent_checkpoint_ep{episode}.pth"
        self.agent.save(str(checkpoint_path))
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def save_final_model(self):
        """Save final trained model"""
        model_path = self.models_dir / "rl_agent_trained.pth"
        self.agent.save(str(model_path))

        # Save training config
        config_path = self.models_dir / "rl_agent_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"✅ Final model saved: {model_path}")

    def generate_training_report(self):
        """Generate training report with visualizations"""
        logger.info("📊 Generating training report...")

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Agent Training Results', fontsize=16)

        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)

        # Plot 2: Episode Profits
        axes[0, 1].plot(self.episode_profits)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Episode Profits (%)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Profit %')
        axes[0, 1].grid(True)

        # Plot 3: Win Rate
        axes[1, 0].plot(self.win_rates)
        axes[1, 0].axhline(y=50, color='r', linestyle='--')
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Rate %')
        axes[1, 0].grid(True)

        # Plot 4: Number of Trades
        axes[1, 1].plot(self.episode_trades)
        axes[1, 1].set_title('Trades per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('# Trades')
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Save plot
        report_path = self.models_dir / "training_report.png"
        plt.savefig(report_path, dpi=150)
        logger.info(f"✅ Report saved: {report_path}")

        # Generate text summary
        summary = {
            'training_date': datetime.now().isoformat(),
            'episodes': self.n_episodes,
            'final_stats': {
                'avg_reward': float(np.mean(self.episode_rewards[-100:])),
                'avg_profit': float(np.mean(self.episode_profits[-100:])),
                'avg_win_rate': float(np.mean(self.win_rates[-100:])),
                'best_profit': float(max(self.episode_profits)),
                'best_win_rate': float(max(self.win_rates))
            },
            'config': self.config
        }

        summary_path = self.models_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"✅ Summary saved: {summary_path}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("📈 TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total Episodes: {self.n_episodes}")
        logger.info(f"Final Avg Reward (last 100): {summary['final_stats']['avg_reward']:.4f}")
        logger.info(f"Final Avg Profit (last 100): {summary['final_stats']['avg_profit']:+.2f}%")
        logger.info(f"Final Avg Win Rate (last 100): {summary['final_stats']['avg_win_rate']:.1f}%")
        logger.info(f"Best Single Episode Profit: {summary['final_stats']['best_profit']:+.2f}%")
        logger.info(f"Best Win Rate: {summary['final_stats']['best_win_rate']:.1f}%")
        logger.info("="*70)


async def main():
    """Main training function"""
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  🤖 NEXLIFY REINFORCEMENT LEARNING TRAINING               ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    # Create trainer
    trainer = RLTrainer()

    # Fetch historical data
    price_data = await trainer.fetch_historical_data(
        symbol='BTC/USDT',
        days=180  # 6 months of data
    )

    # Train agent
    trainer.train(price_data)

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║  ✅ TRAINING COMPLETE!                                     ║
    ║                                                            ║
    ║  Model saved to: models/rl_agent_trained.pth              ║
    ║  Report saved to: models/training_report.png              ║
    ║                                                            ║
    ║  To use the trained agent:                                ║
    ║  1. Set 'use_rl_agent: true' in neural_config.json       ║
    ║  2. Launch Nexlify normally                               ║
    ║  3. The agent will use learned policies!                  ║
    ╚════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import sys

    # Check if data collection only
    if "--collect-only" in sys.argv:
        print("📊 Data collection mode...")
        trainer = RLTrainer()
        asyncio.run(trainer.fetch_historical_data('BTC/USDT', 180))
        print("✅ Data collection complete")
    else:
        # Full training
        asyncio.run(main())
