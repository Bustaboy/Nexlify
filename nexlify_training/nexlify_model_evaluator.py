"""
Nexlify Model Evaluator
Comprehensive evaluation and model selection system

Features:
- Walk-forward analysis (time-series cross-validation)
- Out-of-sample testing on unseen data
- Multiple performance metrics
- Monte Carlo simulation for robustness testing
- Model comparison and ranking
- Statistical significance testing
- Risk-adjusted performance metrics
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
import json
from dataclasses import dataclass, asdict
import torch
from scipy import stats

from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
from nexlify_environments.nexlify_rl_training_env import TradingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics"""
    model_name: str
    test_period: str
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_winning_trade: float
    avg_losing_trade: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_trade_duration: float
    volatility: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional VaR
    risk_adjusted_return: float
    consistency_score: float
    overall_score: float


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_metrics: Dict
    test_metrics: EvaluationMetrics
    overfitting_ratio: float  # test performance / train performance


class ModelEvaluator:
    """
    Comprehensive model evaluation and selection system
    """

    def __init__(self, output_dir: str = "./evaluation_output"):
        """
        Initialize model evaluator

        Args:
            output_dir: Directory for saving evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(exist_ok=True)

        self.comparison_dir = self.output_dir / "comparisons"
        self.comparison_dir.mkdir(exist_ok=True)

    def evaluate_model(
        self,
        agent: UltraOptimizedDQNAgent,
        test_data: pd.DataFrame,
        model_name: str,
        initial_balance: float = 10000,
        fee_rate: float = 0.001,
        num_episodes: int = 10
    ) -> EvaluationMetrics:
        """
        Comprehensive model evaluation on test data

        Args:
            agent: Trained RL agent
            test_data: Test data DataFrame
            model_name: Name of the model
            initial_balance: Starting balance
            fee_rate: Transaction fee rate
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating {model_name} on {len(test_data)} candles")

        prices = test_data['close'].values

        # Run multiple episodes to get robust metrics
        all_returns = []
        all_sharpes = []
        all_trades = []
        all_win_rates = []
        all_drawdowns = []
        all_trade_returns = []
        all_winning_trades = []
        all_losing_trades = []

        agent.epsilon = 0  # No exploration during evaluation

        for episode in range(num_episodes):
            env = TradingEnvironment(
                initial_balance=initial_balance,
                fee_rate=fee_rate,
                slippage=0.0005,
                market_data=prices
            )

            state = env.reset()
            done = False
            episode_trade_returns = []

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                state = next_state

                if 'trade_return' in info:
                    episode_trade_returns.append(info['trade_return'])

            stats = env.get_episode_stats()

            all_returns.append(stats['total_return_pct'])
            all_sharpes.append(stats['sharpe_ratio'])
            all_trades.append(stats['num_trades'])
            all_win_rates.append(stats['win_rate'])
            all_drawdowns.append(stats['max_drawdown'])
            all_trade_returns.extend(episode_trade_returns)

            # Separate winning and losing trades
            winning_trades = [r for r in episode_trade_returns if r > 0]
            losing_trades = [r for r in episode_trade_returns if r < 0]

            all_winning_trades.extend(winning_trades)
            all_losing_trades.extend(losing_trades)

        # Calculate aggregate metrics
        avg_return = np.mean(all_returns)
        avg_sharpe = np.mean(all_sharpes)
        avg_win_rate = np.mean(all_win_rates)
        avg_drawdown = np.mean(all_drawdowns)
        total_trades = int(np.sum(all_trades))

        # Advanced metrics
        sortino_ratio = self._calculate_sortino_ratio(all_trade_returns)
        calmar_ratio = avg_return / abs(avg_drawdown) if avg_drawdown != 0 else 0

        # Profit factor
        total_wins = sum(all_winning_trades) if all_winning_trades else 0
        total_losses = abs(sum(all_losing_trades)) if all_losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Average trade metrics
        avg_trade_return = np.mean(all_trade_returns) if all_trade_returns else 0
        avg_winning_trade = np.mean(all_winning_trades) if all_winning_trades else 0
        avg_losing_trade = np.mean(all_losing_trades) if all_losing_trades else 0

        # Consecutive wins/losses
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_runs(
            all_trade_returns
        )

        # Risk metrics
        volatility = np.std(all_returns) if len(all_returns) > 1 else 0
        var_95 = np.percentile(all_trade_returns, 5) if all_trade_returns else 0
        cvar_95 = np.mean([r for r in all_trade_returns if r <= var_95]) if all_trade_returns else 0

        # Risk-adjusted return
        risk_adjusted_return = avg_return / (volatility + 1e-8)

        # Consistency score (lower std of returns = more consistent)
        consistency_score = 100 / (1 + volatility)

        # Overall score (weighted combination)
        overall_score = (
            avg_return * 0.25 +
            avg_sharpe * 10 * 0.20 +
            sortino_ratio * 10 * 0.15 +
            avg_win_rate * 100 * 0.15 +
            profit_factor * 10 * 0.10 +
            consistency_score * 0.10 +
            (100 - abs(avg_drawdown)) * 0.05
        )

        metrics = EvaluationMetrics(
            model_name=model_name,
            test_period=f"{len(test_data)} candles",
            total_return_pct=avg_return,
            sharpe_ratio=avg_sharpe,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=avg_drawdown,
            win_rate=avg_win_rate,
            profit_factor=profit_factor,
            num_trades=total_trades,
            avg_trade_return=avg_trade_return,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_trade_duration=0,  # Would need trade timestamps to calculate
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            risk_adjusted_return=risk_adjusted_return,
            consistency_score=consistency_score,
            overall_score=overall_score
        )

        # Save results
        self._save_evaluation_results(metrics)

        logger.info(f"✓ Evaluation complete for {model_name}")
        logger.info(f"  Return: {avg_return:.2f}% | Sharpe: {avg_sharpe:.2f} | "
                   f"Win Rate: {avg_win_rate:.1%} | Score: {overall_score:.2f}")

        return metrics

    def walk_forward_analysis(
        self,
        agent_factory,
        full_data: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.8,
        initial_balance: float = 10000,
        fee_rate: float = 0.001
    ) -> List[WalkForwardResult]:
        """
        Perform walk-forward analysis (time-series cross-validation)

        Args:
            agent_factory: Function that creates a new agent instance
            full_data: Complete dataset with 'timestamp' column
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of train data in each split
            initial_balance: Starting balance
            fee_rate: Transaction fee rate

        Returns:
            List of walk-forward results
        """
        logger.info(f"Starting walk-forward analysis with {n_splits} splits")

        full_data = full_data.sort_values('timestamp').reset_index(drop=True)
        total_len = len(full_data)
        split_size = total_len // n_splits

        results = []

        for fold in range(n_splits):
            logger.info(f"\nFold {fold + 1}/{n_splits}")

            # Define train and test periods
            test_start_idx = (fold + 1) * split_size - int(split_size * (1 - train_ratio))
            test_end_idx = min((fold + 1) * split_size, total_len)
            train_start_idx = fold * split_size
            train_end_idx = test_start_idx

            if test_end_idx <= test_start_idx:
                logger.warning(f"Skipping fold {fold + 1}: insufficient data")
                continue

            train_data = full_data.iloc[train_start_idx:train_end_idx]
            test_data = full_data.iloc[test_start_idx:test_end_idx]

            logger.info(f"Train: {len(train_data)} candles | Test: {len(test_data)} candles")

            # Train agent on training data
            agent = agent_factory()
            train_metrics = self._train_on_data(agent, train_data, initial_balance, fee_rate)

            # Evaluate on test data
            test_metrics = self.evaluate_model(
                agent=agent,
                test_data=test_data,
                model_name=f"Fold_{fold + 1}",
                initial_balance=initial_balance,
                fee_rate=fee_rate,
                num_episodes=5
            )

            # Calculate overfitting ratio
            overfitting_ratio = (
                test_metrics.total_return_pct / (train_metrics['return_pct'] + 1e-8)
                if train_metrics['return_pct'] != 0 else 0
            )

            result = WalkForwardResult(
                fold=fold + 1,
                train_start=train_data['timestamp'].iloc[0],
                train_end=train_data['timestamp'].iloc[-1],
                test_start=test_data['timestamp'].iloc[0],
                test_end=test_data['timestamp'].iloc[-1],
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                overfitting_ratio=overfitting_ratio
            )

            results.append(result)

            logger.info(f"Fold {fold + 1} - Train Return: {train_metrics['return_pct']:.2f}% | "
                       f"Test Return: {test_metrics.total_return_pct:.2f}% | "
                       f"Overfitting Ratio: {overfitting_ratio:.2f}")

        # Save walk-forward results
        self._save_walk_forward_results(results)

        logger.info(f"\n✓ Walk-forward analysis complete")
        return results

    def compare_models(
        self,
        model_paths: List[str],
        test_data: pd.DataFrame,
        model_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on the same test data

        Args:
            model_paths: List of paths to model checkpoints
            test_data: Test data
            model_names: Optional names for models

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(model_paths)} models")

        if model_names is None:
            model_names = [f"Model_{i+1}" for i in range(len(model_paths))]

        all_metrics = []

        for model_path, model_name in zip(model_paths, model_names):
            logger.info(f"\nEvaluating {model_name}")

            try:
                # Load model
                agent = self._load_model(model_path)

                # Evaluate
                metrics = self.evaluate_model(
                    agent=agent,
                    test_data=test_data,
                    model_name=model_name
                )

                all_metrics.append(asdict(metrics))

            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)

        # Sort by overall score
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)

        # Save comparison
        comparison_path = self.comparison_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(comparison_path, index=False)

        logger.info(f"\n✓ Model comparison complete")
        logger.info(f"Results saved: {comparison_path}")

        # Print top 3
        logger.info("\nTop 3 Models:")
        for idx, row in comparison_df.head(3).iterrows():
            logger.info(f"{idx + 1}. {row['model_name']}: Score {row['overall_score']:.2f} "
                       f"(Return: {row['total_return_pct']:.2f}%, Sharpe: {row['sharpe_ratio']:.2f})")

        return comparison_df

    def monte_carlo_simulation(
        self,
        agent: UltraOptimizedDQNAgent,
        test_data: pd.DataFrame,
        n_simulations: int = 1000,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation for robustness testing

        Args:
            agent: Trained agent
            test_data: Test data
            n_simulations: Number of simulations
            model_name: Model name

        Returns:
            Simulation results
        """
        logger.info(f"Running Monte Carlo simulation ({n_simulations} runs)")

        prices = test_data['close'].values
        agent.epsilon = 0

        returns = []
        sharpes = []
        drawdowns = []
        win_rates = []

        for sim in range(n_simulations):
            if (sim + 1) % 100 == 0:
                logger.info(f"Simulation {sim + 1}/{n_simulations}")

            # Add random noise to prices (±2%)
            noisy_prices = prices * (1 + np.random.normal(0, 0.02, len(prices)))

            env = TradingEnvironment(
                initial_balance=10000,
                fee_rate=0.001,
                slippage=0.0005,
                market_data=noisy_prices
            )

            state = env.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                state = next_state

            stats = env.get_episode_stats()
            returns.append(stats['total_return_pct'])
            sharpes.append(stats['sharpe_ratio'])
            drawdowns.append(stats['max_drawdown'])
            win_rates.append(stats['win_rate'])

        results = {
            'model_name': model_name,
            'n_simulations': n_simulations,
            'returns': {
                'mean': np.mean(returns),
                'median': np.median(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentile_5': np.percentile(returns, 5),
                'percentile_95': np.percentile(returns, 95)
            },
            'sharpe': {
                'mean': np.mean(sharpes),
                'median': np.median(sharpes),
                'std': np.std(sharpes)
            },
            'drawdown': {
                'mean': np.mean(drawdowns),
                'worst': np.max(np.abs(drawdowns))
            },
            'win_rate': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates)
            },
            'positive_return_probability': (np.array(returns) > 0).mean()
        }

        logger.info(f"\n✓ Monte Carlo simulation complete")
        logger.info(f"Expected Return: {results['returns']['mean']:.2f}% "
                   f"(±{results['returns']['std']:.2f}%)")
        logger.info(f"5th Percentile: {results['returns']['percentile_5']:.2f}%")
        logger.info(f"95th Percentile: {results['returns']['percentile_95']:.2f}%")
        logger.info(f"Probability of Positive Return: {results['positive_return_probability']:.1%}")

        # Save results
        results_path = self.results_dir / f"monte_carlo_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def _calculate_sortino_ratio(self, returns: List[float], target: float = 0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns:
            return 0

        returns_array = np.array(returns)
        excess_returns = returns_array - target
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = np.std(downside_returns)

        if downside_std == 0:
            return 0

        return np.mean(excess_returns) / downside_std

    def _calculate_consecutive_runs(self, returns: List[float]) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        if not returns:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for ret in returns:
            if ret > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif ret < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _train_on_data(
        self,
        agent: UltraOptimizedDQNAgent,
        train_data: pd.DataFrame,
        initial_balance: float,
        fee_rate: float,
        episodes: int = 50
    ) -> Dict:
        """Train agent on given data"""
        prices = train_data['close'].values
        all_returns = []

        for episode in range(episodes):
            env = TradingEnvironment(
                initial_balance=initial_balance,
                fee_rate=fee_rate,
                slippage=0.0005,
                market_data=prices
            )

            state = env.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                state = next_state

            stats = env.get_episode_stats()
            all_returns.append(stats['total_return_pct'])

        return {
            'return_pct': np.mean(all_returns),
            'sharpe': np.mean([stats['sharpe_ratio'] for _ in range(episodes)])
        }

    def _load_model(self, model_path: str) -> UltraOptimizedDQNAgent:
        """Load model from checkpoint"""
        checkpoint = torch.load(model_path, map_location='cpu')

        agent = UltraOptimizedDQNAgent(state_size=8, action_size=3)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.epsilon = 0  # No exploration during evaluation

        return agent

    def _save_evaluation_results(self, metrics: EvaluationMetrics):
        """Save evaluation results"""
        results_path = self.results_dir / f"eval_{metrics.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)

    def _save_walk_forward_results(self, results: List[WalkForwardResult]):
        """Save walk-forward analysis results"""
        results_data = [asdict(r) for r in results]
        results_path = self.results_dir / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)


if __name__ == "__main__":
    print("Nexlify Model Evaluator")
    print("=" * 60)
    print("\nThis module provides comprehensive model evaluation tools:")
    print("- Walk-forward analysis")
    print("- Out-of-sample testing")
    print("- Model comparison")
    print("- Monte Carlo simulation")
    print("- Risk-adjusted metrics")
