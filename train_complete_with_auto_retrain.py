#!/usr/bin/env python3
"""
Nexlify COMPLETE Features Training with Auto-Retraining
=========================================================

This is the ULTIMATE training script combining:
✅ ALL Nexlify features (risk management + strategies)
✅ Multi-start initialization (3 independent runs)
✅ Auto-retraining from best model until marginal improvements plateau

PHASE 1: Multi-Start Initialization
- Runs 3 independent training sessions with different random seeds
- Evaluates each on validation data
- Selects the best performing model

PHASE 2: Auto-Retraining
- Continues training from best initial model
- Tracks marginal improvements
- Stops when improvements < 1.0% or no improvement for 3 iterations
- Maximum 10 retraining iterations

Features included:
✅ Stop-loss orders (2%)
✅ Take-profit orders (5%)
✅ Trailing stops (3%)
✅ Kelly Criterion position sizing
✅ Daily loss limits (5%)
✅ Max concurrent trades (3)
✅ Position size limits (5%)
✅ Multi-pair spot trading
✅ DeFi staking
✅ Liquidity provision
✅ Arbitrage
"""

import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json
import numpy as np
import torch
import random

sys.path.append(str(Path(__file__).parent))

from nexlify_environments.nexlify_complete_strategy_env import (
    CompleteMultiStrategyEnvironment,
    RiskLimits
)
from nexlify_rl_models.nexlify_ultra_optimized_rl_agent import UltraOptimizedDQNAgent
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
from nexlify_data.nexlify_external_features import ExternalFeatureEnricher
from nexlify_preflight_checker import PreFlightChecker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompleteAutoRetrainingOrchestrator:
    """
    Orchestrates multi-start initialization + auto-retraining with marginal improvement tracking
    """

    def __init__(
        self,
        output_dir: str = "./complete_auto_training_output",
        num_initial_runs: int = 3,
        improvement_threshold: float = 1.0,  # Minimum % improvement to continue
        patience: int = 3,  # Number of non-improving iterations before stopping
        max_iterations: int = 10  # Maximum retraining iterations
    ):
        """
        Initialize orchestrator

        Args:
            output_dir: Output directory
            num_initial_runs: Number of independent initial training runs
            improvement_threshold: Minimum improvement % to continue training
            patience: Number of iterations without improvement before stopping
            max_iterations: Maximum number of retraining iterations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_initial_runs = num_initial_runs
        self.improvement_threshold = improvement_threshold
        self.patience = patience
        self.max_iterations = max_iterations

        self.training_history = []
        self.initial_runs_history = []
        self.best_score = float('-inf')
        self.best_model_path = None
        self.no_improvement_count = 0

        logger.info(f"Complete Auto-Retraining Orchestrator initialized")
        logger.info(f"Initial runs: {num_initial_runs}")
        logger.info(f"Improvement threshold: {improvement_threshold}%")
        logger.info(f"Patience: {patience} iterations")
        logger.info(f"Max retraining iterations: {max_iterations}")

    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _train_single_run(
        self,
        run_id: int,
        seed: int,
        env: CompleteMultiStrategyEnvironment,
        episodes: int,
        run_dir: Path
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Run a single training session

        Returns:
            (model_path, best_return, final_stats)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Run {run_id} | Seed: {seed}")
        logger.info(f"{'='*80}")

        self._set_seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create fresh agent
        agent = UltraOptimizedDQNAgent(
            state_size=env.state_size,
            action_size=env.action_size
        )

        best_return = float('-inf')
        best_model_path = None
        final_stats = {}

        # Training loop
        for episode in range(1, episodes + 1):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
                episode_reward += reward
                state = next_state

            stats = env.get_episode_stats()
            final_stats = stats

            if episode % 10 == 0:
                logger.info(
                    f"Run {run_id} | Ep {episode}/{episodes} | "
                    f"Return: {stats['total_return_pct']:+.2f}% | "
                    f"Equity: ${stats['final_equity']:,.2f} | "
                    f"Trades: {stats['total_trades']} | "
                    f"Sharpe: {stats['sharpe_ratio']:.2f} | "
                    f"DD: {stats['max_drawdown']:.1f}% | "
                    f"ε: {agent.epsilon:.3f}"
                )

            # Save best model for this run
            if stats['total_return_pct'] > best_return:
                best_return = stats['total_return_pct']
                model_path = run_dir / f"model_return{best_return:.1f}.pt"

                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.model.state_dict(),
                    'target_model_state_dict': agent.target_model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'epsilon': agent.epsilon,
                    'stats': stats,
                    'seed': seed
                }, model_path)

                best_model_path = str(model_path)

        logger.info(f"✓ Run {run_id} complete | Best return: {best_return:.2f}%")
        return best_model_path, best_return, final_stats

    def _evaluate_on_validation(
        self,
        model_path: str,
        validation_env: CompleteMultiStrategyEnvironment
    ) -> Dict[str, float]:
        """
        Evaluate model on validation environment

        Returns:
            Metrics dict with score, return, sharpe, win_rate, etc.
        """
        # Load model
        checkpoint = torch.load(model_path)
        state_size = validation_env.state_size
        action_size = validation_env.action_size

        agent = UltraOptimizedDQNAgent(state_size=state_size, action_size=action_size)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.epsilon = 0.01  # Minimal exploration for evaluation

        # Run evaluation episodes
        eval_episodes = 10
        returns = []
        sharpes = []
        win_rates = []
        max_drawdowns = []

        for ep in range(eval_episodes):
            state = validation_env.reset()
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done, info = validation_env.step(action)
                state = next_state

            stats = validation_env.get_episode_stats()
            returns.append(stats['total_return_pct'])
            sharpes.append(stats['sharpe_ratio'])
            win_rates.append(stats['win_rate'])
            max_drawdowns.append(stats['max_drawdown'])

        # Calculate overall score (weighted combination)
        avg_return = np.mean(returns)
        avg_sharpe = np.mean(sharpes)
        avg_win_rate = np.mean(win_rates)
        avg_drawdown = np.mean(max_drawdowns)

        # Overall score: prioritize return and sharpe, penalize drawdown
        score = avg_return * 0.4 + avg_sharpe * 10 * 0.3 + avg_win_rate * 100 * 0.2 - avg_drawdown * 0.1

        return {
            'score': score,
            'avg_return': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate': avg_win_rate,
            'avg_drawdown': avg_drawdown,
            'returns_std': np.std(returns)
        }

    def run_initial_training_rounds(
        self,
        training_env: CompleteMultiStrategyEnvironment,
        validation_env: CompleteMultiStrategyEnvironment,
        episodes_per_run: int,
        base_seed: int = 42
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Run multiple independent initial training rounds and select the best

        Returns:
            (best_model_path, best_score, best_metrics)
        """
        logger.info("\n" + "#"*80)
        logger.info("PHASE 1: MULTI-START INITIALIZATION")
        logger.info(f"Running {self.num_initial_runs} independent training sessions")
        logger.info("#"*80 + "\n")

        initial_results = []

        for run_id in range(1, self.num_initial_runs + 1):
            seed = base_seed + run_id * 1000
            run_dir = self.output_dir / "initial_runs" / f"run_{run_id}"

            # Train
            model_path, train_return, train_stats = self._train_single_run(
                run_id=run_id,
                seed=seed,
                env=training_env,
                episodes=episodes_per_run,
                run_dir=run_dir
            )

            # Evaluate on validation data
            logger.info(f"Evaluating Run {run_id} on validation data...")
            val_metrics = self._evaluate_on_validation(model_path, validation_env)

            result = {
                'run_id': run_id,
                'seed': seed,
                'model_path': model_path,
                'train_return': train_return,
                'val_score': val_metrics['score'],
                'val_metrics': val_metrics,
                'train_stats': train_stats
            }
            initial_results.append(result)
            self.initial_runs_history.append(result)

            logger.info(f"Run {run_id} Results:")
            logger.info(f"  Training return: {train_return:.2f}%")
            logger.info(f"  Validation score: {val_metrics['score']:.2f}")
            logger.info(f"  Validation return: {val_metrics['avg_return']:.2f}%")
            logger.info(f"  Validation Sharpe: {val_metrics['avg_sharpe']:.2f}")
            logger.info(f"  Validation win rate: {val_metrics['avg_win_rate']:.1%}")

        # Select best based on validation score
        best_result = max(initial_results, key=lambda x: x['val_score'])

        logger.info("\n" + "="*80)
        logger.info("INITIAL RUNS COMPARISON")
        logger.info("="*80)
        for result in sorted(initial_results, key=lambda x: x['val_score'], reverse=True):
            marker = " ⭐ BEST" if result['run_id'] == best_result['run_id'] else ""
            logger.info(
                f"Run {result['run_id']} | "
                f"Val Score: {result['val_score']:>6.2f} | "
                f"Val Return: {result['val_metrics']['avg_return']:>+6.2f}% | "
                f"Sharpe: {result['val_metrics']['avg_sharpe']:>5.2f}{marker}"
            )
        logger.info("="*80 + "\n")

        logger.info(f"✓ Selected Run {best_result['run_id']} for retraining")
        logger.info(f"  Model: {best_result['model_path']}")

        return best_result['model_path'], best_result['val_score'], best_result['val_metrics']

    def run_retraining_iterations(
        self,
        initial_model_path: str,
        initial_score: float,
        training_env: CompleteMultiStrategyEnvironment,
        validation_env: CompleteMultiStrategyEnvironment,
        episodes_per_iteration: int
    ) -> Tuple[str, float]:
        """
        Continue training from best initial model until marginal improvements plateau

        Returns:
            (final_best_model_path, final_best_score)
        """
        logger.info("\n" + "#"*80)
        logger.info("PHASE 2: AUTO-RETRAINING FROM BEST MODEL")
        logger.info(f"Starting score: {initial_score:.2f}")
        logger.info(f"Improvement threshold: {self.improvement_threshold}%")
        logger.info(f"Patience: {self.patience} iterations")
        logger.info("#"*80 + "\n")

        self.best_score = initial_score
        self.best_model_path = initial_model_path
        self.no_improvement_count = 0

        iteration = 1
        current_model_path = initial_model_path

        while iteration <= self.max_iterations:
            logger.info(f"\n{'='*80}")
            logger.info(f"RETRAINING ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*80}")

            iteration_dir = self.output_dir / "retraining" / f"iteration_{iteration}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            # Load previous best model
            checkpoint = torch.load(current_model_path)
            agent = UltraOptimizedDQNAgent(
                state_size=training_env.state_size,
                action_size=training_env.action_size
            )
            agent.model.load_state_dict(checkpoint['model_state_dict'])
            agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.epsilon = max(0.01, checkpoint.get('epsilon', 0.1) * 0.9)  # Decay epsilon

            logger.info(f"Loaded model from: {current_model_path}")
            logger.info(f"Epsilon: {agent.epsilon:.3f}")

            # Continue training
            best_iteration_return = float('-inf')
            best_iteration_model = None

            for episode in range(1, episodes_per_iteration + 1):
                state = training_env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action = agent.act(state)
                    next_state, reward, done, info = training_env.step(action)
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                    episode_reward += reward
                    state = next_state

                stats = training_env.get_episode_stats()

                if episode % 10 == 0:
                    logger.info(
                        f"Iter {iteration} | Ep {episode}/{episodes_per_iteration} | "
                        f"Return: {stats['total_return_pct']:+.2f}% | "
                        f"Sharpe: {stats['sharpe_ratio']:.2f} | "
                        f"DD: {stats['max_drawdown']:.1f}% | "
                        f"ε: {agent.epsilon:.3f}"
                    )

                # Save if best in this iteration
                if stats['total_return_pct'] > best_iteration_return:
                    best_iteration_return = stats['total_return_pct']
                    model_path = iteration_dir / f"model_return{best_iteration_return:.1f}.pt"

                    torch.save({
                        'iteration': iteration,
                        'episode': episode,
                        'model_state_dict': agent.model.state_dict(),
                        'target_model_state_dict': agent.target_model.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                        'epsilon': agent.epsilon,
                        'stats': stats
                    }, model_path)

                    best_iteration_model = str(model_path)

            # Evaluate iteration's best model
            logger.info(f"\nEvaluating iteration {iteration} on validation data...")
            val_metrics = self._evaluate_on_validation(best_iteration_model, validation_env)
            current_score = val_metrics['score']

            # Calculate improvement
            improvement_pct = ((current_score - self.best_score) / abs(self.best_score)) * 100

            logger.info(f"\nIteration {iteration} Results:")
            logger.info(f"  Score: {current_score:.2f}")
            logger.info(f"  Previous best: {self.best_score:.2f}")
            logger.info(f"  Improvement: {improvement_pct:+.2f}%")
            logger.info(f"  Return: {val_metrics['avg_return']:.2f}%")
            logger.info(f"  Sharpe: {val_metrics['avg_sharpe']:.2f}")
            logger.info(f"  Win Rate: {val_metrics['avg_win_rate']:.1%}")

            # Track history
            iteration_result = {
                'iteration': iteration,
                'score': current_score,
                'improvement_pct': improvement_pct,
                'metrics': val_metrics,
                'model_path': best_iteration_model,
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(iteration_result)

            # Check if improved
            if current_score > self.best_score:
                logger.info(f"🏆 NEW BEST MODEL!")
                self.best_score = current_score
                self.best_model_path = best_iteration_model
                self.no_improvement_count = 0
                current_model_path = best_iteration_model
            else:
                self.no_improvement_count += 1
                logger.info(f"No improvement ({self.no_improvement_count}/{self.patience})")

            # Check stopping criteria
            if self.no_improvement_count >= self.patience:
                logger.info(f"\n⚠️ Stopping: No improvement for {self.patience} iterations")
                break

            if improvement_pct < self.improvement_threshold and iteration > 1:
                logger.info(f"\n⚠️ Stopping: Improvement ({improvement_pct:.2f}%) below threshold ({self.improvement_threshold}%)")
                break

            iteration += 1

        logger.info(f"\n✓ Retraining complete")
        logger.info(f"  Final best score: {self.best_score:.2f}")
        logger.info(f"  Total iterations: {len(self.training_history)}")

        return self.best_model_path, self.best_score

    def run_complete_pipeline(
        self,
        pairs: List[str],
        exchange: str,
        years: int,
        initial_balance: float,
        risk_limits: RiskLimits,
        episodes_initial: int,
        episodes_retrain: int,
        automated: bool = False
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline: multi-start + auto-retraining

        Returns:
            Complete training summary
        """
        start_time = datetime.now()

        logger.info("\n" + "="*80)
        logger.info("NEXLIFY COMPLETE AUTO-RETRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Pairs: {', '.join(pairs)}")
        logger.info(f"Episodes (initial): {episodes_initial}")
        logger.info(f"Episodes (retrain): {episodes_retrain}")
        logger.info(f"Initial runs: {self.num_initial_runs}")
        logger.info(f"Max retraining iterations: {self.max_iterations}")
        logger.info("="*80 + "\n")

        # Fetch historical data
        logger.info("Fetching historical data...")
        fetcher = HistoricalDataFetcher(automated_mode=automated)
        enricher = ExternalFeatureEnricher(automated_mode=automated)

        end_date = datetime.now()
        train_start = end_date - timedelta(days=years * 365)
        val_start = end_date - timedelta(days=90)  # Last 90 days for validation

        # Training data
        train_market_data = {}
        for pair in pairs:
            logger.info(f"Fetching training data for {pair}...")
            config = FetchConfig(
                exchange=exchange,
                symbol=pair,
                timeframe='1h',
                start_date=train_start,
                end_date=val_start,  # Train up to validation period
                cache_enabled=True
            )
            df, quality = fetcher.fetch_historical_data(config)
            if not df.empty:
                df = enricher.enrich_dataframe(df, symbol=pair)
                train_market_data[pair] = df['close'].values
                logger.info(f"✓ {pair}: {len(df)} training candles")

        # Validation data
        val_market_data = {}
        for pair in pairs:
            logger.info(f"Fetching validation data for {pair}...")
            config = FetchConfig(
                exchange=exchange,
                symbol=pair,
                timeframe='1h',
                start_date=val_start,
                end_date=end_date,
                cache_enabled=True
            )
            df, quality = fetcher.fetch_historical_data(config)
            if not df.empty:
                df = enricher.enrich_dataframe(df, symbol=pair)
                val_market_data[pair] = df['close'].values
                logger.info(f"✓ {pair}: {len(df)} validation candles")

        if not train_market_data or not val_market_data:
            logger.error("Failed to fetch sufficient data")
            return {}

        # Create environments
        training_env = CompleteMultiStrategyEnvironment(
            trading_pairs=list(train_market_data.keys()),
            initial_balance=initial_balance,
            market_data=train_market_data,
            risk_limits=risk_limits,
            enable_staking=True,
            enable_defi=True,
            enable_arbitrage=True
        )

        validation_env = CompleteMultiStrategyEnvironment(
            trading_pairs=list(val_market_data.keys()),
            initial_balance=initial_balance,
            market_data=val_market_data,
            risk_limits=risk_limits,
            enable_staking=True,
            enable_defi=True,
            enable_arbitrage=True
        )

        logger.info(f"\nEnvironments created:")
        logger.info(f"  State size: {training_env.state_size}")
        logger.info(f"  Action size: {training_env.action_size}")
        logger.info(f"  Training steps: {len(list(train_market_data.values())[0])}")
        logger.info(f"  Validation steps: {len(list(val_market_data.values())[0])}")

        # Phase 1: Multi-start initialization
        best_initial_model, best_initial_score, initial_metrics = self.run_initial_training_rounds(
            training_env=training_env,
            validation_env=validation_env,
            episodes_per_run=episodes_initial
        )

        # Phase 2: Auto-retraining
        final_model, final_score = self.run_retraining_iterations(
            initial_model_path=best_initial_model,
            initial_score=best_initial_score,
            training_env=training_env,
            validation_env=validation_env,
            episodes_per_iteration=episodes_retrain
        )

        # Training complete
        elapsed = datetime.now() - start_time

        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"Initial best score: {best_initial_score:.2f}")
        logger.info(f"Final best score: {final_score:.2f}")
        logger.info(f"Total improvement: {((final_score - best_initial_score) / abs(best_initial_score)) * 100:+.2f}%")
        logger.info(f"Time elapsed: {elapsed}")
        logger.info(f"Best model: {final_model}")
        logger.info("="*80 + "\n")

        # Save complete summary
        summary = {
            'pipeline_complete': True,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'elapsed_seconds': elapsed.total_seconds(),
            'initial_runs': self.initial_runs_history,
            'retraining_iterations': self.training_history,
            'best_initial_score': best_initial_score,
            'final_best_score': final_score,
            'total_improvement_pct': ((final_score - best_initial_score) / abs(best_initial_score)) * 100,
            'final_model_path': final_model,
            'pairs': pairs,
            'risk_limits': risk_limits.__dict__
        }

        summary_path = self.output_dir / "complete_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"✓ Summary saved: {summary_path}")

        return summary


def main():
    parser = argparse.ArgumentParser(
        description="Nexlify Complete Auto-Retraining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is the ULTIMATE training script combining:
✅ ALL Nexlify features (risk management + strategies)
✅ Multi-start initialization (3 independent runs)
✅ Auto-retraining until marginal improvements plateau

PHASE 1: Runs 3 independent training sessions, picks best
PHASE 2: Auto-retrains from best until improvements < 1%

Examples:
  # Full training pipeline
  python train_complete_with_auto_retrain.py \\
      --pairs BTC/USDT ETH/USDT SOL/USDT \\
      --initial-episodes 300 \\
      --retrain-episodes 200

  # Quick test
  python train_complete_with_auto_retrain.py --quick-test

  # Automated
  python train_complete_with_auto_retrain.py --automated
        """
    )

    parser.add_argument('--pairs', nargs='+', default=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                        help='Trading pairs')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange')
    parser.add_argument('--initial-episodes', type=int, default=300,
                        help='Episodes per initial run')
    parser.add_argument('--retrain-episodes', type=int, default=200,
                        help='Episodes per retraining iteration')
    parser.add_argument('--years', type=int, default=2, help='Years of training data')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--output', type=str, default='./complete_auto_training_output',
                        help='Output directory')
    parser.add_argument('--automated', action='store_true', help='Automated mode')
    parser.add_argument('--skip-preflight', action='store_true', help='Skip pre-flight checks')
    parser.add_argument('--quick-test', action='store_true', help='Quick test mode')

    # Multi-start parameters
    parser.add_argument('--initial-runs', type=int, default=3,
                        help='Number of initial independent runs')

    # Auto-retraining parameters
    parser.add_argument('--improvement-threshold', type=float, default=1.0,
                        help='Minimum improvement %% to continue retraining')
    parser.add_argument('--patience', type=int, default=3,
                        help='Iterations without improvement before stopping')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum retraining iterations')

    # Risk management overrides
    parser.add_argument('--stop-loss', type=float, default=0.02)
    parser.add_argument('--take-profit', type=float, default=0.05)
    parser.add_argument('--trailing-stop', type=float, default=0.03)
    parser.add_argument('--max-position', type=float, default=0.05)
    parser.add_argument('--max-trades', type=int, default=3)
    parser.add_argument('--no-kelly', action='store_true')

    args = parser.parse_args()

    # Quick test adjustments
    if args.quick_test:
        args.pairs = ['BTC/USDT']
        args.years = 1
        args.initial_episodes = 100
        args.retrain_episodes = 100
        args.initial_runs = 2
        args.max_iterations = 3
        logger.info("⚡ Quick test mode")

    # Print configuration
    print("\n" + "="*80)
    print("NEXLIFY COMPLETE AUTO-RETRAINING PIPELINE")
    print("="*80)
    print(f"Pairs: {', '.join(args.pairs)}")
    print(f"Initial runs: {args.initial_runs} × {args.initial_episodes} episodes")
    print(f"Retraining: up to {args.max_iterations} × {args.retrain_episodes} episodes")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\nRisk Management:")
    print(f"  Stop-loss: {args.stop_loss*100:.1f}%")
    print(f"  Take-profit: {args.take_profit*100:.1f}%")
    print(f"  Trailing stop: {args.trailing_stop*100:.1f}%")
    print(f"  Max position: {args.max_position*100:.1f}%")
    print(f"  Max trades: {args.max_trades}")
    print(f"  Kelly Criterion: {'✅' if not args.no_kelly else '❌'}")
    print("="*80 + "\n")

    # Pre-flight checks
    if not args.skip_preflight:
        logger.info("Running pre-flight checks...")
        for pair in args.pairs:
            checker = PreFlightChecker(symbol=pair, exchange=args.exchange)
            can_proceed, _ = checker.run_all_checks(automated_mode=args.automated)
            if not can_proceed:
                logger.error(f"Pre-flight failed for {pair}")
                if not args.automated:
                    return 1
        logger.info("✓ Pre-flight checks passed\n")

    # Create risk limits
    risk_limits = RiskLimits(
        max_position_size=args.max_position,
        max_daily_loss=0.05,
        stop_loss_percent=args.stop_loss,
        take_profit_percent=args.take_profit,
        trailing_stop_percent=args.trailing_stop,
        max_concurrent_trades=args.max_trades,
        use_kelly_criterion=not args.no_kelly,
        kelly_fraction=0.5
    )

    # Create orchestrator
    orchestrator = CompleteAutoRetrainingOrchestrator(
        output_dir=args.output,
        num_initial_runs=args.initial_runs,
        improvement_threshold=args.improvement_threshold,
        patience=args.patience,
        max_iterations=args.max_iterations
    )

    # Run complete pipeline
    summary = orchestrator.run_complete_pipeline(
        pairs=args.pairs,
        exchange=args.exchange,
        years=args.years,
        initial_balance=args.balance,
        risk_limits=risk_limits,
        episodes_initial=args.initial_episodes,
        episodes_retrain=args.retrain_episodes,
        automated=args.automated
    )

    if summary:
        print("\n✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"\nNext steps:")
        print(f"  1. Review results: {args.output}/complete_training_summary.json")
        print(f"  2. Best model: {summary['final_model_path']}")
        print(f"  3. Test in paper trading")
        print(f"  4. Deploy to live (carefully!)")
        return 0
    else:
        print("\n❌ Training pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
