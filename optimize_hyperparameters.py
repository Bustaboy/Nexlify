#!/usr/bin/env python3
"""
Nexlify Hyperparameter Optimization Runner
Automated hyperparameter tuning for RL trading agents using Optuna

Usage:
    python optimize_hyperparameters.py --objective sharpe --trials 100 --symbol BTC/USDT
    python optimize_hyperparameters.py --objective multi --trials 50 --quick-test
    python optimize_hyperparameters.py --objective return --trials 200 --timeout 86400
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from nexlify.optimization import (
    HyperparameterTuner,
    create_custom_search_space,
    create_objective,
    SharpeObjective,
    ReturnObjective,
    DrawdownObjective,
    MultiObjective
)
from nexlify.optimization.analysis_tools import OptimizationAnalyzer, OptimizationVisualizer

# Import training infrastructure
try:
    from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig
    from nexlify_training.nexlify_advanced_training_orchestrator import AdvancedTrainingOrchestrator
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logging.warning("Training modules not available, using mock training function")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HyperparameterOptimizationRunner:
    """
    Runner for hyperparameter optimization
    """

    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        exchange: str = 'binance',
        data_years: int = 2,
        quick_test: bool = False,
        output_dir: str = './optimization_results'
    ):
        """
        Initialize optimization runner

        Args:
            symbol: Trading pair
            exchange: Exchange name
            data_years: Years of historical data
            quick_test: Quick test mode (less data, fewer episodes)
            output_dir: Output directory
        """
        self.symbol = symbol
        self.exchange = exchange
        self.data_years = data_years
        self.quick_test = quick_test
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Cached data
        self.train_data = None
        self.val_data = None

        logger.info(f"Initialized optimization runner:")
        logger.info(f"  Symbol: {symbol}")
        logger.info(f"  Exchange: {exchange}")
        logger.info(f"  Data years: {data_years}")
        logger.info(f"  Quick test: {quick_test}")

    def fetch_data(self) -> None:
        """Fetch and cache training/validation data"""
        logger.info("Fetching historical data...")

        if not TRAINING_AVAILABLE:
            # Mock data for testing
            logger.warning("Using mock data (training modules not available)")
            self.train_data = np.random.randn(1000, 10)
            self.val_data = np.random.randn(200, 10)
            return

        # Fetch real data
        end_date = datetime.now()
        train_start = end_date - timedelta(days=365 * self.data_years)
        val_start = end_date - timedelta(days=90)

        fetcher = HistoricalDataFetcher()

        # Training data
        try:
            train_config = FetchConfig(
                exchange=self.exchange,
                symbol=self.symbol,
                timeframe='1h',
                start_date=train_start,
                end_date=val_start
            )
            self.train_data = fetcher.fetch_data(train_config)
            logger.info(f"Fetched {len(self.train_data)} training candles")
        except Exception as e:
            logger.error(f"Failed to fetch training data: {e}")
            raise

        # Validation data
        try:
            val_config = FetchConfig(
                exchange=self.exchange,
                symbol=self.symbol,
                timeframe='1h',
                start_date=val_start,
                end_date=end_date
            )
            self.val_data = fetcher.fetch_data(val_config)
            logger.info(f"Fetched {len(self.val_data)} validation candles")
        except Exception as e:
            logger.error(f"Failed to fetch validation data: {e}")
            raise

    def create_training_function(self) -> callable:
        """
        Create training function compatible with HyperparameterTuner

        Returns:
            Training function with signature:
                train_func(params, train_data, val_data) -> results_dict
        """
        def train_agent(
            params: Dict[str, Any],
            train_data: Any,
            val_data: Any
        ) -> Dict[str, Any]:
            """
            Train RL agent with given hyperparameters

            Args:
                params: Hyperparameters
                train_data: Training data
                val_data: Validation data

            Returns:
                Dict with training results including metrics
            """
            try:
                if not TRAINING_AVAILABLE:
                    # Mock training results
                    return self._mock_training(params)

                # Real training
                return self._real_training(params, train_data, val_data)

            except Exception as e:
                logger.error(f"Training failed: {e}")
                # Return poor results on failure
                return {
                    'sharpe_ratio': -10.0,
                    'total_return': -1.0,
                    'max_drawdown': -0.5,
                    'final_balance': 5000,
                    'initial_balance': 10000,
                    'returns': [],
                    'balance_history': [10000, 5000]
                }

        return train_agent

    def _mock_training(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock training for testing (returns random results)

        Args:
            params: Hyperparameters

        Returns:
            Mock training results
        """
        # Simulate training based on params
        # Better params -> better results (roughly)
        lr_score = -np.log10(params.get('learning_rate', 0.001))
        gamma_score = params.get('gamma', 0.95)
        batch_size_score = np.log2(params.get('batch_size', 64)) / 10

        base_score = (lr_score + gamma_score + batch_size_score) / 3
        noise = np.random.randn() * 0.1

        sharpe = base_score + noise
        total_return = sharpe * 0.5 + np.random.randn() * 0.1

        # Generate mock returns
        n_returns = 100
        returns = np.random.randn(n_returns) * 0.02 + total_return / n_returns

        # Generate balance history
        balance_history = [10000]
        for ret in returns:
            balance_history.append(balance_history[-1] * (1 + ret))

        return {
            'sharpe_ratio': float(sharpe),
            'total_return': float(total_return),
            'max_drawdown': float(np.random.uniform(-0.3, -0.05)),
            'final_balance': balance_history[-1],
            'initial_balance': 10000,
            'returns': returns.tolist(),
            'balance_history': balance_history,
            'win_rate': float(np.random.uniform(0.4, 0.7)),
            'trades': returns.tolist()
        }

    def _real_training(
        self,
        params: Dict[str, Any],
        train_data: Any,
        val_data: Any
    ) -> Dict[str, Any]:
        """
        Real training using Nexlify training infrastructure

        Args:
            params: Hyperparameters
            train_data: Training data
            val_data: Validation data

        Returns:
            Training results
        """
        # Adjust episodes for quick test
        n_episodes = 100 if self.quick_test else 500

        # Create training config
        training_config = {
            'n_episodes': n_episodes,
            'gamma': params.get('gamma', 0.95),
            'learning_rate': params.get('learning_rate', 0.001),
            'epsilon_decay_steps': params.get('epsilon_decay_steps', 1000),
            'batch_size': params.get('batch_size', 64),
            'hidden_layers': params.get('hidden_layers', [128, 128]),
            'buffer_size': params.get('buffer_size', 100000),
            'n_step': params.get('n_step', 1),
            'target_update_frequency': params.get('target_update_frequency', 100),
            'tau': params.get('tau', 0.01),
        }

        # Create orchestrator
        orchestrator = AdvancedTrainingOrchestrator(
            config=training_config,
            output_dir=str(self.output_dir / 'trials')
        )

        # Train agent
        logger.info(f"Training with config: {json.dumps(training_config, indent=2)}")
        results = orchestrator.train(
            train_data=train_data,
            val_data=val_data
        )

        # Extract metrics
        return {
            'sharpe_ratio': results.get('val_sharpe_ratio', -10.0),
            'total_return': results.get('val_total_return', -1.0),
            'max_drawdown': results.get('val_max_drawdown', -0.5),
            'final_balance': results.get('val_final_balance', 5000),
            'initial_balance': results.get('val_initial_balance', 10000),
            'returns': results.get('val_returns', []),
            'balance_history': results.get('val_balance_history', []),
            'win_rate': results.get('val_win_rate', 0.0),
            'trades': results.get('val_trades', [])
        }

    def run_optimization(
        self,
        objective_type: str = 'sharpe',
        n_trials: int = 100,
        timeout: Optional[int] = None,
        search_space_type: str = 'default',
        sampler: str = 'tpe',
        pruner: str = 'median',
        n_jobs: int = 1
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization

        Args:
            objective_type: Objective type ('sharpe', 'return', 'multi', etc.)
            n_trials: Number of trials
            timeout: Timeout in seconds
            search_space_type: Search space ('default', 'compact', 'advanced')
            sampler: Sampler algorithm ('tpe', 'random', 'cmaes')
            pruner: Pruner strategy ('median', 'none')
            n_jobs: Number of parallel jobs

        Returns:
            Optimization results
        """
        # Fetch data if not cached
        if self.train_data is None or self.val_data is None:
            self.fetch_data()

        # Create objective
        if objective_type == 'multi':
            # Balanced multi-objective
            objective = MultiObjective([
                (SharpeObjective(), 0.4),
                (ReturnObjective(), 0.3),
                (DrawdownObjective(), 0.3)
            ], name='balanced')
        else:
            objective = create_objective(objective_type)

        # Create search space
        search_space = create_custom_search_space(base_space=search_space_type)

        # Create tuner
        tuner = HyperparameterTuner(
            objective=objective,
            search_space=search_space,
            n_trials=n_trials,
            timeout=timeout,
            sampler=sampler,
            pruner=pruner,
            n_jobs=n_jobs,
            output_dir=str(self.output_dir),
            verbose=True
        )

        # Create training function
        train_func = self.create_training_function()

        # Run optimization
        logger.info("\n" + "="*80)
        logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80 + "\n")

        results = tuner.optimize(
            train_func=train_func,
            train_data=self.train_data,
            validation_data=self.val_data
        )

        # Generate report
        report = tuner.generate_report(
            output_path=str(self.output_dir / 'optimization_report.txt')
        )
        print("\n" + report)

        # Create visualizations
        logger.info("\nGenerating visualizations...")
        try:
            analyzer = OptimizationAnalyzer(study=tuner.study)
            visualizer = OptimizationVisualizer(study=tuner.study, analyzer=analyzer)

            viz_dir = self.output_dir / 'visualizations'
            visualizer.create_comprehensive_report(str(viz_dir))

            # Print analysis
            convergence = analyzer.analyze_convergence()
            logger.info("\nConvergence Analysis:")
            logger.info(f"  Best value: {convergence['best_value']:.4f}")
            logger.info(f"  Found at trial: {convergence['best_trial_number']}")
            logger.info(f"  Converged: {convergence['converged']}")

            sensitivity = analyzer.analyze_parameter_sensitivity()
            logger.info("\nTop 5 Most Sensitive Parameters:")
            for i, (param, metrics) in enumerate(list(sensitivity.items())[:5], 1):
                logger.info(f"  {i}. {param}: correlation = {metrics['correlation']:.4f}")

        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Nexlify Hyperparameter Optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data arguments
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading pair (default: BTC/USDT)')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange name (default: binance)')
    parser.add_argument('--data-years', type=int, default=2,
                       help='Years of historical data (default: 2)')

    # Optimization arguments
    parser.add_argument('--objective', type=str, default='sharpe',
                       choices=['sharpe', 'return', 'drawdown', 'multi'],
                       help='Optimization objective (default: sharpe)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds (default: no timeout)')
    parser.add_argument('--search-space', type=str, default='default',
                       choices=['default', 'compact', 'advanced'],
                       help='Search space type (default: default)')
    parser.add_argument('--sampler', type=str, default='tpe',
                       choices=['tpe', 'random', 'cmaes'],
                       help='Sampling algorithm (default: tpe)')
    parser.add_argument('--pruner', type=str, default='median',
                       choices=['median', 'none'],
                       help='Pruning strategy (default: median)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel trials (default: 1)')

    # Other arguments
    parser.add_argument('--output-dir', type=str, default='./optimization_results',
                       help='Output directory (default: ./optimization_results)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (less data, fewer episodes)')

    args = parser.parse_args()

    # Create runner
    runner = HyperparameterOptimizationRunner(
        symbol=args.symbol,
        exchange=args.exchange,
        data_years=args.data_years,
        quick_test=args.quick_test,
        output_dir=args.output_dir
    )

    # Run optimization
    results = runner.run_optimization(
        objective_type=args.objective,
        n_trials=args.trials,
        timeout=args.timeout,
        search_space_type=args.search_space,
        sampler=args.sampler,
        pruner=args.pruner,
        n_jobs=args.n_jobs
    )

    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Best hyperparameters saved to {args.output_dir}/best_params_*.json")
    logger.info(f"Full report available at {args.output_dir}/optimization_report.txt")
    logger.info(f"Visualizations available at {args.output_dir}/visualizations/")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()
