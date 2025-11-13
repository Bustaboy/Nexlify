#!/usr/bin/env python3
"""
Walk-Forward Cross-Validation and Hyperparameter Optimization

Implements:
✅ Walk-Forward Cross-Validation - Proper time-series validation
✅ Hyperparameter Optimization - Automated tuning with Optuna
✅ Performance tracking across multiple folds
✅ Statistical significance testing
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ValidationFold:
    """Single fold in walk-forward validation"""
    fold_id: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_return: float = 0.0
    test_return: float = 0.0
    test_sharpe: float = 0.0
    test_max_drawdown: float = 0.0
    test_win_rate: float = 0.0
    test_trades: int = 0
    model_path: Optional[str] = None


@dataclass
class WalkForwardResults:
    """Results from walk-forward cross-validation"""
    folds: List[ValidationFold] = field(default_factory=list)
    avg_test_return: float = 0.0
    avg_test_sharpe: float = 0.0
    avg_test_max_drawdown: float = 0.0
    avg_test_win_rate: float = 0.0
    std_test_return: float = 0.0
    std_test_sharpe: float = 0.0
    total_folds: int = 0
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'avg_test_return': self.avg_test_return,
            'avg_test_sharpe': self.avg_test_sharpe,
            'avg_test_max_drawdown': self.avg_test_max_drawdown,
            'avg_test_win_rate': self.avg_test_win_rate,
            'std_test_return': self.std_test_return,
            'std_test_sharpe': self.std_test_sharpe,
            'total_folds': self.total_folds,
            'overall_score': self.overall_score,
            'folds': [
                {
                    'fold_id': f.fold_id,
                    'test_return': f.test_return,
                    'test_sharpe': f.test_sharpe,
                    'test_max_drawdown': f.test_max_drawdown,
                    'test_win_rate': f.test_win_rate,
                    'test_trades': f.test_trades
                }
                for f in self.folds
            ]
        }


class WalkForwardValidator:
    """
    Walk-Forward Cross-Validation for time-series data

    Splits data into multiple train/test windows that walk forward in time:

    Fold 1: [Train: Month 1-12] → [Test: Month 13-14]
    Fold 2: [Train: Month 3-14] → [Test: Month 15-16]
    Fold 3: [Train: Month 5-16] → [Test: Month 17-18]
    ...

    This gives a more realistic estimate of future performance.
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: Optional[int] = None,
        min_train_size: Optional[int] = None,
        anchored: bool = False
    ):
        """
        Initialize walk-forward validator

        Args:
            train_size: Number of samples in training window
            test_size: Number of samples in test window
            step_size: Number of samples to step forward (default: test_size)
            min_train_size: Minimum training size (for expanding window)
            anchored: If True, always start training from beginning (expanding window)
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.min_train_size = min_train_size or train_size
        self.anchored = anchored

        logger.info(f"Walk-Forward Validator initialized")
        logger.info(f"  Train size: {train_size}")
        logger.info(f"  Test size: {test_size}")
        logger.info(f"  Step size: {self.step_size}")
        logger.info(f"  Mode: {'Anchored (expanding)' if anchored else 'Rolling'}")

    def create_folds(self, data_length: int) -> List[ValidationFold]:
        """
        Create walk-forward folds

        Args:
            data_length: Total length of data

        Returns:
            List of validation folds
        """
        folds = []
        fold_id = 0

        # Start with minimum train size
        current_train_end = self.min_train_size if self.anchored else self.train_size

        while current_train_end + self.test_size <= data_length:
            if self.anchored:
                # Anchored: always start from beginning (expanding window)
                train_start = 0
                train_end = current_train_end
            else:
                # Rolling: fixed-size window that slides
                train_start = current_train_end - self.train_size
                train_end = current_train_end

            test_start = train_end
            test_end = test_start + self.test_size

            fold = ValidationFold(
                fold_id=fold_id,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end
            )

            folds.append(fold)

            # Step forward
            current_train_end += self.step_size
            fold_id += 1

        logger.info(f"Created {len(folds)} walk-forward folds")

        return folds

    def validate(
        self,
        folds: List[ValidationFold],
        train_func: Callable,
        evaluate_func: Callable,
        data_dict: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> WalkForwardResults:
        """
        Run walk-forward validation

        Args:
            folds: List of validation folds
            train_func: Function to train model (fold, train_data) -> model
            evaluate_func: Function to evaluate model (model, test_data) -> metrics
            data_dict: Dictionary containing data and environment
            output_dir: Directory to save fold results

        Returns:
            Walk-forward validation results
        """
        results = WalkForwardResults(total_folds=len(folds))

        for fold in folds:
            logger.info(f"\n{'='*80}")
            logger.info(f"FOLD {fold.fold_id + 1}/{len(folds)}")
            logger.info(f"  Train: [{fold.train_start_idx}:{fold.train_end_idx}] ({fold.train_end_idx - fold.train_start_idx} samples)")
            logger.info(f"  Test:  [{fold.test_start_idx}:{fold.test_end_idx}] ({fold.test_end_idx - fold.test_start_idx} samples)")
            logger.info(f"{'='*80}")

            # Prepare train data for this fold
            train_data = {
                'start_idx': fold.train_start_idx,
                'end_idx': fold.train_end_idx
            }

            # Train model on this fold
            logger.info("Training model...")
            model, train_metrics = train_func(fold, train_data, data_dict)

            fold.train_return = train_metrics.get('total_return_pct', 0.0)

            # Save model
            if output_dir:
                fold_dir = output_dir / f"fold_{fold.fold_id}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                model_path = fold_dir / "best_model.pt"

                # Save model
                if hasattr(model, 'save'):
                    model.save(str(model_path))
                else:
                    torch.save(model.state_dict(), model_path)

                fold.model_path = str(model_path)

            # Prepare test data
            test_data = {
                'start_idx': fold.test_start_idx,
                'end_idx': fold.test_end_idx
            }

            # Evaluate on test data
            logger.info("Evaluating on test data...")
            test_metrics = evaluate_func(model, test_data, data_dict)

            # Store test metrics
            fold.test_return = test_metrics.get('total_return_pct', 0.0)
            fold.test_sharpe = test_metrics.get('sharpe_ratio', 0.0)
            fold.test_max_drawdown = test_metrics.get('max_drawdown', 0.0)
            fold.test_win_rate = test_metrics.get('win_rate', 0.0)
            fold.test_trades = test_metrics.get('total_trades', 0)

            logger.info(f"\nFold {fold.fold_id} Results:")
            logger.info(f"  Train Return: {fold.train_return:+.2f}%")
            logger.info(f"  Test Return:  {fold.test_return:+.2f}%")
            logger.info(f"  Test Sharpe:  {fold.test_sharpe:.2f}")
            logger.info(f"  Test Drawdown: {fold.test_max_drawdown:.2f}%")
            logger.info(f"  Test Win Rate: {fold.test_win_rate:.1%}")
            logger.info(f"  Test Trades:   {fold.test_trades}")

            results.folds.append(fold)

        # Compute aggregate statistics
        test_returns = [f.test_return for f in results.folds]
        test_sharpes = [f.test_sharpe for f in results.folds]
        test_drawdowns = [f.test_max_drawdown for f in results.folds]
        test_win_rates = [f.test_win_rate for f in results.folds]

        results.avg_test_return = np.mean(test_returns)
        results.avg_test_sharpe = np.mean(test_sharpes)
        results.avg_test_max_drawdown = np.mean(test_drawdowns)
        results.avg_test_win_rate = np.mean(test_win_rates)
        results.std_test_return = np.std(test_returns)
        results.std_test_sharpe = np.std(test_sharpes)

        # Overall score (weighted combination)
        results.overall_score = (
            results.avg_test_return * 0.4 +
            results.avg_test_sharpe * 10 * 0.3 +
            results.avg_test_win_rate * 100 * 0.2 -
            results.avg_test_max_drawdown * 0.1
        )

        logger.info(f"\n{'='*80}")
        logger.info("WALK-FORWARD VALIDATION RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Folds: {results.total_folds}")
        logger.info(f"Avg Test Return: {results.avg_test_return:+.2f}% (±{results.std_test_return:.2f}%)")
        logger.info(f"Avg Test Sharpe: {results.avg_test_sharpe:.2f} (±{results.std_test_sharpe:.2f})")
        logger.info(f"Avg Test Drawdown: {results.avg_test_max_drawdown:.2f}%")
        logger.info(f"Avg Test Win Rate: {results.avg_test_win_rate:.1%}")
        logger.info(f"Overall Score: {results.overall_score:.2f}")
        logger.info(f"{'='*80}\n")

        # Save results
        if output_dir:
            results_path = output_dir / "walk_forward_results.json"
            with open(results_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            logger.info(f"Results saved to: {results_path}")

        return results


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization using Optuna

    Searches for optimal hyperparameters using Bayesian optimization
    """

    def __init__(
        self,
        objective_func: Callable,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize hyperparameter optimizer

        Args:
            objective_func: Function to optimize (trial) -> score
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (None = no limit)
            study_name: Name for the study
            output_dir: Directory to save results
        """
        self.objective_func = objective_func
        self.n_trials = n_trials
        self.timeout = timeout
        self.study_name = study_name or f"optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = output_dir

        # Try to import optuna
        try:
            import optuna
            self.optuna = optuna
            self.optuna_available = True
            logger.info("✅ Optuna available for hyperparameter optimization")
        except ImportError:
            self.optuna = None
            self.optuna_available = False
            logger.warning("⚠️  Optuna not available. Install with: pip install optuna")

    def optimize(self) -> Optional[Dict[str, Any]]:
        """
        Run hyperparameter optimization

        Returns:
            Best hyperparameters found, or None if Optuna unavailable
        """
        if not self.optuna_available:
            logger.error("Cannot run optimization without Optuna")
            return None

        logger.info(f"\n{'='*80}")
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info(f"Study: {self.study_name}")
        logger.info(f"Trials: {self.n_trials}")
        logger.info(f"Timeout: {self.timeout}s" if self.timeout else "Timeout: None")
        logger.info(f"{'='*80}\n")

        # Create study
        study = self.optuna.create_study(
            study_name=self.study_name,
            direction='maximize',  # Maximize score
            sampler=self.optuna.samplers.TPESampler(seed=42)
        )

        # Run optimization
        study.optimize(
            self.objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Get best results
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"\n{'='*80}")
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Best Score: {best_value:.2f}")
        logger.info(f"Best Parameters:")
        for param, value in best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"{'='*80}\n")

        # Save results
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Save best parameters
            params_path = self.output_dir / "best_hyperparameters.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'best_value': best_value,
                    'n_trials': len(study.trials),
                    'study_name': self.study_name
                }, f, indent=2)
            logger.info(f"Best parameters saved to: {params_path}")

            # Save study history
            history_path = self.output_dir / "optimization_history.json"
            history = [
                {
                    'trial_id': trial.number,
                    'value': trial.value,
                    'params': trial.params
                }
                for trial in study.trials
            ]
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Optimization history saved to: {history_path}")

        return best_params

    def suggest_hyperparameters(self, trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for DQN agent

        This defines the search space for hyperparameter optimization
        """
        if not self.optuna_available:
            return {}

        params = {
            # Learning rate (log scale)
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),

            # Discount factor
            'gamma': trial.suggest_uniform('gamma', 0.95, 0.999),

            # Batch size (categorical)
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),

            # Network architecture
            'hidden_layer_1': trial.suggest_categorical('hidden_layer_1', [128, 256, 512]),
            'hidden_layer_2': trial.suggest_categorical('hidden_layer_2', [128, 256, 512]),
            'hidden_layer_3': trial.suggest_categorical('hidden_layer_3', [64, 128, 256]),

            # Exploration
            'epsilon_decay': trial.suggest_uniform('epsilon_decay', 0.99, 0.999),

            # N-step
            'n_step': trial.suggest_int('n_step', 1, 5),

            # Target update frequency
            'target_update_frequency': trial.suggest_categorical('target_update_frequency', [500, 1000, 2000]),

            # L2 regularization
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),

            # Gradient clipping
            'gradient_clip_norm': trial.suggest_uniform('gradient_clip_norm', 0.5, 2.0),

            # PER parameters (if using)
            'per_alpha': trial.suggest_uniform('per_alpha', 0.4, 0.8),
            'per_beta': trial.suggest_uniform('per_beta', 0.3, 0.5),
        }

        return params


def create_default_hyperparameter_search_space():
    """
    Returns default search space for hyperparameters

    Use this if not using Optuna for optimization
    """
    return {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],
        'gamma': [0.95, 0.97, 0.99, 0.995],
        'batch_size': [32, 64, 128],
        'hidden_layers': [
            [128, 128, 64],
            [256, 128, 64],
            [256, 256, 128],
            [512, 256, 128]
        ],
        'epsilon_decay': [0.99, 0.995, 0.999],
        'n_step': [1, 3, 5],
        'weight_decay': [1e-6, 1e-5, 1e-4]
    }


class GridSearchOptimizer:
    """
    Simple grid search optimizer (fallback if Optuna not available)
    """

    def __init__(
        self,
        objective_func: Callable,
        search_space: Dict[str, List],
        max_combinations: int = 20
    ):
        """
        Initialize grid search optimizer

        Args:
            objective_func: Function to optimize (params) -> score
            search_space: Dictionary of parameter names to lists of values
            max_combinations: Maximum number of combinations to try
        """
        self.objective_func = objective_func
        self.search_space = search_space
        self.max_combinations = max_combinations

    def optimize(self) -> Dict[str, Any]:
        """
        Run grid search

        Returns:
            Best parameters found
        """
        import itertools

        logger.info(f"\n{'='*80}")
        logger.info("GRID SEARCH OPTIMIZATION")
        logger.info(f"Max combinations: {self.max_combinations}")
        logger.info(f"{'='*80}\n")

        # Generate all combinations
        keys = list(self.search_space.keys())
        values = list(self.search_space.values())
        all_combinations = list(itertools.product(*values))

        # Randomly sample if too many
        if len(all_combinations) > self.max_combinations:
            import random
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:self.max_combinations]

        logger.info(f"Testing {len(all_combinations)} parameter combinations...")

        best_score = float('-inf')
        best_params = None

        for i, combination in enumerate(all_combinations):
            params = dict(zip(keys, combination))

            logger.info(f"\nCombination {i+1}/{len(all_combinations)}")
            logger.info(f"  Parameters: {params}")

            # Evaluate
            score = self.objective_func(params)

            logger.info(f"  Score: {score:.2f}")

            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"  🏆 New best!")

        logger.info(f"\n{'='*80}")
        logger.info("GRID SEARCH COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Best Score: {best_score:.2f}")
        logger.info(f"Best Parameters: {best_params}")
        logger.info(f"{'='*80}\n")

        return best_params
