"""
Hyperparameter Tuner
Main optimization engine using Optuna for automated hyperparameter tuning
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner, NopPruner
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.trial import Trial, TrialState
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

from nexlify.optimization.hyperparameter_space import (
    HyperparameterSpace,
    DEFAULT_SEARCH_SPACE,
    validate_hyperparameters
)
from nexlify.optimization.objective_functions import (
    ObjectiveFunction,
    create_objective,
    SharpeObjective
)
from nexlify.utils.error_handler import get_error_handler

logger = logging.getLogger(__name__)
error_handler = get_error_handler()


class HyperparameterTuner:
    """
    Automated hyperparameter optimization using Optuna

    Supports:
    - Multiple optimization algorithms (TPE, Random, CMA-ES)
    - Early stopping with pruning
    - Parallel trials
    - Resume from checkpoint
    - Result visualization and analysis
    """

    def __init__(
        self,
        objective: Optional[ObjectiveFunction] = None,
        objective_type: str = 'sharpe',
        search_space: Optional[HyperparameterSpace] = None,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        sampler: str = 'tpe',
        pruner: Optional[str] = 'median',
        pruner_warmup: int = 10,
        pruner_interval: int = 5,
        n_jobs: int = 1,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
        output_dir: str = './optimization_results',
        verbose: bool = True
    ):
        """
        Initialize hyperparameter tuner

        Args:
            objective: ObjectiveFunction instance (if None, creates from objective_type)
            objective_type: Type of objective ('sharpe', 'return', 'drawdown', etc.)
            search_space: HyperparameterSpace instance (if None, uses DEFAULT_SEARCH_SPACE)
            n_trials: Number of optimization trials
            timeout: Timeout in seconds (None = no timeout)
            sampler: Sampling algorithm ('tpe', 'random', 'cmaes')
            pruner: Pruning strategy ('median', 'none', None = no pruning)
            pruner_warmup: Number of trials before pruning starts
            pruner_interval: Steps between pruning checks
            n_jobs: Number of parallel trials (1 = sequential)
            study_name: Name for Optuna study (for resuming)
            storage: Storage URL for study persistence (e.g., 'sqlite:///optuna.db')
            load_if_exists: Load existing study if available
            output_dir: Directory for saving results
            verbose: Enable verbose logging
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )

        # Objective function
        if objective is None:
            self.objective = create_objective(objective_type)
        else:
            self.objective = objective

        # Search space
        if search_space is None:
            self.search_space = HyperparameterSpace(DEFAULT_SEARCH_SPACE)
        else:
            self.search_space = search_space

        # Optimization parameters
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs

        # Sampler
        self.sampler = self._create_sampler(sampler)

        # Pruner
        self.pruner = self._create_pruner(
            pruner, warmup=pruner_warmup, interval=pruner_interval
        )

        # Study management
        self.study_name = study_name or f"nexlify_optim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.load_if_exists = load_if_exists

        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # State
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.optimization_history: List[Dict[str, Any]] = []

        logger.info(f"Initialized HyperparameterTuner:")
        logger.info(f"  Objective: {self.objective.name}")
        logger.info(f"  Search space: {len(self.search_space.get_parameter_names())} parameters")
        logger.info(f"  Trials: {n_trials}")
        logger.info(f"  Sampler: {sampler}")
        logger.info(f"  Pruner: {pruner}")
        logger.info(f"  Parallel jobs: {n_jobs}")

    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler"""
        sampler_map = {
            'tpe': TPESampler,
            'random': RandomSampler,
            'cmaes': CmaEsSampler,
        }

        if sampler_type not in sampler_map:
            raise ValueError(
                f"Unknown sampler '{sampler_type}'. "
                f"Choose from: {list(sampler_map.keys())}"
            )

        sampler_class = sampler_map[sampler_type]
        return sampler_class(seed=42)  # Fixed seed for reproducibility

    def _create_pruner(
        self,
        pruner_type: Optional[str],
        warmup: int = 10,
        interval: int = 5
    ) -> optuna.pruners.BasePruner:
        """Create Optuna pruner"""
        if pruner_type is None or pruner_type == 'none':
            return NopPruner()

        pruner_map = {
            'median': MedianPruner,
        }

        if pruner_type not in pruner_map:
            raise ValueError(
                f"Unknown pruner '{pruner_type}'. "
                f"Choose from: {list(pruner_map.keys())} or 'none'"
            )

        pruner_class = pruner_map[pruner_type]
        return pruner_class(
            n_startup_trials=warmup,
            n_warmup_steps=interval,
            interval_steps=interval
        )
    def optimize(
        self,
        train_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        train_data: Optional[Any] = None,
        validation_data: Optional[Any] = None,
        fixed_params: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization

        Args:
            train_func: Training function with signature:
                train_func(hyperparameters, train_data, validation_data) -> results_dict
                Results dict must contain metrics needed by objective function
            train_data: Training data to pass to train_func
            validation_data: Validation data to pass to train_func
            fixed_params: Parameters to keep fixed (not optimized)
            callbacks: Optional callbacks for Optuna study

        Returns:
            Dict containing:
                - best_params: Best hyperparameters found
                - best_value: Best objective value
                - study: Optuna study object
                - optimization_history: List of trial results

        Example:
            >>> def train_agent(params, train_data, val_data):
            ...     agent = create_agent(**params)
            ...     agent.train(train_data)
            ...     metrics = agent.evaluate(val_data)
            ...     return metrics  # Should contain 'sharpe_ratio' or 'returns'
            >>>
            >>> tuner = HyperparameterTuner(objective_type='sharpe', n_trials=50)
            >>> results = tuner.optimize(train_agent, train_data, val_data)
            >>> print(f"Best params: {results['best_params']}")
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)

        start_time = time.time()

        # Create or load study
        direction = 'maximize' if self.objective.direction == 'maximize' else 'minimize'

        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=self.load_if_exists,
            direction=direction,
            sampler=self.sampler,
            pruner=self.pruner
        )

        # Define objective function for Optuna
        def objective_wrapper(trial: Trial) -> float:
            """Wrapper for Optuna trial"""
            try:
                # Suggest hyperparameters
                params = self.search_space.suggest_hyperparameters(trial)

                # Add fixed parameters
                if fixed_params:
                    params.update(fixed_params)

                # Validate hyperparameters
                is_valid, errors = validate_hyperparameters(params)
                if not is_valid:
                    logger.warning(f"Invalid hyperparameters: {errors}")
                    raise optuna.TrialPruned()

                # Log trial info
                if self.verbose:
                    logger.info(f"\nTrial {trial.number + 1}/{self.n_trials}")
                    logger.info(f"Parameters: {json.dumps(params, indent=2)}")

                # Train with these hyperparameters
                training_results = train_func(params, train_data, validation_data)

                # Calculate objective
                score = self.objective.calculate(training_results)

                # Store trial info
                trial_info = {
                    'trial_number': trial.number,
                    'params': params,
                    'score': score,
                    'training_results': training_results,
                    'timestamp': datetime.now().isoformat()
                }
                self.optimization_history.append(trial_info)

                if self.verbose:
                    logger.info(f"Score: {score:.4f}")

                # Report for pruning (if intermediate values available)
                if 'intermediate_scores' in training_results:
                    for step, value in enumerate(training_results['intermediate_scores']):
                        trial.report(value, step)
                        if trial.should_prune():
                            logger.info(f"Trial pruned at step {step}")
                            raise optuna.TrialPruned()

                return score

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                error_handler.log_error(e, context={'trial': trial.number, 'params': params})
                # Return worst possible score
                return float('-inf') if direction == 'maximize' else float('inf')

        # Run optimization
        logger.info(f"Running optimization with {self.n_trials} trials...")
        logger.info(f"Objective: {self.objective.name} ({direction})")

        self.study.optimize(
            objective_wrapper,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
            show_progress_bar=self.verbose
        )

        # Extract results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        elapsed_time = time.time() - start_time

        # Log results
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Best value: {self.best_value:.4f}")
        logger.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Total trials: {len(self.study.trials)}")
        logger.info(f"Completed trials: {len([t for t in self.study.trials if t.state == TrialState.COMPLETE])}")
        logger.info(f"Pruned trials: {len([t for t in self.study.trials if t.state == TrialState.PRUNED])}")
        logger.info(f"Failed trials: {len([t for t in self.study.trials if t.state == TrialState.FAIL])}")
        logger.info(f"Time elapsed: {elapsed_time:.2f} seconds")
        logger.info("="*80 + "\n")

        # Save results
        self._save_results()

        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'study': self.study,
            'optimization_history': self.optimization_history,
            'elapsed_time': elapsed_time
        }

    def _save_results(self) -> None:
        """Save optimization results to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save best parameters
        params_file = self.output_dir / f'best_params_{timestamp}.json'
        with open(params_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"Saved best parameters to {params_file}")

        # Save optimization history
        history_file = self.output_dir / f'optimization_history_{timestamp}.json'
        with open(history_file, 'w') as f:
            # Convert to serializable format
            history_serializable = []
            for trial in self.optimization_history:
                trial_copy = trial.copy()
                # Remove non-serializable objects
                if 'training_results' in trial_copy:
                    results = trial_copy['training_results']
                    # Convert numpy arrays to lists
                    for key, value in results.items():
                        if isinstance(value, np.ndarray):
                            results[key] = value.tolist()
                history_serializable.append(trial_copy)

            json.dump(history_serializable, f, indent=2)
        logger.info(f"Saved optimization history to {history_file}")

        # Save study
        if self.study:
            study_file = self.output_dir / f'study_{timestamp}.pkl'
            with open(study_file, 'wb') as f:
                pickle.dump(self.study, f)
            logger.info(f"Saved Optuna study to {study_file}")

    def get_parameter_importance(self, n_top: int = 10) -> Dict[str, float]:
        """
        Calculate parameter importance using fANOVA

        Args:
            n_top: Number of top parameters to return

        Returns:
            Dict mapping parameter names to importance scores
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        try:
            importance = optuna.importance.get_param_importances(self.study)
            # Return top N
            sorted_importance = dict(sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_top])
            return sorted_importance
        except Exception as e:
            logger.error(f"Failed to calculate parameter importance: {e}")
            return {}

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot optimization history

        Args:
            save_path: Path to save plot (if None, just displays)
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved optimization history plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot optimization history: {e}")

    def plot_param_importances(self, save_path: Optional[str] = None) -> None:
        """
        Plot parameter importance

        Args:
            save_path: Path to save plot (if None, just displays)
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved parameter importance plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot parameter importances: {e}")

    def plot_parallel_coordinate(self, save_path: Optional[str] = None) -> None:
        """
        Plot parallel coordinate plot of hyperparameters

        Args:
            save_path: Path to save plot (if None, just displays)
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        try:
            import matplotlib.pyplot as plt

            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(self.study)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved parallel coordinate plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Failed to plot parallel coordinate: {e}")

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive optimization report

        Args:
            output_path: Path to save report (if None, returns as string)

        Returns:
            Report as string
        """
        if not self.study:
            raise ValueError("No study available. Run optimize() first.")

        report_lines = []
        report_lines.append("="*80)
        report_lines.append("HYPERPARAMETER OPTIMIZATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Study name: {self.study_name}")
        report_lines.append(f"Objective: {self.objective.name} ({self.objective.direction})")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Best trial
        report_lines.append("BEST TRIAL")
        report_lines.append("-"*80)
        report_lines.append(f"Value: {self.best_value:.6f}")
        report_lines.append("Parameters:")
        for param, value in self.best_params.items():
            report_lines.append(f"  {param}: {value}")
        report_lines.append("")

        # Trial statistics
        report_lines.append("TRIAL STATISTICS")
        report_lines.append("-"*80)
        report_lines.append(f"Total trials: {len(self.study.trials)}")
        report_lines.append(f"Completed: {len([t for t in self.study.trials if t.state == TrialState.COMPLETE])}")
        report_lines.append(f"Pruned: {len([t for t in self.study.trials if t.state == TrialState.PRUNED])}")
        report_lines.append(f"Failed: {len([t for t in self.study.trials if t.state == TrialState.FAIL])}")
        report_lines.append("")

        # Parameter importance
        try:
            importance = self.get_parameter_importance(n_top=10)
            if importance:
                report_lines.append("PARAMETER IMPORTANCE (Top 10)")
                report_lines.append("-"*80)
                for i, (param, score) in enumerate(importance.items(), 1):
                    report_lines.append(f"{i:2d}. {param:30s} {score:.4f}")
                report_lines.append("")
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {e}")

        # Top 5 trials
        report_lines.append("TOP 5 TRIALS")
        report_lines.append("-"*80)
        sorted_trials = sorted(
            [t for t in self.study.trials if t.state == TrialState.COMPLETE],
            key=lambda t: t.value,
            reverse=(self.objective.direction == 'maximize')
        )[:5]

        for i, trial in enumerate(sorted_trials, 1):
            report_lines.append(f"\n{i}. Trial {trial.number}")
            report_lines.append(f"   Value: {trial.value:.6f}")
            report_lines.append(f"   Params: {json.dumps(trial.params, indent=10)}")

        report_lines.append("")
        report_lines.append("="*80)

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved optimization report to {output_path}")

        return report
