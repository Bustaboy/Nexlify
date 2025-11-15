"""
Walk-Forward Validation for Robust Performance Estimation

Implements time-series aware cross-validation with no future data leakage.
Supports rolling and expanding window modes for realistic backtesting.
"""

import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from nexlify.utils.error_handler import get_error_handler

logger = logging.getLogger(__name__)

# Configuration Constants
TRADING_DAYS_PER_YEAR = 252  # Standard assumption for annualization
DEFAULT_RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
DEFAULT_MODEL_DIR = 'models/walk_forward'
DEFAULT_REPORT_DIR = 'reports/walk_forward'


@dataclass
class FoldConfig:
    """Configuration for a single fold in walk-forward validation"""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        """Number of episodes in training window"""
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        """Number of episodes in test window"""
        return self.test_end - self.test_start

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"Train [{self.train_start}-{self.train_end}] "
            f"→ Test [{self.test_start}-{self.test_end}]"
        )


@dataclass
class FoldMetrics:
    """Performance metrics for a single fold"""
    fold_id: int
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    profit_factor: float
    num_trades: int
    avg_trade_duration: float
    volatility: float
    sortino_ratio: float
    calmar_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_id}: "
            f"Return={self.total_return:.2%}, "
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"WinRate={self.win_rate:.2%}"
        )


@dataclass
class WalkForwardResults:
    """Aggregated results from walk-forward validation"""
    fold_configs: List[FoldConfig]
    fold_metrics: List[FoldMetrics]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_fold_id: int
    worst_fold_id: int
    validation_date: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'fold_configs': [asdict(fc) for fc in self.fold_configs],
            'fold_metrics': [fm.to_dict() for fm in self.fold_metrics],
            'mean_metrics': self.mean_metrics,
            'std_metrics': self.std_metrics,
            'best_fold_id': self.best_fold_id,
            'worst_fold_id': self.worst_fold_id,
            'validation_date': self.validation_date
        }

    def summary(self) -> str:
        """Generate summary report"""
        lines = [
            "\n" + "="*70,
            "WALK-FORWARD VALIDATION SUMMARY",
            "="*70,
            f"Validation Date: {self.validation_date}",
            f"Number of Folds: {len(self.fold_configs)}",
            "",
            "Mean Performance Metrics:",
            "-" * 70
        ]

        for metric, value in self.mean_metrics.items():
            std = self.std_metrics.get(metric, 0)
            if 'rate' in metric or 'return' in metric:
                lines.append(f"  {metric:20s}: {value:>8.2%} ± {std:>6.2%}")
            else:
                lines.append(f"  {metric:20s}: {value:>8.2f} ± {std:>6.2f}")

        lines.extend([
            "",
            f"Best Fold:  {self.best_fold_id} (Return: {self.fold_metrics[self.best_fold_id].total_return:.2%})",
            f"Worst Fold: {self.worst_fold_id} (Return: {self.fold_metrics[self.worst_fold_id].total_return:.2%})",
            "="*70
        ])

        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward validation for time-series trading strategies

    Prevents future data leakage by training on past data and testing on
    sequential future periods. Provides realistic performance estimates.

    Modes:
        - rolling: Fixed training window size, slides forward
        - expanding: Growing training window (anchored at start)

    Example:
        >>> validator = WalkForwardValidator(
        ...     total_episodes=2000,
        ...     train_size=1000,
        ...     test_size=200,
        ...     step_size=200,
        ...     mode='rolling'
        ... )
        >>> results = await validator.validate(train_fn, eval_fn)
        >>> print(results.summary())
    """

    def __init__(
        self,
        total_episodes: int,
        train_size: int = 1000,
        test_size: int = 200,
        step_size: int = 200,
        mode: str = 'rolling',
        min_train_size: int = 500,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize walk-forward validator

        Args:
            total_episodes: Total number of episodes available
            train_size: Episodes for training window
            test_size: Episodes for testing window
            step_size: How far to step forward between folds
            mode: 'rolling' (fixed window) or 'expanding' (growing window)
            min_train_size: Minimum training episodes (for expanding mode)
            config: Optional configuration dictionary
        """
        self.total_episodes = total_episodes
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.mode = mode.lower()
        self.min_train_size = min_train_size
        self.config = config or {}

        self.error_handler = get_error_handler()

        # Validate configuration
        self._validate_config()

        # Generate folds
        self.folds = self._generate_folds()

        logger.info(
            f"WalkForwardValidator initialized: {len(self.folds)} folds, "
            f"mode={self.mode}, train_size={train_size}, test_size={test_size}"
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if self.mode not in ['rolling', 'expanding']:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be 'rolling' or 'expanding'"
            )

        if self.train_size <= 0 or self.test_size <= 0:
            raise ValueError("Train and test sizes must be positive")

        if self.step_size <= 0:
            raise ValueError("Step size must be positive")

        if self.train_size + self.test_size > self.total_episodes:
            raise ValueError(
                f"Train size ({self.train_size}) + test size ({self.test_size}) "
                f"exceeds total episodes ({self.total_episodes})"
            )

        if self.min_train_size > self.train_size:
            raise ValueError(
                f"Minimum train size ({self.min_train_size}) cannot exceed "
                f"train size ({self.train_size})"
            )

    def _generate_folds(self) -> List[FoldConfig]:
        """
        Generate fold configurations based on mode

        Returns:
            List of FoldConfig objects
        """
        folds = []
        fold_id = 0

        if self.mode == 'rolling':
            # Rolling window: fixed training size
            current_pos = 0
            while current_pos + self.train_size + self.test_size <= self.total_episodes:
                train_start = current_pos
                train_end = current_pos + self.train_size
                test_start = train_end
                test_end = test_start + self.test_size

                folds.append(FoldConfig(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end
                ))

                current_pos += self.step_size
                fold_id += 1

        elif self.mode == 'expanding':
            # Expanding window: growing training size
            train_start = 0
            test_start = self.train_size

            while test_start + self.test_size <= self.total_episodes:
                test_end = test_start + self.test_size

                folds.append(FoldConfig(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=test_start,
                    test_start=test_start,
                    test_end=test_end
                ))

                test_start += self.step_size
                fold_id += 1

        if not folds:
            raise ValueError(
                "No valid folds could be generated with current parameters. "
                "Try reducing train_size, test_size, or step_size."
            )

        logger.info(f"Generated {len(folds)} folds for walk-forward validation")
        for fold in folds:
            logger.debug(str(fold))

        return folds

    async def validate(
        self,
        train_fn: Callable[[int, int], Any],
        eval_fn: Callable[[Any, int, int], Dict[str, float]],
        save_models: bool = True,
        model_dir: Optional[Path] = None
    ) -> WalkForwardResults:
        """
        Run walk-forward validation

        Args:
            train_fn: Function(train_start, train_end) -> model
                     Trains model on specified episode range
            eval_fn: Function(model, test_start, test_end) -> metrics_dict
                    Evaluates model and returns performance metrics
            save_models: Whether to save models from each fold
            model_dir: Directory to save models (default: models/walk_forward/)

        Returns:
            WalkForwardResults object with aggregated metrics
        """
        logger.info(
            f"Starting walk-forward validation with {len(self.folds)} folds"
        )

        if model_dir is None:
            model_dir = Path(DEFAULT_MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)

        fold_metrics_list = []

        for fold in self.folds:
            logger.info(f"Processing {fold}")

            try:
                # Train on training window
                logger.info(
                    f"  Training on episodes {fold.train_start}-{fold.train_end}"
                )
                model = await self._call_async(
                    train_fn, fold.train_start, fold.train_end
                )

                # Evaluate on test window
                logger.info(
                    f"  Testing on episodes {fold.test_start}-{fold.test_end}"
                )
                metrics = await self._call_async(
                    eval_fn, model, fold.test_start, fold.test_end
                )

                # Create FoldMetrics object
                fold_metrics = self._create_fold_metrics(fold.fold_id, metrics)
                fold_metrics_list.append(fold_metrics)

                logger.info(f"  {fold_metrics}")

                # Save model if requested
                if save_models:
                    model_path = model_dir / f"fold_{fold.fold_id}_model.pt"
                    self._save_model(model, model_path)

            except Exception as e:
                self.error_handler.log_error(
                    e,
                    context={
                        'operation': 'walk_forward_validation',
                        'fold_id': fold.fold_id
                    }
                )
                logger.error(f"Error processing {fold}: {e}")
                # Continue with next fold
                continue

        if not fold_metrics_list:
            raise RuntimeError("No folds completed successfully")

        # Aggregate results
        results = self._aggregate_results(self.folds, fold_metrics_list)

        logger.info("\n" + results.summary())

        return results

    async def _call_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Call function, handling both sync and async functions"""
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    def _create_fold_metrics(
        self,
        fold_id: int,
        metrics: Dict[str, float]
    ) -> FoldMetrics:
        """Create FoldMetrics object from metrics dictionary"""
        return FoldMetrics(
            fold_id=fold_id,
            total_return=metrics.get('total_return', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            profit_factor=metrics.get('profit_factor', 0.0),
            num_trades=int(metrics.get('num_trades', 0)),
            avg_trade_duration=metrics.get('avg_trade_duration', 0.0),
            volatility=metrics.get('volatility', 0.0),
            sortino_ratio=metrics.get('sortino_ratio', 0.0),
            calmar_ratio=metrics.get('calmar_ratio', 0.0)
        )

    def _aggregate_results(
        self,
        folds: List[FoldConfig],
        fold_metrics: List[FoldMetrics]
    ) -> WalkForwardResults:
        """Aggregate metrics across all folds"""
        # Convert to DataFrame for easy aggregation
        metrics_df = pd.DataFrame([fm.to_dict() for fm in fold_metrics])

        # Calculate mean and std for each metric
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'fold_id']

        mean_metrics = metrics_df[numeric_cols].mean().to_dict()
        std_metrics = metrics_df[numeric_cols].std().to_dict()

        # Find best and worst folds by total return
        best_idx = metrics_df['total_return'].idxmax()
        worst_idx = metrics_df['total_return'].idxmin()

        return WalkForwardResults(
            fold_configs=folds,
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_fold_id=int(best_idx),
            worst_fold_id=int(worst_idx),
            validation_date=datetime.now().isoformat()
        )

    def _save_model(self, model: Any, path: Path) -> None:
        """Save model to file"""
        if TORCH_AVAILABLE and isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
            logger.debug(f"Saved model to {path}")
        else:
            logger.warning(f"Model type {type(model)} not supported for saving")

    def generate_report(
        self,
        results: WalkForwardResults,
        output_dir: Optional[Path] = None
    ) -> None:
        """
        Generate comprehensive visual report

        Args:
            results: WalkForwardResults object
            output_dir: Directory to save report (default: reports/walk_forward/)
        """
        if output_dir is None:
            output_dir = Path(DEFAULT_REPORT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create visualizations
        self._plot_fold_comparison(results, output_dir / f'fold_comparison_{timestamp}.png')
        self._plot_performance_stability(results, output_dir / f'performance_stability_{timestamp}.png')
        self._plot_metric_distributions(results, output_dir / f'metric_distributions_{timestamp}.png')

        # Save JSON report
        json_path = output_dir / f'validation_results_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")

        # Save text summary
        summary_path = output_dir / f'summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write(results.summary())
        logger.info(f"Saved summary to {summary_path}")

    def _plot_fold_comparison(
        self,
        results: WalkForwardResults,
        output_path: Path
    ) -> None:
        """Plot comparison of key metrics across folds"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Walk-Forward Validation: Fold Comparison', fontsize=16)

        metrics_df = pd.DataFrame([fm.to_dict() for fm in results.fold_metrics])

        # Total Return
        ax = axes[0, 0]
        ax.bar(metrics_df['fold_id'], metrics_df['total_return'] * 100)
        ax.axhline(
            results.mean_metrics['total_return'] * 100,
            color='r', linestyle='--', label='Mean'
        )
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Total Return (%)')
        ax.set_title('Total Return per Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sharpe Ratio
        ax = axes[0, 1]
        ax.bar(metrics_df['fold_id'], metrics_df['sharpe_ratio'])
        ax.axhline(
            results.mean_metrics['sharpe_ratio'],
            color='r', linestyle='--', label='Mean'
        )
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio per Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Win Rate
        ax = axes[1, 0]
        ax.bar(metrics_df['fold_id'], metrics_df['win_rate'] * 100)
        ax.axhline(
            results.mean_metrics['win_rate'] * 100,
            color='r', linestyle='--', label='Mean'
        )
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate per Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Max Drawdown
        ax = axes[1, 1]
        ax.bar(metrics_df['fold_id'], metrics_df['max_drawdown'] * 100)
        ax.axhline(
            results.mean_metrics['max_drawdown'] * 100,
            color='r', linestyle='--', label='Mean'
        )
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Max Drawdown per Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved fold comparison plot to {output_path}")

    def _plot_performance_stability(
        self,
        results: WalkForwardResults,
        output_path: Path
    ) -> None:
        """Plot performance stability analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Performance Stability Analysis', fontsize=16)

        metrics_df = pd.DataFrame([fm.to_dict() for fm in results.fold_metrics])

        # Cumulative returns over time
        ax = axes[0]
        cumulative_returns = (1 + metrics_df['total_return']).cumprod()
        ax.plot(metrics_df['fold_id'], cumulative_returns, marker='o', linewidth=2)
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Cumulative Return Factor')
        ax.set_title('Cumulative Returns Across Folds')
        ax.grid(True, alpha=0.3)

        # Return stability (rolling std)
        ax = axes[1]
        rolling_std = metrics_df['total_return'].rolling(window=3, min_periods=1).std()
        ax.plot(metrics_df['fold_id'], rolling_std * 100, marker='o', linewidth=2, color='orange')
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Rolling Std of Returns (%)')
        ax.set_title('Return Stability (3-Fold Rolling Std)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved performance stability plot to {output_path}")

    def _plot_metric_distributions(
        self,
        results: WalkForwardResults,
        output_path: Path
    ) -> None:
        """Plot distribution of key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Metric Distributions Across Folds', fontsize=16)

        metrics_df = pd.DataFrame([fm.to_dict() for fm in results.fold_metrics])

        metrics_to_plot = [
            ('total_return', 'Total Return', True),
            ('sharpe_ratio', 'Sharpe Ratio', False),
            ('win_rate', 'Win Rate', True),
            ('max_drawdown', 'Max Drawdown', True),
            ('profit_factor', 'Profit Factor', False),
            ('sortino_ratio', 'Sortino Ratio', False)
        ]

        for idx, (metric, title, as_percent) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]

            data = metrics_df[metric]
            if as_percent:
                data = data * 100

            ax.hist(data, bins=10, edgecolor='black', alpha=0.7)

            mean_val = data.mean()
            ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

            suffix = '%' if as_percent else ''
            ax.set_xlabel(f'{title} {suffix}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{title} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved metric distributions plot to {output_path}")


def calculate_performance_metrics(
    returns: np.ndarray,
    trades: Optional[List[Dict[str, Any]]] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics

    Args:
        returns: Array of returns (not cumulative)
        trades: Optional list of trade dictionaries
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'profit_factor': 0.0,
            'num_trades': 0,
            'avg_trade_duration': 0.0,
            'volatility': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0
        }

    # Total return
    cumulative_returns = np.cumprod(1 + returns)
    total_return = cumulative_returns[-1] - 1

    # Volatility (annualized, assuming daily returns)
    volatility = np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Sharpe ratio
    mean_return = np.mean(returns)
    excess_return = mean_return - (risk_free_rate / TRADING_DAYS_PER_YEAR)  # Daily risk-free rate
    sharpe_ratio = (excess_return / np.std(returns)) * np.sqrt(TRADING_DAYS_PER_YEAR) if np.std(returns) > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    sortino_ratio = (excess_return / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR) if downside_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    # Calmar ratio
    calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # Trade-based metrics
    if trades:
        winning_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        total_wins = sum(t.get('profit', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('profit', 0) for t in losing_trades))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        durations = [t.get('duration', 0) for t in trades if 'duration' in t]
        avg_trade_duration = np.mean(durations) if durations else 0.0

        num_trades = len(trades)
    else:
        # Estimate from returns
        win_rate = len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0.0

        total_wins = np.sum(returns[returns > 0])
        total_losses = abs(np.sum(returns[returns < 0]))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0.0

        num_trades = len(returns)
        avg_trade_duration = 1.0  # Assume 1 period per trade

    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'win_rate': float(win_rate),
        'max_drawdown': float(max_drawdown),
        'profit_factor': float(profit_factor),
        'num_trades': int(num_trades),
        'avg_trade_duration': float(avg_trade_duration),
        'volatility': float(volatility),
        'sortino_ratio': float(sortino_ratio),
        'calmar_ratio': float(calmar_ratio)
    }
