#!/usr/bin/env python3
"""
Nexlify Validation Monitor
==========================
Comprehensive validation monitoring for RL training with proper data splitting,
metric tracking, and performance analysis.

Features:
- Temporal train/val/test splitting (no data leakage)
- Validation metric tracking over time
- Separate validation environment
- Configurable validation frequency
- Result caching and reporting
- Automatic best model selection
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Results from a single validation run"""

    episode: int
    val_return: float
    val_return_pct: float
    val_sharpe: float
    val_win_rate: float
    val_max_drawdown: float
    val_num_trades: int
    val_final_equity: float
    timestamp: str

    # Additional metrics
    val_avg_trade_return: Optional[float] = None
    val_profit_factor: Optional[float] = None
    val_volatility: Optional[float] = None


@dataclass
class DataSplit:
    """Train/validation/test data split with metadata"""

    train_data: np.ndarray
    val_data: np.ndarray
    test_data: np.ndarray

    train_indices: Tuple[int, int]
    val_indices: Tuple[int, int]
    test_indices: Tuple[int, int]

    split_ratios: Tuple[float, float, float]
    total_size: int
    timestamp: str


class ValidationDataSplitter:
    """
    Temporal data splitter that respects chronological order

    Ensures no future data leakage by splitting sequentially:
    [Train | Validation | Test]
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        min_samples_per_split: int = 100
    ):
        """
        Initialize data splitter

        Args:
            train_ratio: Fraction of data for training (default: 0.70)
            val_ratio: Fraction of data for validation (default: 0.15)
            test_ratio: Fraction of data for testing (default: 0.15)
            min_samples_per_split: Minimum samples per split (default: 100)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise ValueError("All split ratios must be positive")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.min_samples_per_split = min_samples_per_split

        logger.info(f"📊 ValidationDataSplitter initialized:")
        logger.info(f"   Train: {train_ratio * 100:.1f}%, Val: {val_ratio * 100:.1f}%, Test: {test_ratio * 100:.1f}%")
        logger.info(f"   Min samples per split: {min_samples_per_split}")

    def split(self, data: np.ndarray) -> DataSplit:
        """
        Split data temporally into train/val/test sets

        Args:
            data: Full dataset (price data or similar)

        Returns:
            DataSplit object with train/val/test arrays and metadata

        Raises:
            ValueError: If data is too small for split
        """
        total_size = len(data)

        # Check minimum size
        min_required = self.min_samples_per_split * 3
        if total_size < min_required:
            raise ValueError(
                f"Dataset too small for splitting. "
                f"Need at least {min_required} samples, got {total_size}"
            )

        # Calculate split points (sequential)
        train_end = int(total_size * self.train_ratio)
        val_end = int(total_size * (self.train_ratio + self.val_ratio))

        # Ensure each split has minimum samples
        if train_end < self.min_samples_per_split:
            raise ValueError(f"Train split too small: {train_end} < {self.min_samples_per_split}")
        if (val_end - train_end) < self.min_samples_per_split:
            raise ValueError(f"Val split too small: {val_end - train_end} < {self.min_samples_per_split}")
        if (total_size - val_end) < self.min_samples_per_split:
            raise ValueError(f"Test split too small: {total_size - val_end} < {self.min_samples_per_split}")

        # Split data (temporal ordering preserved)
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        logger.info(f"✅ Data split completed:")
        logger.info(f"   Train: {len(train_data)} samples (indices 0-{train_end-1})")
        logger.info(f"   Val: {len(val_data)} samples (indices {train_end}-{val_end-1})")
        logger.info(f"   Test: {len(test_data)} samples (indices {val_end}-{total_size-1})")

        return DataSplit(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_indices=(0, train_end),
            val_indices=(train_end, val_end),
            test_indices=(val_end, total_size),
            split_ratios=(self.train_ratio, self.val_ratio, self.test_ratio),
            total_size=total_size,
            timestamp=datetime.now().isoformat()
        )


class ValidationMonitor:
    """
    Monitors validation performance during training

    Tracks validation metrics, manages validation environment,
    and provides insights into model generalization.
    """

    def __init__(
        self,
        validation_frequency: int = 50,
        metrics_to_track: Optional[List[str]] = None,
        save_dir: Optional[Path] = None,
        cache_results: bool = True
    ):
        """
        Initialize validation monitor

        Args:
            validation_frequency: Run validation every N episodes (default: 50)
            metrics_to_track: List of metrics to track (default: all)
            save_dir: Directory to save validation results (default: None)
            cache_results: Whether to cache validation results (default: True)
        """
        self.validation_frequency = validation_frequency
        self.save_dir = Path(save_dir) if save_dir else None
        self.cache_results = cache_results

        # Default metrics
        if metrics_to_track is None:
            metrics_to_track = [
                'val_return',
                'val_return_pct',
                'val_sharpe',
                'val_win_rate',
                'val_max_drawdown',
                'val_num_trades'
            ]
        self.metrics_to_track = metrics_to_track

        # Validation history
        self.validation_results: List[ValidationResult] = []
        self.best_val_result: Optional[ValidationResult] = None
        self.best_val_metric_value: float = -np.inf

        # Tracking
        self.validation_count = 0
        self.last_validation_episode = 0

        # Create save directory if specified
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Validation results will be saved to: {self.save_dir}")

        logger.info(f"🔍 ValidationMonitor initialized:")
        logger.info(f"   Validation frequency: every {validation_frequency} episodes")
        logger.info(f"   Metrics tracked: {', '.join(metrics_to_track)}")
        logger.info(f"   Caching: {'enabled' if cache_results else 'disabled'}")

    def should_validate(self, current_episode: int) -> bool:
        """
        Check if validation should run at current episode

        Args:
            current_episode: Current training episode number

        Returns:
            True if validation should run, False otherwise
        """
        # Always validate on first episode
        if current_episode == 0:
            return True

        # Check frequency
        episodes_since_last = current_episode - self.last_validation_episode
        return episodes_since_last >= self.validation_frequency

    def run_validation(
        self,
        agent,
        val_env,
        current_episode: int,
        num_episodes: int = 5
    ) -> ValidationResult:
        """
        Run validation on validation environment

        Args:
            agent: RL agent to validate
            val_env: Validation environment
            current_episode: Current training episode
            num_episodes: Number of validation episodes to average (default: 5)

        Returns:
            ValidationResult with aggregated metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 VALIDATION RUN (Episode {current_episode})")
        logger.info(f"{'='*80}")

        # Run multiple validation episodes
        episode_returns = []
        episode_return_pcts = []
        episode_sharpes = []
        episode_win_rates = []
        episode_drawdowns = []
        episode_trades = []
        episode_equities = []
        episode_avg_trade_returns = []
        episode_profit_factors = []
        episode_volatilities = []

        for val_ep in range(num_episodes):
            state = val_env.reset()
            done = False
            episode_reward = 0
            trades_this_ep = []

            # Run episode (no training, no exploration)
            while not done:
                # Use greedy policy (no exploration)
                action = agent.act(state, training=False)
                next_state, reward, done, info = val_env.step(action)

                episode_reward += reward
                state = next_state

                # Track trades
                if info.get('trade_executed', False):
                    trades_this_ep.append(info)

            # Calculate episode metrics
            final_equity = val_env._get_current_equity()
            total_return = final_equity - val_env.initial_balance
            return_pct = (total_return / val_env.initial_balance) * 100

            # Calculate Sharpe ratio from equity curve
            if len(val_env.equity_curve) > 1:
                equity_returns = np.diff(val_env.equity_curve) / val_env.equity_curve[:-1]
                sharpe = (
                    np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)
                    if np.std(equity_returns) > 0 else 0
                )
            else:
                sharpe = 0

            # Calculate win rate
            if val_env.total_trades > 0:
                win_rate = (val_env.winning_trades / val_env.total_trades) * 100
            else:
                win_rate = 0

            # Calculate max drawdown
            equity_curve = np.array(val_env.equity_curve)
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0

            # Calculate additional metrics
            if len(trades_this_ep) > 0:
                trade_returns = [t.get('profit', 0) for t in trades_this_ep if 'profit' in t]
                avg_trade_return = np.mean(trade_returns) if trade_returns else 0

                # Profit factor: sum(wins) / sum(losses)
                wins = [r for r in trade_returns if r > 0]
                losses = [abs(r) for r in trade_returns if r < 0]
                profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else 0
            else:
                avg_trade_return = 0
                profit_factor = 0

            # Volatility
            volatility = np.std(equity_returns) if len(val_env.equity_curve) > 1 else 0

            # Record
            episode_returns.append(total_return)
            episode_return_pcts.append(return_pct)
            episode_sharpes.append(sharpe)
            episode_win_rates.append(win_rate)
            episode_drawdowns.append(max_drawdown)
            episode_trades.append(val_env.total_trades)
            episode_equities.append(final_equity)
            episode_avg_trade_returns.append(avg_trade_return)
            episode_profit_factors.append(profit_factor)
            episode_volatilities.append(volatility)

            logger.debug(
                f"   Val Episode {val_ep + 1}/{num_episodes}: "
                f"Return={return_pct:+.2f}%, Sharpe={sharpe:.2f}, "
                f"WinRate={win_rate:.1f}%, Trades={val_env.total_trades}"
            )

        # Aggregate results (average over validation episodes)
        val_result = ValidationResult(
            episode=current_episode,
            val_return=float(np.mean(episode_returns)),
            val_return_pct=float(np.mean(episode_return_pcts)),
            val_sharpe=float(np.mean(episode_sharpes)),
            val_win_rate=float(np.mean(episode_win_rates)),
            val_max_drawdown=float(np.mean(episode_drawdowns)),
            val_num_trades=int(np.mean(episode_trades)),
            val_final_equity=float(np.mean(episode_equities)),
            timestamp=datetime.now().isoformat(),
            val_avg_trade_return=float(np.mean(episode_avg_trade_returns)),
            val_profit_factor=float(np.mean(episode_profit_factors)),
            val_volatility=float(np.mean(episode_volatilities))
        )

        # Log results
        logger.info(f"\n{'─'*80}")
        logger.info(f"📊 VALIDATION RESULTS (averaged over {num_episodes} episodes)")
        logger.info(f"{'─'*80}")
        logger.info(f"   Return: ${val_result.val_return:+,.2f} ({val_result.val_return_pct:+.2f}%)")
        logger.info(f"   Sharpe Ratio: {val_result.val_sharpe:.3f}")
        logger.info(f"   Win Rate: {val_result.val_win_rate:.1f}%")
        logger.info(f"   Max Drawdown: {val_result.val_max_drawdown:.2f}%")
        logger.info(f"   Trades: {val_result.val_num_trades}")
        logger.info(f"   Avg Trade Return: ${val_result.val_avg_trade_return:+,.2f}")
        logger.info(f"   Profit Factor: {val_result.val_profit_factor:.2f}")
        logger.info(f"   Volatility: {val_result.val_volatility:.4f}")
        logger.info(f"{'='*80}\n")

        # Update tracking
        self.validation_results.append(val_result)
        self.validation_count += 1
        self.last_validation_episode = current_episode

        # Save if caching enabled
        if self.cache_results and self.save_dir:
            self._save_result(val_result)

        return val_result

    def update_best(self, val_result: ValidationResult, metric: str = 'val_sharpe') -> bool:
        """
        Update best validation result based on metric

        Args:
            val_result: Latest validation result
            metric: Metric to use for comparison (default: 'val_sharpe')

        Returns:
            True if this is a new best, False otherwise
        """
        metric_value = getattr(val_result, metric, None)
        if metric_value is None:
            logger.warning(f"Metric '{metric}' not found in validation result")
            return False

        # For max_drawdown, lower is better
        if metric == 'val_max_drawdown':
            is_better = metric_value < self.best_val_metric_value if self.best_val_metric_value != -np.inf else True
        else:
            is_better = metric_value > self.best_val_metric_value

        if is_better:
            self.best_val_result = val_result
            self.best_val_metric_value = metric_value
            logger.info(f"🏆 NEW BEST VALIDATION: {metric}={metric_value:.3f}")
            return True

        return False

    def get_validation_history(self) -> pd.DataFrame:
        """
        Get validation history as DataFrame

        Returns:
            DataFrame with all validation results
        """
        if not self.validation_results:
            return pd.DataFrame()

        data = [asdict(result) for result in self.validation_results]
        df = pd.DataFrame(data)
        return df

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of validation metrics

        Returns:
            Dictionary with summary statistics
        """
        if not self.validation_results:
            return {}

        df = self.get_validation_history()

        summary = {
            'num_validations': len(self.validation_results),
            'best_val_sharpe': float(df['val_sharpe'].max()),
            'best_val_return_pct': float(df['val_return_pct'].max()),
            'avg_val_sharpe': float(df['val_sharpe'].mean()),
            'avg_val_return_pct': float(df['val_return_pct'].mean()),
            'avg_win_rate': float(df['val_win_rate'].mean()),
            'best_episode': int(df.loc[df['val_sharpe'].idxmax(), 'episode']),
        }

        return summary

    def _save_result(self, result: ValidationResult):
        """Save validation result to disk"""
        if not self.save_dir:
            return

        # Save individual result
        filename = f"validation_ep{result.episode}.json"
        filepath = self.save_dir / filename

        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        # Update cumulative results file
        cumulative_file = self.save_dir / "validation_history.json"
        all_results = [asdict(r) for r in self.validation_results]

        with open(cumulative_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.debug(f"💾 Validation result saved to {filepath}")

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive validation report

        Args:
            output_path: Path to save report (default: save_dir/validation_report.txt)

        Returns:
            Report as string
        """
        if not self.validation_results:
            return "No validation results available"

        df = self.get_validation_history()
        summary = self.get_metrics_summary()

        report_lines = [
            "=" * 80,
            "VALIDATION MONITORING REPORT",
            "=" * 80,
            "",
            f"Total Validations: {summary['num_validations']}",
            f"Validation Frequency: Every {self.validation_frequency} episodes",
            "",
            "BEST PERFORMANCE",
            "-" * 80,
            f"  Best Sharpe Ratio: {summary['best_val_sharpe']:.3f} (Episode {summary['best_episode']})",
            f"  Best Return: {summary['best_val_return_pct']:+.2f}%",
            "",
            "AVERAGE PERFORMANCE",
            "-" * 80,
            f"  Avg Sharpe Ratio: {summary['avg_val_sharpe']:.3f}",
            f"  Avg Return: {summary['avg_val_return_pct']:+.2f}%",
            f"  Avg Win Rate: {summary['avg_win_rate']:.1f}%",
            "",
            "VALIDATION HISTORY",
            "-" * 80,
        ]

        # Add each validation result
        for result in self.validation_results:
            report_lines.append(
                f"  Episode {result.episode:4d}: "
                f"Return={result.val_return_pct:+7.2f}%, "
                f"Sharpe={result.val_sharpe:6.3f}, "
                f"WinRate={result.val_win_rate:5.1f}%, "
                f"Trades={result.val_num_trades:3d}"
            )

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        # Save if output path specified
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Validation report saved to {output_path}")
        elif self.save_dir:
            output_path = self.save_dir / "validation_report.txt"
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"📄 Validation report saved to {output_path}")

        return report


__all__ = [
    'ValidationMonitor',
    'ValidationDataSplitter',
    'ValidationResult',
    'DataSplit',
]
