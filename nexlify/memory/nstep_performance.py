#!/usr/bin/env python3
"""
N-Step Returns Performance Analysis and Tracking
Provides tools to analyze and visualize n-step return effectiveness
"""

import logging
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class NStepPerformanceTracker:
    """
    Track and analyze n-step return performance

    Monitors:
    - Credit assignment accuracy
    - Return prediction error
    - Optimal n value estimation
    - Comparison between 1-step and n-step learning

    Args:
        n_step: Number of steps to track
        window_size: Rolling window for statistics (default: 1000)

    Example:
        >>> tracker = NStepPerformanceTracker(n_step=5)
        >>> tracker.record_episode(episode_rewards, episode_actions)
        >>> stats = tracker.get_statistics()
    """

    def __init__(self, n_step: int = 5, window_size: int = 1000):
        """Initialize performance tracker"""
        self.n_step = n_step
        self.window_size = window_size

        # Episode tracking
        self.episode_count = 0
        self.total_steps = 0

        # Return accuracy tracking
        self.return_errors = deque(maxlen=window_size)
        self.td_errors = deque(maxlen=window_size)

        # Credit assignment tracking
        self.early_action_rewards = deque(maxlen=window_size)  # Rewards for early actions
        self.late_action_rewards = deque(maxlen=window_size)   # Rewards for late actions

        # N-step comparison
        self.one_step_returns = deque(maxlen=window_size)
        self.n_step_returns = deque(maxlen=window_size)

        # Episode metrics
        self.episode_lengths = deque(maxlen=100)
        self.episode_returns = deque(maxlen=100)
        self.episode_truncations = deque(maxlen=100)  # How often n-step was truncated

        # Statistics
        self.stats = {
            "episodes": 0,
            "total_steps": 0,
            "avg_episode_length": 0.0,
            "avg_episode_return": 0.0,
            "avg_return_error": 0.0,
            "avg_td_error": 0.0,
            "credit_assignment_ratio": 1.0,  # early/late reward ratio
            "truncation_rate": 0.0,
            "n_step_advantage": 0.0,  # How much better n-step is vs 1-step
        }

        logger.info(f"NStepPerformanceTracker initialized (n={n_step}, window={window_size})")

    def record_transition(
        self,
        one_step_return: float,
        n_step_return: float,
        td_error: float,
        is_early_action: bool = False,
    ):
        """
        Record a single transition for analysis

        Args:
            one_step_return: 1-step return estimate
            n_step_return: N-step return estimate
            td_error: Temporal difference error
            is_early_action: Whether this is an early action in episode
        """
        self.total_steps += 1

        # Track returns
        self.one_step_returns.append(one_step_return)
        self.n_step_returns.append(n_step_return)

        # Track errors
        return_error = abs(n_step_return - one_step_return)
        self.return_errors.append(return_error)
        self.td_errors.append(abs(td_error))

        # Track credit assignment
        if is_early_action:
            self.early_action_rewards.append(n_step_return)
        else:
            self.late_action_rewards.append(n_step_return)

    def record_episode(
        self,
        episode_length: int,
        episode_return: float,
        truncation_count: int = 0,
    ):
        """
        Record episode-level metrics

        Args:
            episode_length: Number of steps in episode
            episode_return: Total episode return
            truncation_count: Number of times n-step was truncated
        """
        self.episode_count += 1

        self.episode_lengths.append(episode_length)
        self.episode_returns.append(episode_return)
        self.episode_truncations.append(truncation_count)

        # Update statistics
        self._update_statistics()

    def _update_statistics(self):
        """Update running statistics"""
        self.stats["episodes"] = self.episode_count
        self.stats["total_steps"] = self.total_steps

        # Episode metrics
        if len(self.episode_lengths) > 0:
            self.stats["avg_episode_length"] = np.mean(self.episode_lengths)
            self.stats["avg_episode_return"] = np.mean(self.episode_returns)

        # Return accuracy
        if len(self.return_errors) > 0:
            self.stats["avg_return_error"] = np.mean(self.return_errors)

        if len(self.td_errors) > 0:
            self.stats["avg_td_error"] = np.mean(self.td_errors)

        # Credit assignment
        if len(self.early_action_rewards) > 0 and len(self.late_action_rewards) > 0:
            early_mean = np.mean(self.early_action_rewards)
            late_mean = np.mean(self.late_action_rewards)
            if late_mean != 0:
                self.stats["credit_assignment_ratio"] = early_mean / late_mean

        # Truncation rate
        if len(self.episode_truncations) > 0:
            total_truncations = sum(self.episode_truncations)
            total_transitions = sum(self.episode_lengths)
            if total_transitions > 0:
                self.stats["truncation_rate"] = total_truncations / total_transitions

        # N-step advantage
        if len(self.one_step_returns) > 0 and len(self.n_step_returns) > 0:
            one_step_var = np.var(self.one_step_returns)
            n_step_var = np.var(self.n_step_returns)
            # Lower variance is better (more stable learning)
            if one_step_var > 0:
                self.stats["n_step_advantage"] = (one_step_var - n_step_var) / one_step_var

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics

        Returns:
            Dictionary of performance statistics
        """
        return self.stats.copy()

    def get_optimal_n(self, max_n: int = 10) -> int:
        """
        Estimate optimal n value based on current performance

        This is a heuristic based on:
        - Episode length (longer episodes can use larger n)
        - Return variance (lower is better)
        - Truncation rate (lower is better)

        Args:
            max_n: Maximum n to consider

        Returns:
            Recommended n value
        """
        if len(self.episode_lengths) < 10:
            return self.n_step  # Not enough data

        avg_length = np.mean(self.episode_lengths)
        truncation_rate = self.stats["truncation_rate"]

        # Heuristic: n should be ~10-20% of episode length
        # but adjusted based on truncation rate
        optimal_n = int(avg_length * 0.15)

        # Adjust for high truncation (reduce n)
        if truncation_rate > 0.3:
            optimal_n = int(optimal_n * 0.7)

        # Clamp to reasonable range
        optimal_n = max(1, min(optimal_n, max_n))

        return optimal_n

    def get_report(self) -> str:
        """
        Generate performance report

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("N-Step Returns Performance Report")
        report.append("=" * 60)
        report.append(f"Configuration: n={self.n_step}")
        report.append(f"Episodes: {self.stats['episodes']}")
        report.append(f"Total Steps: {self.stats['total_steps']}")
        report.append("")

        report.append("Episode Metrics:")
        report.append(f"  Avg Length: {self.stats['avg_episode_length']:.1f}")
        report.append(f"  Avg Return: {self.stats['avg_episode_return']:.2f}")
        report.append("")

        report.append("Return Accuracy:")
        report.append(f"  Avg Return Error: {self.stats['avg_return_error']:.4f}")
        report.append(f"  Avg TD Error: {self.stats['avg_td_error']:.4f}")
        report.append("")

        report.append("Credit Assignment:")
        report.append(f"  Early/Late Reward Ratio: {self.stats['credit_assignment_ratio']:.2f}")
        if self.stats['credit_assignment_ratio'] > 1.0:
            report.append("  → Early actions receive MORE credit (good for n-step)")
        else:
            report.append("  → Late actions receive MORE credit")
        report.append("")

        report.append("N-Step Analysis:")
        report.append(f"  Truncation Rate: {self.stats['truncation_rate']:.1%}")
        report.append(f"  N-Step Advantage: {self.stats['n_step_advantage']:.1%}")

        if self.stats['n_step_advantage'] > 0:
            report.append("  → N-step returns show LOWER variance (more stable)")
        else:
            report.append("  → N-step returns show HIGHER variance")
        report.append("")

        # Recommendations
        optimal_n = self.get_optimal_n()
        report.append("Recommendations:")
        if optimal_n != self.n_step:
            report.append(f"  Consider using n={optimal_n} (currently n={self.n_step})")
        else:
            report.append(f"  Current n={self.n_step} appears optimal")

        if self.stats['truncation_rate'] > 0.5:
            report.append("  ⚠️  High truncation rate - consider reducing n")

        if self.stats['n_step_advantage'] < -0.2:
            report.append("  ⚠️  N-step showing higher variance - may want to use 1-step or mixed")

        report.append("=" * 60)

        return "\n".join(report)

    def __repr__(self) -> str:
        return (
            f"NStepPerformanceTracker("
            f"n={self.n_step}, "
            f"episodes={self.episode_count}, "
            f"steps={self.total_steps}, "
            f"advantage={self.stats['n_step_advantage']:.1%})"
        )


def compare_nstep_configurations(
    results: Dict[int, Dict[str, float]]
) -> str:
    """
    Compare performance across different n values

    Args:
        results: Dictionary mapping n -> performance metrics

    Returns:
        Formatted comparison report

    Example:
        >>> results = {
        ...     1: {"avg_return": 100, "variance": 50},
        ...     3: {"avg_return": 120, "variance": 30},
        ...     5: {"avg_return": 125, "variance": 35},
        ... }
        >>> print(compare_nstep_configurations(results))
    """
    report = []
    report.append("=" * 70)
    report.append("N-Step Configuration Comparison")
    report.append("=" * 70)
    report.append(f"{'N':>5} | {'Avg Return':>12} | {'Variance':>12} | {'TD Error':>12}")
    report.append("-" * 70)

    for n in sorted(results.keys()):
        metrics = results[n]
        avg_return = metrics.get("avg_return", 0)
        variance = metrics.get("variance", 0)
        td_error = metrics.get("avg_td_error", 0)

        report.append(f"{n:5d} | {avg_return:12.2f} | {variance:12.4f} | {td_error:12.4f}")

    # Find best configuration
    best_n = max(results.keys(), key=lambda n: results[n].get("avg_return", 0))
    report.append("-" * 70)
    report.append(f"Best Configuration: n={best_n}")
    report.append("=" * 70)

    return "\n".join(report)


# Export classes and functions
__all__ = [
    "NStepPerformanceTracker",
    "compare_nstep_configurations",
]
