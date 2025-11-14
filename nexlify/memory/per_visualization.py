#!/usr/bin/env python3
"""
Prioritized Experience Replay Visualization Tools
Tools for monitoring and visualizing PER statistics during training
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PERStatsTracker:
    """
    Track PER statistics over time for analysis and visualization

    Tracks:
        - Beta annealing progress
        - Mean/max/min priorities
        - Priority distribution
        - Sampling statistics
        - Buffer utilization

    Example:
        >>> tracker = PERStatsTracker()
        >>> tracker.record(agent.get_per_stats(), episode=10)
        >>> tracker.save('per_history.json')
        >>> tracker.plot('per_stats.png')
    """

    def __init__(self):
        """Initialize stats tracker"""
        self.history = {
            "episodes": [],
            "steps": [],
            "beta": [],
            "mean_priority": [],
            "max_priority": [],
            "min_priority": [],
            "total_priority": [],
            "buffer_size": [],
            "buffer_capacity": [],
            "total_samples": [],
            "priority_updates": [],
        }

    def record(
        self,
        per_stats: Dict,
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ):
        """
        Record PER statistics at current time

        Args:
            per_stats: Dictionary from agent.get_per_stats()
            episode: Current episode number
            step: Current training step
        """
        if per_stats is None:
            logger.warning("No PER stats available to record")
            return

        if episode is not None:
            self.history["episodes"].append(episode)
        if step is not None:
            self.history["steps"].append(step)

        # Record stats
        self.history["beta"].append(per_stats.get("beta", 0))
        self.history["mean_priority"].append(per_stats.get("mean_priority", 0))
        self.history["max_priority"].append(per_stats.get("max_priority", 0))
        self.history["min_priority"].append(per_stats.get("min_priority", 0))
        self.history["total_priority"].append(per_stats.get("total_priority", 0))
        self.history["buffer_size"].append(per_stats.get("size", 0))
        self.history["buffer_capacity"].append(per_stats.get("capacity", 0))
        self.history["total_samples"].append(per_stats.get("total_samples", 0))
        self.history["priority_updates"].append(per_stats.get("priority_updates", 0))

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics

        Returns:
            Dictionary of summary statistics
        """
        if len(self.history["beta"]) == 0:
            return {}

        return {
            "final_beta": self.history["beta"][-1] if self.history["beta"] else 0,
            "mean_priority_avg": np.mean(self.history["mean_priority"])
            if self.history["mean_priority"]
            else 0,
            "mean_priority_std": np.std(self.history["mean_priority"])
            if self.history["mean_priority"]
            else 0,
            "max_priority_peak": max(self.history["max_priority"])
            if self.history["max_priority"]
            else 0,
            "total_samples": self.history["total_samples"][-1]
            if self.history["total_samples"]
            else 0,
            "total_updates": self.history["priority_updates"][-1]
            if self.history["priority_updates"]
            else 0,
        }

    def save(self, filepath: str):
        """
        Save history to JSON file

        Args:
            filepath: Path to save file
        """
        import json

        try:
            # Convert numpy types to native Python types
            history_serializable = {}
            for key, values in self.history.items():
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in values
                ]

            with open(filepath, "w") as f:
                json.dump(history_serializable, f, indent=2)

            logger.info(f"✅ PER history saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save PER history: {e}")

    def load(self, filepath: str):
        """
        Load history from JSON file

        Args:
            filepath: Path to load file
        """
        import json

        try:
            with open(filepath, "r") as f:
                self.history = json.load(f)

            logger.info(f"✅ PER history loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load PER history: {e}")

    def plot(self, output_path: str, title: str = "PER Statistics"):
        """
        Create visualization of PER statistics

        Args:
            output_path: Path to save plot
            title: Plot title

        Raises:
            ImportError: If matplotlib is not available
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not available for plotting")
            raise

        if len(self.history["beta"]) == 0:
            logger.warning("No data to plot")
            return

        # Determine x-axis (prefer episodes, fall back to steps or index)
        if len(self.history["episodes"]) > 0:
            x_values = self.history["episodes"]
            x_label = "Episode"
        elif len(self.history["steps"]) > 0:
            x_values = self.history["steps"]
            x_label = "Step"
        else:
            x_values = list(range(len(self.history["beta"])))
            x_label = "Sample"

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=16)

        # Plot 1: Beta annealing
        axes[0, 0].plot(x_values, self.history["beta"], label="Beta", color="blue")
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel("Beta (IS correction)")
        axes[0, 0].set_title("Beta Annealing Progress")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # Plot 2: Priority statistics
        axes[0, 1].plot(
            x_values, self.history["mean_priority"], label="Mean", color="green"
        )
        axes[0, 1].plot(
            x_values,
            self.history["max_priority"],
            label="Max",
            color="red",
            alpha=0.6,
        )
        axes[0, 1].plot(
            x_values,
            self.history["min_priority"],
            label="Min",
            color="orange",
            alpha=0.6,
        )
        axes[0, 1].set_xlabel(x_label)
        axes[0, 1].set_ylabel("Priority")
        axes[0, 1].set_title("Priority Statistics")
        axes[0, 1].set_yscale("log")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

        # Plot 3: Buffer utilization
        buffer_utilization = [
            size / cap if cap > 0 else 0
            for size, cap in zip(
                self.history["buffer_size"], self.history["buffer_capacity"]
            )
        ]
        axes[1, 0].plot(
            x_values, buffer_utilization, label="Utilization", color="purple"
        )
        axes[1, 0].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Full")
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel("Buffer Utilization (%)")
        axes[1, 0].set_title("Replay Buffer Utilization")
        axes[1, 0].set_ylim(0, 1.1)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        # Plot 4: Total priority sum
        axes[1, 1].plot(
            x_values, self.history["total_priority"], label="Total Priority", color="teal"
        )
        axes[1, 1].set_xlabel(x_label)
        axes[1, 1].set_ylabel("Total Priority")
        axes[1, 1].set_title("Total Priority Sum")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()

        # Save plot
        try:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"✅ PER plot saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
        finally:
            plt.close()

    def __repr__(self) -> str:
        n_samples = len(self.history["beta"])
        return f"PERStatsTracker(samples={n_samples})"


def create_per_report(
    per_stats: Dict[str, float],
    tracker: Optional[PERStatsTracker] = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Create a detailed text report of PER statistics

    Args:
        per_stats: Current PER statistics
        tracker: Optional PERStatsTracker for historical data
        output_file: Optional file path to save report

    Returns:
        Report text
    """
    report_lines = [
        "=" * 60,
        "PRIORITIZED EXPERIENCE REPLAY (PER) REPORT",
        "=" * 60,
        "",
        "Current Statistics:",
        "-" * 60,
        f"Buffer Size:       {per_stats.get('size', 0):,} / {per_stats.get('capacity', 0):,}",
        f"Buffer Usage:      {per_stats.get('size', 0) / max(per_stats.get('capacity', 1), 1) * 100:.1f}%",
        "",
        f"Alpha (priority):  {per_stats.get('alpha', 0):.3f}",
        f"Beta (IS weight):  {per_stats.get('beta', 0):.3f}",
        "",
        f"Mean Priority:     {per_stats.get('mean_priority', 0):.6f}",
        f"Max Priority:      {per_stats.get('max_priority', 0):.6f}",
        f"Min Priority:      {per_stats.get('min_priority', 0):.6f}",
        f"Total Priority:    {per_stats.get('total_priority', 0):.2f}",
        "",
        f"Total Samples:     {per_stats.get('total_samples', 0):,}",
        f"Priority Updates:  {per_stats.get('priority_updates', 0):,}",
    ]

    # Add historical summary if available
    if tracker is not None:
        summary = tracker.get_summary()
        if summary:
            report_lines.extend(
                [
                    "",
                    "Historical Summary:",
                    "-" * 60,
                    f"Final Beta:          {summary['final_beta']:.3f}",
                    f"Mean Priority (avg): {summary['mean_priority_avg']:.6f}",
                    f"Mean Priority (std): {summary['mean_priority_std']:.6f}",
                    f"Max Priority (peak): {summary['max_priority_peak']:.6f}",
                    f"Total Samples:       {summary['total_samples']:,}",
                    f"Total Updates:       {summary['total_updates']:,}",
                ]
            )

    report_lines.append("=" * 60)

    report = "\n".join(report_lines)

    # Save to file if requested
    if output_file is not None:
        try:
            with open(output_file, "w") as f:
                f.write(report)
            logger.info(f"✅ PER report saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    return report


# Export main classes
__all__ = ["PERStatsTracker", "create_per_report"]
