#!/usr/bin/env python3
"""
Architecture Comparison Tools
Compare performance of different DQN variants

Compares:
- Standard DQN
- Double DQN
- Dueling DQN
- Double + Dueling DQN

Metrics:
- Training convergence speed
- Q-value overestimation
- Final performance
- Sample efficiency
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ArchitectureComparator:
    """
    Compare different DQN architecture variants

    Runs ablation studies to determine:
    - Which architecture performs best
    - Overestimation reduction from Double DQN
    - Value estimation improvement from Dueling DQN
    - Combined benefit of both
    """

    def __init__(self, output_dir: str = "comparison_results"):
        """
        Initialize architecture comparator

        Args:
            output_dir: Directory to save comparison results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "standard_dqn": [],
            "double_dqn": [],
            "dueling_dqn": [],
            "double_dueling_dqn": [],
        }

        self.comparison_metrics = {}

        logger.info(f"📊 Architecture Comparator initialized (output: {output_dir})")

    def add_result(
        self,
        architecture: str,
        episode: int,
        reward: float,
        loss: float,
        q_value_stats: Optional[Dict[str, Any]] = None,
    ):
        """
        Add training result for an architecture

        Args:
            architecture: Architecture name (standard_dqn, double_dqn, etc.)
            episode: Episode number
            reward: Episode reward
            loss: Training loss
            q_value_stats: Q-value statistics (optional)
        """
        if architecture not in self.results:
            logger.warning(f"Unknown architecture: {architecture}")
            return

        result = {
            "episode": episode,
            "reward": reward,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }

        if q_value_stats:
            result["q_value_stats"] = q_value_stats

        self.results[architecture].append(result)

    def compute_metrics(self, window: int = 100) -> Dict[str, Any]:
        """
        Compute comparison metrics across architectures

        Args:
            window: Moving average window for smoothing

        Returns:
            Dictionary of comparison metrics
        """
        metrics = {}

        for arch_name, results in self.results.items():
            if not results:
                continue

            rewards = [r["reward"] for r in results]
            losses = [r["loss"] for r in results]

            # Smooth rewards
            smoothed_rewards = self._moving_average(rewards, window)

            metrics[arch_name] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "final_reward": np.mean(rewards[-window:]) if len(rewards) >= window else np.mean(rewards),
                "best_reward": np.max(rewards),
                "mean_loss": np.mean(losses),
                "convergence_speed": self._estimate_convergence_speed(smoothed_rewards),
                "sample_efficiency": self._estimate_sample_efficiency(rewards),
            }

            # Add Q-value overestimation metrics if available
            q_stats_list = [r.get("q_value_stats") for r in results if r.get("q_value_stats")]
            if q_stats_list:
                overestimations = [
                    q.get("mean_overestimation", 0)
                    for q in q_stats_list
                    if "mean_overestimation" in q
                ]
                if overestimations:
                    metrics[arch_name]["mean_overestimation"] = np.mean(overestimations)
                    metrics[arch_name]["overestimation_reduction"] = np.mean([
                        q.get("overestimation_reduction", 0)
                        for q in q_stats_list
                        if "overestimation_reduction" in q
                    ])

        self.comparison_metrics = metrics
        return metrics

    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Compute moving average"""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def _estimate_convergence_speed(self, smoothed_rewards: np.ndarray) -> int:
        """
        Estimate convergence speed (episodes to reach 90% of final performance)

        Args:
            smoothed_rewards: Smoothed reward curve

        Returns:
            Number of episodes to convergence
        """
        if len(smoothed_rewards) < 10:
            return len(smoothed_rewards)

        final_perf = np.mean(smoothed_rewards[-10:])
        target = 0.9 * final_perf

        # Find first episode where performance exceeds target
        for i, reward in enumerate(smoothed_rewards):
            if reward >= target:
                return i

        return len(smoothed_rewards)

    def _estimate_sample_efficiency(self, rewards: List[float]) -> float:
        """
        Estimate sample efficiency (area under learning curve)

        Higher = better (learns faster)

        Args:
            rewards: Episode rewards

        Returns:
            Sample efficiency score
        """
        if not rewards:
            return 0.0

        # Normalize to [0, 1]
        min_reward = min(rewards)
        max_reward = max(rewards)

        if max_reward == min_reward:
            return 0.0

        normalized = [(r - min_reward) / (max_reward - min_reward) for r in rewards]

        # Area under curve (trapezoid rule)
        auc = np.trapz(normalized)

        # Normalize by length
        return auc / len(rewards)

    def get_best_architecture(self) -> Tuple[str, Dict[str, Any]]:
        """
        Determine best performing architecture

        Returns:
            Tuple of (architecture_name, metrics)
        """
        if not self.comparison_metrics:
            self.compute_metrics()

        if not self.comparison_metrics:
            return "none", {}

        # Rank by final reward
        best_arch = max(
            self.comparison_metrics.items(),
            key=lambda x: x[1].get("final_reward", -float("inf"))
        )

        return best_arch[0], best_arch[1]

    def generate_report(self) -> str:
        """
        Generate human-readable comparison report

        Returns:
            Report string
        """
        if not self.comparison_metrics:
            self.compute_metrics()

        report = [
            "=" * 80,
            "DQN Architecture Comparison Report",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Architectures Compared:",
        ]

        for arch_name in self.results.keys():
            count = len(self.results[arch_name])
            report.append(f"  - {arch_name}: {count} episodes")

        report.append("")
        report.append("Performance Metrics:")
        report.append("-" * 80)

        # Table header
        report.append(
            f"{'Architecture':<25} {'Final Reward':>15} {'Convergence':>15} {'Efficiency':>15}"
        )
        report.append("-" * 80)

        # Sort by final reward
        sorted_archs = sorted(
            self.comparison_metrics.items(),
            key=lambda x: x[1].get("final_reward", -float("inf")),
            reverse=True,
        )

        for arch_name, metrics in sorted_archs:
            final_reward = metrics.get("final_reward", 0)
            convergence = metrics.get("convergence_speed", 0)
            efficiency = metrics.get("sample_efficiency", 0)

            report.append(
                f"{arch_name:<25} {final_reward:>15.2f} {convergence:>15d} {efficiency:>15.4f}"
            )

        # Overestimation analysis
        report.append("")
        report.append("Overestimation Analysis (Double DQN variants only):")
        report.append("-" * 80)

        for arch_name, metrics in sorted_archs:
            if "mean_overestimation" in metrics:
                overest = metrics["mean_overestimation"]
                reduction = metrics.get("overestimation_reduction", 0)
                report.append(
                    f"{arch_name:<25} Overestimation: {overest:>10.4f}, "
                    f"Reduction: {reduction:>6.2f}%"
                )

        # Best architecture
        best_arch, best_metrics = self.get_best_architecture()
        report.append("")
        report.append("=" * 80)
        report.append(f"🏆 Best Architecture: {best_arch}")
        report.append(f"   Final Reward: {best_metrics.get('final_reward', 0):.2f}")
        report.append(
            f"   Convergence Speed: {best_metrics.get('convergence_speed', 0)} episodes"
        )
        report.append(
            f"   Sample Efficiency: {best_metrics.get('sample_efficiency', 0):.4f}"
        )
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, filename: str = "comparison_report.txt"):
        """
        Save comparison report to file

        Args:
            filename: Output filename
        """
        report = self.generate_report()
        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            f.write(report)

        logger.info(f"📄 Comparison report saved to {output_path}")

    def save_results(self, filename: str = "comparison_results.json"):
        """
        Save detailed results to JSON

        Args:
            filename: Output filename
        """
        output_path = self.output_dir / filename

        data = {
            "results": self.results,
            "metrics": self.comparison_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"💾 Comparison results saved to {output_path}")

    def load_results(self, filename: str = "comparison_results.json"):
        """
        Load previous results from JSON

        Args:
            filename: Input filename
        """
        input_path = self.output_dir / filename

        if not input_path.exists():
            logger.warning(f"Results file not found: {input_path}")
            return

        with open(input_path, "r") as f:
            data = json.load(f)

        self.results = data.get("results", {})
        self.comparison_metrics = data.get("metrics", {})

        logger.info(f"📂 Loaded comparison results from {input_path}")


def run_ablation_study(
    env,
    episodes: int = 1000,
    output_dir: str = "ablation_study",
) -> ArchitectureComparator:
    """
    Run ablation study comparing all DQN variants

    Args:
        env: Trading environment
        episodes: Number of episodes per architecture
        output_dir: Output directory for results

    Returns:
        ArchitectureComparator with results
    """
    from nexlify.strategies.double_dqn_agent import DoubleDQNAgent

    comparator = ArchitectureComparator(output_dir=output_dir)

    # Define architectures to test
    architectures = [
        ("standard_dqn", {"use_double_dqn": False, "use_dueling_dqn": False}),
        ("double_dqn", {"use_double_dqn": True, "use_dueling_dqn": False}),
        ("dueling_dqn", {"use_double_dqn": False, "use_dueling_dqn": True}),
        ("double_dueling_dqn", {"use_double_dqn": True, "use_dueling_dqn": True}),
    ]

    logger.info(f"🧪 Starting ablation study: {len(architectures)} architectures × {episodes} episodes")

    for arch_name, config in architectures:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {arch_name}")
        logger.info(f"{'='*60}")

        # Create agent
        agent = DoubleDQNAgent(
            state_size=env.state_space_n,
            action_size=env.action_space_n,
            config=config,
        )

        # Train
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            loss_total = 0
            steps = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)

                loss = agent.replay()
                loss_total += loss

                episode_reward += reward
                state = next_state
                steps += 1

            # Decay epsilon
            agent.decay_epsilon()

            # Log result
            avg_loss = loss_total / max(steps, 1)
            q_stats = agent.get_q_value_stats()

            comparator.add_result(
                architecture=arch_name,
                episode=episode,
                reward=episode_reward,
                loss=avg_loss,
                q_value_stats=q_stats,
            )

            if (episode + 1) % 100 == 0:
                logger.info(
                    f"  Episode {episode + 1}/{episodes}: "
                    f"Reward={episode_reward:.2f}, Loss={avg_loss:.4f}"
                )

    # Generate and save report
    logger.info("\n" + comparator.generate_report())
    comparator.save_report()
    comparator.save_results()

    return comparator


__all__ = ["ArchitectureComparator", "run_ablation_study"]
