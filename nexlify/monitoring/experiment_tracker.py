"""
Experiment Tracking System
Track, compare, and manage multiple training experiments
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import shutil

import numpy as np

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Comprehensive experiment tracking and comparison

    Features:
    - Log all hyperparameters
    - Track multiple runs simultaneously
    - Compare experiments side-by-side
    - Best model leaderboard
    - Export results to CSV/JSON

    Example:
        >>> tracker = ExperimentTracker()
        >>> exp_id = tracker.create_experiment(
        ...     name="dqn_baseline",
        ...     hyperparameters={'lr': 0.001, 'gamma': 0.99}
        ... )
        >>> tracker.log_result(exp_id, episode=100, profit=500.0)
        >>> tracker.save_experiment(exp_id)
        >>> best = tracker.get_leaderboard()
    """

    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment tracker

        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Active experiments
        self.experiments: Dict[str, Dict[str, Any]] = {}

        # Leaderboard
        self.leaderboard_file = self.experiments_dir / "leaderboard.json"
        self.leaderboard = self._load_leaderboard()

        logger.info(f"ExperimentTracker initialized at {self.experiments_dir}")

    def create_experiment(
        self,
        name: str,
        hyperparameters: Dict[str, Any],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create new experiment

        Args:
            name: Experiment name
            hyperparameters: All hyperparameters
            description: Experiment description
            tags: Tags for categorization

        Returns:
            Experiment ID
        """
        # Generate unique ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id = f"{name}_{timestamp}"

        # Create experiment record
        experiment = {
            'id': exp_id,
            'name': name,
            'created': datetime.now().isoformat(),
            'hyperparameters': hyperparameters,
            'description': description,
            'tags': tags or [],
            'results': [],
            'status': 'running',
            'best_episode': None,
            'final_metrics': {}
        }

        # Add to active experiments
        self.experiments[exp_id] = experiment

        # Create experiment directory
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created experiment: {exp_id}")
        return exp_id

    def log_result(
        self,
        exp_id: str,
        episode: int,
        **metrics
    ) -> None:
        """
        Log episode results for experiment

        Args:
            exp_id: Experiment ID
            episode: Episode number
            **metrics: Episode metrics (profit, sharpe, etc.)
        """
        if exp_id not in self.experiments:
            logger.error(f"Experiment not found: {exp_id}")
            return

        result = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }

        self.experiments[exp_id]['results'].append(result)

        # Update best episode
        self._update_best_episode(exp_id, result)

    def complete_experiment(
        self,
        exp_id: str,
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark experiment as complete

        Args:
            exp_id: Experiment ID
            final_metrics: Final summary metrics
        """
        if exp_id not in self.experiments:
            logger.error(f"Experiment not found: {exp_id}")
            return

        self.experiments[exp_id]['status'] = 'completed'
        self.experiments[exp_id]['completed'] = datetime.now().isoformat()

        if final_metrics:
            self.experiments[exp_id]['final_metrics'] = final_metrics

        # Save experiment
        self.save_experiment(exp_id)

        # Update leaderboard
        self._update_leaderboard(exp_id)

        logger.info(f"Experiment completed: {exp_id}")

    def save_experiment(self, exp_id: str) -> Path:
        """
        Save experiment to disk

        Args:
            exp_id: Experiment ID

        Returns:
            Path to saved file
        """
        if exp_id not in self.experiments:
            logger.error(f"Experiment not found: {exp_id}")
            return None

        exp_dir = self.experiments_dir / exp_id
        filepath = exp_dir / "experiment.json"

        with open(filepath, 'w') as f:
            json.dump(self.experiments[exp_id], f, indent=2)

        logger.info(f"Experiment saved: {filepath}")
        return filepath

    def load_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Load experiment from disk

        Args:
            exp_id: Experiment ID

        Returns:
            Experiment data or None if not found
        """
        filepath = self.experiments_dir / exp_id / "experiment.json"

        if not filepath.exists():
            logger.error(f"Experiment file not found: {filepath}")
            return None

        with open(filepath) as f:
            experiment = json.load(f)

        self.experiments[exp_id] = experiment
        logger.info(f"Experiment loaded: {exp_id}")
        return experiment

    def compare_experiments(
        self,
        exp_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments

        Args:
            exp_ids: List of experiment IDs
            metrics: Metrics to compare (None for all)

        Returns:
            Comparison data
        """
        comparison = {
            'experiments': [],
            'metrics_comparison': {}
        }

        default_metrics = ['profit', 'sharpe', 'win_rate', 'drawdown']
        metrics = metrics or default_metrics

        for exp_id in exp_ids:
            # Load if not in memory
            if exp_id not in self.experiments:
                self.load_experiment(exp_id)

            if exp_id not in self.experiments:
                logger.warning(f"Skipping unknown experiment: {exp_id}")
                continue

            exp = self.experiments[exp_id]
            results = exp['results']

            if not results:
                continue

            # Calculate statistics
            exp_stats = {
                'id': exp_id,
                'name': exp['name'],
                'hyperparameters': exp['hyperparameters'],
                'total_episodes': len(results),
                'best_episode': exp.get('best_episode', {}),
            }

            # Per-metric stats
            for metric in metrics:
                values = [r.get(metric, 0) for r in results if metric in r]

                if values:
                    exp_stats[f'{metric}_mean'] = float(np.mean(values))
                    exp_stats[f'{metric}_std'] = float(np.std(values))
                    exp_stats[f'{metric}_max'] = float(np.max(values))
                    exp_stats[f'{metric}_min'] = float(np.min(values))

            comparison['experiments'].append(exp_stats)

        # Cross-experiment comparison
        for metric in metrics:
            metric_values = []
            for exp in comparison['experiments']:
                if f'{metric}_mean' in exp:
                    metric_values.append({
                        'experiment': exp['name'],
                        'mean': exp[f'{metric}_mean'],
                        'std': exp[f'{metric}_std'],
                        'max': exp[f'{metric}_max']
                    })

            comparison['metrics_comparison'][metric] = metric_values

        return comparison

    def get_leaderboard(
        self,
        metric: str = 'profit',
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard of best experiments

        Args:
            metric: Metric to rank by
            top_n: Number of top experiments to return

        Returns:
            List of top experiments
        """
        # Sort by metric
        sorted_board = sorted(
            self.leaderboard,
            key=lambda x: x.get(f'best_{metric}', float('-inf')),
            reverse=True
        )

        return sorted_board[:top_n]

    def export_comparison(
        self,
        exp_ids: List[str],
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> Path:
        """
        Export experiment comparison

        Args:
            exp_ids: Experiment IDs to compare
            format: Output format ('json' or 'csv')
            output_path: Output file path (auto-generated if None)

        Returns:
            Path to output file
        """
        comparison = self.compare_experiments(exp_ids)

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.experiments_dir / f"comparison_{timestamp}.{format}"
        else:
            output_path = Path(output_path)

        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)

        elif format == 'csv':
            import csv

            # Flatten experiments for CSV
            if comparison['experiments']:
                with open(output_path, 'w', newline='') as f:
                    fieldnames = comparison['experiments'][0].keys()
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()

                    for exp in comparison['experiments']:
                        # Handle nested dicts
                        row = {}
                        for key, value in exp.items():
                            if isinstance(value, dict):
                                row[key] = json.dumps(value)
                            else:
                                row[key] = value
                        writer.writerow(row)

        logger.info(f"Comparison exported to {output_path}")
        return output_path

    def archive_experiment(self, exp_id: str) -> None:
        """
        Archive experiment

        Args:
            exp_id: Experiment ID
        """
        exp_dir = self.experiments_dir / exp_id
        archive_dir = self.experiments_dir / "archived"
        archive_dir.mkdir(exist_ok=True)

        if exp_dir.exists():
            shutil.move(str(exp_dir), str(archive_dir / exp_id))
            logger.info(f"Experiment archived: {exp_id}")

        # Remove from active experiments
        if exp_id in self.experiments:
            del self.experiments[exp_id]

    def delete_experiment(self, exp_id: str, confirm: bool = False) -> None:
        """
        Delete experiment permanently

        Args:
            exp_id: Experiment ID
            confirm: Confirmation flag (safety)
        """
        if not confirm:
            logger.warning(
                f"Delete not confirmed for {exp_id}. "
                "Set confirm=True to delete."
            )
            return

        exp_dir = self.experiments_dir / exp_id

        if exp_dir.exists():
            shutil.rmtree(exp_dir)
            logger.info(f"Experiment deleted: {exp_id}")

        # Remove from active experiments
        if exp_id in self.experiments:
            del self.experiments[exp_id]

        # Remove from leaderboard
        self.leaderboard = [
            e for e in self.leaderboard if e['id'] != exp_id
        ]
        self._save_leaderboard()

    def list_experiments(
        self,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering

        Args:
            status: Filter by status ('running', 'completed')
            tags: Filter by tags

        Returns:
            List of experiment summaries
        """
        # Load all experiments from disk
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != 'archived':
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists() and exp_dir.name not in self.experiments:
                    self.load_experiment(exp_dir.name)

        experiments = list(self.experiments.values())

        # Filter by status
        if status:
            experiments = [e for e in experiments if e['status'] == status]

        # Filter by tags
        if tags:
            experiments = [
                e for e in experiments
                if any(tag in e.get('tags', []) for tag in tags)
            ]

        return experiments

    def _update_best_episode(
        self,
        exp_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Update best episode for experiment"""
        exp = self.experiments[exp_id]
        current_best = exp.get('best_episode')

        if current_best is None:
            exp['best_episode'] = result
            return

        # Compare by profit
        if result.get('profit', 0) > current_best.get('profit', 0):
            exp['best_episode'] = result

    def _update_leaderboard(self, exp_id: str) -> None:
        """Update leaderboard with experiment results"""
        exp = self.experiments[exp_id]
        results = exp['results']

        if not results:
            return

        # Calculate best metrics
        profits = [r.get('profit', 0) for r in results if 'profit' in r]
        sharpes = [r.get('sharpe', 0) for r in results if 'sharpe' in r]

        entry = {
            'id': exp_id,
            'name': exp['name'],
            'created': exp['created'],
            'best_profit': max(profits) if profits else 0,
            'best_sharpe': max(sharpes) if sharpes else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'total_episodes': len(results),
            'hyperparameters': exp['hyperparameters']
        }

        # Remove old entry if exists
        self.leaderboard = [e for e in self.leaderboard if e['id'] != exp_id]

        # Add new entry
        self.leaderboard.append(entry)

        # Save leaderboard
        self._save_leaderboard()

    def _load_leaderboard(self) -> List[Dict[str, Any]]:
        """Load leaderboard from disk"""
        if not self.leaderboard_file.exists():
            return []

        with open(self.leaderboard_file) as f:
            return json.load(f)

    def _save_leaderboard(self) -> None:
        """Save leaderboard to disk"""
        with open(self.leaderboard_file, 'w') as f:
            json.dump(self.leaderboard, f, indent=2)

    def get_experiment_summary(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment summary

        Args:
            exp_id: Experiment ID

        Returns:
            Summary dictionary or None
        """
        if exp_id not in self.experiments:
            self.load_experiment(exp_id)

        if exp_id not in self.experiments:
            return None

        exp = self.experiments[exp_id]
        results = exp['results']

        if not results:
            return {
                'id': exp_id,
                'name': exp['name'],
                'status': exp['status'],
                'total_episodes': 0
            }

        # Calculate summary statistics
        profits = [r.get('profit', 0) for r in results if 'profit' in r]
        sharpes = [r.get('sharpe', 0) for r in results if 'sharpe' in r]

        summary = {
            'id': exp_id,
            'name': exp['name'],
            'status': exp['status'],
            'created': exp['created'],
            'total_episodes': len(results),
            'hyperparameters': exp['hyperparameters'],
            'best_episode': exp.get('best_episode', {}),
            'statistics': {
                'profit': {
                    'mean': float(np.mean(profits)) if profits else 0,
                    'std': float(np.std(profits)) if profits else 0,
                    'max': float(np.max(profits)) if profits else 0,
                    'min': float(np.min(profits)) if profits else 0,
                },
                'sharpe': {
                    'mean': float(np.mean(sharpes)) if sharpes else 0,
                    'std': float(np.std(sharpes)) if sharpes else 0,
                    'max': float(np.max(sharpes)) if sharpes else 0,
                    'min': float(np.min(sharpes)) if sharpes else 0,
                }
            }
        }

        return summary
