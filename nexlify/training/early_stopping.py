#!/usr/bin/env python3
"""
Nexlify Early Stopping System
==============================
Comprehensive early stopping with training phase detection and overfitting monitoring.

Features:
- Early stopping with configurable patience
- Training phase detection (exploration/learning/exploitation)
- Adaptive patience based on training phase
- Overfitting detection and alerts
- Best model weight restoration
- Comprehensive logging and notifications
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Training phase classification"""
    EXPLORATION = "exploration"  # epsilon > 0.7
    LEARNING = "learning"  # 0.3 < epsilon <= 0.7
    EXPLOITATION = "exploitation"  # epsilon <= 0.3


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    patience: int = 30
    min_delta: float = 0.01
    mode: str = 'max'  # 'max' for metrics to maximize, 'min' for minimize
    metric: str = 'val_sharpe'
    restore_best_weights: bool = True

    # Adaptive patience based on training phase
    exploration_patience_multiplier: float = 2.0  # More patience during exploration
    learning_patience_multiplier: float = 1.5
    exploitation_patience_multiplier: float = 1.0

    # Minimum episodes before early stopping can trigger
    min_episodes: int = 100

    # Save best model automatically
    save_best_model: bool = True
    model_save_path: Optional[str] = None


class TrainingPhaseDetector:
    """
    Detects current training phase based on epsilon and other metrics

    Adjusts early stopping patience based on phase:
    - Exploration: More lenient (allow more variation)
    - Learning: Moderate leniency
    - Exploitation: Strict (expect consistent performance)
    """

    def __init__(
        self,
        exploration_threshold: float = 0.7,
        learning_threshold: float = 0.3
    ):
        """
        Initialize training phase detector

        Args:
            exploration_threshold: Epsilon above this = exploration phase
            learning_threshold: Epsilon below this = exploitation phase
        """
        self.exploration_threshold = exploration_threshold
        self.learning_threshold = learning_threshold

        self.phase_history: List[Tuple[int, TrainingPhase]] = []
        self.current_phase = TrainingPhase.EXPLORATION

        logger.info(f"🎯 TrainingPhaseDetector initialized:")
        logger.info(f"   Exploration: ε > {exploration_threshold}")
        logger.info(f"   Learning: {learning_threshold} < ε ≤ {exploration_threshold}")
        logger.info(f"   Exploitation: ε ≤ {learning_threshold}")

    def detect_phase(self, epsilon: float, episode: int) -> TrainingPhase:
        """
        Detect current training phase based on epsilon

        Args:
            epsilon: Current exploration rate
            episode: Current episode number

        Returns:
            Current training phase
        """
        if epsilon > self.exploration_threshold:
            phase = TrainingPhase.EXPLORATION
        elif epsilon > self.learning_threshold:
            phase = TrainingPhase.LEARNING
        else:
            phase = TrainingPhase.EXPLOITATION

        # Track phase changes
        if phase != self.current_phase:
            logger.info(
                f"🔄 Training phase changed: {self.current_phase.value} → {phase.value} "
                f"(Episode {episode}, ε={epsilon:.4f})"
            )
            self.current_phase = phase

        self.phase_history.append((episode, phase))
        return phase

    def get_patience_multiplier(
        self,
        phase: TrainingPhase,
        config: EarlyStoppingConfig
    ) -> float:
        """
        Get patience multiplier for current phase

        Args:
            phase: Current training phase
            config: Early stopping configuration

        Returns:
            Patience multiplier
        """
        if phase == TrainingPhase.EXPLORATION:
            return config.exploration_patience_multiplier
        elif phase == TrainingPhase.LEARNING:
            return config.learning_patience_multiplier
        else:
            return config.exploitation_patience_multiplier


class OverfittingDetector:
    """
    Monitors overfitting by comparing training and validation performance

    Alerts when training performance significantly exceeds validation performance.
    """

    def __init__(
        self,
        overfitting_threshold: float = 0.20,
        window_size: int = 10,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize overfitting detector

        Args:
            overfitting_threshold: Threshold for overfitting score (default: 0.20 = 20%)
            window_size: Window for averaging metrics (default: 10)
            alert_callback: Optional callback when overfitting detected
        """
        self.overfitting_threshold = overfitting_threshold
        self.window_size = window_size
        self.alert_callback = alert_callback

        # Tracking
        self.train_metrics: List[float] = []
        self.val_metrics: List[float] = []
        self.overfitting_scores: List[float] = []
        self.alerts_triggered = 0
        self.chronic_overfitting_count = 0

        logger.info(f"🔍 OverfittingDetector initialized:")
        logger.info(f"   Threshold: {overfitting_threshold * 100:.1f}%")
        logger.info(f"   Window size: {window_size}")

    def update(
        self,
        train_metric: float,
        val_metric: float,
        episode: int
    ) -> Tuple[bool, float]:
        """
        Update overfitting detector with new metrics

        Args:
            train_metric: Training metric value
            val_metric: Validation metric value
            episode: Current episode number

        Returns:
            Tuple of (is_overfitting, overfitting_score)
        """
        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)

        # Calculate overfitting score
        # Score = (train - val) / train (percentage difference)
        if train_metric != 0:
            overfitting_score = (train_metric - val_metric) / abs(train_metric)
        else:
            overfitting_score = 0.0

        self.overfitting_scores.append(overfitting_score)

        # Check if overfitting
        is_overfitting = overfitting_score > self.overfitting_threshold

        # Check for chronic overfitting (persistent over window)
        if len(self.overfitting_scores) >= self.window_size:
            recent_scores = self.overfitting_scores[-self.window_size:]
            avg_score = np.mean(recent_scores)

            if avg_score > self.overfitting_threshold:
                self.chronic_overfitting_count += 1

                if self.chronic_overfitting_count >= 3:  # 3 consecutive windows
                    logger.warning(
                        f"⚠️  CHRONIC OVERFITTING DETECTED (Episode {episode})"
                    )
                    logger.warning(
                        f"   Avg overfitting score: {avg_score * 100:.1f}% "
                        f"(threshold: {self.overfitting_threshold * 100:.1f}%)"
                    )
                    logger.warning(
                        f"   Train metric: {np.mean(self.train_metrics[-self.window_size:]):.3f}, "
                        f"Val metric: {np.mean(self.val_metrics[-self.window_size:]):.3f}"
                    )
                    logger.warning("   Consider: increased regularization, smaller model, more data")

                    self.alerts_triggered += 1

                    # Trigger callback if provided
                    if self.alert_callback:
                        self.alert_callback({
                            'episode': episode,
                            'overfitting_score': avg_score,
                            'train_metric': train_metric,
                            'val_metric': val_metric,
                            'alert_type': 'chronic_overfitting'
                        })
            else:
                self.chronic_overfitting_count = 0

        # Log occasional overfitting warnings
        if is_overfitting and episode % 50 == 0:
            logger.debug(
                f"⚠️  Potential overfitting (Episode {episode}): "
                f"Train={train_metric:.3f}, Val={val_metric:.3f}, "
                f"Score={overfitting_score * 100:.1f}%"
            )

        return is_overfitting, overfitting_score

    def get_overfitting_summary(self) -> Dict[str, Any]:
        """
        Get summary of overfitting analysis

        Returns:
            Dictionary with overfitting statistics
        """
        if not self.overfitting_scores:
            return {}

        return {
            'avg_overfitting_score': float(np.mean(self.overfitting_scores)),
            'max_overfitting_score': float(np.max(self.overfitting_scores)),
            'alerts_triggered': self.alerts_triggered,
            'chronic_overfitting_detected': self.chronic_overfitting_count >= 3,
            'pct_episodes_overfitting': float(
                sum(1 for s in self.overfitting_scores if s > self.overfitting_threshold)
                / len(self.overfitting_scores) * 100
            )
        }


class EarlyStopping:
    """
    Early stopping monitor with training phase awareness

    Stops training when validation metric stops improving.
    Adapts patience based on training phase for more intelligent stopping.
    """

    def __init__(
        self,
        config: Optional[EarlyStoppingConfig] = None,
        phase_detector: Optional[TrainingPhaseDetector] = None,
        overfitting_detector: Optional[OverfittingDetector] = None
    ):
        """
        Initialize early stopping

        Args:
            config: Early stopping configuration
            phase_detector: Optional training phase detector
            overfitting_detector: Optional overfitting detector
        """
        self.config = config or EarlyStoppingConfig()
        self.phase_detector = phase_detector or TrainingPhaseDetector()
        self.overfitting_detector = overfitting_detector

        # Tracking
        self.best_metric = -np.inf if self.config.mode == 'max' else np.inf
        self.best_episode = 0
        self.best_weights = None
        self.patience_counter = 0
        self.current_patience = self.config.patience
        self.stopped = False
        self.stop_episode = None

        # History
        self.metric_history: List[Tuple[int, float]] = []

        logger.info(f"🛑 EarlyStopping initialized:")
        logger.info(f"   Metric: {self.config.metric}")
        logger.info(f"   Mode: {self.config.mode}")
        logger.info(f"   Base patience: {self.config.patience}")
        logger.info(f"   Min delta: {self.config.min_delta}")
        logger.info(f"   Min episodes: {self.config.min_episodes}")
        logger.info(f"   Restore best weights: {self.config.restore_best_weights}")

    def update(
        self,
        metric_value: float,
        episode: int,
        epsilon: float,
        model_weights: Optional[Any] = None,
        train_metric: Optional[float] = None
    ) -> bool:
        """
        Update early stopping monitor

        Args:
            metric_value: Current validation metric value
            episode: Current episode number
            epsilon: Current exploration rate
            model_weights: Current model weights (for restoration)
            train_metric: Optional training metric (for overfitting detection)

        Returns:
            True if training should stop, False otherwise
        """
        # Don't stop before minimum episodes
        if episode < self.config.min_episodes:
            return False

        # Detect training phase and adjust patience
        phase = self.phase_detector.detect_phase(epsilon, episode)
        patience_multiplier = self.phase_detector.get_patience_multiplier(phase, self.config)
        self.current_patience = int(self.config.patience * patience_multiplier)

        # Track metric
        self.metric_history.append((episode, metric_value))

        # Update overfitting detector if available
        if self.overfitting_detector and train_metric is not None:
            self.overfitting_detector.update(train_metric, metric_value, episode)

        # Check if this is a new best
        is_improvement = self._is_improvement(metric_value)

        if is_improvement:
            improvement_amount = abs(metric_value - self.best_metric)
            logger.info(
                f"✨ Validation improved: {self.best_metric:.4f} → {metric_value:.4f} "
                f"(+{improvement_amount:.4f}, Episode {episode})"
            )

            self.best_metric = metric_value
            self.best_episode = episode
            self.patience_counter = 0

            # Save best weights
            if model_weights is not None and self.config.restore_best_weights:
                self.best_weights = model_weights
                logger.debug(f"💾 Saved best model weights (Episode {episode})")
        else:
            self.patience_counter += 1

            if self.patience_counter % 10 == 0:  # Log every 10 episodes without improvement
                logger.info(
                    f"⏳ No improvement for {self.patience_counter} episodes "
                    f"(patience: {self.current_patience}, phase: {phase.value})"
                )

        # Check if should stop
        if self.patience_counter >= self.current_patience:
            self.stopped = True
            self.stop_episode = episode

            logger.info(f"\n{'='*80}")
            logger.info(f"🛑 EARLY STOPPING TRIGGERED (Episode {episode})")
            logger.info(f"{'='*80}")
            logger.info(f"   Reason: No improvement for {self.patience_counter} episodes")
            logger.info(f"   Training phase: {phase.value}")
            logger.info(f"   Best {self.config.metric}: {self.best_metric:.4f} (Episode {self.best_episode})")
            logger.info(f"   Current {self.config.metric}: {metric_value:.4f}")
            logger.info(f"   Episodes saved: {episode - self.best_episode}")

            if self.overfitting_detector:
                overfitting_summary = self.overfitting_detector.get_overfitting_summary()
                if overfitting_summary:
                    logger.info(f"\n   Overfitting Analysis:")
                    logger.info(f"      Avg overfitting score: {overfitting_summary['avg_overfitting_score']*100:.1f}%")
                    logger.info(f"      Chronic overfitting: {'YES' if overfitting_summary['chronic_overfitting_detected'] else 'NO'}")

            logger.info(f"{'='*80}\n")

            return True

        return False

    def _is_improvement(self, metric_value: float) -> bool:
        """
        Check if metric value is an improvement

        Args:
            metric_value: Current metric value

        Returns:
            True if improvement, False otherwise
        """
        if self.config.mode == 'max':
            return metric_value > (self.best_metric + self.config.min_delta)
        else:
            return metric_value < (self.best_metric - self.config.min_delta)

    def restore_best_weights(self, agent) -> bool:
        """
        Restore best weights to agent

        Args:
            agent: RL agent to restore weights to

        Returns:
            True if weights restored, False if no weights available
        """
        if not self.config.restore_best_weights:
            logger.warning("Restore best weights is disabled in config")
            return False

        if self.best_weights is None:
            logger.warning("No best weights available to restore")
            return False

        try:
            # This assumes agent has a method to load weights
            # Implementation depends on agent architecture
            if hasattr(agent, 'load_weights'):
                agent.load_weights(self.best_weights)
            elif hasattr(agent, 'model') and hasattr(agent.model, 'load_state_dict'):
                agent.model.load_state_dict(self.best_weights)
            elif hasattr(agent, 'model') and hasattr(agent.model, 'set_weights'):
                agent.model.set_weights(self.best_weights)
            else:
                logger.warning("Agent doesn't have a recognized method to load weights")
                return False

            logger.info(f"✅ Restored best weights from episode {self.best_episode}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore best weights: {e}")
            return False

    def should_save_checkpoint(self, episode: int) -> bool:
        """
        Check if checkpoint should be saved at this episode

        Args:
            episode: Current episode

        Returns:
            True if should save checkpoint
        """
        return episode == self.best_episode

    def get_summary(self) -> Dict[str, Any]:
        """
        Get early stopping summary

        Returns:
            Dictionary with early stopping statistics
        """
        summary = {
            'stopped': self.stopped,
            'stop_episode': self.stop_episode,
            'best_episode': self.best_episode,
            'best_metric': float(self.best_metric),
            'patience_counter': self.patience_counter,
            'current_patience': self.current_patience,
            'episodes_saved': (self.stop_episode - self.best_episode) if self.stopped else 0,
        }

        # Add overfitting summary if available
        if self.overfitting_detector:
            summary['overfitting'] = self.overfitting_detector.get_overfitting_summary()

        return summary

    def plot_metric_history(self, save_path: Optional[Path] = None):
        """
        Plot metric history with early stopping point marked

        Args:
            save_path: Optional path to save plot
        """
        if not self.metric_history:
            logger.warning("No metric history to plot")
            return

        try:
            import matplotlib.pyplot as plt

            episodes, metrics = zip(*self.metric_history)

            plt.figure(figsize=(12, 6))
            plt.plot(episodes, metrics, 'b-', linewidth=2, label='Validation Metric')
            plt.axvline(x=self.best_episode, color='g', linestyle='--',
                       label=f'Best (Episode {self.best_episode})')

            if self.stopped:
                plt.axvline(x=self.stop_episode, color='r', linestyle='--',
                           label=f'Early Stop (Episode {self.stop_episode})')

            plt.xlabel('Episode')
            plt.ylabel(self.config.metric)
            plt.title(f'Early Stopping Monitor: {self.config.metric}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"📊 Early stopping plot saved to {save_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting metric history: {e}")


__all__ = [
    'EarlyStopping',
    'EarlyStoppingConfig',
    'TrainingPhaseDetector',
    'TrainingPhase',
    'OverfittingDetector',
]
