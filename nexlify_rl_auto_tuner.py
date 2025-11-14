#!/usr/bin/env python3
"""
Automatic Hyperparameter Tuner for RL Trading Agent

Automatically adjusts hyperparameters based on training performance metrics.
Uses adaptive rules and heuristics to optimize learning.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TuningMetrics:
    """Metrics tracked for auto-tuning decisions"""
    episode: int
    avg_return: float  # Average return over last N episodes
    avg_sharpe: float  # Average Sharpe over last N episodes
    win_rate: float  # Win rate over last N episodes
    epsilon: float  # Current exploration rate
    learning_rate: float  # Current learning rate
    avg_loss: float  # Average training loss
    q_value_mean: float  # Average Q-value
    q_value_std: float  # Q-value standard deviation


@dataclass
class TuningState:
    """Current state of the tuning process"""
    # Performance tracking
    recent_returns: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_sharpes: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_win_rates: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_losses: deque = field(default_factory=lambda: deque(maxlen=100))

    # Tuning history
    adjustments_made: List[Dict] = field(default_factory=list)

    # Performance trend
    performance_trend: str = "unknown"  # "improving", "plateau", "declining", "unstable"
    episodes_since_improvement: int = 0
    best_performance: float = float('-inf')

    # Flags
    exploration_boosted: bool = False
    learning_rate_reduced: bool = False


class AutoHyperparameterTuner:
    """
    Automatically tunes hyperparameters during training

    Monitors training performance and makes intelligent adjustments:
    - Adapts epsilon decay based on learning progress
    - Adjusts learning rate if training is unstable
    - Modulates exploration if stuck in local minimum
    - Tunes batch size and update frequency
    """

    def __init__(self,
                 window_size: int = 50,
                 min_episodes_before_tuning: int = 100,
                 tuning_frequency: int = 50,
                 enable_lr_tuning: bool = True,
                 enable_epsilon_tuning: bool = True,
                 enable_architecture_tuning: bool = False,
                 verbose: bool = True):
        """
        Initialize auto-tuner

        Args:
            window_size: Number of episodes to track for metrics
            min_episodes_before_tuning: Minimum episodes before first adjustment
            tuning_frequency: Check for tuning every N episodes
            enable_lr_tuning: Enable learning rate adjustments
            enable_epsilon_tuning: Enable epsilon adjustments
            enable_architecture_tuning: Enable network architecture changes
            verbose: Log tuning decisions
        """
        self.window_size = window_size
        self.min_episodes = min_episodes_before_tuning
        self.tuning_freq = tuning_frequency
        self.enable_lr_tuning = enable_lr_tuning
        self.enable_epsilon_tuning = enable_epsilon_tuning
        self.enable_architecture_tuning = enable_architecture_tuning
        self.verbose = verbose

        self.state = TuningState()

    def update(self, metrics: TuningMetrics) -> Dict[str, any]:
        """
        Update tuner with latest metrics and get adjustments

        Args:
            metrics: Current training metrics

        Returns:
            Dictionary of parameter adjustments to apply
        """
        # Track metrics
        self.state.recent_returns.append(metrics.avg_return)
        self.state.recent_sharpes.append(metrics.avg_sharpe)
        self.state.recent_win_rates.append(metrics.win_rate)
        self.state.recent_losses.append(metrics.avg_loss)

        # Check if it's time to tune
        if metrics.episode < self.min_episodes:
            return {}  # Too early

        if metrics.episode % self.tuning_freq != 0:
            return {}  # Not tuning episode

        # Analyze performance trend
        self._analyze_trend(metrics)

        # Get tuning recommendations
        adjustments = {}

        if self.enable_epsilon_tuning:
            epsilon_adj = self._tune_epsilon(metrics)
            if epsilon_adj:
                adjustments.update(epsilon_adj)

        if self.enable_lr_tuning:
            lr_adj = self._tune_learning_rate(metrics)
            if lr_adj:
                adjustments.update(lr_adj)

        # Log adjustments
        if adjustments and self.verbose:
            self._log_adjustments(metrics.episode, adjustments)

        # Record adjustment
        if adjustments:
            self.state.adjustments_made.append({
                'episode': metrics.episode,
                'adjustments': adjustments,
                'trend': self.state.performance_trend
            })

        return adjustments

    def _analyze_trend(self, metrics: TuningMetrics):
        """Analyze performance trend over recent episodes"""
        if len(self.state.recent_returns) < 20:
            self.state.performance_trend = "unknown"
            return

        # Calculate recent performance
        recent_perf = np.mean(list(self.state.recent_returns)[-20:])

        # Check if improving
        if recent_perf > self.state.best_performance:
            self.state.best_performance = recent_perf
            self.state.performance_trend = "improving"
            self.state.episodes_since_improvement = 0
        else:
            self.state.episodes_since_improvement += self.tuning_freq

        # Detect plateau (no improvement for 200+ episodes)
        if self.state.episodes_since_improvement > 200:
            self.state.performance_trend = "plateau"

        # Detect decline
        if recent_perf < self.state.best_performance * 0.8:
            self.state.performance_trend = "declining"

        # Detect instability (high variance)
        if len(self.state.recent_returns) >= 30:
            returns_std = np.std(list(self.state.recent_returns)[-30:])
            returns_mean = np.mean(list(self.state.recent_returns)[-30:])
            if abs(returns_mean) > 0:
                cv = returns_std / abs(returns_mean)
                if cv > 2.0:  # Coefficient of variation > 2
                    self.state.performance_trend = "unstable"

    def _tune_epsilon(self, metrics: TuningMetrics) -> Optional[Dict]:
        """
        Tune epsilon based on performance

        Rules:
        1. If plateau → increase exploration temporarily
        2. If improving → let current schedule continue
        3. If unstable → reduce exploration
        4. If declining → boost exploration to escape local minimum
        """
        adjustments = {}

        if self.state.performance_trend == "plateau":
            # Stuck in plateau - boost exploration
            if not self.state.exploration_boosted:
                new_epsilon = min(metrics.epsilon * 1.5, 0.5)  # Boost up to 50%
                adjustments['epsilon'] = new_epsilon
                adjustments['epsilon_boost'] = True
                self.state.exploration_boosted = True

                if self.verbose:
                    logger.info(f"   📈 Plateau detected → Boosting exploration: {metrics.epsilon:.3f} → {new_epsilon:.3f}")

        elif self.state.performance_trend == "improving":
            # Learning well - reset boost flag
            if self.state.exploration_boosted:
                self.state.exploration_boosted = False

        elif self.state.performance_trend == "unstable":
            # Too much randomness - reduce exploration faster
            new_epsilon = max(metrics.epsilon * 0.8, 0.01)
            adjustments['epsilon'] = new_epsilon
            adjustments['epsilon_reduce_rate'] = 1.2  # Faster decay

            if self.verbose:
                logger.info(f"   🔻 Instability detected → Reducing exploration: {metrics.epsilon:.3f} → {new_epsilon:.3f}")

        elif self.state.performance_trend == "declining":
            # Performance dropping - might be stuck, boost exploration
            new_epsilon = min(metrics.epsilon * 2.0, 0.6)
            adjustments['epsilon'] = new_epsilon

            if self.verbose:
                logger.info(f"   ⚠️  Decline detected → Boosting exploration: {metrics.epsilon:.3f} → {new_epsilon:.3f}")

        return adjustments if adjustments else None

    def _tune_learning_rate(self, metrics: TuningMetrics) -> Optional[Dict]:
        """
        Tune learning rate based on loss trends

        Rules:
        1. If loss oscillating/exploding → reduce LR
        2. If loss plateaued but performance improving → OK
        3. If both loss and performance plateaued → reduce LR slightly
        4. If loss very low but not learning → increase LR
        """
        adjustments = {}

        if len(self.state.recent_losses) < 50:
            return None

        # Calculate loss statistics
        recent_losses = list(self.state.recent_losses)[-50:]
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]  # Linear trend

        # Check for exploding/oscillating loss
        if loss_std > loss_mean * 0.5 and loss_mean > 0.01:  # High variance
            # Reduce learning rate
            new_lr = metrics.learning_rate * 0.5
            adjustments['learning_rate'] = max(new_lr, 1e-6)
            self.state.learning_rate_reduced = True

            if self.verbose:
                logger.info(f"   🔻 High loss variance → Reducing LR: {metrics.learning_rate:.6f} → {new_lr:.6f}")

        # Check for very low loss but no learning
        elif loss_mean < 0.001 and self.state.performance_trend == "plateau":
            # Might be stuck - slightly increase LR
            new_lr = metrics.learning_rate * 1.2
            adjustments['learning_rate'] = min(new_lr, 0.001)

            if self.verbose:
                logger.info(f"   📈 Low loss + plateau → Increasing LR: {metrics.learning_rate:.6f} → {new_lr:.6f}")

        # Check for steadily increasing loss
        elif loss_trend > 0.0001:  # Loss increasing
            # Reduce learning rate
            new_lr = metrics.learning_rate * 0.7
            adjustments['learning_rate'] = max(new_lr, 1e-6)

            if self.verbose:
                logger.info(f"   ⚠️  Loss increasing → Reducing LR: {metrics.learning_rate:.6f} → {new_lr:.6f}")

        return adjustments if adjustments else None

    def _log_adjustments(self, episode: int, adjustments: Dict):
        """Log tuning adjustments"""
        logger.info(f"\n{'='*80}")
        logger.info(f"🎯 AUTO-TUNING @ Episode {episode}")
        logger.info(f"   Trend: {self.state.performance_trend.upper()}")
        logger.info(f"   Episodes since improvement: {self.state.episodes_since_improvement}")

        for key, value in adjustments.items():
            if key.endswith('_boost') or key.endswith('_rate'):
                continue  # Skip flags
            logger.info(f"   Adjusted {key}: {value}")

        logger.info("="*80 + "\n")

    def get_summary(self) -> Dict:
        """Get summary of tuning history"""
        return {
            'total_adjustments': len(self.state.adjustments_made),
            'performance_trend': self.state.performance_trend,
            'episodes_since_improvement': self.state.episodes_since_improvement,
            'best_performance': self.state.best_performance,
            'recent_avg_return': np.mean(list(self.state.recent_returns)) if self.state.recent_returns else 0.0,
            'recent_avg_sharpe': np.mean(list(self.state.recent_sharpes)) if self.state.recent_sharpes else 0.0,
            'adjustments_history': self.state.adjustments_made
        }

    def should_stop_early(self, metrics: TuningMetrics) -> bool:
        """
        Determine if training should stop early

        Args:
            metrics: Current training metrics

        Returns:
            True if training should stop
        """
        # Stop if declining for 500+ episodes
        if (self.state.performance_trend == "declining" and
            self.state.episodes_since_improvement > 500):
            logger.warning(f"⛔ Early stop: Declining for {self.state.episodes_since_improvement} episodes")
            return True

        # Stop if plateau for 1000+ episodes (even with interventions)
        if (self.state.performance_trend == "plateau" and
            self.state.episodes_since_improvement > 1000):
            logger.warning(f"⛔ Early stop: Plateau for {self.state.episodes_since_improvement} episodes")
            return True

        return False


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create tuner
    tuner = AutoHyperparameterTuner(verbose=True)

    # Simulate training episodes
    print("Simulating auto-tuning during training...\n")

    for episode in range(500):
        # Simulate metrics (in real training, these come from the agent)
        metrics = TuningMetrics(
            episode=episode,
            avg_return=np.random.randn() * 2 + episode * 0.01,  # Slowly improving
            avg_sharpe=np.random.randn() * 0.5,
            win_rate=0.5 + np.random.randn() * 0.05,
            epsilon=max(0.01, 1.0 - episode * 0.001),
            learning_rate=0.0003,
            avg_loss=0.1 / (episode + 1),
            q_value_mean=np.random.randn() * 10,
            q_value_std=5.0
        )

        # Get tuning adjustments
        adjustments = tuner.update(metrics)

        # Apply adjustments (in real training, modify agent config here)
        if 'epsilon' in adjustments:
            metrics.epsilon = adjustments['epsilon']
        if 'learning_rate' in adjustments:
            metrics.learning_rate = adjustments['learning_rate']

    # Print summary
    summary = tuner.get_summary()
    print("\n" + "="*80)
    print("TUNING SUMMARY")
    print("="*80)
    print(f"Total adjustments made: {summary['total_adjustments']}")
    print(f"Final trend: {summary['performance_trend']}")
    print(f"Best performance: {summary['best_performance']:.2f}%")
    print(f"Recent avg return: {summary['recent_avg_return']:.2f}%")
    print("="*80)
