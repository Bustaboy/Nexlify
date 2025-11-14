#!/usr/bin/env python3
"""
Advanced Epsilon Decay Manager
Supports multiple decay strategies for exploration-exploitation trade-off in RL
"""

import numpy as np
import logging
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EpsilonDecayStrategy(ABC):
    """Base class for epsilon decay strategies"""

    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.05):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.current_epsilon = epsilon_start
        self.current_step = 0

        # Monitoring and logging
        self.epsilon_history = []
        self.thresholds_crossed = set()
        self.key_thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]

        logger.info(f"🎯 Initialized {self.__class__.__name__} (start={epsilon_start:.2f}, end={epsilon_end:.2f})")

    @abstractmethod
    def get_epsilon(self, step: int) -> float:
        """
        Calculate epsilon value for given step

        Args:
            step: Current training step

        Returns:
            Epsilon value (clamped to [epsilon_end, epsilon_start])
        """
        pass

    def step(self) -> float:
        """
        Update and return epsilon for next step

        Returns:
            Next epsilon value (after decay)
        """
        # Increment step first, then calculate epsilon
        self.current_step += 1
        self.current_epsilon = self.get_epsilon(self.current_step)

        self.epsilon_history.append({
            'step': self.current_step,
            'epsilon': self.current_epsilon
        })

        # Check and log threshold crossings
        self._check_thresholds()

        return self.current_epsilon

    def _check_thresholds(self):
        """Check if epsilon crossed any key thresholds"""
        for threshold in self.key_thresholds:
            if threshold not in self.thresholds_crossed and self.current_epsilon <= threshold:
                self.thresholds_crossed.add(threshold)
                logger.info(f"🎯 Epsilon crossed threshold {threshold:.1f} at step {self.current_step} (ε={self.current_epsilon:.4f})")

    def reset(self):
        """Reset to initial state"""
        self.current_epsilon = self.epsilon_start
        self.current_step = 0
        self.epsilon_history = []
        self.thresholds_crossed = set()

    def save_history(self, filepath: str):
        """Save epsilon history to JSON file"""
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump({
                    'strategy': self.__class__.__name__,
                    'config': {
                        'epsilon_start': self.epsilon_start,
                        'epsilon_end': self.epsilon_end
                    },
                    'history': self.epsilon_history,
                    'thresholds_crossed': list(self.thresholds_crossed)
                }, f, indent=2)
            logger.info(f"✅ Epsilon history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save epsilon history: {e}")


class LinearEpsilonDecay(EpsilonDecayStrategy):
    """
    Linear epsilon decay from start to end over specified steps

    Formula: epsilon = start - (start - end) * (current_step / total_steps)

    Example:
        - Start: 1.0, End: 0.05, Steps: 2000
        - Step 0: ε = 1.0
        - Step 100: ε ≈ 0.95
        - Step 500: ε ≈ 0.76
        - Step 1000: ε ≈ 0.525
        - Step 2000: ε = 0.05
    """

    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 decay_steps: int = 2000):
        super().__init__(epsilon_start, epsilon_end)
        self.decay_steps = decay_steps
        logger.info(f"📉 Linear decay over {decay_steps} steps")

    def get_epsilon(self, step: int) -> float:
        """
        Calculate epsilon using linear decay

        Args:
            step: Current training step

        Returns:
            Epsilon value (clamped to epsilon_end)
        """
        if step >= self.decay_steps:
            return self.epsilon_end

        # Linear interpolation
        progress = step / self.decay_steps
        epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress

        # Ensure epsilon never goes below epsilon_end
        return max(epsilon, self.epsilon_end)


class ScheduledEpsilonDecay(EpsilonDecayStrategy):
    """
    Scheduled epsilon decay with custom schedules and interpolation

    Supports defining specific epsilon values at specific steps
    and interpolates between them.

    Example:
        schedule = {0: 1.0, 300: 0.7, 1000: 0.3, 2000: 0.05}
        - Step 0-300: Linear decay from 1.0 to 0.7
        - Step 300-1000: Linear decay from 0.7 to 0.3
        - Step 1000-2000: Linear decay from 0.3 to 0.05
        - Step 2000+: Constant at 0.05
    """

    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 schedule: Optional[Dict[int, float]] = None):
        super().__init__(epsilon_start, epsilon_end)

        # Default schedule if none provided
        if schedule is None:
            schedule = {
                0: 1.0,
                300: 0.7,
                1000: 0.3,
                2000: 0.05
            }

        # Ensure schedule starts at 0 and ends with epsilon_end
        if 0 not in schedule:
            schedule[0] = epsilon_start

        # Sort schedule by step
        self.schedule = sorted(schedule.items())
        logger.info(f"📅 Scheduled decay with {len(self.schedule)} milestones: {dict(self.schedule)}")

    def get_epsilon(self, step: int) -> float:
        """
        Calculate epsilon using scheduled decay with interpolation

        Args:
            step: Current training step

        Returns:
            Interpolated epsilon value
        """
        # Find the two schedule points to interpolate between
        for i in range(len(self.schedule) - 1):
            step1, eps1 = self.schedule[i]
            step2, eps2 = self.schedule[i + 1]

            if step1 <= step < step2:
                # Linear interpolation between two points
                progress = (step - step1) / (step2 - step1)
                epsilon = eps1 - (eps1 - eps2) * progress
                return max(epsilon, self.epsilon_end)

        # If beyond last scheduled step, return last epsilon value
        _, last_epsilon = self.schedule[-1]
        return max(last_epsilon, self.epsilon_end)


class ExponentialEpsilonDecay(EpsilonDecayStrategy):
    """
    Exponential epsilon decay with configurable decay rate

    Formula: epsilon = max(epsilon_end, epsilon_start * decay_rate^step)

    Better defaults than simple multiplicative decay (0.995).
    Uses decay_steps to automatically calculate appropriate decay_rate.

    Example:
        - Start: 1.0, End: 0.05, Steps: 2000
        - Decay rate automatically calculated to reach 0.05 at step 2000
        - Step 0: ε = 1.0
        - Step 100: ε ≈ 0.85
        - Step 500: ε ≈ 0.51
        - Step 1000: ε ≈ 0.22
        - Step 2000: ε = 0.05
    """

    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 decay_steps: int = 2000, decay_rate: Optional[float] = None):
        super().__init__(epsilon_start, epsilon_end)

        if decay_rate is None:
            # Auto-calculate decay rate to reach epsilon_end at decay_steps
            # epsilon_end = epsilon_start * decay_rate^decay_steps
            # decay_rate = (epsilon_end / epsilon_start)^(1/decay_steps)
            self.decay_rate = (epsilon_end / epsilon_start) ** (1 / decay_steps)
        else:
            self.decay_rate = decay_rate

        self.decay_steps = decay_steps
        logger.info(f"📊 Exponential decay with rate {self.decay_rate:.6f} over {decay_steps} steps")

    def get_epsilon(self, step: int) -> float:
        """
        Calculate epsilon using exponential decay

        Args:
            step: Current training step

        Returns:
            Epsilon value (clamped to epsilon_end)
        """
        epsilon = self.epsilon_start * (self.decay_rate ** step)
        return max(epsilon, self.epsilon_end)


class EpsilonDecayFactory:
    """
    Factory for creating epsilon decay strategies

    Usage:
        strategy = EpsilonDecayFactory.create('linear', decay_steps=2000)
        epsilon = strategy.step()
    """

    STRATEGIES = {
        'linear': LinearEpsilonDecay,
        'scheduled': ScheduledEpsilonDecay,
        'exponential': ExponentialEpsilonDecay
    }

    @staticmethod
    def create(strategy_type: str = 'linear', **kwargs) -> EpsilonDecayStrategy:
        """
        Create epsilon decay strategy based on type

        Args:
            strategy_type: Type of strategy ('linear', 'scheduled', 'exponential')
            **kwargs: Strategy-specific parameters

        Returns:
            EpsilonDecayStrategy instance

        Raises:
            ValueError: If strategy_type is not recognized
        """
        strategy_type = strategy_type.lower()

        if strategy_type not in EpsilonDecayFactory.STRATEGIES:
            available = ', '.join(EpsilonDecayFactory.STRATEGIES.keys())
            raise ValueError(
                f"Unknown strategy type '{strategy_type}'. "
                f"Available: {available}"
            )

        strategy_class = EpsilonDecayFactory.STRATEGIES[strategy_type]

        # Default to LinearEpsilonDecay for trading
        logger.info(f"🏭 Creating {strategy_type} epsilon decay strategy")

        return strategy_class(**kwargs)

    @staticmethod
    def create_from_config(config: Dict) -> EpsilonDecayStrategy:
        """
        Create epsilon decay strategy from configuration dictionary

        Args:
            config: Configuration dictionary with keys:
                - epsilon_decay_type: Strategy type
                - epsilon_start: Starting epsilon value
                - epsilon_end: Ending epsilon value
                - epsilon_decay_steps: Number of decay steps
                - Other strategy-specific parameters

        Returns:
            EpsilonDecayStrategy instance
        """
        strategy_type = config.get('epsilon_decay_type', 'linear')

        # Extract common parameters
        params = {
            'epsilon_start': config.get('epsilon_start', 1.0),
            'epsilon_end': config.get('epsilon_end', 0.05),
        }

        # Add decay_steps if present
        if 'epsilon_decay_steps' in config:
            params['decay_steps'] = config['epsilon_decay_steps']

        # Add schedule for scheduled decay
        if strategy_type == 'scheduled' and 'epsilon_schedule' in config:
            params['schedule'] = config['epsilon_schedule']

        # Add decay_rate for exponential decay
        if strategy_type == 'exponential' and 'epsilon_decay_rate' in config:
            params['decay_rate'] = config['epsilon_decay_rate']

        return EpsilonDecayFactory.create(strategy_type, **params)


# Export main classes
__all__ = [
    'EpsilonDecayStrategy',
    'LinearEpsilonDecay',
    'ScheduledEpsilonDecay',
    'ExponentialEpsilonDecay',
    'EpsilonDecayFactory'
]
