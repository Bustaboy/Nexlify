#!/usr/bin/env python3
"""
State Normalization for RL Trading Agents

Provides online normalization of state vectors using running statistics.
Essential for stable neural network training.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class StateNormalizer:
    """
    Online state normalization using running statistics

    Features:
    - Feature-wise mean and standard deviation tracking
    - Online updates (Welford's algorithm)
    - Outlier clipping
    - Save/load normalization parameters
    - Warm-up period before normalization
    """

    def __init__(
        self,
        state_size: int,
        clip_range: float = 3.0,
        warmup_samples: int = 100,
        epsilon: float = 1e-8,
        use_running_stats: bool = True
    ):
        """
        Initialize state normalizer

        Args:
            state_size: Size of state vector
            clip_range: Clip normalized values to [-clip_range, +clip_range]
            warmup_samples: Number of samples before applying normalization
            epsilon: Small value to prevent division by zero
            use_running_stats: Use running statistics (True) or batch stats (False)
        """
        self.state_size = state_size
        self.clip_range = clip_range
        self.warmup_samples = warmup_samples
        self.epsilon = epsilon
        self.use_running_stats = use_running_stats

        # Running statistics (Welford's algorithm)
        self.count = 0
        self.mean = np.zeros(state_size, dtype=np.float64)
        self.m2 = np.zeros(state_size, dtype=np.float64)  # Sum of squared differences

        logger.debug(f"StateNormalizer initialized: state_size={state_size}, "
                    f"clip_range={clip_range}, warmup={warmup_samples}")

    def update(self, state: np.ndarray) -> None:
        """
        Update running statistics with new state

        Uses Welford's online algorithm for numerical stability

        Args:
            state: State vector to add to statistics
        """
        if state.shape[0] != self.state_size:
            raise ValueError(
                f"State size mismatch: expected {self.state_size}, got {state.shape[0]}"
            )

        # Convert to float64 for precision
        state = state.astype(np.float64)

        self.count += 1

        # Welford's algorithm
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.m2 += delta * delta2

    def normalize(self, state: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Normalize state using running statistics

        Args:
            state: State vector to normalize
            update_stats: Whether to update statistics with this state

        Returns:
            Normalized state vector
        """
        if update_stats:
            self.update(state)

        # Don't normalize during warmup period
        if self.count < self.warmup_samples:
            return state.astype(np.float32)

        # Calculate standard deviation
        variance = self.m2 / self.count
        std = np.sqrt(variance + self.epsilon)

        # Normalize: (x - mean) / std
        normalized = (state - self.mean) / std

        # Clip outliers
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized.astype(np.float32)

    def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Convert normalized state back to original scale

        Args:
            normalized_state: Normalized state vector

        Returns:
            Denormalized state vector
        """
        if self.count < self.warmup_samples:
            return normalized_state

        variance = self.m2 / self.count
        std = np.sqrt(variance + self.epsilon)

        # Denormalize: x_original = (x_normalized * std) + mean
        denormalized = (normalized_state * std) + self.mean

        return denormalized.astype(np.float32)

    def get_statistics(self) -> Dict[str, np.ndarray]:
        """
        Get current normalization statistics

        Returns:
            Dictionary with mean, std, count
        """
        variance = self.m2 / max(self.count, 1)
        std = np.sqrt(variance + self.epsilon)

        return {
            'mean': self.mean.copy(),
            'std': std.copy(),
            'count': self.count
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save normalization parameters to file

        Args:
            filepath: Path to save file (JSON format)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        statistics = {
            'state_size': self.state_size,
            'clip_range': self.clip_range,
            'warmup_samples': self.warmup_samples,
            'epsilon': self.epsilon,
            'count': int(self.count),
            'mean': self.mean.tolist(),
            'm2': self.m2.tolist()
        }

        with open(filepath, 'w') as f:
            json.dump(statistics, f, indent=2)

        logger.info(f"Normalization parameters saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load normalization parameters from file

        Args:
            filepath: Path to load file (JSON format)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Normalization file not found: {filepath}")

        with open(filepath, 'r') as f:
            statistics = json.load(f)

        # Validate state size
        if statistics['state_size'] != self.state_size:
            raise ValueError(
                f"State size mismatch: file has {statistics['state_size']}, "
                f"expected {self.state_size}"
            )

        # Load parameters
        self.clip_range = statistics['clip_range']
        self.warmup_samples = statistics['warmup_samples']
        self.epsilon = statistics['epsilon']
        self.count = statistics['count']
        self.mean = np.array(statistics['mean'], dtype=np.float64)
        self.m2 = np.array(statistics['m2'], dtype=np.float64)

        logger.info(f"Normalization parameters loaded from {filepath} "
                   f"(count={self.count})")

    def reset(self) -> None:
        """Reset normalization statistics"""
        self.count = 0
        self.mean = np.zeros(self.state_size, dtype=np.float64)
        self.m2 = np.zeros(self.state_size, dtype=np.float64)

        logger.debug("Normalization statistics reset")

    @property
    def is_warmed_up(self) -> bool:
        """Check if normalizer has enough samples"""
        return self.count >= self.warmup_samples

    @property
    def std(self) -> np.ndarray:
        """Get current standard deviation"""
        variance = self.m2 / max(self.count, 1)
        return np.sqrt(variance + self.epsilon)


class BatchNormalizer:
    """
    Batch normalization for offline training

    Simpler than running normalization, computes stats from full dataset
    """

    def __init__(
        self,
        clip_range: float = 3.0,
        epsilon: float = 1e-8
    ):
        """
        Initialize batch normalizer

        Args:
            clip_range: Clip normalized values to [-clip_range, +clip_range]
            epsilon: Small value to prevent division by zero
        """
        self.clip_range = clip_range
        self.epsilon = epsilon

        self.mean = None
        self.std = None

        logger.debug(f"BatchNormalizer initialized: clip_range={clip_range}")

    def fit(self, states: np.ndarray) -> None:
        """
        Fit normalizer to batch of states

        Args:
            states: Array of states (n_samples, state_size)
        """
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0) + self.epsilon

        logger.debug(f"BatchNormalizer fitted on {len(states)} samples")

    def normalize(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize states

        Args:
            states: States to normalize (can be single state or batch)

        Returns:
            Normalized states
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        # Handle single state or batch
        is_single = states.ndim == 1
        if is_single:
            states = states.reshape(1, -1)

        normalized = (states - self.mean) / self.std
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        if is_single:
            normalized = normalized.flatten()

        return normalized.astype(np.float32)

    def denormalize(self, normalized_states: np.ndarray) -> np.ndarray:
        """
        Denormalize states

        Args:
            normalized_states: Normalized states

        Returns:
            Original scale states
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        # Handle single state or batch
        is_single = normalized_states.ndim == 1
        if is_single:
            normalized_states = normalized_states.reshape(1, -1)

        denormalized = (normalized_states * self.std) + self.mean

        if is_single:
            denormalized = denormalized.flatten()

        return denormalized.astype(np.float32)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save normalization parameters"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        statistics = {
            'clip_range': self.clip_range,
            'epsilon': self.epsilon,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None
        }

        with open(filepath, 'w') as f:
            json.dump(statistics, f, indent=2)

        logger.info(f"Batch normalization parameters saved to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """Load normalization parameters"""
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Normalization file not found: {filepath}")

        with open(filepath, 'r') as f:
            statistics = json.load(f)

        self.clip_range = statistics['clip_range']
        self.epsilon = statistics['epsilon']
        self.mean = np.array(statistics['mean']) if statistics['mean'] else None
        self.std = np.array(statistics['std']) if statistics['std'] else None

        logger.info(f"Batch normalization parameters loaded from {filepath}")


__all__ = ['StateNormalizer', 'BatchNormalizer']
