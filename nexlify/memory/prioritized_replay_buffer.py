#!/usr/bin/env python3
"""
Prioritized Experience Replay Buffer
Implements PER with importance sampling for DQN agents
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from nexlify.memory.sumtree import SumTree

logger = logging.getLogger(__name__)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) Buffer

    Samples experiences based on their TD error, allowing the agent
    to learn more from important experiences.

    Features:
        - SumTree for O(log n) sampling
        - Importance sampling (IS) weights to correct for bias
        - Beta annealing for IS weight correction
        - Priority clipping to prevent extreme values
        - Configurable alpha (prioritization strength)

    Args:
        capacity: Maximum buffer size
        alpha: Prioritization exponent (0=uniform, 1=full prioritization)
        beta_start: Initial IS correction (0=no correction, 1=full correction)
        beta_end: Final IS correction (typically 1.0)
        beta_annealing_steps: Steps to anneal beta from start to end
        epsilon: Small constant to prevent zero priorities
        priority_clip: Maximum priority value (None=no clipping)

    Example:
        >>> buffer = PrioritizedReplayBuffer(capacity=100000)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
        >>> buffer.update_priorities(indices, td_errors)
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_annealing_steps: int = 10000,
        epsilon: float = 1e-6,
        priority_clip: Optional[float] = None,
    ):
        """Initialize prioritized replay buffer"""
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if not 0 <= beta_start <= 1:
            raise ValueError(f"Beta start must be in [0, 1], got {beta_start}")
        if not 0 <= beta_end <= 1:
            raise ValueError(f"Beta end must be in [0, 1], got {beta_end}")
        if beta_annealing_steps <= 0:
            raise ValueError(
                f"Beta annealing steps must be positive, got {beta_annealing_steps}"
            )
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_annealing_steps = beta_annealing_steps
        self.epsilon = epsilon
        self.priority_clip = priority_clip

        # Initialize SumTree
        self.tree = SumTree(capacity)

        # Track sampling step for beta annealing
        self.sample_count = 0

        # Statistics
        self.stats = {
            "total_samples": 0,
            "priority_updates": 0,
            "mean_priority": 0.0,
            "max_priority": 1.0,
            "min_priority": epsilon,
        }

        # For backward compatibility with standard ReplayBuffer
        self.max_size = capacity
        self.buffer = None  # Not used, but some code may check for it

        logger.info(
            f"PrioritizedReplayBuffer initialized: "
            f"capacity={capacity}, alpha={alpha}, "
            f"beta={beta_start}→{beta_end} over {beta_annealing_steps} steps"
        )

    def _get_beta(self) -> float:
        """
        Get current beta value with annealing

        Beta increases linearly from beta_start to beta_end over
        beta_annealing_steps.

        Returns:
            Current beta value
        """
        progress = min(1.0, self.sample_count / self.beta_annealing_steps)
        beta = self.beta_start + progress * (self.beta_end - self.beta_start)
        return beta

    def _get_priority(self, td_error: float) -> float:
        """
        Calculate priority from TD error

        Priority = (|TD_error| + epsilon)^alpha

        Args:
            td_error: Temporal difference error

        Returns:
            Priority value
        """
        priority = (abs(td_error) + self.epsilon) ** self.alpha

        # Apply clipping if configured
        if self.priority_clip is not None:
            priority = min(priority, self.priority_clip)

        return priority

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,
    ):
        """
        Add experience to buffer with priority

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            td_error: Optional TD error (if None, uses max priority)
        """
        # Create experience tuple
        experience = (state, action, reward, next_state, done)

        # Calculate priority
        if td_error is not None:
            priority = self._get_priority(td_error)
        else:
            # Use max priority for new experiences
            priority = self.tree.get_max_priority()

        # Add to tree
        self.tree.add(priority, experience)

        # Update statistics
        self._update_stats(priority)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Alias for push() for backward compatibility

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample batch of experiences with importance sampling weights

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, indices, weights):
                - experiences: List of (state, action, reward, next_state, done)
                - indices: Tree indices for priority updates
                - weights: Importance sampling weights

        Raises:
            ValueError: If buffer has fewer experiences than batch_size
        """
        if len(self.tree) < batch_size:
            raise ValueError(
                f"Not enough experiences: {len(self.tree)} < {batch_size}"
            )

        # Increment sample count for beta annealing
        self.sample_count += 1
        beta = self._get_beta()

        # Calculate sampling segments
        segment_size = self.tree.total() / batch_size

        experiences = []
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Sample from each segment
        for i in range(batch_size):
            # Sample uniformly within segment
            a = segment_size * i
            b = segment_size * (i + 1)
            value = np.random.uniform(a, b)

            # Retrieve experience
            idx, priority, experience = self.tree.sample(value)

            experiences.append(experience)
            indices[i] = idx
            priorities[i] = priority

        # Calculate importance sampling weights
        # P(i) = p_i^alpha / sum(p_k^alpha)
        # w_i = (N * P(i))^(-beta) / max_w
        sampling_probabilities = priorities / self.tree.total()
        weights = (len(self.tree) * sampling_probabilities) ** (-beta)

        # Normalize by max weight for stability
        weights = weights / weights.max()

        # Update statistics
        self.stats["total_samples"] += batch_size

        return experiences, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities for sampled experiences

        Args:
            indices: Tree indices from sample()
            td_errors: New TD errors for these experiences

        Raises:
            ValueError: If indices and td_errors have different lengths
        """
        if len(indices) != len(td_errors):
            raise ValueError(
                f"Indices and TD errors must have same length: "
                f"{len(indices)} != {len(td_errors)}"
            )

        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(int(idx), priority)
            self._update_stats(priority)

        self.stats["priority_updates"] += len(indices)

    def _update_stats(self, priority: float):
        """
        Update running statistics

        Args:
            priority: New priority value
        """
        # Update max/min
        self.stats["max_priority"] = max(self.stats["max_priority"], priority)
        self.stats["min_priority"] = min(self.stats["min_priority"], priority)

        # Update mean (exponential moving average)
        alpha = 0.01  # Smoothing factor
        self.stats["mean_priority"] = (
            alpha * priority + (1 - alpha) * self.stats["mean_priority"]
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "size": len(self.tree),
            "capacity": self.capacity,
            "beta": self._get_beta(),
            "alpha": self.alpha,
            "total_priority": self.tree.total(),
        }

    def __len__(self) -> int:
        """
        Get number of experiences in buffer

        Returns:
            Number of stored experiences
        """
        return len(self.tree)

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer("
            f"size={len(self.tree)}/{self.capacity}, "
            f"alpha={self.alpha}, "
            f"beta={self._get_beta():.3f}, "
            f"mean_priority={self.stats['mean_priority']:.4f})"
        )


# Export main class
__all__ = ["PrioritizedReplayBuffer"]
