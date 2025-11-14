#!/usr/bin/env python3
"""
N-Step Replay Buffer for Improved Temporal Credit Assignment
Implements n-step returns to better propagate rewards through time
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NStepReturnCalculator:
    """
    Utility class for calculating n-step returns

    N-step return formula:
        R_t^(n) = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n max_a Q(s_{t+n}, a)

    For terminal states (episode ends), we truncate at the terminal state:
        R_t^(n) = r_t + γr_{t+1} + ... + γ^{k-1}r_{t+k} (where k < n and s_{t+k} is terminal)
    """

    def __init__(self, n_step: int = 5, gamma: float = 0.99):
        """
        Initialize n-step return calculator

        Args:
            n_step: Number of steps to look ahead
            gamma: Discount factor

        Raises:
            ValueError: If n_step < 1 or gamma not in [0, 1]
        """
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self.n_step = n_step
        self.gamma = gamma

        # Precompute gamma powers for efficiency
        self._gamma_powers = np.array([gamma ** i for i in range(n_step)])

    def calculate(
        self,
        rewards: List[float],
        next_q_value: float,
        dones: List[bool]
    ) -> Tuple[float, int]:
        """
        Calculate n-step return from a sequence of rewards

        Args:
            rewards: List of rewards [r_t, r_{t+1}, ..., r_{t+n-1}]
            next_q_value: Q(s_{t+n}, a*) - estimated value at n steps ahead
            dones: List of done flags for each step

        Returns:
            Tuple of (n_step_return, actual_steps_used)
                - n_step_return: Calculated return
                - actual_steps_used: Number of steps actually used (may be < n if episode ended)
        """
        n_steps = len(rewards)
        if n_steps == 0:
            return 0.0, 0

        # Find if episode ended early
        actual_steps = n_steps
        for i, done in enumerate(dones):
            if done:
                actual_steps = i + 1
                break

        # Calculate discounted sum of rewards
        # R = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(n-1)*r_{n-1}
        discounted_return = 0.0
        for i in range(actual_steps):
            discounted_return += self._gamma_powers[i] * rewards[i]

        # Add bootstrapped value if episode didn't end
        if actual_steps == n_steps and not dones[-1]:
            discounted_return += (self.gamma ** actual_steps) * next_q_value

        return discounted_return, actual_steps


class NStepReplayBuffer:
    """
    N-Step Experience Replay Buffer

    Stores sequences of transitions and calculates n-step returns for better
    temporal credit assignment. Particularly useful for trading where actions
    have delayed consequences.

    Features:
        - Efficient circular buffer for storing n-step sequences
        - Proper handling of episode boundaries
        - Incremental return calculation
        - Optional mixing with 1-step returns
        - Compatible with Prioritized Experience Replay

    Args:
        capacity: Maximum buffer size
        n_step: Number of steps to look ahead (default: 5)
        gamma: Discount factor (default: 0.99)
        use_mixed_returns: Mix 1-step and n-step returns (default: False)
        mixed_returns_ratio: Ratio of n-step samples when mixing (default: 0.5)

    Example:
        >>> buffer = NStepReplayBuffer(capacity=100000, n_step=5)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> batch = buffer.sample(batch_size=32)
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 5,
        gamma: float = 0.99,
        use_mixed_returns: bool = False,
        mixed_returns_ratio: float = 0.5,
    ):
        """Initialize n-step replay buffer"""
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}")
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0 <= mixed_returns_ratio <= 1:
            raise ValueError(f"mixed_returns_ratio must be in [0, 1], got {mixed_returns_ratio}")

        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.use_mixed_returns = use_mixed_returns
        self.mixed_returns_ratio = mixed_returns_ratio

        # Main storage for n-step transitions
        # Each entry: (state, action, n_step_return, next_state_n, done, actual_steps)
        self.buffer = deque(maxlen=capacity)

        # Temporary buffer for accumulating n-step sequences
        # Stores recent transitions until we can calculate n-step return
        self.n_step_buffer = deque(maxlen=n_step)

        # Optional 1-step buffer for mixed sampling
        if use_mixed_returns:
            self.one_step_buffer = deque(maxlen=capacity)
        else:
            self.one_step_buffer = None

        # N-step return calculator
        self.return_calculator = NStepReturnCalculator(n_step, gamma)

        # For backward compatibility
        self.max_size = capacity

        # Statistics
        self.stats = {
            "total_pushes": 0,
            "n_step_samples": 0,
            "one_step_samples": 0,
            "avg_actual_steps": 0.0,
            "episode_truncations": 0,
        }

        logger.info(
            f"NStepReplayBuffer initialized: "
            f"capacity={capacity}, n_step={n_step}, gamma={gamma}, "
            f"mixed={use_mixed_returns}"
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        **kwargs
    ):
        """
        Add transition to buffer and calculate n-step return

        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            done: Whether episode ended
            **kwargs: Additional arguments (for compatibility with PER)
        """
        # Add to n-step accumulation buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Also add to 1-step buffer if using mixed returns
        if self.one_step_buffer is not None:
            self.one_step_buffer.append((state, action, reward, next_state, done))

        # Calculate and store n-step return if we have enough transitions
        if len(self.n_step_buffer) >= self.n_step or done:
            self._process_n_step_sequence()

        # If episode ended, flush remaining transitions
        if done:
            self._flush_n_step_buffer()

        self.stats["total_pushes"] += 1

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Alias for push() for backward compatibility"""
        self.push(state, action, reward, next_state, done)

    def _process_n_step_sequence(self):
        """
        Process the oldest transition in n_step_buffer with n-step return
        """
        if len(self.n_step_buffer) == 0:
            return

        # Get the oldest transition (the one we'll calculate return for)
        state_0, action_0, _, _, _ = self.n_step_buffer[0]

        # Collect rewards and dones from all steps
        rewards = [t[2] for t in self.n_step_buffer]
        dones = [t[4] for t in self.n_step_buffer]

        # Get the final next_state (n steps ahead)
        next_state_n = self.n_step_buffer[-1][3]

        # Estimate Q-value at next_state_n (will be replaced during training)
        # For now, use 0 (will be updated with actual Q-values during sampling)
        next_q_value = 0.0

        # Calculate n-step return
        n_step_return, actual_steps = self.return_calculator.calculate(
            rewards=rewards,
            next_q_value=next_q_value,
            dones=dones
        )

        # Check if episode ended early
        episode_ended = any(dones)
        if episode_ended and actual_steps < len(dones):
            self.stats["episode_truncations"] += 1

        # Store n-step transition
        # Format: (state, action, n_step_return, next_state_n, done, actual_steps)
        self.buffer.append((
            state_0,
            action_0,
            n_step_return,
            next_state_n,
            dones[-1],  # Use final done flag
            actual_steps
        ))

        # Update stats
        alpha = 0.01
        self.stats["avg_actual_steps"] = (
            alpha * actual_steps + (1 - alpha) * self.stats["avg_actual_steps"]
        )

    def _flush_n_step_buffer(self):
        """
        Flush all remaining transitions in n_step_buffer when episode ends
        """
        while len(self.n_step_buffer) > 1:
            self._process_n_step_sequence()
            self.n_step_buffer.popleft()

        # Clear the buffer
        self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch of n-step experiences

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of (state, action, n_step_return, next_state_n, done) tuples

        Raises:
            ValueError: If buffer has fewer experiences than batch_size
        """
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough experiences: {len(self.buffer)} < {batch_size}"
            )

        # Decide sampling strategy
        if self.use_mixed_returns and self.one_step_buffer is not None:
            # Sample from both buffers
            n_step_count = int(batch_size * self.mixed_returns_ratio)
            one_step_count = batch_size - n_step_count

            # Sample n-step transitions
            n_step_samples = self._sample_from_buffer(self.buffer, n_step_count)

            # Sample 1-step transitions
            one_step_samples = self._sample_from_buffer(
                self.one_step_buffer, one_step_count
            )

            # Convert 1-step to compatible format (add actual_steps=1)
            one_step_formatted = [
                (s, a, r, ns, d, 1) for s, a, r, ns, d in one_step_samples
            ]

            # Combine
            experiences = n_step_samples + one_step_formatted

            # Update stats
            self.stats["n_step_samples"] += n_step_count
            self.stats["one_step_samples"] += one_step_count
        else:
            # Sample only n-step transitions
            experiences = self._sample_from_buffer(self.buffer, batch_size)
            self.stats["n_step_samples"] += batch_size

        return experiences

    def _sample_from_buffer(
        self, buffer: deque, count: int
    ) -> List[Tuple]:
        """
        Sample uniformly from a buffer

        Args:
            buffer: Buffer to sample from
            count: Number of samples

        Returns:
            List of sampled experiences
        """
        if count == 0:
            return []

        # Uniform sampling
        indices = np.random.choice(len(buffer), size=count, replace=False)
        return [buffer[i] for i in indices]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get buffer statistics

        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "size": len(self.buffer),
            "capacity": self.capacity,
            "n_step": self.n_step,
            "gamma": self.gamma,
            "pending_transitions": len(self.n_step_buffer),
            "one_step_size": len(self.one_step_buffer) if self.one_step_buffer else 0,
        }

    def __len__(self) -> int:
        """
        Get number of n-step experiences in buffer

        Returns:
            Number of stored n-step experiences
        """
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            f"NStepReplayBuffer("
            f"size={len(self.buffer)}/{self.capacity}, "
            f"n_step={self.n_step}, "
            f"gamma={self.gamma}, "
            f"avg_steps={self.stats['avg_actual_steps']:.2f})"
        )


class MixedNStepReplayBuffer:
    """
    Mixed N-Step Replay Buffer that combines 1-step and n-step returns

    This buffer maintains both 1-step and n-step transitions and samples
    from both with a configurable ratio. This can help balance the
    bias-variance tradeoff:
    - 1-step: High bias, low variance
    - n-step: Lower bias, higher variance
    - Mixed: Balanced approach

    Args:
        capacity: Maximum buffer size
        n_step: Number of steps to look ahead
        gamma: Discount factor
        n_step_ratio: Ratio of n-step samples (0.0-1.0)

    Example:
        >>> buffer = MixedNStepReplayBuffer(capacity=100000, n_step=5, n_step_ratio=0.7)
        >>> # 70% n-step, 30% 1-step samples
    """

    def __init__(
        self,
        capacity: int = 100000,
        n_step: int = 5,
        gamma: float = 0.99,
        n_step_ratio: float = 0.5,
    ):
        """Initialize mixed n-step replay buffer"""
        self.buffer = NStepReplayBuffer(
            capacity=capacity,
            n_step=n_step,
            gamma=gamma,
            use_mixed_returns=True,
            mixed_returns_ratio=n_step_ratio,
        )

        # For backward compatibility
        self.capacity = capacity
        self.max_size = capacity
        self.n_step = n_step
        self.gamma = gamma

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.push(state, action, reward, next_state, done)

    def add(self, state, action, reward, next_state, done):
        """Alias for push()"""
        self.push(state, action, reward, next_state, done)

    def sample(self, batch_size: int):
        """Sample mixed batch"""
        return self.buffer.sample(batch_size)

    def get_stats(self):
        """Get statistics"""
        return self.buffer.get_stats()

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"Mixed{self.buffer.__repr__()}"


# Export classes
__all__ = [
    "NStepReturnCalculator",
    "NStepReplayBuffer",
    "MixedNStepReplayBuffer",
]
