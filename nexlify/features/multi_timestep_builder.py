#!/usr/bin/env python3
"""
Multi-Timestep State Builder for RL Trading Agents

Stacks multiple timesteps into a single state vector to capture temporal dynamics.
Essential for agents to understand market momentum and trends.
"""

import logging
from collections import deque
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class MultiTimestepStateBuilder:
    """
    Builds state vectors by stacking multiple timesteps

    Provides temporal context to RL agents by including
    recent history in the state representation.

    Example:
        With lookback=3 and state_size=10:
        - Single timestep state: (10,)
        - Multi-timestep state: (30,) = 3 timesteps * 10 features

    Benefits:
        - Captures momentum and trends
        - Removes need for recurrent networks (LSTM/GRU)
        - Simple and efficient
        - Works with standard feedforward networks
    """

    def __init__(
        self,
        state_size: int,
        lookback: int = 10,
        fill_value: float = 0.0
    ):
        """
        Initialize multi-timestep state builder

        Args:
            state_size: Size of single timestep state
            lookback: Number of timesteps to stack (default: 10)
            fill_value: Value to use for padding before enough history (default: 0.0)
        """
        self.state_size = state_size
        self.lookback = lookback
        self.fill_value = fill_value

        # Ring buffer for efficient history storage
        self.history = deque(maxlen=lookback)

        # Pre-allocated array for padding
        self.padding_state = np.full(state_size, fill_value, dtype=np.float32)

        logger.debug(f"MultiTimestepStateBuilder initialized: "
                    f"state_size={state_size}, lookback={lookback}")

    def add_state(self, state: np.ndarray) -> None:
        """
        Add new state to history

        Args:
            state: State vector to add
        """
        if state.shape[0] != self.state_size:
            raise ValueError(
                f"State size mismatch: expected {self.state_size}, got {state.shape[0]}"
            )

        # Store as copy to avoid reference issues
        self.history.append(state.copy())

    def build(self, include_current: bool = True) -> np.ndarray:
        """
        Build multi-timestep state by stacking history

        Args:
            include_current: Whether to include current state (default: True)

        Returns:
            Stacked state vector of shape (lookback * state_size,)
        """
        if len(self.history) == 0:
            # No history yet, return all padding
            return np.tile(self.padding_state, self.lookback)

        # Determine how many states to use
        num_states = len(self.history) if include_current else len(self.history) - 1
        num_states = min(num_states, self.lookback)

        if num_states <= 0:
            return np.tile(self.padding_state, self.lookback)

        # Get most recent states
        states_to_stack = list(self.history)[-num_states:]

        # Pad if needed (not enough history)
        num_padding = self.lookback - num_states
        if num_padding > 0:
            padding_states = [self.padding_state] * num_padding
            states_to_stack = padding_states + states_to_stack

        # Stack states (oldest to newest)
        stacked = np.concatenate(states_to_stack)

        return stacked.astype(np.float32)

    def get_current_state(self) -> Optional[np.ndarray]:
        """
        Get most recent state

        Returns:
            Current state or None if no history
        """
        if len(self.history) == 0:
            return None

        return self.history[-1].copy()

    def reset(self) -> None:
        """Reset history buffer"""
        self.history.clear()
        logger.debug("Multi-timestep history reset")

    @property
    def output_size(self) -> int:
        """Get size of output stacked state"""
        return self.state_size * self.lookback

    @property
    def is_full(self) -> bool:
        """Check if history buffer is full"""
        return len(self.history) >= self.lookback

    @property
    def history_length(self) -> int:
        """Get current history length"""
        return len(self.history)


class AdaptiveTimestepBuilder:
    """
    Adaptive multi-timestep builder with dynamic lookback

    Adjusts lookback period based on market conditions:
    - High volatility: Shorter lookback (more reactive)
    - Low volatility: Longer lookback (more context)
    """

    def __init__(
        self,
        state_size: int,
        min_lookback: int = 5,
        max_lookback: int = 20,
        volatility_threshold: float = 0.02
    ):
        """
        Initialize adaptive timestep builder

        Args:
            state_size: Size of single timestep state
            min_lookback: Minimum lookback period (default: 5)
            max_lookback: Maximum lookback period (default: 20)
            volatility_threshold: Volatility threshold for adaptation (default: 0.02)
        """
        self.state_size = state_size
        self.min_lookback = min_lookback
        self.max_lookback = max_lookback
        self.volatility_threshold = volatility_threshold

        # Use max lookback for history storage
        self.history = deque(maxlen=max_lookback)
        self.current_lookback = max_lookback

        # Track recent volatility
        self.recent_volatilities = deque(maxlen=20)

        logger.debug(f"AdaptiveTimestepBuilder initialized: "
                    f"lookback range=[{min_lookback}, {max_lookback}]")

    def add_state(self, state: np.ndarray, volatility: Optional[float] = None) -> None:
        """
        Add new state and update lookback based on volatility

        Args:
            state: State vector to add
            volatility: Current market volatility (optional)
        """
        self.history.append(state.copy())

        if volatility is not None:
            self.recent_volatilities.append(volatility)
            self._update_lookback()

    def _update_lookback(self) -> None:
        """Update lookback period based on recent volatility"""
        if len(self.recent_volatilities) < 5:
            return

        avg_volatility = np.mean(list(self.recent_volatilities))

        # High volatility: use shorter lookback (more reactive)
        # Low volatility: use longer lookback (more context)
        if avg_volatility > self.volatility_threshold:
            # High volatility
            target_lookback = self.min_lookback
        else:
            # Low volatility
            target_lookback = self.max_lookback

        # Smooth transition
        self.current_lookback = int(
            0.9 * self.current_lookback + 0.1 * target_lookback
        )

        # Clamp to valid range
        self.current_lookback = max(
            self.min_lookback,
            min(self.current_lookback, self.max_lookback)
        )

    def build(self) -> np.ndarray:
        """Build state using current adaptive lookback"""
        if len(self.history) == 0:
            padding_state = np.zeros(self.state_size, dtype=np.float32)
            return np.tile(padding_state, self.current_lookback)

        # Get most recent states according to current lookback
        num_states = min(len(self.history), self.current_lookback)
        states_to_stack = list(self.history)[-num_states:]

        # Pad if needed
        num_padding = self.current_lookback - num_states
        if num_padding > 0:
            padding_state = np.zeros(self.state_size, dtype=np.float32)
            padding_states = [padding_state] * num_padding
            states_to_stack = padding_states + states_to_stack

        stacked = np.concatenate(states_to_stack)
        return stacked.astype(np.float32)

    def reset(self) -> None:
        """Reset history and lookback"""
        self.history.clear()
        self.recent_volatilities.clear()
        self.current_lookback = self.max_lookback

    @property
    def output_size(self) -> int:
        """Get current output size (may vary)"""
        return self.state_size * self.current_lookback


class TemporalAttentionBuilder:
    """
    Builds state with temporal attention weights

    More recent states get higher weight in the final representation
    """

    def __init__(
        self,
        state_size: int,
        lookback: int = 10,
        decay_factor: float = 0.9
    ):
        """
        Initialize temporal attention builder

        Args:
            state_size: Size of single timestep state
            lookback: Number of timesteps to consider (default: 10)
            decay_factor: Weight decay for older states (default: 0.9)
        """
        self.state_size = state_size
        self.lookback = lookback
        self.decay_factor = decay_factor

        self.history = deque(maxlen=lookback)

        # Pre-compute attention weights (exponential decay)
        weights = np.array([decay_factor ** i for i in range(lookback)])
        weights = weights[::-1]  # Reverse so most recent has highest weight
        self.attention_weights = weights / weights.sum()  # Normalize

        logger.debug(f"TemporalAttentionBuilder initialized: "
                    f"lookback={lookback}, decay={decay_factor}")

    def add_state(self, state: np.ndarray) -> None:
        """Add new state to history"""
        self.history.append(state.copy())

    def build(self) -> np.ndarray:
        """
        Build weighted state representation

        Returns both:
        1. Stacked states (for full temporal info)
        2. Weighted aggregate (for attention-based summary)
        """
        if len(self.history) == 0:
            padding = np.zeros(self.state_size, dtype=np.float32)
            return np.tile(padding, self.lookback)

        # Get states
        num_states = min(len(self.history), self.lookback)
        states_list = list(self.history)[-num_states:]

        # Pad if needed
        if num_states < self.lookback:
            padding = np.zeros(self.state_size, dtype=np.float32)
            padding_states = [padding] * (self.lookback - num_states)
            states_list = padding_states + states_list

        # Stack for output
        stacked = np.concatenate(states_list)

        return stacked.astype(np.float32)

    def build_weighted(self) -> np.ndarray:
        """Build attention-weighted aggregate state"""
        if len(self.history) == 0:
            return np.zeros(self.state_size, dtype=np.float32)

        # Get states
        states_array = np.array(list(self.history))

        # Apply attention weights
        if len(states_array) < self.lookback:
            # Use only available weights
            weights = self.attention_weights[-len(states_array):]
            weights = weights / weights.sum()
        else:
            weights = self.attention_weights

        # Weighted sum
        weighted_state = np.sum(
            states_array * weights[:, np.newaxis],
            axis=0
        )

        return weighted_state.astype(np.float32)

    def reset(self) -> None:
        """Reset history"""
        self.history.clear()


__all__ = [
    'MultiTimestepStateBuilder',
    'AdaptiveTimestepBuilder',
    'TemporalAttentionBuilder'
]
