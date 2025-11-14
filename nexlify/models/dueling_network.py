#!/usr/bin/env python3
"""
Dueling DQN Network Architecture
Implements both standard and dueling DQN architectures

Dueling DQN separates value and advantage estimation:
Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]

Benefits:
- Better state value estimation
- Learns which states are valuable regardless of action
- More stable learning in environments where actions don't always matter
"""

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StandardDQNNetwork(nn.Module):
    """
    Standard DQN architecture
    Simple feedforward network: state → hidden layers → Q-values
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: Tuple[int, ...] = (128, 128, 64),
        activation: str = "relu",
    ):
        """
        Initialize standard DQN network

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: Tuple of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super(StandardDQNNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes

        # Build layers
        layers = []
        in_size = state_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(self._get_activation(activation))
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, action_size))

        self.network = nn.Sequential(*layers)

        logger.info(
            f"✅ Standard DQN Network: {state_size} → {hidden_sizes} → {action_size}"
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: State tensor [batch_size, state_size]

        Returns:
            Q-values tensor [batch_size, action_size]
        """
        return self.network(x)


class DuelingNetwork(nn.Module):
    """
    Dueling DQN Architecture

    Splits network into:
    - Value stream: Estimates state value V(s)
    - Advantage stream: Estimates advantage A(s,a) for each action

    Q-values computed as:
    Q(s,a) = V(s) + [A(s,a) - aggregation(A(s,·))]

    where aggregation can be mean (original paper) or max
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        shared_sizes: Tuple[int, ...] = (128, 128),
        value_sizes: Tuple[int, ...] = (64,),
        advantage_sizes: Tuple[int, ...] = (64,),
        activation: str = "relu",
        aggregation: str = "mean",
    ):
        """
        Initialize Dueling DQN network

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            shared_sizes: Sizes of shared feature extractor layers
            value_sizes: Sizes of value stream hidden layers
            advantage_sizes: Sizes of advantage stream hidden layers
            activation: Activation function ('relu', 'tanh', 'elu')
            aggregation: How to aggregate advantages ('mean' or 'max')
        """
        super(DuelingNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.aggregation = aggregation

        # Shared feature extractor
        shared_layers = []
        in_size = state_size

        for hidden_size in shared_sizes:
            shared_layers.append(nn.Linear(in_size, hidden_size))
            shared_layers.append(self._get_activation(activation))
            in_size = hidden_size

        self.shared_layers = nn.Sequential(*shared_layers)
        shared_out_size = shared_sizes[-1] if shared_sizes else state_size

        # Value stream: V(s)
        value_layers = []
        in_size = shared_out_size

        for hidden_size in value_sizes:
            value_layers.append(nn.Linear(in_size, hidden_size))
            value_layers.append(self._get_activation(activation))
            in_size = hidden_size

        value_layers.append(nn.Linear(in_size, 1))  # Single value output
        self.value_stream = nn.Sequential(*value_layers)

        # Advantage stream: A(s,a)
        advantage_layers = []
        in_size = shared_out_size

        for hidden_size in advantage_sizes:
            advantage_layers.append(nn.Linear(in_size, hidden_size))
            advantage_layers.append(self._get_activation(activation))
            in_size = hidden_size

        advantage_layers.append(
            nn.Linear(in_size, action_size)
        )  # One advantage per action
        self.advantage_stream = nn.Sequential(*advantage_layers)

        logger.info(
            f"✅ Dueling DQN Network initialized:\n"
            f"   Shared: {state_size} → {shared_sizes}\n"
            f"   Value: {shared_sizes[-1] if shared_sizes else state_size} → {value_sizes} → 1\n"
            f"   Advantage: {shared_sizes[-1] if shared_sizes else state_size} → {advantage_sizes} → {action_size}\n"
            f"   Aggregation: {aggregation}"
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function module"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        return activations.get(activation.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture

        Args:
            x: State tensor [batch_size, state_size]

        Returns:
            Q-values tensor [batch_size, action_size]
        """
        # Shared features
        shared_features = self.shared_layers(x)

        # Value and advantage streams
        value = self.value_stream(shared_features)  # [batch_size, 1]
        advantage = self.advantage_stream(shared_features)  # [batch_size, action_size]

        # Combine using dueling architecture formula
        # Q(s,a) = V(s) + [A(s,a) - aggregation(A(s,·))]
        if self.aggregation == "mean":
            # Original paper: subtract mean advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        elif self.aggregation == "max":
            # Alternative: subtract max advantage
            q_values = value + (advantage - advantage.max(dim=1, keepdim=True)[0])
        else:
            raise ValueError(
                f"Unknown aggregation method: {self.aggregation}. Use 'mean' or 'max'"
            )

        return q_values

    def get_value_advantage(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get separate value and advantage estimates (for analysis)

        Args:
            x: State tensor [batch_size, state_size]

        Returns:
            Tuple of (value, advantage) tensors
        """
        shared_features = self.shared_layers(x)
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        return value, advantage


def create_network(
    state_size: int,
    action_size: int,
    use_dueling: bool = True,
    config: Optional[dict] = None,
) -> nn.Module:
    """
    Factory function to create DQN network

    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        use_dueling: If True, use Dueling architecture; else standard DQN
        config: Configuration dictionary with network parameters

    Returns:
        DQN network (StandardDQNNetwork or DuelingNetwork)
    """
    config = config or {}

    if use_dueling:
        return DuelingNetwork(
            state_size=state_size,
            action_size=action_size,
            shared_sizes=tuple(
                config.get("dueling_shared_sizes", [128, 128])
            ),
            value_sizes=tuple(config.get("dueling_value_sizes", [64])),
            advantage_sizes=tuple(
                config.get("dueling_advantage_sizes", [64])
            ),
            activation=config.get("activation", "relu"),
            aggregation=config.get("dueling_aggregation", "mean"),
        )
    else:
        return StandardDQNNetwork(
            state_size=state_size,
            action_size=action_size,
            hidden_sizes=tuple(config.get("hidden_sizes", [128, 128, 64])),
            activation=config.get("activation", "relu"),
        )


__all__ = ["DuelingNetwork", "StandardDQNNetwork", "create_network"]
