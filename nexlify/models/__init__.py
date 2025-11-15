#!/usr/bin/env python3
"""
Nexlify Models Module
Neural network architectures for reinforcement learning and model management
"""

from nexlify.models.dueling_network import DuelingNetwork, StandardDQNNetwork
from nexlify.models.model_manifest import (
    ModelManifest,
    TradingCapabilities,
    TrainingMetadata,
    ModelManager,
)

__all__ = [
    # Neural networks
    "DuelingNetwork",
    "StandardDQNNetwork",
    # Model management
    "ModelManifest",
    "TradingCapabilities",
    "TrainingMetadata",
    "ModelManager",
]
