#!/usr/bin/env python3
"""
Nexlify Feature Engineering Package

Comprehensive state engineering for RL trading agents.
Provides volume, trend, volatility, time, and position features
with normalization and multi-timestep stacking.
"""

from nexlify.features.volume_features import VolumeFeatureEngineer
from nexlify.features.technical_features import (
    TrendFeatureEngineer,
    VolatilityFeatureEngineer
)
from nexlify.features.time_features import (
    TimeFeatureEngineer,
    PositionTimeFeatureEngineer
)
from nexlify.features.state_normalizer import StateNormalizer, BatchNormalizer
from nexlify.features.multi_timestep_builder import (
    MultiTimestepStateBuilder,
    AdaptiveTimestepBuilder,
    TemporalAttentionBuilder
)
from nexlify.features.state_engineering import EnhancedStateEngineer

__all__ = [
    # Feature engineers
    'VolumeFeatureEngineer',
    'TrendFeatureEngineer',
    'VolatilityFeatureEngineer',
    'TimeFeatureEngineer',
    'PositionTimeFeatureEngineer',

    # Normalizers
    'StateNormalizer',
    'BatchNormalizer',

    # Multi-timestep builders
    'MultiTimestepStateBuilder',
    'AdaptiveTimestepBuilder',
    'TemporalAttentionBuilder',

    # Main orchestrator
    'EnhancedStateEngineer'
]
