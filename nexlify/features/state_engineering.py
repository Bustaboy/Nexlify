#!/usr/bin/env python3
"""
Comprehensive State Engineering System for RL Trading Agents

Orchestrates all feature engineering modules to create rich, normalized
state vectors optimized for reinforcement learning.

This module replaces placeholder features with real implementations and
provides a complete state engineering pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nexlify.features.volume_features import VolumeFeatureEngineer
from nexlify.features.technical_features import (
    TrendFeatureEngineer,
    VolatilityFeatureEngineer
)
from nexlify.features.time_features import (
    TimeFeatureEngineer,
    PositionTimeFeatureEngineer
)
from nexlify.features.state_normalizer import StateNormalizer
from nexlify.features.multi_timestep_builder import MultiTimestepStateBuilder

logger = logging.getLogger(__name__)


class EnhancedStateEngineer:
    """
    Complete state engineering system for RL trading agents

    Combines:
    - Volume features (6-7 features)
    - Trend features (8-10 features)
    - Volatility features (5-6 features)
    - Time features (4-13 features, toggleable)
    - Position features (5 features)
    - Multi-timestep stacking (optional)
    - State normalization (optional)

    Total features: 25-35+ depending on configuration
    """

    def __init__(
        self,
        use_volume: bool = True,
        use_trend: bool = True,
        use_volatility: bool = True,
        use_time: bool = True,
        use_position: bool = True,
        use_normalization: bool = True,
        use_multi_timestep: bool = False,
        multi_timestep_lookback: int = 10,
        state_feature_groups: Optional[List[str]] = None,
        normalization_clip_range: float = 3.0,
        normalization_warmup: int = 100
    ):
        """
        Initialize enhanced state engineer

        Args:
            use_volume: Include volume features (default: True)
            use_trend: Include trend features (default: True)
            use_volatility: Include volatility features (default: True)
            use_time: Include time features (default: True)
            use_position: Include position features (default: True)
            use_normalization: Normalize state vectors (default: True)
            use_multi_timestep: Stack multiple timesteps (default: False)
            multi_timestep_lookback: Timesteps to stack if enabled (default: 10)
            state_feature_groups: Specific groups to include (overrides individual flags)
            normalization_clip_range: Clip range for normalization (default: 3.0)
            normalization_warmup: Warmup samples before normalization (default: 100)
        """
        # Feature flags
        if state_feature_groups is not None:
            self.use_volume = 'volume' in state_feature_groups
            self.use_trend = 'trend' in state_feature_groups
            self.use_volatility = 'volatility' in state_feature_groups
            self.use_time = 'time' in state_feature_groups
            self.use_position = 'position' in state_feature_groups
        else:
            self.use_volume = use_volume
            self.use_trend = use_trend
            self.use_volatility = use_volatility
            self.use_time = use_time
            self.use_position = use_position

        self.use_normalization = use_normalization
        self.use_multi_timestep = use_multi_timestep
        self.multi_timestep_lookback = multi_timestep_lookback

        # Initialize feature engineers
        self.volume_engineer = VolumeFeatureEngineer() if self.use_volume else None
        self.trend_engineer = TrendFeatureEngineer() if self.use_trend else None
        self.volatility_engineer = VolatilityFeatureEngineer() if self.use_volatility else None
        self.time_engineer = TimeFeatureEngineer() if self.use_time else None
        self.position_time_engineer = PositionTimeFeatureEngineer() if self.use_position else None

        # Calculate base state size
        self.base_state_size = self._calculate_base_state_size()

        # Initialize normalizer
        if self.use_normalization:
            self.normalizer = StateNormalizer(
                state_size=self.base_state_size,
                clip_range=normalization_clip_range,
                warmup_samples=normalization_warmup
            )
        else:
            self.normalizer = None

        # Initialize multi-timestep builder
        if self.use_multi_timestep:
            self.timestep_builder = MultiTimestepStateBuilder(
                state_size=self.base_state_size,
                lookback=multi_timestep_lookback
            )
        else:
            self.timestep_builder = None

        # Position tracking for position features
        self.position_history = []
        self.current_balance = 0.0
        self.initial_balance = 10000.0
        self.current_position = 0.0
        self.entry_price = 0.0
        self.peak_equity = 0.0

        logger.info(f"EnhancedStateEngineer initialized:")
        logger.info(f"  Volume: {self.use_volume}, Trend: {self.use_trend}, "
                   f"Volatility: {self.use_volatility}")
        logger.info(f"  Time: {self.use_time}, Position: {self.use_position}")
        logger.info(f"  Base state size: {self.base_state_size}")
        logger.info(f"  Normalization: {self.use_normalization}, "
                   f"Multi-timestep: {self.use_multi_timestep}")
        if self.use_multi_timestep:
            logger.info(f"  Final state size: {self.get_state_size()}")

    def _calculate_base_state_size(self) -> int:
        """Calculate total state size based on enabled features"""
        size = 0

        # Position & Portfolio (5)
        if self.use_position:
            size += 5

        # Volume features (6-7)
        if self.use_volume:
            size += self.volume_engineer.get_feature_count(include_price_divergence=True)

        # Trend features (8-10)
        if self.use_trend:
            size += self.trend_engineer.get_feature_count()

        # Volatility features (5-6)
        if self.use_volatility:
            size += self.volatility_engineer.get_feature_count()

        # Time features (4-13)
        if self.use_time:
            size += self.time_engineer.get_feature_count()

        return size

    def build_state(
        self,
        price_df: pd.DataFrame,
        volume_series: Optional[pd.Series] = None,
        timestamp_series: Optional[pd.Series] = None,
        current_balance: float = 10000.0,
        current_position: float = 0.0,
        entry_price: float = 0.0,
        current_price: float = 0.0,
        equity_history: Optional[List[float]] = None,
        update_normalizer: bool = True
    ) -> np.ndarray:
        """
        Build complete state vector from market data and position info

        Args:
            price_df: DataFrame with OHLC data
            volume_series: Volume series (optional if in price_df)
            timestamp_series: Timestamp series (optional if in price_df)
            current_balance: Current cash balance
            current_position: Current position size
            entry_price: Entry price of current position
            current_price: Current market price
            equity_history: Historical equity values
            update_normalizer: Whether to update normalization stats

        Returns:
            State vector (normalized and stacked if configured)
        """
        state_components = []

        # 1. Position & Portfolio features (5)
        if self.use_position:
            position_features = self._build_position_features(
                current_balance,
                current_position,
                entry_price,
                current_price,
                equity_history
            )
            state_components.append(position_features)

        # 2. Volume features (6-7)
        if self.use_volume:
            if volume_series is None:
                if 'volume' in price_df.columns:
                    volume_series = price_df['volume']
                else:
                    raise ValueError("Volume data required but not provided")

            price_series = price_df['close'] if 'close' in price_df.columns else None
            volume_features = self.volume_engineer.get_state_vector(
                volume_series,
                price_series
            )
            state_components.append(volume_features)

        # 3. Trend features (8-10)
        if self.use_trend:
            trend_df = self.trend_engineer.engineer_features(price_df)
            trend_features = trend_df.iloc[-1].values.astype(np.float32)
            state_components.append(trend_features)

        # 4. Volatility features (5-6)
        if self.use_volatility:
            vol_df = self.volatility_engineer.engineer_features(price_df)
            vol_features = vol_df.iloc[-1].values.astype(np.float32)
            state_components.append(vol_features)

        # 5. Time features (4-13)
        if self.use_time:
            if timestamp_series is None:
                if 'timestamp' in price_df.columns:
                    timestamp_series = price_df['timestamp']
                else:
                    # Create dummy timestamps if not available
                    timestamp_series = pd.Series([pd.Timestamp.now()] * len(price_df))

            time_features = self.time_engineer.get_state_vector(timestamp_series)
            state_components.append(time_features)

        # Concatenate all components
        state = np.concatenate(state_components).astype(np.float32)

        # Update position tracking
        self._update_position_tracking(current_position)

        # Normalize if enabled
        if self.use_normalization:
            state = self.normalizer.normalize(state, update_stats=update_normalizer)

        # Add to multi-timestep builder if enabled
        if self.use_multi_timestep:
            self.timestep_builder.add_state(state)
            state = self.timestep_builder.build()

        return state

    def _build_position_features(
        self,
        balance: float,
        position: float,
        entry_price: float,
        current_price: float,
        equity_history: Optional[List[float]]
    ) -> np.ndarray:
        """
        Build position and portfolio features (5 features)

        Returns:
            [balance_norm, position_norm, position_pnl, drawdown, time_in_position]
        """
        # Update internal state
        self.current_balance = balance
        self.current_position = position
        self.entry_price = entry_price

        # 1. Normalized balance
        balance_norm = balance / self.initial_balance

        # 2. Normalized position (relative to buying power)
        if current_price > 0:
            max_position = balance / current_price
            position_norm = position / (max_position + 1e-10)
        else:
            position_norm = 0.0

        # 3. Position P&L
        if position > 0 and entry_price > 0 and current_price > 0:
            position_pnl = (current_price - entry_price) / entry_price
        else:
            position_pnl = 0.0

        # Clip P&L to reasonable range
        position_pnl = np.clip(position_pnl, -1.0, 1.0)

        # 4. Drawdown from peak
        if equity_history and len(equity_history) > 0:
            current_equity = balance + (position * current_price)
            self.peak_equity = max(self.peak_equity, max(equity_history))
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - current_equity) / self.peak_equity
            else:
                drawdown = 0.0
        else:
            drawdown = 0.0

        drawdown = np.clip(drawdown, 0.0, 1.0)

        # 5. Time in position (normalized)
        if self.position_time_engineer:
            time_in_position = self.position_time_engineer.calculate_time_in_position(
                self.position_history,
                normalize=True
            )
        else:
            time_in_position = 0.0

        features = np.array([
            balance_norm,
            position_norm,
            position_pnl,
            drawdown,
            time_in_position
        ], dtype=np.float32)

        return features

    def _update_position_tracking(self, position: float) -> None:
        """Update position history for time-in-position calculation"""
        self.position_history.append(position)

        # Keep last 200 positions
        if len(self.position_history) > 200:
            self.position_history.pop(0)

    def reset(
        self,
        initial_balance: float = 10000.0,
        reset_normalizer: bool = False
    ) -> None:
        """
        Reset state engineer for new episode

        Args:
            initial_balance: Starting balance for new episode
            reset_normalizer: Whether to reset normalization statistics
        """
        self.position_history = []
        self.current_balance = initial_balance
        self.initial_balance = initial_balance
        self.current_position = 0.0
        self.entry_price = 0.0
        self.peak_equity = initial_balance

        if self.use_multi_timestep and self.timestep_builder:
            self.timestep_builder.reset()

        if reset_normalizer and self.normalizer:
            self.normalizer.reset()

        logger.debug("State engineer reset for new episode")

    def get_state_size(self) -> int:
        """
        Get total state size

        Returns:
            State size (accounting for multi-timestep stacking)
        """
        if self.use_multi_timestep:
            return self.base_state_size * self.multi_timestep_lookback
        else:
            return self.base_state_size

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        names = []

        # Position features
        if self.use_position:
            names.extend([
                'balance_norm',
                'position_norm',
                'position_pnl',
                'drawdown',
                'time_in_position'
            ])

        # Volume features
        if self.use_volume:
            names.extend(self.volume_engineer.get_feature_names(
                include_price_divergence=True
            ))

        # Trend features
        if self.use_trend:
            names.extend(self.trend_engineer.get_feature_names())

        # Volatility features
        if self.use_volatility:
            names.extend(self.volatility_engineer.get_feature_names())

        # Time features
        if self.use_time:
            names.extend(self.time_engineer.get_feature_names())

        return names

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category"""
        all_features = self.get_feature_names()

        groups = {}

        if self.use_position:
            groups['position'] = [f for f in all_features if any(
                x in f for x in ['balance', 'position', 'pnl', 'drawdown', 'time_in']
            )]

        if self.use_volume:
            groups['volume'] = [f for f in all_features if 'volume' in f]

        if self.use_trend:
            groups['trend'] = [f for f in all_features if any(
                x in f for x in ['ema', 'adx', 'trend', 'roc', 'momentum', 'di']
            )]

        if self.use_volatility:
            groups['volatility'] = [f for f in all_features if any(
                x in f for x in ['atr', 'bb_', 'vol_']
            )]

        if self.use_time:
            groups['time'] = [f for f in all_features if any(
                x in f for x in ['hour', 'day', 'session', 'weekend', 'month', 'quarter']
            )]

        return groups

    def save_normalization(self, filepath: Union[str, Path]) -> None:
        """Save normalization parameters"""
        if self.normalizer:
            self.normalizer.save(filepath)
        else:
            logger.warning("No normalizer to save")

    def load_normalization(self, filepath: Union[str, Path]) -> None:
        """Load normalization parameters"""
        if self.normalizer:
            self.normalizer.load(filepath)
        else:
            logger.warning("No normalizer to load into")

    def get_config(self) -> Dict:
        """Get configuration dictionary"""
        return {
            'use_volume': self.use_volume,
            'use_trend': self.use_trend,
            'use_volatility': self.use_volatility,
            'use_time': self.use_time,
            'use_position': self.use_position,
            'use_normalization': self.use_normalization,
            'use_multi_timestep': self.use_multi_timestep,
            'multi_timestep_lookback': self.multi_timestep_lookback,
            'base_state_size': self.base_state_size,
            'total_state_size': self.get_state_size()
        }


__all__ = ['EnhancedStateEngineer']
