#!/usr/bin/env python3
"""
Time-Based Feature Engineering for RL Trading Agents

Provides time-based features with cyclical encoding for reinforcement learning.
Captures temporal patterns and market session effects.
"""

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeFeatureEngineer:
    """
    Time-based feature engineering for trading RL agents

    Generates cyclically-encoded time features:
    - Hour of day (sin/cos encoding)
    - Day of week (sin/cos encoding)
    - Market session indicators (Asia/Europe/US)
    - Weekend detection
    """

    def __init__(
        self,
        include_hour: bool = True,
        include_day_of_week: bool = True,
        include_sessions: bool = True,
        include_weekend: bool = True,
        timezone: str = 'UTC'
    ):
        """
        Initialize time feature engineer

        Args:
            include_hour: Include hour cyclical encoding (default: True)
            include_day_of_week: Include day of week encoding (default: True)
            include_sessions: Include market session indicators (default: True)
            include_weekend: Include weekend indicator (default: True)
            timezone: Timezone for time calculations (default: 'UTC')
        """
        self.include_hour = include_hour
        self.include_day_of_week = include_day_of_week
        self.include_sessions = include_sessions
        self.include_weekend = include_weekend
        self.timezone = timezone

        logger.debug(f"TimeFeatureEngineer initialized with timezone={timezone}")

    def engineer_features(
        self,
        timestamp_series: pd.Series
    ) -> pd.DataFrame:
        """
        Engineer time features from timestamp series

        Args:
            timestamp_series: Series of timestamps

        Returns:
            DataFrame with time features
        """
        # Normalize input to a Series so downstream indexing works for both
        # pandas Series and DatetimeIndex inputs.
        if isinstance(timestamp_series, pd.DatetimeIndex):
            timestamp_series = pd.Series(timestamp_series)
        elif not isinstance(timestamp_series, pd.Series):
            timestamp_series = pd.Series(timestamp_series)

        features = pd.DataFrame(index=timestamp_series.index)

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(timestamp_series):
            timestamp_series = pd.to_datetime(timestamp_series)

        # Extract datetime components
        dt = timestamp_series.dt

        # 1. Hour cyclical encoding
        if self.include_hour:
            features['hour_sin'] = self._encode_cyclical(dt.hour, 24, 'sin')
            features['hour_cos'] = self._encode_cyclical(dt.hour, 24, 'cos')

        # 2. Day of week cyclical encoding
        if self.include_day_of_week:
            features['day_of_week_sin'] = self._encode_cyclical(dt.dayofweek, 7, 'sin')
            features['day_of_week_cos'] = self._encode_cyclical(dt.dayofweek, 7, 'cos')

        # 3. Market sessions
        if self.include_sessions:
            session_features = self._calculate_market_sessions(dt)
            for col, values in session_features.items():
                features[col] = values

        # 4. Weekend indicator
        if self.include_weekend:
            features['is_weekend'] = (dt.dayofweek >= 5).astype(float)

        # 5. Additional time patterns
        features['is_month_start'] = dt.is_month_start.astype(float)
        features['is_month_end'] = dt.is_month_end.astype(float)
        features['is_quarter_start'] = dt.is_quarter_start.astype(float)
        features['is_quarter_end'] = dt.is_quarter_end.astype(float)

        return features

    def get_state_vector(
        self,
        timestamp_series: pd.Series,
        current_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get time features as a state vector for the current timestep

        Args:
            timestamp_series: Timestamp series
            current_idx: Index to extract (default: last)

        Returns:
            Numpy array with time features
        """
        features = self.engineer_features(timestamp_series)

        if current_idx is None:
            current_idx = -1

        state_vector = features.iloc[current_idx].values.astype(np.float32)

        return state_vector

    def _encode_cyclical(
        self,
        values: pd.Series,
        period: int,
        encoding: str
    ) -> pd.Series:
        """
        Encode cyclical values using sin or cos

        Args:
            values: Values to encode
            period: Period of the cycle
            encoding: 'sin' or 'cos'

        Returns:
            Encoded series
        """
        if encoding == 'sin':
            return np.sin(2 * np.pi * values / period)
        elif encoding == 'cos':
            return np.cos(2 * np.pi * values / period)
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    def _calculate_market_sessions(self, dt: pd.Series) -> dict:
        """
        Calculate market session indicators (UTC-based)

        Sessions (approximate, UTC time):
        - Asia: 00:00 - 08:00 UTC (Tokyo: 00:00-09:00 JST)
        - Europe: 08:00 - 16:00 UTC (London: 08:00-16:30 GMT)
        - US: 14:30 - 21:00 UTC (NY: 09:30-16:00 EST)

        Note: Crypto markets trade 24/7, but these sessions
        can still affect liquidity and volatility
        """
        hour = dt.hour

        features = {}

        # Asia session (Tokyo hours)
        features['is_asia_session'] = ((hour >= 0) & (hour < 8)).astype(float)

        # Europe session (London hours)
        features['is_europe_session'] = ((hour >= 8) & (hour < 16)).astype(float)

        # US session (New York hours)
        features['is_us_session'] = ((hour >= 14) & (hour < 21)).astype(float)

        # Overlap periods (high liquidity)
        features['is_europe_us_overlap'] = ((hour >= 14) & (hour < 16)).astype(float)
        features['is_asia_europe_overlap'] = (hour == 8).astype(float)

        # Off-hours (low liquidity)
        features['is_off_hours'] = (
            ~(features['is_asia_session'].astype(bool) |
              features['is_europe_session'].astype(bool) |
              features['is_us_session'].astype(bool))
        ).astype(float)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        names = []

        if self.include_hour:
            names.extend(['hour_sin', 'hour_cos'])

        if self.include_day_of_week:
            names.extend(['day_of_week_sin', 'day_of_week_cos'])

        if self.include_sessions:
            names.extend([
                'is_asia_session',
                'is_europe_session',
                'is_us_session',
                'is_europe_us_overlap',
                'is_asia_europe_overlap',
                'is_off_hours'
            ])

        if self.include_weekend:
            names.append('is_weekend')

        # Additional patterns
        names.extend([
            'is_month_start',
            'is_month_end',
            'is_quarter_start',
            'is_quarter_end'
        ])

        return names

    def get_feature_count(self) -> int:
        """Get number of features generated"""
        return len(self.get_feature_names())


class PositionTimeFeatureEngineer:
    """
    Position-related time features for trading RL agents

    Tracks temporal aspects of current position:
    - Time in position
    - Time-weighted returns
    - Position age decay
    """

    def __init__(self, max_position_time: int = 100):
        """
        Initialize position time feature engineer

        Args:
            max_position_time: Maximum expected position duration for normalization
        """
        self.max_position_time = max_position_time
        logger.debug(f"PositionTimeFeatureEngineer initialized with max_time={max_position_time}")

    def calculate_time_in_position(
        self,
        position_history: List[float],
        normalize: bool = True
    ) -> float:
        """
        Calculate time steps since position was opened

        Args:
            position_history: List of position sizes over time
            normalize: Normalize by max_position_time

        Returns:
            Time in position (normalized if requested)
        """
        if not position_history or all(p == 0 for p in position_history):
            return 0.0

        # Count consecutive non-zero positions from end
        time_in_position = 0
        for pos in reversed(position_history):
            if pos != 0:
                time_in_position += 1
            else:
                break

        if normalize:
            return min(time_in_position / self.max_position_time, 1.0)
        else:
            return float(time_in_position)

    def calculate_position_age_decay(
        self,
        time_in_position: float,
        decay_rate: float = 0.05
    ) -> float:
        """
        Calculate position age decay factor

        Exponential decay: exp(-decay_rate * time)

        Encourages closing old positions
        """
        return float(np.exp(-decay_rate * time_in_position))

    def calculate_hold_urgency(
        self,
        time_in_position: float,
        max_hold_time: int = 50
    ) -> float:
        """
        Calculate urgency to close position based on hold time

        Returns value 0-1:
        - 0: Just opened
        - 1: At or beyond max_hold_time
        """
        urgency = min(time_in_position / max_hold_time, 1.0)
        return float(urgency)


__all__ = ['TimeFeatureEngineer', 'PositionTimeFeatureEngineer']
