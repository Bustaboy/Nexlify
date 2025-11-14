#!/usr/bin/env python3
"""
Volume Feature Engineering for RL Trading Agents

Provides volume-based features optimized for reinforcement learning state vectors.
Focuses on compact, normalized features that capture volume dynamics.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class VolumeFeatureEngineer:
    """
    Volume feature engineering for trading RL agents

    Generates compact, normalized volume features that capture:
    - Volume trends and momentum
    - Relative volume positioning
    - Volume volatility and clustering
    """

    def __init__(
        self,
        lookback_short: int = 5,
        lookback_medium: int = 10,
        lookback_long: int = 20,
        max_lookback: int = 50
    ):
        """
        Initialize volume feature engineer

        Args:
            lookback_short: Short-term lookback period (default: 5)
            lookback_medium: Medium-term lookback period (default: 10)
            lookback_long: Long-term lookback period (default: 20)
            max_lookback: Maximum lookback for relative volume (default: 50)
        """
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long
        self.max_lookback = max_lookback

        logger.debug(f"VolumeFeatureEngineer initialized with lookbacks: "
                    f"{lookback_short}/{lookback_medium}/{lookback_long}")

    def engineer_features(
        self,
        volume_series: pd.Series,
        price_series: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Engineer volume features from volume time series

        Args:
            volume_series: Volume time series
            price_series: Optional price series for volume-price features

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=volume_series.index)

        # 1. Volume ratio (current volume / average volume)
        features['volume_ratio'] = self._calculate_volume_ratio(volume_series)

        # 2. Volume momentum
        features['volume_momentum'] = self._calculate_volume_momentum(volume_series)

        # 3. Volume trend (linear regression slope)
        features['volume_trend'] = self._calculate_volume_trend(volume_series)

        # 4. Relative volume (percentile in lookback window)
        features['relative_volume'] = self._calculate_relative_volume(volume_series)

        # 5. Volume volatility
        features['volume_volatility'] = self._calculate_volume_volatility(volume_series)

        # 6. Volume acceleration
        features['volume_acceleration'] = self._calculate_volume_acceleration(volume_series)

        # 7. Volume-price divergence (if price available)
        if price_series is not None:
            features['volume_price_divergence'] = self._calculate_volume_price_divergence(
                volume_series, price_series
            )

        # Fill NaN values with 0 (for early periods)
        features = features.fillna(0.0)

        return features

    def get_state_vector(
        self,
        volume_series: pd.Series,
        price_series: Optional[pd.Series] = None,
        current_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get volume features as a state vector for the current timestep

        Args:
            volume_series: Volume time series
            price_series: Optional price series
            current_idx: Index to extract (default: last)

        Returns:
            Numpy array with volume features
        """
        features = self.engineer_features(volume_series, price_series)

        if current_idx is None:
            current_idx = -1

        # Extract current features
        state_vector = features.iloc[current_idx].values.astype(np.float32)

        return state_vector

    def _calculate_volume_ratio(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate volume ratio: current_volume / avg_volume_20

        Normalized around 1.0:
        - > 1.0: Above average volume
        - < 1.0: Below average volume
        """
        avg_volume = volume_series.rolling(window=self.lookback_long, min_periods=1).mean()
        volume_ratio = volume_series / (avg_volume + 1e-10)

        # Clip extreme values
        volume_ratio = volume_ratio.clip(0, 5.0)

        return volume_ratio

    def _calculate_volume_momentum(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate volume momentum: (volume_5 - volume_20) / volume_20

        Captures whether volume is increasing or decreasing
        """
        volume_short = volume_series.rolling(
            window=self.lookback_short, min_periods=1
        ).mean()
        volume_long = volume_series.rolling(
            window=self.lookback_long, min_periods=1
        ).mean()

        momentum = (volume_short - volume_long) / (volume_long + 1e-10)

        # Clip to reasonable range
        momentum = momentum.clip(-2.0, 2.0)

        return momentum

    def _calculate_volume_trend(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate volume trend using linear regression slope of last 10 volumes

        Positive slope: Volume trending up
        Negative slope: Volume trending down
        """
        def rolling_slope(window_data):
            """Calculate slope of linear regression"""
            if len(window_data) < 2:
                return 0.0

            x = np.arange(len(window_data))
            y = window_data.values

            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return 0.0

            x_clean = x[mask]
            y_clean = y[mask]

            try:
                slope, _, _, _, _ = stats.linregress(x_clean, y_clean)
                # Normalize by mean to get relative slope
                mean_val = np.mean(y_clean)
                if mean_val > 0:
                    return slope / mean_val
                return 0.0
            except:
                return 0.0

        trend = volume_series.rolling(
            window=self.lookback_medium, min_periods=2
        ).apply(rolling_slope, raw=False)

        # Clip to reasonable range
        trend = trend.clip(-1.0, 1.0)

        return trend

    def _calculate_relative_volume(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate relative volume: current_volume / max_volume_lookback

        Shows where current volume stands relative to recent maximum
        """
        max_volume = volume_series.rolling(
            window=self.max_lookback, min_periods=1
        ).max()

        relative_vol = volume_series / (max_volume + 1e-10)

        # Should be between 0 and 1
        relative_vol = relative_vol.clip(0, 1.0)

        return relative_vol

    def _calculate_volume_volatility(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate volume volatility: std(volume_10) / mean(volume_10)

        Coefficient of variation for volume
        High values indicate unstable volume
        """
        rolling_mean = volume_series.rolling(
            window=self.lookback_medium, min_periods=1
        ).mean()
        rolling_std = volume_series.rolling(
            window=self.lookback_medium, min_periods=1
        ).std()

        volatility = rolling_std / (rolling_mean + 1e-10)

        # Clip to reasonable range
        volatility = volatility.clip(0, 3.0)

        return volatility

    def _calculate_volume_acceleration(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate volume acceleration (second derivative)

        Captures rate of change in volume momentum
        """
        # First derivative (momentum)
        volume_diff = volume_series.diff()

        # Second derivative (acceleration)
        volume_accel = volume_diff.diff()

        # Normalize by rolling std to make scale-invariant
        rolling_std = volume_series.rolling(
            window=self.lookback_long, min_periods=1
        ).std()

        normalized_accel = volume_accel / (rolling_std + 1e-10)

        # Clip extreme values
        normalized_accel = normalized_accel.clip(-3.0, 3.0)

        return normalized_accel

    def _calculate_volume_price_divergence(
        self,
        volume_series: pd.Series,
        price_series: pd.Series
    ) -> pd.Series:
        """
        Calculate volume-price divergence

        Detects when volume and price trends diverge:
        - Positive divergence: Volume up, price down (potential reversal)
        - Negative divergence: Volume down, price up (weak rally)
        """
        # Calculate price change
        price_change = price_series.pct_change(self.lookback_short)

        # Calculate volume change
        volume_change = volume_series.pct_change(self.lookback_short)

        # Divergence: opposite signs indicate divergence
        # Multiply by -1 so that:
        # Positive: Volume increasing while price decreasing (bullish divergence)
        # Negative: Volume decreasing while price increasing (bearish divergence)
        divergence = -1 * np.sign(price_change) * volume_change

        # Clip to reasonable range
        divergence = divergence.clip(-2.0, 2.0)

        return divergence

    def get_feature_names(self, include_price_divergence: bool = False) -> List[str]:
        """
        Get list of feature names

        Args:
            include_price_divergence: Whether to include volume-price divergence

        Returns:
            List of feature names
        """
        names = [
            'volume_ratio',
            'volume_momentum',
            'volume_trend',
            'relative_volume',
            'volume_volatility',
            'volume_acceleration'
        ]

        if include_price_divergence:
            names.append('volume_price_divergence')

        return names

    def get_feature_count(self, include_price_divergence: bool = False) -> int:
        """Get number of features generated"""
        return len(self.get_feature_names(include_price_divergence))


__all__ = ['VolumeFeatureEngineer']
