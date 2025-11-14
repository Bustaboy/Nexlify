#!/usr/bin/env python3
"""
Technical Indicators Feature Engineering for RL Trading Agents

Provides trend and volatility features optimized for reinforcement learning.
Includes EMA crossovers, ADX, momentum indicators, ATR, and Bollinger Bands.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendFeatureEngineer:
    """
    Trend feature engineering for trading RL agents

    Generates trend-following indicators:
    - EMA crossovers
    - ADX (trend strength)
    - Rate of change
    - Momentum indicators
    """

    def __init__(
        self,
        ema_short_periods: Optional[List[int]] = None,
        ema_long_periods: Optional[List[int]] = None,
        adx_period: int = 14,
        roc_period: int = 10,
        momentum_periods: Optional[List[int]] = None
    ):
        """
        Initialize trend feature engineer

        Args:
            ema_short_periods: List of short EMA periods (default: [9, 12])
            ema_long_periods: List of long EMA periods (default: [26, 50])
            adx_period: ADX calculation period (default: 14)
            roc_period: Rate of Change period (default: 10)
            momentum_periods: Momentum calculation periods (default: [5, 10, 20])
        """
        self.ema_short_periods = ema_short_periods or [9, 12]
        self.ema_long_periods = ema_long_periods or [26, 50]
        self.adx_period = adx_period
        self.roc_period = roc_period
        self.momentum_periods = momentum_periods or [5, 10, 20]

        logger.debug(f"TrendFeatureEngineer initialized with EMA periods: "
                    f"{self.ema_short_periods} x {self.ema_long_periods}")

    def engineer_features(
        self,
        price_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer trend features from OHLC data

        Args:
            price_df: DataFrame with columns [open, high, low, close]

        Returns:
            DataFrame with trend features
        """
        features = pd.DataFrame(index=price_df.index)

        # 1. EMA crossovers
        ema_features = self._calculate_ema_crossovers(price_df['close'])
        for col, values in ema_features.items():
            features[col] = values

        # 2. ADX (trend strength)
        features['adx'] = self._calculate_adx(price_df)

        # 3. Trend intensity (distance from SMA)
        features['trend_intensity'] = self._calculate_trend_intensity(price_df['close'])

        # 4. Rate of Change
        features['roc'] = self._calculate_roc(price_df['close'])

        # 5. Momentum (multiple periods)
        momentum_features = self._calculate_momentum(price_df['close'])
        for col, values in momentum_features.items():
            features[col] = values

        # 6. Directional movement indicators
        di_features = self._calculate_directional_indicators(price_df)
        for col, values in di_features.items():
            features[col] = values

        # Fill NaN values
        features = features.fillna(0.0)

        return features

    def _calculate_ema_crossovers(self, close_series: pd.Series) -> dict:
        """
        Calculate EMA crossovers: (EMA_short - EMA_long) / price

        Returns normalized crossover values for different period combinations
        """
        features = {}

        for short_period in self.ema_short_periods:
            for long_period in self.ema_long_periods:
                if short_period >= long_period:
                    continue

                ema_short = close_series.ewm(span=short_period, adjust=False).mean()
                ema_long = close_series.ewm(span=long_period, adjust=False).mean()

                # Normalized crossover
                crossover = (ema_short - ema_long) / (close_series + 1e-10)

                # Clip to reasonable range
                crossover = crossover.clip(-0.5, 0.5)

                feature_name = f'ema_{short_period}_{long_period}_cross'
                features[feature_name] = crossover

        return features

    def _calculate_adx(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)

        Measures trend strength (0-100):
        - 0-25: Weak/no trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        """
        high = price_df['high']
        low = price_df['low']
        close = price_df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed indicators using EWM (Wilder's smoothing)
        alpha = 1 / self.adx_period

        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        pos_di = 100 * (pos_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10))
        neg_di = 100 * (neg_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10))

        # DX and ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        # Normalize to 0-1 range
        adx_normalized = adx / 100.0

        return adx_normalized

    def _calculate_trend_intensity(
        self,
        close_series: pd.Series,
        sma_period: int = 50
    ) -> pd.Series:
        """
        Calculate trend intensity: abs(price - SMA50) / SMA50

        Measures how far price is from the moving average
        """
        sma = close_series.rolling(window=sma_period, min_periods=1).mean()

        intensity = abs(close_series - sma) / (sma + 1e-10)

        # Clip to reasonable range
        intensity = intensity.clip(0, 1.0)

        return intensity

    def _calculate_roc(self, close_series: pd.Series) -> pd.Series:
        """
        Calculate Rate of Change (10 period)

        ROC = (price - price_n) / price_n
        """
        roc = close_series.pct_change(periods=self.roc_period)

        # Clip to reasonable range
        roc = roc.clip(-0.5, 0.5)

        return roc

    def _calculate_momentum(self, close_series: pd.Series) -> dict:
        """
        Calculate momentum for multiple periods

        Momentum = price - price_n_days_ago
        """
        features = {}

        for period in self.momentum_periods:
            momentum = close_series - close_series.shift(period)

            # Normalize by price
            normalized_momentum = momentum / (close_series + 1e-10)

            # Clip to reasonable range
            normalized_momentum = normalized_momentum.clip(-0.5, 0.5)

            features[f'momentum_{period}'] = normalized_momentum

        return features

    def _calculate_directional_indicators(self, price_df: pd.DataFrame) -> dict:
        """
        Calculate +DI and -DI (Directional Indicators)

        Useful for identifying trend direction
        """
        high = price_df['high']
        low = price_df['low']
        close = price_df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed indicators
        alpha = 1 / self.adx_period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()

        pos_di = (pos_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10))
        neg_di = (neg_dm.ewm(alpha=alpha, adjust=False).mean() / (atr + 1e-10))

        # Clip to 0-1 range (already normalized by ATR)
        pos_di = pos_di.clip(0, 1.0)
        neg_di = neg_di.clip(0, 1.0)

        return {
            'plus_di': pos_di,
            'minus_di': neg_di
        }

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        names = []

        # EMA crossovers
        for short in self.ema_short_periods:
            for long in self.ema_long_periods:
                if short < long:
                    names.append(f'ema_{short}_{long}_cross')

        # Other features
        names.extend([
            'adx',
            'trend_intensity',
            'roc'
        ])

        # Momentum features
        for period in self.momentum_periods:
            names.append(f'momentum_{period}')

        # Directional indicators
        names.extend(['plus_di', 'minus_di'])

        return names

    def get_feature_count(self) -> int:
        """Get number of features generated"""
        return len(self.get_feature_names())


class VolatilityFeatureEngineer:
    """
    Volatility feature engineering for trading RL agents

    Generates volatility indicators:
    - ATR (Average True Range)
    - Bollinger Bands position and width
    - Historical volatility at multiple timescales
    """

    def __init__(
        self,
        atr_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        volatility_periods: Optional[List[int]] = None
    ):
        """
        Initialize volatility feature engineer

        Args:
            atr_period: ATR calculation period (default: 14)
            bb_period: Bollinger Bands period (default: 20)
            bb_std: Bollinger Bands standard deviations (default: 2.0)
            volatility_periods: Historical volatility periods (default: [10, 30])
        """
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.volatility_periods = volatility_periods or [10, 30]

        logger.debug(f"VolatilityFeatureEngineer initialized with ATR={atr_period}, BB={bb_period}")

    def engineer_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer volatility features from OHLC data

        Args:
            price_df: DataFrame with columns [open, high, low, close]

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=price_df.index)

        # 1. ATR (normalized)
        features['atr_norm'] = self._calculate_atr(price_df)

        # 2. Bollinger Bands position and width
        bb_features = self._calculate_bollinger_bands(price_df['close'])
        for col, values in bb_features.items():
            features[col] = values

        # 3. Historical volatility at multiple periods
        vol_features = self._calculate_historical_volatility(price_df['close'])
        for col, values in vol_features.items():
            features[col] = values

        # 4. Volatility ratio (short/long term)
        features['vol_ratio'] = self._calculate_volatility_ratio(price_df['close'])

        # 5. Intraday volatility
        features['intraday_vol'] = self._calculate_intraday_volatility(price_df)

        # Fill NaN values
        features = features.fillna(0.0)

        return features

    def _calculate_atr(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Average True Range (normalized by price)

        Returns ATR as percentage of price
        """
        high = price_df['high']
        low = price_df['low']
        close = price_df['close']

        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR using Wilder's smoothing (EWM)
        alpha = 1 / self.atr_period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()

        # Normalize by close price
        atr_normalized = atr / (close + 1e-10)

        # Clip to reasonable range
        atr_normalized = atr_normalized.clip(0, 0.5)

        return atr_normalized

    def _calculate_bollinger_bands(self, close_series: pd.Series) -> dict:
        """
        Calculate Bollinger Bands features

        Returns:
        - bb_position: (price - lower) / (upper - lower)
        - bb_width: (upper - lower) / middle
        """
        # Calculate bands
        sma = close_series.rolling(window=self.bb_period, min_periods=1).mean()
        std = close_series.rolling(window=self.bb_period, min_periods=1).std()

        upper_band = sma + (self.bb_std * std)
        lower_band = sma - (self.bb_std * std)

        # BB Position: 0 = at lower band, 1 = at upper band
        bb_position = (close_series - lower_band) / (upper_band - lower_band + 1e-10)
        bb_position = bb_position.clip(0, 1.0)

        # BB Width: relative width of bands
        bb_width = (upper_band - lower_band) / (sma + 1e-10)
        bb_width = bb_width.clip(0, 1.0)

        return {
            'bb_position': bb_position,
            'bb_width': bb_width
        }

    def _calculate_historical_volatility(self, close_series: pd.Series) -> dict:
        """
        Calculate historical volatility (annualized) for multiple periods

        Uses standard deviation of returns * sqrt(252) for annualization
        """
        features = {}

        returns = close_series.pct_change()

        for period in self.volatility_periods:
            vol = returns.rolling(window=period, min_periods=1).std()

            # Annualize (crypto trades 24/7, so we use different factor)
            # For daily data: sqrt(365)
            # For hourly data: sqrt(365 * 24)
            # We'll use a generic sqrt(252) as in traditional markets
            vol_annualized = vol * np.sqrt(252)

            # Clip to reasonable range
            vol_annualized = vol_annualized.clip(0, 5.0)

            features[f'vol_{period}'] = vol_annualized

        return features

    def _calculate_volatility_ratio(
        self,
        close_series: pd.Series,
        short_period: int = 10,
        long_period: int = 30
    ) -> pd.Series:
        """
        Calculate volatility ratio: vol_short / vol_long

        Values > 1: Increasing volatility
        Values < 1: Decreasing volatility
        """
        returns = close_series.pct_change()

        vol_short = returns.rolling(window=short_period, min_periods=1).std()
        vol_long = returns.rolling(window=long_period, min_periods=1).std()

        vol_ratio = vol_short / (vol_long + 1e-10)

        # Clip to reasonable range
        vol_ratio = vol_ratio.clip(0, 3.0)

        return vol_ratio

    def _calculate_intraday_volatility(self, price_df: pd.DataFrame) -> pd.Series:
        """
        Calculate intraday volatility: (high - low) / close

        Measures volatility within each period
        """
        intraday_vol = (price_df['high'] - price_df['low']) / (price_df['close'] + 1e-10)

        # Clip to reasonable range
        intraday_vol = intraday_vol.clip(0, 0.5)

        return intraday_vol

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        names = [
            'atr_norm',
            'bb_position',
            'bb_width'
        ]

        # Historical volatility
        for period in self.volatility_periods:
            names.append(f'vol_{period}')

        names.extend([
            'vol_ratio',
            'intraday_vol'
        ])

        return names

    def get_feature_count(self) -> int:
        """Get number of features generated"""
        return len(self.get_feature_names())


__all__ = ['TrendFeatureEngineer', 'VolatilityFeatureEngineer']
