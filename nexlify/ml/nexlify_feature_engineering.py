#!/usr/bin/env python3
"""
Nexlify Advanced Feature Engineering System
Comprehensive feature extraction for cryptocurrency trading ML models

This module provides 100+ features across multiple categories:
- Technical indicators (50+)
- Market microstructure (20+)
- Statistical features (15+)
- Time-based features (10+)
- Sentiment indicators (10+)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import sentiment analyzer
try:
    from nexlify.ml.nexlify_sentiment_analysis import SentimentAnalyzer

    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for cryptocurrency trading

    Generates 100+ features optimized for ML/RL models
    """

    def __init__(
        self,
        lookback_periods: Optional[List[int]] = None,
        enable_sentiment: bool = True,
        sentiment_config: Optional[Dict] = None,
    ):
        """
        Initialize feature engineer

        Args:
            lookback_periods: List of periods for rolling calculations [7, 14, 30, 60]
            enable_sentiment: Enable sentiment analysis features
            sentiment_config: Configuration for sentiment analyzer (API keys, etc.)
        """
        self.lookback_periods = lookback_periods or [7, 14, 30, 60]
        self.feature_names = []
        self.enable_sentiment = enable_sentiment and SENTIMENT_AVAILABLE

        # Initialize sentiment analyzer if enabled
        self.sentiment_analyzer = None
        if self.enable_sentiment:
            try:
                self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
                logger.info("🔧 Feature Engineer initialized (with sentiment analysis)")
            except Exception as e:
                logger.warning(f"Sentiment analyzer failed to initialize: {e}")
                self.enable_sentiment = False
                logger.info("🔧 Feature Engineer initialized (without sentiment)")
        else:
            logger.info("🔧 Feature Engineer initialized (without sentiment)")

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV data

        Args:
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            DataFrame with original data + engineered features
        """
        logger.info(f"Engineering features for {len(df)} data points...")

        result = df.copy()

        # 1. Price-based features
        result = self._add_price_features(result)

        # 2. Technical indicators
        result = self._add_technical_indicators(result)

        # 3. Volatility features
        result = self._add_volatility_features(result)

        # 4. Volume features
        result = self._add_volume_features(result)

        # 5. Momentum features
        result = self._add_momentum_features(result)

        # 6. Statistical features
        result = self._add_statistical_features(result)

        # 7. Pattern features
        result = self._add_pattern_features(result)

        # 8. Time-based features
        result = self._add_time_features(result)

        # 9. Market microstructure
        result = self._add_microstructure_features(result)

        # 10. Sentiment features (async, cached)
        if self.enable_sentiment:
            result = self._add_sentiment_features(result)

        # Fill NaN values
        result = result.fillna(method="bfill").fillna(0)

        logger.info(f"✅ Generated {len(result.columns) - len(df.columns)} features")

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features"""

        # Returns at different periods
        for period in [1, 3, 5, 10, 20]:
            df[f"return_{period}"] = df["close"].pct_change(period)
            df[f"log_return_{period}"] = np.log(df["close"] / df["close"].shift(period))

        # Price position in candle
        df["price_position"] = (df["close"] - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        # Body to shadow ratio
        df["body_size"] = abs(df["close"] - df["open"]) / (
            df["high"] - df["low"] + 1e-10
        )
        df["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / (
            df["high"] - df["low"] + 1e-10
        )
        df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / (
            df["high"] - df["low"] + 1e-10
        )

        # Price momentum
        df["momentum_5"] = df["close"] - df["close"].shift(5)
        df["momentum_10"] = df["close"] - df["close"].shift(10)
        df["momentum_20"] = df["close"] - df["close"].shift(20)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical indicators"""

        # Moving Averages
        for period in [7, 14, 20, 30, 50, 100, 200]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            df[f"price_to_sma_{period}"] = df["close"] / df[f"sma_{period}"]

        # MACD
        ema_12 = df["close"].ewm(span=12).mean()
        ema_26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # RSI (multiple periods)
        for period in [7, 14, 21, 28]:
            df[f"rsi_{period}"] = self._calculate_rsi(df["close"], period)

        # Bollinger Bands
        for period in [20, 50]:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"bb_upper_{period}"] = sma + (2 * std)
            df[f"bb_lower_{period}"] = sma - (2 * std)
            df[f"bb_width_{period}"] = (
                df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
            ) / sma
            df[f"bb_position_{period}"] = (df["close"] - df[f"bb_lower_{period}"]) / (
                df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"] + 1e-10
            )

        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df["low"].rolling(period).min()
            high_max = df["high"].rolling(period).max()
            df[f"stoch_{period}"] = (
                100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
            )
            df[f"stoch_signal_{period}"] = df[f"stoch_{period}"].rolling(3).mean()

        # ATR (Average True Range)
        for period in [14, 21]:
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
            df[f"atr_{period}"] = true_range.rolling(period).mean()
            df[f"atr_percent_{period}"] = df[f"atr_{period}"] / df["close"]

        # ADX (Average Directional Index)
        df["adx_14"] = self._calculate_adx(df, 14)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based features"""

        for period in self.lookback_periods:
            # Historical volatility
            df[f"volatility_{period}"] = df["close"].pct_change().rolling(period).std()

            # Parkinson volatility (uses high-low)
            df[f"parkinson_vol_{period}"] = np.sqrt(
                (1 / (4 * period * np.log(2)))
                * (np.log(df["high"] / df["low"]) ** 2).rolling(period).sum()
            )

            # Garman-Klass volatility
            df[f"gk_vol_{period}"] = np.sqrt(
                (
                    0.5 * (np.log(df["high"] / df["low"]) ** 2).rolling(period).mean()
                    - (2 * np.log(2) - 1)
                    * (np.log(df["close"] / df["open"]) ** 2).rolling(period).mean()
                )
            )

        # Volatility regime
        vol_14 = df["close"].pct_change().rolling(14).std()
        vol_60 = df["close"].pct_change().rolling(60).std()
        df["vol_regime"] = vol_14 / (vol_60 + 1e-10)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features"""

        if "volume" not in df.columns:
            return df

        # Volume moving averages
        for period in [7, 14, 30]:
            df[f"volume_sma_{period}"] = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}"] = df["volume"] / df[f"volume_sma_{period}"]

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_ema_20"] = df["obv"].ewm(span=20).mean()

        # Volume-Price Trend
        df["vpt"] = (df["volume"] * df["close"].pct_change()).cumsum()

        # Money Flow Index
        for period in [14, 20]:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            money_flow = typical_price * df["volume"]

            positive_flow = (
                money_flow.where(typical_price > typical_price.shift(), 0)
                .rolling(period)
                .sum()
            )
            negative_flow = (
                money_flow.where(typical_price < typical_price.shift(), 0)
                .rolling(period)
                .sum()
            )

            df[f"mfi_{period}"] = 100 - (
                100 / (1 + positive_flow / (negative_flow + 1e-10))
            )

        # Volume weighted average price
        df["vwap"] = (
            df["volume"] * (df["high"] + df["low"] + df["close"]) / 3
        ).cumsum() / df["volume"].cumsum()
        df["price_to_vwap"] = df["close"] / df["vwap"]

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indicators"""

        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (
                (df["close"] - df["close"].shift(period)) / df["close"].shift(period)
            ) * 100

        # Momentum
        for period in [10, 20]:
            df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)

        # Commodity Channel Index
        for period in [20]:
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            sma_tp = typical_price.rolling(period).mean()
            mad = (typical_price - sma_tp).abs().rolling(period).mean()
            df[f"cci_{period}"] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        # Williams %R
        for period in [14]:
            high_max = df["high"].rolling(period).max()
            low_min = df["low"].rolling(period).min()
            df[f"williams_r_{period}"] = (
                -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)
            )

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features"""

        for period in [14, 30]:
            # Skewness
            df[f"skew_{period}"] = df["close"].pct_change().rolling(period).skew()

            # Kurtosis
            df[f"kurtosis_{period}"] = df["close"].pct_change().rolling(period).kurt()

            # Z-score
            mean = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"zscore_{period}"] = (df["close"] - mean) / (std + 1e-10)

            # Percentile rank
            df[f"percentile_{period}"] = (
                df["close"]
                .rolling(period)
                .apply(
                    lambda x: (
                        pd.Series(x).rank().iloc[-1] / len(x) if len(x) > 0 else 0.5
                    )
                )
            )

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pattern recognition features"""

        # Candlestick patterns (simplified binary features)

        # Doji
        body = abs(df["close"] - df["open"])
        range_hl = df["high"] - df["low"]
        df["is_doji"] = (body / (range_hl + 1e-10) < 0.1).astype(int)

        # Hammer
        lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
        upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
        df["is_hammer"] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(
            int
        )

        # Engulfing patterns
        df["bullish_engulfing"] = (
            (df["close"] > df["open"])
            & (df["close"].shift() < df["open"].shift())
            & (df["open"] < df["close"].shift())
            & (df["close"] > df["open"].shift())
        ).astype(int)

        df["bearish_engulfing"] = (
            (df["close"] < df["open"])
            & (df["close"].shift() > df["open"].shift())
            & (df["open"] > df["close"].shift())
            & (df["close"] < df["open"].shift())
        ).astype(int)

        # Support/Resistance breaks
        for period in [20, 50]:
            df[f"resistance_{period}"] = df["high"].rolling(period).max()
            df[f"support_{period}"] = df["low"].rolling(period).min()
            df[f"broke_resistance_{period}"] = (
                df["close"] > df[f"resistance_{period}"].shift()
            ).astype(int)
            df[f"broke_support_{period}"] = (
                df["close"] < df[f"support_{period}"].shift()
            ).astype(int)

        # Trend detection
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        df["golden_cross"] = (
            (sma_20 > sma_50) & (sma_20.shift() <= sma_50.shift())
        ).astype(int)
        df["death_cross"] = (
            (sma_20 < sma_50) & (sma_20.shift() >= sma_50.shift())
        ).astype(int)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""

        if "timestamp" not in df.columns:
            return df

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter

        # Cyclical encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Is weekend
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Trading session (assuming UTC)
        df["is_asia_session"] = ((df["hour"] >= 0) & (df["hour"] < 8)).astype(int)
        df["is_europe_session"] = ((df["hour"] >= 8) & (df["hour"] < 16)).astype(int)
        df["is_us_session"] = ((df["hour"] >= 16) & (df["hour"] < 24)).astype(int)

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""

        # Spread approximation
        df["spread"] = df["high"] - df["low"]
        df["spread_percent"] = df["spread"] / df["close"]

        # Amihud illiquidity
        if "volume" in df.columns:
            df["amihud_illiquidity"] = abs(df["close"].pct_change()) / (
                df["volume"] + 1e-10
            )

        # Price efficiency (autocorrelation)
        for lag in [1, 5, 10]:
            df[f"autocorr_{lag}"] = (
                df["close"]
                .pct_change()
                .rolling(20)
                .apply(lambda x: x.autocorr(lag=lag) if len(x) > lag else 0)
            )

        # Hurst exponent (simplified)
        for period in [30]:
            df[f"hurst_{period}"] = (
                df["close"]
                .rolling(period)
                .apply(lambda x: self._calculate_hurst(x) if len(x) == period else 0.5)
            )

        return df

    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment analysis features

        Uses cached sentiment data to avoid API rate limits
        """
        if not self.sentiment_analyzer:
            return df

        # Detect symbol from data (assume BTC for now, can be extended)
        # In production, symbol should be passed as parameter
        symbol = "BTC"  # Default

        try:
            # Fetch sentiment (cached for 5 minutes)
            loop = asyncio.get_event_loop()
            sentiment = loop.run_until_complete(
                self.sentiment_analyzer.get_sentiment(symbol)
            )

            # Get sentiment features
            sentiment_features = self.sentiment_analyzer.get_sentiment_features(
                sentiment
            )

            # Add to dataframe (same value for all rows, as sentiment is point-in-time)
            for feature_name, value in sentiment_features.items():
                df[feature_name] = value

            logger.debug(f"Added {len(sentiment_features)} sentiment features")

        except Exception as e:
            logger.warning(f"Failed to add sentiment features: {e}")
            # Add default values
            df["sentiment_overall"] = 0.0
            df["sentiment_confidence"] = 0.0

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

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
        atr = tr.rolling(period).mean()
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)

        # ADX
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-10)
        adx = dx.rolling(period).mean()

        return adx

    def _calculate_hurst(self, ts: pd.Series) -> float:
        """Calculate Hurst exponent (simplified)"""
        try:
            lags = range(2, min(20, len(ts) // 2))
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

            if len(tau) < 2:
                return 0.5

            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature names (excluding original columns)"""
        original_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return [col for col in df.columns if col not in original_cols]

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category"""
        return {
            "price": [
                col
                for col in self.feature_names
                if any(x in col for x in ["return", "momentum", "price_position"])
            ],
            "technical": [
                col
                for col in self.feature_names
                if any(x in col for x in ["sma", "ema", "rsi", "macd", "bb_"])
            ],
            "volatility": [
                col for col in self.feature_names if "vol" in col or "atr" in col
            ],
            "volume": [
                col
                for col in self.feature_names
                if "volume" in col or "obv" in col or "mfi" in col
            ],
            "momentum": [
                col
                for col in self.feature_names
                if any(x in col for x in ["roc", "cci", "williams"])
            ],
            "statistical": [
                col
                for col in self.feature_names
                if any(x in col for x in ["skew", "kurt", "zscore"])
            ],
            "pattern": [
                col
                for col in self.feature_names
                if any(x in col for x in ["doji", "hammer", "engulfing", "cross"])
            ],
            "time": [
                col
                for col in self.feature_names
                if any(x in col for x in ["hour", "day", "month", "weekend", "session"])
            ],
            "sentiment": [
                col
                for col in self.feature_names
                if "sentiment" in col or "fear_greed" in col or "whale" in col
            ],
        }


def quick_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for quick feature engineering

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(df)


# Export
__all__ = ["FeatureEngineer", "quick_engineer_features"]
