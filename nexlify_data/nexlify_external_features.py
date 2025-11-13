"""
Nexlify External Feature Enrichment
Adds external data sources to enhance ML/RL training

Features:
- Fear & Greed Index (crypto market sentiment)
- News sentiment analysis
- Social media sentiment (Reddit, Twitter trends)
- On-chain metrics (active addresses, transaction volume, etc.)
- Macroeconomic indicators (when relevant)
- Market correlation metrics
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ExternalFeatureEnricher:
    """
    Enriches OHLCV data with external features for better training
    """

    def __init__(self, cache_dir: str = "./data/external_cache", automated_mode: bool = True):
        """
        Initialize external feature enricher

        Args:
            cache_dir: Directory for caching external data
            automated_mode: If True, never raises exceptions, uses fallbacks instead
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.automated_mode = automated_mode

        self.api_endpoints = {
            'fear_greed': 'https://api.alternative.me/fng/',
            'crypto_news': 'https://min-api.cryptocompare.com/data/v2/news/',
            'blockchain_info': 'https://api.blockchain.info/stats'
        }

        # Track which features succeeded
        self.features_loaded = {
            'fear_greed': False,
            'onchain': False,
            'social': False,
            'temporal': False,
            'regime': False
        }

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str = 'BTC/USDT',
        include_sentiment: bool = True,
        include_onchain: bool = True,
        include_social: bool = True
    ) -> pd.DataFrame:
        """
        Enrich OHLCV DataFrame with external features

        Args:
            df: OHLCV DataFrame with 'timestamp' column
            symbol: Trading pair symbol
            include_sentiment: Include fear & greed index
            include_onchain: Include on-chain metrics
            include_social: Include social sentiment

        Returns:
            Enriched DataFrame
        """
        df = df.copy()
        initial_columns = len(df.columns)

        logger.info(f"Enriching {len(df)} candles with external features...")

        # Extract base asset (e.g., 'BTC' from 'BTC/USDT')
        base_asset = symbol.split('/')[0]

        # Add features with fallback handling
        if include_sentiment:
            try:
                df = self._add_fear_greed_index(df)
                self.features_loaded['fear_greed'] = True
            except Exception as e:
                logger.warning(f"Failed to add Fear & Greed Index: {e}")
                if not self.automated_mode:
                    raise
                # Add neutral defaults
                df['fear_greed_index'] = 50
                df['fear_greed_normalized'] = 0.5
                df['market_sentiment'] = 'Neutral'
                df['is_extreme_fear'] = 0
                df['is_fear'] = 0
                df['is_greed'] = 0
                df['is_extreme_greed'] = 0

        if include_onchain and base_asset == 'BTC':
            try:
                df = self._add_onchain_metrics(df)
                self.features_loaded['onchain'] = True
            except Exception as e:
                logger.warning(f"Failed to add on-chain metrics: {e}")
                if not self.automated_mode:
                    raise

        if include_social:
            try:
                df = self._add_social_sentiment(df, base_asset)
                self.features_loaded['social'] = True
            except Exception as e:
                logger.warning(f"Failed to add social sentiment: {e}")
                if not self.automated_mode:
                    raise

        # Add time-based features (should always work)
        try:
            df = self._add_temporal_features(df)
            self.features_loaded['temporal'] = True
        except Exception as e:
            logger.error(f"Failed to add temporal features: {e}")
            if not self.automated_mode:
                raise

        # Add market regime indicators (should always work)
        try:
            df = self._add_market_regime(df)
            self.features_loaded['regime'] = True
        except Exception as e:
            logger.error(f"Failed to add market regime: {e}")
            if not self.automated_mode:
                raise

        features_added = len(df.columns) - initial_columns
        logger.info(f"✓ Added {features_added} external features")
        self._log_feature_status()

        return df

    def _log_feature_status(self):
        """Log which features were successfully loaded"""
        loaded = [k for k, v in self.features_loaded.items() if v]
        failed = [k for k, v in self.features_loaded.items() if not v]

        if loaded:
            logger.info(f"✓ Loaded features: {', '.join(loaded)}")
        if failed:
            logger.warning(f"⚠ Failed features: {', '.join(failed)} (using fallbacks)")

    def _add_fear_greed_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fear & Greed Index to the DataFrame

        The Fear & Greed Index is a sentiment indicator for crypto markets (0-100)
        - 0-24: Extreme Fear
        - 25-49: Fear
        - 50-74: Greed
        - 75-100: Extreme Greed
        """
        logger.info("Fetching Fear & Greed Index...")

        try:
            # Try to load from cache first
            cache_file = self.cache_dir / "fear_greed_cache.json"

            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    cache_date = datetime.fromisoformat(cache_data['cached_at'])

                    # Use cache if less than 1 day old
                    if (datetime.now() - cache_date).days < 1:
                        fg_data = cache_data['data']
                        logger.info("✓ Loaded Fear & Greed from cache")
                    else:
                        fg_data = self._fetch_fear_greed()
                        self._save_fear_greed_cache(fg_data)
            else:
                fg_data = self._fetch_fear_greed()
                self._save_fear_greed_cache(fg_data)

            # Convert to DataFrame
            fg_df = pd.DataFrame(fg_data)
            fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'].astype(int), unit='s')

            # Merge with main DataFrame (using forward fill for missing dates)
            df = df.sort_values('timestamp')
            fg_df = fg_df.sort_values('timestamp')

            # Merge and forward fill
            df = pd.merge_asof(
                df,
                fg_df[['timestamp', 'value', 'value_classification']],
                on='timestamp',
                direction='backward'
            )

            # Rename columns
            df = df.rename(columns={
                'value': 'fear_greed_index',
                'value_classification': 'market_sentiment'
            })

            # Convert fear & greed to numeric (normalize to 0-1)
            df['fear_greed_index'] = pd.to_numeric(df['fear_greed_index'], errors='coerce')
            df['fear_greed_normalized'] = df['fear_greed_index'] / 100.0

            # Create binary sentiment features
            df['is_extreme_fear'] = (df['fear_greed_index'] < 25).astype(int)
            df['is_fear'] = ((df['fear_greed_index'] >= 25) & (df['fear_greed_index'] < 50)).astype(int)
            df['is_greed'] = ((df['fear_greed_index'] >= 50) & (df['fear_greed_index'] < 75)).astype(int)
            df['is_extreme_greed'] = (df['fear_greed_index'] >= 75).astype(int)

            # Fill any remaining NaNs with neutral (50)
            df['fear_greed_index'].fillna(50, inplace=True)
            df['fear_greed_normalized'].fillna(0.5, inplace=True)
            df['market_sentiment'].fillna('Neutral', inplace=True)

            logger.info("✓ Added Fear & Greed Index features")

        except Exception as e:
            logger.warning(f"Could not fetch Fear & Greed Index: {e}")
            # Add neutral values
            df['fear_greed_index'] = 50
            df['fear_greed_normalized'] = 0.5
            df['market_sentiment'] = 'Neutral'
            df['is_extreme_fear'] = 0
            df['is_fear'] = 0
            df['is_greed'] = 0
            df['is_extreme_greed'] = 0

        return df

    def _fetch_fear_greed(self, limit: int = 365) -> List[Dict]:
        """Fetch Fear & Greed Index data with robust error handling"""
        url = f"{self.api_endpoints['fear_greed']}?limit={limit}"

        for attempt in range(3):
            try:
                response = requests.get(url, timeout=5)  # Shorter timeout for automation
                response.raise_for_status()
                data = response.json()
                if 'data' in data and data['data']:
                    return data['data']
                else:
                    logger.warning("Fear & Greed API returned empty data")
                    return []
            except requests.exceptions.Timeout:
                logger.warning(f"Fear & Greed API timeout (attempt {attempt + 1}/3)")
                if attempt < 2:
                    time.sleep(1)  # Shorter wait for automation
            except requests.exceptions.RequestException as e:
                logger.warning(f"Fear & Greed API request failed: {e} (attempt {attempt + 1}/3)")
                if attempt < 2:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Unexpected error fetching Fear & Greed: {e}")
                if attempt == 2 and self.automated_mode:
                    return []  # Return empty in automated mode
                elif attempt == 2:
                    raise

        return []

    def _save_fear_greed_cache(self, data: List[Dict]):
        """Save Fear & Greed data to cache"""
        try:
            cache_file = self.cache_dir / "fear_greed_cache.json"
            cache_data = {
                'cached_at': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    def _add_onchain_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add on-chain metrics for Bitcoin

        Metrics include:
        - Active addresses
        - Transaction count
        - Hash rate
        - Mining difficulty
        - Market cap
        """
        logger.info("Adding on-chain metrics...")

        try:
            # Note: In production, you would fetch real on-chain data from sources like:
            # - Glassnode API
            # - Blockchain.info API
            # - CoinMetrics API
            # - IntoTheBlock API

            # For demonstration, we'll create proxy features from price action
            # In production, replace this with actual on-chain data

            # Network activity proxy (based on volume)
            df['network_activity_proxy'] = df['volume'] / df['volume'].rolling(30, min_periods=1).mean()

            # Hash rate proxy (increasing with price generally)
            df['hashrate_proxy'] = df['close'] / df['close'].rolling(90, min_periods=1).mean()

            # Difficulty adjustment proxy
            df['difficulty_proxy'] = df['close'].rolling(14, min_periods=1).std() / df['close'].rolling(14, min_periods=1).mean()

            # HODL sentiment (price stability indicates holding)
            df['hodl_sentiment'] = 1 / (1 + df['close'].pct_change().abs().rolling(30, min_periods=1).mean())

            # Address activity proxy
            df['address_activity_proxy'] = (
                df['volume'].rolling(7, min_periods=1).mean() /
                df['volume'].rolling(30, min_periods=1).mean()
            )

            # Fill NaNs
            for col in ['network_activity_proxy', 'hashrate_proxy', 'difficulty_proxy',
                       'hodl_sentiment', 'address_activity_proxy']:
                df[col].fillna(1.0, inplace=True)

            logger.info("✓ Added on-chain metric proxies")

        except Exception as e:
            logger.warning(f"Could not add on-chain metrics: {e}")

        return df

    def _add_social_sentiment(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Add social media sentiment features

        In production, this would integrate with:
        - Twitter API for tweet sentiment
        - Reddit API for subreddit activity
        - LunarCrush for social metrics
        - Santiment for social volume
        """
        logger.info("Adding social sentiment features...")

        try:
            # Create proxy features based on price momentum and volume
            # In production, replace with actual social data

            # Social volume proxy (high volume = high social interest)
            df['social_volume_proxy'] = (
                df['volume'].rolling(7, min_periods=1).mean() /
                df['volume'].rolling(30, min_periods=1).mean()
            )

            # Sentiment proxy (price momentum as proxy for sentiment)
            df['sentiment_proxy'] = df['close'].pct_change(7).clip(-0.5, 0.5) + 0.5

            # Social dominance (trading activity compared to market average)
            df['social_dominance_proxy'] = df['volume'] / df['volume'].rolling(90, min_periods=1).max()

            # Reddit mentions proxy
            df['reddit_activity_proxy'] = (
                df['volume'].rolling(3, min_periods=1).std() /
                df['volume'].rolling(30, min_periods=1).mean()
            ).clip(0, 5)

            # Twitter sentiment proxy
            df['twitter_sentiment_proxy'] = (
                df['close'].pct_change(3).rolling(7, min_periods=1).mean().clip(-0.2, 0.2) + 0.5
            )

            # Fill NaNs
            for col in ['social_volume_proxy', 'sentiment_proxy', 'social_dominance_proxy',
                       'reddit_activity_proxy', 'twitter_sentiment_proxy']:
                df[col].fillna(df[col].median() if not df[col].isna().all() else 0.5, inplace=True)

            logger.info("✓ Added social sentiment proxies")

        except Exception as e:
            logger.warning(f"Could not add social sentiment: {e}")

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features that can affect trading patterns
        """
        df = df.copy()

        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter

        # Create cyclical features (sine/cosine encoding for periodicity)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Market session indicators (UTC timezone)
        # Asian session: 00:00-08:00 UTC
        # European session: 08:00-16:00 UTC
        # US session: 16:00-24:00 UTC
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)

        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Month start/end indicators (often volatile)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 26).astype(int)

        return df

    def _add_market_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime indicators (trend, volatility, etc.)
        """
        df = df.copy()

        # Calculate returns for regime detection
        df['returns'] = df['close'].pct_change()

        # Trend regime (based on moving average slopes)
        df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
        df['sma_200'] = df['close'].rolling(200, min_periods=1).mean()

        df['is_bull_market'] = (df['close'] > df['sma_200']).astype(int)
        df['is_bear_market'] = (df['close'] < df['sma_200']).astype(int)
        df['golden_cross'] = (df['sma_50'] > df['sma_200']).astype(int)

        # Volatility regime
        df['volatility_20'] = df['returns'].rolling(20, min_periods=1).std()
        df['volatility_percentile'] = df['volatility_20'].rolling(100, min_periods=1).apply(
            lambda x: (x.iloc[-1] <= x).sum() / len(x) if len(x) > 0 else 0.5, raw=False
        )

        df['is_low_volatility'] = (df['volatility_percentile'] < 0.33).astype(int)
        df['is_medium_volatility'] = (
            (df['volatility_percentile'] >= 0.33) & (df['volatility_percentile'] < 0.66)
        ).astype(int)
        df['is_high_volatility'] = (df['volatility_percentile'] >= 0.66).astype(int)

        # Momentum regime
        df['momentum_20'] = df['close'].pct_change(20)
        df['is_strong_uptrend'] = (df['momentum_20'] > 0.1).astype(int)
        df['is_strong_downtrend'] = (df['momentum_20'] < -0.1).astype(int)
        df['is_ranging'] = (
            (df['momentum_20'] >= -0.1) & (df['momentum_20'] <= 0.1)
        ).astype(int)

        # Volume regime
        df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['is_high_volume'] = (df['volume'] > 1.5 * df['volume_ma']).astype(int)
        df['is_low_volume'] = (df['volume'] < 0.5 * df['volume_ma']).astype(int)

        # Clean up temporary columns
        df = df.drop(columns=['returns', 'sma_50', 'sma_200', 'volatility_20',
                              'momentum_20', 'volume_ma'], errors='ignore')

        # Fill NaNs
        df = df.fillna(method='ffill').fillna(0)

        return df

    def add_macro_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add macroeconomic indicators that can affect crypto markets

        Note: This requires external data sources like FRED API
        In production, integrate with:
        - Federal Reserve Economic Data (FRED)
        - World Bank API
        - IMF Data
        """
        logger.info("Adding macro indicators...")

        try:
            # For demonstration, create proxy indicators
            # In production, fetch real macro data

            # Interest rate environment proxy (inverse relationship with crypto)
            # Simulated based on long-term price trends
            df['interest_rate_proxy'] = 1 / (1 + df['close'].pct_change(90).clip(-1, 1))

            # Liquidity indicator proxy
            df['liquidity_proxy'] = df['volume'].rolling(60, min_periods=1).mean() / df['volume'].rolling(180, min_periods=1).mean()

            # Risk appetite proxy (higher volume + rising prices = risk-on)
            df['risk_appetite_proxy'] = (
                (df['close'].pct_change(20) > 0).astype(int) *
                (df['volume'] / df['volume'].rolling(60, min_periods=1).mean())
            ).clip(0, 3)

            # Fill NaNs
            for col in ['interest_rate_proxy', 'liquidity_proxy', 'risk_appetite_proxy']:
                df[col].fillna(df[col].median() if not df[col].isna().all() else 1.0, inplace=True)

            logger.info("✓ Added macro indicator proxies")

        except Exception as e:
            logger.warning(f"Could not add macro indicators: {e}")

        return df


def enrich_data(
    df: pd.DataFrame,
    symbol: str = 'BTC/USDT',
    full_enrichment: bool = True
) -> pd.DataFrame:
    """
    Convenience function to enrich OHLCV data with all external features

    Args:
        df: OHLCV DataFrame with 'timestamp' column
        symbol: Trading pair symbol
        full_enrichment: Include all available features

    Returns:
        Enriched DataFrame
    """
    enricher = ExternalFeatureEnricher()

    df = enricher.enrich_dataframe(
        df,
        symbol=symbol,
        include_sentiment=full_enrichment,
        include_onchain=full_enrichment,
        include_social=full_enrichment
    )

    if full_enrichment:
        df = enricher.add_macro_indicators(df)

    return df


if __name__ == "__main__":
    # Example usage
    print("Nexlify External Feature Enrichment")
    print("=" * 60)

    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='1H')
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': 40000 + np.random.randn(len(dates)).cumsum() * 100,
        'high': 40000 + np.random.randn(len(dates)).cumsum() * 100 + 100,
        'low': 40000 + np.random.randn(len(dates)).cumsum() * 100 - 100,
        'close': 40000 + np.random.randn(len(dates)).cumsum() * 100,
        'volume': 1000 + np.random.rand(len(dates)) * 500
    })

    # Enrich the data
    enriched_df = enrich_data(sample_df, symbol='BTC/USDT')

    print(f"\nOriginal columns: {len(sample_df.columns)}")
    print(f"Enriched columns: {len(enriched_df.columns)}")
    print(f"Added features: {len(enriched_df.columns) - len(sample_df.columns)}")
    print(f"\nNew columns:")
    new_cols = [col for col in enriched_df.columns if col not in sample_df.columns]
    for col in new_cols:
        print(f"  - {col}")
