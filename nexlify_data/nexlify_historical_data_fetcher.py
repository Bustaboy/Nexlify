"""
Nexlify Historical Data Fetcher
Multi-source comprehensive historical data fetching for optimal ML/RL training

Features:
- Multi-exchange support (Binance, Coinbase, Kraken, Bitfinex, etc.)
- Multiple cryptocurrencies and timeframes
- Robust error handling with exponential backoff retry
- Data quality validation and preprocessing
- Local caching for faster retraining
- Progress tracking and resumption
- Rate limit handling
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Metrics for assessing data quality"""
    total_candles: int
    missing_candles: int
    duplicate_candles: int
    invalid_ohlc: int  # Where O/H/L/C relationships are invalid
    zero_volume_candles: int
    extreme_price_jumps: int  # Sudden jumps > 50%
    quality_score: float  # 0-100


@dataclass
class FetchConfig:
    """Configuration for data fetching"""
    exchange: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    cache_enabled: bool = True
    validate_quality: bool = True

    def get_cache_key(self) -> str:
        """Generate unique cache key for this configuration"""
        config_str = f"{self.exchange}_{self.symbol}_{self.timeframe}_{self.start_date}_{self.end_date}"
        return hashlib.md5(config_str.encode()).hexdigest()


class HistoricalDataFetcher:
    """
    Comprehensive historical data fetcher with multi-source support
    """

    SUPPORTED_EXCHANGES = ['binance', 'coinbase', 'kraken', 'bitfinex', 'bitstamp', 'huobi']

    # Timeframe mappings (exchange-specific formats)
    TIMEFRAMES = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        '1w': 604800
    }

    def __init__(self, cache_dir: str = "./data/historical_cache", automated_mode: bool = True):
        """
        Initialize the historical data fetcher

        Args:
            cache_dir: Directory for caching downloaded data
            automated_mode: If True, uses fallbacks and never blocks on user input
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.automated_mode = automated_mode

        # Initialize exchanges
        self.exchanges = {}
        self._initialize_exchanges()

        # Track fetch statistics
        self.fetch_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cached_requests': 0,
            'total_candles_fetched': 0,
            'exchanges_available': len(self.exchanges),
            'exchanges_failed': 0
        }

    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        for exchange_name in self.SUPPORTED_EXCHANGES:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchanges[exchange_name] = exchange_class({
                    'enableRateLimit': True,
                    'timeout': 30000,  # 30 second timeout
                })
                logger.info(f"✓ Initialized {exchange_name}")
            except Exception as e:
                logger.warning(f"✗ Could not initialize {exchange_name}: {e}")

    def fetch_historical_data(
        self,
        config: FetchConfig,
        max_retries: int = 5
    ) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """
        Fetch historical data with comprehensive error handling

        Args:
            config: Fetch configuration
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (DataFrame with OHLCV data, Quality metrics)
        """
        logger.info(f"Fetching {config.symbol} from {config.exchange} "
                   f"({config.timeframe}, {config.start_date} to {config.end_date})")

        # Check cache first
        if config.cache_enabled:
            cached_data = self._load_from_cache(config)
            if cached_data is not None:
                logger.info(f"✓ Loaded from cache")
                self.fetch_stats['cached_requests'] += 1
                quality_metrics = self._validate_data_quality(cached_data, config)
                return cached_data, quality_metrics

        # Fetch from exchange with error handling
        try:
            df = self._fetch_with_retry(config, max_retries)

            if df is not None and not df.empty:
                # Validate and clean data
                if config.validate_quality:
                    logger.info("Cleaning data...")
                    df = self._clean_data(df, config)

                    if df.empty:
                        logger.error(f"All data was filtered out during cleaning for {config.symbol}")
                        return pd.DataFrame(), DataQualityMetrics(0, 0, 0, 0, 0, 0, 0.0)

                    quality_metrics = self._validate_data_quality(df, config)
                    logger.info(f"Data quality score: {quality_metrics.quality_score:.1f}/100")
                else:
                    quality_metrics = DataQualityMetrics(
                        total_candles=len(df),
                        missing_candles=0,
                        duplicate_candles=0,
                        invalid_ohlc=0,
                        zero_volume_candles=0,
                        extreme_price_jumps=0,
                        quality_score=100.0
                    )

                # Save to cache
                if config.cache_enabled:
                    self._save_to_cache(df, config, quality_metrics)

                self.fetch_stats['total_candles_fetched'] += len(df)
                return df, quality_metrics
            else:
                logger.error(f"Failed to fetch data for {config.symbol}: No data returned from exchange")
                return pd.DataFrame(), DataQualityMetrics(0, 0, 0, 0, 0, 0, 0.0)

        except Exception as e:
            logger.error(f"Error fetching data for {config.symbol}: {e}")
            if self.automated_mode:
                logger.warning("Automated mode: returning empty DataFrame")
                return pd.DataFrame(), DataQualityMetrics(0, 0, 0, 0, 0, 0, 0.0)
            else:
                raise

    def _fetch_with_retry(
        self,
        config: FetchConfig,
        max_retries: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data with exponential backoff retry logic

        Args:
            config: Fetch configuration
            max_retries: Maximum retry attempts

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        exchange = self.exchanges.get(config.exchange)
        if not exchange:
            logger.error(f"Exchange {config.exchange} not available")
            return None

        # Check if exchange supports the symbol
        try:
            exchange.load_markets()
            if config.symbol not in exchange.markets:
                logger.error(f"{config.symbol} not available on {config.exchange}")
                return None
        except Exception as e:
            logger.warning(f"Could not load markets for {config.exchange}: {e}")

        all_candles = []
        current_start = config.start_date
        timeframe_seconds = self.TIMEFRAMES[config.timeframe]

        # Calculate how many candles we need to fetch
        total_seconds = (config.end_date - config.start_date).total_seconds()
        expected_candles = int(total_seconds / timeframe_seconds)

        logger.info(f"Expected ~{expected_candles} candles to fetch")

        with self._create_progress_bar(expected_candles) as pbar:
            while current_start < config.end_date:
                retry_count = 0
                success = False

                while retry_count < max_retries and not success:
                    try:
                        self.fetch_stats['total_requests'] += 1

                        # Convert datetime to timestamp in milliseconds
                        since = int(current_start.timestamp() * 1000)

                        # Fetch OHLCV data
                        ohlcv = exchange.fetch_ohlcv(
                            symbol=config.symbol,
                            timeframe=config.timeframe,
                            since=since,
                            limit=1000  # Most exchanges support 1000 candles per request
                        )

                        if not ohlcv:
                            logger.warning(f"No data returned for {current_start}")
                            break

                        # Convert to DataFrame
                        df_batch = pd.DataFrame(
                            ohlcv,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )

                        # Convert timestamp to datetime
                        df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'], unit='ms')

                        all_candles.append(df_batch)

                        # Update progress
                        pbar.update(len(df_batch))

                        # Move to next batch
                        last_timestamp = df_batch['timestamp'].iloc[-1]
                        current_start = last_timestamp + timedelta(seconds=timeframe_seconds)

                        self.fetch_stats['successful_requests'] += 1
                        success = True

                        # Small delay to respect rate limits
                        time.sleep(exchange.rateLimit / 1000)

                    except ccxt.RateLimitExceeded as e:
                        retry_count += 1
                        wait_time = min(2 ** retry_count, 60)  # Cap at 60 seconds
                        logger.warning(f"Rate limit exceeded, waiting {wait_time}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)

                    except ccxt.NetworkError as e:
                        retry_count += 1
                        wait_time = min(2 ** retry_count, 60)
                        logger.warning(f"Network error: {e}, retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)

                    except ccxt.ExchangeError as e:
                        logger.error(f"Exchange error: {e}")
                        self.fetch_stats['failed_requests'] += 1
                        break

                    except Exception as e:
                        retry_count += 1
                        logger.error(f"Unexpected error: {e}, attempt {retry_count}/{max_retries}")
                        if retry_count >= max_retries:
                            self.fetch_stats['failed_requests'] += 1
                            break
                        time.sleep(2 ** retry_count)

                if not success:
                    logger.warning(f"Failed to fetch data starting from {current_start}")
                    break

        # Combine all batches
        if all_candles:
            df = pd.concat(all_candles, ignore_index=True)
            raw_count = len(df)

            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['timestamp'])
            after_dedup_count = len(df)

            df = df.sort_values('timestamp').reset_index(drop=True)

            # Filter to requested date range
            df = df[
                (df['timestamp'] >= config.start_date) &
                (df['timestamp'] <= config.end_date)
            ]
            final_count = len(df)

            # Log detailed information
            logger.info(f"✓ Fetched {raw_count} raw candles, removed {raw_count - after_dedup_count} duplicates, "
                       f"filtered to {final_count} candles in date range")

            if final_count == 0 and raw_count > 0:
                logger.warning(f"All {raw_count} fetched candles were outside the requested date range "
                             f"({config.start_date} to {config.end_date})")

            return df

        logger.warning("No candles were successfully fetched from any batch")
        return None

    def _create_progress_bar(self, total: int):
        """Create a simple progress tracker"""
        class SimpleProgress:
            def __init__(self, total):
                self.total = total
                self.current = 0
                self.last_log_percent = 0

            def update(self, n):
                self.current += n
                if self.total > 0:
                    progress = min(100, (self.current / self.total) * 100)
                    # Only log every 10% to reduce noise
                    if progress - self.last_log_percent >= 10 or progress >= 100:
                        logger.info(f"Progress: {progress:.1f}% ({self.current}/{self.total} candles fetched in batches)")
                        self.last_log_percent = progress

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return SimpleProgress(total)

    def _clean_data(self, df: pd.DataFrame, config: FetchConfig) -> pd.DataFrame:
        """
        Clean and validate OHLCV data

        Args:
            df: Raw OHLCV DataFrame
            config: Fetch configuration

        Returns:
            Cleaned DataFrame
        """
        original_count = len(df)

        # Remove rows with any NaN values
        df = df.dropna()

        # Validate OHLC relationships
        valid_ohlc = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        df = df[valid_ohlc]

        # Remove zero or negative prices
        df = df[
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0) &
            (df['volume'] >= 0)
        ]

        # Handle extreme price jumps (> 50% in one candle)
        # Calculate price change
        df['price_change'] = df['close'].pct_change().abs()
        extreme_jumps = df['price_change'] > 0.5

        if extreme_jumps.sum() > 0:
            logger.warning(f"Found {extreme_jumps.sum()} extreme price jumps (>50%), investigating...")
            # For extreme jumps, we'll keep them but flag for review
            # In a production system, you might want to validate these against multiple sources

        df = df.drop(columns=['price_change'])

        cleaned_count = len(df)
        removed_count = original_count - cleaned_count

        if removed_count > 0:
            logger.info(f"Removed {removed_count} invalid candles ({removed_count/original_count*100:.1f}%)")

        return df

    def _validate_data_quality(
        self,
        df: pd.DataFrame,
        config: FetchConfig
    ) -> DataQualityMetrics:
        """
        Validate data quality and generate metrics

        Args:
            df: OHLCV DataFrame
            config: Fetch configuration

        Returns:
            Data quality metrics
        """
        if df.empty:
            return DataQualityMetrics(0, 0, 0, 0, 0, 0, 0.0)

        total_candles = len(df)

        # Check for missing candles
        timeframe_seconds = self.TIMEFRAMES[config.timeframe]
        expected_candles = int(
            (config.end_date - config.start_date).total_seconds() / timeframe_seconds
        )
        missing_candles = max(0, expected_candles - total_candles)

        # Check for duplicates
        duplicate_candles = df.duplicated(subset=['timestamp']).sum()

        # Check OHLC validity
        invalid_ohlc = (
            ~(
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            )
        ).sum()

        # Check for zero volume
        zero_volume_candles = (df['volume'] == 0).sum()

        # Check for extreme price jumps
        price_changes = df['close'].pct_change().abs()
        extreme_price_jumps = (price_changes > 0.5).sum()

        # Calculate quality score (0-100)
        penalties = 0
        penalties += (missing_candles / max(1, expected_candles)) * 30  # Max 30 points
        penalties += (duplicate_candles / max(1, total_candles)) * 20  # Max 20 points
        penalties += (invalid_ohlc / max(1, total_candles)) * 25  # Max 25 points
        penalties += (zero_volume_candles / max(1, total_candles)) * 15  # Max 15 points
        penalties += (extreme_price_jumps / max(1, total_candles)) * 10  # Max 10 points

        quality_score = max(0, 100 - penalties)

        return DataQualityMetrics(
            total_candles=int(total_candles),
            missing_candles=int(missing_candles),
            duplicate_candles=int(duplicate_candles),
            invalid_ohlc=int(invalid_ohlc),
            zero_volume_candles=int(zero_volume_candles),
            extreme_price_jumps=int(extreme_price_jumps),
            quality_score=float(quality_score)
        )

    def _load_from_cache(self, config: FetchConfig) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_key = config.get_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")

        return None

    def _save_to_cache(
        self,
        df: pd.DataFrame,
        config: FetchConfig,
        quality_metrics: DataQualityMetrics
    ):
        """Save data to cache"""
        try:
            cache_key = config.get_cache_key()
            cache_file = self.cache_dir / f"{cache_key}.parquet"

            # Save DataFrame
            df.to_parquet(cache_file, index=False)

            # Convert quality metrics to dict and ensure all values are JSON serializable
            quality_metrics_dict = asdict(quality_metrics)
            # Convert numpy types to native Python types
            for key, value in quality_metrics_dict.items():
                if hasattr(value, 'item'):  # numpy scalar
                    quality_metrics_dict[key] = value.item()
                elif isinstance(value, (np.integer, np.int64)):
                    quality_metrics_dict[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    quality_metrics_dict[key] = float(value)

            # Save metadata
            metadata = {
                'config': {
                    'exchange': config.exchange,
                    'symbol': config.symbol,
                    'timeframe': config.timeframe,
                    'start_date': config.start_date.isoformat(),
                    'end_date': config.end_date.isoformat()
                },
                'quality_metrics': quality_metrics_dict,
                'cached_at': datetime.now().isoformat()
            }

            metadata_file = self.cache_dir / f"{cache_key}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✓ Saved to cache: {cache_file}")

        except Exception as e:
            logger.warning(f"Could not save to cache: {e}")

    def fetch_multi_symbol(
        self,
        exchange: str,
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        cache_enabled: bool = True
    ) -> Dict[str, Tuple[pd.DataFrame, DataQualityMetrics]]:
        """
        Fetch data for multiple symbols

        Args:
            exchange: Exchange name
            symbols: List of trading pairs
            timeframe: Candle timeframe
            start_date: Start date
            end_date: End date
            cache_enabled: Whether to use cache

        Returns:
            Dictionary mapping symbol to (DataFrame, metrics)
        """
        results = {}

        logger.info(f"Fetching {len(symbols)} symbols from {exchange}")

        for symbol in symbols:
            config = FetchConfig(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                cache_enabled=cache_enabled
            )

            df, metrics = self.fetch_historical_data(config)
            results[symbol] = (df, metrics)

            # Small delay between symbols
            time.sleep(1)

        return results

    def get_statistics(self) -> Dict:
        """Get fetcher statistics"""
        stats = self.fetch_stats.copy()

        if stats['total_requests'] > 0:
            stats['success_rate'] = (
                stats['successful_requests'] / stats['total_requests'] * 100
            )
            stats['cache_hit_rate'] = (
                stats['cached_requests'] / stats['total_requests'] * 100
            )
        else:
            stats['success_rate'] = 0
            stats['cache_hit_rate'] = 0

        return stats

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache files

        Args:
            older_than_days: Only clear files older than this many days (None = all)
        """
        files_removed = 0

        for cache_file in self.cache_dir.glob("*"):
            should_remove = True

            if older_than_days is not None:
                file_age_days = (
                    datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                ).days
                should_remove = file_age_days > older_than_days

            if should_remove:
                cache_file.unlink()
                files_removed += 1

        logger.info(f"Removed {files_removed} cache files")


# Convenience function for quick data fetching
def fetch_data(
    exchange: str = 'binance',
    symbol: str = 'BTC/USDT',
    timeframe: str = '1h',
    days: int = 365,
    end_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, DataQualityMetrics]:
    """
    Quick helper to fetch historical data

    Args:
        exchange: Exchange name
        symbol: Trading pair
        timeframe: Candle timeframe
        days: Number of days of history
        end_date: End date (default: now)

    Returns:
        Tuple of (DataFrame, quality metrics)
    """
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=days)

    fetcher = HistoricalDataFetcher()
    config = FetchConfig(
        exchange=exchange,
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )

    return fetcher.fetch_historical_data(config)


if __name__ == "__main__":
    # Example usage
    print("Nexlify Historical Data Fetcher")
    print("=" * 60)

    # Fetch 2 years of BTC data
    fetcher = HistoricalDataFetcher()

    config = FetchConfig(
        exchange='binance',
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime.now()
    )

    df, metrics = fetcher.fetch_historical_data(config)

    print(f"\nFetched {len(df)} candles")
    print(f"Quality score: {metrics.quality_score:.1f}/100")
    print(f"\nData preview:")
    print(df.head())
    print(f"\nStatistics:")
    print(json.dumps(fetcher.get_statistics(), indent=2))
