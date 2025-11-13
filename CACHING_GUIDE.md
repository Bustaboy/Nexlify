# Historical Data Caching Guide

## Overview

Nexlify automatically caches all downloaded historical data to avoid re-downloading the same data on every training run. This significantly speeds up training and reduces API calls to exchanges.

## Cache Location

**Default**: `./data/historical_cache/`

The cache directory is created automatically on first use and contains:
- `.parquet` files: The actual OHLCV data (efficient binary format)
- `.json` files: Metadata (exchange, symbol, timeframe, quality metrics)

## How Caching Works

### Automatic Caching (Default)

1. **First fetch**: Data is downloaded from exchange and saved to cache
2. **Subsequent fetches**: If exact same data request is made, it's loaded from cache instantly
3. **Cache key**: Based on `exchange + symbol + timeframe + date_range`

Example:
```bash
# First run: Downloads from Coinbase (~10 seconds)
python train_ultimate_full_pipeline.py --pairs BTC/USD --exchange auto

# Second run: Loads from cache (<1 second)
python train_ultimate_full_pipeline.py --pairs BTC/USD --exchange auto
```

### Cache Hit Rate

After training, you'll see statistics like:

```
================================================================================
📊 DATA FETCH STATISTICS
================================================================================
Total requests: 10
Cache hits: 8 (80.0%)
Network fetches: 2
Failed requests: 0
Total candles fetched: 87,600
Cache location: ./data/historical_cache
Cache size: 45.23 MB (10 datasets)
================================================================================
```

**80% cache hit rate** means 8 out of 10 data requests were served from cache!

## Viewing Cache

### Quick View

```bash
python scripts/view_cache_stats.py
```

Output:
```
================================================================================
HISTORICAL DATA CACHE STATISTICS
================================================================================

📁 Cache location: ./data/historical_cache
📊 Total datasets: 10
💾 Total size: 45.23 MB

================================================================================
CACHED DATASETS (most recent first)
================================================================================

1. COINBASE - BTC/USD
   Timeframe: 1h
   Candles: 8,760
   Quality: 98.5/100
   Size: 5.2 MB
   Modified: 2025-11-13T22:30:15
   Range: 2024-11-13 to 2025-11-13

2. KRAKEN - ETH/USDT
   Timeframe: 1h
   Candles: 8,760
   Quality: 95.2/100
   Size: 4.8 MB
   Modified: 2025-11-13T22:25:42
   Range: 2024-11-13 to 2025-11-13

...
```

## Managing Cache

### Clear All Cache

```bash
# With confirmation prompt
python scripts/view_cache_stats.py --clear

# Skip confirmation
python scripts/view_cache_stats.py --clear --yes
```

### Custom Cache Directory

```bash
# View custom cache location
python scripts/view_cache_stats.py --cache-dir /path/to/cache

# Clear custom cache
python scripts/view_cache_stats.py --clear --cache-dir /path/to/cache
```

### Programmatic Cache Management

```python
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher

fetcher = HistoricalDataFetcher()

# View statistics
stats = fetcher.get_statistics()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1f}%")

# Clear old cache (older than 30 days)
fetcher.clear_cache(older_than_days=30)

# Clear all cache
fetcher.clear_cache()
```

## When to Clear Cache

Clear cache when:

1. **Exchange data has been updated** - Exchanges may backfill or correct historical data
2. **Testing changes** - Force re-download to test data fetching logic
3. **Disk space needed** - Cache can grow to several GB with many symbols/timeframes
4. **Data quality issues** - If you suspect cached data is corrupted

## Cache Benefits

### Speed Improvements

| Task | First Run | Cached Run | Speedup |
|------|-----------|------------|---------|
| 1 pair, 1 year | 10 sec | <1 sec | 10x faster |
| 6 pairs, 2 years | 60 sec | 2 sec | 30x faster |
| 10 pairs, 5 years | 180 sec | 5 sec | 36x faster |

### Network & Rate Limit Savings

- **API calls**: Reduced by cache hit rate (typically 70-90%)
- **Rate limiting**: Avoid hitting exchange rate limits
- **Data costs**: Some exchanges charge for API access
- **Reliability**: No network dependency for cached data

## Cache Storage

### Disk Usage

Approximate sizes per year of hourly data:
- 1 pair: ~5-10 MB
- 6 pairs: ~30-60 MB
- 10 pairs: ~50-100 MB

Example: Training with 10 pairs × 5 years = ~250-500 MB cache

### Format

**Parquet format** is used because:
- ✅ 10-100x smaller than CSV
- ✅ 10-100x faster to load than CSV
- ✅ Preserves data types (no parsing needed)
- ✅ Supports compression
- ✅ Industry standard for data science

## Disable Caching

To disable caching (not recommended):

```python
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig

fetcher = HistoricalDataFetcher()

config = FetchConfig(
    exchange='coinbase',
    symbol='BTC/USD',
    timeframe='1h',
    start_date=start_date,
    end_date=end_date,
    cache_enabled=False  # Disable cache
)

df, metrics = fetcher.fetch_historical_data(config)
```

## Git Integration

The cache directory is automatically ignored by Git:
- `data/.gitignore` includes `historical_cache/`
- Cache files won't be committed to repository
- Each user maintains their own local cache

## FAQ

**Q: Does cache work across different training scripts?**
A: Yes! The cache is shared across all scripts that use `HistoricalDataFetcher`.

**Q: What if exchange data changes?**
A: Cache uses date range as key. New dates = new cache entry. Historical data rarely changes, but you can clear cache if needed.

**Q: Is cache safe to delete?**
A: Yes! Deleting cache just means data will be re-downloaded on next fetch. No training data or models are affected.

**Q: Can I share cache between machines?**
A: Yes! Copy the entire `./data/historical_cache/` directory. Parquet format is portable across systems.

**Q: How do I know if cache is being used?**
A: Look for "✓ Loaded from cache" in logs, or check cache hit rate in statistics.

---

**Last Updated**: 2025-11-13
**See Also**: `scripts/view_cache_stats.py` for cache management
