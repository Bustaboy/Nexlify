#!/usr/bin/env python3
"""
Quick test script for historical data fetcher improvements
"""

from datetime import datetime, timedelta
from nexlify_data.nexlify_historical_data_fetcher import HistoricalDataFetcher, FetchConfig

# Test with Kraken BTC/USD (as per the user's logs)
fetcher = HistoricalDataFetcher(automated_mode=True)

# Try fetching with a realistic date range (last 30 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

config = FetchConfig(
    exchange='kraken',
    symbol='BTC/USD',
    timeframe='1h',
    start_date=start_date,
    end_date=end_date,
    cache_enabled=False  # Disable cache for testing
)

print(f"Testing fetch from {start_date} to {end_date}")
print("="*80)

df, quality = fetcher.fetch_historical_data(config)

print("\n" + "="*80)
print("RESULTS:")
print(f"Candles fetched: {len(df)}")
if len(df) > 0:
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Quality score: {quality.quality_score:.1f}/100")
    print(f"Missing candles: {quality.missing_candles}")
    print(f"\nFirst 5 candles:")
    print(df.head())
else:
    print("No data fetched!")

print("="*80)
