#!/usr/bin/env python3
"""
View and manage historical data cache

Shows cache statistics, size, and provides utilities to clear cache if needed.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

def get_cache_stats(cache_dir: Path = Path("./data/historical_cache")):
    """Get statistics about the cache"""
    if not cache_dir.exists():
        return {
            'exists': False,
            'total_files': 0,
            'total_size_mb': 0,
            'cached_datasets': []
        }

    parquet_files = list(cache_dir.glob("*.parquet"))
    json_files = list(cache_dir.glob("*.json"))

    total_size = sum(f.stat().st_size for f in parquet_files)

    datasets = []
    for parquet_file in parquet_files:
        json_file = cache_dir / f"{parquet_file.stem}.json"

        file_info = {
            'cache_key': parquet_file.stem,
            'size_mb': parquet_file.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(parquet_file.stat().st_mtime).isoformat(),
            'has_metadata': json_file.exists()
        }

        # Try to load metadata
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                file_info['exchange'] = metadata.get('exchange', 'unknown')
                file_info['symbol'] = metadata.get('symbol', 'unknown')
                file_info['timeframe'] = metadata.get('timeframe', 'unknown')
                file_info['quality_score'] = metadata.get('quality_metrics', {}).get('quality_score', 0)
            except Exception as e:
                file_info['metadata_error'] = str(e)

        # Try to get row count from parquet (requires pandas)
        if PANDAS_AVAILABLE:
            try:
                df = pd.read_parquet(parquet_file)
                file_info['candles'] = len(df)
                if 'timestamp' in df.columns:
                    file_info['date_range'] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
            except Exception as e:
                file_info['read_error'] = str(e)
        else:
            file_info['candles'] = 'N/A (pandas not available)'

        datasets.append(file_info)

    return {
        'exists': True,
        'cache_dir': str(cache_dir),
        'total_files': len(parquet_files),
        'total_size_mb': total_size / (1024 * 1024),
        'cached_datasets': sorted(datasets, key=lambda x: x.get('modified', ''), reverse=True)
    }

def print_cache_stats(stats: dict):
    """Pretty print cache statistics"""
    print("\n" + "="*80)
    print("HISTORICAL DATA CACHE STATISTICS")
    print("="*80)

    if not stats['exists']:
        print("\n❌ Cache directory does not exist")
        print("   Cache will be created on first data fetch")
        print(f"   Location: ./data/historical_cache")
        return

    print(f"\n📁 Cache location: {stats['cache_dir']}")
    print(f"📊 Total datasets: {stats['total_files']}")
    print(f"💾 Total size: {stats['total_size_mb']:.2f} MB")

    if stats['cached_datasets']:
        print(f"\n{'='*80}")
        print("CACHED DATASETS (most recent first)")
        print("="*80)

        for i, dataset in enumerate(stats['cached_datasets'], 1):
            print(f"\n{i}. {dataset.get('exchange', 'unknown').upper()} - {dataset.get('symbol', 'unknown')}")
            print(f"   Timeframe: {dataset.get('timeframe', 'unknown')}")
            print(f"   Candles: {dataset.get('candles', 'unknown'):,}")
            print(f"   Quality: {dataset.get('quality_score', 0):.1f}/100")
            print(f"   Size: {dataset['size_mb']:.2f} MB")
            print(f"   Modified: {dataset['modified']}")
            if 'date_range' in dataset:
                print(f"   Range: {dataset['date_range']}")

    print("\n" + "="*80)

def clear_cache(cache_dir: Path = Path("./data/historical_cache"), confirm: bool = True):
    """Clear all cache files"""
    if not cache_dir.exists():
        print("Cache directory does not exist, nothing to clear")
        return

    parquet_files = list(cache_dir.glob("*.parquet"))
    json_files = list(cache_dir.glob("*.json"))
    total_files = len(parquet_files) + len(json_files)

    if total_files == 0:
        print("Cache is already empty")
        return

    if confirm:
        print(f"\n⚠️  WARNING: This will delete {total_files} cached files")
        response = input("Are you sure? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            return

    deleted = 0
    for f in parquet_files + json_files:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"Error deleting {f}: {e}")

    print(f"\n✓ Deleted {deleted} files from cache")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="View and manage historical data cache")
    parser.add_argument('--clear', action='store_true', help='Clear all cache files')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation when clearing')
    parser.add_argument('--cache-dir', type=str, default='./data/historical_cache',
                       help='Cache directory (default: ./data/historical_cache)')

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)

    if args.clear:
        clear_cache(cache_dir, confirm=not args.yes)
    else:
        stats = get_cache_stats(cache_dir)
        print_cache_stats(stats)

        if stats['exists'] and stats['total_files'] > 0:
            print("\nTIP: Run with --clear to delete cached files and force re-download")

if __name__ == "__main__":
    main()
