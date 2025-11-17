#!/usr/bin/env python3
"""
Quick test script for historical data fetcher improvements
"""

import os
from datetime import datetime, timedelta

import pytest

from nexlify_data.nexlify_historical_data_fetcher import (
    FetchConfig,
    HistoricalDataFetcher,
)


# This module was originally a manual script. It now serves as an integration test that
# is skipped by default to avoid slow, network-heavy collection. Enable by running
# pytest with `-m requires_network` and setting NEXLIFY_ENABLE_NETWORK_TESTS=1.
pytestmark = [pytest.mark.requires_network]

if not os.environ.get("NEXLIFY_ENABLE_NETWORK_TESTS"):
    pytest.skip(
        "Network-dependent smoke test disabled by default. Set "
        "NEXLIFY_ENABLE_NETWORK_TESTS=1 to enable.",
        allow_module_level=True,
    )


def test_fetch_historical_data_smoke():
    """Smoke test for historical data retrieval (requires network).

    This ensures the fetcher can retrieve at least some candles when a network
    connection is available. The assertions are minimal to avoid flakiness from
    remote API variability.
    """

    fetcher = HistoricalDataFetcher(automated_mode=True)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    config = FetchConfig(
        exchange="kraken",
        symbol="BTC/USD",
        timeframe="1h",
        start_date=start_date,
        end_date=end_date,
        cache_enabled=False,
    )

    df, quality = fetcher.fetch_historical_data(config)

    # Basic smoke assertions: data frame should not be empty and quality metadata
    # should reflect the returned data.
    assert not df.empty
    assert "timestamp" in df.columns
    assert quality is not None
    assert quality.total_candles == len(df)
