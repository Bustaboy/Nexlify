# Nexlify Exchange Geo-Blocking Guide

## Problem: Binance Geo-Blocking

If you see this error:
```
Service unavailable from a restricted location according to 'b. Eligibility'
```

**Binance blocks users from certain regions** (USA, and others). This prevents data fetching and training.

---

## Solution 1: Use Alternative Exchanges (Recommended)

### Exchanges That Work Globally

| Exchange | Geo-Restrictions | Data Quality | Speed |
|----------|------------------|--------------|-------|
| **Kraken** | ✅ Works in most regions | ⭐⭐⭐⭐⭐ | Fast |
| **Coinbase** | ✅ Works in most regions | ⭐⭐⭐⭐ | Medium |
| **Bybit** | ⚠️ Some restrictions | ⭐⭐⭐⭐ | Fast |
| **OKX** | ⚠️ Some restrictions | ⭐⭐⭐⭐ | Fast |
| **Binance** | ❌ Many restrictions | ⭐⭐⭐⭐⭐ | Fast |

### Use Kraken Instead of Binance

**Training with Kraken:**
```bash
python train_ultimate_full_pipeline.py \
    --exchange kraken \
    --pairs BTC/USD ETH/USD SOL/USD \
    --initial-episodes 400 \
    --initial-runs 5 \
    --automated
```

**Note:** Kraken uses `BTC/USD` instead of `BTC/USDT`

### Use Coinbase:
```bash
python train_ultimate_full_pipeline.py \
    --exchange coinbase \
    --pairs BTC/USD ETH/USD \
    --initial-episodes 400 \
    --initial-runs 5 \
    --automated
```

---

## Solution 2: Use a VPN

If you need Binance specifically:

1. **Get a VPN** (ProtonVPN, NordVPN, etc.)
2. **Connect to an allowed region** (Europe, Asia - not US)
3. **Run training** with Binance as usual

**Free VPN options:**
- ProtonVPN (free tier available)
- Windscribe (free 10GB/month)

---

## Solution 3: Use Pre-Downloaded Data

Download historical data once from an accessible location, then train offline:

### Step 1: Download Data (from VPN or allowed region)
```python
import ccxt
import pandas as pd
from datetime import datetime, timedelta

# Connect through VPN first
exchange = ccxt.binance()

# Download 2 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

for pair in pairs:
    print(f"Downloading {pair}...")
    ohlcv = exchange.fetch_ohlcv(pair, '1h', limit=1000)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Save to CSV
    filename = pair.replace('/', '_') + '_historical.csv'
    df.to_csv(f'data/{filename}', index=False)
    print(f"Saved {len(df)} candles to {filename}")
```

### Step 2: Train Using Cached Data
```bash
# Training will use cached data automatically
python train_ultimate_full_pipeline.py \
    --pairs BTC/USDT ETH/USDT SOL/USDT \
    --automated
```

---

## Testing Exchange Accessibility

**Run the test suite to see which exchanges work:**
```bash
python test_training_pipeline.py --quick
```

This will test:
1. Kraken (usually works everywhere)
2. Coinbase (works in most regions)
3. Binance (geo-blocked in many regions)
4. Bybit
5. OKX

The test will show which exchanges are accessible from your location.

---

## Recommended: Use Kraken

**Kraken is the best alternative:**
- ✅ Works in 99% of locations (including USA)
- ✅ High-quality historical data
- ✅ Fast API
- ✅ No registration required for public data
- ✅ Same pairs available (use USD instead of USDT)

**Update your training commands:**
```bash
# OLD (Binance - geo-blocked)
python train_ultimate_full_pipeline.py --pairs BTC/USDT ETH/USDT

# NEW (Kraken - works everywhere)
python train_ultimate_full_pipeline.py --exchange kraken --pairs BTC/USD ETH/USD
```

---

## For Live Trading (Not Training)

**Note:** Geo-blocking only affects data fetching for training. For live trading:
- Use Kraken, Coinbase, or other exchanges available in your region
- Or use Binance with VPN
- Training data source doesn't have to match your live trading exchange

---

## Summary

1. **Run test suite first:** `python test_training_pipeline.py --quick`
2. **See which exchanges work** in your region
3. **Use Kraken if Binance is blocked:** Add `--exchange kraken`
4. **Change pairs:** Use `BTC/USD` instead of `BTC/USDT` for Kraken

**Most common fix:**
```bash
python train_ultimate_full_pipeline.py --exchange kraken --pairs BTC/USD ETH/USD --automated
```
