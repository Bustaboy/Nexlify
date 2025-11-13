# Trading Pairs Guide for Multi-Exchange Training

## Recommended Trading Pairs

When using `--exchange auto` for automatic exchange selection, use these widely-available trading pairs:

### Top Tier (Available on ALL exchanges)
These pairs have excellent availability and liquidity across Coinbase, Bitstamp, Bitfinex, and Kraken:

```bash
--pairs BTC/USD BTC/USDT ETH/USD ETH/USDT
```

**Best for**: Maximum data quality, longest historical data, highest reliability

### Second Tier (Available on MOST exchanges)
Good availability but may be missing on 1-2 exchanges:

**⚠️ IMPORTANT: Use USDT pairs for altcoins!**

Most altcoins are NOT available as USD pairs. Use USDT instead:

```bash
--pairs BTC/USD ETH/USD BTC/USDT ETH/USDT \
        SOL/USDT ADA/USDT XRP/USDT MATIC/USDT \
        AVAX/USDT DOT/USDT
```

**Why USDT for altcoins?**
- ✅ USD pairs: Only major coins (BTC, ETH) widely available
- ✅ USDT pairs: Most altcoins available (SOL, ADA, MATIC, etc.)
- ❌ USD pairs for altcoins: Limited exchange support

**Best for**: Diversified portfolio training with altcoins

### ⚠️ Pairs to AVOID with Auto-Selection

**BNB (Binance Coin)**
- ❌ `BNB/USD`, `BNB/USDT`
- **Reason**: BNB is Binance's native token and has very limited availability on other exchanges
- **Impact**: Auto-selection will fail or only find low-quality data
- **Alternative**: Use MATIC, AVAX, or DOT instead

**Exchange-Specific Tokens**
- ❌ FTT, LEO, HT, OKB, CRO
- **Reason**: These are native tokens of specific exchanges
- **Alternative**: Use widely-traded altcoins like SOL, ADA, MATIC

## Exchange Coverage Matrix

### USD Pairs (Traditional Fiat)
| Pair | Coinbase | Bitstamp | Bitfinex | Kraken | Quality Score |
|------|----------|----------|----------|--------|---------------|
| BTC/USD | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| ETH/USD | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| XRP/USD | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ Very Good |
| SOL/USD | ⚠️ | ❌ | ⚠️ | ⚠️ | ⭐⭐ Limited |
| MATIC/USD | ⚠️ | ❌ | ⚠️ | ❌ | ⭐ Very Limited |
| ADA/USD | ⚠️ | ❌ | ⚠️ | ⚠️ | ⭐⭐ Limited |

### USDT Pairs (Stablecoin - RECOMMENDED for Altcoins)
| Pair | Coinbase | Bitstamp | Bitfinex | Kraken | Quality Score |
|------|----------|----------|----------|--------|---------------|
| BTC/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| ETH/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| SOL/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| MATIC/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| ADA/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| XRP/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ Excellent |
| AVAX/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ Very Good |
| DOT/USDT | ✅ | ✅ | ✅ | ✅ | ⭐⭐⭐⭐ Very Good |
| BNB/USDT | ❌ | ❌ | ✅ | ❌ | ⭐ Poor (1 exchange only)

## Recommended Commands

### Maximum Reliability (Top 4 pairs only)
Use only BTC and ETH with both USD and USDT:
```bash
python train_ultimate_full_pipeline.py \
  --pairs BTC/USD ETH/USD BTC/USDT ETH/USDT \
  --exchange auto \
  --automated
```

### Balanced Portfolio (6 pairs) - RECOMMENDED
Mix of USD majors and USDT altcoins:
```bash
python train_ultimate_full_pipeline.py \
  --pairs BTC/USD ETH/USD SOL/USDT ADA/USDT XRP/USDT MATIC/USDT \
  --exchange auto \
  --automated
```

### Large Diversified Portfolio (8+ pairs)
Comprehensive coverage with USDT altcoins:
```bash
python train_ultimate_full_pipeline.py \
  --pairs BTC/USD ETH/USD BTC/USDT ETH/USDT \
          SOL/USDT ADA/USDT XRP/USDT MATIC/USDT \
  --exchange auto \
  --automated
```

### All Altcoins (10 pairs)
If you want more altcoin exposure:
```bash
python train_ultimate_full_pipeline.py \
  --pairs BTC/USDT ETH/USDT SOL/USDT ADA/USDT XRP/USDT \
          MATIC/USDT AVAX/USDT DOT/USDT LINK/USDT ATOM/USDT \
  --exchange auto \
  --automated
```

## Why These Pairs Work Best

### USD vs USDT
Both work well, but there are subtle differences:

**USD pairs:**
- ✅ Better regulated, more traditional
- ✅ Available on US-compliant exchanges (Coinbase, Kraken)
- ✅ Often longer historical data
- ❌ May have slightly lower liquidity on some exchanges

**USDT pairs:**
- ✅ More widely available globally
- ✅ Higher liquidity on most exchanges
- ✅ Better for international users
- ❌ Stablecoin risk (Tether controversies)

**Recommendation**: Use **USD pairs** if training primarily for US markets, **USDT pairs** for global markets, or **mix both** for maximum data diversity.

## Exchange Selection Behavior

When you use `--exchange auto`, the system:

1. **Tests each exchange** for your specified pairs
2. **Scores data quality** based on:
   - Historical data completeness (30%)
   - Data quality metrics (60%)
   - Total data volume (10%)
3. **Selects best exchange** per pair independently
4. **Falls back gracefully** if an exchange is unavailable

### Example Output

```
🔍 Auto-selecting best exchange for BTC/USD...
  → Testing coinbase...
    ✓ Quality: 95.2/100, Candles: 17,520, Score: 92.1
  → Testing bitstamp...
    ✓ Quality: 89.5/100, Candles: 15,840, Score: 86.3
  → Testing kraken...
    ✓ Quality: 71.3/100, Candles: 721, Score: 49.2
✅ Selected: coinbase for BTC/USD (quality: 95.2/100)

🔍 Auto-selecting best exchange for BNB/USD...
  → Testing coinbase...
    ✗ BNB/USD not available
  → Testing bitstamp...
    ✗ BNB/USD not available
  → Testing bitfinex...
    ✓ Quality: 68.1/100, Candles: 450, Score: 47.5
  → Testing kraken...
    ✗ BNB/USD not available
⚠️  Selected: bitfinex for BNB/USD (quality: 68.1/100) - LIMITED DATA
```

## Troubleshooting

### "No exchange could provide data for SYMBOL"

**Cause**: The symbol is not available on any tested exchange.

**Solutions**:
1. Check the symbol format (should be `BTC/USD` not `BTCUSD`)
2. Replace with a more widely-available pair
3. Use `--exchange coinbase` to test a specific exchange
4. Check exchange websites to verify symbol availability

### "Only 1 exchange available for SYMBOL"

**Cause**: Symbol is only on one exchange (like BNB).

**Impact**: No quality comparison possible, must use whatever data is available.

**Solutions**:
1. Replace with a pair available on multiple exchanges
2. Accept the warning and proceed (data quality may be lower)
3. Add `--min-quality 60` to lower quality requirements

### Poor Data Quality Scores

**Cause**: Exchange doesn't have enough historical data for your time range.

**Solutions**:
1. Reduce `--years 2` to request less historical data
2. Use different exchanges with `--exchange coinbase`
3. Accept warnings with `--min-quality 70` (lowered threshold)

## Updated Exchange List

The system now tests these exchanges (in order):
1. **Coinbase** - Best for US users, excellent data quality
2. **Bitstamp** - Long history, reliable data
3. **Bitfinex** - Good altcoin coverage
4. **Kraken** - Variable quality, good as fallback

**Removed:**
- ~~Huobi~~ - API connectivity issues
- ~~Binance~~ - Geoblocking problems in many regions

---

**Last Updated**: 2025-11-13
**Recommended Action**: Replace `BNB/USD` with `MATIC/USD`, `AVAX/USD`, or `DOT/USD` in your training commands.
