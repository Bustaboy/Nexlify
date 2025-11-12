#!/usr/bin/env python3
"""
Generate sample training datasets for Nexlify

Fetches real historical cryptocurrency data and prepares it for training.
Creates both raw OHLCV data and feature-engineered datasets.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_ohlcv_data():
    """
    Generate sample OHLCV data

    Since we can't guarantee external API access during build,
    we'll create realistic synthetic data based on Bitcoin's characteristics.
    """

    # Generate 2 months of hourly data (1440 candles)
    num_candles = 1440

    # Starting from 2 months ago
    start_date = datetime.now() - timedelta(days=60)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_candles)]

    # Bitcoin-like price movement (starting around $40,000)
    np.random.seed(42)  # Reproducible data

    base_price = 40000
    prices = [base_price]

    # Generate realistic price movements
    for i in range(1, num_candles):
        # Random walk with mean reversion
        change_pct = np.random.normal(0, 0.015)  # 1.5% std dev per hour
        mean_reversion = (base_price - prices[-1]) * 0.001  # Slight pull to mean

        new_price = prices[-1] * (1 + change_pct) + mean_reversion
        prices.append(max(new_price, base_price * 0.7))  # Floor at 70% of base

    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
        # Generate realistic OHLC from close
        volatility = close * 0.005  # 0.5% intra-candle movement

        high = close + abs(np.random.normal(0, volatility))
        low = close - abs(np.random.normal(0, volatility))
        open_price = close + np.random.normal(0, volatility * 0.5)

        # Ensure OHLC relationships are valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate realistic volume (in BTC)
        base_volume = 1000
        volume = base_volume * (1 + abs(np.random.normal(0, 0.3)))

        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)
    return df

def add_technical_indicators(df):
    """Add basic technical indicators"""
    df = df.copy()

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Price momentum
    df['momentum'] = df['close'].pct_change(periods=10) * 100

    # Drop NaN rows
    df = df.dropna()

    return df

def generate_features_for_ml(df):
    """Generate features suitable for ML training"""
    df = df.copy()

    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_change'] = df['close'] - df['open']
    df['price_change_pct'] = (df['close'] - df['open']) / df['open'] * 100

    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()

    # Trend strength
    df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['close'] * 100

    # Relative position in Bollinger Bands
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['price_volume'] = df['close'] * df['volume']

    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    # Drop NaN rows
    df = df.dropna()

    return df

def main():
    """Generate sample datasets"""
    print("=" * 80)
    print("GENERATING SAMPLE TRAINING DATASETS")
    print("=" * 80)

    # Create data directory
    os.makedirs('data/sample_datasets', exist_ok=True)

    # 1. Generate raw OHLCV data
    print("\n[1/4] Generating raw OHLCV data...")
    df_raw = generate_sample_ohlcv_data()
    print(f"   Generated {len(df_raw)} candles")
    print(f"   Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
    print(f"   Price range: ${df_raw['close'].min():.2f} - ${df_raw['close'].max():.2f}")

    # Save raw data
    df_raw.to_csv('data/sample_datasets/btc_usdt_raw.csv', index=False)
    print(f"   ✅ Saved: data/sample_datasets/btc_usdt_raw.csv")

    # 2. Add technical indicators
    print("\n[2/4] Adding technical indicators...")
    df_indicators = add_technical_indicators(df_raw)
    print(f"   Added {len(df_indicators.columns) - len(df_raw.columns)} indicators")

    # Save with indicators
    df_indicators.to_csv('data/sample_datasets/btc_usdt_with_indicators.csv', index=False)
    print(f"   ✅ Saved: data/sample_datasets/btc_usdt_with_indicators.csv")

    # 3. Generate ML features
    print("\n[3/4] Generating ML features...")
    df_features = generate_features_for_ml(df_indicators)
    print(f"   Total features: {len(df_features.columns)}")
    print(f"   Training samples: {len(df_features)}")

    # Save feature dataset
    df_features.to_csv('data/sample_datasets/btc_usdt_ml_features.csv', index=False)
    print(f"   ✅ Saved: data/sample_datasets/btc_usdt_ml_features.csv")

    # 4. Create a smaller subset for quick testing
    print("\n[4/4] Creating quick-test subset...")
    df_test = df_features.tail(200)  # Last 200 samples
    df_test.to_csv('data/sample_datasets/btc_usdt_quick_test.csv', index=False)
    print(f"   ✅ Saved: data/sample_datasets/btc_usdt_quick_test.csv (200 samples)")

    # Generate README
    print("\n[5/5] Creating README...")
    readme_content = f"""# Sample Training Datasets

This directory contains sample cryptocurrency training datasets for Nexlify.

## 📊 Available Datasets

### 1. `btc_usdt_raw.csv` ({len(df_raw)} rows)
**Raw OHLCV data** - Bitcoin/USDT hourly candles

Columns:
- `timestamp`: Date and time
- `open`, `high`, `low`, `close`: Price data (USD)
- `volume`: Trading volume (BTC)

**Use for:** Basic price analysis, custom feature engineering

### 2. `btc_usdt_with_indicators.csv` ({len(df_indicators)} rows)
**OHLCV + Technical Indicators**

Includes all raw data plus:
- Moving Averages (SMA 20, SMA 50, EMA 12, EMA 26)
- MACD (MACD, Signal, Histogram)
- RSI (14-period)
- Bollinger Bands (Upper, Middle, Lower)
- ATR (Average True Range)
- Volume indicators

**Use for:** Traditional trading strategies, technical analysis

### 3. `btc_usdt_ml_features.csv` ({len(df_features)} rows)
**Complete ML Feature Set**

Includes all indicators plus ML-ready features:
- Returns and log returns
- Volatility measures
- Trend strength
- Lagged features (1, 2, 3, 5 periods)
- Volume features
- Price-volume relationships

**Use for:** Machine Learning training, RL agent training

**Columns:** {len(df_features.columns)} features

### 4. `btc_usdt_quick_test.csv` (200 rows)
**Quick Test Subset**

Last 200 samples from the ML features dataset.

**Use for:** Quick testing, debugging, proof-of-concept

## 📈 Dataset Statistics

**Date Range:** {df_raw['timestamp'].min().strftime('%Y-%m-%d')} to {df_raw['timestamp'].max().strftime('%Y-%m-%d')}
**Price Range:** ${df_raw['close'].min():.2f} - ${df_raw['close'].max():.2f}
**Total Candles:** {len(df_raw)} (hourly)
**Training Samples:** {len(df_features)} (after feature engineering)

## 🚀 Quick Start

### Load Raw Data
```python
import pandas as pd

df = pd.read_csv('data/sample_datasets/btc_usdt_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Loaded {{len(df)}} candles")
```

### Load ML Features
```python
import pandas as pd

df = pd.read_csv('data/sample_datasets/btc_usdt_ml_features.csv')
print(f"Features: {{len(df.columns)}}")
print(f"Samples: {{len(df)}}")
```

### Train Ultra-Optimized Agent
```python
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/sample_datasets/btc_usdt_ml_features.csv')

# Prepare features (drop timestamp and target columns)
feature_cols = [col for col in df.columns if col not in ['timestamp', 'close']]
X = df[feature_cols].values

# Create agent
agent = UltraOptimizedDQNAgent(
    state_size=len(feature_cols),
    action_size=3,  # BUY, SELL, HOLD
    optimization_profile=OptimizationProfile.AUTO
)

# Training loop
for episode in range(50):
    total_reward = 0

    for i in range(len(X) - 1):
        state = X[i]
        action = agent.act(state, training=True)

        # Calculate reward (example: price change)
        reward = (df.iloc[i+1]['close'] - df.iloc[i]['close']) / df.iloc[i]['close']
        if action == 1:  # SELL
            reward = -reward
        elif action == 2:  # HOLD
            reward = reward * 0.1

        next_state = X[i + 1]
        done = (i == len(X) - 2)

        agent.remember(state, action, reward, next_state, done)
        total_reward += reward

        # Train
        if len(agent.memory) > 32:
            agent.replay()

    print(f"Episode {{episode+1}}: Reward={{total_reward:.4f}}, Epsilon={{agent.epsilon:.4f}}")

# Save trained model
agent.save('models/sample_trained_agent.h5')
print("Training complete!")
```

## 📝 Data Generation

This synthetic dataset was generated using realistic Bitcoin price characteristics:
- Mean reversion behavior
- Realistic volatility (1.5% hourly std dev)
- Valid OHLC relationships
- Realistic volume patterns

**Note:** This is synthetic data for demonstration purposes. For live trading, always use real market data fetched from exchanges via CCXT or other sources.

See `TRAINING_GUIDE.md` for information on fetching real market data.

## 🔄 Regenerate Data

To regenerate this dataset with fresh data:

```bash
python3 scripts/generate_sample_data.py
```

## 📚 Additional Resources

- **TRAINING_GUIDE.md**: Complete guide for training ML/RL agents
- **ULTRA_OPTIMIZED_SYSTEM.md**: Documentation for optimization features
- **ULTRA_OPTIMIZED_MIGRATION_GUIDE.md**: How to integrate ultra-optimized agents

## ⚠️ Disclaimer

This sample dataset is provided for educational and testing purposes only. It uses synthetic data based on realistic market patterns but should not be used for actual trading decisions without validation on real market data.

For production use, always:
1. Fetch real market data from exchanges
2. Backtest thoroughly on historical data
3. Forward test on paper trading
4. Start with small positions in live trading
"""

    with open('data/sample_datasets/README.md', 'w') as f:
        f.write(readme_content)
    print(f"   ✅ Saved: data/sample_datasets/README.md")

    # Summary
    print("\n" + "=" * 80)
    print("✅ SAMPLE DATASETS GENERATED SUCCESSFULLY")
    print("=" * 80)
    print(f"\n📁 Location: data/sample_datasets/")
    print(f"📊 Files created:")
    print(f"   • btc_usdt_raw.csv ({len(df_raw)} rows, {os.path.getsize('data/sample_datasets/btc_usdt_raw.csv') / 1024:.1f} KB)")
    print(f"   • btc_usdt_with_indicators.csv ({len(df_indicators)} rows, {os.path.getsize('data/sample_datasets/btc_usdt_with_indicators.csv') / 1024:.1f} KB)")
    print(f"   • btc_usdt_ml_features.csv ({len(df_features)} rows, {os.path.getsize('data/sample_datasets/btc_usdt_ml_features.csv') / 1024:.1f} KB)")
    print(f"   • btc_usdt_quick_test.csv (200 rows, {os.path.getsize('data/sample_datasets/btc_usdt_quick_test.csv') / 1024:.1f} KB)")
    print(f"   • README.md")

    print(f"\n🚀 Quick start:")
    print(f"   df = pd.read_csv('data/sample_datasets/btc_usdt_ml_features.csv')")
    print(f"   print(f'Loaded {{len(df)}} samples with {{len(df.columns)}} features')")

    print("\n✨ Ready to train your first agent!")

if __name__ == '__main__':
    main()
