# Training Guide: Ultra-Optimized RL/ML System

## Getting Training Data

### Option 1: Fetch from Exchanges (Recommended - Free)

The easiest way is to use CCXT to fetch historical data directly from exchanges:

```python
import ccxt
import pandas as pd

# Initialize exchange (no API key needed for public data)
exchange = ccxt.binance()

# Fetch historical OHLCV data (free, no limits for public data)
ohlcv = exchange.fetch_ohlcv(
    'BTC/USDT',       # Trading pair
    '1h',             # Timeframe: 1m, 5m, 15m, 1h, 4h, 1d
    limit=1000        # Number of candles (max usually 1000-1500 per request)
)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save for later use
df.to_csv('data/btc_usdt_1h.csv', index=False)
print(f"Downloaded {len(df)} candles")
```

**Fetching Large Datasets (Multiple Months):**

```python
import ccxt
import pandas as pd
import time

def fetch_historical_data(symbol, timeframe, since_date, limit=1000):
    """
    Fetch historical data in chunks

    Args:
        symbol: e.g., 'BTC/USDT'
        timeframe: e.g., '1h', '1d'
        since_date: Starting date (e.g., '2023-01-01')
        limit: Candles per request (default 1000)
    """
    exchange = ccxt.binance()

    # Convert date to timestamp
    since = exchange.parse8601(f'{since_date}T00:00:00Z')

    all_data = []

    while True:
        try:
            print(f"Fetching from {pd.to_datetime(since, unit='ms')}")

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Update since to last timestamp
            since = ohlcv[-1][0] + 1

            # Rate limiting
            time.sleep(exchange.rateLimit / 1000)

            # Stop if we've reached current time
            if ohlcv[-1][0] >= exchange.milliseconds():
                break

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
            continue

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'])

    return df

# Example: Fetch 1 year of hourly BTC data
df = fetch_historical_data('BTC/USDT', '1h', '2023-01-01')
df.to_csv('data/btc_usdt_1h_1year.csv', index=False)
print(f"Total candles: {len(df)}")
```

**Supported Exchanges (No API Key Needed for Historical Data):**

```python
import ccxt

# Major exchanges with free historical data
exchanges = {
    'Binance': ccxt.binance(),
    'Coinbase': ccxt.coinbase(),
    'Kraken': ccxt.kraken(),
    'Bybit': ccxt.bybit(),
    'OKX': ccxt.okx(),
    'KuCoin': ccxt.kucoin(),
    'Bitfinex': ccxt.bitfinex(),
}

# Fetch from multiple exchanges
for name, exchange in exchanges.items():
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=100)
        print(f"✅ {name}: {len(ohlcv)} candles")
    except Exception as e:
        print(f"❌ {name}: {e}")
```

### Option 2: Download Pre-Built Datasets

#### **2.1 Kaggle Datasets (Free)**

Popular cryptocurrency datasets on Kaggle:

1. **Bitcoin Historical Data** (2012-present)
   - URL: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
   - Format: CSV with minute-level data
   - Size: ~400MB
   ```bash
   # Install Kaggle CLI
   pip install kaggle

   # Download dataset (requires Kaggle API key)
   kaggle datasets download -d mczielinski/bitcoin-historical-data
   unzip bitcoin-historical-data.zip -d data/
   ```

2. **Cryptocurrency Historical Prices** (Multiple coins)
   - URL: https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory
   - 10+ major cryptocurrencies
   - Daily data from 2013-present

3. **Binance Full History**
   - URL: https://www.kaggle.com/datasets/jorijnsmit/binance-full-history
   - All trading pairs on Binance
   - Hourly data

**How to use Kaggle datasets:**

```python
import pandas as pd

# Load Kaggle dataset
df = pd.read_csv('data/bitcoin_historical_data.csv')

# Resample to 1-hour if needed (from minute data)
df['timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
df_1h = df.resample('1h').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

# Save resampled data
df_1h.to_csv('data/btc_1h_resampled.csv')
```

#### **2.2 CryptoDataDownload (Free, No Registration)**

- **URL**: https://www.cryptodatadownload.com/data/
- **Content**: Historical data for 50+ exchanges
- **Format**: CSV files ready to use
- **Timeframes**: 1min, 5min, 1h, 1d

**Direct download links:**

```bash
# Binance BTC/USDT (1-hour data)
wget https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_1h.csv -O data/binance_btc_1h.csv

# Coinbase BTC/USD (daily)
wget https://www.cryptodatadownload.com/cdd/Coinbase_BTCUSD_d.csv -O data/coinbase_btc_daily.csv

# Kraken ETH/USD (1-hour)
wget https://www.cryptodatadownload.com/cdd/Kraken_ETHUSD_1h.csv -O data/kraken_eth_1h.csv
```

#### **2.3 CoinGecko API (Free, No Key)**

```python
import requests
import pandas as pd
import time

def get_coingecko_data(coin_id, vs_currency='usd', days=365):
    """
    Fetch historical data from CoinGecko

    Args:
        coin_id: e.g., 'bitcoin', 'ethereum'
        vs_currency: 'usd', 'eur', etc.
        days: 1, 7, 14, 30, 90, 180, 365, 'max'
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': days,
        'interval': 'hourly' if days <= 90 else 'daily'
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame({
        'timestamp': [x[0] for x in data['prices']],
        'close': [x[1] for x in data['prices']],
        'volume': [x[1] for x in data['total_volumes']]
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    return df

# Fetch Bitcoin data
df_btc = get_coingecko_data('bitcoin', days=365)
df_btc.to_csv('data/btc_coingecko_1year.csv', index=False)

# Popular coin IDs
coins = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano']
```

#### **2.4 Yahoo Finance (For Traditional Markets)**

```python
import yfinance as yf

# Bitcoin futures
btc = yf.download('BTC-USD', start='2020-01-01', end='2024-01-01', interval='1h')
btc.to_csv('data/btc_yahoo_finance.csv')

# Ethereum
eth = yf.download('ETH-USD', start='2020-01-01', interval='1d')
eth.to_csv('data/eth_yahoo_finance.csv')
```

### Option 3: Data Aggregators (Free Tier Available)

#### **3.1 CryptoCompare API**

- Free tier: 100,000 calls/month
- Historical data: Minute, hour, day
- Multiple exchanges

```python
import requests
import pandas as pd

def fetch_cryptocompare_data(symbol, limit=2000):
    """
    Fetch from CryptoCompare (free, no key for basic use)
    """
    url = f"https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': symbol,  # e.g., 'BTC'
        'tsym': 'USD',
        'limit': limit
    }

    response = requests.get(url, params=params)
    data = response.json()['Data']['Data']

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volumefrom']]

# Fetch Bitcoin data
df = fetch_cryptocompare_data('BTC', limit=2000)
df.to_csv('data/btc_cryptocompare.csv', index=False)
```

#### **3.2 AlphaVantage (Free Tier)**

- Free tier: 5 API calls/minute, 500/day
- Digital currencies, forex, stocks

```python
import requests
import pandas as pd

def fetch_alphavantage_crypto(symbol, market='USD', api_key='demo'):
    """
    Fetch from AlphaVantage (free tier: 5 calls/min)
    Get free API key: https://www.alphavantage.co/support/#api-key
    """
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': symbol,
        'market': market,
        'apikey': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()['Time Series (Digital Currency Daily)']

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    return df

# Example (use your own API key)
# df = fetch_alphavantage_crypto('BTC', api_key='YOUR_FREE_KEY')
```

### Option 4: Pre-Processed Training Datasets

If you want ready-to-use training datasets, I've prepared sample data:

```python
from nexlify.ml import SmartCache

# Initialize smart cache
cache = SmartCache(cache_dir='./data/cache')

# Sample datasets included with Nexlify (if available)
datasets = {
    'btc_1h_6months': 'Training/BTC hourly data (6 months)',
    'eth_1h_6months': 'Training/ETH hourly data (6 months)',
    'multi_asset_1d': 'Training/Multiple assets daily data (1 year)',
}

# Check if sample data exists
import os
if os.path.exists('data/sample_datasets'):
    print("✅ Sample datasets available!")
    df = pd.read_csv('data/sample_datasets/btc_1h_6months.csv')
else:
    print("ℹ️  Sample datasets not found. Use Option 1 to fetch from exchanges.")
```

### Recommended Data Sources Summary

| Source | Best For | Cost | Ease of Use | Data Quality |
|--------|----------|------|-------------|--------------|
| **CCXT (Exchanges)** | Real-time, recent data | Free | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Kaggle** | Large historical datasets | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CryptoDataDownload** | Ready-to-use CSVs | Free | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CoinGecko** | Quick testing | Free | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **CryptoCompare** | Professional use | Free tier | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Complete Example: Download and Prepare Data

```python
import ccxt
import pandas as pd
from nexlify.ml import FeatureEngineer

# Step 1: Download data from Binance (free)
print("Step 1: Downloading data from Binance...")
exchange = ccxt.binance()
ohlcv = []

# Fetch 6 months of hourly data (in chunks)
for i in range(6):  # 6 months
    since = exchange.parse8601(f'2023-{7+i:02d}-01T00:00:00Z')
    chunk = exchange.fetch_ohlcv('BTC/USDT', '1h', since, limit=1000)
    ohlcv.extend(chunk)
    print(f"  Month {i+1}: {len(chunk)} candles")

# Step 2: Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# Step 3: Save raw data
df.to_csv('data/btc_usdt_raw.csv', index=False)
print(f"Step 2: Saved {len(df)} candles to data/btc_usdt_raw.csv")

# Step 4: Engineer features
print("Step 3: Engineering features...")
feature_engineer = FeatureEngineer(enable_sentiment=True)
df_features = feature_engineer.engineer_features(df, symbol='BTC')

# Step 5: Save processed data
df_features.to_csv('data/btc_usdt_features.csv', index=False)
print(f"Step 4: Saved {len(df_features)} rows with {len(df_features.columns)} features")

print("\n✅ Data ready for training!")
print(f"   Raw data: data/btc_usdt_raw.csv")
print(f"   Features: data/btc_usdt_features.csv")
```

Now you can use this data for training:

```python
# Load processed data
df_features = pd.read_csv('data/btc_usdt_features.csv')

# Create agent and train
agent = UltraOptimizedDQNAgent(
    state_size=len(df_features.columns),
    action_size=3,
    optimization_profile=OptimizationProfile.AUTO
)

# Training loop using your data
for episode in range(100):
    # ... training code ...
    pass
```

---

## Quick Start: Train Your First Agent

### 1. Basic Training Loop

```python
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile, FeatureEngineer
import pandas as pd
import numpy as np

# Create agent with AUTO optimization
agent = UltraOptimizedDQNAgent(
    state_size=50,          # Number of features
    action_size=3,          # Actions: BUY, SELL, HOLD
    optimization_profile=OptimizationProfile.AUTO  # Auto-detect best settings
)

# Initialize feature engineer (includes sentiment analysis)
feature_engineer = FeatureEngineer(enable_sentiment=True)

# Training loop
for episode in range(100):
    # Get market data
    df = get_market_data(symbol='BTC/USDT', timeframe='1h', limit=500)

    # Engineer features
    df_features = feature_engineer.engineer_features(df)

    # Train on this data
    for i in range(len(df_features) - 1):
        # Current state
        state = df_features.iloc[i].values

        # Agent decides action
        action = agent.act(state, training=True)

        # Calculate reward (your trading logic)
        reward = calculate_reward(df, i, action)

        # Next state
        next_state = df_features.iloc[i + 1].values
        done = (i == len(df_features) - 2)

        # Remember this experience
        agent.remember(state, action, reward, next_state, done)

        # Train the agent (replay from memory)
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()

    # Save checkpoint
    if episode % 10 == 0:
        agent.save(f'models/agent_episode_{episode}.h5')
        print(f"Episode {episode}: Agent saved")

# Shutdown monitoring threads
agent.shutdown()
```

### 2. Training with Historical Data (Recommended)

```python
import ccxt
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile, FeatureEngineer

# Initialize exchange
exchange = ccxt.binance()

# Create agent
agent = UltraOptimizedDQNAgent(
    state_size=50,
    action_size=3,
    optimization_profile=OptimizationProfile.AUTO
)

# Feature engineer with sentiment
feature_engineer = FeatureEngineer(
    enable_sentiment=True,
    sentiment_config={
        'cache_ttl': 300,  # 5 minutes
    }
)

# Training configuration
CONFIG = {
    'symbol': 'BTC/USDT',
    'timeframe': '1h',
    'train_episodes': 100,
    'initial_balance': 10000,
    'trade_size': 0.1,  # 10% of balance per trade
}

def calculate_reward(df, index, action, position, balance):
    """
    Calculate reward for action

    action: 0=BUY, 1=SELL, 2=HOLD
    position: current position (1=long, -1=short, 0=neutral)
    """
    current_price = df.iloc[index]['close']
    next_price = df.iloc[index + 1]['close']
    price_change = (next_price - current_price) / current_price

    # Reward based on action and market movement
    if action == 0:  # BUY
        if price_change > 0:
            reward = price_change * 100  # Positive reward for correct prediction
        else:
            reward = price_change * 100  # Negative reward for wrong prediction

    elif action == 1:  # SELL
        if price_change < 0:
            reward = abs(price_change) * 100  # Positive reward for correct prediction
        else:
            reward = -price_change * 100  # Negative reward for wrong prediction

    else:  # HOLD
        reward = -0.01  # Small penalty for inaction (encourage trading)

    # Bonus for maintaining balance
    if balance > CONFIG['initial_balance']:
        reward += 0.1  # Small bonus for profit

    return reward

# Fetch historical data (last 6 months)
print("Fetching historical data...")
df = exchange.fetch_ohlcv(
    CONFIG['symbol'],
    CONFIG['timeframe'],
    limit=4000  # ~6 months of hourly data
)
df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Engineer features
print("Engineering features (including sentiment)...")
df_features = feature_engineer.engineer_features(df, symbol='BTC')

# Training
print("Starting training...")
for episode in range(CONFIG['train_episodes']):
    balance = CONFIG['initial_balance']
    position = 0  # 0=neutral, 1=long, -1=short

    total_reward = 0
    trades = 0

    for i in range(len(df_features) - 1):
        # Current state
        state = df_features.iloc[i].values.astype(np.float32)

        # Agent decides action
        action = agent.act(state, training=True)

        # Execute trade simulation
        if action == 0 and position <= 0:  # BUY
            position = 1
            trades += 1
        elif action == 1 and position >= 0:  # SELL
            position = -1
            trades += 1
        # else: HOLD or already in position

        # Calculate reward
        reward = calculate_reward(df, i, action, position, balance)
        total_reward += reward

        # Update balance (simplified)
        balance += reward

        # Next state
        next_state = df_features.iloc[i + 1].values.astype(np.float32)
        done = (i == len(df_features) - 2)

        # Remember experience
        agent.remember(state, action, reward, next_state, done)

        # Train the agent
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()

    # Episode summary
    profit_pct = ((balance - CONFIG['initial_balance']) / CONFIG['initial_balance']) * 100
    print(f"Episode {episode + 1}/{CONFIG['train_episodes']}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Trades: {trades}")
    print(f"  Final Balance: ${balance:.2f} ({profit_pct:+.2f}%)")
    print(f"  Epsilon: {agent.epsilon:.4f}")

    # Save checkpoint
    if (episode + 1) % 10 == 0:
        agent.save(f'models/btc_agent_ep{episode+1}.h5')
        print(f"  Model saved!")

# Final save
agent.save('models/btc_agent_final.h5')
print("\nTraining complete!")

# Get statistics
stats = agent.get_statistics()
print(f"\nHardware used:")
print(f"  GPU: {stats['hardware']['gpu_name']}")
print(f"  Effective cores: {stats['hardware']['effective_cores']}")
print(f"\nOptimizations enabled:")
for key, value in stats['optimizations'].items():
    print(f"  {key}: {value}")

agent.shutdown()
```

## Training with Backtesting Integration

```python
from nexlify.backtesting import nexlify_backtester
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile

# Create backtester with ultra-optimized agent
backtester = nexlify_backtester.Backtester(
    strategy='rl',
    initial_balance=10000,
    commission=0.001
)

# Create and train agent
agent = UltraOptimizedDQNAgent(
    state_size=50,
    action_size=3,
    optimization_profile=OptimizationProfile.AUTO
)

# Train with backtesting
results = backtester.run(
    agent=agent,
    data=historical_data,
    train=True,
    episodes=100
)

# Evaluate results
print(f"Final Balance: ${results['final_balance']:.2f}")
print(f"Total Return: {results['total_return']:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}%")

agent.save('models/backtested_agent.h5')
agent.shutdown()
```

## Transfer Learning: Fine-tune Existing Models

```python
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile

# Load pre-trained model
agent = UltraOptimizedDQNAgent(
    state_size=50,
    action_size=3,
    optimization_profile=OptimizationProfile.AUTO
)

# Load existing weights
agent.load('models/pretrained_btc_agent.h5')

# Reduce epsilon for fine-tuning (less exploration)
agent.epsilon = 0.1
agent.epsilon_min = 0.01
agent.epsilon_decay = 0.995

# Fine-tune on new data (e.g., ETH instead of BTC)
print("Fine-tuning on ETH/USDT...")
for episode in range(20):  # Fewer episodes for fine-tuning
    # ... training loop with ETH data ...
    pass

agent.save('models/finetuned_eth_agent.h5')
agent.shutdown()
```

## Real-Time Learning

```python
import asyncio
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile, FeatureEngineer

async def real_time_trading():
    """Real-time trading with continuous learning"""

    # Create agent
    agent = UltraOptimizedDQNAgent(
        state_size=50,
        action_size=3,
        optimization_profile=OptimizationProfile.BALANCED  # Lower overhead for real-time
    )

    # Load trained model
    agent.load('models/btc_agent_final.h5')

    # Set to exploitation mode (low epsilon)
    agent.epsilon = 0.05  # 5% exploration

    # Feature engineer
    feature_engineer = FeatureEngineer(enable_sentiment=True)

    # Trading state
    last_state = None
    last_action = None
    position = 0

    while True:
        try:
            # Get current market data
            df = await get_latest_market_data('BTC/USDT', limit=500)

            # Engineer features
            df_features = feature_engineer.engineer_features(df, symbol='BTC')
            current_state = df_features.iloc[-1].values.astype(np.float32)

            # If we have a previous action, learn from it
            if last_state is not None:
                # Calculate reward from last action
                reward = calculate_reward_realtime(
                    last_action,
                    position,
                    df.iloc[-2]['close'],
                    df.iloc[-1]['close']
                )

                # Learn from this experience
                agent.remember(last_state, last_action, reward, current_state, False)

                # Train if enough experiences
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

            # Decide next action
            action = agent.act(current_state, training=True)

            # Execute trade
            if action == 0 and position <= 0:  # BUY
                await execute_buy_order('BTC/USDT')
                position = 1
                print("BUY signal - Order executed")

            elif action == 1 and position >= 0:  # SELL
                await execute_sell_order('BTC/USDT')
                position = -1
                print("SELL signal - Order executed")

            else:
                print("HOLD - No action")

            # Save state for next iteration
            last_state = current_state
            last_action = action

            # Save model periodically
            if np.random.random() < 0.01:  # 1% chance each iteration
                agent.save('models/live_agent_checkpoint.h5')
                print("Model checkpoint saved")

            # Wait for next candle
            await asyncio.sleep(3600)  # 1 hour for 1h timeframe

        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

# Run
asyncio.run(real_time_trading())
```

## Training Best Practices

### 1. Start with Good Data
```python
# Use at least 3-6 months of data
# Multiple timeframes for robustness
df_1h = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=4000)
df_4h = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=1000)
df_1d = exchange.fetch_ohlcv('BTC/USDT', '1d', limit=180)
```

### 2. Use Proper Train/Validation Split
```python
# Split data: 80% train, 20% validation
split_index = int(len(df) * 0.8)
train_data = df[:split_index]
val_data = df[split_index:]

# Train on train_data, evaluate on val_data
```

### 3. Monitor Training Progress
```python
# Track metrics
training_metrics = {
    'episode': [],
    'total_reward': [],
    'average_loss': [],
    'epsilon': [],
    'validation_return': []
}

for episode in range(num_episodes):
    # ... training ...

    # Record metrics
    training_metrics['episode'].append(episode)
    training_metrics['total_reward'].append(total_reward)
    training_metrics['average_loss'].append(avg_loss)
    training_metrics['epsilon'].append(agent.epsilon)

    # Validate every 10 episodes
    if episode % 10 == 0:
        val_return = validate_agent(agent, val_data)
        training_metrics['validation_return'].append(val_return)

        # Plot metrics
        plot_training_progress(training_metrics)
```

### 4. Hyperparameter Tuning
```python
# Try different hyperparameters
configs = [
    {'learning_rate': 0.001, 'gamma': 0.95, 'epsilon_decay': 0.995},
    {'learning_rate': 0.0001, 'gamma': 0.99, 'epsilon_decay': 0.999},
    {'learning_rate': 0.0005, 'gamma': 0.98, 'epsilon_decay': 0.997},
]

best_config = None
best_return = -float('inf')

for config in configs:
    agent = UltraOptimizedDQNAgent(
        state_size=50,
        action_size=3,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        epsilon_decay=config['epsilon_decay']
    )

    # Train
    returns = train_agent(agent, train_data)

    # Evaluate
    val_return = validate_agent(agent, val_data)

    if val_return > best_return:
        best_return = val_return
        best_config = config
        agent.save('models/best_agent.h5')

print(f"Best config: {best_config}")
print(f"Best validation return: {best_return:.2f}%")
```

### 5. Use Sentiment Analysis
```python
# Enable sentiment analysis for better predictions
agent = UltraOptimizedDQNAgent(
    state_size=50,
    action_size=3,
    optimization_profile=OptimizationProfile.AUTO,
    enable_sentiment=True,  # IMPORTANT!
    sentiment_config={
        'cryptopanic_api_key': 'your_key',  # Optional but recommended
    }
)
```

## Evaluation and Testing

### 1. Backtest on Unseen Data
```python
# Test on completely unseen data (not used in training)
test_data = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=1000)

agent.load('models/btc_agent_final.h5')
agent.epsilon = 0  # Pure exploitation (no exploration)

# Run backtest
balance = 10000
for i in range(len(test_data) - 1):
    state = get_state(test_data, i)
    action = agent.act(state, training=False)

    # Execute action and update balance
    # ...

final_return = ((balance - 10000) / 10000) * 100
print(f"Test Return: {final_return:.2f}%")
```

### 2. Compare Against Baselines
```python
# Compare against buy-and-hold
buy_hold_return = ((test_data[-1]['close'] - test_data[0]['close']) /
                   test_data[0]['close']) * 100

# Compare against random strategy
random_return = evaluate_random_strategy(test_data)

# Compare against simple MA crossover
ma_return = evaluate_ma_strategy(test_data)

print(f"RL Agent: {rl_return:.2f}%")
print(f"Buy & Hold: {buy_hold_return:.2f}%")
print(f"Random: {random_return:.2f}%")
print(f"MA Crossover: {ma_return:.2f}%")
```

### 3. Risk Metrics
```python
import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_max_drawdown(balance_history):
    """Calculate maximum drawdown"""
    peak = balance_history[0]
    max_dd = 0

    for balance in balance_history:
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak
        if dd > max_dd:
            max_dd = dd

    return max_dd * 100

# Calculate metrics
sharpe = calculate_sharpe_ratio(returns)
max_dd = calculate_max_drawdown(balance_history)

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2f}%")
```

## Troubleshooting Training Issues

### Problem: Agent Not Learning
```python
# Solution 1: Check reward function
# Make sure rewards are scaled properly
reward = np.clip(reward, -1, 1)  # Clip to [-1, 1]

# Solution 2: Adjust learning rate
agent.learning_rate = 0.0001  # Lower learning rate

# Solution 3: Increase replay frequency
if len(agent.memory) > 100:  # Replay more often
    agent.replay()
```

### Problem: Agent Too Conservative (Only HOLDs)
```python
# Solution 1: Penalize HOLD more
if action == 2:  # HOLD
    reward = -0.1  # Stronger penalty

# Solution 2: Increase exploration
agent.epsilon = 0.3  # More exploration
agent.epsilon_min = 0.05
agent.epsilon_decay = 0.999  # Slower decay

# Solution 3: Reward trading actions
if action in [0, 1]:  # BUY or SELL
    reward += 0.05  # Small bonus for taking action
```

### Problem: Training Too Slow
```python
# Solution 1: Use MAXIMUM_PERFORMANCE profile
agent = UltraOptimizedDQNAgent(
    state_size=50,
    action_size=3,
    optimization_profile=OptimizationProfile.MAXIMUM_PERFORMANCE
)

# Solution 2: Reduce data size for initial testing
train_data = train_data[-1000:]  # Use last 1000 candles only

# Solution 3: Reduce replay frequency
if len(agent.memory) > 1000 and episode % 10 == 0:  # Replay less often
    agent.replay()
```

## Advanced: Multi-Asset Training

```python
# Train single agent on multiple cryptocurrencies
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']

agent = UltraOptimizedDQNAgent(
    state_size=50,
    action_size=3,
    optimization_profile=OptimizationProfile.AUTO
)

for episode in range(100):
    # Randomly select symbol
    symbol = np.random.choice(symbols)

    # Fetch data
    df = exchange.fetch_ohlcv(symbol, '1h', limit=500)

    # Train on this symbol
    # ...

# Agent learns general patterns across all assets!
agent.save('models/multi_asset_agent.h5')
```

## Summary

**Recommended Training Workflow:**

1. **Data Collection**: Gather 3-6 months of historical data
2. **Feature Engineering**: Use FeatureEngineer with sentiment analysis
3. **Train/Val Split**: 80/20 split
4. **Initial Training**: 50-100 episodes with AUTO profile
5. **Evaluation**: Test on unseen data, calculate metrics
6. **Hyperparameter Tuning**: Try different configs, keep best
7. **Fine-tuning**: Additional training on specific scenarios
8. **Deployment**: Real-time learning with saved model

**Key Points:**
- ✅ Use `OptimizationProfile.AUTO` for best performance
- ✅ Enable sentiment analysis (`enable_sentiment=True`)
- ✅ Monitor training metrics (reward, loss, epsilon)
- ✅ Validate on unseen data regularly
- ✅ Save checkpoints every 10 episodes
- ✅ Use proper reward function (not too sparse, not too dense)
- ✅ Start with exploration (high epsilon) → exploitation (low epsilon)

**Expected Results:**
- After 50-100 episodes: Agent should show consistent profits
- Sharpe ratio > 1.0: Good risk-adjusted returns
- Max drawdown < 20%: Acceptable risk management
- Better than buy-and-hold: Validates strategy effectiveness
