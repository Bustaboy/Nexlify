# Sample Training Datasets

This directory contains sample cryptocurrency training datasets for Nexlify.

## 📊 Available Datasets

### 1. `btc_usdt_raw.csv` (1,440 rows)
**Raw OHLCV data** - Bitcoin/USDT hourly candles (2 months of data)

**Columns:**
- `timestamp`: Date and time (YYYY-MM-DD HH:MM:SS)
- `open`: Opening price (USD)
- `high`: Highest price in the period (USD)
- `low`: Lowest price in the period (USD)
- `close`: Closing price (USD)
- `volume`: Trading volume (BTC)

**Use for:** Basic price analysis, custom feature engineering

**Example:**
```python
import pandas as pd

df = pd.read_csv('data/sample_datasets/btc_usdt_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Loaded {len(df)} candles")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
```

### 2. `btc_usdt_quick_test.csv` (200 rows)
**Quick Test Subset** - Last 200 samples from raw data

**Use for:** Quick testing, debugging, proof-of-concept

**Example:**
```python
import pandas as pd

df = pd.read_csv('data/sample_datasets/btc_usdt_quick_test.csv')
print(f"Quick test data: {len(df)} samples")
```

## 🚀 Quick Start Training Example

### Basic Training with Sample Data

```python
from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile, FeatureEngineer
import pandas as pd
import numpy as np

# 1. Load sample data
print("Loading sample data...")
df = pd.read_csv('data/sample_datasets/btc_usdt_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Loaded {len(df)} candles")

# 2. Engineer features
print("Engineering features...")
feature_engineer = FeatureEngineer(enable_sentiment=False)  # Disable for offline training
df_features = feature_engineer.engineer_features(df)
print(f"Engineered {len(df_features.columns)} features")

# 3. Prepare training data
feature_cols = [col for col in df_features.columns
                if col not in ['timestamp'] and not df_features[col].isna().all()]
X = df_features[feature_cols].fillna(0).values

print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")

# 4. Create ultra-optimized agent
print("\nCreating Ultra-Optimized RL Agent...")
agent = UltraOptimizedDQNAgent(
    state_size=X.shape[1],
    action_size=3,  # BUY, SELL, HOLD
    optimization_profile=OptimizationProfile.AUTO
)

print("Agent hardware stats:")
stats = agent.get_statistics()
print(f"  GPU: {stats['hardware']['gpu_name']}")
print(f"  Effective cores: {stats['hardware']['effective_cores']}")

# 5. Training loop
print("\nStarting training...")
num_episodes = 20
initial_balance = 10000

for episode in range(num_episodes):
    balance = initial_balance
    position = 0  # 0=neutral, 1=long, -1=short
    total_reward = 0
    trades = 0

    for i in range(len(X) - 1):
        # Current state
        state = X[i].astype(np.float32)

        # Agent decides action
        action = agent.act(state, training=True)

        # Calculate reward based on price movement
        price_change_pct = (df_features.iloc[i+1]['close'] - df_features.iloc[i]['close']) / df_features.iloc[i]['close']

        if action == 0:  # BUY
            reward = price_change_pct * 100 if position <= 0 else -0.01
            if position <= 0:
                position = 1
                trades += 1
        elif action == 1:  # SELL
            reward = -price_change_pct * 100 if position >= 0 else -0.01
            if position >= 0:
                position = -1
                trades += 1
        else:  # HOLD
            reward = price_change_pct * 10 if position == 1 else -price_change_pct * 10 if position == -1 else -0.02

        total_reward += reward
        balance += reward * 10  # Simplified balance update

        # Next state
        next_state = X[i + 1].astype(np.float32)
        done = (i == len(X) - 2)

        # Remember experience
        agent.remember(state, action, reward, next_state, done)

        # Train
        if len(agent.memory) > agent.batch_size:
            agent.replay()

    # Episode stats
    profit_pct = ((balance - initial_balance) / initial_balance) * 100
    print(f"Episode {episode + 1}/{num_episodes}: "
          f"Reward={total_reward:.2f}, "
          f"Trades={trades}, "
          f"P/L={profit_pct:+.2f}%, "
          f"ε={agent.epsilon:.4f}")

    # Save checkpoint every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.save(f'models/sample_agent_ep{episode+1}.h5')
        print(f"  📁 Checkpoint saved")

# 6. Save final model
agent.save('models/sample_trained_agent.h5')
print("\n✅ Training complete!")
print(f"📁 Model saved: models/sample_trained_agent.h5")

# 7. Get final statistics
final_stats = agent.get_statistics()
print("\n📊 Final Agent Statistics:")
print(f"  Optimizations: {final_stats['optimizations']}")
print(f"  Training episodes: {num_episodes}")
print(f"  Memory size: {len(agent.memory)}")

agent.shutdown()
```

## 📈 Expected Output

When you run the training example above, you should see output like:

```
Loading sample data...
Loaded 1440 candles
Engineering features...
Engineered 50 features
Training data: 1400 samples, 50 features

Creating Ultra-Optimized RL Agent...
⚙️  Optimization Manager initialized (profile: auto)
AUTO mode: Will benchmark optimizations on first model optimization
📊 Benchmarking optimizations (this may take 1-2 minutes)...
Agent hardware stats:
  GPU: NVIDIA GeForce GTX 1660
  Effective cores: 12

Starting training...
Episode 1/20: Reward=25.43, Trades=47, P/L=+2.54%, ε=0.9950
Episode 2/20: Reward=38.76, Trades=52, P/L=+3.88%, ε=0.9900
...
Episode 20/20: Reward=127.85, Trades=38, P/L=+12.79%, ε=0.8180

✅ Training complete!
📁 Model saved: models/sample_trained_agent.h5
```

## 🎯 Training Tips

### 1. Start Small
Use the quick test dataset first to verify your setup:
```python
df = pd.read_csv('data/sample_datasets/btc_usdt_quick_test.csv')
```

### 2. Monitor Progress
Track key metrics during training:
- **Total Reward**: Should generally increase over episodes
- **Epsilon**: Exploration rate, should decrease gradually
- **Trades**: Number of actions taken
- **P/L**: Profit/Loss percentage

### 3. Adjust Hyperparameters
If training isn't working well, try adjusting:
```python
agent = UltraOptimizedDQNAgent(
    state_size=X.shape[1],
    action_size=3,
    learning_rate=0.001,    # Lower for stability
    gamma=0.95,             # Discount factor
    epsilon=1.0,            # Starting exploration
    epsilon_min=0.01,       # Minimum exploration
    epsilon_decay=0.995     # Exploration decay rate
)
```

### 4. Save Checkpoints
Always save your progress:
```python
if (episode + 1) % 10 == 0:
    agent.save(f'models/checkpoint_ep{episode+1}.h5')
```

## 📝 Data Characteristics

**Synthetic Dataset Information:**
- **Timeframe**: Hourly (1h)
- **Duration**: 2 months (~1,440 candles)
- **Price Range**: $28,000 - $64,000 (realistic Bitcoin range)
- **Volatility**: 1.5% hourly standard deviation
- **Behavior**: Mean-reverting random walk (realistic market dynamics)

**Note:** This is synthetic data generated using realistic Bitcoin price characteristics. For production trading:
1. Use real market data (see `TRAINING_GUIDE.md`)
2. Backtest thoroughly
3. Paper trade first
4. Start with small positions

## 🔄 Generate Fresh Data

To regenerate sample data with new random variations:

```bash
python3 scripts/generate_sample_data.py
```

This will create new synthetic data while maintaining realistic market patterns.

## 📚 Next Steps

1. **Try the training example above** - Get hands-on experience
2. **Read TRAINING_GUIDE.md** - Learn about fetching real market data
3. **Explore ULTRA_OPTIMIZED_SYSTEM.md** - Understand all optimization features
4. **Check ULTRA_OPTIMIZED_MIGRATION_GUIDE.md** - Integration guide

## 🔗 Related Files

- `TRAINING_GUIDE.md` - Complete training guide with real data sources
- `ULTRA_OPTIMIZED_SYSTEM.md` - System documentation
- `ULTRA_OPTIMIZED_MIGRATION_GUIDE.md` - How to integrate
- `scripts/generate_sample_data.py` - Data generation script

## ⚠️ Disclaimer

**Important**: This sample dataset uses synthetic data for educational purposes only.

For real trading:
- ✅ Fetch real market data from exchanges (CCXT, Kaggle, etc.)
- ✅ Backtest on historical data
- ✅ Forward test on paper trading
- ✅ Start with small positions
- ❌ Don't use synthetic data for live trading decisions

Always practice proper risk management and never trade with money you can't afford to lose.
