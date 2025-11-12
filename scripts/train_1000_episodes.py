#!/usr/bin/env python3
"""
Train Ultra-Optimized RL Agent for 1000 Episodes

This script trains the Nexlify Ultra-Optimized RL agent using the included
sample dataset. Adjust parameters as needed for your use case.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexlify.strategies import UltraOptimizedDQNAgent
from nexlify.ml import OptimizationProfile, FeatureEngineer

def main():
    """Main training function"""

    print("=" * 80)
    print("NEXLIFY ULTRA-OPTIMIZED RL AGENT - 1000 EPISODE TRAINING")
    print("=" * 80)

    # Configuration
    NUM_EPISODES = 1000
    INITIAL_BALANCE = 10000
    SAVE_CHECKPOINT_EVERY = 50  # Save every 50 episodes

    # 1. Load sample data
    print("\n[1/5] Loading sample dataset...")
    try:
        df = pd.read_csv('data/sample_datasets/btc_usdt_raw.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"   ✅ Loaded {len(df)} candles")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    except FileNotFoundError:
        print("   ❌ Sample dataset not found!")
        print("   Please run: python3 scripts/generate_sample_data.py")
        return 1

    # 2. Engineer features
    print("\n[2/5] Engineering features...")
    feature_engineer = FeatureEngineer(enable_sentiment=False)  # Disable for offline training
    df_features = feature_engineer.engineer_features(df)

    # Prepare training data
    feature_cols = [col for col in df_features.columns
                    if col not in ['timestamp'] and not df_features[col].isna().all()]
    X = df_features[feature_cols].fillna(0).values

    print(f"   ✅ Engineered {len(feature_cols)} features")
    print(f"   Training samples: {X.shape[0]}")

    # 3. Create Ultra-Optimized Agent
    print("\n[3/5] Creating Ultra-Optimized RL Agent...")
    agent = UltraOptimizedDQNAgent(
        state_size=X.shape[1],
        action_size=3,  # BUY, SELL, HOLD
        optimization_profile=OptimizationProfile.AUTO,  # Auto-detect best settings
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9995,  # Slower decay for 1000 episodes
        batch_size=64
    )

    # Get hardware stats
    stats = agent.get_statistics()
    print(f"   ✅ Agent created successfully")
    print(f"   Hardware:")
    print(f"      GPU: {stats['hardware']['gpu_name']}")
    print(f"      Effective cores: {stats['hardware']['effective_cores']}")
    print(f"   Optimizations:")
    for key, value in stats['optimizations'].items():
        print(f"      {key}: {value}")

    # 4. Training loop
    print(f"\n[4/5] Starting training for {NUM_EPISODES} episodes...")
    print("=" * 80)

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Training metrics
    training_history = {
        'episode': [],
        'total_reward': [],
        'trades': [],
        'profit_pct': [],
        'epsilon': [],
        'avg_loss': []
    }

    best_reward = -float('inf')
    start_time = datetime.now()

    for episode in range(NUM_EPISODES):
        balance = INITIAL_BALANCE
        position = 0  # 0=neutral, 1=long, -1=short
        total_reward = 0
        trades = 0
        losses = []

        for i in range(len(X) - 1):
            # Current state
            state = X[i].astype(np.float32)

            # Agent decides action
            action = agent.act(state, training=True)

            # Calculate reward based on price movement and action
            current_price = df_features.iloc[i]['close']
            next_price = df_features.iloc[i + 1]['close']
            price_change_pct = (next_price - current_price) / current_price

            # Reward logic
            if action == 0:  # BUY
                if position <= 0:
                    reward = price_change_pct * 100  # Profit from buying
                    position = 1
                    trades += 1
                else:
                    reward = -0.01  # Small penalty for redundant buy
            elif action == 1:  # SELL
                if position >= 0:
                    reward = -price_change_pct * 100  # Profit from selling
                    position = -1
                    trades += 1
                else:
                    reward = -0.01  # Small penalty for redundant sell
            else:  # HOLD
                if position == 1:
                    reward = price_change_pct * 10  # Small profit if holding long
                elif position == -1:
                    reward = -price_change_pct * 10  # Small profit if holding short
                else:
                    reward = -0.02  # Penalty for inaction

            total_reward += reward
            balance += reward * 10  # Simplified balance update

            # Next state
            next_state = X[i + 1].astype(np.float32)
            done = (i == len(X) - 2)

            # Remember experience
            agent.remember(state, action, reward, next_state, done)

            # Train the agent
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    losses.append(loss)

        # Episode statistics
        profit_pct = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
        avg_loss = np.mean(losses) if losses else 0.0

        # Store metrics
        training_history['episode'].append(episode + 1)
        training_history['total_reward'].append(total_reward)
        training_history['trades'].append(trades)
        training_history['profit_pct'].append(profit_pct)
        training_history['epsilon'].append(agent.epsilon)
        training_history['avg_loss'].append(avg_loss)

        # Print progress
        if (episode + 1) % 10 == 0 or episode == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = (elapsed / (episode + 1)) * (NUM_EPISODES - episode - 1)

            print(f"Episode {episode + 1:4d}/{NUM_EPISODES} | "
                  f"Reward: {total_reward:7.2f} | "
                  f"Trades: {trades:3d} | "
                  f"P/L: {profit_pct:+6.2f}% | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ETA: {eta/60:.1f}m")

        # Save checkpoint
        if (episode + 1) % SAVE_CHECKPOINT_EVERY == 0:
            checkpoint_path = f'models/agent_ep{episode+1}.h5'
            agent.save(checkpoint_path)
            print(f"   💾 Checkpoint saved: {checkpoint_path}")

        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save('models/best_agent.h5')
            print(f"   ⭐ New best model! Reward: {total_reward:.2f}")

    # 5. Save final model and results
    print("\n[5/5] Saving final results...")

    # Save final model
    agent.save('models/final_agent_1000ep.h5')
    print(f"   ✅ Final model saved: models/final_agent_1000ep.h5")

    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv('models/training_history_1000ep.csv', index=False)
    print(f"   ✅ Training history saved: models/training_history_1000ep.csv")

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n📊 Training Summary:")
    print(f"   Total episodes: {NUM_EPISODES}")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Time per episode: {total_time/NUM_EPISODES:.2f} seconds")
    print(f"\n   Best reward: {best_reward:.2f}")
    print(f"   Final reward: {total_reward:.2f}")
    print(f"   Final profit: {profit_pct:+.2f}%")
    print(f"   Final epsilon: {agent.epsilon:.4f}")

    # Get final agent statistics
    final_stats = agent.get_statistics()
    print(f"\n📈 Agent Statistics:")
    print(f"   Memory size: {len(agent.memory)}")
    print(f"   Total training steps: {len(X) * NUM_EPISODES}")

    # Calculate moving averages
    window = 100
    if len(history_df) >= window:
        recent_avg_reward = history_df['total_reward'].tail(window).mean()
        recent_avg_profit = history_df['profit_pct'].tail(window).mean()
        print(f"\n   Last {window} episodes average:")
        print(f"      Reward: {recent_avg_reward:.2f}")
        print(f"      Profit: {recent_avg_profit:+.2f}%")

    print(f"\n💾 Saved Models:")
    print(f"   • models/final_agent_1000ep.h5 (final model)")
    print(f"   • models/best_agent.h5 (best performing)")
    print(f"   • models/agent_ep*.h5 (checkpoints every {SAVE_CHECKPOINT_EVERY} episodes)")

    print(f"\n📊 Training History:")
    print(f"   • models/training_history_1000ep.csv")
    print(f"     (Contains: episode, total_reward, trades, profit_pct, epsilon, avg_loss)")

    print(f"\n🎉 Training complete! Your agent is ready to use.")
    print(f"   To load: agent.load('models/final_agent_1000ep.h5')")

    # Shutdown agent
    agent.shutdown()

    return 0

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
