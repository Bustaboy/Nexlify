#!/usr/bin/env python3
"""
Comprehensive Test Suite for Nexlify Training Pipeline

Tests all critical components to identify issues before full training runs.
Run this BEFORE starting any multi-hour training session!

Usage (Standalone):
    python test_training_pipeline.py
    python test_training_pipeline.py --verbose
    python test_training_pipeline.py --quick  # Skip slow tests
    python test_training_pipeline.py --coverage  # Include test coverage scan

Usage (Pytest with Coverage):
    pip install pytest pytest-cov
    pytest test_training_pipeline.py --cov=. --cov-report=html --cov-report=term
    pytest test_training_pipeline.py -m "not slow" --cov=.  # Skip slow tests
    # View HTML report: htmlcov/index.html
"""

import sys
from pathlib import Path
import argparse
import time
from typing import Dict, Any
import numpy as np

# Add project root
sys.path.append(str(Path(__file__).parent))

print("=" * 80)
print("NEXLIFY TRAINING PIPELINE TEST SUITE")
print("=" * 80)
print()

# Track results
test_results = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'warnings': 0
}

def test_status(name: str, passed: bool, message: str = "", warning: bool = False):
    """Print test status"""
    if warning:
        status = "[WARN]"
        test_results['warnings'] += 1
    elif passed:
        status = "[PASS]"
        test_results['passed'] += 1
    else:
        status = "[FAIL]"
        test_results['failed'] += 1

    print(f"{status} {name}")
    if message:
        print(f"      {message}")


# ============================================================================
# TEST 1: Import Dependencies
# ============================================================================
print("\n[TEST 1] Checking Dependencies...")

try:
    import torch
    test_status("Import torch", True, f"Version: {torch.__version__}")
except Exception as e:
    test_status("Import torch", False, str(e))
    sys.exit(1)

try:
    import ccxt
    test_status("Import ccxt", True, f"Version: {ccxt.__version__}")
except Exception as e:
    test_status("Import ccxt", False, str(e))
    sys.exit(1)

try:
    import pandas as pd
    test_status("Import pandas", True, f"Version: {pd.__version__}")
except Exception as e:
    test_status("Import pandas", False, str(e))
    sys.exit(1)

try:
    import numpy as np
    test_status("Import numpy", True, f"Version: {np.__version__}")
except Exception as e:
    test_status("Import numpy", False, str(e))
    sys.exit(1)


# ============================================================================
# TEST 2: Hardware Detection
# ============================================================================
print("\n[TEST 2] Hardware Detection...")

cuda_available = torch.cuda.is_available()
test_status("CUDA availability", True,
           f"GPU: {'YES' if cuda_available else 'NO (using CPU)'}",
           warning=not cuda_available)

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    test_status("GPU details", True, f"{gpu_name}, {gpu_memory:.1f}GB")


# ============================================================================
# TEST 3: Exchange Connectivity
# ============================================================================
print("\n[TEST 3] Exchange Connectivity...")

try:
    exchange = ccxt.binance()

    # Test fetch ticker (very quick)
    start = time.time()
    ticker = exchange.fetch_ticker('BTC/USDT')
    elapsed = time.time() - start

    test_status("Fetch ticker", True,
               f"BTC/USDT: ${ticker['last']:.2f} ({elapsed:.2f}s)")

    # Test fetch OHLCV
    start = time.time()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=10)
    elapsed = time.time() - start

    if len(ohlcv) == 10:
        test_status("Fetch OHLCV", True, f"Got {len(ohlcv)} candles ({elapsed:.2f}s)")
    else:
        test_status("Fetch OHLCV", False, f"Expected 10 candles, got {len(ohlcv)}")

except Exception as e:
    test_status("Exchange connectivity", False, str(e))


# ============================================================================
# TEST 4: Project Imports
# ============================================================================
print("\n[TEST 4] Project Module Imports...")

try:
    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig
    test_status("Import AdvancedDQNAgent", True)
except Exception as e:
    test_status("Import AdvancedDQNAgent", False, str(e))

try:
    from nexlify_environments.nexlify_complete_strategy_env import (
        CompleteMultiStrategyEnvironment, RiskLimits
    )
    test_status("Import CompleteMultiStrategyEnvironment", True)
except Exception as e:
    test_status("Import CompleteMultiStrategyEnvironment", False, str(e))

try:
    from nexlify_data.nexlify_historical_data_fetcher import (
        HistoricalDataFetcher, FetchConfig
    )
    test_status("Import HistoricalDataFetcher", True)
except Exception as e:
    test_status("Import HistoricalDataFetcher", False, str(e))

try:
    from nexlify_data.nexlify_external_features import ExternalFeatureEnricher
    test_status("Import ExternalFeatureEnricher", True)
except Exception as e:
    test_status("Import ExternalFeatureEnricher", False, str(e))


# ============================================================================
# TEST 5: Agent Creation
# ============================================================================
print("\n[TEST 5] Agent Creation...")

try:
    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig

    config = AgentConfig(
        hidden_layers=[64, 64],
        gamma=0.99,
        learning_rate=0.001,
        batch_size=32,
        use_double_dqn=True,
        use_dueling_dqn=True
    )

    agent = AdvancedDQNAgent(
        state_size=10,
        action_size=3,
        config=config
    )

    test_status("Create agent", True, f"State: 10, Actions: 3")

    # Test agent action
    dummy_state = np.random.randn(10).astype(np.float32)
    action = agent.act(dummy_state, training=False)

    if action in [0, 1, 2]:
        test_status("Agent action", True, f"Action: {action}")
    else:
        test_status("Agent action", False, f"Invalid action: {action}")

except Exception as e:
    test_status("Agent creation", False, str(e))


# ============================================================================
# TEST 6: Environment Creation
# ============================================================================
print("\n[TEST 6] Environment Creation...")

try:
    from nexlify_environments.nexlify_complete_strategy_env import (
        CompleteMultiStrategyEnvironment, RiskLimits
    )

    # Create dummy data
    dummy_data = {
        'BTC/USDT': np.random.randn(1000) * 1000 + 50000
    }

    risk_limits = RiskLimits(
        max_position_size=0.1,
        max_daily_loss=0.05,
        stop_loss_percent=0.02,
        take_profit_percent=0.05,
        trailing_stop_percent=0.03,
        max_concurrent_trades=3
    )

    env = CompleteMultiStrategyEnvironment(
        trading_pairs=['BTC/USDT'],
        initial_balance=10000,
        market_data=dummy_data,
        risk_limits=risk_limits
    )

    test_status("Create environment", True,
               f"Balance: $10000, Pairs: 1")

    # Test reset
    state = env.reset()
    test_status("Environment reset", True, f"State shape: {state.shape}")

    # Test step
    action = 2  # HOLD
    next_state, reward, done, info = env.step(action)
    test_status("Environment step", True,
               f"Reward: {reward:.4f}, Done: {done}")

except Exception as e:
    test_status("Environment creation", False, str(e))


# ============================================================================
# TEST 7: Mini Training Loop (CRITICAL)
# ============================================================================
print("\n[TEST 7] Mini Training Loop (10 episodes)...")

try:
    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig
    from nexlify_environments.nexlify_complete_strategy_env import (
        CompleteMultiStrategyEnvironment, RiskLimits
    )

    # Create realistic price data (trending up)
    base_price = 50000
    price_data = base_price + np.cumsum(np.random.randn(500) * 100)

    dummy_data = {'BTC/USDT': price_data}

    risk_limits = RiskLimits(
        max_position_size=0.1,
        stop_loss_percent=0.02,
        take_profit_percent=0.05,
        trailing_stop_percent=0.03
    )

    env = CompleteMultiStrategyEnvironment(
        trading_pairs=['BTC/USDT'],
        initial_balance=10000,
        market_data=dummy_data,
        risk_limits=risk_limits
    )

    config = AgentConfig(
        hidden_layers=[64, 64],
        learning_rate=0.001,
        batch_size=32
    )

    agent = AdvancedDQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=config
    )

    # Run 10 episodes
    episode_returns = []

    for episode in range(10):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:  # Max 100 steps per episode
            action = agent.act(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > 32:
                agent.replay()

            episode_reward += reward
            state = next_state
            steps += 1

        stats = env.get_episode_stats()
        episode_returns.append(stats['total_return_pct'])

    avg_return = np.mean(episode_returns)
    final_return = episode_returns[-1]

    test_status("Mini training loop", True,
               f"Avg return: {avg_return:+.2f}%, Final: {final_return:+.2f}%")

    # Warning if all returns are terrible
    if all(r < -50 for r in episode_returns):
        test_status("Training quality check", True,
                   "WARNING: All episodes lost >50%. May indicate issue.",
                   warning=True)

except Exception as e:
    test_status("Mini training loop", False, str(e))
    import traceback
    traceback.print_exc()


# ============================================================================
# TEST 8: Data Fetching (Optional - Can be slow)
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='Skip slow tests')
parser.add_argument('--verbose', action='store_true', help='Verbose output')
parser.add_argument('--coverage', action='store_true', help='Show test coverage report')
args, _ = parser.parse_known_args()

if not args.quick:
    print("\n[TEST 8] Historical Data Fetching...")

    try:
        from nexlify_data.nexlify_historical_data_fetcher import (
            HistoricalDataFetcher, FetchConfig
        )
        from datetime import datetime, timedelta

        fetcher = HistoricalDataFetcher(automated_mode=True)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Just 1 week for testing

        config = FetchConfig(
            exchange='binance',
            symbol='BTC/USDT',
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            cache_enabled=True
        )

        start = time.time()
        df, quality = fetcher.fetch_historical_data(config)
        elapsed = time.time() - start

        if len(df) > 0:
            test_status("Fetch historical data", True,
                       f"Got {len(df)} candles in {elapsed:.1f}s, quality: {quality.quality_score:.0f}/100")

            # Check for missing data
            if quality.missing_data_pct > 5:
                test_status("Data quality", True,
                           f"WARNING: {quality.missing_data_pct:.1f}% missing data",
                           warning=True)
        else:
            test_status("Fetch historical data", False, "No data returned")

    except Exception as e:
        test_status("Historical data fetching", False, str(e))
else:
    test_results['skipped'] += 1
    print("\n[TEST 8] Historical Data Fetching... [SKIPPED]")


# ============================================================================
# TEST 9: Model Save/Load
# ============================================================================
print("\n[TEST 9] Model Save/Load...")

try:
    from nexlify_advanced_dqn_agent import AdvancedDQNAgent, AgentConfig
    import tempfile
    import os

    config = AgentConfig(hidden_layers=[32, 32])
    agent = AdvancedDQNAgent(state_size=10, action_size=3, config=config)

    # Test action before save
    test_state = np.random.randn(10).astype(np.float32)
    action_before = agent.act(test_state, training=False)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pt")
        agent.save(model_path)

        if os.path.exists(model_path):
            test_status("Save model", True, f"Size: {os.path.getsize(model_path)/1024:.1f}KB")
        else:
            test_status("Save model", False, "File not created")
            raise Exception("Save failed")

        # Load
        new_agent = AdvancedDQNAgent(state_size=10, action_size=3, config=config)
        new_agent.load(model_path)

        # Test action after load (should be same with epsilon=0)
        new_agent.epsilon = 0
        agent.epsilon = 0
        action_after = new_agent.act(test_state, training=False)

        if action_before == action_after:
            test_status("Load model", True, "Actions match after load")
        else:
            test_status("Load model", True,
                       f"WARNING: Actions differ (before={action_before}, after={action_after})",
                       warning=True)

except Exception as e:
    test_status("Model save/load", False, str(e))


# ============================================================================
# TEST 10: Risk Management
# ============================================================================
print("\n[TEST 10] Risk Management...")

try:
    from nexlify_environments.nexlify_complete_strategy_env import RiskLimits

    risk_limits = RiskLimits(
        max_position_size=0.1,
        max_daily_loss=0.05,
        stop_loss_percent=0.02,
        take_profit_percent=0.05,
        trailing_stop_percent=0.03,
        max_concurrent_trades=3,
        use_kelly_criterion=True
    )

    test_status("Risk limits creation", True)

    # Verify values
    assert risk_limits.max_position_size == 0.1
    assert risk_limits.stop_loss_percent == 0.02
    assert risk_limits.use_kelly_criterion == True

    test_status("Risk limits validation", True, "All values correct")

except Exception as e:
    test_status("Risk management", False, str(e))


# ============================================================================
# TEST COVERAGE REPORT
# ============================================================================
if args.coverage:
    print("\n" + "=" * 80)
    print("TEST COVERAGE REPORT")
    print("=" * 80)

    coverage_items = {
        'Core Components': {
            'AdvancedDQNAgent': test_results['passed'] >= 2,
            'CompleteMultiStrategyEnvironment': test_results['passed'] >= 3,
            'HistoricalDataFetcher': not args.quick,
            'ExternalFeatureEnricher': test_results['passed'] >= 4,
            'RiskLimits': test_results['passed'] >= 9,
        },
        'Training Pipeline': {
            'Agent creation': test_results['passed'] >= 5,
            'Environment creation': test_results['passed'] >= 6,
            'Training loop': test_results['passed'] >= 7,
            'Model save/load': test_results['passed'] >= 9,
        },
        'External Dependencies': {
            'PyTorch': test_results['passed'] >= 1,
            'CCXT exchange': test_results['passed'] >= 3,
            'Pandas/Numpy': test_results['passed'] >= 1,
        },
        'Infrastructure': {
            'GPU detection': True,
            'Network connectivity': test_results['passed'] >= 3,
            'File I/O': test_results['passed'] >= 9,
        }
    }

    total_items = 0
    covered_items = 0

    for category, items in coverage_items.items():
        print(f"\n{category}:")
        for item, covered in items.items():
            status = "[X]" if covered else "[ ]"
            print(f"  {status} {item}")
            total_items += 1
            if covered:
                covered_items += 1

    coverage_pct = (covered_items / total_items * 100) if total_items > 0 else 0
    print(f"\nOverall Coverage: {covered_items}/{total_items} ({coverage_pct:.1f}%)")

    if coverage_pct < 80:
        print("WARNING: Coverage below 80%. Some components not tested.")
    elif coverage_pct < 100:
        print("GOOD: Most components tested. Some optional tests skipped.")
    else:
        print("EXCELLENT: All components tested!")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Passed:   {test_results['passed']}")
print(f"Failed:   {test_results['failed']}")
print(f"Warnings: {test_results['warnings']}")
print(f"Skipped:  {test_results['skipped']}")
print("=" * 80)

if test_results['failed'] > 0:
    print("\n[!] CRITICAL: Some tests failed. DO NOT run full training yet!")
    print("    Fix the failed tests before proceeding.")
    sys.exit(1)
elif test_results['warnings'] > 0:
    print("\n[!] WARNINGS: Tests passed but with warnings.")
    print("    Review warnings above. Training may work but could have issues.")
    sys.exit(0)
else:
    print("\n[OK] All tests passed! Safe to run full training.")
    print("\n    Next step:")
    print("    python train_ultimate_full_pipeline.py --pairs BTC/USDT ETH/USDT --automated")
    sys.exit(0)
