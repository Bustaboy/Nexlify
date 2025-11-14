#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced State Engineering

Tests all components of the state engineering system:
- Volume features
- Trend features
- Volatility features
- Time features
- State normalization
- Multi-timestep stacking
- Full integration with TradingEnvironment
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_volume_features():
    """Test volume feature engineering"""
    logger.info("=" * 60)
    logger.info("TEST 1: Volume Feature Engineering")
    logger.info("=" * 60)

    from nexlify.features.volume_features import VolumeFeatureEngineer

    engineer = VolumeFeatureEngineer()

    # Create test data
    volume_data = pd.Series([1000, 1200, 1100, 1300, 1500, 1400, 1600, 1800, 1700, 2000])
    price_data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

    # Generate features
    features = engineer.engineer_features(volume_data, price_data)

    logger.info(f"✓ Generated {len(features.columns)} volume features")
    logger.info(f"  Features: {list(features.columns)}")
    logger.info(f"  Feature count: {engineer.get_feature_count(include_price_divergence=True)}")

    # Check no placeholders (volume_ratio should not be 1.0 for all)
    volume_ratio = features['volume_ratio'].values
    assert not np.all(volume_ratio == 1.0), "Volume ratio still has placeholder values!"

    logger.info(f"✓ Volume ratio range: [{volume_ratio.min():.3f}, {volume_ratio.max():.3f}]")
    logger.info(f"✓ No placeholder values detected")

    return True


def test_trend_features():
    """Test trend feature engineering"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Trend Feature Engineering")
    logger.info("=" * 60)

    from nexlify.features.technical_features import TrendFeatureEngineer

    engineer = TrendFeatureEngineer()

    # Create test OHLC data
    prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112]
    price_df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices
    })

    # Generate features
    features = engineer.engineer_features(price_df)

    logger.info(f"✓ Generated {len(features.columns)} trend features")
    logger.info(f"  Features: {list(features.columns)}")

    # Check EMA crossovers
    assert 'ema_9_26_cross' in features.columns
    assert 'ema_12_26_cross' in features.columns

    # Check ADX
    assert 'adx' in features.columns
    adx_values = features['adx'].values
    logger.info(f"✓ ADX range: [{adx_values.min():.3f}, {adx_values.max():.3f}]")

    return True


def test_volatility_features():
    """Test volatility feature engineering"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Volatility Feature Engineering")
    logger.info("=" * 60)

    from nexlify.features.technical_features import VolatilityFeatureEngineer

    engineer = VolatilityFeatureEngineer()

    # Create test data with varying volatility
    prices = [100, 102, 99, 104, 98, 106, 97, 108, 95, 110]
    price_df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices
    })

    # Generate features
    features = engineer.engineer_features(price_df)

    logger.info(f"✓ Generated {len(features.columns)} volatility features")
    logger.info(f"  Features: {list(features.columns)}")

    # Check ATR
    assert 'atr_norm' in features.columns
    atr = features['atr_norm'].values
    logger.info(f"✓ ATR normalized range: [{atr.min():.3f}, {atr.max():.3f}]")

    # Check Bollinger Bands
    assert 'bb_position' in features.columns
    assert 'bb_width' in features.columns

    bb_pos = features['bb_position'].values
    logger.info(f"✓ BB position range: [{bb_pos.min():.3f}, {bb_pos.max():.3f}]")

    return True


def test_time_features():
    """Test time feature engineering"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Time Feature Engineering")
    logger.info("=" * 60)

    from nexlify.features.time_features import TimeFeatureEngineer

    engineer = TimeFeatureEngineer()

    # Create test timestamps
    timestamps = pd.date_range(start='2024-01-01', periods=24, freq='1h')

    # Generate features
    features = engineer.engineer_features(timestamps)

    logger.info(f"✓ Generated {len(features.columns)} time features")
    logger.info(f"  Features: {list(features.columns)}")

    # Check cyclical encoding
    assert 'hour_sin' in features.columns
    assert 'hour_cos' in features.columns

    hour_sin = features['hour_sin'].values
    logger.info(f"✓ Hour sin range: [{hour_sin.min():.3f}, {hour_sin.max():.3f}]")

    # Check market sessions
    assert 'is_asia_session' in features.columns
    assert 'is_europe_session' in features.columns
    assert 'is_us_session' in features.columns

    logger.info(f"✓ Market session features present")

    return True


def test_state_normalizer():
    """Test state normalization"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: State Normalization")
    logger.info("=" * 60)

    from nexlify.features.state_normalizer import StateNormalizer

    normalizer = StateNormalizer(state_size=5, warmup_samples=10)

    # Generate random states
    np.random.seed(42)
    states = [np.random.randn(5) * 100 + 50 for _ in range(100)]

    # Normalize states
    normalized_states = []
    for state in states:
        normalized = normalizer.normalize(state, update_stats=True)
        normalized_states.append(normalized)

    # Check normalization
    normalized_array = np.array(normalized_states[20:])  # After warmup

    mean = normalized_array.mean(axis=0)
    std = normalized_array.std(axis=0)

    logger.info(f"✓ Processed {len(states)} states")
    logger.info(f"✓ Mean after normalization: {mean}")
    logger.info(f"✓ Std after normalization: {std}")

    # Mean should be close to 0, std close to 1
    assert np.allclose(mean, 0, atol=0.5), f"Mean not close to 0: {mean}"
    assert np.allclose(std, 1, atol=0.5), f"Std not close to 1: {std}"

    logger.info("✓ Normalization working correctly")

    # Test save/load
    normalizer.save('/tmp/test_normalizer.json')
    logger.info("✓ Saved normalization parameters")

    new_normalizer = StateNormalizer(state_size=5)
    new_normalizer.load('/tmp/test_normalizer.json')
    logger.info("✓ Loaded normalization parameters")

    return True


def test_multi_timestep_builder():
    """Test multi-timestep state building"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Multi-Timestep State Builder")
    logger.info("=" * 60)

    from nexlify.features.multi_timestep_builder import MultiTimestepStateBuilder

    builder = MultiTimestepStateBuilder(state_size=5, lookback=3)

    # Add states
    states = [
        np.array([1, 2, 3, 4, 5], dtype=np.float32),
        np.array([2, 3, 4, 5, 6], dtype=np.float32),
        np.array([3, 4, 5, 6, 7], dtype=np.float32),
        np.array([4, 5, 6, 7, 8], dtype=np.float32)
    ]

    for state in states:
        builder.add_state(state)

    # Build stacked state
    stacked = builder.build()

    logger.info(f"✓ State size: {builder.state_size}")
    logger.info(f"✓ Lookback: {builder.lookback}")
    logger.info(f"✓ Output size: {builder.output_size}")
    logger.info(f"✓ Stacked state shape: {stacked.shape}")

    assert stacked.shape[0] == 15, f"Expected size 15, got {stacked.shape[0]}"
    logger.info("✓ Multi-timestep stacking working correctly")

    return True


def test_enhanced_state_engineer():
    """Test complete enhanced state engineering"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Enhanced State Engineer (Full Integration)")
    logger.info("=" * 60)

    from nexlify.features.state_engineering import EnhancedStateEngineer

    # Create with all features
    engineer = EnhancedStateEngineer(
        use_volume=True,
        use_trend=True,
        use_volatility=True,
        use_time=True,
        use_position=True,
        use_normalization=True,
        use_multi_timestep=False
    )

    logger.info(f"✓ Base state size: {engineer.base_state_size}")
    logger.info(f"✓ Total state size: {engineer.get_state_size()}")

    # Create test data
    prices = list(range(100, 150))
    price_df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [1000000] * len(prices),
        'timestamp': pd.date_range(start='2024-01-01', periods=len(prices), freq='1h')
    })

    # Build state
    state = engineer.build_state(
        price_df=price_df,
        volume_series=price_df['volume'],
        timestamp_series=price_df['timestamp'],
        current_balance=10000,
        current_position=0.5,
        entry_price=120,
        current_price=130,
        equity_history=[10000, 10500, 11000]
    )

    logger.info(f"✓ Generated state shape: {state.shape}")
    logger.info(f"✓ State range: [{state.min():.3f}, {state.max():.3f}]")

    # Get feature names
    feature_names = engineer.get_feature_names()
    logger.info(f"✓ Total features: {len(feature_names)}")

    # Get feature groups
    groups = engineer.get_feature_importance_groups()
    for group_name, features in groups.items():
        logger.info(f"  {group_name}: {len(features)} features")

    return True


def test_trading_environment_integration():
    """Test integration with TradingEnvironment"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: TradingEnvironment Integration")
    logger.info("=" * 60)

    from nexlify.environments.nexlify_rl_training_env import TradingEnvironment

    # Test 1: Legacy state (backward compatibility)
    logger.info("\nTest 8a: Legacy state system")
    env_legacy = TradingEnvironment(
        initial_balance=10000,
        use_enhanced_state=False,
        state_size=12
    )

    state = env_legacy.reset()
    logger.info(f"✓ Legacy state size: {state.shape}")
    assert state.shape[0] == 12

    # Test 2: Enhanced state (no multi-timestep)
    logger.info("\nTest 8b: Enhanced state (no multi-timestep)")
    env_enhanced = TradingEnvironment(
        initial_balance=10000,
        use_enhanced_state=True,
        state_feature_groups=['volume', 'trend', 'volatility'],
        use_state_normalization=True,
        use_multi_timestep=False
    )

    state = env_enhanced.reset()
    logger.info(f"✓ Enhanced state size: {state.shape}")
    logger.info(f"✓ Environment state_size: {env_enhanced.state_size}")
    assert state.shape[0] == env_enhanced.state_size

    # Run a few steps
    for i in range(5):
        action = np.random.randint(0, 3)
        next_state, reward, done, info = env_enhanced.step(action)
        logger.info(f"  Step {i+1}: action={action}, reward={reward:.3f}, done={done}")

    logger.info("✓ Environment step working correctly")

    # Test 3: Enhanced state with multi-timestep
    logger.info("\nTest 8c: Enhanced state (with multi-timestep)")
    env_multi = TradingEnvironment(
        initial_balance=10000,
        use_enhanced_state=True,
        state_feature_groups=['volume', 'trend'],
        use_state_normalization=True,
        use_multi_timestep=True,
        multi_timestep_lookback=5
    )

    state = env_multi.reset()
    logger.info(f"✓ Multi-timestep state size: {state.shape}")
    logger.info(f"✓ Environment state_size: {env_multi.state_size}")
    assert state.shape[0] == env_multi.state_size

    return True


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED STATE ENGINEERING TEST SUITE")
    logger.info("=" * 60)

    tests = [
        ("Volume Features", test_volume_features),
        ("Trend Features", test_trend_features),
        ("Volatility Features", test_volatility_features),
        ("Time Features", test_time_features),
        ("State Normalizer", test_state_normalizer),
        ("Multi-Timestep Builder", test_multi_timestep_builder),
        ("Enhanced State Engineer", test_enhanced_state_engineer),
        ("Trading Environment Integration", test_trading_environment_integration)
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.error(f"\n✗ {test_name} FAILED: {e}", exc_info=True)
            results[test_name] = "FAILED"

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        logger.info(f"{status} {test_name}: {result}")

        if result == "PASSED":
            passed += 1
        else:
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Total: {len(tests)} tests")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info("=" * 60)

    if failed == 0:
        logger.info("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
