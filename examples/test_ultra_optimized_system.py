#!/usr/bin/env python3
"""
Comprehensive Test Suite for Ultra-Optimized Nexlify System

Tests ALL integrations:
- OptimizationManager with all profiles
- GPU optimizations (NVIDIA/AMD)
- Hyperthreading/SMT optimization
- Multi-GPU detection
- Thermal monitoring
- Smart caching
- Model compilation
- Quantization
- Sentiment analysis
- Ultra-optimized RL agent
- Complete end-to-end workflow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_1_optimization_manager():
    """Test 1: Optimization Manager Initialization"""
    print("\n" + "=" * 80)
    print("TEST 1: Optimization Manager")
    print("=" * 80)

    from nexlify.ml.nexlify_optimization_manager import create_optimizer, OptimizationProfile

    # Test all profiles
    profiles = [
        OptimizationProfile.AUTO,
        OptimizationProfile.ULTRA_LOW_OVERHEAD,
        OptimizationProfile.BALANCED,
        OptimizationProfile.MAXIMUM_PERFORMANCE
    ]

    for profile in profiles:
        print(f"\n   Testing {profile.value}...")
        try:
            optimizer = create_optimizer(profile)
            optimizer.shutdown()
            print(f"   ✅ {profile.value} works!")
        except Exception as e:
            print(f"   ❌ {profile.value} failed: {e}")
            return False

    return True


def test_2_gpu_detection():
    """Test 2: GPU Detection and Optimization"""
    print("\n" + "=" * 80)
    print("TEST 2: GPU Detection and Optimization")
    print("=" * 80)

    from nexlify.ml.nexlify_gpu_optimizations import create_gpu_optimizer

    try:
        optimizer = create_gpu_optimizer()

        if optimizer.capabilities:
            print(f"\n   ✅ GPU Detected: {optimizer.capabilities.name}")
            print(f"      Vendor: {optimizer.capabilities.vendor.value}")
            print(f"      VRAM: {optimizer.capabilities.vram_gb:.1f} GB")
            print(f"      Tensor Cores: {optimizer.capabilities.has_tensor_cores}")
            print(f"      Optimal Batch: {optimizer.capabilities.optimal_batch_size}")

            # Apply optimizations
            optimizer.apply_optimizations()
            print(f"   ✅ GPU optimizations applied")
        else:
            print(f"\n   ⚠️  No GPU detected (CPU-only system)")

        return True

    except Exception as e:
        print(f"   ❌ GPU detection failed: {e}")
        return False


def test_3_cpu_topology():
    """Test 3: CPU Topology Detection (HT/SMT)"""
    print("\n" + "=" * 80)
    print("TEST 3: CPU Topology Detection")
    print("=" * 80)

    from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

    try:
        monitor = EnhancedDynamicResourceMonitor()
        topology = monitor.cpu_topology

        print(f"\n   ✅ CPU Detected:")
        print(f"      Physical Cores: {topology.physical_cores}")
        print(f"      Logical Cores: {topology.logical_cores}")
        print(f"      HT/SMT: {'✓' if topology.has_ht_smt else '✗'}")
        if topology.has_ht_smt:
            print(f"      Effective Cores: {topology.effective_cores:.1f}")
            print(f"      HT Efficiency: {topology.ht_efficiency*100:.0f}%")

        # Test worker calculation
        workers = monitor.calculate_optimal_workers('preprocessing')
        print(f"      Optimal Workers: {workers}")

        monitor.stop_monitoring()
        return True

    except Exception as e:
        print(f"   ❌ CPU topology detection failed: {e}")
        return False


def test_4_thermal_monitoring():
    """Test 4: Thermal Monitoring"""
    print("\n" + "=" * 80)
    print("TEST 4: Thermal Monitoring")
    print("=" * 80)

    from nexlify.ml.nexlify_thermal_monitor import create_thermal_monitor

    try:
        monitor = create_thermal_monitor(check_interval=1.0)
        monitor.start_monitoring()

        # Wait for first snapshot
        time.sleep(2)

        snapshot = monitor.take_snapshot()
        print(f"\n   ✅ Thermal Monitoring Active:")
        print(f"      GPU Temps: {snapshot.gpu_temps}")
        print(f"      Thermal State: {snapshot.thermal_state.value}")
        print(f"      On Battery: {snapshot.on_battery}")

        stats = monitor.get_stats_summary()
        if stats['available']:
            print(f"      Throttling: {stats['is_throttling']}")

        monitor.stop_monitoring()
        return True

    except Exception as e:
        print(f"   ❌ Thermal monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_smart_cache():
    """Test 5: Smart Cache with LZ4 Compression"""
    print("\n" + "=" * 80)
    print("TEST 5: Smart Cache with LZ4 Compression")
    print("=" * 80)

    from nexlify.ml.nexlify_smart_cache import create_smart_cache
    import tempfile

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = create_smart_cache(tmpdir, memory_cache_mb=100)

            # Test data
            test_data = np.random.rand(1000, 100)

            # Write
            cache.put('test_data', test_data)
            print(f"\n   ✅ Data cached")

            # Read
            retrieved = cache.get('test_data')
            if retrieved is not None and np.array_equal(test_data, retrieved):
                print(f"   ✅ Data retrieved correctly")
            else:
                print(f"   ❌ Data retrieval failed")
                return False

            # Stats
            cache.print_stats()

            cache.shutdown()
            return True

    except Exception as e:
        print(f"   ❌ Smart cache failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_sentiment_analysis():
    """Test 6: Sentiment Analysis"""
    print("\n" + "=" * 80)
    print("TEST 6: Sentiment Analysis")
    print("=" * 80)

    from nexlify.ml.nexlify_sentiment_analysis import SentimentAnalyzer
    import asyncio

    try:
        analyzer = SentimentAnalyzer()

        # Get sentiment
        async def get_sentiment():
            return await analyzer.get_sentiment("BTC")

        sentiment = asyncio.run(get_sentiment())

        print(f"\n   ✅ Sentiment Analysis:")
        print(f"      Overall Score: {sentiment.overall_score:.2f}")
        print(f"      Confidence: {sentiment.confidence:.0%}")
        if sentiment.fear_greed_index:
            print(f"      Fear & Greed: {sentiment.fear_greed_index:.0f}/100")
        if sentiment.news_sentiment:
            print(f"      News Sentiment: {sentiment.news_sentiment:.2f}")

        # Get features
        features = analyzer.get_sentiment_features(sentiment)
        print(f"      Features: {len(features)}")

        return True

    except Exception as e:
        print(f"   ❌ Sentiment analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_model_compilation():
    """Test 7: Model Compilation"""
    print("\n" + "=" * 80)
    print("TEST 7: Model Compilation")
    print("=" * 80)

    from nexlify.ml.nexlify_model_compilation import ModelCompiler, CompilationMode

    try:
        # Simple test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        model = TestModel()
        compiler = ModelCompiler()

        # Test compilation
        example_input = torch.randn(1, 10)

        if compiler.has_torch_compile:
            print(f"\n   ✅ torch.compile available")
            compiled = compiler.compile(model, example_inputs=example_input)
            print(f"   ✅ Model compiled successfully")
        else:
            print(f"\n   ⚠️  torch.compile not available (PyTorch < 2.0)")
            compiled = compiler.compile(
                model,
                backend=compiler._select_backend(),
                example_inputs=example_input
            )
            print(f"   ✅ Model compiled with fallback backend")

        return True

    except Exception as e:
        print(f"   ❌ Model compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_8_quantization():
    """Test 8: Automatic Quantization"""
    print("\n" + "=" * 80)
    print("TEST 8: Automatic Quantization")
    print("=" * 80)

    from nexlify.ml.nexlify_quantization import AutoQuantizer, QuantizationMethod

    try:
        # Simple test model
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 20)
                self.fc2 = torch.nn.Linear(20, 5)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = TestModel()
        quantizer = AutoQuantizer()

        # Get original size
        original_size = quantizer._get_model_size(model)
        print(f"\n   Original model size: {original_size:.2f} MB")

        # Quantize
        quantized = quantizer.quantize(model, method=QuantizationMethod.DYNAMIC)
        quantized_size = quantizer._get_model_size(quantized)

        print(f"   Quantized model size: {quantized_size:.2f} MB")
        print(f"   ✅ Reduction: {original_size / quantized_size:.2f}x smaller")

        return True

    except Exception as e:
        print(f"   ❌ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_9_feature_engineering():
    """Test 9: Feature Engineering with Sentiment"""
    print("\n" + "=" * 80)
    print("TEST 9: Feature Engineering with Sentiment")
    print("=" * 80)

    from nexlify.ml.nexlify_feature_engineering import FeatureEngineer

    try:
        # Create sample OHLCV data
        data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1H'),
            'open': np.random.rand(100) * 100 + 40000,
            'high': np.random.rand(100) * 100 + 40100,
            'low': np.random.rand(100) * 100 + 39900,
            'close': np.random.rand(100) * 100 + 40000,
            'volume': np.random.rand(100) * 1000000
        })

        # Engineer features (with sentiment)
        engineer = FeatureEngineer(enable_sentiment=True)
        features = engineer.engineer_features(data)

        print(f"\n   ✅ Feature Engineering:")
        print(f"      Original columns: {len(data.columns)}")
        print(f"      Total columns: {len(features.columns)}")
        print(f"      Features added: {len(features.columns) - len(data.columns)}")

        # Check for sentiment features
        sentiment_features = [col for col in features.columns if 'sentiment' in col]
        print(f"      Sentiment features: {len(sentiment_features)}")

        return True

    except Exception as e:
        print(f"   ❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_10_ultra_optimized_agent():
    """Test 10: Ultra-Optimized RL Agent (Full Integration)"""
    print("\n" + "=" * 80)
    print("TEST 10: Ultra-Optimized RL Agent (Full Integration)")
    print("=" * 80)

    from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
    from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

    try:
        # Create agent with BALANCED profile (faster than AUTO for testing)
        agent = create_ultra_optimized_agent(
            state_size=50,
            action_size=3,
            profile=OptimizationProfile.BALANCED,
            enable_sentiment=False  # Disable for faster testing
        )

        print(f"\n   ✅ Agent Created:")

        # Test state generation
        state = np.random.rand(50)

        # Test action selection
        action = agent.act(state, training=False)
        print(f"      Action selected: {action}")

        # Test experience storage
        next_state = np.random.rand(50)
        agent.remember(state, action, 1.0, next_state, False)
        print(f"      Experience stored: {len(agent.memory)} memories")

        # Fill memory with random experiences
        print(f"\n      Filling replay buffer...")
        for i in range(100):
            s = np.random.rand(50)
            a = np.random.randint(0, 3)
            r = np.random.rand()
            ns = np.random.rand(50)
            d = False
            agent.remember(s, a, r, ns, d)

        print(f"      Replay buffer: {len(agent.memory)} experiences")

        # Test training
        print(f"\n      Training on batch...")
        loss = agent.replay()
        print(f"      ✅ Training successful! Loss: {loss:.4f}")

        # Get statistics
        stats = agent.get_statistics()
        print(f"\n   📊 Agent Statistics:")
        for key, value in stats.items():
            print(f"      {key}: {value}")

        # Cleanup
        agent.shutdown()
        print(f"\n   ✅ Agent shutdown successful")

        return True

    except Exception as e:
        print(f"   ❌ Ultra-optimized agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_11_end_to_end_workflow():
    """Test 11: End-to-End Training Workflow"""
    print("\n" + "=" * 80)
    print("TEST 11: End-to-End Training Workflow")
    print("=" * 80)

    from nexlify.strategies.nexlify_ultra_optimized_rl_agent import create_ultra_optimized_agent
    from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

    try:
        # Create sample market data
        market_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='1H'),
            'open': np.random.rand(200) * 100 + 40000,
            'high': np.random.rand(200) * 100 + 40100,
            'low': np.random.rand(200) * 100 + 39900,
            'close': np.random.rand(200) * 100 + 40000,
            'volume': np.random.rand(200) * 1000000
        })

        print(f"\n   📊 Market Data: {len(market_data)} candles")

        # Create agent
        agent = create_ultra_optimized_agent(
            state_size=50,  # Will be updated after feature engineering
            action_size=3,  # Buy, Sell, Hold
            profile=OptimizationProfile.BALANCED,
            enable_sentiment=False  # Disable for faster testing
        )

        print(f"   ✅ Agent created")

        # Engineer features
        print(f"\n   🔧 Engineering features...")
        features = agent.engineer_features(market_data)
        print(f"      Features: {len(features.columns)} columns")

        # Simulate training episodes
        print(f"\n   🎮 Simulating training episodes...")
        num_episodes = 5

        for episode in range(num_episodes):
            state = np.random.rand(50)
            episode_reward = 0

            for step in range(20):
                # Select action
                action = agent.act(state, training=True)

                # Simulate environment step
                next_state = np.random.rand(50)
                reward = np.random.rand() - 0.5
                done = (step == 19)

                # Store experience
                agent.remember(state, action, reward, next_state, done)

                # Train
                if len(agent.memory) >= agent.batch_size:
                    loss = agent.replay()

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Update epsilon
            agent.update_epsilon()

            agent.episodes += 1

            print(f"      Episode {episode + 1}/{num_episodes}: "
                  f"Reward: {episode_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, "
                  f"Memory: {len(agent.memory)}")

        # Get final statistics
        print(f"\n   📊 Final Statistics:")
        stats = agent.get_statistics()
        for key, value in stats.items():
            if key not in ['architecture']:  # Skip long output
                print(f"      {key}: {value}")

        # Cleanup
        agent.shutdown()

        print(f"\n   ✅ End-to-end workflow completed successfully!")
        return True

    except Exception as e:
        print(f"   ❌ End-to-end workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUITE - ULTRA-OPTIMIZED NEXLIFY SYSTEM")
    print("=" * 80)

    tests = [
        ("Optimization Manager", test_1_optimization_manager),
        ("GPU Detection", test_2_gpu_detection),
        ("CPU Topology", test_3_cpu_topology),
        ("Thermal Monitoring", test_4_thermal_monitoring),
        ("Smart Cache", test_5_smart_cache),
        ("Sentiment Analysis", test_6_sentiment_analysis),
        ("Model Compilation", test_7_model_compilation),
        ("Quantization", test_8_quantization),
        ("Feature Engineering", test_9_feature_engineering),
        ("Ultra-Optimized Agent", test_10_ultra_optimized_agent),
        ("End-to-End Workflow", test_11_end_to_end_workflow),
    ]

    results = []
    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {name} crashed: {e}")
            results.append((name, False))
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}  {name}")

    print(f"\n   Total: {len(tests)} tests")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")

    if failed == 0:
        print(f"\n   🎉 ALL TESTS PASSED! System is fully operational!")
    else:
        print(f"\n   ⚠️  Some tests failed. Please review errors above.")

    print("=" * 80)


if __name__ == "__main__":
    main()
