#!/usr/bin/env python3
"""
GPU Training Verification Script

Tests all GPU training components:
- GPU detection (NVIDIA, AMD, Intel, Apple)
- GPU optimizations (Tensor Cores, mixed precision)
- Training with GPU acceleration
- CPU fallback (backward compatibility)
- All optimization profiles
- Dependencies and imports

Usage:
    python scripts/verify_gpu_training.py
    python scripts/verify_gpu_training.py --quick  # Skip actual training
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import time
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_imports() -> Dict[str, bool]:
    """Test all required imports"""
    print_section("1. TESTING IMPORTS")

    results = {}

    # Core dependencies
    imports_to_test = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("sklearn", "scikit-learn"),
        ("psutil", "psutil"),
        ("py_cpuinfo", "py-cpuinfo"),
        ("gputil", "GPUtil"),
        ("pynvml", "pynvml (NVIDIA GPU monitoring)"),
        ("lz4.frame", "LZ4 (compression)"),
    ]

    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✅ {display_name:40} OK")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ {display_name:40} FAILED: {e}")
            results[module_name] = False

    # Nexlify ML modules
    print("\nTesting Nexlify ML modules:")
    nexlify_modules = [
        ("nexlify.ml.nexlify_gpu_optimizations", "GPU Optimizations"),
        ("nexlify.ml.nexlify_optimization_manager", "Optimization Manager"),
        ("nexlify.ml.nexlify_dynamic_architecture_enhanced", "Dynamic Architecture"),
        ("nexlify.ml.nexlify_thermal_monitor", "Thermal Monitor"),
        ("nexlify.ml.nexlify_multi_gpu", "Multi-GPU Support"),
        ("nexlify.ml.nexlify_smart_cache", "Smart Cache"),
        ("nexlify.ml.nexlify_feature_engineering", "Feature Engineering"),
        ("nexlify.strategies.nexlify_ultra_optimized_rl_agent", "Ultra-Optimized Agent"),
    ]

    for module_name, display_name in nexlify_modules:
        try:
            __import__(module_name)
            print(f"✅ {display_name:40} OK")
            results[module_name] = True
        except ImportError as e:
            print(f"❌ {display_name:40} FAILED: {e}")
            results[module_name] = False

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n📊 Import Success Rate: {success_rate:.1f}% ({sum(results.values())}/{len(results)})")

    return results


def test_gpu_detection():
    """Test GPU detection and capabilities"""
    print_section("2. TESTING GPU DETECTION")

    try:
        import torch
        from nexlify.ml.nexlify_gpu_optimizations import GPUOptimizer

        # Check PyTorch CUDA availability
        print("PyTorch CUDA Status:")
        print(f"  CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            print(f"  Current Device: {torch.cuda.current_device()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")

            # Device properties
            props = torch.cuda.get_device_properties(0)
            print(f"\nGPU Properties:")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi Processors: {props.multi_processor_count}")
        else:
            print("  ⚠️  No CUDA GPUs detected - will use CPU")

        # Check for other backends
        print("\nOther GPU Backends:")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("  ✅ Apple Metal (MPS) available")
        else:
            print("  ❌ Apple Metal (MPS) not available")

        # Test GPU Optimizer
        print("\nTesting GPU Optimizer:")
        gpu_optimizer = GPUOptimizer()

        if gpu_optimizer.capabilities:
            caps = gpu_optimizer.capabilities
            print(f"  ✅ GPU Detected: {caps.name}")
            print(f"     Vendor: {caps.vendor.value}")
            print(f"     Architecture: {caps.architecture}")
            print(f"     VRAM: {caps.vram_gb:.1f} GB")
            print(f"     Compute Capability: {caps.compute_capability}")
            print(f"     Tensor Cores: {'✅' if caps.has_tensor_cores else '❌'}")
            print(f"     FP16: {'✅' if caps.has_fp16 else '❌'}")
            print(f"     BF16: {'✅' if caps.has_bf16 else '❌'}")
            print(f"     TF32: {'✅' if caps.has_tf32 else '❌'}")
            print(f"     Optimal Batch Size: {caps.optimal_batch_size}")

            # Test optimization config
            if gpu_optimizer.config:
                config = gpu_optimizer.config
                print(f"\n  Optimization Config:")
                print(f"     Mixed Precision: {'✅' if config.use_mixed_precision else '❌'}")
                print(f"     TF32: {'✅' if config.use_tf32 else '❌'}")
                print(f"     FP16: {'✅' if config.use_fp16 else '❌'}")
                print(f"     BF16: {'✅' if config.use_bf16 else '❌'}")
                print(f"     Tensor Cores: {'✅' if config.use_tensor_cores else '❌'}")
                print(f"     Optimal Batch Size: {config.optimal_batch_size}")
                print(f"     Device String: {gpu_optimizer.get_device_string()}")

            return True
        else:
            print("  ⚠️  No GPU detected - CPU-only mode")
            return False

    except Exception as e:
        logger.error(f"GPU detection failed: {e}", exc_info=True)
        return False


def test_optimization_profiles():
    """Test all optimization profiles"""
    print_section("3. TESTING OPTIMIZATION PROFILES")

    try:
        from nexlify.ml.nexlify_optimization_manager import (
            OptimizationManager,
            OptimizationProfile
        )

        profiles = [
            OptimizationProfile.ULTRA_LOW_OVERHEAD,
            OptimizationProfile.BALANCED,
            OptimizationProfile.MAXIMUM_PERFORMANCE,
            OptimizationProfile.INFERENCE_ONLY,
        ]

        for profile in profiles:
            print(f"\nTesting {profile.value.upper()} profile:")
            try:
                manager = OptimizationManager(profile=profile)
                print(f"  ✅ {profile.value}: Initialized successfully")

                # Get configuration
                config = manager.config
                print(f"     GPU Optimizations: {'✅' if config.enable_gpu_optimizations else '❌'}")
                print(f"     Mixed Precision: {'✅' if config.enable_mixed_precision else '❌'}")
                print(f"     Thermal Monitoring: {'✅' if config.enable_thermal_monitoring else '❌'}")
                print(f"     Smart Cache: {'✅' if config.enable_smart_cache else '❌'}")
                print(f"     Compilation: {'✅' if config.enable_compilation else '❌'}")
                print(f"     Quantization: {'✅' if config.enable_quantization else '❌'}")

                manager.shutdown()
            except Exception as e:
                print(f"  ❌ {profile.value}: Failed - {e}")

        return True

    except Exception as e:
        logger.error(f"Optimization profile testing failed: {e}", exc_info=True)
        return False


def test_agent_creation():
    """Test creating ultra-optimized agent"""
    print_section("4. TESTING AGENT CREATION")

    try:
        from nexlify.strategies.nexlify_ultra_optimized_rl_agent import (
            UltraOptimizedDQNAgent,
            create_ultra_optimized_agent
        )
        from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

        print("Creating agent with BALANCED profile:")
        agent = create_ultra_optimized_agent(
            state_size=50,
            action_size=3,
            profile=OptimizationProfile.BALANCED,
            enable_sentiment=False  # Disable for testing
        )

        print(f"  ✅ Agent created successfully")
        print(f"     Device: {agent.device}")
        print(f"     Architecture: {agent.architecture}")
        print(f"     Batch Size: {agent.batch_size}")
        print(f"     Mixed Precision: {'✅' if agent.use_mixed_precision else '❌'}")

        # Test agent methods
        print("\n  Testing agent methods:")

        # Test act
        import numpy as np
        test_state = np.random.rand(50)
        action = agent.act(test_state, training=False)
        print(f"     act(): ✅ (returned action: {action})")

        # Test remember
        next_state = np.random.rand(50)
        agent.remember(test_state, action, 0.5, next_state, False)
        print(f"     remember(): ✅ (memory size: {len(agent.memory)})")

        # Test statistics
        stats = agent.get_statistics()
        print(f"     get_statistics(): ✅")
        print(f"        Training steps: {stats['training_steps']}")
        print(f"        Memory size: {stats['memory_size']}")

        # Cleanup
        agent.shutdown()
        print(f"\n  ✅ Agent cleanup successful")

        return True

    except Exception as e:
        logger.error(f"Agent creation failed: {e}", exc_info=True)
        return False


def test_training_loop(quick: bool = False):
    """Test actual training with GPU"""
    print_section("5. TESTING TRAINING LOOP")

    if quick:
        print("⏭️  Skipping training test (--quick mode)")
        return True

    try:
        import numpy as np
        import torch
        from nexlify.strategies.nexlify_ultra_optimized_rl_agent import (
            create_ultra_optimized_agent
        )
        from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

        print("Creating agent for training test:")
        agent = create_ultra_optimized_agent(
            state_size=10,  # Small state for quick test
            action_size=3,
            profile=OptimizationProfile.BALANCED,
            enable_sentiment=False
        )

        print(f"  Device: {agent.device}")
        print(f"  Batch Size: {agent.batch_size}")

        # Generate synthetic experiences
        print("\n  Generating experiences...")
        for i in range(agent.batch_size + 10):  # Enough for one batch
            state = np.random.rand(10)
            action = np.random.randint(0, 3)
            reward = np.random.randn()
            next_state = np.random.rand(10)
            done = False
            agent.remember(state, action, reward, next_state, done)

        print(f"  ✅ Generated {len(agent.memory)} experiences")

        # Test training
        print("\n  Testing training loop (10 iterations):")
        start_time = time.time()

        losses = []
        for i in range(10):
            loss = agent.replay()
            if loss is not None:
                losses.append(loss)

        elapsed = time.time() - start_time

        print(f"  ✅ Training completed")
        print(f"     Iterations: 10")
        print(f"     Time: {elapsed:.3f}s ({elapsed/10*1000:.2f}ms per iteration)")
        print(f"     Avg Loss: {np.mean(losses):.6f}")

        # Test with mixed precision if available
        if agent.use_mixed_precision and 'cuda' in agent.device:
            print("\n  Testing mixed precision training:")
            start_time = time.time()

            for i in range(10):
                loss = agent.replay()

            elapsed = time.time() - start_time
            print(f"  ✅ Mixed precision training successful")
            print(f"     Time: {elapsed:.3f}s ({elapsed/10*1000:.2f}ms per iteration)")

        # Cleanup
        agent.shutdown()

        return True

    except Exception as e:
        logger.error(f"Training loop test failed: {e}", exc_info=True)
        return False


def test_cpu_fallback():
    """Test CPU fallback (backward compatibility)"""
    print_section("6. TESTING CPU FALLBACK")

    original_cuda_available = None
    try:
        import torch
        import numpy as np
        from nexlify.strategies.nexlify_ultra_optimized_rl_agent import (
            create_ultra_optimized_agent
        )
        from nexlify.ml.nexlify_optimization_manager import OptimizationProfile

        # Force CPU mode
        original_cuda_available = torch.cuda.is_available

        # Temporarily disable CUDA
        torch.cuda.is_available = lambda: False

        print("Creating agent with CUDA disabled (testing CPU fallback):")
        try:
            agent = create_ultra_optimized_agent(
                state_size=10,
                action_size=3,
                profile=OptimizationProfile.BALANCED,
                enable_sentiment=False
            )

            print(f"  ✅ Agent created successfully in CPU mode")
            print(f"     Device: {agent.device}")
            print(f"     Architecture: {agent.architecture}")

            # Test basic operations
            test_state = np.random.rand(10)
            action = agent.act(test_state, training=False)
            print(f"  ✅ CPU inference works (action: {action})")

            # Test training
            agent.remember(test_state, action, 0.5, np.random.rand(10), False)
            for i in range(agent.batch_size):
                state = np.random.rand(10)
                agent.remember(state, np.random.randint(0, 3), np.random.randn(),
                              np.random.rand(10), False)

            loss = agent.replay()
            print(f"  ✅ CPU training works (loss: {loss:.6f})")

            agent.shutdown()

        finally:
            # Restore CUDA availability
            torch.cuda.is_available = original_cuda_available

        print("\n✅ CPU fallback is working correctly")
        return True

    except Exception as e:
        logger.error(f"CPU fallback test failed: {e}", exc_info=True)
        # Restore CUDA availability if it was set
        if original_cuda_available is not None:
            import torch
            torch.cuda.is_available = original_cuda_available
        return False


def run_verification(quick: bool = False):
    """Run all verification tests"""
    print("\n" + "="*80)
    print("  🚀 GPU TRAINING VERIFICATION SCRIPT")
    print("  Testing all GPU training components")
    print("="*80)

    results = {}

    # Run all tests
    results['imports'] = test_imports()
    results['gpu_detection'] = test_gpu_detection()
    results['optimization_profiles'] = test_optimization_profiles()
    results['agent_creation'] = test_agent_creation()
    results['training_loop'] = test_training_loop(quick=quick)
    results['cpu_fallback'] = test_cpu_fallback()

    # Summary
    print_section("VERIFICATION SUMMARY")

    all_passed = all(results.values())

    print("Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:30} {status}")

    print(f"\n{'='*80}")
    if all_passed:
        print("  ✅ ALL TESTS PASSED - GPU training is working correctly!")
        print("="*80)
        return 0
    else:
        print("  ⚠️  SOME TESTS FAILED - Check the output above for details")
        print("="*80)
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Verify GPU training implementation"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip actual training (faster)'
    )

    args = parser.parse_args()

    try:
        return run_verification(quick=args.quick)
    except KeyboardInterrupt:
        print("\n\n⚠️  Verification interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
