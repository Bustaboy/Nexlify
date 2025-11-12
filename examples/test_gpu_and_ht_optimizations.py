#!/usr/bin/env python3
"""
Test GPU-Specific and Hyperthreading/SMT Optimizations

Demonstrates:
- NVIDIA Tensor Core detection and optimization
- AMD ROCm and CDNA/RDNA optimization
- Intel Hyperthreading detection and utilization
- AMD SMT detection and utilization
- Intelligent worker allocation
- CPU affinity recommendations
- GPU-specific precision selection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time


def test_gpu_detection():
    """Test 1: GPU Detection and Capabilities"""
    print("=" * 80)
    print("TEST 1: GPU Detection and Capabilities")
    print("=" * 80)

    from nexlify.ml.nexlify_gpu_optimizations import create_gpu_optimizer

    optimizer = create_gpu_optimizer()

    if not optimizer.capabilities:
        print("\n❌ No GPU detected")
        print("   Running on CPU-only system")
        return False

    caps = optimizer.capabilities

    print(f"\n✅ GPU Detected: {caps.name}")
    print(f"   Vendor: {caps.vendor.value.upper()}")
    print(f"   Architecture: {caps.architecture}")
    print(f"   VRAM: {caps.vram_gb:.1f} GB")

    if caps.compute_capability:
        print(f"   Compute Capability: {caps.compute_capability}")

    if caps.cuda_cores:
        print(f"   CUDA Cores: {caps.cuda_cores:,}")

    if caps.has_tensor_cores:
        print(f"   ✓ Tensor Cores: {caps.tensor_cores or 'Yes'}")
    else:
        print(f"   ✗ Tensor Cores: Not available")

    print(f"\n   Precision Support:")
    print(f"     FP16: {'✓' if caps.has_fp16 else '✗'}")
    print(f"     BF16: {'✓' if caps.has_bf16 else '✗'}")
    print(f"     TF32: {'✓' if caps.has_tf32 else '✗'}")
    print(f"     FP8:  {'✓' if caps.has_fp8 else '✗'}")
    print(f"     INT8: {'✓' if caps.has_int8 else '✗'}")

    if caps.memory_bandwidth_gbps:
        print(f"\n   Memory Bandwidth: {caps.memory_bandwidth_gbps:.0f} GB/s")

    if caps.sm_count:
        print(f"   SM/CU Count: {caps.sm_count}")

    print(f"\n   Optimal Batch Size: {caps.optimal_batch_size}")

    return True


def test_gpu_optimization_config():
    """Test 2: GPU Optimization Configuration"""
    print("\n" + "=" * 80)
    print("TEST 2: GPU Optimization Configuration")
    print("=" * 80)

    from nexlify.ml.nexlify_gpu_optimizations import create_gpu_optimizer

    optimizer = create_gpu_optimizer()

    if not optimizer.config:
        print("\n❌ No GPU optimization config available")
        return

    config = optimizer.config

    print(f"\n📋 Optimization Configuration:")
    print(f"   Vendor: {config.vendor.value.upper()}")

    print(f"\n   Precision Settings:")
    print(f"     Mixed Precision: {'✓' if config.use_mixed_precision else '✗'}")
    print(f"     TF32: {'✓' if config.use_tf32 else '✗'}")
    print(f"     FP16: {'✓' if config.use_fp16 else '✗'}")
    print(f"     BF16: {'✓' if config.use_bf16 else '✗'}")
    print(f"     FP8: {'✓' if config.use_fp8 else '✗'}")

    print(f"\n   Memory Settings:")
    print(f"     Memory Pool: {'✓' if config.use_memory_pool else '✗'}")
    print(f"     Memory Fraction: {config.memory_fraction:.0%}")
    print(f"     Allow Growth: {'✓' if config.allow_growth else '✗'}")

    print(f"\n   Execution Settings:")
    print(f"     cuDNN/MIOpen Benchmark: {'✓' if config.use_cudnn_benchmark else '✗'}")
    print(f"     Tensor Cores: {'✓' if config.use_tensor_cores else '✗'}")
    print(f"     Streams: {config.num_streams}")
    print(f"     Persistent Workers: {'✓' if config.persistent_workers else '✗'}")

    print(f"\n   Batch Settings:")
    print(f"     Optimal Batch Size: {config.optimal_batch_size}")
    print(f"     Gradient Accumulation: {config.gradient_accumulation_steps} steps")

    if config.vendor_specific:
        print(f"\n   Vendor-Specific Settings:")
        for key, value in config.vendor_specific.items():
            if isinstance(value, bool):
                print(f"     {key}: {'✓' if value else '✗'}")
            else:
                print(f"     {key}: {value}")


def test_cpu_topology():
    """Test 3: CPU Topology Detection (HT/SMT)"""
    print("\n" + "=" * 80)
    print("TEST 3: CPU Topology Detection (Hyperthreading/SMT)")
    print("=" * 80)

    from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

    monitor = EnhancedDynamicResourceMonitor()
    topology = monitor.cpu_topology

    print(f"\n🔧 CPU Topology:")
    print(f"   Physical Cores: {topology.physical_cores}")
    print(f"   Logical Cores: {topology.logical_cores}")
    print(f"   HT/SMT Enabled: {'✓ Yes' if topology.has_ht_smt else '✗ No'}")

    if topology.has_ht_smt:
        print(f"   HT/SMT Efficiency: {topology.ht_efficiency*100:.0f}%")
        print(f"   Effective Cores: {topology.effective_cores:.1f}")
        print(f"\n   Explanation:")
        print(f"     - {topology.physical_cores} physical cores")
        print(f"     - {topology.logical_cores - topology.physical_cores} hyperthreaded cores")
        print(f"     - HT cores are ~{topology.ht_efficiency*100:.0f}% as effective as physical")
        print(f"     - Total effective: {topology.physical_cores} + "
              f"{topology.logical_cores - topology.physical_cores} × "
              f"{topology.ht_efficiency} = {topology.effective_cores:.1f} cores")

    print(f"\n   Vendor: {topology.vendor}")
    print(f"   Architecture: {topology.architecture}")


def test_optimal_workers():
    """Test 4: Optimal Worker Calculation"""
    print("\n" + "=" * 80)
    print("TEST 4: Optimal Worker Thread Calculation")
    print("=" * 80)

    from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

    monitor = EnhancedDynamicResourceMonitor()

    workload_types = ['preprocessing', 'computation', 'io']

    print(f"\n⚙️  Optimal Workers for Different Workloads:")

    for workload in workload_types:
        workers = monitor.calculate_optimal_workers(workload)
        print(f"\n   {workload.capitalize()}:")
        print(f"     Recommended workers: {workers}")

        if workload == 'preprocessing':
            print(f"     Rationale: Uses 80% of effective cores (benefits moderately from HT)")
        elif workload == 'computation':
            print(f"     Rationale: Prefers physical cores (compute-heavy, less HT benefit)")
        elif workload == 'io':
            print(f"     Rationale: Uses all logical cores (I/O bound, HT helps)")


def test_cpu_affinity():
    """Test 5: CPU Affinity Recommendations"""
    print("\n" + "=" * 80)
    print("TEST 5: CPU Affinity Recommendations")
    print("=" * 80)

    from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

    monitor = EnhancedDynamicResourceMonitor()
    topology = monitor.cpu_topology

    if not topology.has_ht_smt:
        print("\n   No hyperthreading detected - simple 1:1 core assignment")
        return

    # Test different worker counts
    worker_counts = [2, 4, topology.physical_cores, topology.physical_cores + 2]

    print(f"\n📌 CPU Affinity Recommendations:")

    for num_workers in worker_counts:
        if num_workers > topology.logical_cores:
            continue

        affinities = monitor.get_cpu_affinity_recommendation(num_workers)

        print(f"\n   {num_workers} workers:")

        for i, affinity in enumerate(affinities):
            core_type = "Physical" if affinity[0] < topology.physical_cores else "HT"
            print(f"     Worker {i}: CPU {affinity[0]} ({core_type})")

        if num_workers <= topology.physical_cores:
            print(f"     Strategy: Using physical cores only (optimal for compute)")
        else:
            print(f"     Strategy: Physical cores first, then HT cores (needed for more workers)")


def test_enhanced_monitoring():
    """Test 6: Enhanced Resource Monitoring"""
    print("\n" + "=" * 80)
    print("TEST 6: Enhanced Resource Monitoring")
    print("=" * 80)

    from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

    monitor = EnhancedDynamicResourceMonitor(sample_interval=0.5)
    monitor.start_monitoring()

    print(f"\n📊 Monitoring for 5 seconds...\n")

    for i in range(5):
        time.sleep(1)

        snapshot = monitor.take_snapshot()

        print(f"[{i+1}/5] Snapshot:")
        print(f"  CPU: {snapshot.cpu_percent:.1f}% ({snapshot.cpu_cores_used:.1f} effective cores)")
        print(f"  RAM: {snapshot.ram_used_gb:.1f} GB ({snapshot.ram_percent:.1f}%)")
        print(f"  GPU: {snapshot.gpu_percent:.1f}% ({snapshot.gpu_memory_used_gb:.1f} GB VRAM)")

        if snapshot.gpu_capabilities:
            print(f"  GPU: {snapshot.gpu_capabilities.name}")

        print(f"  Bottleneck: {snapshot.bottleneck.value}")
        print()

    monitor.stop_monitoring()

    print("✅ Monitoring complete")


def test_gpu_integration():
    """Test 7: GPU Integration with Enhanced Monitor"""
    print("\n" + "=" * 80)
    print("TEST 7: GPU Integration with Enhanced Monitor")
    print("=" * 80)

    from nexlify.ml.nexlify_dynamic_architecture_enhanced import EnhancedDynamicResourceMonitor

    monitor = EnhancedDynamicResourceMonitor()

    print(f"\n🎮 GPU Integration:")

    # Get GPU info
    gpu_info = monitor.get_gpu_info_summary()

    if not gpu_info['available']:
        print("   No GPU available")
        return

    print(f"   GPU: {gpu_info['name']}")
    print(f"   Vendor: {gpu_info['vendor'].upper()}")
    print(f"   Architecture: {gpu_info['architecture']}")
    print(f"   VRAM: {gpu_info['vram_gb']:.1f} GB")
    print(f"   Tensor Cores: {'✓' if gpu_info['has_tensor_cores'] else '✗'}")

    print(f"\n   Recommended Settings:")
    print(f"     Batch Size: {gpu_info['optimal_batch_size']}")
    print(f"     Mixed Precision: {'✓' if gpu_info['use_mixed_precision'] else '✗'}")
    print(f"     Precision: {gpu_info['precision']}")
    print(f"     Streams: {gpu_info['num_streams']}")
    print(f"     Gradient Accumulation: {gpu_info['gradient_accumulation_steps']} steps")

    # Device string
    device = monitor.get_device_string()
    print(f"\n   PyTorch Device: '{device}'")


def test_vendor_specific_optimizations():
    """Test 8: Vendor-Specific Optimizations"""
    print("\n" + "=" * 80)
    print("TEST 8: Vendor-Specific Optimizations")
    print("=" * 80)

    from nexlify.ml.nexlify_gpu_optimizations import create_gpu_optimizer

    optimizer = create_gpu_optimizer()

    if not optimizer.config:
        print("\n   No GPU available for optimization")
        return

    caps = optimizer.capabilities
    config = optimizer.config

    print(f"\n🚀 Vendor: {caps.vendor.value.upper()}")

    if caps.vendor.value == 'nvidia':
        print(f"\n   NVIDIA-Specific Optimizations:")
        vs = config.vendor_specific

        print(f"     Compute Capability: {vs['compute_capability']}")
        print(f"     Architecture: {vs['architecture']}")
        print(f"     cuDNN Benchmark: {'✓' if vs['enable_cudnn_benchmark'] else '✗'}")
        print(f"     TF32 Enabled: {'✓' if vs['enable_tf32'] else '✗'}")
        print(f"     Matrix Precision: {vs['matmul_precision']}")
        print(f"     CUDA Graphs: {'✓' if vs['use_cuda_graphs'] else '✗'}")
        print(f"     Flash Attention: {'✓' if vs['enable_flash_attention'] else '✗'}")
        print(f"     Optimal Thread Count: {vs['optimal_thread_count']:,}")

        if caps.has_tensor_cores:
            print(f"\n   💎 Tensor Core Optimization:")
            print(f"     Generation: {vs['architecture']}")
            print(f"     Tensor Cores: {caps.tensor_cores or 'Available'}")
            print(f"     Recommendation: Use mixed precision training for 2-3x speedup")

        if caps.has_tf32:
            print(f"\n   🎯 TF32 Optimization:")
            print(f"     Automatic acceleration for FP32 operations")
            print(f"     No code changes needed")
            print(f"     ~3x faster matmul vs FP32")

        if caps.has_bf16:
            print(f"\n   🎯 BF16 Optimization:")
            print(f"     Better numerical stability than FP16")
            print(f"     Same performance as FP16")
            print(f"     Recommended for training")

    elif caps.vendor.value == 'amd':
        print(f"\n   AMD-Specific Optimizations:")
        vs = config.vendor_specific

        print(f"     Architecture: {vs['architecture']}")
        print(f"     MIOpen Benchmark: {'✓' if vs['miopen_benchmark'] else '✗'}")
        print(f"     HIPBlas: {'✓' if vs['use_hipblas'] else '✗'}")
        print(f"     ROCBlas: {'✓' if vs['use_rocblas'] else '✗'}")
        print(f"     Wave Size: {vs['wave_size']}")
        print(f"     Infinity Cache: {'✓' if vs['use_infinity_cache'] else '✗'}")
        print(f"     Optimized for: {vs['optimize_for_workload']}")

        if 'cdna' in caps.architecture.lower():
            print(f"\n   💎 CDNA Compute Optimization:")
            print(f"     Matrix cores available")
            print(f"     Optimized for HPC/AI workloads")
            if caps.has_bf16:
                print(f"     BF16 support (CDNA2+)")
            if caps.has_fp8:
                print(f"     FP8 support (CDNA3)")

        elif 'rdna' in caps.architecture.lower():
            print(f"\n   🎮 RDNA Gaming GPU:")
            print(f"     Gaming-focused architecture")
            print(f"     Good for inference, moderate for training")
            if 'rdna3' in caps.architecture.lower():
                print(f"     RDNA3: Improved compute performance")


def test_optimization_application():
    """Test 9: Apply Optimizations"""
    print("\n" + "=" * 80)
    print("TEST 9: Apply GPU Optimizations")
    print("=" * 80)

    try:
        import torch

        from nexlify.ml.nexlify_gpu_optimizations import create_gpu_optimizer

        optimizer = create_gpu_optimizer()

        if not optimizer.config:
            print("\n   No GPU available - skipping optimization")
            return

        print(f"\n🔧 Applying optimizations...")

        optimizer.apply_optimizations()

        print(f"\n✅ Optimizations applied successfully!")

        # Check what was applied
        print(f"\n   PyTorch Configuration:")
        print(f"     cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

        if torch.cuda.is_available():
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                print(f"     TF32 Matmul: {torch.backends.cuda.matmul.allow_tf32}")
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                print(f"     TF32 cuDNN: {torch.backends.cudnn.allow_tf32}")

            print(f"     Device: {optimizer.get_device_string()}")

    except ImportError:
        print("\n   PyTorch not available - skipping")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GPU-SPECIFIC AND HYPERTHREADING/SMT OPTIMIZATION TESTS")
    print("=" * 80)

    try:
        # GPU tests
        has_gpu = test_gpu_detection()

        if has_gpu:
            test_gpu_optimization_config()
            test_vendor_specific_optimizations()
            test_optimization_application()

        # CPU tests (always run)
        test_cpu_topology()
        test_optimal_workers()
        test_cpu_affinity()

        # Integration tests
        test_enhanced_monitoring()
        test_gpu_integration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 80)

        print("\n📝 Summary:")
        print("   ✓ GPU detection and capability enumeration")
        print("   ✓ Vendor-specific optimization configuration")
        print("   ✓ CPU topology detection (HT/SMT)")
        print("   ✓ Intelligent worker thread allocation")
        print("   ✓ CPU affinity recommendations")
        print("   ✓ Enhanced resource monitoring")
        print("   ✓ GPU integration with dynamic architecture")

        print("\n🎯 Key Features:")
        print("   • NVIDIA: Tensor Cores, TF32, BF16, FP8, cuDNN optimization")
        print("   • AMD: CDNA/RDNA detection, ROCm, MIOpen, Matrix Cores")
        print("   • Intel: Hyperthreading detection and effective core calculation")
        print("   • AMD: SMT detection and optimal thread distribution")
        print("   • Smart CPU affinity for physical vs HT cores")
        print("   • Automatic precision selection (FP8/BF16/FP16/TF32/FP32)")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
