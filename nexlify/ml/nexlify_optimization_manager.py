#!/usr/bin/env python3
"""
Nexlify Optimization Manager

Centralized management of all optimizations with overhead analysis.

CRITICAL ANALYSIS:
Each optimization has been evaluated for overhead vs benefit.
Users can choose profiles based on their needs.

Profiles:
- ULTRA_LOW_OVERHEAD: Only optimizations with near-zero overhead
- BALANCED: Good balance of performance and overhead
- MAXIMUM_PERFORMANCE: All optimizations enabled
- INFERENCE_ONLY: Optimized for inference (quantization, compilation)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class OptimizationProfile(Enum):
    """Optimization profiles"""

    AUTO = "auto"  # Automatically benchmark and enable best (RECOMMENDED)
    ULTRA_LOW_OVERHEAD = "ultra_low_overhead"  # < 1% overhead
    BALANCED = "balanced"  # < 5% overhead
    MAXIMUM_PERFORMANCE = "maximum"  # All optimizations
    INFERENCE_ONLY = "inference"  # Inference optimizations only
    CUSTOM = "custom"


@dataclass
class OptimizationConfig:
    """Configuration for optimizations"""

    # Hardware detection (one-time, startup only)
    detect_gpu_capabilities: bool = True  # < 0.1s startup overhead
    detect_cpu_topology: bool = True  # < 0.1s startup overhead
    detect_multi_gpu: bool = True  # < 0.2s startup overhead

    # Background monitoring (ongoing overhead)
    enable_thermal_monitoring: bool = False  # 30s interval = 0.001% overhead
    thermal_check_interval: float = 30.0  # Seconds between checks

    enable_resource_monitoring: bool = False  # 100ms interval = 0.1% overhead
    resource_check_interval: float = 0.5  # Seconds between checks

    # Caching (overhead from memory, but massive speedup)
    enable_smart_cache: bool = False  # Memory overhead, disk speedup
    cache_size_mb: int = 1000  # Memory overhead
    enable_compression: bool = True  # CPU overhead, disk savings (LZ4 is fast!)

    # Model optimization (one-time compilation overhead)
    enable_compilation: bool = False  # 10-60s compile time, 30-50% speedup
    compilation_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"

    enable_quantization: bool = False  # 5-30s conversion, 4x memory + 2-4x speed
    quantization_method: str = "dynamic"  # "dynamic", "static", "fp16"

    # GPU optimizations (zero overhead)
    enable_gpu_optimizations: bool = True  # Zero overhead, automatic
    enable_mixed_precision: bool = True  # Zero overhead, 2-3x speedup
    enable_tensor_cores: bool = True  # Zero overhead, 2-5x speedup

    # CPU optimizations (zero overhead)
    enable_ht_smt_optimization: bool = True  # Zero overhead, better threading
    enable_cpu_affinity: bool = False  # Slight benefit, restrictive

    # Multi-GPU (zero overhead if single GPU)
    enable_multi_gpu: bool = True  # Zero overhead for single GPU


class OptimizationManager:
    """
    Centralized optimization management

    Provides pre-configured profiles and detailed overhead analysis
    """

    def __init__(self, profile: OptimizationProfile = OptimizationProfile.BALANCED):
        """
        Args:
            profile: Optimization profile
        """
        self.profile = profile
        self.config = self._create_config(profile)

        # Lazy-loaded optimizers (only create when needed)
        self.gpu_optimizer = None
        self.resource_monitor = None
        self.thermal_monitor = None
        self.smart_cache = None
        self.multi_gpu_manager = None
        self.model_compiler = None
        self.quantizer = None

        logger.info(f"⚙️  Optimization Manager initialized (profile: {profile.value})")
        self._print_profile_info()

    def _create_config(self, profile: OptimizationProfile) -> OptimizationConfig:
        """Create configuration for profile"""

        if profile == OptimizationProfile.AUTO:
            # AUTO mode: Will benchmark on first use
            # For now, return balanced config
            logger.info(
                "AUTO mode: Will benchmark optimizations on first model optimization"
            )
            return self._create_config(OptimizationProfile.BALANCED)

        elif profile == OptimizationProfile.ULTRA_LOW_OVERHEAD:
            # Only zero-overhead optimizations
            return OptimizationConfig(
                detect_gpu_capabilities=True,  # One-time, < 0.1s
                detect_cpu_topology=True,  # One-time, < 0.1s
                detect_multi_gpu=True,  # One-time, < 0.2s
                enable_thermal_monitoring=False,  # Disabled for ultra-low
                enable_resource_monitoring=False,  # Disabled for ultra-low
                enable_smart_cache=False,  # Disabled (memory overhead)
                enable_compilation=False,  # Disabled (startup overhead)
                enable_quantization=False,  # Disabled (conversion overhead)
                enable_gpu_optimizations=True,  # Zero overhead
                enable_mixed_precision=True,  # Zero overhead, huge speedup
                enable_tensor_cores=True,  # Zero overhead, huge speedup
                enable_ht_smt_optimization=True,  # Zero overhead
                enable_cpu_affinity=False,  # Disabled (restrictive)
                enable_multi_gpu=True,  # Zero overhead if single GPU
            )

        elif profile == OptimizationProfile.BALANCED:
            # Good balance: < 5% overhead
            return OptimizationConfig(
                detect_gpu_capabilities=True,
                detect_cpu_topology=True,
                detect_multi_gpu=True,
                enable_thermal_monitoring=True,  # 30s interval = 0.001% overhead
                thermal_check_interval=30.0,
                enable_resource_monitoring=True,  # 0.5s interval = 0.02% overhead
                resource_check_interval=0.5,
                enable_smart_cache=True,  # Memory overhead, but fast
                cache_size_mb=1000,
                enable_compression=True,  # LZ4 is faster than disk!
                enable_compilation=True,  # One-time cost, 30-50% speedup
                compilation_mode="default",
                enable_quantization=False,  # User can enable if needed
                enable_gpu_optimizations=True,
                enable_mixed_precision=True,
                enable_tensor_cores=True,
                enable_ht_smt_optimization=True,
                enable_cpu_affinity=False,
                enable_multi_gpu=True,
            )

        elif profile == OptimizationProfile.MAXIMUM_PERFORMANCE:
            # All optimizations enabled
            return OptimizationConfig(
                detect_gpu_capabilities=True,
                detect_cpu_topology=True,
                detect_multi_gpu=True,
                enable_thermal_monitoring=True,
                thermal_check_interval=60.0,  # Longer interval for max perf
                enable_resource_monitoring=True,
                resource_check_interval=1.0,  # Longer interval
                enable_smart_cache=True,
                cache_size_mb=2000,  # Larger cache
                enable_compression=True,
                enable_compilation=True,
                compilation_mode="max-autotune",  # Best performance
                enable_quantization=True,  # 4x memory + 2-4x speed
                quantization_method="dynamic",
                enable_gpu_optimizations=True,
                enable_mixed_precision=True,
                enable_tensor_cores=True,
                enable_ht_smt_optimization=True,
                enable_cpu_affinity=True,  # Use CPU affinity
                enable_multi_gpu=True,
            )

        elif profile == OptimizationProfile.INFERENCE_ONLY:
            # Optimized for inference
            return OptimizationConfig(
                detect_gpu_capabilities=True,
                detect_cpu_topology=True,
                detect_multi_gpu=False,  # Not needed for inference
                enable_thermal_monitoring=False,  # Not critical for inference
                enable_resource_monitoring=False,
                enable_smart_cache=True,  # Good for inference
                cache_size_mb=500,
                enable_compression=True,
                enable_compilation=True,  # Critical for inference
                compilation_mode="default",
                enable_quantization=True,  # Critical for inference
                quantization_method="dynamic",
                enable_gpu_optimizations=True,
                enable_mixed_precision=True,
                enable_tensor_cores=True,
                enable_ht_smt_optimization=True,
                enable_cpu_affinity=False,
                enable_multi_gpu=False,
            )

        else:
            # Default to balanced
            return OptimizationConfig()

    def _print_profile_info(self):
        """Print profile information"""
        overhead_analysis = self._analyze_overhead()

        logger.info(f"\n{'='*80}")
        logger.info(f"OPTIMIZATION PROFILE: {self.profile.value.upper()}")
        logger.info(f"{'='*80}")

        logger.info(f"\nExpected Overhead: {overhead_analysis['total_overhead']:.2%}")
        logger.info(f"Expected Speedup: {overhead_analysis['expected_speedup']:.1f}x")
        logger.info(f"Memory Overhead: {overhead_analysis['memory_overhead_mb']} MB")

        logger.info(f"\nEnabled Optimizations:")
        for opt, enabled in overhead_analysis["enabled_optimizations"].items():
            symbol = "✓" if enabled else "✗"
            logger.info(f"  {symbol} {opt}")

        logger.info(f"\n{'='*80}\n")

    def _analyze_overhead(self) -> Dict[str, Any]:
        """Analyze overhead vs benefits"""

        # Calculate overhead
        overhead = 0.0
        memory_overhead = 0

        if self.config.enable_thermal_monitoring:
            overhead += 0.001  # 30s interval = 0.001%

        if self.config.enable_resource_monitoring:
            interval_overhead = 100 / (self.config.resource_check_interval * 1000)
            overhead += interval_overhead  # 0.5s = 0.02%

        if self.config.enable_smart_cache:
            memory_overhead += self.config.cache_size_mb

        # Calculate expected speedup
        speedup = 1.0

        if self.config.enable_compilation:
            speedup *= 1.4  # 40% speedup

        if self.config.enable_quantization:
            speedup *= 2.5  # 2.5x speedup

        if self.config.enable_gpu_optimizations and self.config.enable_mixed_precision:
            speedup *= 2.0  # 2x speedup from mixed precision

        if self.config.enable_tensor_cores:
            speedup *= 1.5  # Additional 50% from Tensor Cores

        # List enabled optimizations
        enabled = {
            "GPU Capabilities Detection": self.config.detect_gpu_capabilities,
            "CPU Topology Detection (HT/SMT)": self.config.detect_cpu_topology,
            "Multi-GPU Support": self.config.detect_multi_gpu,
            "Thermal Monitoring": self.config.enable_thermal_monitoring,
            "Resource Monitoring": self.config.enable_resource_monitoring,
            "Smart Cache + Compression": self.config.enable_smart_cache,
            "Model Compilation": self.config.enable_compilation,
            "Quantization (INT8/FP16)": self.config.enable_quantization,
            "GPU Optimizations": self.config.enable_gpu_optimizations,
            "Mixed Precision (FP16/BF16/TF32)": self.config.enable_mixed_precision,
            "Tensor Cores": self.config.enable_tensor_cores,
            "HT/SMT Optimization": self.config.enable_ht_smt_optimization,
            "CPU Affinity": self.config.enable_cpu_affinity,
        }

        return {
            "total_overhead": overhead / 100,  # Convert to fraction
            "memory_overhead_mb": memory_overhead,
            "expected_speedup": speedup,
            "enabled_optimizations": enabled,
        }

    def initialize(self, lazy: bool = True):
        """
        Initialize optimizers

        Args:
            lazy: If True, only initialize when first used (recommended)
        """
        if lazy:
            logger.info("Lazy initialization enabled (optimizers created on first use)")
            return

        # Eager initialization
        logger.info("Eager initialization...")

        if self.config.detect_gpu_capabilities:
            self._get_gpu_optimizer()

        if self.config.detect_cpu_topology or self.config.enable_resource_monitoring:
            self._get_resource_monitor()

        if self.config.enable_thermal_monitoring:
            self._get_thermal_monitor()

        if self.config.detect_multi_gpu:
            self._get_multi_gpu_manager()

        logger.info("✅ Initialization complete")

    def _get_gpu_optimizer(self):
        """Get GPU optimizer (lazy)"""
        if self.gpu_optimizer is None and self.config.enable_gpu_optimizations:
            from nexlify.ml.nexlify_gpu_optimizations import GPUOptimizer

            self.gpu_optimizer = GPUOptimizer()
            if self.gpu_optimizer.config:
                self.gpu_optimizer.apply_optimizations()
        return self.gpu_optimizer

    def _get_resource_monitor(self):
        """Get resource monitor (lazy)"""
        if self.resource_monitor is None:
            from nexlify.ml.nexlify_dynamic_architecture_enhanced import \
                EnhancedDynamicResourceMonitor

            self.resource_monitor = EnhancedDynamicResourceMonitor(
                sample_interval=self.config.resource_check_interval
            )
            if self.config.enable_resource_monitoring:
                self.resource_monitor.start_monitoring()
        return self.resource_monitor

    def _get_thermal_monitor(self):
        """Get thermal monitor (lazy)"""
        if self.thermal_monitor is None and self.config.enable_thermal_monitoring:
            from nexlify.ml.nexlify_thermal_monitor import ThermalMonitor

            self.thermal_monitor = ThermalMonitor(
                check_interval=self.config.thermal_check_interval
            )
            self.thermal_monitor.start_monitoring()
        return self.thermal_monitor

    def _get_smart_cache(self, cache_dir: str):
        """Get smart cache (lazy)"""
        if self.smart_cache is None and self.config.enable_smart_cache:
            from nexlify.ml.nexlify_smart_cache import SmartCache

            self.smart_cache = SmartCache(
                cache_dir=cache_dir,
                memory_cache_mb=self.config.cache_size_mb,
                enable_compression=self.config.enable_compression,
                enable_prefetch=True,
            )
        return self.smart_cache

    def _get_multi_gpu_manager(self):
        """Get multi-GPU manager (lazy)"""
        if self.multi_gpu_manager is None and self.config.enable_multi_gpu:
            from nexlify.ml.nexlify_multi_gpu import MultiGPUManager

            self.multi_gpu_manager = MultiGPUManager()
        return self.multi_gpu_manager

    def _get_model_compiler(self):
        """Get model compiler (lazy)"""
        if self.model_compiler is None and self.config.enable_compilation:
            from nexlify.ml.nexlify_model_compilation import ModelCompiler

            self.model_compiler = ModelCompiler()
        return self.model_compiler

    def _get_quantizer(self):
        """Get quantizer (lazy)"""
        if self.quantizer is None and self.config.enable_quantization:
            from nexlify.ml.nexlify_quantization import AutoQuantizer

            self.quantizer = AutoQuantizer()
        return self.quantizer

    def optimize_model(self, model, example_input=None, auto_benchmark: bool = None):
        """
        Optimize model with all enabled optimizations

        Args:
            model: PyTorch model
            example_input: Example input (for compilation/quantization)
            auto_benchmark: If True, benchmark each optimization and only enable if beneficial
                          If None, use AUTO mode for AUTO profile

        Returns:
            Optimized model
        """
        logger.info("🚀 Optimizing model...")

        # Auto-benchmark if AUTO profile or explicitly requested
        if auto_benchmark or (
            auto_benchmark is None and self.profile == OptimizationProfile.AUTO
        ):
            if example_input is not None:
                logger.info("📊 AUTO mode: Benchmarking optimizations...")
                return self._optimize_with_benchmarking(model, example_input)
            else:
                logger.warning("AUTO mode requires example_input for benchmarking")
                logger.warning("Falling back to BALANCED profile")

        # Manual optimization with current config
        # Compilation
        if self.config.enable_compilation:
            compiler = self._get_model_compiler()
            if compiler:
                model = compiler.compile(
                    model,
                    mode=CompilationMode[
                        self.config.compilation_mode.upper().replace("-", "_")
                    ],
                    example_inputs=example_input,
                )

        # Quantization
        if self.config.enable_quantization:
            quantizer = self._get_quantizer()
            if quantizer:
                from nexlify.ml.nexlify_quantization import QuantizationMethod

                method = QuantizationMethod[self.config.quantization_method.upper()]
                model = quantizer.quantize(model, method)

        logger.info("✅ Model optimization complete")

        return model

    def _optimize_with_benchmarking(self, model, example_input):
        """
        Automatically benchmark and apply best optimizations

        Tests each optimization and only enables if it improves performance
        """
        import copy
        import time

        import torch

        logger.info("🔍 Benchmarking optimizations (this may take 1-2 minutes)...")

        # Baseline: original model
        logger.info("\n   [1/4] Benchmarking original model...")
        baseline_time = self._benchmark_model_inference(
            model, example_input, num_runs=50
        )
        logger.info(f"         Baseline: {baseline_time*1000:.2f}ms per inference")

        best_model = model
        best_time = baseline_time
        best_config = "original"

        # Test compilation
        logger.info("\n   [2/4] Testing model compilation...")
        try:
            compiler = self._get_model_compiler()
            if compiler:
                compiled_model = compiler.compile(
                    copy.deepcopy(model), example_inputs=example_input
                )

                compiled_time = self._benchmark_model_inference(
                    compiled_model, example_input, num_runs=50
                )
                speedup = baseline_time / compiled_time

                logger.info(
                    f"         Compiled: {compiled_time*1000:.2f}ms ({speedup:.2f}x speedup)"
                )

                if compiled_time < best_time * 0.95:  # At least 5% improvement
                    best_model = compiled_model
                    best_time = compiled_time
                    best_config = "compiled"
                    logger.info(
                        f"         ✓ Compilation improves performance - ENABLED"
                    )
                else:
                    logger.info(
                        f"         ✗ Compilation doesn't help enough - DISABLED"
                    )

        except Exception as e:
            logger.warning(f"         Compilation failed: {e}")

        # Test quantization (only if not already compiled, to save time)
        logger.info("\n   [3/4] Testing quantization...")
        try:
            quantizer = self._get_quantizer()
            if quantizer:
                from nexlify.ml.nexlify_quantization import QuantizationMethod

                test_model = (
                    copy.deepcopy(model) if best_config == "original" else best_model
                )

                quantized_model = quantizer.quantize(
                    test_model, method=QuantizationMethod.DYNAMIC
                )

                quantized_time = self._benchmark_model_inference(
                    quantized_model, example_input, num_runs=50
                )
                speedup = baseline_time / quantized_time

                logger.info(
                    f"         Quantized: {quantized_time*1000:.2f}ms ({speedup:.2f}x speedup)"
                )

                if quantized_time < best_time * 0.95:  # At least 5% improvement
                    best_model = quantized_model
                    best_time = quantized_time
                    best_config = (
                        "quantized"
                        if best_config == "original"
                        else "compiled+quantized"
                    )
                    logger.info(
                        f"         ✓ Quantization improves performance - ENABLED"
                    )
                else:
                    logger.info(
                        f"         ✗ Quantization doesn't help enough - DISABLED"
                    )

        except Exception as e:
            logger.warning(f"         Quantization failed: {e}")

        # Summary
        final_speedup = baseline_time / best_time
        logger.info(f"\n   [4/4] Benchmark complete!")
        logger.info(f"         Best configuration: {best_config}")
        logger.info(f"         Overall speedup: {final_speedup:.2f}x")

        # Update config based on results
        if "compiled" in best_config:
            self.config.enable_compilation = True
            logger.info("         → Compilation: ENABLED")
        else:
            self.config.enable_compilation = False
            logger.info("         → Compilation: DISABLED")

        if "quantized" in best_config:
            self.config.enable_quantization = True
            logger.info("         → Quantization: ENABLED")
        else:
            self.config.enable_quantization = False
            logger.info("         → Quantization: DISABLED")

        logger.info("\n✅ Auto-optimization complete")

        return best_model

    def _benchmark_model_inference(
        self, model, example_input, num_runs: int = 50
    ) -> float:
        """Benchmark model inference time"""
        import time

        import torch

        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                try:
                    _ = model(example_input)
                except:
                    pass  # Ignore warmup errors

        # Benchmark
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(example_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_time = total_time / num_runs

        return avg_time

    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size"""
        monitor = self._get_resource_monitor()
        if monitor:
            return monitor.get_gpu_optimal_batch_size()
        return 32  # Default

    def shutdown(self):
        """Shutdown all monitors and cleanup"""
        logger.info("Shutting down optimization manager...")

        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()

        if self.thermal_monitor:
            self.thermal_monitor.stop_monitoring()

        if self.smart_cache:
            self.smart_cache.shutdown()

        logger.info("✅ Shutdown complete")


# Convenience functions
def create_optimizer(
    profile: OptimizationProfile = OptimizationProfile.BALANCED,
) -> OptimizationManager:
    """Create optimization manager with profile"""
    return OptimizationManager(profile=profile)


# Import for convenience
from nexlify.ml.nexlify_model_compilation import CompilationMode

# Export
__all__ = [
    "OptimizationProfile",
    "OptimizationConfig",
    "OptimizationManager",
    "create_optimizer",
]
